#!/usr/bin/env python3
"""One-time provisioning. Runs with OS Python, before Gestur's venv exists."""
import datetime
import fcntl
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import time
import urllib.request

BOOTSTRAP = Path('/opt/gestur-bootstrap')
CONFIG = Path('/etc/gestur/first-boot.json')
STATE = Path('/var/lib/gestur-first-boot')
LOG = Path('/var/log/gestur-first-boot.log')


def read_settings(path):
    settings = json.loads(path.read_text())
    if not re.fullmatch(r'[A-Z]{2}', settings.get('wifi_country', '')):
        raise ValueError('Falta un país Wi-Fi válido en first-boot.json.')
    admin = settings.get('admin_user', '')
    if not re.fullmatch(r'[a-z_][a-z0-9_-]{0,31}', admin) or admin in ('root', 'gestur', 'gestur-portal'):
        raise ValueError('Indica el usuario administrador creado con Raspberry Pi Imager.')
    if not re.fullmatch(r'[0-9a-f]{40,64}', settings.get('revision', '')):
        raise ValueError('La revisión del paquete no es válida.')
    if 'hostname' in settings and (not isinstance(settings['hostname'], str)
            or not re.fullmatch(r'[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?', settings['hostname'])):
        raise ValueError('El hostname debe tener 1–63 letras minúsculas, números o guiones, sin .local ni guiones en los extremos.')
    return settings


def check_host(settings):
    if os.geteuid() != 0 or platform.system() != 'Linux' or platform.machine() != 'aarch64':
        raise RuntimeError('El primer arranque requiere root y Raspberry Pi OS Lite de 64 bits.')
    model = Path('/proc/device-tree/model').read_text().rstrip('\x00')
    if not model.startswith('Raspberry Pi 5'):
        raise RuntimeError('Esta imagen de Gestur está preparada para Raspberry Pi 5.')
    # These modules consult the real OS accounts only when running on the Pi.
    import grp
    import pwd
    admin = pwd.getpwnam(settings['admin_user'])
    sudo = grp.getgrnam('sudo')
    if admin.pw_uid < 1000 or (admin.pw_gid != sudo.gr_gid and admin.pw_name not in sudo.gr_mem):
        raise RuntimeError('Completa la creación del usuario administrador con Raspberry Pi Imager.')
    if admin.pw_shell.endswith(('nologin', 'false')):
        raise RuntimeError('El usuario administrador necesita una sesión de acceso.')


def portal_ready():
    """Wait for a real HTTP response; systemctl alone cannot prove Node started."""
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    for _ in range(30):
        try:
            with opener.open('http://127.0.0.1/api/session', timeout=2) as response:
                body = json.load(response)
                if response.status == 200 and isinstance(body, dict) and isinstance(body.get('authenticated'), bool):
                    return
        except (OSError, ValueError):
            pass
        time.sleep(1)
    raise RuntimeError('El portal no responde en el puerto 80; la instalación sigue pendiente.')


def atomic_json(path, content):
    temp = path.with_suffix('.tmp')
    with temp.open('w') as output:
        os.chmod(temp, 0o600)
        json.dump(content, output, ensure_ascii=False)
        output.write('\n')
        output.flush()
        os.fsync(output.fileno())
    temp.replace(path)
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def provision(*, bootstrap=BOOTSTRAP, config=CONFIG, state=STATE, log_path=LOG,
              run=subprocess.run, host_check=check_host, ready=portal_ready, sync=os.sync):
    """Dependencies are injectable so failure/reboot tests never modify the host."""
    settings = read_settings(config)
    host_check(settings)
    if (bootstrap / '.source-revision').read_text().strip() != settings['revision']:
        raise RuntimeError('La revisión del código no coincide con la imagen preparada.')
    state.mkdir(mode=0o700, parents=True, exist_ok=True)
    state.chmod(0o700)
    # systemd serializes starts; the lock also covers a manual invocation.
    with (state / 'lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        marker = state / 'complete.json'
        if marker.exists():
            print('Gestur ya está instalado; no se repite el primer arranque.', flush=True)
            return
        log_path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND | os.O_NOFOLLOW, 0o600)
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, 'a', buffering=1) as log:
            stamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
            log.write(f'\n[{stamp}] Iniciando instalación {settings["revision"]}\n')
            print(f'Instalando Gestur. Progreso: sudo tail -f {log_path}', flush=True)
            env = {**os.environ, 'DEBIAN_FRONTEND': 'noninteractive', 'GESTUR_UNATTENDED': '1',
                   'PATH': '/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin'}

            def command(args):
                return run(args, check=True, env=env, stdout=log, stderr=subprocess.STDOUT,
                           stdin=subprocess.DEVNULL, umask=0o022)

            try:
                # Resume packages left half-configured by a power loss. dpkg
                # keeps its own lock; if another updater owns it we retry later.
                command(['/usr/bin/dpkg', '--configure', '--pending'])
                command(['/usr/bin/raspi-config', 'nonint', 'do_wifi_country', settings['wifi_country']])
                hostname_args = ['--hostname', settings['hostname']] if 'hostname' in settings else []
                command(['/bin/bash', str(bootstrap / 'gestur.sh'), 'install', *hostname_args])
                command(['/usr/bin/systemctl', 'is-active', '--quiet', 'gestur-portal.service'])
                ready()
                # Persist the runtime before publishing the durable completion
                # marker, so a power loss cannot certify unwritten packages.
                sync()
                atomic_json(marker, {'revision': settings['revision'], 'completed_at':
                            datetime.datetime.now(datetime.timezone.utc).isoformat()})
            except Exception as error:
                log.write(f'Instalación pendiente: {error}\n')
                raise
            log.write('Instalación completada. Abre el portal para completar la configuración inicial.\n')
            print('Gestur instalado. Reiniciando para iniciar el expositor.', flush=True)
            try:
                command(['/usr/bin/systemctl', '--no-block', 'reboot'])
            except subprocess.CalledProcessError:
                # Installation is already durable: do not reinstall on reboot failure.
                log.write('No se pudo solicitar el reinicio. Ejecuta sudo reboot.\n')
                print('Instalación completada. Ejecuta sudo reboot para iniciar el expositor.', flush=True)


def main():
    try:
        provision()
    except Exception as error:
        print(f'Primer arranque pendiente: {error}. Se reintentará en 2 minutos. Registro: {LOG}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
