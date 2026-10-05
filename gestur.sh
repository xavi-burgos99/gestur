#!/usr/bin/env bash
# Install the checked-out revision, including feature branches. Never pulls main.
set -euo pipefail
# Provisioning runs with a private log/state umask; runtimes must remain usable
# by the unprivileged viewer and portal service accounts.
umask 022
export DEBIAN_FRONTEND=noninteractive
export PIP_NO_INPUT=1
GESTUR_SOURCE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
GESTUR_PREFIX=/opt/gestur
GESTUR_STATE=/var/lib/gestur

require_host() {
    if [[ $(id -u) != 0 ]]; then
        echo "Ejecuta: sudo bash $0 $*" >&2
        exit 1
    fi
    if [[ $(uname -s) != Linux || $(uname -m) != aarch64 ]]; then
        echo "El instalador requiere Raspberry Pi OS / Debian de 64 bits (aarch64)." >&2
        exit 1
    fi
}

install_gestur() {
    require_host install
    if systemctl is-active --quiet display-manager.service || \
        systemctl is-enabled --quiet display-manager.service; then
        echo "Gestur necesita una sesión de expositor exclusiva en tty1/:0." >&2
        echo "Usa Raspberry Pi OS Lite o desactiva antes tu gestor de escritorio." >&2
        exit 1
    fi
    echo "Instalando la copia local de Gestur desde $GESTUR_SOURCE"
    apt-get -o DPkg::Lock::Timeout=300 -o Acquire::Retries=3 -o APT::Update::Error-Mode=any update
    apt-get -o DPkg::Lock::Timeout=300 -o Acquire::Retries=3 \
        -o Dpkg::Options::=--force-confdef -o Dpkg::Options::=--force-confold install -y --no-install-recommends \
        ca-certificates curl rsync python3 python3-venv \
        xserver-xorg xinit openbox x11-xserver-utils \
        mesa-utils libgl1-mesa-dri libglx-mesa0 libegl1 libgles2 \
        libglib2.0-0 libsm6 libxext6 libxrender1 libportaudio2 libgomp1

    if ! id -u gestur >/dev/null 2>&1; then
        useradd --create-home --shell /bin/bash gestur
    fi
    usermod -a -G video,render,input gestur
    install -d -o root -g root -m 755 "$GESTUR_PREFIX"
    install -d -o gestur -g gestur -m 2770 "$GESTUR_STATE" "$GESTUR_STATE/models"
    install -d -o gestur -g gestur -m 750 /var/log/gestur
    if [[ "$GESTUR_SOURCE" != "$GESTUR_PREFIX" ]]; then
        rsync -a --chown=root:root \
            --exclude=.git --exclude=.venv --exclude=.bootstrap --exclude=.python \
            --exclude=node_modules --exclude=dist --exclude=__pycache__ \
            --exclude=.pytest_cache --exclude=.ruff_cache --exclude=.coverage \
            --exclude='.coverage.*' --exclude=htmlcov --exclude=coverage.xml \
            --exclude=data --exclude=models \
            --exclude=models_compressed --exclude='capitell.*' --exclude=.dev-data \
            --exclude=.cache --exclude=artifacts --exclude='*.local.json' \
            "$GESTUR_SOURCE/" "$GESTUR_PREFIX/"
    fi

    # Remove only the old flat runtime modules after copying the package layout.
    for retired in control_system controller device_metrics pose_detector \
        pose_hand_tracker primary_person render_scheduler runtime_config \
        runtime_state screen_settings tracking_geometry tracking_session visualizer; do
        rm -f "$GESTUR_PREFIX/$retired.py"
    done

    # Pi OS can ship Python 3.13+, but the verified ARM64 MediaPipe wheel
    # targets 3.12. A private, managed interpreter leaves system Python intact.
    python3 -m venv "$GESTUR_PREFIX/.bootstrap"
    "$GESTUR_PREFIX/.bootstrap/bin/python" -m pip install --disable-pip-version-check uv==0.12.18
    export UV_PYTHON_INSTALL_DIR="$GESTUR_PREFIX/.python"
    export UV_PYTHON_BIN_DIR="$GESTUR_PREFIX/.python/bin"
    "$GESTUR_PREFIX/.bootstrap/bin/uv" python install 3.12.14
    if [[ -x "$GESTUR_PREFIX/.venv/bin/python" ]] && \
        ! "$GESTUR_PREFIX/.venv/bin/python" -c 'import sys; assert sys.version_info[:2] == (3, 12)'; then
        mv "$GESTUR_PREFIX/.venv" "$GESTUR_PREFIX/.venv-backup-$(date +%s)"
    fi
    if [[ ! -x "$GESTUR_PREFIX/.venv/bin/python" ]]; then
        "$GESTUR_PREFIX/.bootstrap/bin/uv" venv --python 3.12.14 --managed-python "$GESTUR_PREFIX/.venv"
    fi
    "$GESTUR_PREFIX/.bootstrap/bin/uv" pip install \
        --only-binary :all: --python "$GESTUR_PREFIX/.venv/bin/python" -r "$GESTUR_PREFIX/requirements/runtime.txt"
    "$GESTUR_PREFIX/.venv/bin/python" "$GESTUR_PREFIX/scripts/provision_models.py"
    chown -R root:root "$GESTUR_PREFIX"
    if [[ ! -f "$GESTUR_STATE/config.json" ]]; then
        TASK_CONFIG=$(mktemp "$GESTUR_STATE/.config.XXXXXX")
        install -o gestur -g gestur -m 660 "$GESTUR_PREFIX/config/default.json" "$TASK_CONFIG"
        mv -T "$TASK_CONFIG" "$GESTUR_STATE/config.json"
    fi
    # Validate without opening a camera or changing existing parameters.
    (cd "$GESTUR_PREFIX" && .venv/bin/python -c \
        'from gestur.runtime_config import load_config; load_config("/var/lib/gestur/config.json")')

    # Complete all downloads and portal checks before enabling the kiosk login.
    bash "$GESTUR_PREFIX/scripts/install-portal.sh" "$GESTUR_PREFIX" "$@"
    # The display controller may be card0 or card1. Select vc4 by its DRM name,
    # so Xorg does not make the separate v3d render-only device the primary GPU.
    bash "$GESTUR_PREFIX/scripts/configure-xorg.sh" install
    install -o root -g root -m 755 "$GESTUR_PREFIX/scripts/kiosk-session.sh" /usr/local/bin/gestur-session
    install -d /etc/systemd/system/getty@tty1.service.d
    if [[ -f /etc/systemd/system/getty@tty1.service.d/override.conf ]] && \
        [[ ! -f /etc/systemd/system/getty@tty1.service.d/override.conf.before-gestur ]]; then
        cp -p /etc/systemd/system/getty@tty1.service.d/override.conf \
            /etc/systemd/system/getty@tty1.service.d/override.conf.before-gestur
    fi
    cat > /etc/systemd/system/getty@tty1.service.d/override.conf <<'GETTY'
[Service]
ExecStart=
ExecStart=-/sbin/agetty --autologin gestur --noclear %I $TERM
GETTY
    touch /home/gestur/.bash_profile
    if [[ ! -f /home/gestur/.bash_profile.before-gestur ]]; then
        cp -p /home/gestur/.bash_profile /home/gestur/.bash_profile.before-gestur
    fi
    # Migrate the old installer's managed block, preserving other profile content.
    sed -i '/# Gestur autostart/,/^fi/d' /home/gestur/.bash_profile
    cat >> /home/gestur/.bash_profile <<'PROFILE'
# Gestur autostart
if [ "$(tty)" = /dev/tty1 ] && [ -z "${DISPLAY:-}" ]; then
    exec startx /usr/local/bin/gestur-session -- :0 vt1 -keeptty
fi
PROFILE
    chown gestur:gestur /home/gestur/.bash_profile
    # Use stock KMS/Mesa. No gpu_mem edits, experimental firmware or full OS upgrade.
    systemctl daemon-reload
    echo "Instalación completada. Reinicia para arrancar el expositor."
    echo "Configuración conservada en $GESTUR_STATE/config.json"
}

uninstall_gestur() {
    require_host uninstall
    # Also cancel retries if uninstalling after an interrupted first boot.
    if [[ -f /etc/systemd/system/gestur-first-boot.timer ]]; then
        systemctl disable --now gestur-first-boot.timer
        systemctl stop gestur-first-boot.service
    fi
    if [[ -f /etc/systemd/system/gestur-portal.service ]]; then
        systemctl disable --now gestur-portal.service
        rm -f /etc/systemd/system/gestur-portal.service
    fi
    rm -f /etc/sudoers.d/gestur-wifi /usr/local/libexec/gestur-wifi \
        /etc/sudoers.d/gestur-device /usr/local/libexec/gestur-device
    if [[ -f /etc/systemd/system/getty@tty1.service.d/override.conf.before-gestur ]]; then
        mv /etc/systemd/system/getty@tty1.service.d/override.conf.before-gestur \
            /etc/systemd/system/getty@tty1.service.d/override.conf
    elif [[ -f /etc/systemd/system/getty@tty1.service.d/override.conf ]] && \
        grep -q -- '--autologin gestur' /etc/systemd/system/getty@tty1.service.d/override.conf; then
        rm /etc/systemd/system/getty@tty1.service.d/override.conf
    fi
    if [[ -f /home/gestur/.bash_profile ]]; then
        sed -i '/# Gestur autostart/,/^fi/d' /home/gestur/.bash_profile
    fi
    rm -f /usr/local/bin/gestur-session
    bash "$GESTUR_SOURCE/scripts/configure-xorg.sh" uninstall
    systemctl daemon-reload
    echo "Arranque y portal retirados. Código, usuarios, Wi-Fi, modelos y configuración conservados."
}

main() {
    local action=${1:-}
    if [[ $# -gt 0 ]]; then
        shift
    fi
    case "$action" in
        install)
            # Validate before touching packages, files, networking or users.
            if [[ $# -gt 0 ]]; then
                if [[ $# != 2 || "$1" != --hostname || ! "$2" =~ ^[a-z0-9]([a-z0-9-]{0,61}[a-z0-9])?$ ]]; then
                    echo 'Usa --hostname con 1–63 letras minúsculas, números o guiones, sin .local ni guiones en los extremos.' >&2
                    return 1
                fi
            fi
            install_gestur "$@" </dev/null
            ;;
        uninstall)
            if [[ $# != 0 ]]; then
                echo 'uninstall no admite opciones.' >&2
                return 1
            fi
            uninstall_gestur
            ;;
        *)
            echo "Uso: sudo bash $0 install [--hostname nombre] | uninstall" >&2
            return 1
            ;;
    esac
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    main "$@"
fi
