# Instalación automática en el primer arranque

Preparado para **Raspberry Pi 5 con Raspberry Pi OS Lite de 64 bits**. El código se añade a la partición raíz antes de arrancar la tarjeta. Al encender la Pi, un servicio instala la revisión incluida, comprueba el portal y reinicia una vez para iniciar el expositor. Los siguientes arranques no repiten la instalación.

La imagen inicial necesita Internet para descargar paquetes, Python, Node y los modelos de reconocimiento. Conecta **Ethernet** durante esta instalación: al final, `wlan0` pasa a ser el punto de acceso. No se incluyen modelos 3D, claves de administración ni datos del ordenador que prepara la imagen. No se utiliza Blender.

## Preparar una tarjeta sin arrancarla

1. En Raspberry Pi Imager, selecciona **Raspberry Pi 5 → Raspberry Pi OS Lite (64-bit)**. Personaliza el nombre del equipo, el usuario administrador y su contraseña, el país Wi-Fi y SSH. Elige un usuario distinto de `gestur` y `gestur-portal`, que están reservados a los servicios. Escribe la tarjeta y **no la arranques todavía**.
2. En un ordenador Linux, una máquina virtual Linux con acceso a la tarjeta u otra Raspberry Pi, monta en lectura/escritura la partición **rootfs (ext4)** de esa tarjeta. En el ejemplo está en `/mnt/gestur-root`. No es la partición FAT `bootfs`. macOS no monta ext4 en escritura de forma nativa.
3. En ese ordenador, obtén una copia limpia de esta rama y prepara la raíz montada:

```bash
git clone --branch web-portal --single-branch https://github.com/xavi-burgos99/gestur.git
cd gestur
sudo python3 scripts/prepare-image.py \
  --root /mnt/gestur-root \
  --wifi-country ES \
  --admin-user expositor
```

Sustituye `expositor` por **el mismo usuario configurado en Imager**, `ES` por el país de uso y la ruta por tu montaje real. La herramienta requiere Python 3.9+ y Git en el ordenador que prepara la tarjeta. El repositorio debe tener sus cambios guardados en un commit: se exporta exactamente `HEAD`, sin descargar otra rama durante el primer arranque.

4. Desmonta la tarjeta de forma segura. Colócala en la Pi 5, conecta pantalla HDMI, alimentación adecuada y Ethernet con Internet, y enciéndela. No hace falta ejecutar el instalador por SSH: comienza automáticamente después de la personalización inicial de Pi OS. La duración depende de la tarjeta y la conexión.
5. Tras completarse, la Pi reinicia e inicia la bienvenida con la figura 3D y el QR. La red abierta será **GESTUR-XXXX**, según la MAC de esa Raspberry. Conéctate y abre el QR o `http://10.42.0.1:3000`.

La clave de administración es distinta en cada dispositivo y se obtiene con el usuario administrador, por SSH o consola:

```bash
sudo cat /etc/gestur/portal-token
```

## Incorporarlo a una imagen propia

La misma herramienta acepta la raíz montada de un `.img` o el directorio raíz de un proceso de construcción de imágenes. No monta discos, no graba tarjetas ni ejecuta binarios ARM en el ordenador de preparación. Rechaza `/`, imágenes de 32 bits, gestores de escritorio habilitados y raíces con una instalación o datos previos de Gestur. Repetir la preparación con la misma revisión y opciones es idempotente.

Configura también el usuario inicial en el proceso de creación de la imagen. Copiar un `firstrun.sh` suelto a la partición de arranque no garantiza que Pi OS lo ejecute. Los mecanismos de personalización varían entre versiones y una imagen local seleccionada como personalizada en Imager puede no aceptar las mismas opciones que la imagen oficial. La ruta de tarjeta descrita arriba evita esa diferencia, personalizando primero la imagen oficial. Consulta los [formatos oficiales de Raspberry Pi Imager](https://github.com/raspberrypi/rpi-imager/blob/main/doc/os_customisation_formats.md) y la [guía de instalación de Raspberry Pi](https://www.raspberrypi.com/documentation/computers/getting-started.html).

Prepara la imagen **antes de su primera instalación de Gestur**, para que cada Pi genere su propia clave y su SSID a partir de su MAC. La herramienta no convierte en plantilla un sistema ya instalado ni elimina datos para hacerlo.

## Estado y recuperación

```bash
# Estado del instalador y sus próximos reintentos
sudo systemctl status gestur-first-boot.service gestur-first-boot.timer
sudo journalctl -u gestur-first-boot.service -b

# Detalle de los pasos de instalación (solo root)
sudo tail -f /var/log/gestur-first-boot.log

# Tras corregir un problema, reintentar sin esperar los dos minutos
sudo systemctl start gestur-first-boot.service
```

El temporizador inicia el servicio a partir de los 30 segundos; este espera a los servicios de personalización de Pi OS y a la red. La red disponible no implica acceso a Internet: si una descarga o instalación falla, el servicio vuelve a intentarlo a los dos minutos y también en el siguiente arranque. APT espera bloqueos de otros instaladores y reintenta descargas. Una instalación interrumpida intenta terminar la configuración de paquetes pendientes. No se fuerza ningún bloqueo ni se marca el trabajo como terminado en caso de fallo.

La marca `/var/lib/gestur-first-boot/complete.json` se publica de forma atómica después de guardar los archivos y comprobar que el servicio web responde. Antes de ella, cualquier error mantiene la instalación pendiente. Después de ella, no se reinstala, aunque no se pudiera solicitar el reinicio: en ese caso el registro pide ejecutar `sudo reboot`. No borres esta marca para actualizar; utiliza el instalador normal con la nueva revisión.

Para detener los reintentos mientras corriges la imagen o la red:

```bash
sudo systemctl stop gestur-first-boot.timer gestur-first-boot.service
# Reanudar
sudo systemctl start gestur-first-boot.timer
```

El estado y el registro del primer arranque son privados de root; la clave del portal no se imprime en el journal. La instalación sigue usando las cuentas sin privilegios del visor y del portal. El usuario administrador debe existir y pertenecer a `sudo` antes de empezar. Si no coincide con `--admin-user`, corrige `/etc/gestur/first-boot.json` como root y vuelve a iniciar el servicio.

## Verificación

```bash
python3 -m pytest tests/test_first_boot.py tests/test_prepare_image.py -q
bash -n gestur.sh scripts/install-portal.sh
```

Las pruebas usan raíces temporales y comandos simulados: verifican la preparación, exclusiones, permisos, fallos, reintentos y la marca de finalización sin instalar paquetes ni reiniciar el equipo de desarrollo. Queda pendiente validar el arranque completo sobre una Raspberry Pi 5 física.
