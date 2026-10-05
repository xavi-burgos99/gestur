# Instalación de Gestur

## 1. Materiales

- Raspberry Pi 5 y fuente USB-C oficial de 27 W.
- MicroSD de **16 GB como mínimo**; 32 GB o más recomendado para guardar modelos.
- Lector de microSD y ordenador con Git, SSH y SCP.
- Webcam USB compatible con V4L2; calidad media-alta recomendada.
- Pantalla y cable **micro-HDMI a HDMI**.
- Ventilador oficial **Raspberry Pi 5 Active Cooler**.
- Cable Ethernet y router con Internet: necesarios para esta instalación,
  opcionales después.

## 2. Preparar la microSD

Abre [Raspberry Pi Imager](https://www.raspberrypi.com/software/), selecciona
**Raspberry Pi 5 → Raspberry Pi OS Lite (64-bit)** y la microSD. Personaliza:

- Nombre del equipo: `gestur`.
- País: España (`ES`); ciudad: Madrid; zona horaria: `Europe/Madrid`.
- Idioma/región: español (`es_ES.UTF-8`); teclado: español (`es`).
- Usuario: `gestur` (recomendado); contraseña segura.
- **No configurar Wi-Fi: deja sus campos vacíos.**
- Activar SSH con contraseña o con tu clave pública.
- Desactivar Raspberry Pi Connect.

Graba la tarjeta y espera a que termine la verificación. Si Imager no ofrece
alguno de los campos regionales, configúralo después con `sudo raspi-config`.

## 3. Arrancar y comprobar la conexión

Introduce la microSD con el sistema recién instalado, conecta el ventilador,
la webcam, la pantalla y Ethernet al router. Conecta la alimentación al final.
Espera **entre 2 y 5 minutos aproximadamente** durante el primer arranque.
Desde un ordenador en la misma red:

```bash
ssh gestur@gestur.local
```

Usa la contraseña o clave SSH configurada en Imager. Si elegiste otro usuario,
sustituye `gestur` por ese usuario en los comandos. Si `.local` no resuelve,
usa la IP del dispositivo, por ejemplo `ssh gestur@192.168.1.93`.
Una vez comprobado el acceso, ejecuta `exit` para volver al ordenador.

## 4. Copiar e instalar

En el ordenador, dentro de la copia local del repositorio que quieras instalar,
crea un paquete del último commit y envíalo a la Raspberry:

```bash
git archive --format=tar.gz -o ../gestur-install.tar.gz HEAD
scp ../gestur-install.tar.gz gestur@gestur.local:~/gestur-install.tar.gz
```

Entra por SSH y ejecuta:

```bash
ssh gestur@gestur.local
mkdir -p ~/gestur
tar -xzf ~/gestur-install.tar.gz -C ~/gestur
sudo bash ~/gestur/gestur.sh install --hostname gestur
sudo reboot
```

Espera a que el instalador termine antes de reiniciar; descarga sus dependencias
por Ethernet y no solicita configuración de Gestur. `sudo` puede pedir la
contraseña del usuario del sistema, también si accedes mediante clave SSH.

## 5. Abrir el portal

Tras reiniciar, espera unos minutos y abre desde la misma red
**[http://gestur.local/](http://gestur.local/)** o `http://IP-DEL-DISPOSITIVO/`.
**Usa HTTP, no HTTPS, y no añadas ningún puerto.**

Completa el asistente: deja el hostname vacío para conservar `gestur`, configura
una contraseña y confírmala. Esa contraseña servirá para el portal y el punto de
acceso Wi-Fi. Tras el reinicio, vuelve a abrir el portal e inicia sesión.
Ya puedes subir tu primer modelo.

Referencia: [guía oficial de Raspberry Pi](https://www.raspberrypi.com/documentation/computers/getting-started.html).
