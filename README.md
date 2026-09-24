# Gestur

Visor de objetos 3D controlado por cámara para exposiciones, orientado a Raspberry Pi 5. Conserva Panda3D, el capitell original y sus controles de cabeza, proximidad y retorno al reposo.

## Instalación en Raspberry Pi 5

Requiere Raspberry Pi OS Lite/Debian **de 64 bits**, pantalla HDMI y cámara USB compatible con OpenCV/V4L2. En una imagen Desktop hay que desactivar antes el gestor de escritorio: el instalador comprueba que la pantalla esté disponible para el expositor. Usa refrigeración adecuada. La instalación descarga dependencias y modelos; después el expositor funciona sin Internet.

```bash
git clone https://github.com/xavi-burgos99/gestur.git
cd gestur
git switch web-portal
sudo bash gestur.sh install
sudo reboot
```

El instalador despliega **la copia local de la rama elegida** en `/opt/gestur`; no hace `pull main`. Crea un Python 3.12.14 privado, instala versiones fijadas con ruedas ARM64, aprovisiona los modelos Lite verificados y configura el arranque del controlador completo. No modifica el Python del sistema, el firmware ni la memoria GPU. La configuración y los modelos de usuario se conservan al reinstalar.

- Configuración: `/var/lib/gestur/config.json`.
- Modelos importados: `/var/lib/gestur/models`.
- Registro del expositor: `/var/log/gestur/viewer.log`.
- `sudo bash gestur.sh uninstall` retira el arranque automático y conserva los datos.

## Portal de administración

Esta rama incorpora las mejoras de rendimiento y añade un portal **React + Mantine + Node/Fastify**, con componentes locales que funcionan sin Internet. El instalador también prepara Node, el servicio web y el punto de acceso.

En instalaciones nuevas la red es **GESTUR-XXXX**, donde XXXX son los últimos cuatro caracteres de la MAC permanente de `wlan0`, sin separadores. Se crea **abierta**. Configura previamente el país Wi-Fi desde Raspberry Pi Imager o `raspi-config`; usa Ethernet para instalar, pues activar el punto de acceso puede interrumpir una conexión Wi-Fi existente.

Conéctate a esa red y abre **http://10.42.0.1:3000**. La clave de administración del portal aparece al terminar la instalación y puede recuperarse con `sudo cat /etc/gestur/portal-token`. Esta clave protege los cambios del dispositivo y es independiente de la contraseña opcional de la red Wi-Fi.

- **Modelos 3D**: el Capitel está siempre disponible. Importa un ZIP con un único modelo OBJ + MTL + texturas, glTF + BIN + texturas o GLB. Se validan rutas, referencias, límites y carga real con Panda3D antes de añadirlo. Los modelos importados se centran y encuadran automáticamente. El panel distingue el modelo seleccionado del que el visor está mostrando.
- **Parámetros**: ajusta sensibilidad, suavizado, umbrales y asignaciones de cabeza, pinza u orientación de manos a rotación, desplazamiento y escala. Las manos se pueden activar cuando hagan falta.
- **Configuración**: en **Punto de acceso Wi-Fi** puedes cambiar el nombre, **Añadir contraseña**, **Cambiar contraseña** o **Eliminar contraseña**. Los cambios se aplican con unos segundos de margen para avisar antes de la desconexión; si fallan, se intenta recuperar la configuración anterior.

Reinstalar conserva los modelos, los parámetros, la clave de administración y la red existente. Si ya había un punto de acceso, su dirección puede ser distinta de `10.42.0.1`. Consulta [docs/portal.md](docs/portal.md) para desarrollo, permisos, formatos y recuperación.

```bash
cd portal
npm ci
npm test
npm run build
# Variables y comando de desarrollo en docs/portal.md
```

## Desarrollo local

Python **3.11 o 3.12** (MediaPipe 0.10.18 tiene ruedas ARM64 verificadas; sus versiones posteriores no son intercambiables para esta plataforma).

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python scripts/provision_models.py
.venv/bin/python controller.py capitell.obj --windowed
```

El cursor queda oculto en el expositor. `Esc` cierra el visor. Con `--windowed` se conserva una ventana para desarrollo; la visibilidad del cursor es configurable en el JSON. La cámara predeterminada es la USB de índice 0. Las cámaras CSI que no expongan V4L2 requieren un adaptador de captura adicional.

```bash
# Sólo el visor, sin abrir la cámara
.venv/bin/python controller.py capitell.obj --no-camera --windowed

# Seguimiento opcional de manos; asignar una entrada de mano en config para controlarlo
.venv/bin/python controller.py --config config/default.json --hands

# Comprobar que los modelos están disponibles sin descargar nada
.venv/bin/python scripts/provision_models.py --check
```

## Rendimiento y reconocimiento

La captura guarda únicamente el fotograma más reciente. La inferencia de pose y manos tiene frecuencias limitadas e independientes; el renderizador procesa los controles en su propio hilo a la frecuencia de pantalla. No se acumulan fotogramas pendientes ni se modifica Panda3D desde el hilo de cámara.

Se usan **Pose Landmarker Lite** y, al activar manos, **Palm Detection Lite + Hand Landmark Lite**. El seguimiento de manos incorpora orientación de palma, pinza y gestos geométricos, sin otra red de clasificación. Las manos están desactivadas por defecto para conservar el perfil del capitell y ahorrar trabajo; pueden activarse desde configuración.

Se mantienen los **491.038 triángulos y la textura 2048 × 2048 del capitell**. No hay reducción de malla o resolución de textura. Se agrupan nodos compatibles y se evitan transformaciones redundantes. MSAA 2× es el valor inicial; se puede elegir 0/2/4 muestras. El código anterior no activaba explícitamente el antialiasing, por lo que no se atribuye a él un coste medido.

Los valores de 60 FPS de render y 24 detecciones/s son **objetivos configurables, no resultados garantizados en una Pi**. Ver [la guía de medición](docs/performance.md) para comprobar tiempos de fotograma, carga y temperatura en el dispositivo real.

## Controles y parámetros

El perfil predeterminado conserva:

- Desplazamiento horizontal de la cabeza: giro proporcional y giro continuo en los extremos 25 % / 75 %.
- Altura de la cabeza: inclinación.
- Proximidad: escala 1 a 1,75 con transición suave.
- Ausencia: parada del giro y retorno al reposo tras 3 segundos.

El archivo [config/default.json](config/default.json) define captura, render y asignaciones de gestos. [docs/configuration.md](docs/configuration.md) explica las entradas, salidas, límites, suavizado y modos. Los controles, el modelo activo, el límite de FPS y el cursor se actualizan al cambiar el archivo. Cambiar cámara, frecuencia de inferencia, seguimiento de manos, pantalla completa o MSAA reinicia el visor automáticamente para recrear sus recursos.

## Verificación

```bash
.venv/bin/python -m pip install pytest
.venv/bin/python -m pytest tests -q
bash -n gestur.sh scripts/kiosk-session.sh
```

Las pruebas cubren pérdida y recuperación de seguimiento, cadencia de control, orientación circular, configuración inválida, transferencia entre hilos y carga del capitell. La precisión visual con personas, los controladores gráficos y el rendimiento térmico requieren validación en la instalación real.

## Créditos

Proyecto académico de Xavier Burgos (`xavi@dzin.es`), Escola d'Enginyeria de la Universitat Autònoma de Barcelona, curso 2024/2025. Dirección de Fernando Vilariño, Centre de Visió per Computador. Agradecimientos a Fran Iglesias, Fundación Épica – La Fura dels Baus y Cátedra UAB–Cruïlla (TSI-100929-2023-2).

Para uso y distribución, consulta los términos específicos del proyecto académico.
