# Gestur

Visor de objetos 3D controlado por cámara para exposiciones, orientado a Raspberry Pi 5. Utiliza Panda3D y controles de cabeza, manos, proximidad y retorno al reposo. La biblioteca empieza vacía; no incluye modelos de exposición.

## Instalación en Raspberry Pi 5

Requiere Raspberry Pi OS Lite/Debian **de 64 bits**, pantalla HDMI y cámara USB compatible con OpenCV/V4L2. En una imagen Desktop hay que desactivar antes el gestor de escritorio: el instalador comprueba que la pantalla esté disponible para el expositor. Usa refrigeración adecuada. La instalación descarga dependencias y modelos de reconocimiento; después el expositor funciona sin Internet.

```bash
git clone https://github.com/xavi-burgos99/gestur.git
cd gestur
git switch gestur-integrated
sudo bash gestur.sh install
sudo reboot
```

El instalador despliega **la copia local de la rama elegida** en `/opt/gestur`; no hace `pull main`. Crea un Python 3.12.14 privado, instala versiones fijadas con ruedas ARM64, aprovisiona los modelos Lite verificados, instala Assimp y meshoptimizer y configura el arranque del controlador completo. No modifica el Python del sistema, el firmware ni la memoria GPU. La configuración y los modelos de usuario se conservan al reinstalar.

- Configuración: `/var/lib/gestur/config.json`.
- Modelos importados: `/var/lib/gestur/models`.
- Registro del expositor: `/var/log/gestur/viewer.log`.
- `sudo bash gestur.sh uninstall` retira el arranque automático y conserva los datos.

También puedes dejar una tarjeta o imagen de **Raspberry Pi OS Lite de 64 bits** preparada para instalar Gestur automáticamente en el primer arranque. `scripts/prepare-image.py` incorpora esta revisión a la raíz montada; la Pi instala con Internet por Ethernet, reintenta si falla y reinicia al terminar. Después muestra la bienvenida con QR y crea su red `GESTUR-XXXX`, sin modelos 3D iniciales. Consulta [la preparación de la imagen y el primer arranque](docs/first-boot.md).

## Portal de administración

Esta rama incorpora las mejoras de rendimiento y añade un portal **React + Mantine + Node/Fastify**, con componentes locales que funcionan sin Internet. El instalador también prepara Node, el servicio web y el punto de acceso.

En instalaciones nuevas la red es **GESTUR-XXXX**, donde XXXX son los últimos cuatro caracteres de la MAC permanente de `wlan0`, sin separadores. Se crea **abierta**. Configura previamente el país Wi-Fi desde Raspberry Pi Imager o `raspi-config`; usa Ethernet para instalar, pues activar el punto de acceso puede interrumpir una conexión Wi-Fi existente.

Conéctate a esa red y abre **http://10.42.0.1**. La clave de administración del portal aparece al terminar la instalación y puede recuperarse con `sudo cat /etc/gestur/portal-token`. Esta clave protege los cambios del dispositivo y es independiente de la contraseña opcional de la red Wi-Fi.

- **Modelos 3D**: biblioteca vacía de inicio. Sube un archivo 3D o un ZIP con el modelo, materiales y texturas. Se admiten OBJ, glTF/GLB, FBX, STL, PLY, DAE, 3DS y otros formatos de malla; se reparan referencias a recursos que estén en el paquete y se convierte a GLB autocontenido. Solo por encima de 1.000.000 de triángulos aparece una propuesta fija de aproximadamente 500.000: puedes aceptarla o conservar el original. La Raspberry realiza el trabajo y el portal muestra su estado. Los modelos se centran y encuadran automáticamente.
- **Parámetros**: ajusta sensibilidad, suavizado, umbrales y asignaciones de cabeza, pinza u orientación de manos a rotación, desplazamiento y escala. Las manos se pueden activar cuando hagan falta.
- **Configuración**: en **Punto de acceso Wi-Fi** puedes cambiar el nombre, **Añadir contraseña**, **Cambiar contraseña** o **Eliminar contraseña**. Los cambios se aplican con unos segundos de margen para avisar antes de la desconexión; si fallan, se intenta recuperar la configuración anterior.

Sin selección, el expositor muestra una figura 3D procedural y un QR con «Escanea el QR para comenzar». La URL usa la IP del punto de acceso o de la red local.

Reinstalar conserva los modelos importados, los parámetros, la clave de administración y la red existente. La antigua selección del capitel incluido pasa a la bienvenida. Si ya había un punto de acceso, su dirección puede ser distinta de `10.42.0.1`. Consulta [docs/portal.md](docs/portal.md) para desarrollo, permisos, formatos y recuperación.

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
.venv/bin/python controller.py --windowed
```

El cursor queda oculto en el expositor. `Esc` cierra el visor. Con `--windowed` se conserva una ventana para desarrollo; la visibilidad del cursor es configurable en el JSON. La cámara predeterminada es la USB de índice 0. Las cámaras CSI que no expongan V4L2 requieren un adaptador de captura adicional.

```bash
# Sólo el visor, sin abrir la cámara
.venv/bin/python controller.py --no-camera --windowed

# Seguimiento opcional de manos; asignar una entrada de mano en config para controlarlo
.venv/bin/python controller.py --config config/default.json --hands

# Comprobar que los modelos están disponibles sin descargar nada
.venv/bin/python scripts/provision_models.py --check
```

## Rendimiento y reconocimiento

La captura guarda únicamente el fotograma más reciente. Se cargan sólo los detectores necesarios para los controles activos; las partes ausentes se buscan a menor frecuencia y las inferencias comparten un presupuesto de trabajo. Abrir, cambiar o recuperar la cámara se hace fuera del hilo de dibujo. El visor procesa controles a la frecuencia configurada, dibuja inmediatamente al cambiar la escena y reduce los redibujados en reposo. Conserva resolución, texturas, geometría y MSAA.

Se usan **Pose Landmarker Lite** y, al activar y asignar controles de manos, **Palm Detection Lite + Hand Landmark Lite**. Los tres giros se calculan desde un marco ortogonal de palma; se conserva el giro de imagen anterior como entrada independiente. Pinza y apertura se calculan sin otra red de clasificación. La [comparación de alternativas](docs/performance-research.md) incluye velocidad, detecciones válidas y límites de las pruebas; no atribuye precisión a un modelo sólo por ser más nuevo.

El objetivo de simplificación de **500.000 triángulos** toma como referencia los 491.038 del antiguo capitel. La reducción es opcional y solo se ofrece en mallas de más de un millón de triángulos; nunca se aplica sin aceptar la propuesta. El resultado conserva materiales y UV en la medida que permite el formato de origen; se informa del recuento final. Se agrupan nodos compatibles y se evitan transformaciones redundantes. Se solicitan 2 muestras de MSAA inicialmente; se puede elegir 0/2/4, pero el framebuffer puede conceder otra cantidad: en la Pi probada se obtuvieron 4. El código anterior no activaba explícitamente el antialiasing, por lo que no se atribuye a él un coste medido.

Los valores de 60 FPS de render y 24 detecciones/s son **objetivos configurables, no resultados garantizados en una Pi**. Ver [la guía de medición](docs/performance.md) para comprobar tiempos de fotograma, carga y temperatura en el dispositivo real.

En una **Pi 5 física de 4 GB**, el capitel original alcanzó **38,62 FPS** a 1080p con cámara, pose y manos activas durante 60 segundos, conservando geometría y textura. La CPU media del proceso desde el segundo 5 fue 73,45 % de un núcleo y la temperatura máxima, 59,5 °C. Un ensayo separado de **30 minutos con replay y giro continuo** terminó a **39,18 FPS**, con CPU total media de **20,53 %**, máximo térmico de **60,6 °C** y todas las lecturas de throttling a cero. El [informe de validación y sus JSON](docs/pi5-validation.md) registra condiciones, versión medida y límites. También se verificó un reinicio real con arranque automático del controlador, gráfica V3D acelerada y portal en HTTP 80.

## Controles y parámetros

El perfil predeterminado conserva:

- Desplazamiento horizontal de la cabeza: giro proporcional y giro continuo en los extremos 25 % / 75 %.
- Altura de la cabeza: inclinación.
- Proximidad: escala 1 a 1,75 con transición suave.
- Ausencia: parada del giro y retorno al reposo tras 3 segundos.

El archivo [config/default.json](config/default.json) define captura, render y asignaciones de gestos. [docs/configuration.md](docs/configuration.md) explica las entradas, salidas, límites, suavizado y modos. Los controles, el modelo activo, el límite de FPS y el cursor se actualizan al cambiar el archivo. Cambiar cámara o parámetros de seguimiento reemplaza el detector en segundo plano. Sólo pantalla completa y MSAA reinician el visor para recrear la ventana.

## Verificación

```bash
.venv/bin/python -m pip install pytest
.venv/bin/python -m pytest tests -q
bash -n gestur.sh scripts/kiosk-session.sh
```

Las pruebas cubren pérdida y recuperación de seguimiento, cadencia de control, orientación circular, configuración inválida, transferencia entre hilos, bienvenida sin modelos e importación y simplificación de mallas de prueba. La instalación manual, el reinicio con arranque automático, el portal HTTP 80, la importación nativa, la cámara y la salida gráfica V3D se han comprobado en la Pi descrita en el informe, junto con 30 minutos de carga térmica con replay. Quedan por evaluar la precisión de los gestos y las sesiones prolongadas con visitantes; el proceso completo de imagen preparada y primer arranque también está pendiente.

## Créditos

Proyecto académico de Xavier Burgos (`xavi@dzin.es`), Escola d'Enginyeria de la Universitat Autònoma de Barcelona, curso 2024/2025. Dirección de Fernando Vilariño, Centre de Visió per Computador. Agradecimientos a Fran Iglesias, Fundación Épica – La Fura dels Baus y Cátedra UAB–Cruïlla (TSI-100929-2023-2).

Para uso y distribución, consulta los términos específicos del proyecto académico.
