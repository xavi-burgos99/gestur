# Seguimiento ligero y sin red

El controlador conserva `head`, `torso`, `left_hand` y `right_hand`, sus campos
`detected`, `x`, `y`, `pitch`, `yaw`, `roll`, y `head.scale` entre 0 y 1. Los
controles de posición de cabeza y zoom del capitell siguen utilizando ese
contrato. Se mantienen `run`, `stop`, `subscribe`, `unsubscribe` y
`get_current_data`; construir un tracker ya no abre la cámara.

## Modelos y despliegue

Se utiliza **MediaPipe Tasks**, en modo VIDEO y CPU:

| Parte | Modelo | Tamaño instalado |
| --- | --- | --- |
| Cuerpo | Pose Landmarker **Lite**, float16, versión 1 | 5.78 MB |
| Manos | Palm Detection **Lite** + Hand Landmark **Lite** | 4.05 MB en el paquete task |

El paquete de manos se construye con dos modelos Lite oficiales. La tarea de
manos distribuida directamente por Google contiene los modelos **Full** (7.82 MB),
por lo que descargarla y llamarla «Lite» no reduciría su tamaño. El instalador
añade a los Lite los metadatos de normalización RGB `[0, 1]` que exige Tasks,
conservando los pesos y el orden de los tensores. El manifiesto fija SHA-256 y
tamaño de los originales y de los modelos procesados; el ZIP usa fechas y orden
fijos. No se añade un clasificador neuronal de gestos.

```sh
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python scripts/provision_models.py
.venv/bin/python scripts/provision_models.py --check
```

Los modelos quedan en `tracking_models/`. Sólo el instalador necesita Internet;
`--check` y el seguimiento funcionan sin conexión. Una descarga inválida no
sustituye un archivo existente. Para preparar una instalación aislada, copiar
este directorio completo después de verificarlo. `--model-dir` permite preparar
otra ubicación; pasar esa misma ruta a `PoseHandTracker(model_dir=...)` si no se
usa la ubicación por defecto.

Se fija MediaPipe 0.10.18 porque publica wheels Linux ARM64 para Python 3.11 y
3.12; las versiones 0.10.20, 0.10.21, 0.10.31 y 0.10.35 consultadas no los publican.
El instalador prepara Python 3.12 y un entorno virtual, incluso si el sistema
utiliza Python 3.13. No instalar varios paquetes OpenCV en ese entorno: MediaPipe
requiere `opencv-contrib-python`, que ya proporciona `cv2`.

En macOS ARM64, el wheel oficial 0.10.18 tiene una discrepancia: el nombre indica
`universal2` y el binario contiene ARM64 y x86_64, pero su metadato interno WHEEL
indica sólo x86_64. `pip check` puede advertir que la plataforma no es compatible
aunque la inferencia funciona. Esto no debe ocultarse corrigiendo archivos del
entorno manualmente; la validación del hardware objetivo se hace en Linux ARM64.

## Cadencia, pérdida de seguimiento y rotaciones

- Un hilo drena la cámara a un único hueco de memoria. La inferencia toma siempre
  la imagen más reciente; no se acumula una cola de imágenes antiguas.
- Pose y manos tienen frecuencias máximas independientes (`inference_fps` y
  `hand_fps`). Las manos sólo se cargan si `use_hands` está activado. El render
  continúa a su propia frecuencia.
- Las marcas de captura usan reloj monotónico; Tasks recibe milisegundos
  estrictamente crecientes. `get_metrics()` permite observar imágenes capturadas,
  inferencias de cada modelo, errores de cámara, duración de inferencia y edad
  de la imagen al terminar. Los objetivos de FPS no son una garantía de FPS.
- El filtro exponencial utiliza el tiempo real transcurrido, admite cero
  suavizado y trata los ángulos circularmente. Una transición de 179° a −179°
  cruza 180°, sin girar por cero.
- Las partes con landmarks de baja visibilidad/presencia no se consideran
  detectadas. Una detección vacía desactiva esa parte inmediatamente; una cámara
  atascada caduca sus datos. No se inventan manos durante el arranque ni se
  mezclan ángulos cero en detecciones perdidas.
- Las orientaciones de palma y las distancias de gestos utilizan landmarks
  tridimensionales en metros. `rotation`/`roll` mide la dirección visible
  muñeca→base del dedo medio: arriba = 0°, derecha = +90°, izquierda = −90°.
  `pitch`/`yaw` describen la inclinación de la normal de la palma.
- Tasks invierte la asignación de etiquetas respecto a la antigua API
  Solutions. Se corrige izquierda/derecha **al aplicar espejo**, y se respeta
  `invert_hands` como inversión adicional. No se debe aplicar la regla de
  selfie de Solutions directamente a los resultados de Tasks.

Cada mano añade `rotation`, `pinch` y `openness`. `pinch` es la distancia
pulgar–índice dividida por el ancho de la palma, limitada a `[0, 1]`: cero significa
contacto. `openness` es la fracción de los cuatro dedos largos extendidos.
`gesture` distingue `pinch`, `open`, `fist` o `unknown` mediante geometría; esta
etiqueta no tiene un modelo adicional ni sustituye una evaluación de precisión
con usuarios reales. Para controles continuos se recomienda el valor `pinch`.

La escala de cabeza ahora usa los centros de **los dos ojos** (2 y 5). El código
anterior medía entre dos landmarks del mismo ojo (1 y 2). El rango nuevo por
defecto es 0.02–0.20 del ancho de imagen, limitado a 0–1; puede calibrarse con
`head_scale_min` y `head_scale_max`. La corrección cambia la sensibilidad del
zoom respecto al error anterior, por lo que conviene ajustarla a la distancia
real de la cámara en la exposición.

## Verificación y límites

```sh
.venv/bin/python -m pytest tests/test_tracking_geometry.py tests/test_tracking_runtime.py tests/test_tracking_models.py -q
```

Las pruebas sin cámara cubren rotaciones positivas/negativas y cercanas a ±180°,
pinza invariante a escala/rotación, puño, visibilidad, espejo, filtros con distintas
frecuencias, colas acotadas, cadencias distintas, reinicio, callbacks y fallos de
cámara/inferencia. Si los modelos se han aprovisionado también se ejecuta
inferencia CPU real en ambos Tasks con imágenes vacías.

Además se verificó manualmente en el entorno de desarrollo que los modelos Lite
producen dos manos y sus landmarks métricos con la imagen oficial
`right_hands.jpg`, y una pose con `pose.jpg`. Es una prueba de integración, no una
medición de precisión ni de rendimiento en Raspberry Pi. Queda por medir en
una Pi 5 con la cámara, iluminación, refrigeración, resolución de pantalla y
modelo capitell de la instalación. Los modelos Lite pueden cambiar la precisión
frente a Full; la geometría y el filtrado corrigen errores concretos del código,
pero no permiten prometer que todas las oclusiones se resuelvan.

## Referencias verificadas

- [Pose Landmarker Python y ejemplo de Raspberry Pi](https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker/python).
- [Modelos oficiales de pose](https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker).
- [Hand Landmarker: modelos y tracking](https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker).
- [Código del paquete de manos y soporte de Lite](https://github.com/google-ai-edge/mediapipe/blob/v0.10.18/mediapipe/tasks/cc/vision/hand_landmarker/hand_landmarker_graph.cc).
- [Orden de tensores y asignación de handedness de Tasks](https://github.com/google-ai-edge/mediapipe/blob/v0.10.18/mediapipe/tasks/cc/vision/hand_landmarker/hand_landmarks_detector_graph.cc).
- [Wheels oficiales de MediaPipe 0.10.18](https://pypi.org/project/mediapipe/0.10.18/#files).

## Medición en la Raspberry Pi

El antiguo `pose_hand_tracker.py` se sustituye por una utilidad de medición con
tiempo limitado. Cuenta inferencias completadas del motor, porque un callback
puede representar una actualización de estado sin una inferencia nueva.

```sh
.venv/bin/python pose_hand_tracker.py --seconds 30 --camera 0 --output artifacts/pose-pi5.json
.venv/bin/python pose_hand_tracker.py --seconds 30 --hands --output artifacts/pose-hands-pi5.json
```

El informe incluye contadores, FPS observados, CPU (100% = un núcleo), memoria
máxima del proceso y percentiles de las últimas latencias muestreadas a 20 Hz.
La fracción de muestras con una parte detectada no es una medida de precisión;
requiere una persona delante de la cámara. El tiempo de carga y cierre se
excluye de la medición, y el renderizador no participa en esta utilidad. El
comando termina al alcanzar la duración o ante un error de cámara/inferencia;
no publica comparaciones de velocidad ni calidad sin haberlas medido.
