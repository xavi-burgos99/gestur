# Mediciones de motores de manos

Medido el 27 de septiembre de 2026 en un Mac M2 Pro, ARM64, macOS 27 y
Python 3.12.14. No son resultados de Raspberry Pi. Los cuatro JSON conservan
las mediciones originales de procesos separados: 180 iteraciones medidas y
20 de calentamiento. Los modelos de palma y mano de LiteRT permanecieron
residentes simultáneamente.

- `litert-default.json`: LiteRT sin especificar el número de hilos.
- `litert-1-thread.json`: LiteRT con `num_threads=1`.
- `litert-2-threads.json`: LiteRT con `num_threads=2`.
- `mediapipe-0.10.18.json`: Tasks VIDEO, CPU, hasta dos manos.

LiteRT ejecutó `invoke()` sobre tensores sintéticos fijos, generados con
`numpy.random.default_rng(42)`. No implementa en esta prueba los recortes,
NMS, asociación, seguimiento ni interpretación de gestos. MediaPipe procesó
la misma imagen estática preparada a 640×480 en cada llamada. Su temporizador
incluye `detect_for_video()` y registrar el número de manos, pero excluye
decodificar, redimensionar, convertir a RGB y construir `mp.Image`, acciones
realizadas previamente. El nombre histórico `mediapipe_full_image_VIDEO` de
su JSON significa que Tasks recibe la imagen completa, no que se mida toda
la adquisición y preparación de la imagen.

**Las latencias de ambos modos no son una comparación directa del pipeline.**
MediaPipe realiza seguimiento y evita repetir la detección de palmas cuando
puede; detectó dos manos en todas las iteraciones medidas. LiteRT ejecuta
las dos redes por separado sin esa lógica. Ninguna prueba mide precisión,
gestos reales, rotación 3D ni rendimiento con el renderizador funcionando.

`memory_peak_MB` es el máximo RSS del proceso observado hasta cada etapa,
en MB decimales. No es memoria actual, tamaño instalado ni consumo total del
sistema. La implementación completa de un backend LiteRT necesitaría más
memoria que este prototipo. Los datos de CPU son tiempo de CPU del proceso,
sumando sus hilos; no equivalen a milisegundos de pared.

## Entornos utilizados

Ambos modos: NumPy 1.26.4 y OpenCV-contrib-python 4.11.0.86 (`cv2` 4.11.0),
con `cv2.setNumThreads(1)`.

MediaPipe: 0.10.18, protobuf 4.25.9 y FlatBuffers 25.12.19, en el entorno
existente del proyecto, sin modificarlo.

LiteRT: ai-edge-litert 2.2.0, protobuf 7.36.2, ml_dtypes 0.5.4,
FlatBuffers 25.12.19, backports.strenum 1.2.8, tqdm 4.70.1 y
typing_extensions 4.16.0, en un entorno temporal nuevo. En la medición inicial
el prototipo importó OpenCV desde el entorno existente sin modificarlo.
Después se instaló la misma versión de OpenCV normalmente en el entorno
temporal y se verificaron ambos modos del script reutilizable. Ese script
ya no añade directorios de otros entornos a `sys.path`.

## Reproducción

Desde la raíz del repositorio, con los modelos locales ya provisionados y
un entorno que contenga el motor seleccionado, NumPy y OpenCV:

```sh
python scripts/benchmark_hand_runtime.py --mode lite --models-dir tracking_models --threads default --output /tmp/litert-default.json
python scripts/benchmark_hand_runtime.py --mode lite --models-dir tracking_models --threads 1 --output /tmp/litert-1-thread.json
python scripts/benchmark_hand_runtime.py --mode lite --models-dir tracking_models --threads 2 --output /tmp/litert-2-threads.json
python scripts/benchmark_hand_runtime.py --mode mediapipe --models-dir tracking_models --image /ruta/local/woman_hands.jpg --output /tmp/mediapipe.json
```

Ejecutar cada comando en un proceso nuevo, secuencialmente, sin otros
benchmarks simultáneos. El script no descarga modelos ni imágenes y no abre
la cámara. El valor `default` depende del motor y de la plataforma: su
comportamiento observado en el Mac no establece cuántos hilos usará en Pi5.

La imagen de prueba es [woman_hands.jpg de MediaPipe](https://storage.googleapis.com/mediapipe-assets/woman_hands.jpg),
67.719 bytes, SHA256
`70cbeb38e198c9862202e0979c21a99b40ca980d3e7b250176c85b1636a40f12`.
El bundle local `hand_landmarker_lite.task` tiene 4.055.094 bytes, SHA256
`28984ffd6aaf10e44a054356a7e768806178bf1db084efe0aec9d636a7ed2c87`.
Los JSON LiteRT incluyen los hashes de los dos modelos individuales.

Documentación primaria: [Interpreter y control de hilos](https://developers.google.com/edge/api/tflite/python/tf/lite/Interpreter),
[publicación LiteRT 2.2.0](https://pypi.org/project/ai-edge-litert/2.2.0/),
[pipeline de MediaPipe Hands](https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/hands.md).
