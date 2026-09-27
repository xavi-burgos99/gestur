# Medir el rendimiento de Gestur

No se ha medido todavía esta versión en una Raspberry Pi física. El cambio reduce trabajo redundante y separa los relojes de captura/inferencia/control/render; no promete un número de FPS o una mejora porcentual sin medir el dispositivo.

## Medición reproducible en Pi 5

Usa siempre la misma pantalla/resolución, cámara, iluminación, distancia al visitante y refrigeración. Espera a que termine la carga inicial del modelo elegido. Cierra otros procesos intensivos y observa el indicador de temperatura/throttling de la Pi.

```bash
cd /opt/gestur
# Selecciona un modelo en el portal antes de medir.
# Ejecutar desde la sesión gráfica, deteniendo antes la otra instancia del visor.
.venv/bin/python controller.py --no-camera --benchmark-seconds 60 --metrics /tmp/render.json
.venv/bin/python controller.py --benchmark-seconds 60 --metrics /tmp/pose.json
.venv/bin/python controller.py --hands --benchmark-seconds 60 --metrics /tmp/pose-hands.json
vcgencmd measure_temp
vcgencmd get_throttled
```

Los JSON contienen FPS observados y percentiles p50/p95/p99 del intervalo entre fotogramas, además de la configuración usada y métricas del detector cuando está activo. El tiempo empieza después de crear los modelos/cámara. Las muestras quedan limitadas en memoria. Compara especialmente p95/p99: una media alta puede ocultar tirones. `--show-fps --verbose` permite observar la cadencia durante interacción.

Para aislar el coste de antialiasing, copia `config/default.json` a un archivo local y repite con `render.antialias_samples` en 0, 2 y 4. No cambies simultáneamente frecuencia, resolución y antialiasing. Por defecto hay MSAA 2×; el proyecto original no solicitaba muestras de forma explícita.

## Decisiones

- Panda3D se conserva: cambiar de motor sin datos no garantiza reducir el coste de una escena de 491.038 triángulos.
- La importación solo ofrece simplificación para más de 1.000.000 de triángulos, con objetivo de 500.000 (referencia: los 491.038 del antiguo capitel). La reducción requiere aceptar la propuesta. Mide importación y reducción por separado del render; ambas compiten por CPU y memoria mientras se ejecutan.
- Las transformaciones se aplican sólo desde el hilo de render; el callback de cámara entrega una copia en un buzón de un único elemento.
- El control continuo y las transiciones se calculan por tiempo transcurrido, por lo que no esperan una detección nueva para cada imagen.
- Si dejan de llegar resultados durante 350 ms, se envía ausencia al control: no se perpetúa un giro con una cámara desconectada.
- La captura descarta fotogramas antiguos y las dos tareas Lite tienen frecuencias separadas. Se limita la paralelización de bibliotecas numéricas en el arranque para evitar competir por todos los núcleos.
- No se activa supersampling, postprocesado, sombras ni compresión de texturas con pérdidas.
- Un cambio de objeto grande puede pausar brevemente el visor durante su carga. Si no hay un modelo disponible, se muestra la bienvenida con QR y se registra el error de carga.

## Referencias de implementación

- [Panda3D: agrupar objetos estáticos](https://docs.panda3d.org/1.10/python/optimization/performance-issues/too-many-meshes).
- [Panda3D: antialiasing y MSAA](https://docs.panda3d.org/1.10/python/programming/render-attributes/antialiasing).
- [Panda3D: caché de modelos](https://docs.panda3d.org/1.10/python/programming/scene-graph/model-files).
- [MediaPipe: ejemplo Raspberry Pi](https://github.com/google-ai-edge/mediapipe-samples/tree/main/examples/pose_landmarker/raspberry_pi).
- [uv: Python privado y versiones específicas](https://docs.astral.sh/uv/guides/install-python/).
