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
# Activar también una asignación de manos en Parámetros antes de esta medición:
.venv/bin/python controller.py --hands --benchmark-seconds 60 --metrics /tmp/pose-hands.json
vcgencmd measure_temp
vcgencmd get_throttled
```

Los JSON separan dibujos reales, actualizaciones de control, detecciones y política de reposo. El contador de dibujo usa el recorrido real de cámara de Panda3D; no confunde un tick sin renderizar con una imagen nueva. Los percentiles p50/p95/p99 miden intervalos entre dibujos: en reposo se esperan unos 100 ms porque la imagen es la misma. Durante interacción, comprueba además la cadencia de control y sus percentiles. Las muestras quedan limitadas en memoria.

El benchmark del controlador incluye el arranque asíncrono de cámara/modelos y conserva el visor disponible durante ese arranque. Para medir únicamente inferencia con arranque excluido, usa `pose_hand_tracker.py`, explicado en [seguimiento](tracking.md). `--hands` permite el detector, pero el controlador sólo lo carga si hay al menos una asignación de manos activa. Para comparar pose y pose+manos debe haber una persona ante la cámara; de lo contrario se mide el sondeo de ausencia.

`--show-fps --verbose` muestra dibujo y control por separado. Con el arranque habitual, `/var/lib/gestur/runtime-status.json` se actualiza una vez por segundo con CPU del proceso, CPU total del sistema, temperatura y frecuencia de CPU cuando Linux las ofrece. La CPU del proceso usa 100 % por núcleo; puede superar 100 %. El presupuesto de inferencia del 60 % mide tiempo de trabajo entre pausas, no equivale a un uso del 60 % de CPU. El indicador térmico avisa desde 78 °C; no modifica el firmware ni certifica ausencia de throttling.

Para verificar estabilidad térmica, repite con `--benchmark-seconds 1800 --verbose`, redirige el registro a un archivo y compara el principio y el final con la misma interacción. La Raspberry Pi [reduce la frecuencia al alcanzar sus límites térmicos](https://www.raspberrypi.com/news/heating-and-cooling-raspberry-pi-5/); optimizar software no sustituye medir la instalación y su refrigeración. No se ha ejecutado aquí esa prueba física.

Para aislar el coste de antialiasing, copia `config/default.json` a un archivo local y repite con `render.antialias_samples` en 0, 2 y 4. No cambies simultáneamente frecuencia, resolución y antialiasing. Por defecto hay MSAA 2×; el proyecto original no solicitaba muestras de forma explícita.

## Decisiones

- Panda3D se conserva: cambiar de motor sin datos no garantiza reducir el coste de una escena de 491.038 triángulos.
- La importación solo ofrece simplificación para más de 1.000.000 de triángulos, con objetivo de 500.000 (referencia: los 491.038 del antiguo capitel). La reducción requiere aceptar la propuesta. Mide importación y reducción por separado del render; ambas compiten por CPU y memoria mientras se ejecutan.
- Las transformaciones se aplican sólo desde el hilo de render; el callback de cámara entrega una copia en un buzón de un único elemento.
- El control continuo y las transiciones se calculan por tiempo transcurrido, por lo que no esperan una detección nueva para cada imagen.
- Una parte perdida se publica inmediatamente tras su inferencia. Si la cámara se atasca, los datos caducan por detector según la cadencia, con máximo adaptativo de 0,8 segundos. El buzón tiene también un límite de 0,8 segundos como respaldo si el trabajador deja de publicar.
- La captura descarta fotogramas antiguos. Las dos tareas Lite tienen frecuencias separadas y presupuesto conjunto; tras dos segundos sin una parte, su detector sondea a 3 FPS hasta recuperarla. Se limita la paralelización de bibliotecas numéricas en el arranque para evitar competir por todos los núcleos.
- Sin un modelo seleccionado, sin controles activos o con `--no-camera`, no se carga el reconocimiento. La creación, sustitución, cierre y recuperación de cámara ocurren fuera del hilo de dibujo; los resultados de una sesión anterior quedan revocados al cambiarla.
- Una escena inmóvil se refresca a 10 FPS, la bienvenida animada a 30 FPS y un cambio de transformación se dibuja en el mismo ciclo de control, hasta el límite configurado. Se preserva el framebuffer en los ciclos inactivos. El contador de FPS distingue dibujo y control.
- No se activa supersampling, postprocesado, sombras ni compresión de texturas con pérdidas.
- Un cambio de objeto grande puede pausar brevemente el visor durante su carga. Si no hay un modelo disponible, se muestra la bienvenida con QR y se registra el error de carga.

## Evidencia local y comparación de alternativas

- [Pruebas de dibujo y consumo del reloj de Panda3D](render-performance.md): escena sintética comparable en complejidad al capitel, calidad constante y diferencia entre refresco inmóvil e interacción.
- [Geometría y comparación de modelos](performance-research.md): Lite frente a Full, alternativas faciales y compatibilidad del runtime reciente.
- Los ensayos matemáticos cubren los giros combinados de palma, ambas manos y la pérdida/recuperación de detección. Corrigen errores del cálculo anterior; no sustituyen evaluar la precisión con una cámara y personas reales.
- La validación de integración ejecutó el controlador, ambos modelos nativos y el visor de 524.288 triángulos durante cinco segundos. La captura se sustituyó por una imagen oficial repetida; se comprobó inferencia, dibujo y cierre sin errores, sin atribuirlo a una prueba de cámara USB o Pi. Las pruebas de ciclo de vida cubren cambios durante una carga lenta, reconexión con callbacks tardíos y una cámara que sigue ocupada tras solicitar su cierre.
- En Linux ARM64 se verificaron los servicios de primer arranque y la copia de los nuevos módulos con las opciones del instalador. Se mantienen las dependencias fijadas y el Python privado. Esto no equivale a instalar y ejecutar la aplicación completa en hardware Pi.

## Referencias de implementación

- [Panda3D: agrupar objetos estáticos](https://docs.panda3d.org/1.10/python/optimization/performance-issues/too-many-meshes).
- [Panda3D: antialiasing y MSAA](https://docs.panda3d.org/1.10/python/programming/render-attributes/antialiasing).
- [Panda3D: caché de modelos](https://docs.panda3d.org/1.10/python/programming/scene-graph/model-files).
- [MediaPipe: ejemplo Raspberry Pi](https://github.com/google-ai-edge/mediapipe-samples/tree/main/examples/pose_landmarker/raspberry_pi).
- [uv: Python privado y versiones específicas](https://docs.astral.sh/uv/guides/install-python/).
