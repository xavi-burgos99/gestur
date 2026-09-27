# Pruebas experimentales

Scripts usados en las [mediciones de Raspberry Pi 5](../../docs/pi5-validation.md).
No se importan desde la aplicación ni cambian sus dependencias. Los scripts de
medición y exportación se conservan byte por byte; el
[manifiesto de resultados](../../docs/benchmarks/pi5/manifest.json) registra sus hashes.

## Nuevas entradas y gráficos

`benchmark_new_inputs.py` comprueba durante 60 segundos el contrato de los
controles de torso y proximidad de cada mano con los modelos instalados. No
modifica la configuración ni inicia el visor. Distingue entre un error y falta
de cobertura cuando una parte no aparece en la imagen; no mide precisión.

```sh
.venv/bin/python scripts/research/benchmark_new_inputs.py --project /opt/gestur --source replay --image /ruta/woman_hands.jpg --seconds 60 --output /tmp/new-inputs.json
```

`plot_pi_system.py` genera un PNG a partir de una captura de telemetría existente
sin modificarla. Requiere Matplotlib en el ordenador de análisis, no en la Pi
instalada. La traza JSONL no se distribuye con los agregados del repositorio.

```sh
python scripts/research/plot_pi_system.py /ruta/ensayo /tmp/telemetry.png
```

La comprobación `scripts/check-portal-import.py` prueba una subida real a través
del portal HTTP local y exige una biblioteca inicialmente vacía. Se ejecuta
como root para leer la clave local y verificar el GLB sin exponer credenciales.
Sube un OBJ con textura, comprueba su conversión y selección automática, y elimina
únicamente su paquete de prueba. El visor puede mostrar esa fixture durante el
ensayo. Al finalizar, verifica que la biblioteca vuelva a quedar vacía y sin
selección; no cambia los demás parámetros. No ejecutarlo sobre una biblioteca
con modelos ni mientras otra persona utilice el portal.

## Repetir las comprobaciones del portal

Estos ensayos usan exclusivamente el portal HTTP local, en el puerto 80 por
defecto, y la biblioteca estándar de Python. La clave de administración se lee
de `/etc/gestur/portal-token` y permanece en memoria. Ambos esperan hasta 15
segundos a que `GET /api/session` responda antes de iniciar sesión; las llamadas
posteriores conservan sus errores y límites propios.

```sh
sudo python3 scripts/check-portal-import.py > /tmp/portal-import-result.json
sudo python3 scripts/research/smoke_portal_api.py > /tmp/portal-api-result.json
```

`check-portal-import.py` conserva el ensayo de importación ejecutado en la Pi,
con la espera inicial añadida para repetirlo inmediatamente después de un
reinicio del servicio. Comprueba dos triángulos, UV, material y los bytes de la
textura PNG, incluida la reparación de su ruta. La primera importación debe
seleccionarse automáticamente. La limpieza comprueba el marcador, el nombre y
el hash de su paquete, y lo conserva si aparecen otros modelos, subidas o cambios
de configuración. Lo retira mediante la API de modelos, verifica la selección vacía y
comprueba que la configuración inicial se haya restaurado sin escribir una copia
antigua sobre los ajustes actuales. Se mantiene en `scripts/` sin duplicarlo aquí.

`smoke_portal_api.py` conserva el ensayo de sesión, biblioteca vacía, runtime,
configuración y Wi-Fi. Obtiene el SSID únicamente del AP configurado que expone
`/api/wifi`; exige formato `GESTUR-[A-F0-9]{4}`, red abierta y activa, y ausencia
de otro cambio Wi-Fi pendiente antes de modificarlo. No busca otras redes ni
adapta el ensayo a un AP personalizado. Añade, cambia y elimina una contraseña
aleatoria, e intenta restaurar y verificar la red abierta original incluso
cuando una comprobación falla; informa si no puede confirmarlo. Las contraseñas
no se guardan y el JSON omite el SSID literal. Ejecutarlo desde Ethernet, porque
los cambios reinician el AP.

`--help` muestra las rutas y opciones locales disponibles. Son pruebas con
efectos reales al ejecutarlas; la lectura de esta documentación no las inicia.

## MobRecon en ONNX

`benchmark_mobrecon.py` era `benchmark_onnx.py` durante la medición. Necesita
Python 3.12, `numpy==1.26.4` y `onnxruntime==1.30.0` en **otro entorno virtual**.
No instalar estas dependencias en `.venv` de producción. No necesita PyTorch,
OpenMesh ni MANO para ejecutar el ONNX ya preparado.

Pasar rutas explícitas al modelo exportado y su referencia PyTorch:

```sh
python scripts/research/benchmark_mobrecon.py --model /ruta/mobrecon_dsconv.onnx --reference /ruta/pytorch-reference.npz --threads 1 --frames 180 --warmup 20 --output /tmp/mobrecon-1.json
python scripts/research/benchmark_mobrecon.py --model /ruta/mobrecon_dsconv.onnx --reference /ruta/pytorch-reference.npz --threads 2 --frames 180 --warmup 20 --output /tmp/mobrecon-2.json
```

Son procesos independientes sobre un recorte de una mano. La espera activa está
desactivada por defecto. Se comprueban salidas finitas y error máximo frente a
PyTorch menor que `1e-4`: paridad de exportación, no precisión de reconocimiento.
Los archivos necesarios no están incluidos; sus hashes están en
[mobrecon-provenance.json](../../docs/benchmarks/pi5/mobrecon-provenance.json).

## Reconstruir la exportación

Copiar `scripts/research/mobrecon/` a un directorio de trabajo externo al checkout.
Su [manifest.json](mobrecon/manifest.json) fija el commit oficial
`3c87e958d4855f890e3884ec94bcfc0f99422c3d`, las URLs y los SHA-256. Se incluyen
únicamente el código necesario y su [licencia MIT](mobrecon/source/LICENSE), sin
modificar los archivos de los autores.

Para reproducir la muestra hay que obtener de esas fuentes y verificar por hash:

- `mobrecon_densestack_dsconv.pt`, junto a `export_mobrecon.py`.
- `source/template/transform.pkl` y `source/template/j_reg.npy`.
- `hand-crop.jpg`, recorte de los autores usado en la prueba.

No se incluyen pesos, plantillas, regresor, imágenes ni resultados binarios.
El permiso de distribución de las plantillas y el regresor sigue pendiente de
revisión; este experimento no concede derechos sobre esos recursos. No requiere
descargar `MANO_RIGHT.pkl`, aceptar una licencia con cuenta ni ejecutar Blender.

La preparación se realizó en macOS ARM64 con CPU; las versiones están en
[requirements-preparation.txt](mobrecon/requirements-preparation.txt). Es un
registro del entorno, no un instalador de producción ni una garantía de wheels
para todas las plataformas. Usar un entorno aislado en la máquina de preparación.
OpenMesh 1.2.1 necesitó compilación local con CMake previamente instalado y
`--no-build-isolation`; no es necesario compilarlo en la Pi para medir ONNX.

Desde ese directorio externo y con su Python de preparación:

```sh
python prepare_geometry.py
python export_mobrecon.py --image hand-crop.jpg --threads 1 --frames 30 --warmup 5 --export mobrecon_dsconv.onnx --output pytorch-results.json
```

El segundo comando genera también `pytorch-reference.npz`. La geometría conserva
el algoritmo espiral oficial; el checkpoint se carga con `weights_only=True` y
comprobación estricta de parámetros. La salida es una malla relativa de 778
vértices y 21 puntos 2D. No incluye detector, asociación de manos, alineación
absoluta con la cámara ni evaluación de orientación 3D.

## Comparación facial con cámara

`benchmark_face_camera.py` necesita las dependencias de seguimiento ya usadas por
Gestur: MediaPipe 0.10.18, NumPy 1.26.4 y OpenCV 4.11.0.86. Puede ejecutarse con
el Python existente sin instalar nada. Usa la geometría del checkout indicado
por `--project`; el valor histórico por defecto es `/opt/gestur` y se puede cambiar.

```sh
.venv/bin/python scripts/research/benchmark_face_camera.py --project "$PWD" --camera 0 --seconds 45 --fps 3 --warmup 3 --face-model /ruta/face_landmarker.task --output /tmp/face-camera.json
```

La cámara debe estar libre, con una persona visible. El script nunca descarga
modelos ni guarda imágenes o coordenadas; conserva estadísticas. `--face-model`
es opcional y añade FaceLandmarker a Pose y los dos detectores faciales. Todas
las redes reciben el mismo fotograma y permanecen residentes: CPU y RSS conjuntos
no representan un único backend. No mide precisión 3D ni latencia cámara→pantalla.
Los tiempos por red excluyen captura y preparación RGB compartida.

## Comparaciones de renderizado

Estos tres scripts conservan exactamente las versiones temporales utilizadas en
la Pi. Se ejecutan desde su sesión gráfica, con el visor habitual detenido por el
operador, y usan las dependencias existentes. No modifican archivos ni parámetros
de producción, VSync, geometría, texturas o antialiasing. Los modelos e imágenes de
prueba se proporcionan mediante rutas locales; no están incluidos.

| Script | Comparación y alcance de la medición |
| --- | --- |
| `gestur_clock_probe.py` | Modos `limited60`, `normal`, `limited61` y `limited120`; conserva el callback de dibujo y registra el modo en el manifiesto. |
| `gestur_render_only_probe.py` | Sin captura ni inferencia, en pantalla completa. Exige 1920×1080 y MSAA real 4×, conserva el modelo y registra su inventario real. CPU y FPS excluyen los 0,5 s de calentamiento; los percentiles de dibujo los incluyen. |
| `gestur_no_draw_callback_probe.py` | Elimina únicamente el callback de dibujo. Cuenta desde una tarea posterior a `igLoop` cuando la ventana está activa y es válida: **envíos estimados**, no recorridos de dibujo comprobados ni fotogramas presentados físicamente. |

En el tercer script, los nombres heredados `render_fps`, `frames` y `frame_ms_*`
mantienen esa interpretación de envíos en todos los resultados, incluida la
telemetría. Su manifiesto lo indica expresamente. Los otros dos scripts cuentan
el callback del recorrido de dibujo; tampoco miden presentación física, tiempo
de GPU ni latencia óptica. No interpretar una frecuencia de GPU observada como
porcentaje de utilización.

Ejemplos desde el checkout instalado, usando directorios de salida nuevos:

```sh
.venv/bin/python scripts/research/gestur_clock_probe.py --project-root "$PWD" --clock-case limited60 --source replay --image /ruta/imagen.jpg --model /ruta/modelo.glb --motion continuous --duration 20 --output-dir /tmp/clock-limited60
.venv/bin/python scripts/research/gestur_render_only_probe.py --project-root "$PWD" --model /ruta/modelo.glb --seconds 20 --output /tmp/render-only.json
.venv/bin/python scripts/research/gestur_no_draw_callback_probe.py --project-root "$PWD" --source replay --image /ruta/imagen.jpg --model /ruta/modelo.glb --motion continuous --duration 20 --output-dir /tmp/no-draw-callback
```

Las comparaciones breves en la Pi no mostraron una mejora relevante al retirar
el límite del reloj o el callback. No justifican cambiar la configuración ni la
instrumentación del producto. El ensayo sin reconocimiento es un diagnóstico de
carga distinta, no una promesa de alcanzar 60 FPS con reconocimiento activo.

SHA-256 de los archivos originales de `/tmp` y de estas copias:

| Archivo | SHA-256 |
| --- | --- |
| `gestur_clock_probe.py` | `418aef5c7f7da031f828d15898bc4692df870610fbe4ab46164520294c947bdd` |
| `gestur_render_only_probe.py` | `ef9c34d531838c52e6890eba65e416830c14085f4b06d737e5e5149b67d8638a` |
| `gestur_no_draw_callback_probe.py` | `b0cb1deb8c2e106dff3a6fef84845da76b4e57aa77ad264407b1654e300bfb35` |
