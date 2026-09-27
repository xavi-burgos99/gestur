# Alternativas de seguimiento de manos para Raspberry Pi 5

Investigación del 27 de septiembre de 2026. Objetivo: una cámara RGB, dos manos,
posición y orientación tridimensional, junto al visor de Gestur en Raspberry Pi
OS Lite de 64 bits. No se ha utilizado una Raspberry física en estas pruebas.

**Hay arquitecturas alternativas reales, pero esta investigación no demuestra
todavía que alguna reconozca mejor y consuma menos que el sistema actual en la
Pi.** MobRecon merece una comparación como modelo distinto. LiteRT directo es
la vía más concreta para reducir el entorno de ejecución, conservando por ahora
las redes actuales. Son dos experimentos diferentes; cambiar el motor no equivale
a cambiar el modelo.

## Qué necesita conservar Gestur

Los controles necesitan orientación de palma, profundidad relativa, lateralidad,
pinza y apertura, además de posición en imagen. Un estimador exclusivamente 2D
no sustituye por sí solo esos datos 3D. La evaluación debe incluir detección inicial,
recortes, dos manos, asociación temporal y recuperación tras perderlas.

El bundle actual de manos ocupa **4.055.094 bytes**. No es el tamaño del programa:
las dependencias, la memoria del proceso y el tiempo de CPU se miden por separado.
Más parámetros o un archivo mayor tampoco prueban por sí solos mayor latencia.

## Modelos con arquitecturas diferentes

| Alternativa | Evidencia comprobada | Encaje y decisión |
| --- | --- | --- |
| **MobRecon, DenseStack + DSConv** | Reconstrucción 3D monocular; código y checkpoint públicos. El artículo da 8,1 M parámetros y 439,9 M operaciones multiplicación-suma para esta variante completa. | Primer candidato experimental de arquitectura distinta. Falta convertir y medir el conjunto en Pi; no hay mejora demostrada frente a nuestro Lite. |
| **MobileHand** | Reconstrucción 3D monocular, pesos de unos 15,15 MB. El rendimiento anunciado de 75 FPS CPU corresponde a un Ryzen 7 3700X. | Candidato secundario. Necesita MANO y aclarar la licencia del código antes de incorporarlo al producto. Los 75 FPS no son de Raspberry. |
| **UmeTrack, de Meta** | Pesos públicos de 17,13 MB y seguimiento temporal 3D, con una o varias vistas. | Interesante para investigación, menos directo para nuestra webcam: entrenamiento egocéntrico, calibración y regiones de mano proporcionadas al tracker. |
| **SPLite Hand** | El artículo mide 15 FPS de inferencia en CPU de Pi 5 y un modelo INT8 de 18 MB. | Es evidencia específica de Pi, pero no he localizado código y pesos públicos para reproducirlo. No compara contra MediaPipe. |
| **RTMPose-Hand** | La conversión LiteRT publicada pesa 27,83 MB y produce 21 puntos **2D**. Su autor mide 58,1 ms en Pi 5 con cuatro hilos, por mano recortada. | No cubre directamente los giros 3D. La cifra excluye detección, segunda mano y visor; no justifica esta sustitución. |

Fuentes y condiciones de las comparaciones:

- **MobRecon:** [artículo CVPR](https://openaccess.thecvf.com/content/CVPR2022/html/Chen_MobRecon_Mobile-Friendly_Hand_Mesh_Reconstruction_From_Monocular_Image_CVPR_2022_paper.html),
  [código y recursos oficiales](https://github.com/SeanChenxy/HandMesh).
  La variante más rápida del artículo, GhostStack + DSConv, tiene 5,3 M parámetros
  y anuncia 83 FPS en CPU Apple A14; no he localizado su checkpoint. El resultado
  de DenseStack + DSConv es 67 FPS en ese A14, no en Pi. El checkpoint identificado
  [mobrecon_densestack_dsconv.pt](https://drive.google.com/file/d/1qqEstFnV3GClpGWNEAZmR0jnxrcZH-9y/view)
  ocupa 45.656.319 bytes, según el listado del almacenamiento oficial.
  No confundirlo con el SpiralConv recomendado por el
  repositorio para imágenes reales. El código usa licencia MIT; su preparación
  también requiere MANO, con condiciones propias. No se ha ejecutado ni convertido
  este modelo durante la investigación.
- **MobileHand:** [repositorio](https://github.com/gmntu/mobilehand) y artículo
  enlazado allí. Incluye demostración de webcam, pero no una comparación contra
  Gestur ni una validación de consumo térmico. No se ha encontrado una licencia
  explícita en el repositorio revisado. El hardware de la cifra de 75 FPS consta
  en el [artículo de los autores](https://www.researchgate.net/publication/347025951_MobileHand_Real-Time_3D_Hand_Shape_and_Pose_Estimation_from_Color_Image).
- **UmeTrack:** [repositorio y licencia](https://github.com/facebookresearch/UmeTrack),
  [artículo](https://arxiv.org/abs/2211.00099) y
  [tracker](https://github.com/facebookresearch/UmeTrack/blob/main/lib/tracker/tracker.py).
  El modo monocular del código requiere un esqueleto conocido; la estimación de
  tamaño desconocido utiliza varias vistas. No aporta una cadena completa de
  localización inicial para nuestra cámara. No he encontrado un benchmark CPU ARM
  comparable. El repositorio está archivado y utiliza CC BY-NC 4.0.
- **SPLite:** [artículo, tablas 2 y 4](https://arxiv.org/html/2510.16396v3).
  Son 50 repeticiones del paso de la red: 15 FPS frente a 6 de su MobRecon base
  y 35 de MobileHand. Hay afirmaciones del texto que no coinciden con esa tabla;
  aquí se conservan los números tabulados. No mide detección, dos manos, visor ni
  estabilidad térmica prolongada. Sus resultados no permiten declarar mejor
  precisión que MediaPipe en los movimientos de Gestur.
- **RTMPose:** [modelo y prueba del autor de la conversión](https://huggingface.co/litert-community/RTMPose-Hand-LiteRT).
  Publica 138 MB de pico de memoria y 58,1 ms con una versión nightly de LiteRT.
  La misma página conserva un fallo de XNNPACK en otra configuración: la
  compatibilidad depende del runtime. Se han inspeccionado los grafos ONNX
  oficiales: el estimador tiene 55.080.248 bytes y salida SimCC X/Y; el detector
  separado, 4.010.667 bytes. El estimador ejecutó una inferencia sintética con
  OpenCV 4.11.0 y devolvió tensores finitos; el detector falló al importar `TopK`.
  Esto no evalúa precisión. El JSON auxiliar de tamaño de
  entrada del ZIP no coincide con el grafo; debe respetarse el tensor real de
  256 × 256.

## Qué ha publicado Meta y qué ofrecen las soluciones de VR

La intuición sobre Meta tiene fundamento: **UmeTrack sí publica un modelo de
seguimiento de manos**. Otros anuncios se refieren a datos o a funciones de las
gafas, no a una biblioteca intercambiable para Raspberry:

- [HOT3D](https://github.com/facebookresearch/hot3d) y
  [hand_tracking_toolkit](https://github.com/facebookresearch/hand_tracking_toolkit)
  aportan datos, geometría, lectura y evaluación. No son un nuevo detector listo
  para una webcam.
- [Aria Gen 2](https://facebookresearch.github.io/projectaria_tools/gen2/ark/device/on_device_mp)
  obtiene seguimiento de manos mediante aceleración en las gafas. Su frecuencia
  de salida no representa tiempo de inferencia en CPU de Raspberry.
- [Quest](https://developers.meta.com/horizon/documentation/unity/unity-handtracking-overview/)
  expone el seguimiento mediante el sistema del visor. En la documentación
  revisada no aparece un runtime autónomo con pesos para nuestra cámara USB.
- [SAM 3D Body](https://github.com/facebookresearch/sam-3d-body) sí ofrece código
  y checkpoints, pero sus backbones son de 631 M y 840 M parámetros. No lo
  seleccionaría para este objetivo de bajo consumo.
- Fuera de Meta, [Monado Mercury](https://monado.freedesktop.org/handtracking/)
  es seguimiento abierto con inferencia CPU, pero su integración publicada utiliza
  cámaras estéreo calibradas. No sustituye directamente una única webcam RGB.

También se revisaron [LiteSpiralGCN](https://github.com/minqili/LiteSpiralGCN) y
[Fast-HaMeR](https://github.com/hunainahmedj/Fast-HaMeR). No se encontró una prueba
reproducible de ventaja en CPU Pi frente al sistema actual. LiteSpiralGCN incluye
dependencias geométricas y MANO. Fast-HaMeR compara con HaMeR, mucho más pesado;
no se localizó un checkpoint público del estudiante listo para esta comparación.
El nombre «Lite» o una mejora frente a un modelo grande no bastan para elegirlos.

## Prueba local de un motor más pequeño: LiteRT

También se revisaron [ncnn](https://github.com/Tencent/ncnn) y
[MNN](https://github.com/alibaba/MNN), motores orientados a ejecución en
dispositivos. Son alternativas de despliegue, no estimadores de manos por sí
solos. ncnn permite una integración C++ con pocas dependencias, pero habría que
convertir los modelos y verificar operadores y resultados. No se midieron aquí;
no hay base para afirmar que sean más rápidos que LiteRT en esta aplicación.
Se priorizó LiteRT para la prueba porque acepta los archivos actuales sin
convertir sus pesos.

[LiteRT](https://developers.google.com/edge/litert) ejecuta modelos TFLite sin
necesitar toda la API de MediaPipe. La
[distribución 2.2.0](https://pypi.org/project/ai-edge-litert/2.2.0/) ofrece wheel
oficial para Linux ARM64; la de Python 3.12 ocupa unos 14,1 MB comprimida.

Se probaron los dos archivos Lite actuales en un entorno temporal: carga e
inferencia CPU correctas, incluidas las salidas de 21 puntos 3D, presencia y
lateralidad. **Esto valida compatibilidad de las redes, no un tracker terminado
ni precisión con imágenes reales.** Gestur sigue utilizando MediaPipe.

Equipo local: Apple M2 Pro, macOS ARM64, Python 3.12.14, NumPy 1.26.4,
OpenCV 4.11.0.86. Procesos independientes, ambos intérpretes retenidos,
20 inferencias de calentamiento y 180 medidas; tensores sintéticos constantes.
Valores medianos en milisegundos:

| Red | Hilos | Tiempo transcurrido | Tiempo de CPU del proceso |
| --- | ---: | ---: | ---: |
| Detector de palma | 1 | 10,33 | 10,24 |
| Detector de palma | 2 | 6,04 | 11,51 |
| Puntos de una mano | 1 | 4,89 | 4,85 |
| Puntos de una mano | 2 | 2,81 | 5,40 |

Dos hilos reducen la espera, pero aumentan el trabajo de CPU por inferencia en
esta prueba. No se han medido vatios ni temperaturas. Para priorizar consumo,
compararía primero un hilo con dos a la misma frecuencia de seguimiento.

El pico RSS del proceso LiteRT con ambas redes y OpenCV fue **109,5 MB** con un
hilo. El proceso MediaPipe 0.10.18 con la cadena HandLandmarker de dos manos
y sus hilos por defecto alcanzó **178,2 MB**. **No son funcionalidades equivalentes:** LiteRT aún carece de
decodificación de detecciones, NMS, recortes orientados y asociación temporal.
La diferencia no debe anunciarse como ahorro de memoria del producto terminado.
El benchmark MediaPipe dio 10,72 ms por llamada a `detect_for_video` en modo VIDEO
sobre una imagen estática preparada. No incluye captura, conversión de imagen
por fotograma, seguimiento de cabeza ni visor; tampoco se compara directamente
con una sola invocación de la red.

En disco, JAX, JAXlib, SciPy y Matplotlib suman aproximadamente 391 MiB instalados
en el entorno macOS actual. JAX y SciPy no estaban cargados al importar MediaPipe:
quitar esas dependencias reduce instalación, pero no demuestra ahorro equivalente
de RAM ni CPU. Mientras el seguimiento de cabeza siga usando MediaPipe, cambiar
solo manos tampoco elimina toda esa dependencia del proyecto.

Resultados originales: [mediciones JSON y procedimiento](benchmarks/hand-runtime-2026-09-27/README.md).
Herramienta: [benchmark_hand_runtime.py](../scripts/benchmark_hand_runtime.py).

## Decisión y prueba que falta

1. **Probar MobRecon DenseStack + DSConv como alternativa 3D real.** Exportar
   primero una inferencia CPU mínima, verificar sus puntos y recortes, y compararla
   con Lite sobre las mismas secuencias. Sus operaciones de muestreo y malla
   requieren comprobar compatibilidad y rendimiento al exportar; no se ha
   localizado un ONNX oficial preparado. La promesa de eficiencia del artículo
   justifica experimentar; no justifica sustituir ya el detector.
2. **Probar LiteRT directo como reducción de dependencias y control de ejecución.**
   Portar detección, recortes, asociación y recuperación conservando los puntos
   tridimensionales; verificar equivalencia antes de medir. Es otra biblioteca,
   con el mismo modelo, y debe describirse así.
3. **Elegir con el conjunto funcionando en la Pi.** Comparar cero, una y dos
   manos; cabeza activa; giros combinados, perfil, oclusión, salida y reentrada.
   Usar secuencias con orientación de referencia y anotaciones de pinza y apertura:
   una predicción más estable también puede ser sistemáticamente incorrecta.
   Medir CPU total, RSS, latencia cámara→movimiento p50/p95, saltos de orientación,
   frecuencia del visor, temperatura y throttling durante al menos 30 minutos,
   con la misma cámara, calidad gráfica, refrigeración y frecuencia de inferencia.

La sustitución se aceptaría si mejora el consumo sostenido manteniendo la
precisión y continuidad necesarias, o mejora claramente los giros dentro del
presupuesto de recursos. No se ha incorporado un nuevo modelo al runtime, no se
han cambiado dependencias de la aplicación y no se ha modificado `main`.
