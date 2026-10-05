# Seguimiento: geometría y comparación de modelos

Resultados locales del 27 de septiembre de 2026. Equipo: Apple M2 Pro, macOS
ARM64, Python 3.12.14, MediaPipe 0.10.18, NumPy 1.26.4 y OpenCV 4.11.0.
La inferencia utiliza el delegado CPU. **Estos valores no son mediciones de
Raspberry Pi 5 ni una evaluación de precisión con usuarios.**

La investigación posterior de arquitecturas distintas, proyectos de Meta y
motores de ejecución está en
[Alternativas de seguimiento de manos](hand-tracking-alternatives.md).

## Corrección de orientación de la mano

El cálculo anterior extraía pitch y yaw mediante dos `atan2` independientes sobre
la normal de la palma. Una rotación combinada de muñeca alteraba ambos resultados,
aunque uno de esos ejes no hubiese cambiado en el marco elegido.

Ahora se construye un marco ortogonal con la muñeca, el nudillo del dedo medio y
los nudillos del índice y el meñique. Gram-Schmidt elimina la componente de la
línea de nudillos paralela a los dedos. El lado de la mano procede del resultado
del modelo, antes de corregir el nombre anatómico por el espejo.

La convención en coordenadas de cámara —X hacia la derecha, Y hacia abajo, Z
hacia el fondo— es `Rz(roll) · Ry(-yaw) · Rx(pitch)`. Mantiene el signo anterior
para rotaciones simples y separa correctamente los tres ejes en rotaciones
combinadas, dentro de esta convención:

- `pitch` y `roll`: entre −180° y 180°.
- `yaw`: entre −90° y 90°.
- En yaw = ±90°, pitch y roll no son independientes. Se escoge roll = 0 y un
  pitch que representa el mismo marco. Los ángulos Euler no garantizan
  continuidad al atravesar esta singularidad.
- `rotation` conserva el giro de la línea muñeca–dedo medio proyectada en la
  imagen, utilizado por los controles existentes. Es distinto del nuevo `roll`
  tridimensional cuando se combinan giros.
- Si esa línea apunta hacia la cámara, `rotation` pasa a `None`; la orientación
  tridimensional, la pinza y la apertura pueden seguir siendo válidas.

Las pruebas de geometría usan manos sintéticas transformadas rígidamente. Cubren
los tres ejes, giros combinados, ambas manos, espejo, perfil, degeneración de la
palma y el paso por ±180°. Demuestran el cálculo matemático; no demuestran que los
landmarks estimados por la cámara sean exactos.

## HandLandmarker Lite frente a Full

Se usa [woman_hands.jpg, recurso oficial de MediaPipe](https://storage.googleapis.com/mediapipe-assets/woman_hands.jpg),
ajustado sin deformación sobre un lienzo blanco de 640 × 480. Cada modelo procesa
120 fotogramas medidos después de 10 de calentamiento, en modo `VIDEO`, dos manos
y umbrales de detección, presencia y seguimiento de 0,5.

Hay dos escenarios: imagen estática y movimiento afín determinista, con giro en
el plano de ±20°, pequeñas variaciones de escala y traslación. **Esta
transformación bidimensional no simula inclinación tridimensional de la mano.**

| Modelo | Escenario | Dos manos detectadas | Mediana | p95 | CPU mediana |
| --- | --- | ---: | ---: | ---: | ---: |
| Lite | Estático | 120/120 | 11,18 ms | 12,59 ms | 12,04 ms |
| Full | Estático | 120/120 | 22,94 ms | 24,64 ms | 23,63 ms |
| Lite | Movimiento afín | 120/120 | 11,22 ms | 13,06 ms | 12,09 ms |
| Full | Movimiento afín | 120/120 | 22,83 ms | 24,50 ms | 23,65 ms |

El tiempo incluye conversión BGR→RGB, creación de la imagen MediaPipe e
inferencia. Excluye la transformación sintética y el análisis geométrico.

Para comparar repetibilidad, se centra cada mano en la muñeca, se normaliza por
su anchura de palma y se deshace el giro conocido de la imagen. El desplazamiento
RMS mediano de los 21 puntos entre fotogramas consecutivos es:

| Modelo | Estático | Movimiento afín |
| --- | ---: | ---: |
| Lite | 0,00281 anchos de palma | 0,01758 anchos de palma |
| Full | 0,00246 anchos de palma | 0,01274 anchos de palma |

Full ofrece menor variación en esta muestra, con aproximadamente el doble de
coste de inferencia. No mejora todas las métricas: por ejemplo, el p95 del cambio
de roll compensado con movimiento pasa de 0,47° a 0,52°. Un resultado más estable
puede seguir siendo incorrecto. Sin anotaciones tridimensionales o un vídeo real
de referencia, estos datos no permiten afirmar una mejora general de precisión.

El modelo [Full oficial](https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker)
pesa 7.819.105 bytes, SHA-256
`fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1`.
El bundle Lite preparado por el proyecto pesa 4.055.094 bytes. Ambos utilizan la
misma familia de modelos; Full añade capacidad, no una arquitectura nueva.

El script no abre la cámara ni descarga archivos. Para repetir con bundles e
imagen guardados localmente:

```bash
python scripts/benchmark_hand_models.py \
  --image /ruta/woman_hands.jpg \
  --model lite=tracking_models/hand_landmarker_lite.task \
  --model full=/ruta/hand_landmarker_full.task \
  --width 640 --height 480 --frames 120 --warmup 10 \
  --output /ruta/comparacion-manos.json
```

El informe incluye versiones, hashes de todos los archivos, continuidad de
detección, tiempo CPU y percentiles de variación. El hash de la imagen oficial
usada aquí es `70cbeb38e198c9862202e0979c21a99b40ca980d3e7b250176c85b1636a40f12`.

## FaceLandmarker como sustituto de Pose Lite

Sobre el recurso oficial `pose.jpg` a 640 × 427, con espejo, Pose Lite detectó una
cabeza válida en los 60 fotogramas medidos después de 10 de calentamiento.
FaceLandmarker detectó cero, con una cara, modo `VIDEO`, expresiones desactivadas
y matriz facial activada. Se repitió sin espejo con el mismo resultado.

Pose Lite necesitó una mediana de 10,04 ms, p95 10,72 ms. Los 1,71 ms de
FaceLandmarker corresponden a detección fallida, **no a seguimiento más rápido**.
Por eso no se sustituye universalmente el modelo de cuerpo por el facial.

También se probó `mp.solutions.face_detection.FaceDetection(model_selection=1)`
con el detector Full Range Sparse incluido en MediaPipe 0.10.18: 676.746 bytes,
SHA-256 `2c3728e6da56f21e21a320433396fb06d40d9088f2247c05e5635a688d45dfe1`.
Sobre la misma imagen no detectó una cara válida en ninguno de los 60 fotogramas,
con o sin espejo; medianas de 4,93 y 4,91 ms. Pose Lite detectó la cabeza en
60/60, con medianas de 10,26 y 10,46 ms en esa ejecución.

Reducir la imagen al 75% y al 50% dentro del mismo lienzo, con y sin espejo,
produjo el mismo resultado: 0/60 para el detector facial y 60/60 para Pose Lite
en cada combinación. Esto comprueba menor tamaño aparente, no distancia física
ni precisión de los puntos. La validez facial requería seis puntos finitos y
distancia entre ojos distinta de cero, con confianza mínima de 0,5. No se
redujeron umbrales para forzar detecciones. Aunque bastasen nariz y ojos para
los controles de posición y escala, esta muestra tampoco respalda sustituir
Pose Lite por Full Range Sparse.

## MediaPipe 1.0.1 y otras alternativas

La [distribución oficial 1.0.1](https://pypi.org/project/mediapipe/1.0.1/)
incluye una wheel Linux ARM64. Se probó en un entorno separado bajo `/tmp`,
igualando NumPy, OpenCV y Matplotlib a las versiones anteriores. No se modificó
el entorno del proyecto.

En este macOS, tanto el bundle Lite como el Full oficial abortan al crear el
HandLandmarker CPU porque un servicio nativo Metal no está disponible. Además,
el import de `MetadataWriter` utilizado por el provisionador actual falla porque
el paquete ya no incluye `mediapipe.tasks.cc`. No hay una medición válida que
justifique migrar en esta rama. El fallo local **no demuestra** que la versión
falle en Raspberry Pi OS: requiere una prueba ARM64 Linux independiente.

[OpenCV Zoo](https://raw.githubusercontent.com/opencv/opencv_zoo/main/models/handpose_estimation_mediapipe/README.md)
ofrece una conversión ONNX de los modelos MediaPipe. Cambiar el formato no aporta
una arquitectura distinta; el propio proyecto avisa de degradación importante de
precisión en la versión int8. No se utiliza como sustitución sin una comparación
favorable de coste y resultados.
