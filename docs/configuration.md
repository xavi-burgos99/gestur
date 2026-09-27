# Configuración de GESTUR

El visor y el portal comparten `/var/lib/gestur/config.json`. Se puede cambiar la ruta con `GESTUR_CONFIG` o con la opción de configuración del visor. Una instalación sin archivo utiliza `config/default.json`. El contrato versionado está en `config/schema.json` (JSON Schema Draft 7); `schema_version` debe ser `1`. El portal valida antes de guardar y sustituye el archivo de forma atómica. No se guardan contraseñas Wi-Fi en este archivo: las gestiona el servicio de red.

La configuración completa contiene `active_model`, `tracking`, `render` y `controls`. `active_model: null` es el valor inicial: muestra la bienvenida con figura procedural y QR, sin abrir la cámara. Al importar se guarda un GLB autocontenido y la selección usa `<uuid>/model.glb`; también se conservan los paquetes OBJ/glTF/GLB de instalaciones anteriores. La selección antigua `capitell.obj` se migra a `null`; no se borran modelos del usuario.

`GESTUR_PORTAL_URL` permite indicar la URL pública local del QR cuando se usa un proxy o una dirección distinta. Sin esa variable se detecta la dirección del punto de acceso o la LAN y se usa el puerto 3000. Nunca se añade la clave de administración al QR.

## Valores iniciales para Raspberry Pi 5

- Cámara: 640 × 480, inferencia de cabeza a un máximo de 24 Hz y manos a 15 Hz cuando están activadas.
- Manos desactivadas de inicio para reducir el trabajo. Activarlas permite usar posición, orientación de la palma, apertura y pinza.
- Renderizado: 60 FPS como objetivo, antialiasing MSAA de 2 muestras, pantalla completa y cursor oculto. Son objetivos configurables; el rendimiento real depende del modelo, la cámara y la Raspberry Pi.
- Suavizado del seguimiento: 60 ms. Suavizado de los controles: 90 ms; el zoom usa un tercio para conservar su respuesta más rápida.

Los cambios de controles se aplican al recargar la configuración. Los ajustes de cámara, seguimiento y creación de la ventana requieren reiniciar el visor; el controlador solicita el reinicio al servicio. Un archivo inválido se rechaza y el proceso en marcha conserva la última configuración válida.

## Asignaciones de movimientos

`controls.mappings` es una lista ordenada. Cada elemento tiene `id`, `input`, `output`, `mode`, `enabled`, `scale`, `invert` y `center`. Cada salida admite una sola asignación activa; los duplicados se rechazan. El campo `id` es único y admite letras, números, guiones y guiones bajos. Se permiten hasta 32 asignaciones.

| Entrada | Valor |
| --- | --- |
| `head_x`, `head_y` | Posición normalizada de la cabeza, normalmente de 0 a 1 |
| `head_scale` | Tamaño aparente de la cabeza, entre 0 y 1; aproxima el acercamiento y alejamiento, no mide profundidad. Mantiene la calibración del zoom de Capitell |
| `head_pitch`, `head_yaw`, `head_roll` | Giro vertical, giro horizontal e inclinación lateral de la cabeza, calculados con los puntos de seguimiento existentes. Los grados se normalizan a 0–1 con 0° en 0,5 y se suavizan por el ángulo más corto |
| `hands_center_x`, `hands_center_y` | Centro de las manos visibles |
| `hands_distance`, `hands_separation_x` | Distancia entre ambas manos o separación horizontal |
| `left_hand_x`, `left_hand_y`, `right_hand_x`, `right_hand_y` | Posición normalizada de la muñeca de cada mano, normalmente de 0 a 1 |
| `left_hand_rotation`, `right_hand_rotation` | Orientación de la palma en la imagen; 0° hacia arriba, +90° hacia la derecha. Se normaliza internamente a 0–1 y se suaviza por el ángulo más corto |
| `left_hand_pitch`, `right_hand_pitch` | Inclinación de la mano calculada con las coordenadas 3D del modelo; grados normalizados a 0–1, con suavizado angular |
| `left_hand_yaw`, `right_hand_yaw` | Giro lateral de la mano calculado con las coordenadas 3D del modelo; grados normalizados a 0–1, con suavizado angular |
| `left_hand_pinch`, `right_hand_pinch` | Distancia pulgar–índice dividida por el ancho de la palma: 0 es contacto, 1 es una apertura de al menos un ancho de palma |
| `left_hand_openness`, `right_hand_openness` | Proporción de dedos extendidos, sin contar el pulgar: 0 es puño y 1 son los cuatro dedos extendidos. El seguimiento suaviza los pasos intermedios |

Las entradas nuevas reutilizan el seguimiento actual; no cargan modelos adicionales. La posición y los giros de cabeza son señales distintas y pueden asignarse por separado. Una cabeza o mano no detectada produce una entrada ausente, nunca un cero fabricado. No hay una entrada de profundidad para las manos.

La inclinación `head_roll` se mide con la línea de los ojos, sin distinguir sus extremos: tiene periodo de 180° y se informa entre −90° y +90°. Sus dos filtros respetan ese periodo para evitar saltos al cruzar el límite. Conserva la misma escala numérica que los demás ángulos (45° equivale a 0,625), y al perder el seguimiento vuelve a una orientación neutra de 360°, no a un objeto invertido. `head_pitch` y `head_yaw` mantienen el periodo de 360°.

Las salidas `rotation_yaw`, `rotation_pitch` y `rotation_roll` corresponden a los tres canales de rotación que ya usaba el visor. `position_x`, `position_y` y `position_z` desplazan el objeto. `scale_uniform` cambia su escala.

- `absolute`: las rotaciones usan `(entrada - center) × 2 × scale` grados; los desplazamientos usan `(entrada - center) × scale`. El zoom usa `1 + (entrada - center) × scale`, limitado a 0,1–5. `invert` invierte el movimiento alrededor de `center`. Las entradas angulares de cabeza y manos pueden cruzar ±180° sin saltar al ángulo opuesto.
- `hybrid`: sólo admite salidas de rotación. En la zona central conserva el giro proporcional. Más allá de `left_threshold` o `right_threshold` añade giro continuo hasta `continuous_speed` grados por segundo. Debe cumplirse `left_threshold < center < right_threshold`.
- `stepped`: sólo admite `scale_uniform`. Alterna entre `small_scale` y `large_scale` con `threshold`, un margen `hysteresis` que evita alternancias por ruido, y una transición de `transition_ms`. `scale` modifica la sensibilidad alrededor del centro; `invert` cambia su dirección. El tamaño pequeño debe ser menor o igual al grande y `threshold ± hysteresis` debe permanecer entre 0 y 1.

La configuración inicial conserva cuatro controles: cabeza horizontal → giro continuo roll (umbrales 25 % y 75 %, máximo 100°/s, invertido); cabeza horizontal → yaw de hasta 10°; cabeza vertical → pitch de hasta 30°; tamaño de cabeza → zoom 1×/1,75× con umbral 0,4 y transición de 750 ms. El margen del zoom es 0,02.

El tiempo transcurrido, y no la cantidad de fotogramas, determina el suavizado y la velocidad. Si se pierde el seguimiento, el giro continuo se detiene inmediatamente. Tras `reset_timeout_seconds` (3 s), vuelve a la orientación neutra por el recorrido angular más corto durante `reset_duration_seconds` (1 s). Si la persona reaparece durante ese retorno, se continúa desde el ángulo que se ve en pantalla.

Ejemplo de una asignación adicional para girar con la palma derecha:

```json
{
  "id": "right_palm_roll",
  "input": "right_hand_rotation",
  "output": "rotation_roll",
  "mode": "absolute",
  "enabled": true,
  "scale": 180,
  "invert": false,
  "center": 0.5
}
```

Hay que activar `tracking.use_hands` y desactivar la asignación anterior que controle `rotation_roll`, antes de guardar. Para ampliar al cerrar la pinza, usar `right_hand_pinch`, salida `scale_uniform`, modo `absolute`, `scale: 2`, `center: 0.5` e `invert: true`.
