# Telemetría por ventanas de 60 segundos

| Ventana nominal, s | FPS de dibujo | CPU sistema media, % | Temperatura media / máx., °C | RSS medio / máx., MiB | Muestras hardware / estados FPS |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0–60 | 38,67 | 21,65 | 58,76 / 61,70 | 442,63 / 461,45 | 60 / 59 |
| 60–120 | — | 21,23 | 58,40 / 58,40 | 428,52 / 428,52 | 1 / 0 |

Los intervalos son [inicio, fin). FPS usa las diferencias entre los estados reales primero y último de cada ventana; los extremos exactos están en el CSV y el JSON. No interpola bordes ni añade un contador inicial. Una ventana final parcial se conserva como tal.

CPU: media aritmética de muestras válidas desde el segundo 5, porcentaje del sistema completo. Temperatura y RSS incluyen todas las muestras. Valores ausentes se omiten; «—» significa sin datos suficientes. El RSS es memoria residente del proceso en MiB. No son medidas de precisión ni latencia óptica.
