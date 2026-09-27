# Rendimiento del dibujo

El visor mantiene Panda3D, la geometría completa, las texturas originales, la resolución y el antialiasing configurados. El bucle de control conserva su objetivo de 60 Hz. Cuando cambia una transformación, la escena se dibuja en ese mismo ciclo. Un modelo inmóvil conserva la última imagen y se vuelve a dibujar a 10 Hz; la figura de bienvenida se anima a 30 Hz. Si el objetivo configurado es menor, se respeta ese límite.

## Cómo funciona

La tarea de control usa orden 10. La decisión de dibujar se toma en orden 49, inmediatamente antes de `igLoop`, que usa orden 50. Se mantienen las tareas normales de eventos, entrada y ventana de Panda3D. Su documentación describe este orden de ejecución y la función de cada tarea. [Tareas de Panda3D](https://docs.panda3d.org/1.10/python/programming/tasks-and-events/tasks).

El visor cambia `GraphicsOutput.set_active()` para omitir el dibujo en los ciclos sin cambios. Los cambios de modelo, pantalla de bienvenida, QR, configuración, tamaño y estado de ventana solicitan una actualización inmediata. Las transiciones relevantes dibujan dos ciclos para actualizar los buffers. El refresco de reposo también cubre exposiciones de ventana sin un cambio de propiedades. [GraphicsOutput](https://docs.panda3d.org/1.10/python/reference/panda3d.core.GraphicsOutput), [GraphicsWindow](https://docs.panda3d.org/1.10/python/reference/panda3d.core.GraphicsWindow).

Los FPS de dibujo se cuentan en el callback de dibujo de la región de la cámara principal. El callback llama a `upcall()` exactamente una vez para conservar el dibujo normal y después registra el tiempo. `get_render_status()` devuelve estos datos, los ciclos de control y los dibujos omitidos. La opción `--show-fps` muestra por separado dibujo y control. Son recorridos reales de dibujo; no representan una medición de la frecuencia física del monitor. [DisplayRegion](https://docs.panda3d.org/1.10/python/reference/panda3d.core.DisplayRegion).

El límite de FPS de Panda3D combina suspensión y espera activa. El código de `ClockObject.wait_until()` resta `sleep-precision` del tiempo de suspensión y espera activamente al final. El valor observado por defecto en Panda3D 1.10.16 era 10 ms. Se usa un margen de 4 ms en Linux y macOS para reducir ese trabajo. Windows conserva su configuración. [Código oficial de ClockObject 1.10.16](https://github.com/panda3d/panda3d/blob/v1.10.16/panda/src/putil/clockObject.cxx#L455-L484).

El margen se escogió midiendo CPU y regularidad del control en el **Apple M2 Pro**. En ese equipo, 1 ms redujo la CPU pero también bajó el control a unos 54 Hz, con un percentil 95 de unos 19,6 ms. Los márgenes de 2 y 3 ms tampoco mantuvieron igual regularidad. Con 4 ms el control volvió a aproximadamente 60 Hz, conservando un ahorro de CPU importante en esa prueba.

## Medición reproducible

Desde el entorno Python de Gestur, con acceso al backend gráfico de Panda3D:

```bash
.venv/bin/python scripts/benchmark_render.py --seconds 5 --output render-results.json
```

La prueba genera temporalmente un modelo de 524.288 triángulos con una textura de 2048 × 2048. Solicita un framebuffer de 1920 × 1080 con MSAA 2× y registra las muestras obtenidas. No abre una ventana de escritorio ni inicia la cámara. Compara el dibujo continuo y margen anterior, el cambio de reloj aislado, el reposo optimizado, el movimiento continuo y la bienvenida. Cada escenario tiene calentamiento previo. El archivo temporal se elimina al terminar.

Ejemplo medido el 27 de septiembre de 2026: Apple M2 Pro, macOS 27.0 arm64, Panda3D 1.10.16; tres segundos por escenario:

| Escenario | Dibujos | Ciclos de control | Dibujo por segundo | Control por segundo | CPU del proceso, % de un núcleo | Intervalo de control p95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Reposo anterior, continuo, margen 10 ms | 180 | 180 | 59,829 | 59,829 | 51,01 % | 16,718 ms |
| Solo reloj de 4 ms, dibujo continuo | 180 | 180 | 59,999 | 59,999 | 11,52 % | 16,692 ms |
| Reposo optimizado | 30 | 180 | 10,000 | 59,999 | 10,16 % | 16,692 ms |
| Movimiento continuo optimizado | 179 | 179 | 59,647 | 59,647 | 11,85 % | 16,824 ms |
| Bienvenida optimizada, 1.536 triángulos | 90 | 180 | 29,940 | 59,881 | 13,18 % | 16,728 ms |

En esta muestra, el reposo necesitó seis veces menos recorridos de dibujo y el tiempo de CPU del proceso bajó aproximadamente un 80 %. El movimiento siguió dibujándose en todos los ciclos de control. Estos resultados corresponden a una carga sintética en el M2 Pro; ese porcentaje no se extrapola a la Pi. El valor de CPU usa `time.process_time()`, normalizado a un núcleo; no incluye una medición del uso o la energía de la GPU.

## Mediciones posteriores en Pi 5 física

En la Pi 5 de 4 GB con V3D 7.1.7.0, el capitel real de 491.038 triángulos y textura 2048 × 2048 alcanzó **38,62 FPS** con cámara, pose y manos durante 60 segundos. La pantalla era de 1920 × 1080; se solicitaron 2 muestras de MSAA y se obtuvieron **4**. La CPU media desde el segundo 5 fue 73,45 % de un núcleo y la temperatura máxima, 59,5 °C. Se mantuvieron geometría, textura, sincronización vertical y calidad gráfica.

Cuatro ajustes del reloj con replay y movimiento continuo dieron 39,32–39,54 FPS, sin mejora útil. El ensayo sin captura ni inferencia dio 45,549 FPS y CPU de 5,35 % de un núcleo durante 20 segundos. Retirar el callback produjo 39,39 envíos estimados por segundo; esa última cifra no verifica recorridos de dibujo ni presentación física. Se mantienen el reloj y el callback del producto. No se ha determinado el cuello de botella ni se ha medido ocupación de GPU.

Consulta las [condiciones, scripts y resultados originales](pi5-validation.md). Para repetir el caso con tu modelo, desde una sesión gráfica acelerada y con la cámara libre:

```bash
.venv/bin/python scripts/benchmark_pi_system.py --source camera --duration 60 --model /ruta/capitel-original/capitell.obj --motion controls --output-dir /tmp/capitel-render-new
```

La [guía de medición](performance.md) explica el registro térmico y el corte por temperatura superior a 85 °C. El ensayo sostenido completó **30 minutos con replay y giro continuo**, a **39,18 FPS**, con CPU total media de **20,53 %** y temperatura máxima de **60,6 °C**. Las 1.800 lecturas de throttling fueron cero. Conservó el capitel, textura, 1080p y MSAA real de 4 muestras; los [resultados por minuto](benchmarks/pi5/capitel-endurance/windows-60s.md) identifican la carga y sus límites. El reinicio del último despliegue sigue pendiente. No se han medido consumo eléctrico ni latencia física de presentación.

## Comprobaciones

```bash
.venv/bin/python -m pytest tests/test_render_scheduling.py -q
```

Las pruebas verifican la cadencia con variaciones del reloj y usan Panda3D con un framebuffer offscreen real. Comprueban que la imagen permanece idéntica durante los ciclos inactivos, que un cambio de control se ve en el mismo ciclo, que se conserva la textura y el MSAA, que el cambio de tamaño despierta el dibujo y que funcionan las transiciones modelo–bienvenida–modelo y la actualización del QR. La animación de bienvenida produce 30 dibujos por cada 60 ciclos de control.
