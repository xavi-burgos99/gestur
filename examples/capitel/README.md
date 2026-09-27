# Capitel de ejemplo

Modelo original del proyecto: **491.038 triángulos**, con su material y textura
de 2048 × 2048. Los tres archivos se conservan sin cambios de geometría ni de
textura.

Para subirlo al portal, comprime juntos `capitell.obj`, `capitell.mtl` y
`capitell.jpg` en un ZIP y selecciona **Subir modelo**. Desde este directorio:

```sh
zip capitel.zip capitell.obj capitell.mtl capitell.jpg
```

El portal convierte el conjunto en un GLB con la textura incluida. No necesita
simplificación: está por debajo del umbral de un millón de triángulos.

Este ejemplo no se añade automáticamente a la biblioteca de nuevas
instalaciones. Al importar el primer modelo se selecciona automáticamente;
después se conserva el último modelo elegido, también tras reiniciar.
