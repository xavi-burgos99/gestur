# Bases generadas para los iconos de movimiento

Estas dos imágenes se generaron con la herramienta integrada de generación de imágenes, sin API externa, clave de API, CLI ni Blender. Se conservan como PNG separados con su canal alfa original; no se recortaron ni retocaron mediante código. La cabeza se generó primero y se inspeccionó antes de utilizarla como referencia de estilo para la mano.

Archivos preparados para la interfaz:

- `portal/public/motion-icons/head-base.png`: cabeza genérica con giro leve hacia la derecha del observador.
- `portal/public/motion-icons/hand-base.png`: mano derecha vista desde la palma, con cinco dedos y el pulgar a la izquierda del observador.

## Criterios que deben conservarse

La cabeza y la mano comparten contornos verdes redondeados y rellenos claros. La paleta solicitada fue `#41675b` para el trazo y `#e4ede8` para el relleno; la generación raster puede aproximar estos colores. El fondo es transparente. Los sujetos están centrados y dejan margen para las indicaciones de movimiento.

No deben añadirse texto, letras, números, flechas, sombras, volumen realista, fondos decorativos ni logotipos a las bases. Las flechas y los indicadores se dibujarán como elementos vectoriales en la interfaz. La mano izquierda se obtendrá reflejando esta misma base en la interfaz; no se generará otra mano. Las variantes de gesto deben reutilizar las bases para mantener el mismo estilo.

Estos archivos se reutilizan en `portal/src/MotionIcon.jsx`. El catálogo de `portal/src/control-catalog.mjs` asigna a cada gesto la misma base y una indicación vectorial de su movimiento. El editor de Parámetros muestra estos iconos al elegir y asignar controles.

## Prompt definitivo de la cabeza

```text
Use case: infographic-diagram.
Asset type: base PNG para iconos funcionales de gestos de una interfaz web; una sola cabeza aislada, lienzo cuadrado 1:1 con canal alfa auténtico y fondo totalmente transparente.
Primary request: pictograma sobrio de una cabeza humana genérica neutral, vista casi frontal con giro suave de 20 grados hacia la derecha del observador, nariz sencilla y claramente orientada hacia la derecha, ojos mínimos, sin cabello detallado, con cuello corto y SIN hombros.
Style: dibujo plano limpio de aspecto vectorial, contornos uniformes redondeados de aproximadamente 20 px en un lienzo de 1024 px. Todas las líneas de igual grosor, curvas suaves y pocos detalles. Trazo #41675b; relleno sólido #e4ede8, sin otros colores.
Composition: cabeza centrada ópticamente, ocupa aproximadamente el 60% del ancho y el 70% del alto. Deja amplios márgenes transparentes en los cuatro lados para añadir flechas vectoriales después. Todo el sujeto está dentro del encuadre.
Constraints: rostro neutral sin identidad reconocible, una sola cabeza y un cuello; relleno completamente opaco dentro del dibujo, fondo alfa. No texto, letras, números, flechas, símbolos adicionales, sombras, degradados, iluminación, volumen realista, textura, pelo elaborado, adornos, borde de lienzo, logotipo ni marca de agua. No dibujes una cuadrícula de transparencia. Devuelve solamente el pictograma.
```

## Prompt definitivo de la mano

Se adjuntó `head-base.png` únicamente como referencia de estilo.

```text
Use case: infographic-diagram.
Asset type: base PNG para iconos funcionales de gestos de una interfaz web; una única mano aislada, lienzo cuadrado 1:1 con canal alfa auténtico y fondo totalmente transparente.
Input images: la imagen adjunta de una cabeza es SOLO referencia de estilo de trazo, colores y sencillez; NO debe aparecer una cabeza en el resultado.
Primary request: pictograma sobrio de una única mano humana DERECHA vista desde la PALMA, vertical con muñeca corta abajo. Exactamente cinco dedos anatómicamente correctos, abiertos moderadamente, el pulgar se proyecta hacia la IZQUIERDA del observador; índice, medio, anular y meñique en orden natural, con el medio más alto y el meñique más corto a la derecha. La forma de palma y pulgar debe ser inequívoca. Evita líneas detalladas de huellas o uñas.
Style: igual al pictograma de referencia: dibujo plano limpio de aspecto vectorial, contornos uniformes redondeados, todas las líneas de igual grosor, curvas suaves y pocos detalles. Trazo #41675b; relleno sólido #e4ede8, sin otros colores. Dos pequeñas líneas discretas de pliegue de la palma como máximo si mejoran su lectura.
Composition: mano centrada ópticamente, ocupa aproximadamente el 60% del ancho y el 70% del alto. Deja amplios márgenes transparentes en los cuatro lados para añadir flechas vectoriales después. Todos los dedos y la muñeca están dentro del encuadre.
Constraints: relleno completamente opaco dentro del dibujo, fondo alfa. No brazos largos, segunda mano, sexto dedo, dedos fusionados, texto, letras, números, flechas, símbolos, sombras, degradados, iluminación, volumen realista, textura, adornos, borde de lienzo, logotipo ni marca de agua. No dibujes una cuadrícula de transparencia. Devuelve solamente el pictograma de mano.
```
