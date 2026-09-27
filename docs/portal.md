# Portal local de Gestur

El portal utiliza React 19, Mantine 9 (componentes accesibles), Vite y Fastify 5. El navegador del administrador renderiza la interfaz; la Raspberry sirve archivos estáticos y una API. No se añade otra escena 3D ni otro detector al proceso del portal.

## Instalación

En la rama del portal, `sudo ./gestur.sh install` instala también el servicio. El hook separado es `sudo bash /opt/gestur/scripts/install-portal.sh /opt/gestur`; requiere que el instalador principal haya creado el usuario `gestur`, su entorno Python y los modelos de reconocimiento. Raspberry Pi OS de 64 bits con NetworkManager; configura el país WLAN con Raspberry Pi Imager o `sudo raspi-config` antes de instalar.

El hook instala Assimp y bubblewrap desde la distribución, y comprueba el aislamiento del conversor con el usuario del servicio. Después de la compilación y la poda de dependencias, `scripts/check-importer.mjs` verifica el motor de simplificación, la conversión de imágenes y un modelo real con Assimp/Panda3D. También habilita mDNS para el enlace de respaldo `<hostname>.local`. También instala Node 22.23.3 ARM64/x64 desde nodejs.org y verifica su archivo con el manifiesto SHA-256 oficial si no hay Node >=22.12. El runtime privado vive en `/opt/gestur-node`, sin sustituir archivos de paquetes del sistema. `npm ci`, compilación y poda de dependencias se ejecutan como usuario sin privilegios. El código y el helper privilegiado quedan propiedad de root. Los datos permanecen en `/var/lib/gestur` al reinstalar.

El servicio `gestur-portal` escucha en el puerto HTTP 80 como usuario `gestur-portal`, con `CAP_NET_BIND_SERVICE` concedida solo a esa unidad. Se abre por IP o por `http://gestur.local` (o `<hostname>.local` si cambiaste el nombre), sin indicar puerto. Para una instalación nueva, conéctate a **GESTUR-XXXX**, donde XXXX son los últimos cuatro dígitos de la MAC permanente de `wlan0`, y abre **http://10.42.0.1**. La red es abierta por defecto. Si ya había un único perfil AP en `wlan0`, se conserva su SSID, contraseña y configuración IP; utiliza la IP de ese perfil. Si hay varios, el instalador se detiene para que el administrador elija el UUID en `/etc/gestur/wifi-profile.json` (objeto `{"uuid":"UUID","interface":"wlan0"}` propiedad root y modo 0600).

La clave de administración independiente se genera una sola vez y se muestra al instalar. Se puede consultar físicamente/por SSH con `sudo cat /etc/gestur/portal-token`. No hay una clave predeterminada compartida. Las sesiones expiran a las ocho horas y se invalidan al reiniciar el servicio. El portal está pensado para una red local de confianza: su HTTP local no cifra el tráfico; no debe publicarse en Internet sin un proxy HTTPS y controles de acceso.

## Modelos 3D

La biblioteca está **vacía** en instalaciones nuevas. No se distribuyen el capitel ni modelos de ejemplo. El expositor muestra una figura 3D procedural con el texto «Escanea el QR para comenzar» y el enlace al portal. El QR se genera localmente; usa la IP del punto de acceso o la red local y no contiene la clave de administración. `GESTUR_PORTAL_URL` permite establecer una dirección alternativa. La antigua selección del capitel se migra a la bienvenida; las importaciones del usuario se conservan.

### Subir un modelo

Sube un archivo directamente cuando contenga todo lo necesario (por ejemplo GLB, STL o PLY), o un **ZIP con un único modelo principal** y sus materiales, texturas y archivos BIN. Se admiten GLB, glTF, OBJ, FBX, DAE/Collada, STL, PLY, 3DS, OFF, DirectX X, LWO, ASE, DXF, AC, MS3D, COB y B3D. El soporte concreto depende del lector de Assimp instalado y del contenido del archivo; formatos propietarios, extensiones glTF y shaders específicos de un programa pueden requerir exportar de nuevo desde ese programa. No se promete conservar animaciones ni reproducir motores de materiales externos.

La importación busca recursos **solo dentro del paquete**: normaliza separadores de Windows, rutas exportadas absolutas, URI codificadas y diferencias de mayúsculas; intenta recuperar referencias mediante coincidencias de carpeta y nombre de archivo. Los nombres Unicode están permitidos. Una coincidencia ambigua o un archivo ausente produce un aviso o un error que identifica el recurso; no se descargan texturas ni se inventan imágenes. Si faltan materiales, incluye el MTL y las texturas en el ZIP. Para varios modelos independientes, sube cada uno por separado.

Assimp convierte el modelo; el portal reúne geometría, materiales y texturas en un GLB autocontenido y comprueba que Panda3D pueda cargarlo. El usuario ve el progreso mientras el dispositivo trabaja. Los procesos de conversión Linux se ejecutan con **bubblewrap**, sin acceso a la red ni a datos ajenos a la importación. Tienen límites de tiempo y recursos; no se ejecutan scripts incluidos en un modelo. Solo se publica una importación terminada y válida, por lo que un fallo no sustituye el modelo activo.

### Simplificación opcional

Solo si hay **más de 1.000.000 de triángulos**, el trabajo espera la decisión del usuario. El modal muestra el recuento actual, un objetivo fijo de **aproximadamente 500.000 triángulos** y el porcentaje de reducción. Esa referencia es similar a los 491.038 del capitel utilizado anteriormente. No hay deslizadores ni valores que configurar.

- **Reducir e importar**: meshoptimizer reduce la malla localmente en la Raspberry Pi, conservando materiales y coordenadas UV; se valida de nuevo y se registra el recuento final. Para preservar bordes y limitar el error geométrico, algunas mallas pueden quedar por encima del objetivo; se muestra el recuento real y un aviso si la diferencia supera el 5 %.
- **Continuar sin simplificar**: se publica el modelo convertido conservando su geometría.

La reducción de mallas grandes puede tardar minutos. Solo hay una importación en curso; recargar el portal recupera su estado. Los trabajos interrumpidos por un reinicio se notifican como fallidos en lugar de darse por completados; una propuesta que ya esperaba confirmación puede recuperarse y responderse después del reinicio. El modelo activo no cambia hasta seleccionarlo en la colección. «Mostrar bienvenida» permite volver al QR sin borrar modelos.

Límites de subida: 100 MiB por archivo, 250 MiB al descomprimir, 500 entradas y ratio de expansión máximo 100× por archivo (con margen de 1 MiB para archivos pequeños). Se rechazan enlaces, rutas que salgan del ZIP, nombres duplicados sin distinguir mayúsculas y ZIP cifrados. No se incluyen los temporales ni los datos de desarrollo en una instalación nueva.

Los paquetes terminados se guardan en `/var/lib/gestur/models/<uuid>/` con metadatos `.gestur-model.json`. La selección persistente usa `<uuid>/model.glb` o `<uuid>/simplified.glb` al reducir; el original se conserva dentro del paquete. El portal distingue **Seleccionado** de **En pantalla** consultando el estado real del visualizador cada tres segundos; a los diez segundos sin actualización muestra el visor desconectado.

## Parámetros

Primero se elige un movimiento del modelo y después su gesto. El selector muestra, con iconos, la parte del cuerpo (Cabeza, Cuerpo, Manos o Combinado), la mano cuando corresponde, el tipo de movimiento y el eje. Las opciones aparecen al completar el paso anterior. Las rotaciones se llaman Giro horizontal, Giro vertical e Inclinación lateral; los desplazamientos, Horizontal, Vertical y Profundidad. Las manos añaden Apertura de la mano y Combinado ofrece Distancia entre manos. «Otros gestos» conserva la pinza, el giro de mano en pantalla y los movimientos conjuntos anteriores.

La UI edita el contrato compartido `config/schema.json` / `config/default.json`; ver [configuration.md](configuration.md). Conserva intensidad, inversión, respuesta proporcional/continua/dos tamaños y umbrales al cambiar un gesto. Cada salida admite un gesto activo. Incluye reconocimiento de manos, suavizado, regreso al centro, cámara, frecuencias, antialiasing y cursor oculto. Los cambios de controles se aplican en vivo; cambios de captura, MSAA o pantalla completa reinician el visualizador automáticamente.

La configuración se valida y escribe con reemplazo atómico. Una configuración inválida o de versión futura se conserva y se muestra un error para evitar una migración destructiva. Solo las versiones explícitamente admitidas pueden editarse.

## Punto de acceso Wi-Fi

La sección permite renombrar la red, **Añadir contraseña**, **Cambiar contraseña** y **Eliminar contraseña**. La contraseña nueva debe tener 8–63 caracteres ASCII; quitarla crea una red abierta. Renombrar conserva la contraseña actual.

La API confirma un cambio pendiente antes de aplicarlo cinco segundos después, porque el AP puede desconectar al navegador. Vuelve a conectarte a la red indicada; «Comprobar estado» consulta el estado real. El resultado persiste en `/var/lib/gestur/wifi-job.json` sin la contraseña; si el servicio se interrumpe muestra que debe comprobarse la red. Un fallo de activación intenta restaurar la configuración previa.

El backend no es root. Su regla sudo permite solo `gestur-wifi status` y `gestur-wifi apply`, y el helper root valida de nuevo el JSON de entrada. Usa la API D-Bus de NetworkManager para un UUID previamente elegido por el instalador; no ejecuta comandos enviados por el cliente, no expone claves Wi-Fi en respuestas/logs/argumentos de procesos y nunca da éxito simulado cuando NetworkManager no está disponible.

La unidad conserva la elevación de ese helper: no activa `NoNewPrivileges` ni limita el conjunto de capacidades a la de abrir puertos. Antes de ejecutar el conversor aislado, `setpriv` de `util-linux` elimina las capacidades heredables y ambientales solo de ese proceso hijo; así bubblewrap arranca sin heredar el permiso HTTP y mantiene sus espacios de nombres, `--cap-drop ALL` y límites de recursos. No se conceden capacidades al binario de Node ni se cambian los puertos privilegiados del sistema.

## Desarrollo y verificación

Desde `portal/`:

```bash
npm ci
npm test
npm run build
GESTUR_ADMIN_TOKEN=development-test-only-not-a-real-secret \
GESTUR_CONFIG="$PWD/.dev-data/config.json" \
GESTUR_MODELS_DIR="$PWD/.dev-data/models" \
GESTUR_PYTHON=/ruta/al/gestur/.venv/bin/python \
HOST=127.0.0.1 npm start
```

Abre http://127.0.0.1:3000. El validador usa `../scripts/check_model.py` y necesita el entorno Python del proyecto. Para importar también se requiere Assimp y la simplificación usa las dependencias Node del portal; no se sustituye por una aprobación simulada en desarrollo. En macOS instala Assimp para probar importaciones; el conversor se aísla con `sandbox-exec`. `GESTUR_ASSIMP` permite indicar la ruta del ejecutable. `GESTUR_PYTHON` debe apuntar al Python con los requisitos del visor. En equipos sin el helper Linux, Configuración muestra «Wi-Fi no disponible». `npm run dev` inicia Vite en 5173 y delega `/api` al backend 3000.

```bash
python3 -m unittest discover -s portal/test -p 'test_*.py'
bash -n scripts/install-portal.sh
npm --prefix portal audit --omit=dev
sudo systemctl status gestur-portal
sudo journalctl -u gestur-portal -n 50
```

Las pruebas API/ZIP y del helper usan directorios temporales y dobles de NetworkManager; no modifican el Wi-Fi del ordenador. La activación real y la reconexión deben verificarse en Raspberry Pi con NetworkManager y `wlan0`.

Referencias técnicas: [Mantine + Vite](https://mantine.dev/guides/vite/), [Fastify](https://fastify.dev/docs/latest/), [NetworkManager Update/GetSecrets](https://networkmanager.dev/docs/api/latest/gdbus-org.freedesktop.NetworkManager.Settings.Connection.html), [MAC permanente del dispositivo Wi-Fi](https://networkmanager.dev/docs/api/latest/gdbus-org.freedesktop.NetworkManager.Device.Wireless.html).

Referencias de importación: [formatos de Assimp](https://github.com/assimp/assimp/blob/master/doc/Fileformats.md), [simplificación con meshoptimizer](https://github.com/zeux/meshoptimizer#simplification).
