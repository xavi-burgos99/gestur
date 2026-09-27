# Portal local de Gestur

El portal utiliza React 19, Mantine 9 (componentes accesibles), Vite y Fastify 5. El navegador del administrador renderiza la interfaz; la Raspberry sirve archivos estáticos y una API. No se añade otra escena 3D ni otro detector al proceso del portal.

## Instalación

En la rama del portal, `sudo ./gestur.sh install` instala también el servicio. El hook separado es `sudo bash /opt/gestur/scripts/install-portal.sh /opt/gestur`; requiere que el instalador principal haya creado el usuario `gestur`, su entorno Python y los modelos de reconocimiento. Raspberry Pi OS de 64 bits con NetworkManager; configura el país WLAN con Raspberry Pi Imager o `sudo raspi-config` antes de instalar.

La instalación no pide datos por consola. Ambos comandos aceptan `--hostname sala-1` para fijar el nombre sin `.local`: entre 1 y 63 letras minúsculas, números o guiones, sin guiones en los extremos. Si se omite, una instalación nueva usa `gestur-xxxx`, con los últimos cuatro caracteres de la MAC permanente de Wi-Fi en minúsculas. Al reinstalar se conserva el nombre actual salvo que se pase esa opción.

El hook instala Assimp y bubblewrap desde la distribución, y comprueba el aislamiento del conversor con el usuario del servicio. Después de la compilación y la poda de dependencias, `scripts/check-importer.mjs` verifica el motor de simplificación, la conversión de imágenes y un modelo real con Assimp/Panda3D. También habilita mDNS para el enlace de respaldo `<hostname>.local`. También instala Node 22.23.3 ARM64/x64 desde nodejs.org y verifica su archivo con el manifiesto SHA-256 oficial si no hay Node >=22.12. El runtime privado vive en `/opt/gestur-node`, sin sustituir archivos de paquetes del sistema. `npm ci`, compilación y poda de dependencias se ejecutan como usuario sin privilegios. El código y el helper privilegiado quedan propiedad de root. Los datos permanecen en `/var/lib/gestur` al reinstalar.

El servicio `gestur-portal` escucha en el puerto HTTP 80 como usuario `gestur-portal`, con `CAP_NET_BIND_SERVICE` concedida solo a esa unidad. Se abre por IP o por `http://gestur-xxxx.local` (o `<hostname>.local` si cambiaste el nombre), sin indicar puerto. Para una instalación nueva, conéctate a **GESTUR-XXXX**, donde XXXX son los últimos cuatro dígitos de la MAC permanente de `wlan0`, y abre **http://10.42.0.1**. La red es abierta por defecto. Al reinstalar se conserva el SSID, la contraseña y la configuración IP del AP existente; utiliza la IP de ese perfil. Si hay varios perfiles de AP y ninguno seleccionado, el instalador se detiene para que el administrador elija el UUID en `/etc/gestur/wifi-profile.json` (objeto `{"uuid":"UUID","interface":"wlan0"}` propiedad root y modo 0600).

Una instalación nueva queda pendiente de la configuración inicial en el portal. El instalador no crea ni imprime una clave. El estado de acceso vive en `/etc/gestur/device.json`, propiedad de `root:gestur-portal` y modo 0640: el servicio puede leerlo y solo el helper privilegiado puede modificarlo. Al actualizar una versión con `/etc/gestur/portal-token`, se conserva esa clave y su acceso sin reabrir la configuración inicial; esa clave anterior sigue siendo recuperable con `sudo cat /etc/gestur/portal-token`. El portal está pensado para una red local de confianza: su HTTP local no cifra el tráfico; no debe publicarse en Internet sin un proxy HTTPS y controles de acceso.

## Configuración inicial y restablecimiento

El primer acceso tiene dos pasos: nombre opcional del dispositivo (vacío conserva
el actual) y contraseña con confirmación. La contraseña elegida protege tanto el
portal como el punto de acceso; debe tener entre 8 y 63 caracteres ASCII para ser
compatible con WPA2. El portal conserva un hash scrypt, nunca la contraseña en
claro. NetworkManager conserva la credencial necesaria para el Wi-Fi.

Al guardar aparece el aviso de reinicio y reconexión. Las operaciones se confirman
antes de modificar la red y su estado público persiste sin secretos. El portal
no da por terminado un cambio solo porque haya perdido la conexión. Desde
**Configuración → Dispositivo** se puede cambiar posteriormente el hostname.

**Borrar contenido y ajustes** exige escribir `BORRAR`: elimina los modelos y
los ajustes de Gestur, invalida el acceso anterior, restaura `gestur-xxxx` y la
red abierta `GESTUR-XXXX`, y vuelve a mostrar el asistente tras reiniciar.
No reinstala el sistema operativo ni modifica usuarios o credenciales SSH.
Las modificaciones de Wi-Fi posteriores al asistente son independientes de la
contraseña de acceso al portal. Actualizar una instalación anterior conserva su
acceso, sus modelos y su configuración; no equivale a restablecerla.

## Modelos 3D

En **Ajustar modelo** puedes guardar o quitar una **URL** opcional. Se admiten
enlaces HTTP/HTTPS de hasta 2048 bytes UTF-8, sin credenciales. Si hay una URL,
el visor muestra un QR blanco con fondo transparente en la esquina inferior
derecha, separado de ambos bordes y más pequeño que el de bienvenida. Vaciar
el campo elimina el QR. Se aplica al guardar, sin recargar la geometría, y se
conserva al reiniciar; Gestur no visita ni descarga el enlace.

La biblioteca está **vacía** en instalaciones nuevas. El [capitel original](../examples/capitel/README.md) está disponible en el repositorio para importarlo, pero no se instala como modelo predeterminado. El primer modelo importado se selecciona automáticamente; las siguientes importaciones conservan el último elegido. La selección se guarda y se recupera al reiniciar. Si falta el archivo elegido, se selecciona otro modelo disponible.

Solo una biblioteca vacía muestra la bienvenida: figura 3D blanca al 15 % de opacidad a tamaño de pantalla, título «GESTUR», QR de módulos blancos sobre fondo transparente, «Escanea el QR para comenzar» y «o accede a {url}». El QR se genera localmente y no contiene la clave de administración. Usa la IP de la red del dispositivo, o la del punto de acceso sin LAN. `GESTUR_PORTAL_URL` permite fijar una URL con IP válida. Mientras no haya IP disponible se muestra la espera de conexión y se reintenta. Un fallo al cargar un modelo se informa como error; no se sustituye por la bienvenida si hay modelos disponibles.

### Subir un modelo

Sube un archivo directamente cuando contenga todo lo necesario (por ejemplo GLB, STL o PLY), o un **ZIP con un único modelo principal** y sus materiales, texturas y archivos BIN. Se admiten GLB, glTF, OBJ, FBX, DAE/Collada, STL, PLY, 3DS, OFF, DirectX X, LWO, ASE, DXF, AC, MS3D, COB y B3D. El soporte concreto depende del lector de Assimp instalado y del contenido del archivo; formatos propietarios, extensiones glTF y shaders específicos de un programa pueden requerir exportar de nuevo desde ese programa. No se promete conservar animaciones ni reproducir motores de materiales externos.

La importación busca recursos **solo dentro del paquete**: normaliza separadores de Windows, rutas exportadas absolutas, URI codificadas y diferencias de mayúsculas; intenta recuperar referencias mediante coincidencias de carpeta y nombre de archivo. Los nombres Unicode están permitidos. Una coincidencia ambigua o un archivo ausente produce un aviso o un error que identifica el recurso; no se descargan texturas ni se inventan imágenes. Si faltan materiales, incluye el MTL y las texturas en el ZIP. Para varios modelos independientes, sube cada uno por separado.

Assimp convierte el modelo; el portal reúne geometría, materiales y texturas en un GLB autocontenido y comprueba que Panda3D pueda cargarlo. El usuario ve el progreso mientras el dispositivo trabaja. Los procesos de conversión Linux se ejecutan con **bubblewrap**, sin acceso a la red ni a datos ajenos a la importación. Tienen límites de tiempo y recursos; no se ejecutan scripts incluidos en un modelo. Solo se publica una importación terminada y válida, por lo que un fallo no sustituye el modelo activo.

### Simplificación opcional

Solo si hay **más de 1.000.000 de triángulos**, el trabajo espera la decisión del usuario. El modal muestra el recuento actual, un objetivo fijo de **aproximadamente 500.000 triángulos** y el porcentaje de reducción. Esa referencia es similar a los 491.038 del capitel utilizado anteriormente. No hay deslizadores ni valores que configurar.

- **Reducir e importar**: meshoptimizer reduce la malla localmente en la Raspberry Pi, conservando materiales y coordenadas UV; se valida de nuevo y se registra el recuento final. Para preservar bordes y limitar el error geométrico, algunas mallas pueden quedar por encima del objetivo; se muestra el recuento real y un aviso si la diferencia supera el 5 %.
- **Continuar sin simplificar**: se publica el modelo convertido conservando su geometría.

La reducción de mallas grandes puede tardar minutos. Solo hay una importación en curso; recargar el portal recupera su estado. Los trabajos interrumpidos por un reinicio se notifican como fallidos en lugar de darse por completados; una propuesta que ya esperaba confirmación puede recuperarse y responderse después del reinicio. El primer modelo válido se activa automáticamente; después, «Mostrar en pantalla» cambia la selección. No se puede vaciar la selección mientras queden modelos disponibles.

Límites de subida: 100 MiB por archivo, 250 MiB al descomprimir, 500 entradas y ratio de expansión máximo 100× por archivo (con margen de 1 MiB para archivos pequeños). Se rechazan enlaces, rutas que salgan del ZIP, nombres duplicados sin distinguir mayúsculas y ZIP cifrados. No se incluyen los temporales ni los datos de desarrollo en una instalación nueva.

### Gestionar modelos

El menú de cada modelo permite **Ajustar modelo** y **Eliminar**. En los ajustes
puedes cambiar el nombre y corregir su orientación con giros de 0°, 90°, 180° o
270° sobre los ejes X, Y y Z. Esta orientación se guarda con el modelo y se aplica
antes de los movimientos controlados por gestos, también tras reiniciar. No se
reescribe la geometría ni la textura.

El borrado pide confirmación y elimina el modelo junto con su archivo original.
Si era el modelo seleccionado, se activa otro disponible. Al eliminar el último
modelo, el visor vuelve a la bienvenida. Durante una importación se bloquean
los cambios en los modelos hasta que termine.

Los paquetes terminados se guardan en `/var/lib/gestur/models/<uuid>/` con metadatos `.gestur-model.json`. La selección persistente usa `<uuid>/model.glb` o `<uuid>/simplified.glb` al reducir; el original se conserva dentro del paquete. El portal distingue **Seleccionado** de **En pantalla** consultando el estado real del visualizador cada tres segundos; a los diez segundos sin actualización muestra el visor desconectado.

## Parámetros

Los **presets** guardan las secciones completas de seguimiento, renderizado y
controles, incluidos los ajustes avanzados y las asignaciones de gestos. Se
guardan desde el borrador visible, sin aplicarlo al visor. Cargar un preset
aplica sus parámetros; si hay cambios sin guardar, se pide confirmación antes
de sustituirlos. No cambian el modelo elegido, sus metadatos ni la red.
Guardar con el mismo nombre sobrescribe el preset; también pueden eliminarse.
Se conservan hasta 50 presets en `/var/lib/gestur/presets.json`, sobreviven al
reinicio y se eliminan al borrar contenido y ajustes.

Primero se elige un movimiento del modelo y después su gesto. El selector muestra, con iconos, la parte del cuerpo (Cabeza, Cuerpo, Manos o Combinado), la mano cuando corresponde, el tipo de movimiento y el eje. Las opciones aparecen al completar el paso anterior. Las rotaciones se llaman Giro horizontal, Giro vertical e Inclinación lateral; los desplazamientos, Horizontal, Vertical y Profundidad. Cada mano ofrece también Apertura de la mano y Pinza, sin pedir un eje. Combinado ofrece Distancia entre manos.

El selector permite 29 gestos. Las asignaciones anteriores de giro de mano en pantalla (2D), centro de ambas manos y separación horizontal se siguen mostrando y admiten ajustes, pero ya no se pueden seleccionar. Al cambiar uno de esos gestos, el selector empieza sin selección en Parte del cuerpo y exige completar una opción válida antes de asignarla. Abrir o cancelar el selector conserva la asignación y sus valores.

La UI edita el contrato compartido `config/schema.json` / `config/default.json`; ver [configuration.md](configuration.md). Conserva intensidad, inversión, respuesta proporcional/continua/dos tamaños y umbrales al cambiar un gesto. Cada salida admite un gesto activo. Incluye reconocimiento de manos, suavizado, regreso al centro, cámara, frecuencias, antialiasing y cursor oculto. Los cambios de controles se aplican en vivo; cambios de captura, MSAA o pantalla completa reinician el visualizador automáticamente.

La configuración se valida y escribe con reemplazo atómico. Una configuración inválida o de versión futura se conserva y se muestra un error para evitar una migración destructiva. Solo las versiones explícitamente admitidas pueden editarse.

## Punto de acceso Wi-Fi

La sección permite renombrar la red, **Añadir contraseña**, **Cambiar contraseña** y **Eliminar contraseña**. La contraseña nueva debe tener 8–63 caracteres ASCII; quitarla crea una red abierta. Renombrar conserva la contraseña actual.

La API confirma un cambio pendiente antes de aplicarlo cinco segundos después, porque el AP puede desconectar al navegador. Vuelve a conectarte a la red indicada; «Comprobar estado» consulta el estado real. El resultado persiste en `/var/lib/gestur/wifi-job.json` sin la contraseña; si el servicio se interrumpe muestra que debe comprobarse la red. Un fallo de activación intenta restaurar la configuración previa.

El backend no es root. Sus reglas sudo permiten solo `gestur-wifi status`, `gestur-wifi apply` y las acciones `status`, `hostname`, `onboarding` y `reset` de `gestur-device`. Los helpers root validan de nuevo el JSON de entrada; la acción `bootstrap` está reservada al instalador y no se concede al portal. El helper de Wi-Fi usa la API D-Bus de NetworkManager para un UUID previamente elegido por el instalador; no ejecuta comandos enviados por el cliente, no expone claves Wi-Fi en respuestas/logs/argumentos de procesos y nunca da éxito simulado cuando NetworkManager no está disponible.

La unidad permite escritura en `/var/lib/gestur` y monta `/etc/gestur` y `/etc/hosts` como rutas modificables por los helpers. Sus permisos de propietario siguen impidiendo la escritura directa al usuario del portal: el directorio de estado es `root:root` 0755 y las credenciales, `root:gestur-portal` 0640. Los helpers son `root:root` 0755; los valores de fábrica para restablecer el dispositivo están en `/etc/gestur/default.json`, `root:root` 0644.

La unidad conserva la elevación de ese helper: no activa `NoNewPrivileges` ni limita el conjunto de capacidades a la de abrir puertos. Antes de ejecutar el conversor aislado, `setpriv` de `util-linux` elimina las capacidades heredables y ambientales solo de ese proceso hijo; así bubblewrap arranca sin heredar el permiso HTTP y mantiene sus espacios de nombres, `--cap-drop ALL` y límites de recursos. No se conceden capacidades al binario de Node ni se cambian los puertos privilegiados del sistema.

Los conversores reciben un directorio `/proc` vacío y conservan el espacio de nombres de PID. Así no ven los procesos del host ni necesitan montar procfs dentro de la unidad. En el kernel de la Pi probada, ese montaje fallaba porque `ProtectKernelTunables` crea submontajes protegidos en `/proc` y Linux impide volver a exponerlos desde un espacio de nombres sin privilegios. La unidad conserva `ProtectKernelTunables=true` y sus demás protecciones. Se comprobó esta configuración en la Pi con Assimp, Node, Sharp y meshoptimizer. Fuentes: [`mount_too_revealing` en Linux](https://github.com/torvalds/linux/blob/v6.18/fs/namespace.c#L5785-L5860) y [submontajes de systemd](https://github.com/systemd/systemd/blob/v257/src/core/namespace.c#L131-L154).

Antes de abrir el puerto HTTP, `ExecStartPre` ejecuta `check-importer.mjs` con las mismas restricciones y credenciales del servicio. Comprueba Assimp, las dependencias de texturas y simplificación, y la carga del GLB en Panda3D. Si falla, el portal no se declara arrancado. El instalador adapta tanto esa comprobación como el servidor a la ruta de Node instalada; la comprobación preliminar mediante `runuser` no sustituye este paso.

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
