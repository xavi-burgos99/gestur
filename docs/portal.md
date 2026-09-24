# Portal local de Gestur

El portal utiliza React 19, Mantine 9 (componentes accesibles), Vite y Fastify 5. El navegador del administrador renderiza la interfaz; la Raspberry sirve archivos estáticos y una API. No se añade otra escena 3D ni otro detector al proceso del portal.

## Instalación

En la rama del portal, `sudo ./gestur.sh install` instala también el servicio. El hook separado es `sudo bash /opt/gestur/scripts/install-portal.sh /opt/gestur`; requiere que el instalador principal haya creado el usuario `gestur`, su entorno Python y los modelos. Raspberry Pi OS de 64 bits con NetworkManager; configura el país WLAN con Raspberry Pi Imager o `sudo raspi-config` antes de instalar.

El hook instala Node 22.23.3 ARM64/x64 desde nodejs.org y verifica su archivo con el manifiesto SHA-256 oficial si no hay Node >=22.12. El runtime privado vive en `/opt/gestur-node`, sin sustituir archivos de paquetes del sistema. `npm ci`, compilación y poda de dependencias se ejecutan como usuario sin privilegios. El código y el helper privilegiado quedan propiedad de root. Los datos permanecen en `/var/lib/gestur` al reinstalar.

El servicio `gestur-portal` escucha en el puerto 3000. Para una instalación nueva, conéctate a **GESTUR-XXXX**, donde XXXX son los últimos cuatro dígitos de la MAC permanente de `wlan0`, y abre **http://10.42.0.1:3000**. La red es abierta por defecto. Si ya había un único perfil AP en `wlan0`, se conserva su SSID, contraseña y configuración IP; utiliza la IP de ese perfil. Si hay varios, el instalador se detiene para que el administrador elija el UUID en `/etc/gestur/wifi-profile.json` (objeto `{"uuid":"UUID","interface":"wlan0"}` propiedad root y modo 0600).

La clave de administración independiente se genera una sola vez y se muestra al instalar. Se puede consultar físicamente/por SSH con `sudo cat /etc/gestur/portal-token`. No hay una clave predeterminada compartida. Las sesiones expiran a las ocho horas y se invalidan al reiniciar el servicio. El portal está pensado para una red local de confianza: su HTTP local no cifra el tráfico; no debe publicarse en Internet sin un proxy HTTPS y controles de acceso.

## Modelos 3D

**Capitel** (`capitell.obj`) permanece siempre disponible y no puede borrarse desde el portal. Importa un ZIP que contenga **exactamente una** entrada `.obj`, `.gltf` o `.glb`, junto a sus archivos MTL, BIN y texturas. Para OBJ, usa una biblioteca MTL por directiva `mtllib`; se admiten las opciones de textura Wavefront habituales. Se conservan subcarpetas y nombres con espacios; los componentes del camino al modelo deben empezar por letra o número y usar caracteres ASCII: letras, números, guiones, puntos, espacios y guiones bajos.

Límites: 100 MiB comprimidos, 250 MiB descomprimidos, 500 entradas y ratio de expansión máximo 100× por archivo (con margen de 1 MiB para archivos pequeños). Se rechazan enlaces, rutas absolutas o que salgan del ZIP, nombres duplicados sin distinguir mayúsculas y ZIP cifrados. Los recursos deben estar dentro del paquete; glTF puede incluir URI `data:`. El modelo se carga además con el mismo motor 3D en un subproceso sin ventana/cámara, con límite de tiempo; una importación fallida elimina sus temporales y no modifica la selección.

Los paquetes se guardan en `/var/lib/gestur/models/<uuid>/` con metadatos `.gestur-model.json`. La selección persistente es `<uuid>/<entrada>`; el modelo original sigue siendo `capitell.obj`. El portal distingue **Seleccionado** de **En pantalla** consultando el estado real del visualizador cada tres segundos; a los diez segundos sin actualización muestra el visor desconectado. Los errores de carga se muestran sin inventar un renderizado correcto.

## Parámetros

La UI edita el contrato compartido `config/schema.json` / `config/default.json`; ver [configuration.md](configuration.md). Permite cambiar entradas (cabeza, centro/distancia/separación de manos, giros en tres ejes y pinza), salidas, intensidad, inversión, respuesta proporcional/continua/dos tamaños y umbrales. Cada salida admite un gesto activo. Incluye reconocimiento de manos, suavizado, regreso al centro, cámara, frecuencias, antialiasing y cursor oculto. Los cambios de controles se aplican en vivo; cambios de captura, MSAA o pantalla completa reinician el visualizador automáticamente.

La configuración se valida y escribe con reemplazo atómico. Una configuración inválida o de versión futura se conserva y se muestra un error para evitar una migración destructiva. Solo las versiones explícitamente admitidas pueden editarse.

## Punto de acceso Wi-Fi

La sección permite renombrar la red, **Añadir contraseña**, **Cambiar contraseña** y **Eliminar contraseña**. La contraseña nueva debe tener 8–63 caracteres ASCII; quitarla crea una red abierta. Renombrar conserva la contraseña actual.

La API confirma un cambio pendiente antes de aplicarlo cinco segundos después, porque el AP puede desconectar al navegador. Vuelve a conectarte a la red indicada; «Comprobar estado» consulta el estado real. El resultado persiste en `/var/lib/gestur/wifi-job.json` sin la contraseña; si el servicio se interrumpe muestra que debe comprobarse la red. Un fallo de activación intenta restaurar la configuración previa.

El backend no es root. Su regla sudo permite solo `gestur-wifi status` y `gestur-wifi apply`, y el helper root valida de nuevo el JSON de entrada. Usa la API D-Bus de NetworkManager para un UUID previamente elegido por el instalador; no ejecuta comandos enviados por el cliente, no expone claves Wi-Fi en respuestas/logs/argumentos de procesos y nunca da éxito simulado cuando NetworkManager no está disponible.

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

Abre http://127.0.0.1:3000. El validador usa `../scripts/check_model.py` y necesita el entorno Python del proyecto; no se sustituye por una aprobación simulada en desarrollo. En equipos sin el helper Linux, Configuración muestra «Wi-Fi no disponible». `npm run dev` inicia Vite en 5173 y delega `/api` al backend 3000.

```bash
python3 -m unittest discover -s portal/test -p 'test_*.py'
bash -n scripts/install-portal.sh
npm --prefix portal audit --omit=dev
sudo systemctl status gestur-portal
sudo journalctl -u gestur-portal -n 50
```

Las pruebas API/ZIP y del helper usan directorios temporales y dobles de NetworkManager; no modifican el Wi-Fi del ordenador. La activación real y la reconexión deben verificarse en Raspberry Pi con NetworkManager y `wlan0`.

Referencias técnicas: [Mantine + Vite](https://mantine.dev/guides/vite/), [Fastify](https://fastify.dev/docs/latest/), [NetworkManager Update/GetSecrets](https://networkmanager.dev/docs/api/latest/gdbus-org.freedesktop.NetworkManager.Settings.Connection.html), [MAC permanente del dispositivo Wi-Fi](https://networkmanager.dev/docs/api/latest/gdbus-org.freedesktop.NetworkManager.Device.Wireless.html).
