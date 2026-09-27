# Gestur

A camera-controlled 3D viewer for Raspberry Pi 5 installations. Visitors move a
model with head, body, and hand gestures; a local web portal manages models and
settings. After installation, the viewer and portal work without Internet access.

## Requirements

- Raspberry Pi 5 running **Raspberry Pi OS Lite, 64-bit**.
- HDMI display, compatible USB/V4L2 camera, and adequate cooling.
- Internet access over Ethernet during installation.
- Wi-Fi country configured in Raspberry Pi Imager or `raspi-config`.

The installer manages its own Python 3.12 environment and Node runtime. It does
not replace the system Python, update firmware, or change GPU memory settings.
CSI cameras need a capture adapter if they do not expose a compatible V4L2 device.

## Install

```bash
git clone --branch gestur-integrated https://github.com/xavi-burgos99/gestur.git
cd gestur
sudo bash gestur.sh install
sudo reboot
```

Installation is unattended and uses the checked-out source. It never pulls
`main`. To set a hostname during installation:

```bash
sudo bash gestur.sh install --hostname exhibition-1
```

Hostnames accept lowercase letters, digits, and internal hyphens, without
`.local`. Otherwise, new installations use `gestur-xxxx.local`, where `xxxx` is
the last four hexadecimal digits of the Wi-Fi adapter's permanent MAC address.

The access point initially opens as **GESTUR-XXXX**. Connect and open
**http://10.42.0.1**, the device's LAN IP, or its `.local` address, without a port.
The onboarding asks for an optional hostname, then a password and confirmation.
That password initially protects both the portal and the access point. Later
Wi-Fi password changes affect only the access point.

Upgrading preserves models, settings, hostname, and existing access credentials.
A fresh library contains no 3D models. An empty viewer shows the welcome screen
and a QR code linking to the device's IP address.

To prepare an OS image that installs Gestur on its first boot, follow the
[first-boot guide](docs/first-boot.md). To remove autostart and the portal while
keeping application data, run `sudo bash gestur.sh uninstall`.

## Portal

| Tab | Available settings |
| --- | --- |
| **Modelos 3D** | Import, select, rename, orient, and delete models; add an optional URL for a white, transparent QR code in the viewer. |
| **Parámetros** | Assign gestures to movements; adjust tracking, smoothing, lighting, exposure, standby behavior, and rendering; save, load, overwrite, and delete presets. |
| **Configuración** | Change the hostname and access point; erase content and settings to return to onboarding with an open access point. |

Upload a 3D file or a ZIP containing its materials and textures. Supported formats
include GLB, glTF, OBJ, FBX, STL, PLY, COLLADA, and other mesh formats supported by
Assimp. Import repairs resource references within the uploaded package and
produces a self-contained GLB. Limits are 100 MB per upload, 250 MB unpacked, and
500 files. Blender is not required.

Models above one million triangles offer an optional reduction to approximately
500,000 triangles. The Pi performs simplification only after confirmation. The
[original capitel example](examples/capitel/README.md) contains 491,038 triangles
and is not installed into the library automatically.

The first upload is selected automatically. The last selected model survives a
restart; deleting it selects another available model. The welcome screen appears
only when the library is empty. See the [portal guide](docs/portal.md) for details.

## Data and operation

| Path on the Pi | Purpose |
| --- | --- |
| `/opt/gestur` | Installed application and private runtimes |
| `/var/lib/gestur/config.json` | Active parameters and model selection |
| `/var/lib/gestur/presets.json` | Saved parameter presets |
| `/var/lib/gestur/models/` | Imported models and their metadata |
| `/var/log/gestur/viewer.log` | Viewer log |
| `/etc/gestur/` | Protected device state and access credentials |

The viewer starts on the device display; `gestur-portal.service` serves HTTP on
port 80 as an unprivileged account. Root-owned helpers perform the limited
hostname, Wi-Fi, onboarding, and reset operations.

```bash
systemctl status gestur-portal.service
journalctl -u gestur-portal.service -n 50
sudo tail -n 50 /var/log/gestur/viewer.log
```

## Development

Use Python **3.11 or 3.12** and Node **22.12 or newer**. Runtime versions are pinned
in `requirements.txt`; development tools are separate in `requirements-dev.txt`.

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt -r requirements-dev.txt
.venv/bin/python scripts/provision_models.py
npm --prefix portal ci
make check
make test
make build
```

Run the viewer locally:

```bash
.venv/bin/python controller.py --windowed
.venv/bin/python controller.py --windowed --no-camera
```

`Esc` closes the viewer. Hand recognition is optional and requires an active hand
control. Configure it through the portal or `config/default.json`. The
[portal development instructions](docs/portal.md#desarrollo-y-verificación) explain
how to run the API and Vite locally. Linux device settings are unavailable on a
regular development machine.

See [CONTRIBUTING.md](CONTRIBUTING.md) for formatting and review conventions, and
[architecture](docs/architecture.md) for module responsibilities. Code comments
and docstrings are in English; visitor-facing and portal copy remains Spanish.

## Performance

Capture keeps the newest frame instead of accumulating a queue. Inference loads
only the detectors required by active controls and reduces work when no person
or hand is present. Rendering responds to scene changes and draws less often
while idle. Resolution, geometry, textures, and antialiasing are configurable
without tying the control loop to every draw.

The production tracker uses MediaPipe Pose Landmarker Lite and optional Palm
Detection Lite / Hand Landmark Lite. Hand rotation, pinching, and opening are
computed from landmarks. Experimental trackers are kept outside the runtime.

The [Pi 5 validation report](docs/pi5-validation.md) includes the actual hardware,
versions, workloads, temperatures, and limitations of the measurements. A 60 FPS
setting is a target, not a guaranteed result. A historical 30-minute capitel trial
averaged 39.18 FPS at 1080p with replay input and continuous rotation, with no
reported throttling. Those results are not a precision study of visitor gestures.

## Documentation

Start at the [documentation index](docs/README.md) for installation, configuration,
tracking, performance, experiments, and licensing information. Published benchmark
results retain the revision and hashes measured at the time; later refactoring
does not constitute a new hardware benchmark.

## Credits and licensing

Created by **Xavier Burgos** — [xavi@dzin.es](mailto:xavi@dzin.es).

Originally an academic project at the Escola d'Enginyeria, Universitat Autònoma de
Barcelona, 2024/2025, supervised by Fernando Vilariño at the Centre de Visió per
Computador. Acknowledgements: Fran Iglesias, Fundación Épica – La Fura dels Baus,
and Cátedra UAB–Cruïlla (TSI-100929-2023-2).

The project license is under review. The intended policy is free noncommercial
use with attribution, and separate written permission for commercial use. This
statement is not a license grant. See the [licensing review](docs/licensing.md)
and [dependency audit](docs/licensing-audit.md) before redistributing Gestur.
Third-party components and example assets are not relicensed by that policy;
see [third-party notices](THIRD_PARTY_NOTICES.md).
