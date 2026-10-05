# Architecture

Gestur has two local processes: a Python viewer with camera tracking, and a Node
portal. They exchange durable configuration and model metadata through files;
the portal does not need to run inside the graphics or inference loop.

## Viewer and tracking

| Module | Responsibility |
| --- | --- |
| `gestur/controller.py` | Start the application, reload settings, coordinate tracking and drawing, and publish runtime status. |
| `gestur/runtime_config.py` | Validate and migrate configuration; resolve model files and metadata within their library. |
| `gestur/runtime_state.py` | Transfer the latest tracking snapshot between threads and collect bounded frame timing samples. |
| `gestur/tracking_session.py` | Own tracker lifecycle, background startup, cancellation, and reconnection. |
| `gestur/pose_detector.py` | Capture the latest frame, run the required MediaPipe tasks within an inference budget, and extract gesture inputs. |
| `scripts/benchmark_tracking.py` | Run the standalone tracking benchmark, separately from the runtime package. |
| `gestur/primary_person.py` | Keep tracking associated with the first selected person rather than switching between visitors. |
| `gestur/tracking_geometry.py` | Convert landmarks into position, orientation, opening, and pinching measurements. |
| `gestur/control_system.py` | Map gesture inputs to object transforms, smoothing, and standby behavior. |
| `gestur/visualizer.py` | Load models, manage lighting and QR overlays, and render the scene. |
| `gestur/render_scheduler.py` | Decide whether another draw is necessary without slowing control updates. |
| `gestur/device_metrics.py` | Sample process load and device temperature for diagnostics. |

These root modules remain in place because the installed entry points and model
validation subprocess import them directly. Heavy native libraries and camera
access stay behind the boundaries that need them. Do not move a lazy import to
module scope solely to satisfy an import-order preference.

## Portal

`portal/src/main.jsx` owns session state and top-level navigation. The three
pages are separate components: `ModelsPage.jsx`, `ParametersPage.jsx`, and
`SettingsPage.jsx`. Focused editors handle controls, device operations, and
presets. `api.mjs` owns the shared HTTP client and authentication error handling.

The server separates responsibilities:

- `app.mjs`: HTTP routes, session/origin checks, and serialized device jobs.
- `credentials.mjs`: credential verification and session revision identity.
- `store.mjs`: configuration validation, atomic persistence, and model catalog.
- `presets.mjs`: complete parameter snapshots, kept separate from active models.
- `models.mjs`: upload lifecycle, extraction, and durable import jobs.
- `package-model.mjs`: resource resolution and portable model packaging.
- `native-model.mjs`, `model-worker.mjs`, and model-check modules: isolated
  conversion, simplification, and compatibility checks.
- `device.mjs` and `wifi.mjs`: narrowly defined calls to privileged helpers.

## Installation and privileges

`gestur.sh` installs the checked-out revision. Files under `deployment/` define
systemd and graphics configuration. `scripts/first-boot.py` and
`scripts/prepare-image.py` handle repeatable image provisioning.

The portal runs as `gestur-portal`; the display session runs as `gestur`.
Root-owned `scripts/gestur-device.py` and `scripts/gestur-wifi.py` validate their
own inputs and implement the exact operations allowed by sudo. Device mutation
jobs acknowledge the HTTP request before changing networking or rebooting.
They publish recoverable status without persisting passwords in job records.

## Data boundaries

`config/default.json` and `config/schema.json` define the shared runtime contract.
Application data lives under `/var/lib/gestur`; protected credentials and factory
defaults live under `/etc/gestur`. Model selection is durable. A model URL is
metadata and can update its overlay without reloading the mesh. Presets contain
all three parameter groups (`tracking`, `render`, `controls`) and preserve the
latest active model when applied.

`tests/` and `portal/test/` exercise these contracts with temporary data. Research
scripts and vendored experimental code under `scripts/research/` are not imported
by production code. Published benchmark reports keep the provenance of the
revision tested, even after the current source is reformatted or reorganized.
