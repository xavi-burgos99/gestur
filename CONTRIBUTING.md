# Contributing

## Development workflow

Create a Python 3.11 or 3.12 virtual environment and install both requirements
files. Install portal dependencies with `npm --prefix portal ci`. The production
installer uses only `requirements.txt`; formatters and test tools stay off the
appliance's runtime dependency list.

Before submitting a change:

```bash
make format
make check
make test
make build
```

`make test` runs Python tests, including the portal's privileged-helper fixtures,
and the Node test suite. Native rendering tests need an available graphics
backend; on a headless Linux host use an Xvfb display with Mesa. Optional native
import tests report a skip when their platform-specific tools are unavailable.
A skip does not verify the missing integration.

Use targeted tests while working, then the complete checks before integration.
Test observable behavior and failure recovery rather than source formatting or
private implementation details. Do not modify real device networking, credentials,
or model libraries from automated tests.

## Code conventions

- Python: Ruff, 88 columns, four spaces, double quotes, `snake_case` functions and
  variables, `PascalCase` classes. Keep imports explicit and sorted.
- JavaScript/JSX: Prettier, 80 columns, two spaces, double quotes, semicolons.
  Use `camelCase` functions and variables, and `PascalCase` React components.
- Shell: Bash, four spaces, quoted expansions, and `set -euo pipefail` for
  provisioning scripts. Keep generated configuration blocks readable.
- Use English for identifiers, comments, and docstrings. Keep existing Spanish
  interface text and validation messages unless the change concerns wording.
- Explain invariants, units, ownership, security boundaries, or non-obvious
  decisions. Avoid comments that restate an assignment or narrate an edit.
- Keep functions focused, but do not introduce a general framework to eliminate
  a small, domain-specific difference. Validate external input at its boundary.

Editor defaults are in `.editorconfig`; Python rules are in `pyproject.toml` and
portal rules in `portal/.prettierrc.json`. Do not format vendored research source:
its upstream license, content, and manifest checksums must remain intact.

## Architecture and compatibility

The [architecture guide](docs/architecture.md) identifies each module's owner and
boundary. Keep the root Python entry points and installation paths compatible:
the installer, systemd units, image preparation, model checker, and research
scripts use them directly.

- Changes to settings must preserve `config/schema.json`, Python validation, API
  validation, migration, and default values as one contract.
- Keep camera and inference work outside the render thread. Closing a tracker
  must stop its worker before another tracker acquires the camera.
- Never replace a user's configuration with defaults after a read failure.
- Keep root-only operations in the restricted device helpers. Do not broaden
  sudo rules or invoke shell strings assembled from client input.
- Keep imports in their isolated worker and confine model resources to the
  uploaded package. Preserve resource limits and atomic publication.
- A preset contains tracking, render, and control parameters; it must not replace
  the latest model selection, model metadata, credentials, or network settings.

## Repository contents

Keep source, configuration schemas, tests, documentation, and small reproducible
fixtures in Git. Keep environments, compiled portal assets, downloaded tracking
weights, temporary measurements, credentials, and ZIP copies of the capitel out.
Do not delete the original example, upstream notices, or published benchmark
provenance as part of routine cleanup.

Historical benchmark records describe their original revision. If benchmark
code changes, preserve old measurements and record a new revision for a new run;
do not rewrite old hashes to match the current source.

## Licensing

Read [the licensing review](docs/licensing.md) and
[third-party notices](THIRD_PARTY_NOTICES.md) before adding dependencies, code,
weights, images, or models. Record the exact version, provenance, license, and
redistribution obligations. A repository license does not establish the rights
to separately downloaded model weights or training assets.

The maintainer must settle the project license and contribution terms before
accepting external contributions that would affect future commercial licensing.
