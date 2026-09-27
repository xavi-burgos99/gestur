PYTHON ?= .venv/bin/python
RUFF ?= .venv/bin/ruff
NPM ?= npm
PRETTIER = portal/node_modules/.bin/prettier --config portal/.prettierrc.json

.PHONY: format check test build

format:
	$(RUFF) check --fix .
	$(RUFF) format .
	$(NPM) --prefix portal run format
	$(PRETTIER) --write 'scripts/*.mjs' 'config/*.json' tracking_models/manifest.json

check:
	$(RUFF) check .
	$(RUFF) format --check .
	$(NPM) --prefix portal run format:check
	$(PRETTIER) --check 'scripts/*.mjs' 'config/*.json' tracking_models/manifest.json
	@for script in gestur.sh scripts/configure-xorg.sh scripts/install-portal.sh scripts/kiosk-session.sh; do \
		bash -n "$$script" || exit 1; \
	done

test:
	$(PYTHON) -m pytest -q
	$(NPM) --prefix portal test

build:
	$(NPM) --prefix portal run build
