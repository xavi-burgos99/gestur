"""Versioned configuration shared by the renderer and the Node administration portal.

Validation uses the checked-in JSON Schema without a Python package dependency.
Only the schema keywords used in config/schema.json are needed here.
"""

import copy
import json
import math
import os
import re
import tempfile
from pathlib import Path

CONFIG_DIR = Path(__file__).resolve().parent / "config"
SYSTEM_CONFIG_PATH = Path("/var/lib/gestur/config.json")
MAX_CONFIG_BYTES = 1024 * 1024

# JSON numbers exclude Python booleans, despite bool being an int subclass.
_SCHEMA_TYPES = {
    "object": lambda item: isinstance(item, dict),
    "array": lambda item: isinstance(item, list),
    "string": lambda item: isinstance(item, str),
    "boolean": lambda item: isinstance(item, bool),
    "number": lambda item: type(item) in (int, float) and math.isfinite(item),
    "integer": lambda item: (
        type(item) in (int, float) and math.isfinite(item) and int(item) == item
    ),
    "null": lambda item: item is None,
}


class ConfigurationError(ValueError):
    """A configuration is malformed or outside supported operating limits."""


def default_config():
    """Return a fresh copy; callers can never mutate shared defaults."""
    return json.loads((CONFIG_DIR / "default.json").read_text(encoding="utf-8"))


def config_path(path=None):
    return Path(
        path or os.environ.get("GESTUR_CONFIG", SYSTEM_CONFIG_PATH)
    ).expanduser()


def _validate(value, schema, location="config"):
    kind = schema.get("type")
    kinds = kind if isinstance(kind, list) else [kind]
    if kind and not any(_SCHEMA_TYPES[item](value) for item in kinds):
        raise ConfigurationError(f"{location}: expected {kind}")
    if "enum" in schema and value not in schema["enum"]:
        raise ConfigurationError(f"{location}: unsupported value {value!r}")
    if "const" in schema and value != schema["const"]:
        raise ConfigurationError(f"{location}: must be {schema['const']!r}")
    if isinstance(value, dict):
        missing = set(schema.get("required", [])) - value.keys()
        if missing:
            raise ConfigurationError(
                f"{location}: missing {', '.join(sorted(missing))}"
            )
        properties = schema.get("properties", {})
        if schema.get("additionalProperties") is False:
            unknown = value.keys() - properties.keys()
            if unknown:
                raise ConfigurationError(
                    f"{location}: unknown keys {', '.join(sorted(unknown))}"
                )
        for key, child_schema in properties.items():
            if key in value:
                _validate(value[key], child_schema, f"{location}.{key}")
    elif isinstance(value, list):
        if len(value) > schema.get("maxItems", math.inf):
            raise ConfigurationError(f"{location}: too many items")
        for index, item in enumerate(value):
            _validate(item, schema.get("items", {}), f"{location}[{index}]")
    elif isinstance(value, str):
        if (
            not schema.get("minLength", 0)
            <= len(value)
            <= schema.get("maxLength", math.inf)
        ):
            raise ConfigurationError(f"{location}: invalid text length")
        if "pattern" in schema and re.search(schema["pattern"], value) is None:
            raise ConfigurationError(f"{location}: invalid format")
    elif type(value) in (int, float):
        if (
            not schema.get("minimum", -math.inf)
            <= value
            <= schema.get("maximum", math.inf)
        ):
            raise ConfigurationError(f"{location}: value outside supported range")
    for child_schema in schema.get("allOf", []):
        _validate(value, child_schema, location)
    if "if" in schema:
        try:
            _validate(value, schema["if"], location)
        except ConfigurationError:
            _validate(value, schema.get("else", {}), location)
        else:
            _validate(value, schema.get("then", {}), location)


def validate_config(config):
    """Validate a complete configuration and return a detached copy.

    The extra relationships below are also enforced by the portal; JSON Schema
    alone cannot express comparisons between two property values.
    """
    config = copy.deepcopy(config)
    # Version 1 predates these optional behaviors. Fill absent fields and
    # recognized retired presets; reject explicitly invalid values as before.
    if isinstance(config, dict) and config.get("schema_version") == 1:
        for section, key, value in (
            ("render", "ambient_light", "none"),
            ("render", "exposure", 50),
            ("controls", "idle_mode", "return"),
        ):
            if isinstance(config.get(section), dict):
                config[section].setdefault(key, value)
        if isinstance(config.get("render"), dict):
            preset = config["render"].get("ambient_light")
            if isinstance(preset, str):
                config["render"]["ambient_light"] = {
                    "soft": "studio",
                    "warm": "sunset",
                    "cool": "gallery",
                    "contrast": "rim",
                }.get(preset, preset)
    schema = json.loads((CONFIG_DIR / "schema.json").read_text(encoding="utf-8"))
    _validate(config, schema)
    ids = set()
    enabled_outputs = set()
    for mapping in config["controls"]["mappings"]:
        if mapping["id"] in ids:
            raise ConfigurationError(f"Duplicate mapping id: {mapping['id']}")
        ids.add(mapping["id"])
        if mapping["enabled"]:
            if mapping["output"] in enabled_outputs:
                raise ConfigurationError(
                    "Each movement may have only one enabled gesture mapping"
                )
            enabled_outputs.add(mapping["output"])
        if mapping["mode"] == "hybrid":
            if (
                not mapping["left_threshold"]
                < mapping["center"]
                < mapping["right_threshold"]
            ):
                raise ConfigurationError(
                    f"{mapping['id']}: left_threshold < center < right_threshold required"
                )
        if mapping["mode"] == "stepped":
            if mapping["small_scale"] > mapping["large_scale"]:
                raise ConfigurationError(
                    f"{mapping['id']}: small_scale cannot exceed large_scale"
                )
            if (
                not 0
                <= mapping["threshold"] - mapping["hysteresis"]
                <= mapping["threshold"] + mapping["hysteresis"]
                <= 1
            ):
                raise ConfigurationError(
                    f"{mapping['id']}: threshold and hysteresis must stay within [0, 1]"
                )
    return config


def load_config(path=None):
    """Read a complete validated configuration, using defaults if absent."""
    source = config_path(path)
    try:
        with source.open("r", encoding="utf-8") as handle:
            serialized = handle.read(MAX_CONFIG_BYTES + 1)
    except FileNotFoundError:
        return validate_config(default_config())
    except OSError as exc:
        raise ConfigurationError(f"Cannot read configuration {source}: {exc}") from exc
    if len(serialized.encode("utf-8")) > MAX_CONFIG_BYTES:
        raise ConfigurationError("Configuration exceeds 1 MiB")
    try:
        config = json.loads(serialized)
    except (json.JSONDecodeError, ValueError) as exc:
        raise ConfigurationError(f"Invalid JSON in {source}: {exc}") from exc
    # The old bundled exhibition is no longer shipped. Imported packages have
    # their own directory and must retain their selection during this migration.
    if isinstance(config, dict) and config.get("active_model") == "capitell.obj":
        config["active_model"] = None
    return validate_config(config)


def available_models(models_dir):
    """Return complete imported models in the portal's deterministic order.

    Called on startup/config or catalog changes, never in the drawing loop.
    Hidden import workspaces and missing or escaping entrypoints are excluded.
    """
    root = Path(models_dir).resolve()
    try:
        packages = sorted(root.iterdir(), key=lambda entry: entry.name)
    except FileNotFoundError:
        return []
    models = []
    for package in packages:
        if not re.fullmatch(r"[a-f0-9-]{36}", package.name):
            continue
        try:
            package_root = package.resolve(strict=True)
            if package_root == root or not package_root.is_relative_to(root):
                continue
            metadata = json.loads(
                (package / ".gestur-model.json").read_text(encoding="utf-8")
            )
            if not isinstance(metadata, dict):
                continue
            entrypoint = metadata.get("entrypoint")
            if not isinstance(entrypoint, str) or not re.fullmatch(
                r"[A-Za-z0-9][A-Za-z0-9_. -]*(/[A-Za-z0-9][A-Za-z0-9_. -]*)*\.(obj|gltf|glb)",
                entrypoint,
            ):
                continue
            model_path = (package_root / entrypoint).resolve(strict=True)
            if model_path.is_relative_to(package_root) and model_path.is_file():
                models.append(f"{package.name}/{entrypoint}")
        except (OSError, ValueError, TypeError):
            continue
    return models


def validate_model_orientation(orientation=None):
    """Fixed import rotation uses Cartesian degrees, independently of gestures."""
    if orientation is None:
        return {"x": 0, "y": 0, "z": 0}
    if (
        not isinstance(orientation, dict)
        or set(orientation) != {"x", "y", "z"}
        or any(
            type(value) is not int or value not in (0, 90, 180, 270)
            for value in orientation.values()
        )
    ):
        raise ConfigurationError(
            "La orientación requiere x, y, z de 0, 90, 180 o 270 grados"
        )
    return dict(orientation)


def _load_model_metadata(model_id, models_dir):
    """Read a package's small metadata file only after the catalog has changed."""
    if model_id is None:
        return {}
    root = Path(models_dir).resolve()
    package = (root / model_id.split("/", 1)[0]).resolve(strict=True)
    if package == root or not package.is_relative_to(root):
        raise ConfigurationError(
            "La orientación debe pertenecer a la biblioteca de Gestur"
        )
    source = (package / ".gestur-model.json").resolve(strict=True)
    if not source.is_relative_to(package):
        raise ConfigurationError(
            "Los metadatos deben estar dentro del paquete del modelo"
        )
    with source.open(encoding="utf-8") as handle:
        serialized = handle.read(MAX_CONFIG_BYTES + 1)
    if len(serialized.encode("utf-8")) > MAX_CONFIG_BYTES:
        raise ConfigurationError("Los metadatos del modelo superan 1 MiB")
    metadata = json.loads(serialized)
    if not isinstance(metadata, dict):
        raise ConfigurationError("Los metadatos del modelo no son válidos")
    return metadata


def load_model_orientation(model_id, models_dir):
    return validate_model_orientation(
        _load_model_metadata(model_id, models_dir).get("orientation")
    )


def validate_model_url(value):
    """Validate a QR destination without resolving it or making a request."""
    from urllib.parse import urlsplit

    if value is None:
        return None
    message = "La URL debe ser HTTP o HTTPS, sin credenciales, y ocupar como máximo 2048 bytes"
    if not isinstance(value, str) or re.search(
        r"[\x00-\x1f\x7f-\x9f\ud800-\udfff]", value
    ):
        raise ConfigurationError(message)
    value = value.strip()
    if not value:
        return None
    if (
        len(value) > 2048
        or len(value.encode("utf-8")) > 2048
        or re.search(r"\s|\\", value)
        or not re.match(r"https?://", value, re.I)
    ):
        raise ConfigurationError(message)
    try:
        parsed = urlsplit(value)
        if not parsed.hostname or "@" in parsed.netloc:
            raise ValueError("Missing host or userinfo")
        # Accessing .port rejects malformed or out-of-range ports.
        parsed.port
    except ValueError as error:
        raise ConfigurationError(message) from error
    return value


def load_model_url(model_id, models_dir):
    metadata = _load_model_metadata(model_id, models_dir)
    try:
        return validate_model_url(metadata.get("url"))
    except ConfigurationError:
        # An invalid old link must not prevent the object from being displayed.
        return None


def reconcile_model_selection(config, models_dir, *, preferred_model=None):
    """Resolve an existing selection without writing shared portal state.

    The portal owns persistence, serialized with explicit user choices. The
    viewer only derives a fallback so it can boot before the portal service,
    without overwriting a concurrent selection made by the portal.
    """
    models = available_models(models_dir)
    result = copy.deepcopy(config)
    if result["active_model"] not in models:
        result["active_model"] = (
            preferred_model if preferred_model in models else next(iter(models), None)
        )
    return result


def save_config(config, path=None):
    """Write and fsync before atomic replace so live readers see whole files."""
    validated = validate_config(config)
    destination = config_path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            os.fchmod(handle.fileno(), 0o640)
            json.dump(validated, handle, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        directory_fd = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return validated


class ConfigReloader:
    """Optional mtime-based reload helper; invalid edits retain the last good state."""

    def __init__(self, path=None):
        self.path = config_path(path)
        self.current = load_config(self.path)
        self._signature = self._stat()

    def _stat(self):
        try:
            stat = self.path.stat()
            return stat.st_ino, stat.st_mtime_ns, stat.st_size
        except FileNotFoundError:
            return None

    def reload_if_changed(self):
        signature = self._stat()
        if signature == self._signature:
            return None
        updated = load_config(self.path)
        self.current = updated
        self._signature = signature
        return copy.deepcopy(updated)
