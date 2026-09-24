"""Versioned configuration shared by the renderer and the Node administration portal.

Validation uses the checked-in JSON Schema without a Python package dependency.
Only the schema keywords used in config/schema.json are needed here.
"""

import copy
import json
import math
import os
from pathlib import Path
import re
import tempfile


CONFIG_DIR = Path(__file__).resolve().parent / "config"
SYSTEM_CONFIG_PATH = Path("/var/lib/gestur/config.json")
MAX_CONFIG_BYTES = 1024 * 1024


class ConfigurationError(ValueError):
    """A configuration is malformed or outside supported operating limits."""


def default_config():
    """Return a fresh copy; callers can never mutate shared defaults."""
    return json.loads((CONFIG_DIR / "default.json").read_text(encoding="utf-8"))


def config_path(path=None):
    return Path(path or os.environ.get("GESTUR_CONFIG", SYSTEM_CONFIG_PATH)).expanduser()


def _validate(value, schema, location="config"):
    kind = schema.get("type")
    types = {
        "object": lambda item: isinstance(item, dict),
        "array": lambda item: isinstance(item, list),
        "string": lambda item: isinstance(item, str),
        "boolean": lambda item: isinstance(item, bool),
        "number": lambda item: type(item) in (int, float) and math.isfinite(item),
        "integer": lambda item: type(item) in (int, float) and math.isfinite(item) and int(item) == item,
    }
    if kind and not types[kind](value):
        raise ConfigurationError(f"{location}: expected {kind}")
    if "enum" in schema and value not in schema["enum"]:
        raise ConfigurationError(f"{location}: unsupported value {value!r}")
    if "const" in schema and value != schema["const"]:
        raise ConfigurationError(f"{location}: must be {schema['const']!r}")
    if isinstance(value, dict):
        missing = set(schema.get("required", [])) - value.keys()
        if missing:
            raise ConfigurationError(f"{location}: missing {', '.join(sorted(missing))}")
        properties = schema.get("properties", {})
        if schema.get("additionalProperties") is False:
            unknown = value.keys() - properties.keys()
            if unknown:
                raise ConfigurationError(f"{location}: unknown keys {', '.join(sorted(unknown))}")
        for key, child_schema in properties.items():
            if key in value:
                _validate(value[key], child_schema, f"{location}.{key}")
    elif isinstance(value, list):
        if len(value) > schema.get("maxItems", math.inf):
            raise ConfigurationError(f"{location}: too many items")
        for index, item in enumerate(value):
            _validate(item, schema.get("items", {}), f"{location}[{index}]")
    elif isinstance(value, str):
        if not schema.get("minLength", 0) <= len(value) <= schema.get("maxLength", math.inf):
            raise ConfigurationError(f"{location}: invalid text length")
        if "pattern" in schema and re.search(schema["pattern"], value) is None:
            raise ConfigurationError(f"{location}: invalid format")
    elif type(value) in (int, float):
        if not schema.get("minimum", -math.inf) <= value <= schema.get("maximum", math.inf):
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
    schema = json.loads((CONFIG_DIR / "schema.json").read_text(encoding="utf-8"))
    _validate(config, schema)
    ids = set()
    for mapping in config["controls"]["mappings"]:
        if mapping["id"] in ids:
            raise ConfigurationError(f"Duplicate mapping id: {mapping['id']}")
        ids.add(mapping["id"])
        if mapping["mode"] == "hybrid":
            if not mapping["left_threshold"] < mapping["center"] < mapping["right_threshold"]:
                raise ConfigurationError(f"{mapping['id']}: left_threshold < center < right_threshold required")
        if mapping["mode"] == "stepped":
            if mapping["small_scale"] > mapping["large_scale"]:
                raise ConfigurationError(f"{mapping['id']}: small_scale cannot exceed large_scale")
            if not 0 <= mapping["threshold"] - mapping["hysteresis"] <= mapping["threshold"] + mapping["hysteresis"] <= 1:
                raise ConfigurationError(f"{mapping['id']}: threshold and hysteresis must stay within [0, 1]")
    return copy.deepcopy(config)


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
    return validate_config(config)


def save_config(config, path=None):
    """Write and fsync before atomic replace so live readers see whole files."""
    validated = validate_config(config)
    destination = config_path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=destination.parent,
                                         prefix=f".{destination.name}.", delete=False) as handle:
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
