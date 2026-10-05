"""Per-model QR metadata is optional, bounded, and confined to its package."""

import json

import pytest

from runtime_config import ConfigurationError, load_model_url, validate_model_url


@pytest.mark.parametrize(
    "value, expected",
    [
        (None, None),
        ("", None),
        ("   ", None),
        (
            "  https://example.org/pieza?q=capitel#detalle  ",
            "https://example.org/pieza?q=capitel#detalle",
        ),
        ("http://10.42.0.1:8080/model", "http://10.42.0.1:8080/model"),
        ("https://[2001:db8::1]/model", "https://[2001:db8::1]/model"),
        ("HTTPS://example.org/", "HTTPS://example.org/"),
        ("https://example.org/niño", "https://example.org/niño"),
        (
            "https://example.org/" + "a" * (2048 - len("https://example.org/")),
            "https://example.org/" + "a" * (2048 - len("https://example.org/")),
        ),
    ],
)
def test_valid_model_urls_are_preserved_without_requests(value, expected):
    assert validate_model_url(value) == expected


@pytest.mark.parametrize(
    "value",
    [
        42,
        True,
        [],
        {},
        "example.org",
        "//example.org",
        "ftp://example.org/file",
        "javascript:alert(1)",
        "data:text/plain,hello",
        "file:///etc/passwd",
        "https://user:password@example.org/",
        "https://user@example.org/",
        "https://@example.org/",
        "http:///example.org",
        "https://example.org:65536",
        "https://example.org/a b",
        "https://example.org/\\other",
        "https://example.org/\nother",
        " https://example.org/\t",
        "https://example.org/\x00",
        "https://example.org/\x7f",
        "https://example.org/\x85",
        "https://example.org/" + "a" * 2048,
        "https://example.org/" + "ñ" * 1100,
        "https://example.org/\ud800",
    ],
)
def test_invalid_urls_are_rejected_before_the_viewer_can_generate_a_qr(value):
    with pytest.raises(ConfigurationError):
        validate_model_url(value)


def test_url_changes_read_fresh_metadata_and_legacy_models_have_no_qr(tmp_path):
    package = tmp_path / "11111111-1111-1111-1111-111111111111"
    package.mkdir()
    info = package / ".gestur-model.json"
    model_id = f"{package.name}/model.glb"
    metadata = {"entrypoint": "model.glb", "name": "Capitel"}
    info.write_text(json.dumps(metadata))
    assert load_model_url(model_id, tmp_path) is None
    assert load_model_url(None, tmp_path / "missing") is None
    metadata["url"] = "https://example.org/capitel"
    info.write_text(json.dumps(metadata))
    assert load_model_url(model_id, tmp_path) == metadata["url"]
    for value in (None, "", "  ", "file:///etc/passwd", {"unexpected": "object"}):
        metadata["url"] = value
        info.write_text(json.dumps(metadata))
        assert load_model_url(model_id, tmp_path) is None


def test_url_metadata_cannot_read_files_outside_the_model_library(tmp_path):
    library = tmp_path / "models"
    library.mkdir()
    package = library / "11111111-1111-1111-1111-111111111111"
    package.mkdir()
    info = package / ".gestur-model.json"
    outside = tmp_path / "private.json"
    outside.write_text('{"url":"https://example.org/private"}')
    info.symlink_to(outside)
    with pytest.raises(ConfigurationError):
        load_model_url(f"{package.name}/model.glb", library)
    info.unlink()
    package.rmdir()
    package.symlink_to(tmp_path)
    with pytest.raises(ConfigurationError):
        load_model_url(f"{package.name}/model.glb", library)
