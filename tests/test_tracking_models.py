"""Provisioning safety and optional real CPU inference (no camera required)."""

import hashlib
import json
import zipfile
from pathlib import Path

import pytest

from scripts import provision_models


def asset(data):
    return {
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
        "url": "https://example.invalid/model",
    }


def manifest(tmp_path, monkeypatch):
    raw, processed = b"raw model", b"processed model"
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "assets": {"hand.tflite": asset(raw)},
                "hand_bundle": {
                    "filename": "hand.task",
                    "members": {"hand_detector.tflite": "hand.tflite"},
                    "processed_assets": {"hand_detector.tflite": asset(processed)},
                },
            }
        )
    )
    monkeypatch.setattr(provision_models, "MANIFEST", manifest_path)
    return raw, processed


def test_offline_integrity_check_never_uses_network(tmp_path, monkeypatch):
    raw, processed = manifest(tmp_path, monkeypatch)
    (tmp_path / "hand.tflite").write_bytes(raw)
    with zipfile.ZipFile(tmp_path / "hand.task", "w") as archive:
        archive.writestr("hand_detector.tflite", processed)

    def network(*args, **kwargs):
        pytest.fail("Offline verification must not access the network")

    monkeypatch.setattr(provision_models.urllib.request, "urlopen", network)
    provision_models.provision(tmp_path, check=True)
    (tmp_path / "hand.tflite").write_bytes(b"corrupt")
    with pytest.raises(RuntimeError, match="checksum"):
        provision_models.provision(tmp_path, check=True)


def test_corrupt_download_does_not_replace_existing_file(tmp_path, monkeypatch):
    import io

    manifest(tmp_path, monkeypatch)
    path = tmp_path / "hand.tflite"
    path.write_bytes(b"previous version")
    monkeypatch.setattr(
        provision_models.urllib.request,
        "urlopen",
        lambda *a, **kw: io.BytesIO(b"invalid"),
    )
    with pytest.raises(RuntimeError, match="Checksum"):
        provision_models.provision(tmp_path)
    assert path.read_bytes() == b"previous version"


def test_bundle_generation_is_deterministic_and_verified(tmp_path, monkeypatch):
    raw, processed = manifest(tmp_path, monkeypatch)
    (tmp_path / "hand.tflite").write_bytes(raw)
    monkeypatch.setattr(
        provision_models, "with_normalization_metadata", lambda *_: processed
    )
    provision_models.provision(tmp_path)
    first = (tmp_path / "hand.task").read_bytes()
    provision_models.provision(tmp_path)
    assert (tmp_path / "hand.task").read_bytes() == first
    provision_models.provision(tmp_path, check=True)
    with zipfile.ZipFile(tmp_path / "hand.task", "a") as archive:
        archive.writestr("unexpected.txt", b"data")
    with pytest.raises(RuntimeError, match="Contenido inesperado"):
        provision_models.provision(tmp_path, check=True)


def test_real_lite_tasks_accept_cpu_video_frames():
    model_dir = Path(__file__).resolve().parents[1] / "tracking_models"
    if not all(
        (model_dir / name).exists()
        for name in ("pose_landmarker_lite.task", "hand_landmarker_lite.task")
    ):
        pytest.skip("Run scripts/provision_models.py to test installed Lite models")
    np = pytest.importorskip("numpy")
    pytest.importorskip("mediapipe")
    from pose_detector import _MediaPipeBackend

    provision_models.provision(model_dir, check=True)
    backend = _MediaPipeBackend(model_dir, True, True, 0.5)
    try:
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        image = backend.image(frame, True)
        for timestamp in (1, 51):
            assert backend.pose.detect_for_video(image, timestamp).pose_landmarks == []
            assert backend.hands.detect_for_video(image, timestamp).hand_landmarks == []
    finally:
        backend.close()
