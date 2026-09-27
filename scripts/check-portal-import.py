#!/usr/bin/env python3
"""Real portal import smoke on the Pi, using only Python's standard library.

Run locally on the Pi after deployment (normally with sudo for token/file access).
Uses HTTP 80 and requires an empty, otherwise idle library. The first import is
automatically selected by the portal. Inspect and delete through the API ONLY
the finished UUID package whose name, upload hash and private fixture marker
match this run, then verify the empty library.
A terminal import-job record remains as diagnostic history; it is not a model.
"""

import argparse
import base64
import hashlib
import http.cookiejar
import io
import json
import math
import re
import signal
import struct
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
import zipfile
import zlib
from pathlib import Path


class SmokeError(Exception):
    pass


class ConnectionFailure(SmokeError):
    pass


def require(condition, message):
    if not condition:
        raise SmokeError(message)


class Client:
    def __init__(self, base):
        self.base = base.rstrip("/")
        self.opener = urllib.request.build_opener(
            urllib.request.ProxyHandler({}),
            urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()),
        )

    def call(
        self,
        method,
        path,
        value=None,
        raw=None,
        content_type=None,
        expected=200,
        timeout=15,
    ):
        data = (
            raw
            if raw is not None
            else json.dumps(value).encode()
            if value is not None
            else None
        )
        headers = {"X-Gestur-Request": "1", "Origin": self.base}
        if data is not None:
            headers["Content-Type"] = content_type or "application/json"
        request = urllib.request.Request(
            self.base + path, data=data, headers=headers, method=method
        )
        try:
            with self.opener.open(request, timeout=timeout) as response:
                require(
                    response.status == expected, f"HTTP inesperado en {method} {path}"
                )
                result = json.loads(response.read(2 * 1024 * 1024))
                require(isinstance(result, dict), "Respuesta API no válida")
                return result
        except urllib.error.HTTPError as error:
            # Never print request bodies, cookie headers or the administrator token.
            raise SmokeError(f"HTTP {error.code} en {method} {path}") from None
        except (urllib.error.URLError, TimeoutError):
            raise ConnectionFailure(
                f"No se pudo conectar con {method} {path}"
            ) from None
        except json.JSONDecodeError:
            raise SmokeError(
                f"No se pudo obtener una respuesta válida en {method} {path}"
            ) from None


def wait_ready(client, timeout=15):
    """Wait only before login; import/configuration calls are never retried here."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        remaining = deadline - time.monotonic()
        try:
            state = client.call(
                "GET", "/api/session", timeout=min(2, max(0.01, remaining))
            )
        except ConnectionFailure:
            pass
        else:
            require(
                type(state.get("authenticated")) is bool,
                "Respuesta de disponibilidad no válida",
            )
            return
        remaining = deadline - time.monotonic()
        if remaining > 0:
            time.sleep(min(0.25, remaining))
    raise SmokeError(
        "El portal no estuvo disponible durante los 15 segundos de espera inicial"
    )


def png_texture():
    def chunk(kind, content):
        return (
            struct.pack(">I", len(content))
            + kind
            + content
            + struct.pack(">I", zlib.crc32(kind + content) & 0xFFFFFFFF)
        )

    rows = b"\x00\xff\x40\x20\x20\xc0\x80\x00\x20\x60\xe0\xf0\xd0\x40"
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", 2, 2, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(rows))
        + chunk(b"IEND", b"")
    )


def fixture(name, marker):
    texture = png_texture()
    data = io.BytesIO()
    with zipfile.ZipFile(data, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("smoke-owner.txt", marker)
        archive.writestr(
            "model.obj",
            "\n".join(
                [
                    "mtllib material.mtl",
                    "o import_smoke",
                    "v -1 -1 0",
                    "v 1 -1 0",
                    "v 1 1 0",
                    "v -1 1 0",
                    "vt 0 0",
                    "vt 1 0",
                    "vt 1 1",
                    "vt 0 1",
                    "vn 0 0 1",
                    "usemtl checker",
                    "f 1/1/1 2/2/1 3/3/1",
                    "f 1/1/1 3/3/1 4/4/1",
                    "",
                ]
            ),
        )
        # A moved Windows export: the real file must be resolved from the ZIP.
        archive.writestr(
            "material.mtl",
            "newmtl checker\nKd 1 1 1\nmap_Kd C:\\export\\textures\\GRID.PNG\n",
        )
        archive.writestr("textures/grid.png", texture)
    payload = data.getvalue()
    boundary = "gestur-smoke-" + uuid.uuid4().hex
    multipart = (
        (
            f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="{name}.zip"\r\n'
            "Content-Type: application/zip\r\n\r\n"
        ).encode()
        + payload
        + f"\r\n--{boundary}--\r\n".encode()
    )
    return payload, texture, multipart, f"multipart/form-data; boundary={boundary}"


def inspect_glb(path, original_texture):
    require(
        path.stat().st_size < 8 * 1024 * 1024,
        "GLB inesperadamente grande para esta fixture",
    )
    data = path.read_bytes()
    require(len(data) >= 20, "GLB incompleto")
    magic, version, length = struct.unpack_from("<III", data)
    require(
        (magic, version, length) == (0x46546C67, 2, len(data)), "Cabecera GLB inválida"
    )
    chunks, offset = {}, 12
    while offset < len(data):
        require(offset + 8 <= len(data), "Chunk GLB incompleto")
        size, kind = struct.unpack_from("<II", data, offset)
        offset += 8
        require(
            size % 4 == 0 and offset + size <= len(data),
            "Longitud de chunk GLB inválida",
        )
        chunks[kind] = data[offset : offset + size]
        offset += size
    doc = json.loads(chunks[0x4E4F534A])
    binary = chunks.get(0x004E4942, b"")
    require(doc.get("asset", {}).get("version") == "2.0", "Versión glTF inesperada")
    accessors = doc.get("accessors", [])
    triangles, material_indices = 0, set()
    for mesh in doc.get("meshes", []):
        for primitive in mesh.get("primitives", []):
            require(
                primitive.get("mode", 4) == 4,
                "El resultado no contiene triángulos normalizados",
            )
            require(
                "POSITION" in primitive["attributes"]
                and "TEXCOORD_0" in primitive["attributes"],
                "Faltan posiciones o coordenadas UV",
            )
            accessor = primitive.get("indices", primitive["attributes"]["POSITION"])
            count = accessors[accessor]["count"]
            require(count % 3 == 0, "Número de índices inválido")
            triangles += count // 3
            material_indices.add(primitive["material"])
    require(triangles == 2, f"Se esperaban 2 triángulos, recibidos {triangles}")
    textures = []
    for material_id in material_indices:
        material = doc["materials"][material_id]
        index = material["pbrMetallicRoughness"]["baseColorTexture"]["index"]
        image = doc["images"][doc["textures"][index]["source"]]
        if "bufferView" in image:
            view = doc["bufferViews"][image["bufferView"]]
            require(view.get("buffer", 0) == 0, "Textura fuera del buffer GLB")
            start = view.get("byteOffset", 0)
            encoded = binary[start : start + view["byteLength"]]
        else:
            uri = image.get("uri", "")
            require(
                uri.startswith("data:image/png;base64,"), "La textura no está embebida"
            )
            encoded = base64.b64decode(uri.split(",", 1)[1], validate=True)
        require(
            encoded == original_texture,
            "La textura PNG no conserva sus bytes originales",
        )
        textures.append(
            {"width": 2, "height": 2, "sha256": hashlib.sha256(encoded).hexdigest()}
        )
    require(bool(textures), "No hay textura aplicada a los materiales")
    return {
        "triangles": triangles,
        "uv_present": True,
        "material_textures": textures,
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def own_job(client, name, job_id=None):
    job = client.call("GET", "/api/imports/current").get("job")
    if not job or job.get("name") != name or (job_id and job.get("id") != job_id):
        return None
    require(
        bool(re.fullmatch(r"[a-f0-9-]{36}", job.get("id", ""))),
        "UUID de importación inválido",
    )
    return job


def wait_terminal(client, name, job_id, timeout):
    deadline = time.monotonic() + timeout
    while True:
        job = client.call("GET", "/api/imports/" + job_id).get("job")
        require(
            job and job.get("id") == job_id and job.get("name") == name,
            "La importación consultada no corresponde a esta prueba",
        )
        if job.get("state") in ("completed", "failed", "awaiting_decision"):
            return job
        require(
            time.monotonic() < deadline,
            "Tiempo de espera agotado; la importación aún está en curso",
        )
        time.sleep(1)


def without_selection(config):
    return {key: value for key, value in config.items() if key != "active_model"}


def library_entries(root):
    """Include unpublished uploads too; this smoke must have exclusive use."""
    return {
        item.name
        for item in root.iterdir()
        if not item.name.startswith(".")
        or (
            item.name.startswith((".import-", ".upload-"))
            and item.name != ".import-job.json"
        )
    }


def cleanup_fixture(
    client, root, name, marker, upload_hash, job_id, timeout, config_before
):
    job = own_job(client, name, job_id)
    if job and job.get("state") == "processing":
        job = wait_terminal(client, name, job["id"], timeout)
    if job and job.get("state") == "awaiting_decision":
        # This should never occur for two triangles; finish only our job so the
        # shared importer is not left awaiting consent. Publication selects it.
        client.call(
            "POST",
            f"/api/imports/{job['id']}/decision",
            {"simplify": False},
            expected=202,
        )
        job = wait_terminal(client, name, job["id"], timeout)
    if job is not None:
        job_id = job["id"]
    if job_id is None:
        return {"removed_own_package": False, "no_owned_job_found": True}
    require(
        bool(re.fullmatch(r"[a-f0-9-]{36}", job_id)), "UUID no seguro para limpieza"
    )
    package = root / job_id
    if not package.exists():
        return {
            "removed_own_package": False,
            "package_absent": True,
            "pending_workspace": (root / (".import-" + job_id)).exists(),
        }
    require(
        job is not None and job.get("state") in ("completed", "failed"),
        "La importación actual ya no pertenece a esta prueba; se conserva el paquete",
    )
    require(
        not package.is_symlink() and package.resolve().parent == root,
        "Ruta insegura; se conserva el paquete",
    )
    metadata = json.loads((package / ".gestur-model.json").read_text())
    require(
        metadata.get("name") == name, "El paquete no pertenece a la prueba; se conserva"
    )
    require(
        (package / "source/smoke-owner.txt").read_text() == marker,
        "No coincide el marcador; se conserva el paquete",
    )
    require(
        hashlib.sha256((package / "original-upload.zip").read_bytes()).hexdigest()
        == upload_hash,
        "No coincide el archivo subido; se conserva el paquete",
    )
    config = client.call("GET", "/api/config")["config"]
    active = config.get("active_model")
    model_id = job_id + "/model.glb"
    require(
        config_before is not None
        and config_before.get("active_model") is None
        and without_selection(config) == without_selection(config_before),
        "Se han cambiado otros ajustes durante la prueba; se conserva el paquete",
    )
    require(
        active in (None, model_id),
        "Se ha seleccionado otro modelo; se conserva el paquete",
    )
    catalog = client.call("GET", "/api/models")
    require(
        [model.get("id") for model in catalog.get("models", [])] == [model_id]
        and catalog.get("active") in (None, model_id),
        "Hay modelos ajenos a esta prueba; se conserva el paquete",
    )
    require(
        library_entries(root) == {job_id},
        "Hay otros paquetes o subidas en la biblioteca; se conserva el paquete",
    )
    require(
        own_job(client, name, job_id) is not None,
        "Ha cambiado la importación actual; se conserva el paquete",
    )
    deleted = client.call("DELETE", "/api/models", value={"id": model_id})
    require(
        deleted.get("config", {}).get("active_model") is None,
        "El borrado del último modelo no ha vaciado la selección",
    )
    require(not package.exists(), "El paquete sigue presente tras el borrado")
    # Never PUT an old snapshot over concurrent edits. The API owns fallback
    # selection and serializes it with deletion and other configuration changes.
    restored = client.call("GET", "/api/config")["config"]
    require(
        restored == config_before,
        "La configuración no se ha restaurado tras retirar la fixture",
    )
    return {
        "removed_own_package": True,
        "deleted_through_api": True,
        "job_id": job_id,
        "terminal_import_record_retained": True,
        "configuration_restored": True,
    }


def run(args):
    client = Client(args.base_url)
    name = "gestur-import-smoke-" + uuid.uuid4().hex
    marker = "GESTUR isolated import smoke\n" + uuid.uuid4().hex + "\n"
    upload, texture, multipart, content_type = fixture(name, marker)
    digest = hashlib.sha256(upload).hexdigest()
    result = {
        "ok": False,
        "fixture_name": name,
        "base_url": args.base_url,
        "scope": "POST real OBJ+MTL+PNG ZIP; selección automática temporal en biblioteca vacía",
        "upload_sha256": digest,
    }
    config_before = None
    job_id = None
    logged_in = False
    upload_attempted = False
    try:
        wait_ready(client)
        token = args.token_file.read_text().strip()
        require(len(token) >= 24, "La clave local no es válida")
        session = client.call("POST", "/api/session", {"token": token})
        token = None
        require(session.get("authenticated") is True, "No se obtuvo sesión autenticada")
        logged_in = True
        config_before = client.call("GET", "/api/config")["config"]
        catalog = client.call("GET", "/api/models")
        require(
            catalog.get("models") == [] and catalog.get("active") is None,
            "La biblioteca debe estar vacía y sin selección antes de probar",
        )
        require(
            config_before.get("active_model") is None,
            "Hay un modelo activo; no se modifica",
        )
        require(
            not library_entries(args.models_dir),
            "Hay paquetes o subidas previas en el directorio; no se modifica",
        )
        current = client.call("GET", "/api/imports/current").get("job")
        require(
            not current
            or current.get("state") not in ("processing", "awaiting_decision"),
            "Hay otra importación pendiente; no se interfiere",
        )
        upload_attempted = True
        accepted = client.call(
            "POST",
            "/api/models",
            raw=multipart,
            content_type=content_type,
            expected=202,
        )["job"]
        require(
            accepted.get("name") == name, "La respuesta no corresponde a la fixture"
        )
        job_id = accepted["id"]
        require(bool(re.fullmatch(r"[a-f0-9-]{36}", job_id)), "UUID no válido")
        result["job_id"] = job_id
        job = wait_terminal(client, name, job_id, args.timeout)
        result["import_state"] = job.get("state")
        require(
            job.get("state") == "completed",
            f"Importación no completada: {job.get('state')}",
        )
        require(
            job.get("proposal") is None,
            "Una fixture pequeña no debe solicitar simplificación",
        )
        model = job["model"]
        require(
            model["id"] == job_id + "/model.glb" and model.get("triangles") == 2,
            "Metadatos del modelo inesperados",
        )
        require(
            model.get("sourceFormat") == "OBJ" and model.get("simplified") is False,
            "No se ha verificado el camino OBJ nativo esperado",
        )
        selected = client.call("GET", "/api/config")["config"]
        catalog = client.call("GET", "/api/models")
        require(
            selected.get("active_model") == model["id"]
            and catalog.get("active") == model["id"],
            "La primera importación no se ha seleccionado automáticamente",
        )
        require(
            without_selection(selected) == without_selection(config_before),
            "La importación ha modificado otros ajustes",
        )
        require(
            [entry.get("id") for entry in catalog.get("models", [])] == [model["id"]],
            "Han aparecido otros modelos durante la prueba",
        )
        result["automatic_selection_verified"] = True
        result["glb"] = inspect_glb(args.models_dir / job_id / "model.glb", texture)
        result["repair_warnings"] = model.get("warnings", [])
        result["conversion_verified"] = True
    except (Exception, KeyboardInterrupt) as error:
        result["error_type"] = type(error).__name__
        result["error"] = (
            str(error)
            if isinstance(error, SmokeError)
            else "La prueba no pudo completarse; consulta el tipo de error"
        )
    finally:
        if logged_in:
            if upload_attempted:
                try:
                    result["cleanup"] = cleanup_fixture(
                        client,
                        args.models_dir,
                        name,
                        marker,
                        digest,
                        job_id,
                        args.timeout,
                        config_before,
                    )
                except (Exception, KeyboardInterrupt) as error:
                    result["cleanup_error"] = (
                        str(error)
                        if isinstance(error, SmokeError)
                        else type(error).__name__
                    )
            try:
                restored = client.call("GET", "/api/config")["config"]
                after = client.call("GET", "/api/models")
                result["library_empty"] = after.get("models") == []
                result["selection_restored"] = (
                    after.get("active") is None and restored.get("active_model") is None
                )
                result["configuration_restored"] = restored == config_before
            except Exception as error:
                result["verification_error"] = type(error).__name__
            try:
                client.call("DELETE", "/api/session")
            except Exception:
                result["session_logout_failed"] = True
        result["ok"] = bool(
            result.get("conversion_verified")
            and result.get("automatic_selection_verified")
            and result.get("library_empty")
            and result.get("selection_restored")
            and result.get("configuration_restored")
            and not result.get("error")
            and not result.get("cleanup_error")
            and not result.get("verification_error")
        )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return 0 if result["ok"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1",
        help="Local HTTP portal; default port 80",
    )
    parser.add_argument(
        "--token-file", type=Path, default=Path("/etc/gestur/portal-token")
    )
    parser.add_argument(
        "--models-dir", type=Path, default=Path("/var/lib/gestur/models")
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=120,
        help="Seconds for import, and bounded cleanup (1..600)",
    )
    args = parser.parse_args()
    url = urllib.parse.urlsplit(args.base_url)
    if (
        url.scheme != "http"
        or url.hostname not in ("127.0.0.1", "localhost", "::1")
        or url.username
        or url.password
        or url.query
        or url.fragment
        or url.path not in ("", "/")
    ):
        parser.error("--base-url debe ser una URL HTTP local, sin credenciales ni ruta")
    if not math.isfinite(args.timeout) or not 1 <= args.timeout <= 600:
        parser.error("--timeout debe estar entre 1 y 600 segundos")
    args.base_url = args.base_url.rstrip("/")
    args.models_dir = args.models_dir.resolve()
    if not args.models_dir.is_dir():
        parser.error("--models-dir debe existir")

    def interrupted(*_):
        raise KeyboardInterrupt()

    signal.signal(signal.SIGTERM, interrupted)
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
