#!/usr/bin/env python3
"""Inventory installed Python and locked Node dependencies and preserve notices.

Run with the project's Python after installing requirements and npm ci. This is
an evidence collector, not an SPDX license classifier or legal approval. It does
not install packages, import application code, or contact the network.
"""

import argparse
import hashlib
import json
import platform
import re
import shutil
import subprocess
from collections import Counter
from datetime import date
from importlib import metadata
from pathlib import Path

from packaging.requirements import Requirement

ROOT = Path(__file__).resolve().parents[1]
NOTICE_NAME = re.compile(
    r"^(licen[cs]e|copying|copyright|notice|authors)([._-].*)?$", re.I
)


def canonical(name):
    return re.sub(r"[-_.]+", "-", name).lower()


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def revision_evidence():
    revision = None
    try:
        directory = subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        if Path(directory).resolve() == ROOT:
            revision = subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=ROOT,
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
    except (OSError, subprocess.CalledProcessError):
        pass
    marker = ROOT / ".source-revision"
    declared = None
    if marker.is_file():
        text = marker.read_text().strip()
        if re.fullmatch(r"[0-9a-fA-F]{40}", text):
            declared = text.lower()
    return {"git_revision": revision, "source_revision_marker": declared}


def notice_path(path):
    return bool(NOTICE_NAME.fullmatch(path.name)) or any(
        part.lower() in ("licenses", "licences") for part in path.parts[:-1]
    )


def collect(path, output):
    data = path.read_bytes()
    if b"\0" in data or len(data) > 4 * 1024 * 1024:
        return None
    digest = sha256(data)
    # Content addressing preserves exact upstream bytes and deduplicates common
    # license texts while retaining every original path in the inventory.
    target = output / "texts" / (digest + ".txt")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    return {"sha256": digest, "bytes": len(data), "text": "texts/" + target.name}


def python_inventory(output):
    direct = {}
    for line in (ROOT / "requirements.txt").read_text().splitlines():
        if line.strip() and not line.lstrip().startswith("#"):
            req = Requirement(line)
            direct[canonical(req.name)] = str(req.specifier)
    installed = {canonical(d.metadata["Name"]): d for d in metadata.distributions()}
    runtime, pending = set(), list(direct)
    while pending:
        name = pending.pop()
        if name in runtime:
            continue
        runtime.add(name)
        dist = installed.get(name)
        if dist:
            for raw in dist.requires or []:
                req = Requirement(raw)
                if req.marker is None or req.marker.evaluate({"extra": ""}):
                    pending.append(canonical(req.name))
    rows = []
    for name, dist in sorted(installed.items()):
        m = dist.metadata
        notices = []
        for file in sorted(dist.files or []):
            if notice_path(file):
                path = Path(dist.locate_file(file))
                if path.is_file():
                    copied = collect(path, output)
                    if copied:
                        notices.append({"original_path": str(file), **copied})
        declared = m.get("License-Expression") or m.get("License")
        # Long license bodies remain complete in the evidence files above.
        if declared and len(declared) > 300:
            declared = declared.splitlines()[0] + " [full text in notices]"
        rows.append(
            {
                "name": m["Name"],
                "version": dist.version,
                "scope": "runtime" if name in runtime else "environment-only",
                "direct": name in direct,
                "requirement": direct.get(name),
                "license_metadata": declared,
                "license_classifiers": [
                    c for c in m.get_all("Classifier", []) if c.startswith("License ::")
                ],
                "project_urls": m.get_all("Project-URL", []),
                "homepage": m.get("Home-page"),
                "requires_dist": dist.requires or [],
                "notices": notices,
                "wheel_tags": (dist.read_text("WHEEL") or "").splitlines(),
            }
        )
    return rows, sorted(runtime - installed.keys())


def node_inventory(output):
    portal = ROOT / "portal"
    lock = json.loads((portal / "package-lock.json").read_text())
    root = lock["packages"][""]
    direct = set(root.get("dependencies", {})) | set(root.get("devDependencies", {}))
    rows = []
    for path, locked in sorted(lock["packages"].items()):
        if not path:
            continue
        folder = portal / path
        manifest = folder / "package.json"
        actual = json.loads(manifest.read_text()) if manifest.is_file() else {}
        name = actual.get("name", path.rsplit("node_modules/", 1)[-1])
        notices = []
        # Restrict traversal to this package: nested node_modules have their own
        # lock entries. Preserve README licensing tables in native bundles too.
        candidates = []
        if folder.is_dir():
            for item in folder.iterdir():
                if item.is_file() and (
                    notice_path(item)
                    or name.startswith("@img/")
                    and item.name in ("README.md", "versions.json")
                ):
                    candidates.append(item)
                elif item.is_dir() and item.name.lower() in ("licenses", "licences"):
                    candidates.extend(f for f in item.rglob("*") if f.is_file())
        for item in sorted(candidates):
            copied = collect(item, output)
            if copied:
                notices.append(
                    {"original_path": item.relative_to(folder).as_posix(), **copied}
                )
        if actual and not notices:
            for item in folder.iterdir():
                if item.is_file() and item.name.lower().startswith("readme"):
                    if re.search(r"licen[cs]e", item.read_text(errors="replace"), re.I):
                        copied = collect(item, output)
                        if copied:
                            notices.append({"original_path": item.name, **copied})
        rows.append(
            {
                "name": name,
                "version": locked.get("version"),
                "lock_path": path,
                "direct": name in direct and path == "node_modules/" + name,
                "dev": locked.get("dev", False),
                "optional": locked.get("optional", False),
                "installed": bool(actual),
                "installed_version": actual.get("version"),
                "license_metadata": locked.get("license"),
                "resolved": locked.get("resolved"),
                "integrity": locked.get("integrity"),
                "os": locked.get("os"),
                "cpu": locked.get("cpu"),
                "repository": actual.get("repository"),
                "author": actual.get("author"),
                "dependencies": locked.get("dependencies", {}),
                "notices": notices,
            }
        )
    return rows


def system_inventory(output):
    """Optional on-device image inventory; never installs or fetches anything."""
    if not shutil.which("dpkg-query"):
        raise SystemExit("--system requires dpkg-query on the system being audited")
    listing = subprocess.check_output(
        [
            "dpkg-query",
            "-W",
            "-f=${db:Status-Status}\t${binary:Package}\t${Version}\t${source:Package}\t${source:Version}\n",
        ],
        text=True,
    )
    rows = []
    for line in sorted(listing.splitlines()):
        status, name, version, source, source_version = line.split("\t")
        if status != "installed":
            continue
        notice = Path("/usr/share/doc") / name.split(":")[0] / "copyright"
        copied = collect(notice, output) if notice.is_file() else None
        rows.append(
            {
                "name": name,
                "version": version,
                "source": source,
                "source_version": source_version,
                "copyright": copied,
            }
        )
    common = []
    for path in sorted(Path("/usr/share/common-licenses").glob("*")):
        if path.is_file():
            copied = collect(path, output)
            if copied:
                common.append({"name": path.name, **copied})
    return {
        "packages": rows,
        "common_licenses": common,
        "limitation": "Copyright/source metadata only; corresponding source archives are not collected.",
    }


def write_index(report, output):
    lines = [
        "# Dependency notice index",
        "",
        "Generated by `scripts/audit_licenses.py`. License fields are upstream metadata, not a legal conclusion. The JSON records original paths, dependency edges, versions, integrity hashes and platform scope. Exact upstream notice bytes are stored once by SHA-256.",
        "",
    ]
    for title, key in (("Python environment", "python"), ("Node lockfile", "node")):
        lines += [
            "## " + title,
            "",
            "| Package | Version | Scope | Declared license | Preserved notices |",
            "| --- | --- | --- | --- | --- |",
        ]
        for row in report[key]:
            scope = row.get("scope") or ("dev" if row.get("dev") else "runtime")
            if key == "node" and not row["installed"]:
                scope += "; not installed on audited host"
            links = ", ".join(
                f"[{n['original_path']}]({n['text']})" for n in row["notices"]
            )
            license_name = (
                (row.get("license_metadata") or "NOT DECLARED")
                .replace("|", "\\|")
                .replace("\n", " ")
            )
            lines.append(
                f"| {row['name']} | {row['version']} | {scope} | {license_name} | {links or 'Not captured; see audit limitations'} |"
            )
        lines.append("")
    (output / "DEPENDENCIES.md").write_text("\n".join(lines).rstrip() + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "docs/licenses")
    parser.add_argument(
        "--system",
        action="store_true",
        help="Also read installed Debian package/copyright metadata",
    )
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    python, missing = python_inventory(output)
    node = node_inventory(output)
    report = {
        "schema_version": 1,
        "audit_date": date.today().isoformat(),
        **revision_evidence(),
        "host": {
            "system": platform.system(),
            "machine": platform.machine(),
            "python": platform.python_version(),
        },
        "inputs": {
            str(p): sha256((ROOT / p).read_bytes())
            for p in (Path("requirements.txt"), Path("portal/package-lock.json"))
        },
        "limitations": [
            "Installed host wheels are not a Raspberry Pi image SBOM.",
            "Missing optional platform packages are inventoried from lock metadata only.",
            "Embedded native libraries may need additional corresponding source and notices.",
            "Environment-only Python packages are included but not implied runtime dependencies.",
            "No network fetch, application execution, or package installation performed.",
            "A source_revision_marker is a declared installation marker, not verified Git state; it may be stale after an update.",
        ],
        "python": python,
        "missing_runtime_python": missing,
        "node": node,
    }
    if args.system:
        report["system"] = system_inventory(output)
    (output / "inventory.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    )
    write_index(report, output)
    print(
        json.dumps(
            {
                "python": len(python),
                "node_locked": len(node),
                "node_installed": sum(r["installed"] for r in node),
                "node_licenses": dict(Counter(r["license_metadata"] for r in node)),
                "missing_runtime_python": missing,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
