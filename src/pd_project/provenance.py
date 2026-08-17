"""Content fingerprints, Git-state checks, and artifact manifest helpers."""

from __future__ import annotations

import csv
import copy
import hashlib
import json
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any, Iterable

from .config import sha256_file


MANIFEST_COLUMNS = (
    "artifact",
    "stage",
    "timestamp_utc",
    "git_commit",
    "config_sha256",
    "raw_data_sha256",
    "processed_data_sha256",
    "artifact_sha256",
    "seed",
    "participant",
    "run",
    "condition",
    "model",
    "fit_status",
    "path",
)


def sha256_named_files(paths: Iterable[str | Path]) -> str:
    """Hash filenames and contents in deterministic sorted path order."""

    files = sorted((Path(path) for path in paths), key=lambda path: path.name)
    if not files:
        raise ValueError("Cannot fingerprint an empty file collection.")
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.name.encode("utf-8"))
        digest.update(sha256_file(path).encode("ascii"))
    return digest.hexdigest()


def sha256_mapping(mapping: dict[str, Any]) -> str:
    payload = json.dumps(mapping, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def data_transform_contract_hash(data_config: dict[str, Any]) -> str:
    """Hash data semantics while excluding the run-A-derived resolved cutoff values."""

    contract = copy.deepcopy(data_config)
    contract.get("rt_sensitivity", {}).pop("resolved_seconds", None)
    return sha256_mapping(contract)


def runtime_metadata() -> dict[str, Any]:
    """Capture numerical runtime versions without local filesystem paths."""

    packages = {}
    for package in ("numpy", "pandas", "scipy", "PyYAML"):
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = "missing"
    return {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "system": platform.system(),
        "machine": platform.machine(),
        "packages": packages,
        "byteorder": sys.byteorder,
    }


def clean_git_commit(root: str | Path) -> str:
    """Return HEAD only when the repository has no tracked/untracked changes."""

    repository = Path(root)
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError("This stage requires a committed Git repository state.")
    dirty = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if dirty.strip():
        raise RuntimeError("This stage requires a clean committed working tree.")
    return result.stdout.strip()


def append_manifest(path: str | Path, records: list[dict[str, Any]]) -> None:
    """Append provenance rows using one stable manifest schema."""

    manifest = Path(path)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    exists = manifest.exists()
    with manifest.open("a", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=MANIFEST_COLUMNS, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        for record in records:
            writer.writerow({column: record.get(column, "") for column in MANIFEST_COLUMNS})
