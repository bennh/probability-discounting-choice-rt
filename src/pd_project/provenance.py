"""Content fingerprints, Git-state checks, and artifact manifest helpers."""

from __future__ import annotations

import csv
import copy
import hashlib
import json
from importlib.metadata import PackageNotFoundError, version
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
from typing import Any, Iterable
import zipfile

from .config import sha256_file


MANIFEST_COLUMNS = (
    "artifact",
    "stage",
    "timestamp_utc",
    "git_commit",
    "config_sha256",
    "raw_data_sha256",
    "processed_data_sha256",
    "data_pipeline_sha256",
    "raw_archive_sha256",
    "raw_source_mode",
    "artifact_sha256",
    "seed",
    "participant",
    "run",
    "condition",
    "model",
    "fit_status",
    "path",
)

DATA_PIPELINE_FILES = (
    "src/pd_project/config.py",
    "src/pd_project/data.py",
    "src/pd_project/provenance.py",
    "scripts/prepare_data.py",
)


def sha256_named_files(paths: Iterable[str | Path]) -> str:
    """Hash an unambiguously framed, location-independent file collection."""

    files = sorted(Path(path).resolve() for path in paths)
    if not files:
        raise ValueError("Cannot fingerprint an empty file collection.")
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Cannot fingerprint missing files: {missing}")
    common_parent = Path(os.path.commonpath([str(path.parent) for path in files]))
    records = [
        {
            "path": path.relative_to(common_parent).as_posix(),
            "sha256": sha256_file(path),
        }
        for path in files
    ]
    payload = json.dumps(records, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def sha256_mapping(mapping: dict[str, Any]) -> str:
    payload = json.dumps(mapping, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def data_transform_contract_hash(data_config: dict[str, Any]) -> str:
    """Hash data semantics while excluding the run-A-derived resolved cutoff values."""

    contract = copy.deepcopy(data_config)
    contract.get("rt_sensitivity", {}).pop("resolved_seconds", None)
    return sha256_mapping(contract)


def data_pipeline_source_hash(repository_root: str | Path) -> str:
    """Hash the source files that define canonical data preparation semantics."""

    root = Path(repository_root)
    paths = [root / relative_path for relative_path in DATA_PIPELINE_FILES]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing data-pipeline source files: {missing}")
    return sha256_named_files(paths)


def materialize_archive_generation(
    archive_path: str | Path,
    destination: str | Path,
    *,
    archive_hash: str,
    raw_glob: str,
) -> None:
    """Safely extract or verify one immutable, content-addressed archive generation."""

    archive = Path(archive_path).resolve()
    target_directory = Path(destination).resolve()
    if sha256_file(archive) != archive_hash:
        raise RuntimeError("The supplied raw archive hash does not match the archive bytes.")
    target_directory.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(
        prefix=f".{archive_hash[:12]}_", dir=target_directory.parent
    ) as staging_name:
        staging = Path(staging_name).resolve()
        with zipfile.ZipFile(archive) as source_archive:
            for member in source_archive.infolist():
                member_target = (staging / member.filename).resolve()
                if not member_target.is_relative_to(staging):
                    raise RuntimeError(f"Unsafe path in raw archive: {member.filename}")
            source_archive.extractall(staging)

        staged_files = sorted(path for path in staging.glob(raw_glob) if path.is_file())
        if not staged_files:
            raise FileNotFoundError("The raw archive contained no MATLAB participant files.")
        staged_snapshot = {
            path.relative_to(staging).as_posix(): sha256_file(path)
            for path in staged_files
        }

        marker = target_directory / ".complete"
        if target_directory.exists():
            if not marker.is_file() or marker.read_text(encoding="utf-8").strip() != archive_hash:
                raise RuntimeError(
                    f"Incomplete or inconsistent archive extraction exists at {target_directory}."
                )
            existing_files = sorted(
                path for path in target_directory.glob(raw_glob) if path.is_file()
            )
            existing_snapshot = {
                path.relative_to(target_directory).as_posix(): sha256_file(path)
                for path in existing_files
            }
            if existing_snapshot != staged_snapshot:
                raise RuntimeError(
                    "Cached raw extraction differs from the configured archive; "
                    "quarantine it and rerun preparation."
                )
            return

        (staging / ".complete").write_text(archive_hash + "\n", encoding="utf-8")
        staging.rename(target_directory)


def raw_source_snapshot(
    repository_root: str | Path,
    data_config: dict[str, Any],
    *,
    source_mode: str,
    expected_archive_sha256: str | None = None,
) -> tuple[list[Path], str | None]:
    """Resolve the exact raw source generation recorded by a canonical audit."""

    root = Path(repository_root).resolve()
    raw_directory = (root / data_config["raw_directory"]).resolve()
    archive_path = (root / data_config["raw_archive"]).resolve()
    raw_glob = data_config.get("raw_glob", "*.mat")

    if source_mode == "archive":
        if not archive_path.is_file():
            raise FileNotFoundError(f"Audited raw archive is missing: {archive_path}")
        archive_hash = sha256_file(archive_path)
        if archive_hash != expected_archive_sha256:
            raise RuntimeError(
                "The configured raw archive differs from the archive used to prepare data."
            )
        source_directory = raw_directory / "extracted" / archive_hash
        materialize_archive_generation(
            archive_path,
            source_directory,
            archive_hash=archive_hash,
            raw_glob=raw_glob,
        )
        files = sorted(path for path in source_directory.glob(raw_glob) if path.is_file())
        current_archive_hash: str | None = archive_hash
    elif source_mode == "direct_mat":
        if archive_path.exists():
            raise RuntimeError(
                "A raw archive now exists but the canonical audit used direct MAT files; "
                "rerun data preparation to select one source generation."
            )
        files = [
            path
            for path in sorted(raw_directory.glob(raw_glob))
            if path.is_file()
            and "extracted" not in path.relative_to(raw_directory).parts
        ]
        current_archive_hash = None
    else:
        raise ValueError(f"Unknown raw source mode: {source_mode!r}")

    if not files:
        raise FileNotFoundError("The audited raw source contains no MATLAB files.")
    return files, current_archive_hash


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
    if exists:
        with manifest.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            existing_columns = tuple(reader.fieldnames or ())
            existing_rows = list(reader)
        if any(None in row for row in existing_rows):
            raise RuntimeError(
                "Cannot migrate a malformed manifest containing rows wider than its header."
            )
        if existing_columns != MANIFEST_COLUMNS:
            unknown_columns = sorted(set(existing_columns) - set(MANIFEST_COLUMNS))
            if unknown_columns:
                raise RuntimeError(
                    "Cannot migrate manifest with unknown columns: "
                    f"{unknown_columns}."
                )
            with tempfile.NamedTemporaryFile(
                "w",
                encoding="utf-8",
                newline="",
                dir=manifest.parent,
                prefix=f".{manifest.name}.",
                suffix=".tmp",
                delete=False,
            ) as stream:
                temporary_path = Path(stream.name)
                writer = csv.DictWriter(stream, fieldnames=MANIFEST_COLUMNS)
                writer.writeheader()
                for row in existing_rows:
                    writer.writerow(
                        {column: row.get(column, "") for column in MANIFEST_COLUMNS}
                    )
            os.replace(temporary_path, manifest)
    with manifest.open("a", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=MANIFEST_COLUMNS, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        for record in records:
            writer.writerow({column: record.get(column, "") for column in MANIFEST_COLUMNS})
