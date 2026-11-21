"""Canonical I/O and integrity verification for artifact registry manifests."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .deployment_manifest_schema import (
    ArtifactManifest,
    ArtifactManifestValidationError,
)

DEFAULT_ARTIFACT_MANIFEST_FILENAME = "artifact-registry.json"


class ManifestDecodeError(ValueError):
    """Raised when JSON syntax or object-key identity is ambiguous."""


@dataclass(frozen=True)
class ArtifactVerification:
    artifact_id: str
    status: str
    error_code: str | None
    detail: str


def _unique_object(pairs: Iterable[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ManifestDecodeError(f"duplicate JSON field: {key}.")
        result[key] = value
    return result


def _canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def load_artifact_manifest(path: str | Path) -> ArtifactManifest:
    manifest_path = Path(path)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"artifact manifest not found: {manifest_path}")
    try:
        payload = json.loads(
            manifest_path.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_object,
        )
    except ManifestDecodeError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ManifestDecodeError(f"invalid artifact manifest JSON: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ArtifactManifestValidationError("artifact manifest root must be an object.")
    return ArtifactManifest.from_dict(payload)


def write_artifact_manifest(
    manifest: ArtifactManifest | Mapping[str, Any],
    path: str | Path,
) -> Path:
    normalized = (
        manifest if isinstance(manifest, ArtifactManifest) else ArtifactManifest.from_dict(manifest)
    )
    output = Path(path)
    _atomic_write_text(output, _canonical_json(normalized.to_dict()))
    return output


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_path(root: Path, relative: str) -> Path:
    resolved_root = root.resolve()
    candidate = (resolved_root / Path(relative)).resolve()
    try:
        candidate.relative_to(resolved_root)
    except ValueError as exc:
        raise ArtifactManifestValidationError(
            f"artifact path escapes verification root: {relative}"
        ) from exc
    return candidate


def _verify_one(manifest: ArtifactManifest, root: Path, index: int) -> ArtifactVerification:
    artifact = manifest.artifacts[index]
    path = _artifact_path(root, artifact.path)
    if not path.exists():
        return ArtifactVerification(
            artifact.artifact_id,
            "UNVERIFIABLE",
            "ARTIFACT_MISSING",
            f"declared artifact does not exist: {artifact.path}",
        )
    if not path.is_file():
        return ArtifactVerification(
            artifact.artifact_id,
            "VIOLATION",
            "ARTIFACT_NOT_FILE",
            f"declared artifact is not a regular file: {artifact.path}",
        )
    actual_size = path.stat().st_size
    if actual_size != artifact.size_bytes:
        return ArtifactVerification(
            artifact.artifact_id,
            "VIOLATION",
            "ARTIFACT_SIZE_MISMATCH",
            f"expected {artifact.size_bytes} bytes, observed {actual_size}",
        )
    actual_hash = _sha256(path)
    if actual_hash != artifact.sha256:
        return ArtifactVerification(
            artifact.artifact_id,
            "VIOLATION",
            "ARTIFACT_HASH_MISMATCH",
            f"expected {artifact.sha256}, observed {actual_hash}",
        )
    return ArtifactVerification(artifact.artifact_id, "PASS", None, "artifact verified")


def verify_artifact_manifest(
    manifest: ArtifactManifest | Mapping[str, Any],
    *,
    root: str | Path,
) -> tuple[ArtifactVerification, ...]:
    normalized = (
        manifest if isinstance(manifest, ArtifactManifest) else ArtifactManifest.from_dict(manifest)
    )
    root_path = Path(root)
    return tuple(
        _verify_one(normalized, root_path, index) for index in range(len(normalized.artifacts))
    )


def read_artifact_manifest_dict(path: str | Path) -> dict[str, Any]:
    return load_artifact_manifest(path).to_dict()


__all__ = [
    "ArtifactVerification",
    "DEFAULT_ARTIFACT_MANIFEST_FILENAME",
    "ManifestDecodeError",
    "load_artifact_manifest",
    "read_artifact_manifest_dict",
    "verify_artifact_manifest",
    "write_artifact_manifest",
]
