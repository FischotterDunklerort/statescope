"""Project-neutral, atomic artifact writing and registration."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any

from .deployment_manifest_schema import ArtifactRecord


class UnsafeArtifactPathError(ValueError):
    """Raised when an artifact path is absolute, ambiguous, or escapes its root."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _metadata(value: Mapping[str, Any] | None) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError("metadata must be a mapping.")
    return {str(key): item for key, item in value.items()}


class ArtifactWriter:
    """Write immutable evidence beneath one caller-provided artifact root."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def _target(self, relative_path: str | Path) -> tuple[str, Path]:
        raw = str(relative_path)
        path = PurePosixPath(raw)
        drive_like = len(raw) >= 2 and raw[1] == ":"
        if "\\" in raw or path.is_absolute() or drive_like or ".." in path.parts:
            raise UnsafeArtifactPathError(
                "artifact path must be a relative POSIX path without traversal."
            )
        canonical = path.as_posix()
        if raw != canonical or canonical in {"", "."}:
            raise UnsafeArtifactPathError("artifact path must be canonical and relative.")
        candidate = (self.root / Path(canonical)).resolve()
        try:
            candidate.relative_to(self.root)
        except ValueError as exc:
            raise UnsafeArtifactPathError("artifact path escapes the configured root.") from exc
        return canonical, candidate

    @staticmethod
    def _atomic_write(path: Path, content: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=path.parent,
                prefix=f".{path.name}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temporary = Path(handle.name)
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()

    @staticmethod
    def _record(
        path: str,
        file_path: Path,
        *,
        artifact_id: str,
        kind: str,
        semantic_id: str,
        media_type: str,
        metadata: Mapping[str, Any] | None,
    ) -> ArtifactRecord:
        return ArtifactRecord.from_dict(
            {
                "artifact_id": artifact_id,
                "kind": kind,
                "path": path,
                "sha256": _sha256(file_path),
                "size_bytes": file_path.stat().st_size,
                "semantic_id": semantic_id,
                "media_type": media_type,
                "metadata": _metadata(metadata),
            }
        )

    def write_bytes(
        self,
        relative_path: str | Path,
        content: bytes,
        *,
        artifact_id: str,
        kind: str,
        semantic_id: str,
        media_type: str = "application/octet-stream",
        metadata: Mapping[str, Any] | None = None,
    ) -> ArtifactRecord:
        if not isinstance(content, bytes):
            raise TypeError("content must be bytes.")
        canonical, target = self._target(relative_path)
        self._atomic_write(target, content)
        return self._record(
            canonical,
            target,
            artifact_id=artifact_id,
            kind=kind,
            semantic_id=semantic_id,
            media_type=media_type,
            metadata=metadata,
        )

    def write_text(
        self,
        relative_path: str | Path,
        content: str,
        *,
        artifact_id: str,
        kind: str,
        semantic_id: str,
        media_type: str = "text/plain",
        metadata: Mapping[str, Any] | None = None,
    ) -> ArtifactRecord:
        if not isinstance(content, str):
            raise TypeError("content must be a string.")
        return self.write_bytes(
            relative_path,
            content.encode("utf-8"),
            artifact_id=artifact_id,
            kind=kind,
            semantic_id=semantic_id,
            media_type=media_type,
            metadata=metadata,
        )

    def write_json(
        self,
        relative_path: str | Path,
        payload: Any,
        *,
        artifact_id: str,
        kind: str,
        semantic_id: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> ArtifactRecord:
        content = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        return self.write_text(
            relative_path,
            content,
            artifact_id=artifact_id,
            kind=kind,
            semantic_id=semantic_id,
            media_type="application/json",
            metadata=metadata,
        )

    def register_existing(
        self,
        relative_path: str | Path,
        *,
        artifact_id: str,
        kind: str,
        semantic_id: str,
        media_type: str = "application/octet-stream",
        metadata: Mapping[str, Any] | None = None,
    ) -> ArtifactRecord:
        canonical, target = self._target(relative_path)
        if not target.is_file():
            raise FileNotFoundError(f"artifact does not exist: {canonical}")
        return self._record(
            canonical,
            target,
            artifact_id=artifact_id,
            kind=kind,
            semantic_id=semantic_id,
            media_type=media_type,
            metadata=metadata,
        )


__all__ = ["ArtifactWriter", "UnsafeArtifactPathError"]
