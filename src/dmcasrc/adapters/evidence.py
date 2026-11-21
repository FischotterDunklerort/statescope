"""Path-safe raw evidence validation and stable manifest writing."""

from __future__ import annotations

import hashlib
import json
import mimetypes
import os
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any

from .models import EvidenceReference


class EvidencePathError(ValueError):
    pass


def resolve_evidence_path(workspace: str | Path, relative: str) -> Path:
    root = Path(workspace).resolve()
    raw = str(relative)
    posix = PurePosixPath(raw)
    if (
        "\\" in raw
        or posix.is_absolute()
        or ".." in posix.parts
        or raw != posix.as_posix()
        or raw in {"", "."}
    ):
        raise EvidencePathError("evidence path must be canonical relative POSIX")
    resolved = (root / Path(raw)).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise EvidencePathError("evidence path escapes workspace") from exc
    return resolved


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def evidence_reference(
    workspace: str | Path,
    relative: str,
    *,
    role: str,
    media_type: str | None = None,
) -> EvidenceReference:
    path = resolve_evidence_path(workspace, relative)
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("rb") as handle:
        handle.read(1)
    detected = media_type or mimetypes.guess_type(path.name)[0]
    return EvidenceReference(
        path=PurePosixPath(relative).as_posix(),
        sha256=sha256_file(path),
        media_type=detected or "application/octet-stream",
        role=role,
    )


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
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


def stable_manifest(references: tuple[EvidenceReference, ...]) -> dict[str, Any]:
    ordered = sorted(references, key=lambda item: (item.path, item.role))
    return {
        "schema_version": "dmcasrc.adapter-evidence-manifest.initial",
        "files": [item.to_dict() for item in ordered],
    }


__all__ = [
    "EvidencePathError",
    "atomic_json",
    "evidence_reference",
    "resolve_evidence_path",
    "sha256_file",
    "stable_manifest",
]
