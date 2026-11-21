"""Strict, versioned schema for the public artifact registry manifest.

The filename is retained because the repository layout was frozen before the
clean-room rewrite.  The object defined here is not a trading deployment
manifest: it is the project-neutral registry that binds audit evidence to
immutable artifacts.
"""

from __future__ import annotations

import copy
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import PurePosixPath
from typing import Any

ARTIFACT_REGISTRY_SCHEMA_VERSION = "dmcasrc.artifact-registry.initial"
KNOWN_REGISTRY_STATES = frozenset({"draft", "frozen", "revoked"})
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_ROOT_FIELDS = frozenset(
    {
        "schema_version",
        "registry_id",
        "created_at",
        "contract_version",
        "status",
        "producer",
        "environment",
        "artifacts",
    }
)
_ARTIFACT_FIELDS = frozenset(
    {
        "artifact_id",
        "kind",
        "path",
        "sha256",
        "size_bytes",
        "semantic_id",
        "media_type",
        "metadata",
    }
)
_PRODUCER_FIELDS = frozenset({"name", "version"})
_ENVIRONMENT_FIELDS = frozenset({"python"})
_METADATA_FIELDS = frozenset({"training_data_refs", "config_artifact_id"})


class ArtifactManifestValidationError(ValueError):
    """Raised when registry evidence does not satisfy the frozen schema."""


def _mapping(name: str, value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ArtifactManifestValidationError(
            f"{name} must be an object, got {type(value).__name__}."
        )
    return value


def _reject_unknown(name: str, value: Mapping[str, Any], allowed: frozenset[str]) -> None:
    unknown = sorted(str(key) for key in value if str(key) not in allowed)
    if unknown:
        raise ArtifactManifestValidationError(f"{name} has unknown field(s): {unknown}.")


def _required_string(name: str, value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ArtifactManifestValidationError(f"{name} must be a non-empty string.")
    return value.strip()


def _required_fields(name: str, value: Mapping[str, Any], fields: frozenset[str]) -> None:
    missing = sorted(fields.difference(str(key) for key in value))
    if missing:
        raise ArtifactManifestValidationError(f"{name} is missing required field(s): {missing}.")


def _canonical_relative_path(name: str, value: Any) -> str:
    raw = _required_string(name, value)
    path = PurePosixPath(raw)
    drive_like = len(raw) >= 2 and raw[1] == ":"
    if "\\" in raw or path.is_absolute() or drive_like or ".." in path.parts:
        raise ArtifactManifestValidationError(
            f"{name} must be a canonical relative POSIX path without traversal."
        )
    if raw != path.as_posix() or raw in {".", ""}:
        raise ArtifactManifestValidationError(f"{name} must be a canonical relative path.")
    return raw


def _timestamp(name: str, value: Any) -> str:
    raw = _required_string(name, value)
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ArtifactManifestValidationError(f"{name} must be an ISO-8601 timestamp.") from exc
    if parsed.tzinfo is None:
        raise ArtifactManifestValidationError(f"{name} must include a timezone.")
    return raw


def _string_map(name: str, value: Any) -> dict[str, str]:
    source = _mapping(name, value)
    normalized: dict[str, str] = {}
    for raw_key, raw_value in source.items():
        key = _required_string(f"{name}.<key>", raw_key)
        normalized[key] = _required_string(f"{name}.{key}", raw_value)
    return normalized


def _json_mapping(name: str, value: Any) -> dict[str, Any]:
    source = _mapping(name, value)
    result = {str(key): copy.deepcopy(item) for key, item in source.items()}
    if any(not isinstance(key, str) or not key for key in source):
        raise ArtifactManifestValidationError(f"{name} keys must be non-empty strings.")
    return result


def _metadata(name: str, value: Any) -> dict[str, Any]:
    """Strict artifact metadata: only training_data_refs / config_artifact_id."""
    source = _mapping(name, value)
    _reject_unknown(name, source, _METADATA_FIELDS)
    result: dict[str, Any] = {}
    if "training_data_refs" in source:
        refs = source["training_data_refs"]
        if not isinstance(refs, list) or not refs:
            raise ArtifactManifestValidationError(
                f"{name}.training_data_refs must be a non-empty array of strings."
            )
        if any(not isinstance(item, str) or not item for item in refs):
            raise ArtifactManifestValidationError(
                f"{name}.training_data_refs must contain only non-empty strings."
            )
        if len(refs) != len(set(refs)):
            raise ArtifactManifestValidationError(
                f"{name}.training_data_refs must not contain duplicates."
            )
        result["training_data_refs"] = list(refs)
    if "config_artifact_id" in source:
        cid = source["config_artifact_id"]
        if not isinstance(cid, str) or not cid.strip():
            raise ArtifactManifestValidationError(
                f"{name}.config_artifact_id must be a non-empty string."
            )
        result["config_artifact_id"] = cid.strip()
    return result


def _producer(name: str, value: Any) -> dict[str, str]:
    """Strict producer: exactly {name, version}, both required non-empty strings."""
    source = _mapping(name, value)
    _required_fields(name, source, _PRODUCER_FIELDS)
    _reject_unknown(name, source, _PRODUCER_FIELDS)
    return {
        "name": _required_string(f"{name}.name", source["name"]),
        "version": _required_string(f"{name}.version", source["version"]),
    }


def _environment(name: str, value: Any) -> dict[str, str]:
    """Strict environment: exactly {python}, required non-empty string."""
    source = _mapping(name, value)
    _required_fields(name, source, _ENVIRONMENT_FIELDS)
    _reject_unknown(name, source, _ENVIRONMENT_FIELDS)
    return {"python": _required_string(f"{name}.python", source["python"])}


@dataclass(frozen=True)
class ArtifactRecord:
    """Identity and integrity evidence for one immutable artifact."""

    artifact_id: str
    kind: str
    path: str
    sha256: str
    size_bytes: int
    semantic_id: str
    media_type: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "artifact_id", _required_string("artifact.artifact_id", self.artifact_id)
        )
        object.__setattr__(self, "kind", _required_string("artifact.kind", self.kind))
        object.__setattr__(self, "path", _canonical_relative_path("artifact.path", self.path))
        digest = _required_string("artifact.sha256", self.sha256)
        if not _SHA256_PATTERN.fullmatch(digest):
            raise ArtifactManifestValidationError(
                "artifact.sha256 must be 64 lowercase hex characters."
            )
        object.__setattr__(self, "sha256", digest)
        size = self.size_bytes
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise ArtifactManifestValidationError(
                "artifact.size_bytes must be a non-negative integer."
            )
        object.__setattr__(
            self, "semantic_id", _required_string("artifact.semantic_id", self.semantic_id)
        )
        object.__setattr__(
            self, "media_type", _required_string("artifact.media_type", self.media_type)
        )
        object.__setattr__(self, "metadata", _metadata("artifact.metadata", self.metadata))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ArtifactRecord:
        source = _mapping("artifact", payload)
        _required_fields("artifact", source, _ARTIFACT_FIELDS)
        _reject_unknown("artifact", source, _ARTIFACT_FIELDS)
        digest = _required_string("artifact.sha256", source["sha256"])
        if not _SHA256_PATTERN.fullmatch(digest):
            raise ArtifactManifestValidationError(
                "artifact.sha256 must be 64 lowercase hex characters."
            )
        size = source["size_bytes"]
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise ArtifactManifestValidationError(
                "artifact.size_bytes must be a non-negative integer."
            )
        return cls(
            artifact_id=_required_string("artifact.artifact_id", source["artifact_id"]),
            kind=_required_string("artifact.kind", source["kind"]),
            path=_canonical_relative_path("artifact.path", source["path"]),
            sha256=digest,
            size_bytes=size,
            semantic_id=_required_string("artifact.semantic_id", source["semantic_id"]),
            media_type=_required_string("artifact.media_type", source["media_type"]),
            metadata=_metadata("artifact.metadata", source["metadata"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "kind": self.kind,
            "path": self.path,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "semantic_id": self.semantic_id,
            "media_type": self.media_type,
            "metadata": copy.deepcopy(self.metadata),
        }


@dataclass(frozen=True)
class ArtifactManifest:
    """Versioned collection of artifacts supporting an audit or claim."""

    registry_id: str
    created_at: str
    contract_version: str
    status: str
    producer: dict[str, str]
    environment: dict[str, str]
    artifacts: tuple[ArtifactRecord, ...]
    schema_version: str = ARTIFACT_REGISTRY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        version = _required_string("schema_version", self.schema_version)
        if version != ARTIFACT_REGISTRY_SCHEMA_VERSION:
            raise ArtifactManifestValidationError(f"unsupported schema_version: {version}.")
        object.__setattr__(self, "schema_version", version)
        object.__setattr__(self, "registry_id", _required_string("registry_id", self.registry_id))
        object.__setattr__(self, "created_at", _timestamp("created_at", self.created_at))
        object.__setattr__(
            self,
            "contract_version",
            _required_string("contract_version", self.contract_version),
        )
        status = _required_string("status", self.status)
        if status not in KNOWN_REGISTRY_STATES:
            raise ArtifactManifestValidationError(f"unknown registry status: {status}.")
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "producer", _producer("producer", self.producer))
        object.__setattr__(self, "environment", _environment("environment", self.environment))
        if not isinstance(self.artifacts, tuple):
            raise ArtifactManifestValidationError("artifacts must be a tuple.")
        for item in self.artifacts:
            if not isinstance(item, ArtifactRecord):
                raise ArtifactManifestValidationError(
                    "artifacts must contain ArtifactRecord instances."
                )
        ids = [item.artifact_id for item in self.artifacts]
        if len(ids) != len(set(ids)):
            raise ArtifactManifestValidationError("duplicate artifact_id values are forbidden.")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ArtifactManifest:
        source = _mapping("artifact manifest root", payload)
        _required_fields("artifact manifest", source, _ROOT_FIELDS)
        _reject_unknown("artifact manifest", source, _ROOT_FIELDS)
        version = _required_string("schema_version", source["schema_version"])
        if version != ARTIFACT_REGISTRY_SCHEMA_VERSION:
            raise ArtifactManifestValidationError(f"unsupported schema_version: {version}.")
        status = _required_string("status", source["status"])
        if status not in KNOWN_REGISTRY_STATES:
            raise ArtifactManifestValidationError(f"unknown registry status: {status}.")
        raw_artifacts = source["artifacts"]
        if not isinstance(raw_artifacts, list):
            raise ArtifactManifestValidationError("artifacts must be a list.")
        artifacts = tuple(ArtifactRecord.from_dict(item) for item in raw_artifacts)
        ids = [item.artifact_id for item in artifacts]
        if len(ids) != len(set(ids)):
            raise ArtifactManifestValidationError("duplicate artifact_id values are forbidden.")
        return cls(
            registry_id=_required_string("registry_id", source["registry_id"]),
            created_at=_timestamp("created_at", source["created_at"]),
            contract_version=_required_string("contract_version", source["contract_version"]),
            status=status,
            producer=_producer("producer", source["producer"]),
            environment=_environment("environment", source["environment"]),
            artifacts=artifacts,
            schema_version=version,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "registry_id": self.registry_id,
            "created_at": self.created_at,
            "contract_version": self.contract_version,
            "status": self.status,
            "producer": dict(self.producer),
            "environment": dict(self.environment),
            "artifacts": [artifact.to_dict() for artifact in self.artifacts],
        }


__all__ = [
    "ARTIFACT_REGISTRY_SCHEMA_VERSION",
    "KNOWN_REGISTRY_STATES",
    "ArtifactManifest",
    "ArtifactManifestValidationError",
    "ArtifactRecord",
]
