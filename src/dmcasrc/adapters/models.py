"""Strict, project-neutral models for adapter evidence collection."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from dmcasrc.domain.graph import PipelineGraph
from dmcasrc.domain.ledger import CandidateLedger
from dmcasrc.infrastructure.deployment_manifest_schema import ArtifactManifest

ADAPTER_CONTEXT_SCHEMA_VERSION = "dmcasrc.adapter-run-context.initial"
ADAPTER_BUNDLE_SCHEMA_VERSION = "dmcasrc.adapter-evidence-bundle.initial"
ADAPTER_SUMMARY_SCHEMA_VERSION = "dmcasrc.adapter-run-summary.initial"
KNOWN_OBLIGATIONS = frozenset(f"O{index}" for index in range(1, 13))
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class AdapterModelError(ValueError):
    pass


def _keys(name: str, payload: Mapping[str, Any], expected: set[str]) -> None:
    actual = {str(key) for key in payload}
    if actual != expected:
        raise AdapterModelError(
            f"{name} fields differ: missing={sorted(expected - actual)}, "
            f"unknown={sorted(actual - expected)}"
        )


def _text(name: str, value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise AdapterModelError(f"{name} must be a non-empty string")
    return value.strip()


def _strings(name: str, value: Any) -> tuple[str, ...]:
    if not isinstance(value, list | tuple):
        raise AdapterModelError(f"{name} must be an array")
    result = tuple(_text(name, item) for item in value)
    if len(result) != len(set(result)):
        raise AdapterModelError(f"{name} must not contain duplicates")
    return tuple(sorted(result))


def _timestamp(name: str, value: Any) -> str:
    raw = _text(name, value)
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError as exc:
        raise AdapterModelError(f"{name} must be ISO-8601") from exc
    if parsed.tzinfo is None:
        raise AdapterModelError(f"{name} must include a timezone")
    return raw


@dataclass(frozen=True)
class AdapterRunContext:
    schema_version: str
    project_id: str
    project_version: str
    project_commit: str
    run_id: str
    workspace: str
    config_ref: str
    output_dir: str
    environment_ref: str
    source_manifest_ref: str

    def __post_init__(self) -> None:
        if self.schema_version != ADAPTER_CONTEXT_SCHEMA_VERSION:
            raise AdapterModelError("unsupported adapter context schema")
        for name in (
            "project_id",
            "project_version",
            "project_commit",
            "run_id",
            "workspace",
            "config_ref",
            "output_dir",
            "environment_ref",
            "source_manifest_ref",
        ):
            object.__setattr__(self, name, _text(name, getattr(self, name)))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> AdapterRunContext:
        expected = {
            "schema_version",
            "project_id",
            "project_version",
            "project_commit",
            "run_id",
            "workspace",
            "config_ref",
            "output_dir",
            "environment_ref",
            "source_manifest_ref",
        }
        if not isinstance(payload, Mapping):
            raise AdapterModelError("adapter context must be an object")
        _keys("adapter context", payload, expected)
        return cls(**{key: payload[key] for key in expected})

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "project_id": self.project_id,
            "project_version": self.project_version,
            "project_commit": self.project_commit,
            "run_id": self.run_id,
            "workspace": self.workspace,
            "config_ref": self.config_ref,
            "output_dir": self.output_dir,
            "environment_ref": self.environment_ref,
            "source_manifest_ref": self.source_manifest_ref,
        }


@dataclass(frozen=True)
class AdapterScope:
    project_id: str
    adapter_version: str
    supported_obligations: tuple[str, ...]
    partial_obligations: tuple[str, ...]
    unverifiable_obligations: tuple[str, ...]
    not_applicable_obligations: tuple[str, ...]
    observed_layers: tuple[str, ...]
    evidence_sources: tuple[str, ...]
    limitations: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "project_id", _text("project_id", self.project_id))
        object.__setattr__(
            self, "adapter_version", _text("adapter_version", self.adapter_version)
        )
        categories = (
            "supported_obligations",
            "partial_obligations",
            "unverifiable_obligations",
            "not_applicable_obligations",
        )
        seen: set[str] = set()
        for name in categories:
            values = _strings(name, getattr(self, name))
            unknown = set(values) - KNOWN_OBLIGATIONS
            if unknown:
                raise AdapterModelError(f"{name} has unknown obligations: {sorted(unknown)}")
            overlap = seen.intersection(values)
            if overlap:
                raise AdapterModelError(
                    f"obligations occur in multiple scope categories: {sorted(overlap)}"
                )
            seen.update(values)
            object.__setattr__(self, name, values)
        if seen != KNOWN_OBLIGATIONS:
            raise AdapterModelError(
                f"scope must classify O1-O12 exactly, missing={sorted(KNOWN_OBLIGATIONS-seen)}"
            )
        for name in ("observed_layers", "evidence_sources", "limitations"):
            values = _strings(name, getattr(self, name))
            if not values:
                raise AdapterModelError(f"{name} must not be empty")
            object.__setattr__(self, name, values)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> AdapterScope:
        expected = {
            "project_id",
            "adapter_version",
            "supported_obligations",
            "partial_obligations",
            "unverifiable_obligations",
            "not_applicable_obligations",
            "observed_layers",
            "evidence_sources",
            "limitations",
        }
        if not isinstance(payload, Mapping):
            raise AdapterModelError("adapter scope must be an object")
        _keys("adapter scope", payload, expected)
        return cls(**{key: payload[key] for key in expected})

    def to_dict(self) -> dict[str, Any]:
        return {
            "project_id": self.project_id,
            "adapter_version": self.adapter_version,
            "supported_obligations": list(self.supported_obligations),
            "partial_obligations": list(self.partial_obligations),
            "unverifiable_obligations": list(self.unverifiable_obligations),
            "not_applicable_obligations": list(self.not_applicable_obligations),
            "observed_layers": list(self.observed_layers),
            "evidence_sources": list(self.evidence_sources),
            "limitations": list(self.limitations),
        }


@dataclass(frozen=True)
class EvidenceReference:
    path: str
    sha256: str
    media_type: str
    role: str

    def __post_init__(self) -> None:
        for name in ("path", "media_type", "role"):
            object.__setattr__(self, name, _text(name, getattr(self, name)))
        digest = _text("sha256", self.sha256)
        if not _SHA256.fullmatch(digest):
            raise AdapterModelError("sha256 must be 64 lowercase hex characters")
        object.__setattr__(self, "sha256", digest)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> EvidenceReference:
        if not isinstance(payload, Mapping):
            raise AdapterModelError("evidence reference must be an object")
        _keys("evidence reference", payload, {"path", "sha256", "media_type", "role"})
        return cls(**payload)

    def to_dict(self) -> dict[str, str]:
        return {
            "path": self.path,
            "sha256": self.sha256,
            "media_type": self.media_type,
            "role": self.role,
        }


@dataclass(frozen=True)
class AdapterRunSummary:
    schema_version: str
    project_id: str
    run_id: str
    started_at: str
    finished_at: str
    exit_code: int
    status: str
    semantic_hash: str
    probe_id: str | None

    def __post_init__(self) -> None:
        if self.schema_version != ADAPTER_SUMMARY_SCHEMA_VERSION:
            raise AdapterModelError("unsupported adapter summary schema")
        for name in ("project_id", "run_id", "status"):
            object.__setattr__(self, name, _text(name, getattr(self, name)))
        object.__setattr__(self, "started_at", _timestamp("started_at", self.started_at))
        object.__setattr__(self, "finished_at", _timestamp("finished_at", self.finished_at))
        if isinstance(self.exit_code, bool) or not isinstance(self.exit_code, int):
            raise AdapterModelError("exit_code must be an integer")
        if not _SHA256.fullmatch(_text("semantic_hash", self.semantic_hash)):
            raise AdapterModelError("semantic_hash must be 64 lowercase hex characters")
        if self.probe_id is not None:
            object.__setattr__(self, "probe_id", _text("probe_id", self.probe_id))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> AdapterRunSummary:
        expected = {
            "schema_version",
            "project_id",
            "run_id",
            "started_at",
            "finished_at",
            "exit_code",
            "status",
            "semantic_hash",
            "probe_id",
        }
        if not isinstance(payload, Mapping):
            raise AdapterModelError("adapter summary must be an object")
        _keys("adapter summary", payload, expected)
        return cls(**{key: payload[key] for key in expected})

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "project_id": self.project_id,
            "run_id": self.run_id,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "exit_code": self.exit_code,
            "status": self.status,
            "semantic_hash": self.semantic_hash,
            "probe_id": self.probe_id,
        }


@dataclass(frozen=True)
class AdapterEvidenceBundle:
    graph: PipelineGraph
    ledger: CandidateLedger
    artifacts: ArtifactManifest
    scope: AdapterScope
    raw_evidence_refs: tuple[EvidenceReference, ...]
    run_summary: AdapterRunSummary
    schema_version: str = ADAPTER_BUNDLE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != ADAPTER_BUNDLE_SCHEMA_VERSION:
            raise AdapterModelError("unsupported adapter bundle schema")
        if not isinstance(self.graph, PipelineGraph):
            raise AdapterModelError("graph must be PipelineGraph")
        if not isinstance(self.ledger, CandidateLedger):
            raise AdapterModelError("ledger must be CandidateLedger")
        if not isinstance(self.artifacts, ArtifactManifest):
            raise AdapterModelError("artifacts must be ArtifactManifest")
        if not isinstance(self.scope, AdapterScope):
            raise AdapterModelError("scope must be AdapterScope")
        if not isinstance(self.raw_evidence_refs, tuple) or not self.raw_evidence_refs:
            raise AdapterModelError("raw_evidence_refs must be a non-empty tuple")
        if any(not isinstance(item, EvidenceReference) for item in self.raw_evidence_refs):
            raise AdapterModelError("raw_evidence_refs has invalid item")
        if not isinstance(self.run_summary, AdapterRunSummary):
            raise AdapterModelError("run_summary must be AdapterRunSummary")
        if self.scope.project_id != self.run_summary.project_id:
            raise AdapterModelError("bundle project identity mismatch")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "graph": self.graph.to_dict(),
            "ledger": self.ledger.to_dict(),
            "artifacts": self.artifacts.to_dict(),
            "scope": self.scope.to_dict(),
            "raw_evidence_refs": [
                item.to_dict()
                for item in sorted(self.raw_evidence_refs, key=lambda ref: ref.path)
            ],
            "run_summary": self.run_summary.to_dict(),
        }


__all__ = [
    "ADAPTER_BUNDLE_SCHEMA_VERSION",
    "ADAPTER_CONTEXT_SCHEMA_VERSION",
    "ADAPTER_SUMMARY_SCHEMA_VERSION",
    "AdapterEvidenceBundle",
    "AdapterModelError",
    "AdapterRunContext",
    "AdapterRunSummary",
    "AdapterScope",
    "EvidenceReference",
    "KNOWN_OBLIGATIONS",
]
