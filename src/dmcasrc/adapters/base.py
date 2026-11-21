"""Project adapter protocol; audit rules intentionally do not belong here."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from dmcasrc.domain.graph import PipelineGraph
from dmcasrc.domain.ledger import CandidateLedger
from dmcasrc.infrastructure.deployment_manifest_schema import ArtifactManifest

from ._scientific import NativeSemanticRule
from .models import AdapterEvidenceBundle, AdapterRunContext, AdapterScope


@runtime_checkable
class ProjectAdapter(Protocol):
    project_id: str

    def version(self) -> str: ...

    def scope(self, context: AdapterRunContext) -> AdapterScope: ...

    def discover_graph(self, context: AdapterRunContext) -> PipelineGraph: ...

    def build_ledger(
        self,
        context: AdapterRunContext,
        graph: PipelineGraph,
    ) -> CandidateLedger: ...

    def register_artifacts(
        self,
        context: AdapterRunContext,
        graph: PipelineGraph,
        ledger: CandidateLedger,
    ) -> ArtifactManifest: ...

    def collect(self, context: AdapterRunContext) -> AdapterEvidenceBundle: ...


@runtime_checkable
class ScientificProjectAdapter(ProjectAdapter, Protocol):
    def native_fact_rules(self) -> tuple[NativeSemanticRule, ...]: ...


def validate_project_adapter(value: Any) -> ProjectAdapter:
    required = (
        "project_id",
        "version",
        "scope",
        "discover_graph",
        "build_ledger",
        "register_artifacts",
        "collect",
    )
    missing = [name for name in required if not hasattr(value, name)]
    if missing:
        raise TypeError(f"project adapter missing required members: {missing}")
    if not isinstance(value.project_id, str) or not value.project_id:
        raise TypeError("project_id must be a non-empty string")
    version = value.version()
    if not isinstance(version, str) or not version or version in {"latest", "unknown"}:
        raise ValueError("adapter version must be pinned")
    return value


def validate_scientific_project_adapter(value: Any) -> ScientificProjectAdapter:
    validated = validate_project_adapter(value)
    if not hasattr(validated, "native_fact_rules"):
        raise TypeError("scientific project adapter missing native_fact_rules")
    rules = validated.native_fact_rules()
    if not isinstance(rules, tuple) or not rules:
        raise TypeError("native_fact_rules must return a non-empty tuple")
    if any(not isinstance(item, NativeSemanticRule) for item in rules):
        raise TypeError("native_fact_rules returned an invalid rule")
    return validated


__all__ = [
    "ProjectAdapter",
    "ScientificProjectAdapter",
    "validate_project_adapter",
    "validate_scientific_project_adapter",
]
