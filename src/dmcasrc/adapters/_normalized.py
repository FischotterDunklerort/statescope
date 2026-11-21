"""Parsing for the normalized evidence files emitted by project runners."""

from __future__ import annotations

import csv
import json
from typing import Any

from dmcasrc.domain.graph import GraphEdge, GraphNode, PipelineGraph
from dmcasrc.domain.ledger import CandidateLedger, CandidateRecord
from dmcasrc.infrastructure.deployment_manifest_schema import ArtifactManifest

from .evidence import evidence_reference, resolve_evidence_path, sha256_file
from .models import (
    AdapterModelError,
    AdapterRunContext,
    AdapterRunSummary,
    AdapterScope,
    EvidenceReference,
)


def _json(context: AdapterRunContext, relative: str) -> Any:
    path = resolve_evidence_path(context.workspace, relative)
    return json.loads(path.read_text(encoding="utf-8"))


def read_graph(
    context: AdapterRunContext,
    *,
    required_layers: frozenset[str] = frozenset(),
) -> PipelineGraph:
    normalized = resolve_evidence_path(context.workspace, "graph.json")
    if normalized.is_file():
        graph = PipelineGraph.from_dict(
            json.loads(normalized.read_text(encoding="utf-8"))
        )
        layers = {node.layer for node in graph.nodes}
        missing = required_layers - layers
        if missing:
            raise AdapterModelError(
                f"graph missing required layers: {sorted(missing)}"
            )
        return graph

    path = resolve_evidence_path(context.workspace, "graph-events.jsonl")
    nodes: list[GraphNode] = []
    edges: list[GraphEdge] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        event = json.loads(line)
        if not isinstance(event, dict) or set(event) != {"event_type", "payload"}:
            raise AdapterModelError(f"invalid graph event at line {line_number}")
        if event["event_type"] == "node":
            nodes.append(GraphNode.from_dict(event["payload"]))
        elif event["event_type"] == "edge":
            edges.append(GraphEdge.from_dict(event["payload"]))
        else:
            raise AdapterModelError(f"unknown graph event: {event['event_type']}")
    graph = PipelineGraph(nodes=tuple(nodes), edges=tuple(edges))
    layers = {node.layer for node in graph.nodes}
    missing = required_layers - layers
    if missing:
        raise AdapterModelError(f"graph missing required layers: {sorted(missing)}")
    return graph


def read_ledger(
    context: AdapterRunContext,
    graph: PipelineGraph,
) -> CandidateLedger:
    normalized = resolve_evidence_path(context.workspace, "ledger.json")
    if normalized.is_file():
        ledger = CandidateLedger.from_dict(
            json.loads(normalized.read_text(encoding="utf-8"))
        )
        known = {node.node_id for node in graph.nodes}
        dangling = sorted({record.node_id for record in ledger.records} - known)
        if dangling:
            raise AdapterModelError(
                f"ledger references unknown graph nodes: {dangling}"
            )
        if not ledger.records:
            raise AdapterModelError("candidate ledger must not be empty")
        return ledger

    path = resolve_evidence_path(context.workspace, "candidates.csv")
    records: list[CandidateRecord] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        expected = {
            "candidate_key",
            "layer",
            "node_id",
            "decision_ts",
            "fold",
            "action",
            "outcome_ref",
            "evidence_json",
        }
        if set(reader.fieldnames or ()) != expected:
            raise AdapterModelError("candidate CSV has missing or unknown columns")
        for row in reader:
            records.append(
                CandidateRecord(
                    candidate_key=row["candidate_key"],
                    layer=row["layer"],
                    node_id=row["node_id"],
                    decision_ts=row["decision_ts"],
                    fold=row["fold"],
                    action=row["action"],
                    outcome_ref=row["outcome_ref"],
                    evidence=json.loads(row["evidence_json"]),
                )
            )
    if not records:
        raise AdapterModelError("candidate ledger must not be empty")
    known = {node.node_id for node in graph.nodes}
    dangling = sorted({record.node_id for record in records} - known)
    if dangling:
        raise AdapterModelError(f"ledger references unknown graph nodes: {dangling}")
    return CandidateLedger(records=tuple(records))


def read_artifacts(context: AdapterRunContext) -> ArtifactManifest:
    payload = _json(context, "artifact-manifest.json")
    manifest = ArtifactManifest.from_dict(payload)
    ids = {artifact.artifact_id for artifact in manifest.artifacts}
    for artifact in manifest.artifacts:
        path = resolve_evidence_path(context.workspace, artifact.path)
        if not path.is_file():
            raise AdapterModelError(f"artifact file missing: {artifact.path}")
        if path.stat().st_size != artifact.size_bytes:
            raise AdapterModelError(f"artifact size mismatch: {artifact.path}")
        if sha256_file(path) != artifact.sha256:
            raise AdapterModelError(f"artifact hash mismatch: {artifact.path}")
        for reference in artifact.metadata.get("training_data_refs", []):
            if reference not in ids:
                raise AdapterModelError(f"dangling training artifact: {reference}")
        config_id = artifact.metadata.get("config_artifact_id")
        if config_id is not None and config_id not in ids:
            raise AdapterModelError(f"dangling config artifact: {config_id}")
    return manifest


def read_scope(context: AdapterRunContext, expected_project: str) -> AdapterScope:
    scope = AdapterScope.from_dict(_json(context, "scope.json"))
    if scope.project_id != expected_project:
        raise AdapterModelError("scope project mismatch")
    return scope


def read_summary(context: AdapterRunContext, expected_project: str) -> AdapterRunSummary:
    summary = AdapterRunSummary.from_dict(_json(context, "run-summary.json"))
    if summary.project_id != expected_project or summary.run_id != context.run_id:
        raise AdapterModelError("run summary identity mismatch")
    return summary


def read_raw_references(context: AdapterRunContext) -> tuple[EvidenceReference, ...]:
    hashes = _json(context, "hashes.json")
    if (
        not isinstance(hashes, dict)
        or set(hashes) != {"schema_version", "algorithm", "files"}
        or hashes["schema_version"]
        not in {
            "dmcasrc.project-evidence-hashes.initial",
            "dmcasrc.project-evidence-hashes.subsequent",
            "dmcasrc.project-evidence-hashes.prior",
        }
        or hashes["algorithm"] != "sha256"
        or not isinstance(hashes["files"], dict)
    ):
        raise AdapterModelError("invalid project evidence hashes manifest")
    references: list[EvidenceReference] = []
    for relative, expected in sorted(hashes["files"].items()):
        reference = evidence_reference(
            context.workspace,
            relative,
            role=(
                "raw"
                if relative.startswith(("raw/", "vendor/"))
                else "normalized"
            ),
        )
        if reference.sha256 != expected:
            raise AdapterModelError(f"project evidence hash mismatch: {relative}")
        references.append(reference)
    if not references:
        raise AdapterModelError("project evidence hashes manifest is empty")
    return tuple(references)


__all__ = [
    "read_artifacts",
    "read_graph",
    "read_ledger",
    "read_raw_references",
    "read_scope",
    "read_summary",
]
