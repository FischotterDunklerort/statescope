"""G6.1 T7: identity provenance graph (B.5).

Builds a provenance chain over identity/artifact obligations:
  ARTIFACT_IDENTITY -> MODEL_IDENTITY -> THRESHOLD_IDENTITY ->
  EXECUTION_IDENTITY

Each node records the declared identity value (expected vs observed) from
the canonical obligation evaluation. A directed edge parent -> child exists
when the child obligation consumes the parent's artifact/output identity.
Inconsistency is detected when an observed identity differs from the
expected identity declared for the same artifact.

Outcome-blind: derives purely from canonical observation structure
(identity evidence fields) and the pipeline graph.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from ..domain.contracts import CanonicalObservation
from ..domain.graph import PipelineGraph

_IDENTITY_OBLIGATION_ORDER = (
    "ARTIFACT_IDENTITY",
    "MODEL_IDENTITY",
    "THRESHOLD_IDENTITY",
    "EXECUTION_IDENTITY",
    "CANDIDATE_IDENTITY",
    "ANCESTOR_CLOSURE",
)


@dataclass(frozen=True)
class IdentityProvenanceNode:
    obligation_id: str
    node_id: str
    expected_identity: Any
    observed_identity: Any
    consistent: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "obligation_id": self.obligation_id,
            "node_id": self.node_id,
            "expected_identity": self.expected_identity,
            "observed_identity": self.observed_identity,
            "consistent": self.consistent,
        }


@dataclass(frozen=True)
class IdentityProvenanceGraph:
    nodes: tuple[IdentityProvenanceNode, ...]
    edges: tuple[tuple[str, str], ...]
    inconsistencies: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "dmcasrc.identity-provenance-graph.initial",
            "nodes": [node.to_dict() for node in self.nodes],
            "edges": [
                {"parent": parent, "child": child} for parent, child in self.edges
            ],
            "inconsistencies": list(self.inconsistencies),
        }

    @property
    def consistent(self) -> bool:
        return not self.inconsistencies and all(node.consistent for node in self.nodes)


def _identity_fields(observation: CanonicalObservation) -> dict[str, Any]:
    """Extract expected/observed identity from evidence refs when present.

    Evidence refs follow ``expected:<id>`` or ``observed:<id>`` conventions
    from the canonical identity evaluators; falls back to empty markers.
    """
    expected: list[str] = []
    observed: list[str] = []
    for ref in observation.evidence_refs:
        if ref.startswith("expected:"):
            expected.append(ref.split(":", 1)[1])
        elif ref.startswith("observed:"):
            observed.append(ref.split(":", 1)[1])
    return {
        "expected": tuple(sorted(set(expected))) or ("UNKNOWN",),
        "observed": tuple(sorted(set(observed))) or ("UNKNOWN",),
    }


def build_identity_provenance_graph(
    observations: Sequence[CanonicalObservation],
    graph: PipelineGraph,
) -> IdentityProvenanceGraph:
    """T7: build the identity provenance graph from canonical observations."""
    identity_obs = [
        item
        for item in observations
        if item.obligation_id in _IDENTITY_OBLIGATION_ORDER
    ]
    nodes: list[IdentityProvenanceNode] = []
    by_obligation: dict[str, IdentityProvenanceNode] = {}
    inconsistencies: list[str] = []
    for obs in sorted(identity_obs, key=lambda item: item.observation_id):
        fields = _identity_fields(obs)
        expected = fields["expected"]
        observed = fields["observed"]
        consistent = expected == observed
        node = IdentityProvenanceNode(
            obligation_id=obs.obligation_id,
            node_id=obs.node_id,
            expected_identity=expected,
            observed_identity=observed,
            consistent=consistent,
        )
        nodes.append(node)
        by_obligation[obs.obligation_id] = node
        if not consistent:
            inconsistencies.append(
                f"{obs.obligation_id}@{obs.node_id}: expected={expected} observed={observed}"
            )

    # chain edges: identity obligations in canonical order consume upstream identity
    edges: list[tuple[str, str]] = []
    rank = {name: index for index, name in enumerate(_IDENTITY_OBLIGATION_ORDER)}
    present = [node for node in nodes if node.obligation_id in rank]
    present.sort(key=lambda node: (rank[node.obligation_id], node.node_id))
    for index in range(1, len(present)):
        parent = present[index - 1]
        child = present[index]
        # edge only when the child's node is downstream of the parent's node
        try:
            downstream = parent.node_id in set(graph.ancestors(child.node_id))
        except KeyError:
            downstream = False
        if downstream or parent.node_id == child.node_id:
            edges.append((parent.obligation_id, child.obligation_id))

    return IdentityProvenanceGraph(
        nodes=tuple(nodes),
        edges=tuple(sorted(set(edges))),
        inconsistencies=tuple(inconsistencies),
    )


__all__ = [
    "IdentityProvenanceGraph",
    "IdentityProvenanceNode",
    "build_identity_provenance_graph",
]
