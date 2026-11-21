"""G6.1 obligation-layer-edge consistency (T3).

Enforces that an observation's declared primary layer and affected layers
match the canonical obligation definition, and that the first-invalid-edge
is bound to a canonical edge whose endpoints live in the affected layers.
An identity violation without a canonical edge reports
``EDGE_NOT_APPLICABLE`` instead of fabricating an edge.

This is a pure structural validator: it reads only the canonical obligation
contract, the observation, and the pipeline graph. It never reads scenario
truth or auditor outcomes.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from ..domain.contracts import (
    _CANONICAL_OBLIGATION_DEFINITIONS,
    CanonicalObservation,
)
from ..domain.graph import PipelineGraph

# identity obligations have no meaningful canonical data-flow edge
_IDENTITY_OBLIGATIONS = frozenset(
    item[0]
    for item in _CANONICAL_OBLIGATION_DEFINITIONS
    if item[0].endswith("_IDENTITY") or item[0] in ("ANCESTOR_CLOSURE", "CANDIDATE_IDENTITY")
)

_OBLIGATION_LAYERS = {
    item[0]: {"primary": item[2], "affected": frozenset(item[3])}
    for item in _CANONICAL_OBLIGATION_DEFINITIONS
}


@dataclass(frozen=True)
class LayerEdgeConsistency:
    """Consistency verdict for one observation."""

    observation_id: str
    obligation_id: str
    primary_layer_consistent: bool
    affected_layers_consistent: bool
    edge_bound_to_affected_layers: bool
    edge_not_applicable: bool
    issues: tuple[str, ...] = field(default_factory=tuple)

    @property
    def consistent(self) -> bool:
        return (
            self.primary_layer_consistent
            and self.affected_layers_consistent
            and (self.edge_bound_to_affected_layers or self.edge_not_applicable)
            and not self.issues
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "observation_id": self.observation_id,
            "obligation_id": self.obligation_id,
            "primary_layer_consistent": self.primary_layer_consistent,
            "affected_layers_consistent": self.affected_layers_consistent,
            "edge_bound_to_affected_layers": self.edge_bound_to_affected_layers,
            "edge_not_applicable": self.edge_not_applicable,
            "consistent": self.consistent,
            "issues": list(self.issues),
        }


def _edge_layer_ids(edge: Any, graph: PipelineGraph) -> frozenset[str]:
    if edge is None:
        return frozenset()
    source = getattr(edge, "source", None) or (edge.get("source_id") if isinstance(edge, dict) else None)
    target = getattr(edge, "target", None) or (edge.get("target_id") if isinstance(edge, dict) else None)
    layers = set()
    for node_id in (source, target):
        if node_id:
            for node in graph.nodes:
                if node.node_id == node_id:
                    layers.add(node.layer)
    return frozenset(layers)


def check_observation_consistency(
    observation: CanonicalObservation,
    graph: PipelineGraph,
) -> LayerEdgeConsistency:
    """Validate one observation against the canonical obligation contract."""
    issues: list[str] = []
    definition = _OBLIGATION_LAYERS.get(observation.obligation_id)
    if definition is None:
        return LayerEdgeConsistency(
            observation_id=observation.observation_id,
            obligation_id=observation.obligation_id,
            primary_layer_consistent=False,
            affected_layers_consistent=False,
            edge_bound_to_affected_layers=False,
            edge_not_applicable=False,
            issues=(f"unknown obligation {observation.obligation_id}",),
        )

    primary_ok = observation.primary_layer == definition["primary"]
    if not primary_ok:
        issues.append(
            f"primary_layer {observation.primary_layer!r} != canonical {definition['primary']!r}"
        )
    affected_ok = set(observation.affected_layers) <= set(definition["affected"])
    if not affected_ok:
        issues.append(
            f"affected_layers {sorted(observation.affected_layers)} not subset of "
            f"{sorted(definition['affected'])}"
        )

    edge = observation.declared_first_invalid_edge
    edge_not_applicable = False
    edge_bound = False
    if observation.obligation_id in _IDENTITY_OBLIGATIONS:
        # identity violations are allowed to have no canonical edge
        edge_not_applicable = edge is None
        if edge is not None:
            edge_layers = _edge_layer_ids(edge, graph)
            edge_bound = bool(edge_layers) and edge_layers <= set(definition["affected"])
            if not edge_bound:
                issues.append("identity edge endpoints outside affected layers")
    else:
        if edge is None:
            issues.append("non-identity violation missing first_invalid_edge")
        else:
            edge_layers = _edge_layer_ids(edge, graph)
            # both endpoints must lie inside the obligation's affected layers
            edge_bound = bool(edge_layers) and edge_layers <= set(definition["affected"])
            if not edge_bound:
                issues.append("first_invalid_edge endpoints outside affected layers")

    return LayerEdgeConsistency(
        observation_id=observation.observation_id,
        obligation_id=observation.obligation_id,
        primary_layer_consistent=primary_ok,
        affected_layers_consistent=affected_ok,
        edge_bound_to_affected_layers=edge_bound,
        edge_not_applicable=edge_not_applicable,
        issues=tuple(issues),
    )


def check_observations_consistency(
    observations: Sequence[CanonicalObservation],
    graph: PipelineGraph,
) -> dict[str, Any]:
    """Validate a full observation set; returns aggregate verdict dict."""
    rows = [check_observation_consistency(item, graph) for item in observations]
    return {
        "schema": "dmcasrc.obligation-layer-edge-consistency.initial",
        "checked": len(rows),
        "consistent": sum(1 for row in rows if row.consistent),
        "inconsistent": sum(1 for row in rows if not row.consistent),
        "results": [row.to_dict() for row in rows],
    }


__all__ = [
    "LayerEdgeConsistency",
    "check_observation_consistency",
    "check_observations_consistency",
]
