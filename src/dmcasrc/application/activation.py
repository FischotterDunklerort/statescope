"""G6.1 Mechanism Activation Instrumentation (B.1).

Determines whether each mechanism had an activation *opportunity* on a given
scenario, entirely from outcome-blind inputs (canonical observation structure,
pipeline graph topology, scenario truth structure). It never reads auditor
outcomes, scored results, scenario ids, or fault families.

Three mechanisms are instrumented:

  ancestor_required
      The scenario contains cross-layer reachability that a local-only audit
      would miss: at least one observation node has non-trivial ancestors, or
      the scenario truth declares an affected layer distinct from the primary
      layer. This is the activation opportunity for A1 (ancestor closure).

  resolver_required
      The scenario plausibly yields multiple candidate violations that
      compete for the primary finding: at least one observation declares
      multiple invalid nodes / affected layers, or the truth declares more
      than one affected obligation/layer. This is the activation opportunity
      for A2 (causal primary-finding resolver).

  propagation_required
      The scenario carries candidate-level witness evidence that must be
      propagated along dependencies: at least one observation lists more
      than one candidate witness, or the truth lists more than one expected
      candidate witness. This is the activation opportunity for A5
      (candidate witness propagation / closure).

Both a truth-side view (from a frozen scenario-truth catalog row, when
available) and a runtime view (from a CanonicalEvidenceGraph) are supported.
The truth-side view is outcome-blind by construction; the runtime view is
computed purely from the pre-audit evidence graph.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from ..domain.graph import PipelineGraph


@dataclass(frozen=True)
class ActivationOpportunity:
    """Outcome-blind activation opportunity for the three instrumented mechanisms."""

    ancestor_required: bool
    resolver_required: bool
    propagation_required: bool
    ancestor_required_count: int = 0
    resolver_required_count: int = 0
    propagation_required_count: int = 0
    rationale: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ancestor_required": self.ancestor_required,
            "resolver_required": self.resolver_required,
            "propagation_required": self.propagation_required,
            "ancestor_required_count": self.ancestor_required_count,
            "resolver_required_count": self.resolver_required_count,
            "propagation_required_count": self.propagation_required_count,
            "rationale": list(self.rationale),
        }

    @property
    def any_required(self) -> bool:
        return self.ancestor_required or self.resolver_required or self.propagation_required

    @property
    def activated_mechanisms(self) -> tuple[str, ...]:
        out = []
        if self.ancestor_required:
            out.append("ancestor_required")
        if self.resolver_required:
            out.append("resolver_required")
        if self.propagation_required:
            out.append("propagation_required")
        return tuple(out)


# ---------------------------------------------------------------------------
# Truth-side computation (outcome-blind by construction).
# ---------------------------------------------------------------------------


def _truth_affected_layers(truth: Mapping[str, Any]) -> tuple[str, ...]:
    raw = truth.get("expected_affected_layers") or truth.get("affected_layers") or ()
    if isinstance(raw, str):
        return (raw,)
    return tuple(raw)


def _truth_affected_obligations(truth: Mapping[str, Any]) -> tuple[str, ...]:
    raw = truth.get("expected_affected_obligations") or truth.get("affected_obligations") or ()
    if isinstance(raw, str):
        return (raw,)
    return tuple(raw)


def _truth_witness_ids(truth: Mapping[str, Any]) -> tuple[str, ...]:
    raw = (
        truth.get("expected_candidate_witness_ids")
        or truth.get("candidate_witness_ids")
        or ()
    )
    if isinstance(raw, str):
        return (raw,)
    return tuple(raw)


def _truth_complexity(truth: Mapping[str, Any]) -> Mapping[str, Any]:
    value = truth.get("complexity") or {}
    return value if isinstance(value, Mapping) else {}


def activation_opportunity_from_truth(
    truth: Mapping[str, Any],
) -> ActivationOpportunity:
    """Compute activation opportunity from a frozen scenario-truth row.

    The decision uses only structural truth fields (affected layers,
    affected obligations, candidate witness ids, complexity metadata). It is
    outcome-blind: it never reads ``expected_status``/``expected_clean_status``
    or any auditor-derived field.
    """
    layers = _truth_affected_layers(truth)
    obligations = _truth_affected_obligations(truth)
    witnesses = _truth_witness_ids(truth)
    complexity = _truth_complexity(truth)

    ancestor_count = 0
    resolver_count = 0
    propagation_count = 0
    rationale: list[str] = []

    # A1: cross-layer reachability. Affected layer differs from primary layer,
    # or explicit ancestor-depth metadata is positive.
    primary_layer = truth.get("expected_primary_layer") or truth.get("primary_layer")
    cross_layer = bool(layers) and primary_layer is not None and set(layers) - {primary_layer}
    try:
        ancestor_depth = int(complexity.get("ancestor_depth", 0) or 0)
    except (TypeError, ValueError):
        ancestor_depth = 0
    if cross_layer or ancestor_depth > 0:
        ancestor_count = max(len(layers), ancestor_depth)
        rationale.append(
            f"cross-layer truth: layers={sorted(set(layers))}, primary={primary_layer}, "
            f"ancestor_depth={ancestor_depth}"
        )

    # A2: multiple competing violations. More than one affected obligation,
    # or more than one affected layer implies resolver arbitration matters.
    if len(obligations) > 1 or len(set(layers)) > 1:
        resolver_count = max(len(obligations), len(set(layers)))
        rationale.append(
            f"multi-candidate truth: obligations={sorted(set(obligations))}, "
            f"layers={sorted(set(layers))}"
        )

    # A5: candidate witness propagation. More than one candidate witness
    # requires propagation/closure of candidate-level evidence.
    if len(witnesses) > 1:
        propagation_count = max(0, len(witnesses) - 1)
        rationale.append(f"witness propagation truth: witnesses={len(witnesses)}")

    return ActivationOpportunity(
        ancestor_required=ancestor_count > 0,
        resolver_required=resolver_count > 0,
        propagation_required=propagation_count > 0,
        ancestor_required_count=ancestor_count,
        resolver_required_count=resolver_count,
        propagation_required_count=propagation_count,
        rationale=tuple(rationale),
    )


# ---------------------------------------------------------------------------
# Runtime-side computation (from the pre-audit canonical evidence graph).
# ---------------------------------------------------------------------------


def activation_opportunity_from_graph(
    graph: PipelineGraph,
    observations: Sequence[Any],
) -> ActivationOpportunity:
    """Compute activation opportunity from the canonical evidence graph.

    Fully outcome-blind: uses only the graph topology and the canonical
    observation structure (affected layers, invalid node ids, candidate
    witness ids, declared edge).
    """
    ancestor_count = 0
    resolver_count = 0
    propagation_count = 0
    rationale: list[str] = []

    for obs in observations:
        node_id = getattr(obs, "node_id", None)
        affected_layers = tuple(getattr(obs, "affected_layers", ()) or ())
        invalid_nodes = tuple(getattr(obs, "invalid_node_ids", ()) or ())
        witnesses = tuple(getattr(obs, "candidate_witness_ids", ()) or ())
        primary_layer = getattr(obs, "primary_layer", None)

        if node_id:
            try:
                anc = graph.ancestors(node_id)
            except KeyError:
                anc = ()
            if anc:
                ancestor_count += len(anc)

        cross_layer = (
            bool(affected_layers)
            and primary_layer is not None
            and set(affected_layers) - {primary_layer}
        )
        if cross_layer:
            ancestor_count = max(ancestor_count, len(affected_layers))
            rationale.append(f"obs {getattr(obs, 'observation_id', '?')} cross-layer")

        if len(invalid_nodes) > 1 or cross_layer:
            resolver_count = max(resolver_count, len(invalid_nodes), len(affected_layers))
            rationale.append(f"obs {getattr(obs, 'observation_id', '?')} multi-violation")

        if len(witnesses) > 1:
            propagation_count = max(propagation_count, len(witnesses) - 1)
            rationale.append(f"obs {getattr(obs, 'observation_id', '?')} witness propagation")

    return ActivationOpportunity(
        ancestor_required=ancestor_count > 0,
        resolver_required=resolver_count > 0,
        propagation_required=propagation_count > 0,
        ancestor_required_count=ancestor_count,
        resolver_required_count=resolver_count,
        propagation_required_count=propagation_count,
        rationale=tuple(sorted(set(rationale))),
    )


__all__ = [
    "ActivationOpportunity",
    "activation_opportunity_from_graph",
    "activation_opportunity_from_truth",
]
