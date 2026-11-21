"""Pre-outcome mechanism and audit-budget policies for G5.6/G6.

Every policy evaluation starts from the same immutable canonical evidence
graph.  Ablations change the evaluator path; they are not post-processing of
the FULL result.

CLAIM_WEIGHTED uses a deterministic marginal claim-closure utility ordering
(G6.4-T1): a full ordering is built by repeatedly picking the eligible slot
with the highest lexicographic structural priority over the runtime claim
dependency structure; budget selection is always a prefix of that ordering.
The policy is outcome-blind: it reads only the runtime evidence graph, the
claim dependency structure and generic configuration.

``budget_policy_order`` is the single runtime ordering source
(G6.4-T1-R1): FULL, UNIFORM and CLAIM_WEIGHTED all share the same outer
budget semantics (limit = ceil(ratio * N), selected = W[:limit]) and
differ only in how W is constructed.  ``scheduled_allocation`` remains as
the historical G6.1 T6 compatibility path and is not used by the future
runtime evaluation.
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, Literal

from dmcasrc.application.activation import ActivationOpportunity, activation_opportunity_from_graph
from dmcasrc.application.audit import audit_scientific
from dmcasrc.application.causal_resolution import (
    EDGE_NOT_APPLICABLE,
    FindingKind,
    PrimaryResolution,
    classify_finding,
    resolve_minimal_causal_roots,
)
from dmcasrc.application.closure import (
    WitnessClosure,
    ancestor_closure_expand,
    witness_closure,
)
from dmcasrc.domain.contracts import (
    CanonicalObservation,
    ScientificAuditFinding,
    ScientificAuditReport,
    ScientificClaim,
)
from dmcasrc.domain.graph import PipelineGraph

BudgetStrategy = Literal["FULL", "UNIFORM", "CLAIM_WEIGHTED"]

_IDENTITY_OBLIGATIONS = frozenset(
    {
        "CANDIDATE_IDENTITY",
        "CALIBRATION_IDENTITY",
        "THRESHOLD_IDENTITY",
        "MODEL_IDENTITY",
        "EXECUTION_IDENTITY",
        "ARTIFACT_IDENTITY",
    }
)


def _canonical_sha256(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class MechanismPolicy:
    """Single-mechanism policy; ``full`` is the production default."""

    policy_id: str = "A0_FULL"
    ancestor_closure: bool = True
    causal_resolver: bool = True
    identity_closure: bool = True
    claim_dependency_gate: bool = True
    witness_propagation: bool = True

    def __post_init__(self) -> None:
        if not self.policy_id:
            raise ValueError("policy_id must be non-empty")
        expected = {
            "A0_FULL": (True, True, True, True, True),
            "A1_NO_ANCESTOR_CLOSURE": (False, True, True, True, True),
            "A2_NO_CAUSAL_RESOLVER": (True, False, True, True, True),
            "A3_NO_IDENTITY_CLOSURE": (True, True, False, True, True),
            "A4_NO_CLAIM_DEPENDENCY_GATE": (True, True, True, False, True),
            "A5_NO_WITNESS_PROPAGATION": (True, True, True, True, False),
        }
        actual = (
            self.ancestor_closure,
            self.causal_resolver,
            self.identity_closure,
            self.claim_dependency_gate,
            self.witness_propagation,
        )
        if self.policy_id in expected and actual != expected[self.policy_id]:
            raise ValueError(f"{self.policy_id} does not match its frozen mechanism definition")

    @classmethod
    def variants(cls) -> tuple[MechanismPolicy, ...]:
        return (
            cls(),
            cls("A1_NO_ANCESTOR_CLOSURE", ancestor_closure=False),
            cls("A2_NO_CAUSAL_RESOLVER", causal_resolver=False),
            cls("A3_NO_IDENTITY_CLOSURE", identity_closure=False),
            cls("A4_NO_CLAIM_DEPENDENCY_GATE", claim_dependency_gate=False),
            cls("A5_NO_WITNESS_PROPAGATION", witness_propagation=False),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy_id": self.policy_id,
            "ancestor_closure": self.ancestor_closure,
            "causal_resolver": self.causal_resolver,
            "identity_closure": self.identity_closure,
            "claim_dependency_gate": self.claim_dependency_gate,
            "witness_propagation": self.witness_propagation,
        }

    @property
    def config_sha256(self) -> str:
        return _canonical_sha256(self.to_dict())


@dataclass(frozen=True)
class AuditBudgetPolicy:
    policy_id: str = "FULL"
    ratio: float = 1.0
    allocation_strategy: BudgetStrategy = "FULL"
    seed: int = 0

    def __post_init__(self) -> None:
        if self.ratio not in {0.25, 0.5, 0.75, 1.0}:
            raise ValueError("ratio must be one of 0.25, 0.5, 0.75, 1.0")
        if self.allocation_strategy not in {"FULL", "UNIFORM", "CLAIM_WEIGHTED"}:
            raise ValueError("unknown allocation strategy")
        if self.allocation_strategy == "FULL" and self.ratio != 1.0:
            raise ValueError("FULL allocation requires ratio=1.0")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy_id": self.policy_id,
            "ratio": self.ratio,
            "allocation_strategy": self.allocation_strategy,
            "seed": self.seed,
        }

    @property
    def config_sha256(self) -> str:
        return _canonical_sha256(self.to_dict())


@dataclass(frozen=True)
class CanonicalEvidenceGraph:
    """Raw, pre-audit facts shared by every G6 evaluator."""

    graph: PipelineGraph
    observations: tuple[CanonicalObservation, ...]
    claims: tuple[ScientificClaim, ...]
    schema_version: str = "dmcasrc.canonical-evidence-graph.initial"

    def __post_init__(self) -> None:
        coordinates = [
            (item.candidate_id, item.node_id, item.obligation_id)
            for item in self.observations
        ]
        if len(coordinates) != len(set(coordinates)):
            raise ValueError("canonical evidence graph has duplicate evaluation slots")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "graph": self.graph.to_dict(),
            "observations": [
                item.to_dict() for item in sorted(self.observations, key=_coordinate)
            ],
            "claims": [
                {
                    "claim_id": item.claim_id,
                    "required_obligation_ids": list(item.required_obligation_ids),
                    "candidate_ids": list(item.candidate_ids),
                    "node_ids": None if item.node_ids is None else list(item.node_ids),
                }
                for item in sorted(self.claims, key=lambda value: value.claim_id)
            ],
        }

    @property
    def sha256(self) -> str:
        return _canonical_sha256(self.to_dict())


@dataclass(frozen=True)
class BudgetPriorityComponents:
    """Structural marginal-utility components for one selected budget slot.

    Every field is a runtime-visible structural fact (claim dependency
    scope, candidate/node/obligation identity).  No outcome, scenario,
    fault, truth or historical field can ever appear here: the scheduler
    input has no such metadata.
    """

    rank: int
    slot_id: str
    candidate_id: str
    node_id: str
    obligation_id: str
    newly_decidable_claim_count: int
    unresolved_dependency_reduction: int
    remaining_dependency_pressure: float
    dependent_claim_count: int
    identity_dependency_count: int
    evidence_dependency_count: int
    structural_cost: float
    effective_priority: tuple[Any, ...]
    canonical_rank: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "slot_id": self.slot_id,
            "candidate_id": self.candidate_id,
            "node_id": self.node_id,
            "obligation_id": self.obligation_id,
            "newly_decidable_claim_count": self.newly_decidable_claim_count,
            "unresolved_dependency_reduction": self.unresolved_dependency_reduction,
            "remaining_dependency_pressure": round(self.remaining_dependency_pressure, 12),
            "dependent_claim_count": self.dependent_claim_count,
            "identity_dependency_count": self.identity_dependency_count,
            "evidence_dependency_count": self.evidence_dependency_count,
            "structural_cost": self.structural_cost,
            "effective_priority": [float(v) if isinstance(v, float) else v for v in self.effective_priority],
            "canonical_rank": self.canonical_rank,
        }


@dataclass(frozen=True)
class BudgetAllocationTrace:
    """Deterministic, structural budget-allocation trace (G6.4-T1).

    Records why the scheduler picked each slot on the runtime structure.
    The trace never records whether a slot later detected anything: the
    allocation layer and the outcome layer stay separated.
    """

    policy_name: str
    eligible_slot_count: int
    budget_ratio: float
    budget_limit: int
    canonical_order: tuple[str, ...]
    effective_ordering: tuple[str, ...]
    effective_ordering_digest: str
    selected_slots: tuple[str, ...]
    selected_slots_digest: str
    ordering_differs_from_canonical: bool
    first_divergence_index: int | None
    priority_components: tuple[BudgetPriorityComponents, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy_name": self.policy_name,
            "eligible_slot_count": self.eligible_slot_count,
            "budget_ratio": self.budget_ratio,
            "budget_limit": self.budget_limit,
            "canonical_order": list(self.canonical_order),
            "effective_ordering": list(self.effective_ordering),
            "effective_ordering_digest": self.effective_ordering_digest,
            "selected_slots": list(self.selected_slots),
            "selected_slots_digest": self.selected_slots_digest,
            "ordering_differs_from_canonical": self.ordering_differs_from_canonical,
            "first_divergence_index": self.first_divergence_index,
            "priority_components": [item.to_dict() for item in self.priority_components],
        }


@dataclass(frozen=True)
class AuditResourceCounters:
    eligible_obligation_slots: int
    obligation_checks: int
    budget_exhausted_slots: int
    disabled_mechanism_slots: int
    ancestor_edge_traversals: int
    evidence_validations: int
    witness_expansions: int
    claim_evaluations: int
    audit_seconds: float
    setup_seconds: float = 0.0
    pipeline_seconds: float = 0.0
    total_wallclock_seconds: float = 0.0
    raw_evidence_bytes: int | str = "NOT_RECORDED"
    normalized_bytes: int | str = "NOT_RECORDED"
    ledger_bytes: int | str = "NOT_RECORDED"
    artifact_bytes: int | str = "NOT_RECORDED"
    peak_memory_bytes: int | str = "NOT_RECORDED"

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


@dataclass(frozen=True)
class MechanismEvaluation:
    mechanism_policy: MechanismPolicy
    budget_policy: AuditBudgetPolicy
    evidence_graph_sha256: str
    report: ScientificAuditReport
    primary_resolution: PrimaryResolution
    counters: AuditResourceCounters
    evaluated_coordinates: tuple[str, ...]
    budget_exhausted_coordinates: tuple[str, ...]
    disabled_coordinates: tuple[str, ...]
    claim_decision_status: str
    activation_opportunity: ActivationOpportunity | None = None
    selected_ordering: tuple[str, ...] = ()
    witness_closures: tuple[WitnessClosure, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "dmcasrc.mechanism-evaluation.initial",
            "mechanism_policy": self.mechanism_policy.to_dict(),
            "mechanism_policy_sha256": self.mechanism_policy.config_sha256,
            "budget_policy": self.budget_policy.to_dict(),
            "budget_policy_sha256": self.budget_policy.config_sha256,
            "evidence_graph_sha256": self.evidence_graph_sha256,
            "report": self.report.to_dict(),
            "primary_resolution": self.primary_resolution.to_dict(),
            "counters": self.counters.to_dict(),
            "evaluated_coordinates": list(self.evaluated_coordinates),
            "budget_exhausted_coordinates": list(self.budget_exhausted_coordinates),
            "disabled_coordinates": list(self.disabled_coordinates),
            "claim_decision_status": self.claim_decision_status,
            "activation_opportunity": (
                None
                if self.activation_opportunity is None
                else self.activation_opportunity.to_dict()
            ),
            "selected_ordering": list(self.selected_ordering),
            "witness_closures": [item.to_dict() for item in self.witness_closures],
        }


def _coordinate(item: CanonicalObservation) -> tuple[str, str, str]:
    return item.candidate_id, item.node_id, item.obligation_id


def _coordinate_text(item: CanonicalObservation) -> str:
    return "|".join(_coordinate(item))


def _stable_hash(seed: int, item: CanonicalObservation) -> str:
    value = f"{seed}|{_coordinate_text(item)}".encode()
    return hashlib.sha256(value).hexdigest()


# ── G6.4-T1: claim dependency structure for structural scheduling ──


def _claim_dependency_units(claim: ScientificClaim) -> frozenset[tuple[str, ...]]:
    """The dependency units a claim requires before it is structurally decidable.

    Mirrors the runtime claim gate semantics (``evaluate_scientific_claims``):
    with ``node_ids is None`` a (candidate, obligation) pair is the unit
    (any node covering the pair counts); with ``node_ids`` set, units are
    (candidate, node, obligation) triples.
    """
    if claim.node_ids is None:
        return frozenset(
            (candidate_id, obligation_id)
            for candidate_id in claim.candidate_ids
            for obligation_id in claim.required_obligation_ids
        )
    return frozenset(
        (candidate_id, node_id, obligation_id)
        for candidate_id in claim.candidate_ids
        for node_id in claim.node_ids
        for obligation_id in claim.required_obligation_ids
    )


def _slot_covers_unit(slot: CanonicalObservation, unit: tuple[str, ...]) -> bool:
    if len(unit) == 2:
        return unit == (slot.candidate_id, slot.obligation_id)
    return unit == (slot.candidate_id, slot.node_id, slot.obligation_id)


def _slot_claim_coverage(
    evidence: CanonicalEvidenceGraph,
) -> tuple[
    dict[str, dict[int, frozenset[tuple[str, ...]]]],
    tuple[frozenset[tuple[str, ...]], ...],
]:
    """Index: slot coordinate -> {claim index -> units of that claim covered}.

    Claims are indexed in claim_id order; the index is purely structural.
    """
    claims = sorted(evidence.claims, key=lambda item: item.claim_id)
    claim_units = tuple(_claim_dependency_units(item) for item in claims)
    coverage: dict[str, dict[int, frozenset[tuple[str, ...]]]] = {}
    for slot in evidence.observations:
        entry: dict[int, frozenset[tuple[str, ...]]] = {}
        for claim_index, units in enumerate(claim_units):
            covered = frozenset(unit for unit in units if _slot_covers_unit(slot, unit))
            if covered:
                entry[claim_index] = covered
        coverage[_coordinate_text(slot)] = entry
    return coverage, claim_units


def _claim_weighted_ordering(
    evidence: CanonicalEvidenceGraph,
) -> tuple[tuple[CanonicalObservation, ...], tuple[BudgetPriorityComponents, ...]]:
    """Deterministic marginal claim-closure utility ordering (G6.4-T1).

    Builds a complete ordering W by repeatedly selecting the not-yet-selected
    eligible slot with the highest lexicographic structural priority::

        priority(s | S) = (
            newly_decidable_claim_count(s | S),      # higher better
            unresolved_dependency_reduction(s | S),  # higher better
            remaining_dependency_pressure(s | S),    # higher better
            dependent_claim_count(s),                # higher better
            identity_dependency_count(s),            # higher better
            evidence_dependency_count(s),            # higher better
        )

    with the final tie-break falling back to canonical coordinate rank
    (ascending).  All inputs are runtime-visible structural facts; no
    outcome/scenario/fault/historical data is read anywhere on this path.
    """
    observations = tuple(sorted(evidence.observations, key=_coordinate))
    if not observations:
        return (), ()
    if not evidence.claims:
        # No claim structural utility exists: degrades to canonical order.
        return observations, ()

    canonical_rank = {_coordinate_text(item): index for index, item in enumerate(observations)}
    by_key = {_coordinate_text(item): item for item in observations}
    coverage, _ = _slot_claim_coverage(evidence)

    claim_units = [
        _claim_dependency_units(item)
        for item in sorted(evidence.claims, key=lambda claim: claim.claim_id)
    ]
    remaining_units: list[set[tuple[str, ...]]] = [set(units) for units in claim_units]

    remaining_keys: list[str] = list(canonical_rank)
    selected: list[CanonicalObservation] = []
    components: list[BudgetPriorityComponents] = []

    while remaining_keys:
        best_key: str | None = None
        best_priority: tuple[Any, ...] | None = None
        best_components: dict[str, Any] | None = None
        for key in remaining_keys:
            slot = by_key[key]
            newly = 0
            reduction = 0
            pressure = 0.0
            for claim_index, covered in coverage[key].items():
                still = remaining_units[claim_index]
                if not still:
                    continue
                newly_covered = covered & still
                if not newly_covered:
                    continue
                if newly_covered == still:
                    newly += 1
                reduction += 1
                pressure += 1.0 / len(still)
            dependent = len(coverage[key])
            identity = dependent if slot.obligation_id in _IDENTITY_OBLIGATIONS else 0
            evidence_count = dependent - identity
            priority = (
                newly,
                reduction,
                pressure,
                dependent,
                identity,
                evidence_count,
                -canonical_rank[key],
            )
            if best_priority is None or priority > best_priority:
                best_priority = priority
                best_key = key
                best_components = {
                    "newly": newly,
                    "reduction": reduction,
                    "pressure": pressure,
                    "dependent": dependent,
                    "identity": identity,
                    "evidence": evidence_count,
                }
        assert best_key is not None and best_components is not None
        slot = by_key[best_key]
        selected.append(slot)
        for claim_index, covered in coverage[best_key].items():
            remaining_units[claim_index] -= covered
        components.append(
            BudgetPriorityComponents(
                rank=len(selected) - 1,
                slot_id=_coordinate_text(slot),
                candidate_id=slot.candidate_id,
                node_id=slot.node_id,
                obligation_id=slot.obligation_id,
                newly_decidable_claim_count=best_components["newly"],
                unresolved_dependency_reduction=best_components["reduction"],
                remaining_dependency_pressure=best_components["pressure"],
                dependent_claim_count=best_components["dependent"],
                identity_dependency_count=best_components["identity"],
                evidence_dependency_count=best_components["evidence"],
                structural_cost=1.0,
                effective_priority=tuple(best_components[key] for key in ("newly", "reduction", "pressure", "dependent", "identity", "evidence")),
                canonical_rank=canonical_rank[best_key],
            )
        )
        remaining_keys.remove(best_key)

    return tuple(selected), tuple(components)


def claim_weighted_order(evidence: CanonicalEvidenceGraph) -> tuple[CanonicalObservation, ...]:
    """Public full ordering for the CLAIM_WEIGHTED budget strategy."""
    return _claim_weighted_ordering(evidence)[0]


def budget_allocation_trace(
    evidence: CanonicalEvidenceGraph,
    policy: AuditBudgetPolicy,
) -> BudgetAllocationTrace:
    """Deterministic structural allocation trace for one budget policy.

    The effective ordering comes from the single runtime ordering source
    ``budget_policy_order`` -- the same source ``evaluate_evidence_graph``
    uses -- so the trace can never diverge from the runtime selection.
    Digests are SHA-256 over the canonical JSON serialization of the
    coordinate list, so they are stable and platform independent.
    """
    observations = tuple(sorted(evidence.observations, key=_coordinate))
    canonical_order = tuple(_coordinate_text(item) for item in observations)
    total = len(observations)
    budget_limit = min(math.ceil(policy.ratio * total), total) if total else 0

    if policy.allocation_strategy == "CLAIM_WEIGHTED":
        ordered, components = _claim_weighted_ordering(evidence)
    else:
        ordered = budget_policy_order(evidence, policy)
        components = ()
    effective = tuple(_coordinate_text(item) for item in ordered)

    selected = effective[:budget_limit]
    first_divergence = None
    # B905: zip intentionally truncates to the shorter ordering; the trace must
    # report the first divergence without raising when mechanism-selected
    # coordinates differ in length from the canonical observation order.
    for index, (actual, canonical) in enumerate(zip(effective, canonical_order)):  # noqa: B905
        if actual != canonical:
            first_divergence = index
            break
    return BudgetAllocationTrace(
        policy_name=policy.policy_id,
        eligible_slot_count=total,
        budget_ratio=policy.ratio,
        budget_limit=budget_limit,
        canonical_order=canonical_order,
        effective_ordering=effective,
        effective_ordering_digest=_canonical_sha256(list(effective)),
        selected_slots=selected,
        selected_slots_digest=_canonical_sha256(list(selected)),
        ordering_differs_from_canonical=effective != canonical_order,
        first_divergence_index=first_divergence,
        priority_components=components,
    )


# T6: multi-stage audit scheduler stage priorities (highest first).
STAGE_PRIORITY: tuple[str, ...] = ("claim", "obligation", "ancestor", "witness")


def stage_of(item: CanonicalObservation, graph: PipelineGraph) -> str:
    """Classify an evaluation slot into a T6 stage."""
    obligation = item.obligation_id
    if obligation in _IDENTITY_OBLIGATIONS:
        return "ancestor"
    if obligation in {
        "CLAIM_DEPENDENCY_INTEGRITY",
        "OUTCOME_FEEDBACK_ISOLATION",
        "SELECTION_EXECUTION_CONSISTENCY",
        "PREDICTION_ORIGIN",
        "ASSESSMENT_SELECTION_ISOLATION",
    }:
        return "claim"
    if len(item.candidate_witness_ids) > 1:
        return "witness"
    return "obligation"


def _legacy_stage_weighted_order(
    evidence: CanonicalEvidenceGraph,
    policy: AuditBudgetPolicy,
    *,
    stage_weights: Mapping[str, float],
) -> tuple[CanonicalObservation, ...]:
    """G6.1 T6 multi-stage budget allocation (legacy, frozen semantics).

    Divides the budget across stages (claim > obligation > ancestor >
    witness) with the frozen stage weights.  This is the historical
    G6.1/G6.2-era scheduler behaviour for every allocation strategy at
    ratio < 1.0: the allocation strategy is NOT consulted here.

    Kept verbatim as a historical compatibility path.  Future G6.4
    budget-policy evaluation must go through ``budget_policy_order`` so
    that UNIFORM and CLAIM_WEIGHTED share one outer budget semantics.
    """
    observations = tuple(sorted(evidence.observations, key=_coordinate))
    total = len(observations)
    budget = min(math.ceil(policy.ratio * total), total)
    per_stage: dict[str, list[CanonicalObservation]] = {stage: [] for stage in STAGE_PRIORITY}
    for item in observations:
        per_stage[stage_of(item, evidence.graph)].append(item)
    for stage in STAGE_PRIORITY:
        per_stage[stage].sort(key=lambda item: _stable_hash(policy.seed, item))
    stage_budgets: dict[str, int] = {}
    remaining = budget
    for index, stage in enumerate(STAGE_PRIORITY):
        if index == len(STAGE_PRIORITY) - 1:
            stage_budgets[stage] = remaining
        else:
            share = math.ceil(budget * stage_weights[stage])
            share = min(share, len(per_stage[stage]), remaining)
            stage_budgets[stage] = share
            remaining -= share
    selected: list[CanonicalObservation] = []
    for stage in STAGE_PRIORITY:
        selected.extend(per_stage[stage][: stage_budgets[stage]])
    selected_ids = {_coordinate_text(item) for item in selected}
    remainder = [
        item for item in observations if _coordinate_text(item) not in selected_ids
    ]
    # deterministic tie-break on full coordinate within each block
    return tuple(
        sorted(selected, key=lambda item: _coordinate_text(item))
        + sorted(remainder, key=lambda item: _coordinate_text(item))
    )


def scheduled_allocation(
    evidence: CanonicalEvidenceGraph,
    policy: AuditBudgetPolicy,
    *,
    stage_weights: Mapping[str, float] | None = None,
) -> tuple[CanonicalObservation, ...]:
    """Historical compatibility path (G6.1 T6 multi-stage scheduler).

    Frozen behaviour: with ratio == 1.0 the order is the FULL deterministic
    order from ``allocation_order``; with ratio < 1.0 the budget is divided
    across stages (claim > obligation > ancestor > witness) regardless of
    the allocation strategy.

    Future G6.4 budget-policy evaluation must NOT use this path: it would
    give UNIFORM stage quotas that CLAIM_WEIGHTED does not have.  Use
    ``budget_policy_order`` instead, which is the single runtime ordering
    source for ``evaluate_evidence_graph`` and ``budget_allocation_trace``.

    Returns the FULL observation set re-ordered so that the budget-selected
    slots come first and the exhausted ones follow, preserving the
    ``ordered[:limit]`` / ``ordered[limit:]`` contract of the evaluator.
    """
    if policy.ratio >= 1.0:
        return allocation_order(evidence, policy)
    weights = stage_weights or {
        "claim": 0.40,
        "obligation": 0.30,
        "ancestor": 0.20,
        "witness": 0.10,
    }
    return _legacy_stage_weighted_order(evidence, policy, stage_weights=weights)


def budget_policy_order(
    evidence: CanonicalEvidenceGraph,
    policy: AuditBudgetPolicy,
) -> tuple[CanonicalObservation, ...]:
    """Single runtime ordering source for budget policies (G6.4-T1-R1).

    Returns the complete deterministic ordering W for a budget policy:

    - FULL:            canonical coordinate ordering
    - UNIFORM:         deterministic stable-hash ordering, independent of
                       claim structure / stage / truth / outcome
    - CLAIM_WEIGHTED:  deterministic marginal claim-closure ordering

    Budget selection is always the same outer semantics applied to W::

        limit    = ceil(ratio * len(W))
        selected = W[:limit]
        exhausted = W[limit:]

    UNIFORM and CLAIM_WEIGHTED therefore differ ONLY in how W is
    constructed; neither strategy carries stage quotas here.  Both
    ``evaluate_evidence_graph`` and ``budget_allocation_trace`` must call
    this same source (never re-implement scheduler decisions).
    """
    observations = tuple(sorted(evidence.observations, key=_coordinate))
    if policy.allocation_strategy == "FULL":
        return observations
    if policy.allocation_strategy == "UNIFORM":
        return tuple(sorted(observations, key=lambda item: _stable_hash(policy.seed, item)))
    # CLAIM_WEIGHTED: deterministic marginal claim-closure utility ordering.
    return claim_weighted_order(evidence)


def allocation_order(
    evidence: CanonicalEvidenceGraph,
    policy: AuditBudgetPolicy,
) -> tuple[CanonicalObservation, ...]:
    """Legacy name for ``budget_policy_order`` (kept for frozen tests)."""
    return budget_policy_order(evidence, policy)


def _topological_resolution(
    findings: Sequence[ScientificAuditFinding], graph: PipelineGraph
) -> PrimaryResolution:
    violations = [item for item in findings if item.status == "VIOLATION"]
    if not violations:
        return PrimaryResolution(None, None, None, None, None, None, (), (), (), ())
    node_rank = {node: index for index, node in enumerate(graph.topological_order())}

    def node_id(item: ScientificAuditFinding) -> str:
        parts = item.finding_id.split(":")
        return parts[1] if len(parts) >= 3 else ""

    primary = sorted(
        violations,
        key=lambda item: (
            node_rank.get(node_id(item), len(node_rank)),
            item.obligation_id,
            item.finding_id,
        ),
    )[0]
    violated_nodes = {node_id(item) for item in violations if node_id(item)}
    classifications = tuple(classify_finding(item, graph, violated_nodes) for item in violations)
    primary_classification = next(
        item for item in classifications if item.finding_id == primary.finding_id
    )
    edge_not_applicable = None
    if primary.first_invalid_edge is None and primary_classification.kind in {
        FindingKind.DIRECT_IDENTITY_VIOLATION,
        FindingKind.DERIVED_CONTRACT_FAILURE,
    }:
        edge_not_applicable = EDGE_NOT_APPLICABLE
    return PrimaryResolution(
        primary.finding_id,
        primary_classification.kind,
        primary.primary_layer,
        primary.obligation_id,
        primary.first_invalid_edge,
        edge_not_applicable,
        tuple(item.finding_id for item in violations),
        tuple(item.first_invalid_edge for item in violations if item.first_invalid_edge is not None),
        tuple(sorted({item.primary_layer for item in violations})),
        tuple(sorted({item.obligation_id for item in violations})),
        classifications,
    )


def evaluate_evidence_graph(
    evidence: CanonicalEvidenceGraph,
    *,
    mechanism_policy: MechanismPolicy | None = None,
    budget_policy: AuditBudgetPolicy | None = None,
) -> MechanismEvaluation:
    """Evaluate one variant directly from the immutable raw evidence graph.

    The ordering comes from the single runtime ordering source
    ``budget_policy_order``; outer budget semantics are identical for every
    strategy: ``limit = ceil(ratio * N)``, ``selected = W[:limit]``,
    ``exhausted = W[limit:]`` (G6.4-T1-R1).
    """
    mechanism = mechanism_policy or MechanismPolicy()
    budget = budget_policy or AuditBudgetPolicy()
    ordered = budget_policy_order(evidence, budget)
    limit = math.ceil(budget.ratio * len(ordered))
    selected = ordered[:limit]
    exhausted = ordered[limit:]
    disabled: list[CanonicalObservation] = []
    prepared: list[CanonicalObservation] = []
    witness_closures: list[WitnessClosure] = []
    ancestor_traversals = 0
    witness_expansions = 0
    for item in selected:
        if not mechanism.identity_closure and item.obligation_id in _IDENTITY_OBLIGATIONS:
            disabled.append(item)
            continue
        current = item
        if not mechanism.ancestor_closure:
            current = replace(
                current,
                affected_layers=(current.primary_layer,),
                invalid_node_ids=(current.node_id,),
                declared_first_invalid_edge=None,
            )
        else:
            ancestor_traversals += len(evidence.graph.ancestors(current.node_id))
            current = ancestor_closure_expand(current, evidence.graph)
        closure = witness_closure(
            current,
            evidence.graph,
            propagation_enabled=mechanism.witness_propagation,
        )
        # ``candidate_witness_ids`` remains the frozen finding schema.  Its
        # evaluator representation is now the explicit complete closure; the
        # additive ``witness_closures`` trace records each tier and path.
        current = replace(current, candidate_witness_ids=closure.complete_witness_ids)
        witness_closures.append(closure)
        witness_expansions += len(closure.propagated_witness_ids)
        prepared.append(current)

    claims = evidence.claims if mechanism.claim_dependency_gate else ()
    started = time.perf_counter()
    report = audit_scientific(evidence.graph, tuple(prepared), tuple(claims))
    audit_elapsed = time.perf_counter() - started
    started_resolve = time.perf_counter()
    # Keep the A2 fallback as an independent, legacy topological path.  FULL
    # receives its result only as a diagnostic baseline; its selected primary
    # is determined solely by the violation causal graph and minimal root set.
    default_resolution = _topological_resolution(report.findings, evidence.graph)
    resolution = (
        resolve_minimal_causal_roots(
            report.findings,
            evidence.graph,
            default_primary_finding_id=default_resolution.primary_finding_id,
        )
        if mechanism.causal_resolver
        else default_resolution
    )
    resolve_elapsed = time.perf_counter() - started_resolve
    counters = AuditResourceCounters(
        eligible_obligation_slots=len(ordered),
        obligation_checks=len(prepared),
        budget_exhausted_slots=len(exhausted),
        disabled_mechanism_slots=len(disabled),
        ancestor_edge_traversals=ancestor_traversals,
        evidence_validations=sum(len(item.evidence_refs) for item in prepared),
        witness_expansions=witness_expansions,
        claim_evaluations=len(claims),
        audit_seconds=round(audit_elapsed, 9),
        setup_seconds=round(resolve_elapsed, 9),
        pipeline_seconds=round(audit_elapsed + resolve_elapsed, 9),
        total_wallclock_seconds=round(audit_elapsed + resolve_elapsed, 9),
    )
    activation = activation_opportunity_from_graph(evidence.graph, evidence.observations)
    return MechanismEvaluation(
        mechanism_policy=mechanism,
        budget_policy=budget,
        evidence_graph_sha256=evidence.sha256,
        report=report,
        primary_resolution=resolution,
        counters=counters,
        evaluated_coordinates=tuple(sorted(_coordinate_text(item) for item in prepared)),
        budget_exhausted_coordinates=tuple(sorted(_coordinate_text(item) for item in exhausted)),
        disabled_coordinates=tuple(sorted(_coordinate_text(item) for item in disabled)),
        claim_decision_status=(
            "PRODUCED" if mechanism.claim_dependency_gate else "NOT_PRODUCED_ABLATION"
        ),
        activation_opportunity=activation,
        selected_ordering=tuple(_coordinate_text(item) for item in selected),
        witness_closures=tuple(witness_closures),
    )


__all__ = [
    "AuditBudgetPolicy",
    "AuditResourceCounters",
    "BudgetAllocationTrace",
    "BudgetPriorityComponents",
    "CanonicalEvidenceGraph",
    "MechanismEvaluation",
    "MechanismPolicy",
    "STAGE_PRIORITY",
    "allocation_order",
    "budget_allocation_trace",
    "budget_policy_order",
    "claim_weighted_order",
    "evaluate_evidence_graph",
    "scheduled_allocation",
    "stage_of",
]
