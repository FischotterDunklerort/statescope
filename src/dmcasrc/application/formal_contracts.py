"""Executable finite-trace contracts for the AutoFact supplementary path.

This is intentionally a compact runtime-monitoring layer, not a theorem prover
or a whole-program Python verifier.  It makes a representative subset of the
canonical evidence responsibilities explicit, preserves UNKNOWN semantics, and
is designed to feed facts back into the existing canonical audit and claim
gate rather than replace either component.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Any

from dmcasrc.autofact.events import EvidenceSource, EventType, RuntimeEvent, trace_sha256
from dmcasrc.domain.contracts import CanonicalContractSpecification


class ContractClass(StrEnum):
    STATE = "STATE_PREDICATE"
    TEMPORAL = "TEMPORAL_TRACE"
    RELATIONAL = "RELATIONAL_NONINTERFERENCE"


class FormalStatus(StrEnum):
    PASS = "PASS"
    VIOLATION = "VIOLATION"
    UNVERIFIABLE = "UNVERIFIABLE"
    NOT_ESTIMABLE = "NOT_ESTIMABLE"


class CandidateIdentityPolicy(StrEnum):
    """Candidate-binding policy used by the finite-trace P1 monitor.

    ``LEGACY_ID_ONLY`` is retained for replay compatibility with the original
    AutoFact supplementary receipts.  New confirmation protocols should use
    ``STRICT_SELECTION_LINEAGE``: declarations can fill a genuinely missing
    semantic role, but cannot override a runtime identity or provenance
    conflict.
    """

    LEGACY_ID_ONLY = "LEGACY_ID_ONLY"
    STRICT_SELECTION_LINEAGE = "STRICT_SELECTION_LINEAGE"


@dataclass(frozen=True)
class FormalContract:
    """A concise, machine-readable finite-trace contract specification."""

    contract_id: str
    obligation_id: str
    contract_class: ContractClass
    description: str
    required_event_types: tuple[EventType, ...]
    assumptions: tuple[str, ...]
    predicate: str
    violation_condition: str
    unknown_semantics: str
    node_id: str
    affected_claims: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in (
            "contract_id",
            "obligation_id",
            "description",
            "predicate",
            "violation_condition",
            "unknown_semantics",
            "node_id",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty string.")
        if not isinstance(self.contract_class, ContractClass):
            raise ValueError("contract_class must be a ContractClass.")
        if not self.required_event_types or any(
            not isinstance(item, EventType) for item in self.required_event_types
        ):
            raise ValueError("required_event_types must contain EventType values.")
        for name in ("assumptions", "affected_claims"):
            values = getattr(self, name)
            if not isinstance(values, tuple) or any(not isinstance(item, str) or not item for item in values):
                raise ValueError(f"{name} must be a tuple of non-empty strings.")
            if len(values) != len(set(values)):
                raise ValueError(f"{name} must not contain duplicates.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract_id": self.contract_id,
            "obligation_id": self.obligation_id,
            "contract_class": self.contract_class.value,
            "description": self.description,
            "required_event_types": [item.value for item in self.required_event_types],
            "assumptions": list(self.assumptions),
            "predicate": self.predicate,
            "violation_condition": self.violation_condition,
            "unknown_semantics": self.unknown_semantics,
            "node_id": self.node_id,
            "affected_claims": list(self.affected_claims),
        }


@dataclass(frozen=True)
class FormalContractResult:
    """One monitor outcome and its evidence/witness path."""

    contract_id: str
    obligation_id: str
    status: FormalStatus
    reason_codes: tuple[str, ...]
    witness_event_ids: tuple[str, ...]
    missing_required_evidence: tuple[str, ...]
    candidate_ids: tuple[str, ...]
    node_id: str
    invalid_node_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in ("contract_id", "obligation_id", "node_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty string.")
        if not isinstance(self.status, FormalStatus):
            raise ValueError("status must be a FormalStatus.")
        for name in (
            "reason_codes",
            "witness_event_ids",
            "missing_required_evidence",
            "candidate_ids",
            "invalid_node_ids",
        ):
            values = getattr(self, name)
            if not isinstance(values, tuple) or any(not isinstance(item, str) or not item for item in values):
                raise ValueError(f"{name} must be a tuple of non-empty strings.")
            if len(values) != len(set(values)):
                raise ValueError(f"{name} must not contain duplicates.")
        if not self.reason_codes:
            raise ValueError("reason_codes must not be empty.")
        if not self.candidate_ids:
            raise ValueError("candidate_ids must not be empty.")
        if self.status in {FormalStatus.UNVERIFIABLE, FormalStatus.NOT_ESTIMABLE} and not self.missing_required_evidence:
            raise ValueError("unverifiable/not-estimable results require missing evidence details.")
        if self.status in {FormalStatus.PASS, FormalStatus.VIOLATION} and self.missing_required_evidence:
            raise ValueError("only unknown outcomes may include missing_required_evidence.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract_id": self.contract_id,
            "obligation_id": self.obligation_id,
            "status": self.status.value,
            "reason_codes": list(self.reason_codes),
            "witness_event_ids": list(self.witness_event_ids),
            "missing_required_evidence": list(self.missing_required_evidence),
            "candidate_ids": list(self.candidate_ids),
            "node_id": self.node_id,
            "invalid_node_ids": list(self.invalid_node_ids),
        }


@dataclass(frozen=True)
class FormalTraceEvaluation:
    """Results for a fixed finite trace and, optionally, a paired trace."""

    trace_sha256: str
    event_count: int
    results: tuple[FormalContractResult, ...]
    comparison_trace_sha256: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.trace_sha256, str) or len(self.trace_sha256) != 64:
            raise ValueError("trace_sha256 must be a SHA-256 string.")
        if isinstance(self.event_count, bool) or not isinstance(self.event_count, int) or self.event_count < 0:
            raise ValueError("event_count must be a non-negative integer.")
        if not isinstance(self.results, tuple) or not self.results:
            raise ValueError("results must be a non-empty tuple.")
        if any(not isinstance(item, FormalContractResult) for item in self.results):
            raise ValueError("results must contain FormalContractResult values.")
        if len({item.contract_id for item in self.results}) != len(self.results):
            raise ValueError("each contract may appear at most once in an evaluation.")
        if self.comparison_trace_sha256 is not None and len(self.comparison_trace_sha256) != 64:
            raise ValueError("comparison_trace_sha256 must be a SHA-256 string or None.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_sha256": self.trace_sha256,
            "event_count": self.event_count,
            "comparison_trace_sha256": self.comparison_trace_sha256,
            "results": [item.to_dict() for item in self.results],
        }


_DEFAULT_CONTRACTS = (
    FormalContract(
        contract_id="FC-T1-TEMPORAL-AVAILABILITY",
        obligation_id="TIME_AVAILABILITY",
        contract_class=ContractClass.TEMPORAL,
        description="Decision inputs observed at a decision boundary must already be available.",
        required_event_types=(EventType.PREDICT, EventType.CANDIDATE_SELECT),
        assumptions=(
            "The producer supplies timezone-aware decision_ts and input_available_ts fields when time semantics are observable.",
        ),
        predicate="G(event in {PREDICT,CANDIDATE_SELECT} with timing metadata -> all(input_available_ts <= decision_ts))",
        violation_condition="An observed input availability time is later than its event decision boundary.",
        unknown_semantics="Missing or non-timezone-aware timing metadata yields UNVERIFIABLE, never PASS.",
        node_id="model.predict",
    ),
    FormalContract(
        contract_id="FC-T2-TRANSFORM-FIT-BOUNDARY",
        obligation_id="ASSESSMENT_FIT_ISOLATION",
        contract_class=ContractClass.TEMPORAL,
        description="Declared/derived top-level preprocessing fit membership must exclude assessment members.",
        required_event_types=(EventType.DATA_SPLIT, EventType.TRANSFORM_FIT),
        assumptions=(
            "A boundary-scoped transform event identifies the members used to fit that transform.",
            "A split event identifies assessment members in the same run scope.",
        ),
        predicate="G(boundary_transform_fit -> source_ids ∩ assessment_member_ids = empty)",
        violation_condition="A source member used by a boundary-scoped transform fit also belongs to assessment data.",
        unknown_semantics="Incomplete membership is UNVERIFIABLE; unrelated framework metadata is optional and non-blocking.",
        node_id="preprocessing.fit",
    ),
    FormalContract(
        contract_id="FC-P1-CANDIDATE-IDENTITY-CLOSURE",
        obligation_id="CANDIDATE_IDENTITY",
        contract_class=ContractClass.STATE,
        description="Each selected candidate must have a previously observed candidate-creation identity.",
        required_event_types=(EventType.CANDIDATE_CREATE, EventType.CANDIDATE_SELECT),
        assumptions=("Selection events identify the selected candidate by a stable candidate_id.",),
        predicate="G(CANDIDATE_SELECT(c) -> previously CANDIDATE_CREATE(c))",
        violation_condition="A selected candidate has no corresponding observed creation event.",
        unknown_semantics="No selection or no candidate-creation evidence is UNVERIFIABLE.",
        node_id="selector.selection",
    ),
    FormalContract(
        contract_id="FC-P2-ARTIFACT-PROVENANCE-CLOSURE",
        obligation_id="ARTIFACT_IDENTITY",
        contract_class=ContractClass.STATE,
        description="A released artifact-dependent claim must bind each required artifact to an observed identity.",
        required_event_types=(EventType.ARTIFACT_WRITE, EventType.CLAIM_RELEASE),
        assumptions=(
            "Claim release declares its required_artifact_ids and artifact_bindings when artifact dependence applies.",
        ),
        predicate="G(CLAIM_RELEASE(required_artifact_ids) -> every binding matches an observed ARTIFACT_WRITE identity)",
        violation_condition="A required artifact is absent or its release binding differs from the observed artifact identity.",
        unknown_semantics="No release or omitted artifact binding metadata is UNVERIFIABLE.",
        node_id="artifact.model",
    ),
    FormalContract(
        contract_id="FC-P3-EVIDENCE-CLOSURE",
        obligation_id="ANCESTOR_CLOSURE",
        contract_class=ContractClass.STATE,
        description="Every PASS result that requires runtime evidence must retain a resolvable event witness.",
        required_event_types=(EventType.DATA_LOAD, EventType.MODEL_FIT, EventType.PREDICT),
        assumptions=("Event IDs are unique within the finite trace.",),
        predicate="forall PASS contract result r: witness_event_ids(r) are nonempty and all belong to trace",
        violation_condition="A satisfied contract result lacks an event witness or references an unknown event.",
        unknown_semantics="When no prerequisite contract is evaluable, evidence closure is UNVERIFIABLE.",
        node_id="model.predict",
    ),
    FormalContract(
        contract_id="FC-G1-CLAIM-RELEASE-SAFETY",
        obligation_id="CLAIM_DEPENDENCY_INTEGRITY",
        contract_class=ContractClass.TEMPORAL,
        description="A claim release is safe only after all mandatory listed contracts are satisfied.",
        required_event_types=(EventType.CLAIM_RELEASE,),
        assumptions=(
            "A release event lists the contract IDs required by its claim policy.",
            "The default policy does not permit unresolved mandatory evidence.",
        ),
        predicate="G(CLAIM_RELEASE(required_contract_ids) -> all(required result status = PASS))",
        violation_condition="A release occurs while a listed prerequisite is VIOLATION, UNVERIFIABLE, NOT_ESTIMABLE, or absent.",
        unknown_semantics="No release or no required_contract_ids declaration is UNVERIFIABLE.",
        node_id="claim.performance",
    ),
    FormalContract(
        contract_id="FC-R1-SELECTION-NONINTERFERENCE",
        obligation_id="ASSESSMENT_SELECTION_ISOLATION",
        contract_class=ContractClass.RELATIONAL,
        description="Paired traces with equal permitted pre-assessment inputs must not select different candidates before assessment influence is allowed.",
        required_event_types=(EventType.CANDIDATE_SELECT,),
        assumptions=(
            "Each pre-assessment selection event exposes permitted_input_identity.",
            "The paired traces differ only in assessment information for the checked decision boundary.",
        ),
        predicate="trace_a.permitted_inputs = trace_b.permitted_inputs -> trace_a.selection = trace_b.selection",
        violation_condition="Equivalent permitted-input projections lead to different pre-assessment candidate selections.",
        unknown_semantics="Missing paired trace, selection metadata, or an unmet equivalence premise yields UNVERIFIABLE/NOT_ESTIMABLE.",
        node_id="selector.selection",
    ),
)


def default_formal_contracts() -> tuple[FormalContract, ...]:
    """Return the immutable initial finite-trace contract set."""

    return _DEFAULT_CONTRACTS


def learned_fit_contract() -> FormalContract:
    """Optional subsequent boundary for captured estimator and transform fit calls.

    The original T2 contract and default initial inventory remain available for exact
    replay. This requires positional lineage for every scoped observed fit; it
    does not prove completeness of all uninstrumented program operations.
    """
    return FormalContract(
        contract_id="FC-T3-LEARNED-FIT-BOUNDARY", obligation_id="ASSESSMENT_FIT_ISOLATION",
        contract_class=ContractClass.STATE,
        description="Captured learned operations must not fit assessment members.",
        required_event_types=(EventType.DATA_SPLIT, EventType.TRANSFORM_FIT, EventType.MODEL_FIT),
        assumptions=("Assessment role is explicitly declared.",
                     "Supported fit and native indexing hooks retain exact member identity."),
        predicate="forall captured scoped fits: source_ids intersect assessment = empty",
        violation_condition="An observed learned fit consumes an assessment member.",
        unknown_semantics="Missing fit membership or assessment role remains UNVERIFIABLE.",
        node_id="preprocessing.fit",
    )


def _events(events: Sequence[RuntimeEvent], event_type: EventType) -> tuple[RuntimeEvent, ...]:
    return tuple(item for item in events if item.event_type == event_type)


from functools import lru_cache

@lru_cache(maxsize=8)
def _ordered_unique_members(value):
    if len(value)!=len(set(value)):return None
    return tuple(sorted(value))


def _attribute_list(event: RuntimeEvent, key: str) -> tuple[str, ...] | None:
    value = event.attributes.get(key)
    if not isinstance(value, list) or any(not isinstance(item, str) or not item for item in value):
        return None
    # Key by immutable current contents, never object identity or a prior PASS.
    # Mutated policies/data therefore cannot reuse a stale normalized member set.
    return _ordered_unique_members(tuple(value))


def _attribute_strings(event: RuntimeEvent, key: str) -> tuple[str, ...] | None:
    value = event.attributes.get(key)
    if isinstance(value, str) and value:
        return (value,)
    return _attribute_list(event, key)


def _parse_time(value: str) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (AttributeError, ValueError):
        return None
    return parsed if parsed.tzinfo is not None else None


def _result(
    contract: FormalContract,
    status: FormalStatus,
    reason: str,
    candidates: tuple[str, ...],
    *,
    witnesses: Sequence[str] = (),
    missing: Sequence[str] = (),
    invalid_nodes: Sequence[str] = (),
) -> FormalContractResult:
    return FormalContractResult(
        contract_id=contract.contract_id,
        obligation_id=contract.obligation_id,
        status=status,
        reason_codes=(reason,),
        witness_event_ids=tuple(sorted(set(witnesses))),
        missing_required_evidence=tuple(sorted(set(missing))),
        candidate_ids=candidates,
        node_id=contract.node_id,
        invalid_node_ids=tuple(sorted(set(invalid_nodes))),
    )


def _temporal_availability(
    contract: FormalContract,
    events: Sequence[RuntimeEvent],
    candidates: tuple[str, ...],
) -> FormalContractResult:
    relevant = tuple(
        item
        for item in events
        if item.event_type in {EventType.PREDICT, EventType.CANDIDATE_SELECT}
    )
    if not relevant:
        return _result(
            contract,
            FormalStatus.UNVERIFIABLE,
            "MISSING_DECISION_EVENT",
            candidates,
            missing=("PREDICT_or_CANDIDATE_SELECT",),
        )
    # A framework hook can observe a prediction call without observing its
    # scientific decision boundary.  Such an unscoped operational event is
    # useful provenance but must not invalidate a separate, explicit boundary
    # annotation in Hybrid mode.  Conversely, if no event carries timing
    # semantics, the contract remains fail-closed and UNVERIFIABLE.
    scoped = tuple(
        item
        for item in relevant
        if item.attributes.get("contract_scope") == "temporal_availability"
        or "decision_ts" in item.attributes
        or "input_available_ts" in item.attributes
    )
    if not scoped:
        return _result(
            contract,
            FormalStatus.UNVERIFIABLE,
            "TEMPORAL_AVAILABILITY_EVIDENCE_MISSING",
            candidates,
            missing=("temporal_decision_boundary",),
        )
    missing: list[str] = []
    witnesses: list[str] = []
    violations: list[RuntimeEvent] = []
    for event in scoped:
        decision = _parse_time(event.attributes.get("decision_ts"))
        available_values = _attribute_strings(event, "input_available_ts")
        if decision is None:
            missing.append(f"{event.event_id}:decision_ts")
            continue
        if not available_values:
            missing.append(f"{event.event_id}:input_available_ts")
            continue
        available = [_parse_time(item) for item in available_values]
        if any(item is None for item in available):
            missing.append(f"{event.event_id}:timezone_aware_input_available_ts")
            continue
        witnesses.append(event.event_id)
        if any(item is not None and item > decision for item in available):
            violations.append(event)
    if missing:
        return _result(
            contract,
            FormalStatus.UNVERIFIABLE,
            "TEMPORAL_AVAILABILITY_EVIDENCE_MISSING",
            candidates,
            witnesses=witnesses,
            missing=missing,
        )
    if violations:
        invalid_nodes = [
            str(item.attributes.get("source_node_id", "data.assessment")) for item in violations
        ]
        return _result(
            contract,
            FormalStatus.VIOLATION,
            "INPUT_AVAILABLE_AFTER_DECISION",
            candidates,
            witnesses=[item.event_id for item in violations],
            invalid_nodes=invalid_nodes,
        )
    return _result(
        contract,
        FormalStatus.PASS,
        "TEMPORAL_AVAILABILITY_SATISFIED",
        candidates,
        witnesses=witnesses,
    )


def _transform_fit_boundary(
    contract: FormalContract,
    events: Sequence[RuntimeEvent],
    candidates: tuple[str, ...],
) -> FormalContractResult:
    assessment: set[str] = set()
    split_witnesses: list[str] = []
    for event in _events(events, EventType.DATA_SPLIT):
        role = event.attributes.get("split_role") or event.fold_role
        members = _attribute_list(event, "member_ids")
        if role == "assessment" and members is not None:
            assessment.update(members)
            split_witnesses.append(event.event_id)
    fits = [
        event
        for event in events
        if event.event_type in ({EventType.TRANSFORM_FIT, EventType.MODEL_FIT}
                               if contract.contract_id == "FC-T3-LEARNED-FIT-BOUNDARY"
                               else {EventType.TRANSFORM_FIT})
        if event.attributes.get("contract_scope") == "assessment_fit_boundary"
        or "source_ids" in event.attributes
    ]
    if not assessment:
        return _result(
            contract,
            FormalStatus.UNVERIFIABLE,
            "ASSESSMENT_MEMBERSHIP_MISSING",
            candidates,
            missing=("assessment_member_ids",),
        )
    if not fits:
        return _result(
            contract,
            FormalStatus.UNVERIFIABLE,
            "TRANSFORM_FIT_MEMBERSHIP_MISSING",
            candidates,
            witnesses=split_witnesses,
            missing=("boundary_scoped_transform_fit",),
        )
    missing: list[str] = []
    overlap: set[str] = set()
    witnesses = list(split_witnesses)
    for event in fits:
        source_ids = _attribute_list(event, "source_ids")
        if source_ids is None:
            missing.append(f"{event.event_id}:source_ids")
            continue
        witnesses.append(event.event_id)
        overlap.update(assessment.intersection(source_ids))
    if overlap:
        return _result(
            contract, FormalStatus.VIOLATION, "ASSESSMENT_MEMBER_USED_FOR_TRANSFORM_FIT",
            candidates, witnesses=witnesses, invalid_nodes=("data.assessment",),
        )
    if missing:
        return _result(
            contract,
            FormalStatus.UNVERIFIABLE,
            "TRANSFORM_FIT_MEMBERSHIP_MISSING",
            candidates,
            witnesses=witnesses,
            missing=missing,
        )
    return _result(
        contract,
        FormalStatus.PASS,
        "TRANSFORM_FIT_BOUNDARY_SATISFIED",
        candidates,
        witnesses=witnesses,
    )


def _identity_observation(event: RuntimeEvent) -> tuple[str | None, bool, bool]:
    """Return (identity token, canonical-present, conflicting-fields).

    Candidate display IDs are intentionally not treated as canonical identity
    when an event supplies a canonical/fingerprint field.  This permits safe
    aliases and object re-instantiation while retaining a visible conflict if
    an event carries two incompatible identity declarations.
    """

    attributes = event.attributes or {}
    canonical_keys = (
        "canonical_candidate_id",
        "canonical_identity",
        "candidate_fingerprint",
        "identity_fingerprint",
        "lineage_identity",
        "semantic_id",
    )
    values = [
        str(attributes[key])
        for key in canonical_keys
        if isinstance(attributes.get(key), str) and attributes.get(key)
    ]
    distinct = tuple(dict.fromkeys(values))
    if len(distinct) > 1:
        return None, True, True
    if distinct:
        return distinct[0], True, False
    return event.candidate_id, False, False


def _lineage_event_ids(event: RuntimeEvent) -> tuple[str, ...]:
    """Normalize optional provenance links carried by a selection event."""

    values: list[str] = list(event.parent_event_ids)
    attributes = event.attributes or {}
    for key in ("lineage_event_ids", "candidate_lineage_event_ids", "provenance_event_ids"):
        value = attributes.get(key)
        if isinstance(value, (list, tuple)):
            values.extend(item for item in value if isinstance(item, str) and item)
    return tuple(dict.fromkeys(values))


def _candidate_identity_strict(
    contract: FormalContract,
    events: Sequence[RuntimeEvent],
    candidates: tuple[str, ...],
) -> FormalContractResult:
    """Require runtime selection plus unique identity/provenance closure.

    The monitor is deliberately conservative.  It returns UNVERIFIABLE for a
    declaration/runtime conflict, missing selector observation, missing
    lineage, or ambiguous canonical matches; a genuinely selected but
    uncreated candidate remains a hard VIOLATION.
    """

    created = tuple(item for item in _events(events, EventType.CANDIDATE_CREATE) if item.candidate_id)
    runtime_selected = tuple(
        item
        for item in _events(events, EventType.CANDIDATE_SELECT)
        if item.source != EvidenceSource.DECLARED
    )
    declared_selected = tuple(
        item
        for item in _events(events, EventType.CANDIDATE_SELECT)
        if item.source == EvidenceSource.DECLARED
    )
    if not runtime_selected:
        return _result(
            contract,
            FormalStatus.UNVERIFIABLE,
            "RUNTIME_SELECTION_EVENT_MISSING",
            candidates,
            witnesses=[item.event_id for item in declared_selected],
            missing=("runtime_observed_CANDIDATE_SELECT",),
        )

    witnesses: list[str] = []
    for selected in runtime_selected:
        token, _canonical, conflicting = _identity_observation(selected)
        if conflicting or not token:
            return _result(
                contract,
                FormalStatus.UNVERIFIABLE,
                "RUNTIME_SELECTION_IDENTITY_CONFLICT",
                candidates,
                witnesses=[selected.event_id],
                missing=(f"{selected.event_id}:unambiguous_identity",),
            )
        matches = [item for item in created if _identity_observation(item)[0] == token]
        if not matches:
            return _result(
                contract,
                FormalStatus.VIOLATION,
                "SELECTED_CANDIDATE_NOT_CREATED",
                candidates,
                witnesses=[selected.event_id],
                invalid_nodes=("data.assessment",),
            )
        lineage = set(_lineage_event_ids(selected))
        if len(matches) > 1:
            linked = [item for item in matches if item.event_id in lineage]
            if len(linked) != 1:
                return _result(
                    contract,
                    FormalStatus.UNVERIFIABLE,
                    "CANDIDATE_LINEAGE_AMBIGUOUS",
                    candidates,
                    witnesses=[selected.event_id, *(item.event_id for item in matches)],
                    missing=(f"{selected.event_id}:unique_candidate_lineage",),
                )
            matches = linked
        elif lineage and matches[0].event_id not in lineage:
            return _result(
                contract,
                FormalStatus.UNVERIFIABLE,
                "CANDIDATE_LINEAGE_CONFLICT",
                candidates,
                witnesses=[selected.event_id, matches[0].event_id, *lineage],
                missing=(f"{selected.event_id}:candidate_lineage_consistency",),
            )
        elif not lineage:
            return _result(
                contract,
                FormalStatus.UNVERIFIABLE,
                "CANDIDATE_LINEAGE_MISSING",
                candidates,
                witnesses=[selected.event_id, matches[0].event_id],
                missing=(f"{selected.event_id}:candidate_lineage",),
            )
        witnesses.extend((selected.event_id, matches[0].event_id))

    if declared_selected:
        runtime_tokens = tuple(_identity_observation(item)[0] for item in runtime_selected)
        for declared in declared_selected:
            token, _canonical, conflicting = _identity_observation(declared)
            if conflicting or not token or token not in runtime_tokens:
                return _result(
                    contract,
                    FormalStatus.UNVERIFIABLE,
                    "DECLARATION_RUNTIME_IDENTITY_CONFLICT",
                    candidates,
                    witnesses=[*witnesses, declared.event_id, *[item.event_id for item in runtime_selected]],
                    missing=(f"{declared.event_id}:identity_conflict_resolution",),
                )
            witnesses.append(declared.event_id)

    return _result(
        contract,
        FormalStatus.PASS,
        "CANDIDATE_IDENTITY_LINEAGE_CLOSED",
        candidates,
        witnesses=witnesses,
    )


def _candidate_identity(
    contract: FormalContract,
    events: Sequence[RuntimeEvent],
    candidates: tuple[str, ...],
    *,
    policy: CandidateIdentityPolicy = CandidateIdentityPolicy.LEGACY_ID_ONLY,
) -> FormalContractResult:
    if policy == CandidateIdentityPolicy.STRICT_SELECTION_LINEAGE:
        return _candidate_identity_strict(contract, events, candidates)
    created = {item.candidate_id for item in _events(events, EventType.CANDIDATE_CREATE) if item.candidate_id}
    selected = _events(events, EventType.CANDIDATE_SELECT)
    if not selected:
        return _result(
            contract,
            FormalStatus.UNVERIFIABLE,
            "CANDIDATE_SELECTION_MISSING",
            candidates,
            missing=("CANDIDATE_SELECT",),
        )
    if not created:
        return _result(
            contract,
            FormalStatus.UNVERIFIABLE,
            "CANDIDATE_CREATION_MISSING",
            candidates,
            witnesses=[item.event_id for item in selected],
            missing=("CANDIDATE_CREATE",),
        )
    missing_ids = [item for item in selected if not item.candidate_id]
    if missing_ids:
        return _result(
            contract,
            FormalStatus.UNVERIFIABLE,
            "SELECTED_CANDIDATE_ID_MISSING",
            candidates,
            witnesses=[item.event_id for item in selected],
            missing=[f"{item.event_id}:candidate_id" for item in missing_ids],
        )
    unknown = [item for item in selected if item.candidate_id not in created]
    if unknown:
        return _result(
            contract,
            FormalStatus.VIOLATION,
            "SELECTED_CANDIDATE_NOT_CREATED",
            candidates,
            witnesses=[item.event_id for item in unknown],
            invalid_nodes=("data.assessment",),
        )
    return _result(
        contract,
        FormalStatus.PASS,
        "CANDIDATE_IDENTITY_CLOSED",
        candidates,
        witnesses=[item.event_id for item in selected],
    )


def _artifact_provenance(
    contract: FormalContract,
    events: Sequence[RuntimeEvent],
    candidates: tuple[str, ...],
) -> FormalContractResult:
    releases = _events(events, EventType.CLAIM_RELEASE)
    if not releases:
        return _result(
            contract,
            FormalStatus.UNVERIFIABLE,
            "CLAIM_RELEASE_MISSING",
            candidates,
            missing=("CLAIM_RELEASE",),
        )
    writes: dict[str, list[RuntimeEvent]] = {}
    for event in _events(events, EventType.ARTIFACT_WRITE):
        artifact_id = event.attributes.get("artifact_id")
        if isinstance(artifact_id, str) and artifact_id and event.artifact_sha256:
            writes.setdefault(artifact_id, []).append(event)
    witnesses: list[str] = []
    missing: list[str] = []
    violations: list[str] = []
    for release in releases:
        required = _attribute_list(release, "required_artifact_ids")
        bindings = release.attributes.get("artifact_bindings")
        if required is None:
            missing.append(f"{release.event_id}:required_artifact_ids")
            continue
        if not isinstance(bindings, Mapping):
            missing.append(f"{release.event_id}:artifact_bindings")
            continue
        witnesses.append(release.event_id)
        for artifact_id in required:
            preceding = [event for event in writes.get(artifact_id, ())
                         if event.logical_time < release.logical_time]
            latest = max(preceding, key=lambda event: event.logical_time) if preceding else None
            observed = None if latest is None else (latest.artifact_sha256, latest.event_id)
            bound = bindings.get(artifact_id)
            if observed is None:
                violations.append(f"{release.event_id}:{artifact_id}:unobserved")
                continue
            if not isinstance(bound, str) or not bound:
                missing.append(f"{release.event_id}:artifact_bindings.{artifact_id}")
                continue
            witnesses.append(observed[1])
            if bound != observed[0]:
                violations.append(f"{release.event_id}:{artifact_id}:identity_mismatch")
    if violations:
        return _result(
            contract, FormalStatus.VIOLATION, "ARTIFACT_PROVENANCE_NOT_CLOSED",
            candidates, witnesses=witnesses, invalid_nodes=("artifact.model",),
        )
    if missing:
        return _result(
            contract,
            FormalStatus.UNVERIFIABLE,
            "ARTIFACT_BINDING_EVIDENCE_MISSING",
            candidates,
            witnesses=witnesses,
            missing=missing,
        )
    return _result(
        contract,
        FormalStatus.PASS,
        "ARTIFACT_PROVENANCE_CLOSED",
        candidates,
        witnesses=witnesses,
    )


def _evidence_closure(
    contract: FormalContract,
    prior: Sequence[FormalContractResult],
    event_ids: set[str],
    candidates: tuple[str, ...],
) -> FormalContractResult:
    passed = [item for item in prior if item.status == FormalStatus.PASS]
    if not passed:
        return _result(
            contract,
            FormalStatus.UNVERIFIABLE,
            "NO_SATISFIED_CONTRACT_TO_CLOSE",
            candidates,
            missing=("satisfied_contract_result",),
        )
    witnesses: list[str] = []
    invalid: list[str] = []
    for item in passed:
        if not item.witness_event_ids:
            invalid.append(f"{item.contract_id}:witness_event_ids")
            continue
        witnesses.extend(item.witness_event_ids)
        invalid.extend(
            f"{item.contract_id}:{event_id}"
            for event_id in item.witness_event_ids
            if event_id not in event_ids
        )
    if invalid:
        return _result(
            contract,
            FormalStatus.VIOLATION,
            "SATISFIED_CONTRACT_HAS_NO_VALID_WITNESS",
            candidates,
            witnesses=witnesses,
            invalid_nodes=("data.assessment",),
        )
    return _result(
        contract,
        FormalStatus.PASS,
        "EVIDENCE_PATH_CLOSED",
        candidates,
        witnesses=witnesses,
    )


def _claim_release_safety(
    contract: FormalContract,
    events: Sequence[RuntimeEvent],
    prior: Sequence[FormalContractResult],
    candidates: tuple[str, ...],
) -> FormalContractResult:
    releases = _events(events, EventType.CLAIM_RELEASE)
    if not releases:
        return _result(
            contract,
            FormalStatus.UNVERIFIABLE,
            "CLAIM_RELEASE_MISSING",
            candidates,
            missing=("CLAIM_RELEASE",),
        )
    by_id = {item.contract_id: item for item in prior}
    missing: list[str] = []
    unsatisfied: list[str] = []
    witnesses: list[str] = []
    for event in releases:
        required = _attribute_list(event, "required_contract_ids")
        if required is None:
            missing.append(f"{event.event_id}:required_contract_ids")
            continue
        witnesses.append(event.event_id)
        for contract_id in required:
            result = by_id.get(contract_id)
            if result is None:
                # The policy is known here: the release explicitly named this
                # prerequisite.  A missing monitor result therefore cannot be
                # treated as an unknown policy; releasing anyway is unsafe.
                # (An omitted required_contract_ids field remains
                # UNVERIFIABLE above because its policy is not observable.)
                unsatisfied.append(f"{event.event_id}:{contract_id}:MISSING")
            elif result.status != FormalStatus.PASS:
                unsatisfied.append(f"{event.event_id}:{contract_id}:{result.status.value}")
                witnesses.extend(result.witness_event_ids)
            else:
                witnesses.extend(result.witness_event_ids)
    if missing:
        return _result(
            contract,
            FormalStatus.UNVERIFIABLE,
            "CLAIM_POLICY_OR_DEPENDENCY_MISSING",
            candidates,
            witnesses=witnesses,
            missing=missing,
        )
    if unsatisfied:
        return _result(
            contract,
            FormalStatus.VIOLATION,
            "CLAIM_RELEASE_WITH_UNSATISFIED_CONTRACT",
            candidates,
            witnesses=witnesses,
            invalid_nodes=("claim.performance",),
        )
    return _result(
        contract,
        FormalStatus.PASS,
        "CLAIM_RELEASE_SAFETY_SATISFIED",
        candidates,
        witnesses=witnesses,
    )


def _relational_selection(
    contract: FormalContract,
    events: Sequence[RuntimeEvent],
    comparison_events: Sequence[RuntimeEvent] | None,
    candidates: tuple[str, ...],
) -> FormalContractResult:
    if comparison_events is None:
        return _result(
            contract,
            FormalStatus.UNVERIFIABLE,
            "PAIRED_TRACE_MISSING",
            candidates,
            missing=("comparison_trace",),
        )

    def selections(source: Sequence[RuntimeEvent]) -> tuple[RuntimeEvent, ...]:
        return tuple(
            item
            for item in source
            if item.event_type == EventType.CANDIDATE_SELECT
            and item.attributes.get("assessment_influence_allowed") is False
        )

    left = selections(events)
    right = selections(comparison_events)
    if not left or not right:
        return _result(
            contract,
            FormalStatus.UNVERIFIABLE,
            "PRE_ASSESSMENT_SELECTION_MISSING",
            candidates,
            missing=("pre_assessment_candidate_selection",),
        )

    def signature(source: Sequence[RuntimeEvent]) -> tuple[tuple[str, str], ...] | None:
        rows: list[tuple[str, str]] = []
        for event in source:
            permitted = event.attributes.get("permitted_input_identity")
            if not isinstance(permitted, str) or not permitted or not event.candidate_id:
                return None
            rows.append((permitted, event.candidate_id))
        return tuple(rows)

    left_signature = signature(left)
    right_signature = signature(right)
    if left_signature is None or right_signature is None:
        return _result(
            contract,
            FormalStatus.UNVERIFIABLE,
            "PERMITTED_INPUT_PROJECTION_MISSING",
            candidates,
            missing=("permitted_input_identity",),
        )
    left_inputs = tuple(item[0] for item in left_signature)
    right_inputs = tuple(item[0] for item in right_signature)
    if left_inputs != right_inputs:
        return _result(
            contract,
            FormalStatus.NOT_ESTIMABLE,
            "RELATIONAL_PREMISE_NOT_MET",
            candidates,
            witnesses=[item.event_id for item in (*left, *right)],
            missing=("equivalent_permitted_input_projection",),
        )
    if tuple(item[1] for item in left_signature) != tuple(item[1] for item in right_signature):
        return _result(
            contract,
            FormalStatus.VIOLATION,
            "ASSESSMENT_VARIATION_CHANGED_SELECTION",
            candidates,
            witnesses=[item.event_id for item in (*left, *right)],
            invalid_nodes=("data.assessment",),
        )
    return _result(
        contract,
        FormalStatus.PASS,
        "RELATIONAL_SELECTION_ISOLATION_SATISFIED",
        candidates,
        witnesses=[item.event_id for item in (*left, *right)],
    )


_MONITORS = {
    "FC-T1-TEMPORAL-AVAILABILITY": _temporal_availability,
    "FC-T2-TRANSFORM-FIT-BOUNDARY": _transform_fit_boundary,
    "FC-P1-CANDIDATE-IDENTITY-CLOSURE": _candidate_identity,
    "FC-P2-ARTIFACT-PROVENANCE-CLOSURE": _artifact_provenance,
}


def _scope_candidates(events: Sequence[RuntimeEvent], candidate_ids: Sequence[str] | None) -> tuple[str, ...]:
    if candidate_ids is not None:
        values = tuple(sorted(set(candidate_ids)))
        if not values or any(not isinstance(item, str) or not item for item in values):
            raise ValueError("candidate_ids must be a non-empty sequence of strings when supplied.")
        return values
    observed = tuple(sorted({item.candidate_id for item in events if item.candidate_id}))
    return observed or ("trace:global",)


def evaluate_formal_contracts(
    events: Sequence[RuntimeEvent],
    *,
    contracts: Sequence[FormalContract] | None = None,
    candidate_ids: Sequence[str] | None = None,
    comparison_events: Sequence[RuntimeEvent] | None = None,
    candidate_identity_policy: CandidateIdentityPolicy = CandidateIdentityPolicy.LEGACY_ID_ONLY,
) -> FormalTraceEvaluation:
    """Evaluate the supported finite-trace semantics in dependency order.

    The claim-release and evidence-closure monitors deliberately consume the
    preceding monitor outcomes, which gives an explicit finite trace ordering
    without adding a heavyweight LTLf dependency.
    """

    if any(not isinstance(item, RuntimeEvent) for item in events):
        raise TypeError("events must contain RuntimeEvent values.")
    if not isinstance(candidate_identity_policy, CandidateIdentityPolicy):
        raise TypeError("candidate_identity_policy must be a CandidateIdentityPolicy.")
    ordered_events = tuple(sorted(events, key=lambda item: (item.logical_time, item.event_id)))
    event_ids = [item.event_id for item in ordered_events]
    if len(event_ids) != len(set(event_ids)):
        raise ValueError("runtime event IDs must be unique.")
    selected = tuple(default_formal_contracts() if contracts is None else contracts)
    if not selected or any(not isinstance(item, FormalContract) for item in selected):
        raise ValueError("contracts must contain FormalContract values.")
    if len({item.contract_id for item in selected}) != len(selected):
        raise ValueError("formal contract IDs must be unique.")
    scope = _scope_candidates(ordered_events, candidate_ids)
    results: list[FormalContractResult] = []
    for contract in selected:
        monitor = _MONITORS.get(contract.contract_id)
        if contract.contract_id == "FC-T3-LEARNED-FIT-BOUNDARY":
            monitor = _transform_fit_boundary
        if monitor is not None:
            if contract.contract_id == "FC-P1-CANDIDATE-IDENTITY-CLOSURE":
                results.append(
                    _candidate_identity(
                        contract,
                        ordered_events,
                        scope,
                        policy=candidate_identity_policy,
                    )
                )
            else:
                results.append(monitor(contract, ordered_events, scope))
        elif contract.contract_id == "FC-P3-EVIDENCE-CLOSURE":
            results.append(_evidence_closure(contract, results, set(event_ids), scope))
        elif contract.contract_id == "FC-G1-CLAIM-RELEASE-SAFETY":
            results.append(_claim_release_safety(contract, ordered_events, results, scope))
        elif contract.contract_id == "FC-R1-SELECTION-NONINTERFERENCE":
            results.append(_relational_selection(contract, ordered_events, comparison_events, scope))
        else:
            raise ValueError(f"no executable monitor is registered for {contract.contract_id!r}.")
    return FormalTraceEvaluation(
        trace_sha256=trace_sha256(ordered_events),
        event_count=len(ordered_events),
        results=tuple(results),
        comparison_trace_sha256=None if comparison_events is None else trace_sha256(comparison_events),
    )


_RELATIONAL_OBLIGATIONS = {
    "ASSESSMENT_FIT_ISOLATION",
    "ASSESSMENT_CALIBRATION_ISOLATION",
    "ASSESSMENT_THRESHOLD_ISOLATION",
    "ASSESSMENT_SEARCH_ISOLATION",
    "ASSESSMENT_SELECTION_ISOLATION",
    "OUTCOME_FEEDBACK_ISOLATION",
}
_TEMPORAL_OBLIGATIONS = {
    "TIME_AVAILABILITY",
    "TIME_ALIGNMENT",
    "LABEL_TIME_ALIGNMENT",
    "SELECTION_EXECUTION_CONSISTENCY",
    "CLAIM_DEPENDENCY_INTEGRITY",
}


def formal_contract_inventory() -> dict[str, Any]:
    """Inventory all authoritative subsequent responsibilities and initial monitor coverage."""

    contracts = default_formal_contracts()
    by_obligation: dict[str, list[str]] = {}
    for contract in contracts:
        by_obligation.setdefault(contract.obligation_id, []).append(contract.contract_id)
    obligations = []
    for obligation in CanonicalContractSpecification.default().obligations:
        if obligation.obligation_id in _RELATIONAL_OBLIGATIONS:
            classification = ContractClass.RELATIONAL.value
        elif obligation.obligation_id in _TEMPORAL_OBLIGATIONS:
            classification = ContractClass.TEMPORAL.value
        else:
            classification = ContractClass.STATE.value
        implemented = tuple(sorted(by_obligation.get(obligation.obligation_id, ())))
        obligations.append(
            {
                "obligation_id": obligation.obligation_id,
                "evidence_key": obligation.evidence_key,
                "primary_layer": obligation.primary_layer,
                "conceptual_classification": classification,
                "executable_contract_ids": list(implemented),
                "initial_status": "FORMALIZED" if implemented else "NOT_FORMALIZED_initial",
            }
        )
    return {
        "schema_version": "dmcasrc.autofact-formal-contract-inventory.initial",
        "canonical_contract_version": CanonicalContractSpecification.default().version,
        "formal_contract_count": len(contracts),
        "canonical_responsibility_count": len(obligations),
        "formalized_responsibility_count": sum(bool(item["executable_contract_ids"]) for item in obligations),
        "candidate_identity_policies": {
            CandidateIdentityPolicy.LEGACY_ID_ONLY.value: "Backward-compatible candidate-id closure for historical receipts.",
            CandidateIdentityPolicy.STRICT_SELECTION_LINEAGE.value: "Runtime selector plus unique candidate provenance lineage; declaration/runtime conflicts remain UNVERIFIABLE.",
        },
        "contracts": [item.to_dict() for item in contracts],
        "responsibility_families": obligations,
        "scope_limitations": [
            "Finite-trace runtime monitors only; no whole-program Python verification.",
            "The relational check is a paired-trace executable non-interference contract, not HyperLTL model checking.",
            "Unobserved semantic roles remain UNVERIFIABLE or NOT_ESTIMABLE rather than inferred from event count alone.",
        ],
    }


__all__ = [
    "ContractClass",
    "CandidateIdentityPolicy",
    "FormalContract",
    "FormalContractResult",
    "FormalStatus",
    "FormalTraceEvaluation",
    "default_formal_contracts",
    "evaluate_formal_contracts",
    "formal_contract_inventory",
]
