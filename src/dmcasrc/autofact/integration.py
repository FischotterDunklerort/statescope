"""Bridge AutoFact contracts into the existing canonical audit and claim gate."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from dmcasrc.application.audit import audit_scientific_facts
from dmcasrc.application.formal_contracts import (
    CandidateIdentityPolicy,
    FormalContract,
    FormalStatus,
    FormalTraceEvaluation,
    evaluate_formal_contracts,
)
from dmcasrc.domain.contracts import CanonicalEvidenceFact, ScientificAuditReport, ScientificClaim
from dmcasrc.domain.graph import PipelineGraph

from .events import RuntimeEvent


def _stable_id(namespace: str, payload: Any) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=repr)
    return f"{namespace}:{hashlib.sha256(encoded.encode('utf-8')).hexdigest()}"


def canonical_facts_from_formal_evaluation(
    evaluation: FormalTraceEvaluation,
    *,
    source_profile: str = "autofact",
) -> tuple[CanonicalEvidenceFact, ...]:
    """Translate monitor outcomes into core-owned, status-free evidence facts.

    The bridge encodes PASS/VIOLATION as an expected-vs-observed fact and
    preserves UNVERIFIABLE/NOT_ESTIMABLE as explicit missing evidence.  The
    existing ``observe_canonical_facts`` function remains the code that assigns
    canonical observation statuses and the existing claim gate makes release
    decisions.
    """

    if not isinstance(evaluation, FormalTraceEvaluation):
        raise TypeError("evaluation must be FormalTraceEvaluation.")
    if not isinstance(source_profile, str) or not source_profile:
        raise ValueError("source_profile must be a non-empty string.")
    facts: list[CanonicalEvidenceFact] = []
    coordinates: set[tuple[str, str, str]] = set()
    for result in evaluation.results:
        for candidate_id in result.candidate_ids:
            coordinate = (candidate_id, result.node_id, result.obligation_id)
            if coordinate in coordinates:
                raise ValueError(
                    "formal contract mapping produced duplicate canonical coordinates: "
                    f"{coordinate!r}."
                )
            coordinates.add(coordinate)
            payload: dict[str, Any] = {
                "formal_contract_id": result.contract_id,
                "formal_status": result.status.value,
                "witness_event_ids": list(result.witness_event_ids),
                "reason_codes": list(result.reason_codes),
                "evidence_source": "DERIVED",
            }
            if result.status == FormalStatus.PASS:
                payload.update({"expected": True, "observed": True})
            elif result.status == FormalStatus.VIOLATION:
                payload.update(
                    {
                        "expected": True,
                        "observed": False,
                        "invalid_node_ids": list(result.invalid_node_ids),
                    }
                )
            else:
                missing = list(result.missing_required_evidence)
                if result.status == FormalStatus.NOT_ESTIMABLE:
                    missing.append("relational_premise_not_estimable")
                payload["missing_required_evidence"] = sorted(set(missing))
            identity_payload = {
                "trace_sha256": evaluation.trace_sha256,
                "candidate_id": candidate_id,
                "contract_id": result.contract_id,
                "node_id": result.node_id,
                "obligation_id": result.obligation_id,
            }
            canonical_identity = _stable_id("autofact.canonical", identity_payload)
            evidence_event = result.witness_event_ids[0] if result.witness_event_ids else result.contract_id
            facts.append(
                CanonicalEvidenceFact(
                    fact_id=_stable_id("autofact.fact", identity_payload),
                    candidate_id=candidate_id,
                    node_id=result.node_id,
                    obligation_id=result.obligation_id,
                    primary_layer=_primary_layer(result.obligation_id),
                    affected_layers=_affected_layers(result.obligation_id),
                    source_profile=source_profile,
                    native_object_ref=f"dmcasrc.application.formal_contracts:{result.contract_id}",
                    native_source_path="autofact/normalized-trace.jsonl",
                    native_fact_type=result.contract_id,
                    canonical_identity=canonical_identity,
                    evidence_sha256=evaluation.trace_sha256,
                    evidence_ref=f"autofact://{evaluation.trace_sha256}#{evidence_event}",
                    payload=payload,
                )
            )
    return tuple(facts)


def _primary_layer(obligation_id: str) -> str:
    from dmcasrc.domain.contracts import CanonicalContractSpecification

    return CanonicalContractSpecification.default().by_id(obligation_id).primary_layer


def _affected_layers(obligation_id: str) -> tuple[str, ...]:
    from dmcasrc.domain.contracts import CanonicalContractSpecification

    return CanonicalContractSpecification.default().by_id(obligation_id).affected_layers


@dataclass(frozen=True)
class AutoFactAudit:
    """Trace monitor output plus the unmodified canonical audit result."""

    evaluation: FormalTraceEvaluation
    facts: tuple[CanonicalEvidenceFact, ...]
    report: ScientificAuditReport

    def to_dict(self) -> dict[str, Any]:
        return {
            "evaluation": self.evaluation.to_dict(),
            "facts": [item.to_dict() for item in self.facts],
            "report": self.report.to_dict(),
        }


def audit_autofact_trace(
    graph: PipelineGraph,
    events: Sequence[RuntimeEvent],
    claims: Sequence[ScientificClaim],
    *,
    contracts: Sequence[FormalContract] | None = None,
    candidate_ids: Sequence[str] | None = None,
    comparison_events: Sequence[RuntimeEvent] | None = None,
    source_profile: str = "autofact",
    candidate_identity_policy: CandidateIdentityPolicy = CandidateIdentityPolicy.LEGACY_ID_ONLY,
) -> AutoFactAudit:
    """Run AutoFact -> formal monitor -> core observation -> existing claim gate."""

    evaluation = evaluate_formal_contracts(
        events,
        contracts=contracts,
        candidate_ids=candidate_ids,
        comparison_events=comparison_events,
        candidate_identity_policy=candidate_identity_policy,
    )
    facts = canonical_facts_from_formal_evaluation(evaluation, source_profile=source_profile)
    report = audit_scientific_facts(graph, facts, claims)
    return AutoFactAudit(evaluation=evaluation, facts=facts, report=report)


__all__ = [
    "AutoFactAudit",
    "audit_autofact_trace",
    "canonical_facts_from_formal_evaluation",
]
