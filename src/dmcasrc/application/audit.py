"""Deterministic candidate-level evaluation of the frozen proof obligations."""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from dmcasrc.application.claim_gate import evaluate_claims, evaluate_scientific_claims
from dmcasrc.domain.contracts import (
    ALLOWED_PREDICTION_ORIGINS,
    AuditExecutionSummary,
    AuditFinding,
    AuditReport,
    CanonicalEvidenceFact,
    CanonicalObservation,
    ClaimAction,
    ClaimManifest,
    ClaimScope,
    ClaimSpecification,
    ContractSpecification,
    Obligation,
    ScientificAuditFinding,
    ScientificAuditReport,
    ScientificClaim,
    validate_obligation_ids,
)
from dmcasrc.domain.graph import PipelineGraph
from dmcasrc.domain.ledger import CandidateLedger, CandidateRecord, parse_timestamp
from dmcasrc.infrastructure.deployment_manifest_loader import ArtifactVerification
from dmcasrc.infrastructure.deployment_manifest_schema import ArtifactManifest

_HASH = re.compile(r"^[0-9a-f]{64}$")
_EVIDENCE_FIELDS = {
    "O1": {"source_keys", "capability_status"},
    "O2": {"available_ts", "capability_status"},
    "O3": {
        "assessment_keys",
        "fit_keys",
        "calibration_keys",
        "threshold_keys",
        "capability_status",
    },
    "O4": {"origin", "capability_status"},
    "O5": {"observed_node_ids", "capability_status"},
    "O6": {
        "artifact_id",
        "fit_candidate_keys",
        "applies_to_candidate_keys",
        "capability_status",
    },
    "O7": {
        "artifact_id",
        "fit_candidate_keys",
        "applies_to_candidate_keys",
        "capability_status",
    },
    "O8": {
        "artifact_id",
        "semantic_id",
        "training_data_refs",
        "config_artifact_id",
        "config_hash",
        "capability_status",
    },
    "O9": {
        "decision_execution_id",
        "action_execution_id",
        "evaluation_execution_id",
        "capability_status",
    },
    "O10": {"artifact_ids", "capability_status"},
    "O11": {"consumer_roles", "capability_status"},
    "O12": {"claim_ids", "artifact_ids", "capability_status"},
}


@dataclass(frozen=True)
class _Result:
    status: str
    error_code: str | None
    detail: str
    first_invalid_edge: str | None = None
    evidence_refs: tuple[str, ...] = ()


def _pass(detail: str = "proof obligation satisfied", refs: Sequence[str] = ()) -> _Result:
    return _Result("PASS", None, detail, evidence_refs=tuple(sorted(set(refs))))


def _violation(code: str, detail: str, edge: str | None = None) -> _Result:
    return _Result("VIOLATION", code, detail, edge)


def _unverifiable(code: str, detail: str) -> _Result:
    return _Result("UNVERIFIABLE", code, detail)


def _error(code: str, detail: str) -> _Result:
    return _Result("ERROR", code, detail)


def _evidence(record: CandidateRecord, obligation: Obligation) -> Mapping[str, Any] | _Result:
    value = record.evidence.get(obligation.evidence_key)
    # 1. Namespace does not exist
    if value is None:
        return _unverifiable(
            "EVIDENCE_MISSING",
            f"missing structured evidence for {obligation.evidence_key}",
        )
    # 2. Value is not a Mapping
    if not isinstance(value, Mapping):
        return _error(
            "EVIDENCE_OBJECT_INVALID",
            f"evidence for {obligation.evidence_key} must be an object",
        )
    # 3. Check unknown fields
    unknown = sorted(set(value) - _EVIDENCE_FIELDS[obligation.obligation_id])
    if unknown:
        return _Result(
            "ERROR",
            "EVIDENCE_FIELD_UNKNOWN",
            f"unknown evidence field(s): {unknown}",
        )
    # 4. Check capability_status
    capability = value.get("capability_status")
    if capability is not None:
        if capability not in ("unsupported", "not_applicable"):
            return _error(
                "CAPABILITY_STATUS_INVALID",
                f"unknown capability_status: {capability!r}",
            )
        # Check for capability+evidence conflict
        substantive_keys = set(value) - {"capability_status"}
        if substantive_keys:
            return _error(
                "CAPABILITY_EVIDENCE_CONFLICT",
                f"capability_status present alongside substantive evidence fields: {sorted(substantive_keys)}",
            )
    if capability == "unsupported":
        return _Result("TOOL_UNSUPPORTED", "CAPABILITY_UNSUPPORTED", "adapter declared unsupported")
    if capability == "not_applicable":
        return _Result("NOT_APPLICABLE", None, "obligation is not applicable")
    return value


def _list(value: Mapping[str, Any], key: str) -> list[Any] | None:
    item = value.get(key)
    return item if isinstance(item, list) else None


def _candidate_identity(
    record: CandidateRecord, evidence: Mapping[str, Any], context: _Context
) -> _Result:
    source_keys = _list(evidence, "source_keys")
    if not source_keys:
        return _unverifiable("CANDIDATE_IDENTITY_EVIDENCE_MISSING", "source keys are absent")
    if all(item == record.candidate_key for item in source_keys):
        return _pass()
    return _violation("CANDIDATE_IDENTITY_MISMATCH", "candidate key differs across layers")


def _time_availability(
    record: CandidateRecord, evidence: Mapping[str, Any], context: _Context
) -> _Result:
    available = _list(evidence, "available_ts")
    if available is None or not available:
        return _unverifiable("TIME_EVIDENCE_MISSING", "available timestamps are absent")
    try:
        decision = parse_timestamp("decision_ts", record.decision_ts)
        observed = [parse_timestamp("available_ts", item) for item in available]
    except ValueError:
        return _error("TIME_FORMAT_INVALID", "time evidence is not valid ISO-8601")
    if all(item <= decision for item in observed):
        return _pass()
    return _violation("AVAILABLE_AFTER_DECISION", "an input became available after decision_ts")


def _fold_role(record: CandidateRecord, evidence: Mapping[str, Any], context: _Context) -> _Result:
    names = ("assessment_keys", "fit_keys", "calibration_keys", "threshold_keys")
    groups = {name: _list(evidence, name) for name in names}
    if any(value is None for value in groups.values()):
        return _unverifiable("FOLD_ROLE_EVIDENCE_MISSING", "one or more fold-role sets are absent")
    assessment = set(groups["assessment_keys"] or [])
    reused = assessment.intersection(
        set(groups["fit_keys"] or [])
        | set(groups["calibration_keys"] or [])
        | set(groups["threshold_keys"] or [])
    )
    if reused:
        return _violation("ASSESSMENT_REUSED", f"assessment keys reused: {sorted(reused)}")
    return _pass()


def _prediction_origin(
    record: CandidateRecord, evidence: Mapping[str, Any], context: _Context
) -> _Result:
    origin = evidence.get("origin")
    if not isinstance(origin, str) or not origin:
        return _unverifiable("PREDICTION_ORIGIN_MISSING", "prediction origin is absent")
    if origin in ALLOWED_PREDICTION_ORIGINS:
        return _pass()
    return _violation(
        "PREDICTION_ORIGIN_INVALID",
        f"prediction origin is '{origin}', allowed: {sorted(ALLOWED_PREDICTION_ORIGINS)}",
    )


def _o5_observed_nodes(evidence: Mapping[str, Any], graph: PipelineGraph) -> _Result | list[str]:
    """Validate observed_node_ids; return the list or an error _Result."""
    if "observed_node_ids" not in evidence:
        return _unverifiable("ANCESTOR_EVIDENCE_MISSING", "observed ancestor nodes are absent")
    observed = evidence["observed_node_ids"]
    if not isinstance(observed, list):
        return _error("ANCESTOR_EVIDENCE_INVALID", "observed_node_ids must be a list of strings")
    if not all(isinstance(item, str) and item for item in observed):
        return _error(
            "ANCESTOR_EVIDENCE_INVALID", "observed_node_ids must contain non-empty strings"
        )
    if len(observed) != len(set(observed)):
        return _error("ANCESTOR_EVIDENCE_INVALID", "observed_node_ids contains duplicates")
    known_nodes = {node.node_id for node in graph.nodes}
    unknown = [item for item in observed if item not in known_nodes]
    if unknown:
        return _error(
            "ANCESTOR_EVIDENCE_INVALID", f"unknown nodes in observed_node_ids: {sorted(unknown)}"
        )
    return observed


def _o5_ancestor_finding(
    expected_nodes: list[str],
    candidate_key: str,
    phase1: dict[tuple[str, str, str], AuditFinding],
) -> _Result | None:
    """Reduce ancestor phase-1 findings to a single O5 result (None if all PASS)."""
    has_violation = False
    violator_nodes: set[str] = set()
    has_error = False
    has_unverifiable = False
    obligation_ids = [oid for oid in _NON_O5_EVALUATORS if oid != "O5"]
    for nid in expected_nodes:
        for obligation_id in obligation_ids:
            finding = phase1.get((candidate_key, nid, obligation_id))
            if finding is None:
                continue
            if finding.status == "VIOLATION":
                has_violation = True
                violator_nodes.add(nid)
            elif finding.status == "ERROR":
                has_error = True
            elif finding.status in {"UNVERIFIABLE", "TOOL_UNSUPPORTED"}:
                has_unverifiable = True
    if has_violation:
        return _violation(
            "ANCESTOR_OBLIGATION_VIOLATION",
            f"ancestor obligation violations at nodes: {sorted(violator_nodes)}",
        )
    if has_error:
        return _error("ANCESTOR_AUDIT_ERROR", "ancestor audit errors detected")
    if has_unverifiable:
        return _unverifiable(
            "ANCESTOR_OBLIGATION_UNVERIFIABLE",
            "ancestor obligations could not be verified",
        )
    return None


def _evaluate_o5_closure(
    record: CandidateRecord,
    evidence: Mapping[str, Any],
    graph: PipelineGraph,
    ledger: CandidateLedger,
    phase1: dict[tuple[str, str, str], AuditFinding],
) -> _Result:
    """Phase-2 O5: ancestor proof closure.

    Examines all ancestor nodes (plus current) for violations, errors,
    or unverifiable statuses, and produces the appropriate O5 finding.
    """
    observed_or_error = _o5_observed_nodes(evidence, graph)
    if isinstance(observed_or_error, _Result):
        return observed_or_error
    observed = observed_or_error

    expected_nodes = list(graph.relevant_node_ids(record.node_id))
    unobserved = [nid for nid in expected_nodes if nid not in observed]
    if unobserved:
        return _unverifiable(
            "ANCESTOR_NODE_UNOBSERVED",
            f"expected nodes not observed: {sorted(unobserved)}",
        )

    # Ensure every expected node has a ledger record
    missing_records = [
        nid for nid in expected_nodes if ledger.by_candidate_node(record.candidate_key, nid) is None
    ]
    if missing_records:
        first_edge = graph.first_invalid_edge(record.node_id, set(missing_records))
        return _Result(
            "UNVERIFIABLE",
            "ANCESTOR_RECORD_MISSING",
            f"missing ledger records for nodes: {sorted(missing_records)}",
            first_invalid_edge=first_edge,
        )

    ancestor = _o5_ancestor_finding(expected_nodes, record.candidate_key, phase1)
    if ancestor is not None:
        if ancestor.status == "VIOLATION":
            violators = {
                n
                for n in expected_nodes
                if any(
                    phase1.get((record.candidate_key, n, oid))
                    and phase1[(record.candidate_key, n, oid)].status == "VIOLATION"
                    for oid in _NON_O5_EVALUATORS
                    if oid != "O5"
                )
            }
            return _Result(
                "VIOLATION",
                "ANCESTOR_OBLIGATION_VIOLATION",
                ancestor.detail,
                first_invalid_edge=graph.first_invalid_edge(record.node_id, violators),
            )
        return ancestor
    return _pass()


@dataclass(frozen=True)
class _FittedIdentityCodes:
    evidence_missing: str
    artifact_unknown: str
    assessment_reused: str
    scope_mismatch: str


_CALIBRATION_CODES = _FittedIdentityCodes(
    evidence_missing="CALIBRATION_EVIDENCE_MISSING",
    artifact_unknown="CALIBRATION_ARTIFACT_UNKNOWN",
    assessment_reused="CALIBRATION_ASSESSMENT_REUSED",
    scope_mismatch="CALIBRATION_SCOPE_MISMATCH",
)

_THRESHOLD_CODES = _FittedIdentityCodes(
    evidence_missing="THRESHOLD_EVIDENCE_MISSING",
    artifact_unknown="THRESHOLD_ARTIFACT_UNKNOWN",
    assessment_reused="THRESHOLD_ASSESSMENT_REUSED",
    scope_mismatch="THRESHOLD_SCOPE_MISMATCH",
)


def _fitted_identity(
    record: CandidateRecord,
    evidence: Mapping[str, Any],
    context: _Context,
    *,
    codes: _FittedIdentityCodes,
) -> _Result:
    artifact_id = evidence.get("artifact_id")
    fit_keys = _list(evidence, "fit_candidate_keys")
    applies_to = _list(evidence, "applies_to_candidate_keys")
    if not isinstance(artifact_id, str) or fit_keys is None or applies_to is None:
        return _unverifiable(codes.evidence_missing, "identity evidence is incomplete")
    if artifact_id not in context.artifacts:
        return _violation(codes.artifact_unknown, f"unknown artifact: {artifact_id}")
    if record.candidate_key in fit_keys:
        return _violation(codes.assessment_reused, "current candidate was used for fitting")
    if record.candidate_key not in applies_to:
        return _violation(codes.scope_mismatch, "identity does not apply to candidate")
    return _pass(refs=(artifact_id,))


def _calibration_identity(
    record: CandidateRecord, evidence: Mapping[str, Any], context: _Context
) -> _Result:
    return _fitted_identity(record, evidence, context, codes=_CALIBRATION_CODES)


def _threshold_identity(
    record: CandidateRecord, evidence: Mapping[str, Any], context: _Context
) -> _Result:
    return _fitted_identity(record, evidence, context, codes=_THRESHOLD_CODES)


def _model_training_refs(training_data_refs_raw: Any, context: _Context) -> _Result | set[str]:
    """Validate training_data_refs; return the ref set or an error _Result."""
    if not isinstance(training_data_refs_raw, list) or not training_data_refs_raw:
        return _error("MODEL_TRAINING_REFS_INVALID", "training_data_refs must be a non-empty list")
    if not all(isinstance(item, str) and item for item in training_data_refs_raw):
        return _error(
            "MODEL_TRAINING_REFS_INVALID", "training_data_refs must be a list of non-empty strings"
        )
    if len(training_data_refs_raw) != len(set(training_data_refs_raw)):
        return _error("MODEL_TRAINING_REFS_INVALID", "training_data_refs contains duplicates")
    refs = set(training_data_refs_raw)
    unknown = [ref for ref in sorted(refs) if ref not in context.artifacts]
    if unknown:
        return _violation(
            "MODEL_TRAINING_ARTIFACT_UNKNOWN", f"unknown training artifact: {unknown}"
        )
    return refs


def _model_metadata_binding(
    artifact: Any, training_refs: set[str], config_artifact_id: str
) -> _Result | None:
    """Verify model metadata matches evidence; return None when consistent."""
    metadata = getattr(artifact, "metadata", None) or {}
    meta_training = metadata.get("training_data_refs")
    meta_config = metadata.get("config_artifact_id")
    if meta_training is None or meta_config is None:
        return _unverifiable(
            "MODEL_METADATA_BINDING_MISSING",
            "model metadata missing training_data_refs or config_artifact_id",
        )
    meta_training_set = set(meta_training) if isinstance(meta_training, list) else set()
    if meta_training_set != training_refs:
        return _violation(
            "MODEL_TRAINING_BINDING_MISMATCH",
            "metadata training_data_refs differs from evidence",
        )
    if meta_config != config_artifact_id:
        return _violation(
            "MODEL_CONFIG_BINDING_MISMATCH",
            f"metadata config_artifact_id '{meta_config}' != evidence '{config_artifact_id}'",
        )
    return None


def _model_artifact_verifications(all_artifact_ids: list[str], context: _Context) -> _Result | None:
    """Check integrity verification for every bound artifact; None if all pass."""
    for aid in all_artifact_ids:
        ver = context.verifications.get(aid)
        if ver is None:
            return _unverifiable(
                "MODEL_ARTIFACT_VERIFICATION_MISSING",
                f"verification missing for artifact: {aid}",
            )
        if ver.status == "VIOLATION":
            return _violation(
                "MODEL_ARTIFACT_INTEGRITY_FAILED",
                f"integrity violation for artifact: {aid}",
            )
        if ver.status != "PASS":
            return _unverifiable(
                "MODEL_ARTIFACT_UNVERIFIED",
                f"artifact {aid} verification status is {ver.status}",
            )
    return None


def _model_identity(
    record: CandidateRecord, evidence: Mapping[str, Any], context: _Context
) -> _Result:
    required = (
        "artifact_id",
        "semantic_id",
        "training_data_refs",
        "config_artifact_id",
        "config_hash",
    )
    missing_fields = [key for key in required if key not in evidence]
    if missing_fields:
        return _unverifiable(
            "MODEL_IDENTITY_EVIDENCE_MISSING",
            f"model identity is incomplete, missing: {sorted(missing_fields)}",
        )

    artifact_id = evidence["artifact_id"]
    config_artifact_id = evidence["config_artifact_id"]

    # 1. Model artifact existence and kind
    artifact = context.artifacts.get(artifact_id)
    if artifact is None:
        return _violation("MODEL_ARTIFACT_UNKNOWN", f"unknown artifact: {artifact_id}")
    if getattr(artifact, "kind", None) != "model":
        return _violation(
            "MODEL_ARTIFACT_KIND_INVALID",
            f"artifact {artifact_id} kind is '{getattr(artifact, 'kind', None)}', expected 'model'",
        )

    # 2. Semantic ID match
    if evidence["semantic_id"] != artifact.semantic_id:
        return _violation(
            "MODEL_SEMANTIC_ID_MISMATCH",
            f"evidence semantic_id '{evidence['semantic_id']}' != artifact '{artifact.semantic_id}'",
        )

    # 3. Training data refs validation
    refs_or_error = _model_training_refs(evidence["training_data_refs"], context)
    if isinstance(refs_or_error, _Result):
        return refs_or_error
    training_refs = refs_or_error

    # 4. Config artifact existence
    if config_artifact_id not in context.artifacts:
        return _violation(
            "MODEL_CONFIG_ARTIFACT_UNKNOWN",
            f"unknown config artifact: {config_artifact_id}",
        )

    # 5. Config hash format and match
    config_hash = evidence["config_hash"]
    if not isinstance(config_hash, str) or not _HASH.fullmatch(config_hash):
        return _error("MODEL_CONFIG_HASH_INVALID", "config_hash must be a 64-char hex string")
    config_artifact_sha = getattr(context.artifacts[config_artifact_id], "sha256", "") or ""
    if config_hash != config_artifact_sha:
        return _violation(
            "MODEL_CONFIG_HASH_MISMATCH",
            f"evidence config_hash '{config_hash}' != artifact sha256 '{config_artifact_sha}'",
        )

    # 6. Model metadata binding
    binding = _model_metadata_binding(artifact, training_refs, config_artifact_id)
    if binding is not None:
        return binding

    # 7. Verification checks for model, config, and all training data artifacts
    all_artifact_ids = [artifact_id, config_artifact_id] + sorted(training_refs)
    ver_error = _model_artifact_verifications(all_artifact_ids, context)
    if ver_error is not None:
        return ver_error

    return _pass(refs=tuple(sorted(set(all_artifact_ids))))


def _execution_identity(
    record: CandidateRecord, evidence: Mapping[str, Any], context: _Context
) -> _Result:
    fields = ("decision_execution_id", "action_execution_id", "evaluation_execution_id")
    values = [evidence.get(field) for field in fields]
    if any(not isinstance(value, str) or not value for value in values):
        return _unverifiable("EXECUTION_IDENTITY_MISSING", "execution identity is incomplete")
    if len(set(values)) != 1:
        return _violation(
            "EXECUTION_IDENTITY_MISMATCH", "decision/action/evaluation identities differ"
        )
    return _pass(refs=(values[0],))


def _artifact_integrity(
    record: CandidateRecord, evidence: Mapping[str, Any], context: _Context
) -> _Result:
    artifact_ids = _list(evidence, "artifact_ids")
    if not artifact_ids:
        return _unverifiable("ARTIFACT_REFERENCES_MISSING", "artifact references are absent")
    unknown = set(artifact_ids) - set(context.artifacts)
    if unknown:
        return _violation("ARTIFACT_UNKNOWN", f"unknown artifacts: {sorted(unknown)}")
    results = [context.verifications.get(artifact_id) for artifact_id in artifact_ids]
    if any(result is None for result in results):
        return _unverifiable("ARTIFACT_VERIFICATION_MISSING", "integrity verification is absent")
    if any(result.status == "VIOLATION" for result in results if result is not None):
        return _violation("ARTIFACT_INTEGRITY_FAILED", "one or more artifacts failed integrity")
    if any(result.status != "PASS" for result in results if result is not None):
        return _unverifiable("ARTIFACT_UNVERIFIED", "one or more artifacts could not be verified")
    return _pass(refs=tuple(str(item) for item in artifact_ids))


def _outcome_consumption(
    record: CandidateRecord, evidence: Mapping[str, Any], context: _Context
) -> _Result:
    roles = _list(evidence, "consumer_roles")
    if roles is None:
        return _unverifiable("OUTCOME_CONSUMPTION_MISSING", "outcome consumers are not declared")
    forbidden = {"fit", "calibration", "threshold", "selection", "development"}
    leaked = forbidden.intersection(str(item) for item in roles)
    if leaked:
        return _violation("ASSESSMENT_OUTCOME_REUSED", f"outcome reused by: {sorted(leaked)}")
    return _pass()


def _claim_binding(
    record: CandidateRecord, evidence: Mapping[str, Any], context: _Context
) -> _Result:
    claim_ids = _list(evidence, "claim_ids")
    artifact_ids = _list(evidence, "artifact_ids")
    if not claim_ids or not artifact_ids:
        return _unverifiable("CLAIM_BINDING_EVIDENCE_MISSING", "claim binding is incomplete")
    known_claims = {claim.claim_id: claim for claim in context.claims.claims}
    unknown = set(claim_ids) - set(known_claims)
    if unknown:
        return _violation("CLAIM_UNKNOWN", f"unknown claims: {sorted(unknown)}")
    required = {
        artifact_id
        for claim_id in claim_ids
        for artifact_id in known_claims[claim_id].required_artifact_ids
    }
    if not required.issubset(set(artifact_ids)):
        return _violation("CLAIM_ARTIFACT_BINDING_INCOMPLETE", "required artifacts are not bound")
    if not set(artifact_ids).issubset(context.artifacts):
        return _violation("CLAIM_ARTIFACT_UNKNOWN", "claim references an unknown artifact")
    return _pass(refs=tuple(str(item) for item in artifact_ids))


_NON_O5_EVALUATORS: dict[str, Callable[[CandidateRecord, Mapping[str, Any], _Context], _Result]] = {
    "O1": _candidate_identity,
    "O2": _time_availability,
    "O3": _fold_role,
    "O4": _prediction_origin,
    "O6": _calibration_identity,
    "O7": _threshold_identity,
    "O8": _model_identity,
    "O9": _execution_identity,
    "O10": _artifact_integrity,
    "O11": _outcome_consumption,
    "O12": _claim_binding,
}


@dataclass(frozen=True)
class _Context:
    graph: PipelineGraph
    artifacts: dict[str, Any]
    verifications: dict[str, ArtifactVerification]
    claims: ClaimManifest


def _finding(
    record: CandidateRecord,
    obligation: Obligation,
    context: _Context,
) -> AuditFinding:
    evidence = _evidence(record, obligation)
    if isinstance(evidence, _Result):
        result = evidence
    else:
        try:
            result = _NON_O5_EVALUATORS[obligation.obligation_id](record, evidence, context)
        except Exception as exc:
            result = _Result("ERROR", "AUDIT_EVALUATION_ERROR", f"{type(exc).__name__}: {exc}")
    if result.status == "VIOLATION" and result.first_invalid_edge is None:
        result = _Result(
            status=result.status,
            error_code=result.error_code,
            detail=result.detail,
            first_invalid_edge=context.graph.first_invalid_edge(
                record.node_id,
                {record.node_id},
            ),
            evidence_refs=result.evidence_refs,
        )
    return AuditFinding(
        candidate_key=record.candidate_key,
        node_id=record.node_id,
        layer=record.layer,
        obligation_id=obligation.obligation_id,
        status=result.status,
        error_code=result.error_code,
        detail=result.detail,
        first_invalid_edge=result.first_invalid_edge,
        evidence_refs=result.evidence_refs,
    )


@dataclass(frozen=True)
class _AuditPlan:
    """Evaluation mode and resource limits for a single audit run."""

    mode: str  # "full" | "budgeted" | "single_candidate"
    budget: int | None  # None = unlimited; max fully-evaluated findings
    candidate_key: str | None  # None = all candidates


def _validate_audit_inputs(
    graph: PipelineGraph,
    ledger: CandidateLedger,
    contract: ContractSpecification,
    claims: ClaimManifest,
    evidence_keys: set[str],
) -> None:
    validate_obligation_ids(
        contract,
        (item for claim in claims.claims for item in claim.required_obligation_ids),
    )
    unknown_evidence = sorted(
        {key for record in ledger.records for key in record.evidence if key not in evidence_keys}
    )
    if unknown_evidence:
        raise ValueError(f"ledger contains unknown evidence namespace(s): {unknown_evidence}.")
    graph_nodes = {node.node_id for node in graph.nodes}
    unknown_nodes = sorted({record.node_id for record in ledger.records} - graph_nodes)
    if unknown_nodes:
        raise ValueError(f"ledger references unknown graph nodes: {unknown_nodes}.")


def _build_context(
    graph: PipelineGraph,
    registry: ArtifactManifest,
    claims: ClaimManifest,
    artifact_verifications: Sequence[ArtifactVerification],
) -> _Context:
    return _Context(
        graph=graph,
        artifacts={item.artifact_id: item for item in registry.artifacts},
        verifications={item.artifact_id: item for item in artifact_verifications},
        claims=claims,
    )


def _budget_exhausted_finding(record: CandidateRecord, obligation_id: str) -> AuditFinding:
    return AuditFinding(
        candidate_key=record.candidate_key,
        node_id=record.node_id,
        layer=record.layer,
        obligation_id=obligation_id,
        status="UNVERIFIABLE",
        error_code="AUDIT_BUDGET_EXHAUSTED",
        detail="budget exhausted; obligation not evaluated",
        first_invalid_edge=None,
        evidence_refs=(),
    )


def _evaluate_o5(
    record: CandidateRecord,
    o5_obligation: Obligation,
    context: _Context,
    graph: PipelineGraph,
    ledger: CandidateLedger,
    phase1_findings: dict[tuple[str, str, str], AuditFinding],
) -> AuditFinding:
    evidence = _evidence(record, o5_obligation)
    if isinstance(evidence, _Result):
        result = evidence
    else:
        try:
            result = _evaluate_o5_closure(record, evidence, graph, ledger, phase1_findings)
        except Exception as exc:
            result = _Result("ERROR", "AUDIT_EVALUATION_ERROR", f"{type(exc).__name__}: {exc}")
    return AuditFinding(
        candidate_key=record.candidate_key,
        node_id=record.node_id,
        layer=record.layer,
        obligation_id="O5",
        status=result.status,
        error_code=result.error_code,
        detail=result.detail,
        first_invalid_edge=result.first_invalid_edge,
        evidence_refs=result.evidence_refs,
    )


def _evaluate_record(
    record: CandidateRecord,
    contract: ContractSpecification,
    context: _Context,
    graph: PipelineGraph,
    ledger: CandidateLedger,
    phase1_findings: dict[tuple[str, str, str], AuditFinding],
    o5_obligation: Obligation,
    budget_remaining: int,
) -> tuple[list[AuditFinding], int]:
    """Evaluate all obligations for one record under a budget.

    Returns the per-record findings and the updated remaining budget
    (``-1`` means unlimited).
    """
    findings: list[AuditFinding] = []
    unlimited = budget_remaining < 0
    for obligation in contract.obligations:
        if obligation.obligation_id == "O5":
            continue
        if not unlimited and budget_remaining == 0:
            findings.append(_budget_exhausted_finding(record, obligation.obligation_id))
        else:
            finding = _finding(record, obligation, context)
            phase1_findings[(record.candidate_key, record.node_id, obligation.obligation_id)] = (
                finding
            )
            findings.append(finding)
            if not unlimited:
                budget_remaining -= 1
    # Phase 2: O5 ancestor closure (ancestors are already in phase1_findings
    # because records are iterated in topological order within a candidate).
    if not unlimited and budget_remaining == 0:
        findings.append(_budget_exhausted_finding(record, "O5"))
    else:
        findings.append(
            _evaluate_o5(record, o5_obligation, context, graph, ledger, phase1_findings)
        )
        if not unlimited:
            budget_remaining -= 1
    return findings, budget_remaining


def _build_execution_summary(
    plan: _AuditPlan,
    findings: tuple[AuditFinding, ...],
    scoped_candidate_nodes: set[tuple[str, str]],
    expected_obligation_count: int,
    registry: ArtifactManifest,
) -> AuditExecutionSummary:
    budget_code = "AUDIT_BUDGET_EXHAUSTED"
    completed_by_pair = {pair: 0 for pair in scoped_candidate_nodes}
    for finding in findings:
        if finding.error_code != budget_code:
            pair = (finding.candidate_key, finding.node_id)
            completed_by_pair[pair] = completed_by_pair.get(pair, 0) + 1

    fully_audited_pairs = {
        pair for pair, count in completed_by_pair.items() if count == expected_obligation_count
    }
    partially_audited_pairs = {
        pair for pair, count in completed_by_pair.items() if 0 < count < expected_obligation_count
    }
    unaudited_pairs = {pair for pair, count in completed_by_pair.items() if count == 0}
    incomplete_pairs = partially_audited_pairs | unaudited_pairs
    scoped_candidates = {candidate for candidate, _node in scoped_candidate_nodes}
    scoped_nodes = {node for _candidate, node in scoped_candidate_nodes}
    unaudited_keys = {
        candidate
        for candidate in scoped_candidates
        if any(pair[0] == candidate for pair in incomplete_pairs)
    }
    audited_keys = scoped_candidates - unaudited_keys
    unaudited_nodes = {
        node for node in scoped_nodes if any(pair[1] == node for pair in incomplete_pairs)
    }
    audited_nodes = scoped_nodes - unaudited_nodes
    evaluated_count = sum(completed_by_pair.values())
    stop_reason: str | None
    if plan.mode == "budgeted" and incomplete_pairs:
        stop_reason = "budget_exhausted"
    elif plan.mode == "single_candidate":
        stop_reason = "candidate_filtered"
    else:
        stop_reason = None
    artifact_ids = tuple(sorted({item.artifact_id for item in registry.artifacts}))
    return AuditExecutionSummary(
        audit_mode=plan.mode,
        audited_candidate_count=len(audited_keys),
        audited_obligation_count=evaluated_count,
        audited_candidate_keys=tuple(sorted(audited_keys)),
        audited_node_ids=tuple(sorted(audited_nodes)),
        fully_audited_candidate_nodes=tuple(sorted(fully_audited_pairs)),
        partially_audited_candidate_nodes=tuple(sorted(partially_audited_pairs)),
        unaudited_candidate_nodes=tuple(sorted(unaudited_pairs)),
        unaudited_candidate_keys=tuple(sorted(unaudited_keys)),
        unaudited_node_ids=tuple(sorted(unaudited_nodes)),
        stop_reason=stop_reason,
        audit_time=None,
        artifact_ids=artifact_ids,
    )


def _scoped_ledger_for(
    plan: _AuditPlan,
    ledger: CandidateLedger,
    original_candidate_keys: set[str],
) -> CandidateLedger:
    """Return the ledger scoped to a single candidate (or unchanged)."""
    if plan.candidate_key is None:
        return ledger
    if plan.candidate_key not in original_candidate_keys:
        raise ValueError(f"candidate_key not found in ledger: {plan.candidate_key!r}.")
    filtered = tuple(
        record for record in ledger.records if record.candidate_key == plan.candidate_key
    )
    return CandidateLedger(schema_version=ledger.schema_version, records=filtered)


def _obligation_ranks(contract: ContractSpecification) -> dict[str, int]:
    return {obligation.obligation_id: idx for idx, obligation in enumerate(contract.obligations)}


def _not_applicable_obligations(findings: tuple[AuditFinding, ...]) -> tuple[str, ...]:
    return tuple(
        sorted(
            {
                finding.obligation_id
                for finding in findings
                if finding.status in {"NOT_APPLICABLE", "TOOL_UNSUPPORTED"}
            }
        )
    )


def _claims_for_plan(plan: _AuditPlan, claims: ClaimManifest) -> ClaimManifest:
    """Return the claim manifest visible to the selected audit mode.

    A single-candidate trace is an intentional projection, not an incomplete
    full scan. Global claims and claims that explicitly include the target are
    projected to that candidate. Claims scoped only to other candidates are
    outside the trace and therefore do not produce claim decisions.
    """
    if plan.candidate_key is None:
        return claims
    target = plan.candidate_key
    projected: list[ClaimSpecification] = []
    for claim in claims.claims:
        candidate_ids = claim.scope.candidate_ids
        if candidate_ids is not None and target not in candidate_ids:
            continue
        projected.append(
            ClaimSpecification(
                claim_id=claim.claim_id,
                required_obligation_ids=claim.required_obligation_ids,
                scope=ClaimScope(
                    candidate_ids=(target,),
                    node_ids=claim.scope.node_ids,
                ),
                required_artifact_ids=claim.required_artifact_ids,
            )
        )
    return ClaimManifest(
        claims=tuple(projected),
        schema_version=claims.schema_version,
    )


def _run_audit(
    graph: PipelineGraph,
    ledger: CandidateLedger,
    registry: ArtifactManifest,
    contract: ContractSpecification,
    *,
    claim_manifest: ClaimManifest | Mapping[str, Any],
    artifact_verifications: Sequence[ArtifactVerification] = (),
    plan: _AuditPlan,
) -> AuditReport:
    """Shared core for full, budgeted, and single-candidate audit modes."""
    claims = (
        claim_manifest
        if isinstance(claim_manifest, ClaimManifest)
        else ClaimManifest.from_dict(claim_manifest)
    )
    evidence_keys = {item.evidence_key for item in contract.obligations}
    _validate_audit_inputs(graph, ledger, contract, claims, evidence_keys)
    context = _build_context(graph, registry, claims, artifact_verifications)
    original_candidate_keys = {record.candidate_key for record in ledger.records}
    original_node_ids = {node.node_id for node in graph.nodes}
    scoped_ledger = _scoped_ledger_for(plan, ledger, original_candidate_keys)
    scoped_candidate_keys = {record.candidate_key for record in scoped_ledger.records}
    scoped_candidate_nodes = {
        (record.candidate_key, record.node_id) for record in scoped_ledger.records
    }

    node_rank = {nid: idx for idx, nid in enumerate(graph.topological_order())}
    obligation_rank = _obligation_ranks(contract)
    sorted_records = sorted(
        scoped_ledger.records,
        key=lambda item: (item.candidate_key, node_rank[item.node_id], item.layer, item.node_id),
    )
    o5_obligation = contract.by_id("O5")
    phase1_findings: dict[tuple[str, str, str], AuditFinding] = {}
    all_findings: list[AuditFinding] = []
    budget_remaining = -1 if plan.budget is None else plan.budget
    for record in sorted_records:
        record_findings, budget_remaining = _evaluate_record(
            record,
            contract,
            context,
            graph,
            scoped_ledger,
            phase1_findings,
            o5_obligation,
            budget_remaining,
        )
        all_findings.extend(record_findings)

    all_findings.sort(
        key=lambda finding: (
            finding.candidate_key,
            node_rank[finding.node_id],
            obligation_rank[finding.obligation_id],
        )
    )
    findings = tuple(all_findings)
    evaluation_claims = _claims_for_plan(plan, claims)
    decisions, _resolved = evaluate_claims(
        evaluation_claims,
        findings,
        registry,
        obligation_order=obligation_rank,
        known_candidate_ids=scoped_candidate_keys,
        known_node_ids=original_node_ids,
    )
    not_applicable = _not_applicable_obligations(findings)
    execution = _build_execution_summary(
        plan,
        findings,
        scoped_candidate_nodes,
        len(contract.obligations),
        registry,
    )
    return AuditReport(contract.version, findings, decisions, not_applicable, execution)


def audit(
    graph: PipelineGraph,
    ledger: CandidateLedger,
    registry: ArtifactManifest,
    contract: ContractSpecification,
    *,
    claim_manifest: ClaimManifest | Mapping[str, Any],
    artifact_verifications: Sequence[ArtifactVerification] = (),
) -> AuditReport:
    """Full audit: evaluate every candidate and obligation without short-circuiting.

    Phase 1 — all obligations except O5.
    Phase 2 — O5 ancestor proof closure using phase-1 results.
    """
    return _run_audit(
        graph,
        ledger,
        registry,
        contract,
        claim_manifest=claim_manifest,
        artifact_verifications=artifact_verifications,
        plan=_AuditPlan("full", None, None),
    )


def audit_budgeted(
    graph: PipelineGraph,
    ledger: CandidateLedger,
    registry: ArtifactManifest,
    contract: ContractSpecification,
    *,
    claim_manifest: ClaimManifest | Mapping[str, Any],
    artifact_verifications: Sequence[ArtifactVerification] = (),
    budget: int,
) -> AuditReport:
    """Resource-limited audit that stops after ``budget`` fully-evaluated findings.

    Pairs not evaluated due to budget exhaustion are reported as
    ``UNVERIFIABLE / AUDIT_BUDGET_EXHAUSTED`` and listed in the execution
    summary's ``unaudited_candidate_keys`` with ``stop_reason="budget_exhausted"``.
    """
    if isinstance(budget, bool) or not isinstance(budget, int) or budget < 0:
        raise ValueError("budget must be a non-negative integer.")
    return _run_audit(
        graph,
        ledger,
        registry,
        contract,
        claim_manifest=claim_manifest,
        artifact_verifications=artifact_verifications,
        plan=_AuditPlan("budgeted", budget, None),
    )


def audit_candidate(
    graph: PipelineGraph,
    ledger: CandidateLedger,
    registry: ArtifactManifest,
    contract: ContractSpecification,
    *,
    claim_manifest: ClaimManifest | Mapping[str, Any],
    artifact_verifications: Sequence[ArtifactVerification] = (),
    candidate_key: str,
) -> AuditReport:
    """Single-candidate trace: audit one candidate's complete obligation chain.

    Filters the ledger to the target candidate and runs the full obligation
    evaluation (including O5 ancestor closure) for that candidate only. The
    execution summary reports ``audit_mode="single_candidate"`` and
    ``stop_reason="candidate_filtered"`` with the remaining candidates listed
    as ``unaudited_candidate_keys``.
    """
    if not isinstance(candidate_key, str) or not candidate_key:
        raise ValueError("candidate_key must be a non-empty string.")
    return _run_audit(
        graph,
        ledger,
        registry,
        contract,
        claim_manifest=claim_manifest,
        artifact_verifications=artifact_verifications,
        plan=_AuditPlan("single_candidate", None, candidate_key),
    )


def audit_scientific(
    graph: PipelineGraph,
    observations: Sequence[CanonicalObservation],
    claims: Sequence[ScientificClaim],
) -> ScientificAuditReport:
    """Run the G5.2 native-fact observation audit and selective claim gate."""
    known_nodes = {node.node_id for node in graph.nodes}
    unknown_nodes = sorted({observation.node_id for observation in observations} - known_nodes)
    if unknown_nodes:
        raise ValueError(f"observations reference unknown graph nodes: {unknown_nodes}.")
    coordinates = [(item.candidate_id, item.node_id, item.obligation_id) for item in observations]
    if len(coordinates) != len(set(coordinates)):
        raise ValueError("duplicate canonical observation coordinates are forbidden.")

    decisions = evaluate_scientific_claims(claims, observations)
    findings: list[ScientificAuditFinding] = []
    for observation in sorted(observations, key=lambda item: item.observation_id):
        affected = tuple(
            decision
            for decision in decisions
            if observation.observation_id in decision.blocking_finding_ids
        )
        actions = tuple(
            ClaimAction(
                claim_id=decision.claim_id,
                status=decision.status,
                reason_codes=decision.reason_codes,
            )
            for decision in affected
        )
        edge = None
        if observation.status == "VIOLATION":
            edge = observation.declared_first_invalid_edge
            if edge is None:
                edge = graph.first_invalid_edge_object(
                    observation.node_id,
                    set(observation.invalid_node_ids),
                )
        findings.append(
            ScientificAuditFinding(
                finding_id=observation.observation_id,
                status=observation.status,
                obligation_id=observation.obligation_id,
                primary_layer=observation.primary_layer,
                affected_layers=observation.affected_layers,
                first_invalid_edge=edge,
                candidate_witness_ids=observation.candidate_witness_ids,
                affected_claim_ids=tuple(item.claim_id for item in affected),
                claim_actions=actions,
                evidence_refs=observation.evidence_refs,
                reason_codes=observation.reason_codes,
                missing_required_evidence=observation.missing_required_evidence,
            )
        )
    statuses = {item.status for item in findings}
    if "VIOLATION" in statuses:
        status = "VIOLATION"
    elif "UNVERIFIABLE" in statuses:
        status = "UNVERIFIABLE"
    else:
        status = "PASS"
    return ScientificAuditReport(
        status=status,
        findings=tuple(findings),
        claim_decisions=decisions,
    )


def _string_list(payload: Mapping[str, Any], key: str) -> tuple[str, ...] | None:
    value = payload.get(key)
    if not isinstance(value, list) or any(not isinstance(item, str) or not item for item in value):
        return None
    return tuple(sorted(set(value)))


def _isolation_observation(
    fact: CanonicalEvidenceFact,
    subject_key: str,
    missing_reason: str,
    violation_reason: str,
) -> tuple[str, tuple[str, ...], tuple[str, ...]]:
    assessment = _string_list(fact.payload, "assessment_source_ids")
    subject = _string_list(fact.payload, subject_key)
    missing = []
    if assessment is None:
        missing.append("assessment_source_ids")
    if subject is None:
        missing.append(subject_key)
    if missing:
        reason = missing_reason if subject_key in missing else "MISSING_ASSESSMENT_SOURCE"
        return "UNVERIFIABLE", (reason,), tuple(missing)
    if set(assessment).intersection(subject):
        return "VIOLATION", (violation_reason,), ()
    return "PASS", ("OBLIGATION_SATISFIED",), ()


_ISOLATION_FIELDS = {
    "ASSESSMENT_FIT_ISOLATION": (
        "fit_source_ids",
        "MISSING_FIT_SOURCE",
        "ASSESSMENT_ROWS_CONSUMED_BY_FIT",
    ),
    "ASSESSMENT_CALIBRATION_ISOLATION": (
        "calibration_source_ids",
        "MISSING_CALIBRATION_SOURCE",
        "ASSESSMENT_ROWS_CONSUMED_BY_CALIBRATION",
    ),
    "ASSESSMENT_THRESHOLD_ISOLATION": (
        "threshold_source_ids",
        "MISSING_THRESHOLD_SOURCE",
        "ASSESSMENT_ROWS_CONSUMED_BY_THRESHOLD",
    ),
    "ASSESSMENT_SEARCH_ISOLATION": (
        "search_source_ids",
        "MISSING_SEARCH_SOURCE",
        "ASSESSMENT_ROWS_CONSUMED_BY_SEARCH",
    ),
    "ASSESSMENT_SELECTION_ISOLATION": (
        "selection_source_ids",
        "MISSING_SELECTION_SOURCE",
        "ASSESSMENT_ROWS_CONSUMED_BY_SELECTION",
    ),
}


def _evaluate_canonical_fact(
    fact: CanonicalEvidenceFact,
) -> tuple[str, tuple[str, ...], tuple[str, ...]]:
    explicit_missing = _string_list(fact.payload, "missing_required_evidence")
    if explicit_missing:
        return "UNVERIFIABLE", ("EVIDENCE_INCOMPLETE",), explicit_missing
    if "expected" in fact.payload or "observed" in fact.payload:
        missing = tuple(field for field in ("expected", "observed") if field not in fact.payload)
        if missing:
            return (
                "UNVERIFIABLE",
                (f"MISSING_{fact.obligation_id}_EVIDENCE",),
                missing,
            )
        if fact.payload["expected"] != fact.payload["observed"]:
            return "VIOLATION", (f"{fact.obligation_id}_MISMATCH",), ()
        return "PASS", ("OBLIGATION_SATISFIED",), ()
    if fact.obligation_id in _ISOLATION_FIELDS:
        return _isolation_observation(fact, *_ISOLATION_FIELDS[fact.obligation_id])
    if fact.obligation_id == "MODEL_IDENTITY":
        fields = ("expected_identity", "observed_identity")
        missing = tuple(field for field in fields if field not in fact.payload)
        if missing:
            return "UNVERIFIABLE", ("MISSING_MODEL_IDENTITY",), missing
        if fact.payload[fields[0]] != fact.payload[fields[1]]:
            return "VIOLATION", ("MODEL_IDENTITY_MISMATCH",), ()
        return "PASS", ("OBLIGATION_SATISFIED",), ()
    if fact.obligation_id == "ARTIFACT_IDENTITY":
        fields = ("manifest_sha256", "actual_sha256")
        missing = tuple(field for field in fields if field not in fact.payload)
        if missing:
            return "UNVERIFIABLE", ("MISSING_ARTIFACT_HASH",), missing
        if fact.payload[fields[0]] != fact.payload[fields[1]]:
            return "VIOLATION", ("ARTIFACT_HASH_MISMATCH",), ()
        return "PASS", ("OBLIGATION_SATISFIED",), ()
    fields = ("expected", "observed")
    missing = tuple(field for field in fields if field not in fact.payload)
    if missing:
        return (
            "UNVERIFIABLE",
            (f"MISSING_{fact.obligation_id}_EVIDENCE",),
            missing,
        )
    if fact.payload[fields[0]] != fact.payload[fields[1]]:
        return "VIOLATION", (f"{fact.obligation_id}_MISMATCH",), ()
    return "PASS", ("OBLIGATION_SATISFIED",), ()


def observe_canonical_facts(
    facts: Sequence[CanonicalEvidenceFact],
) -> tuple[CanonicalObservation, ...]:
    """Evaluate traceable facts without accepting adapter-authored statuses."""
    coordinates = [(item.candidate_id, item.node_id, item.obligation_id) for item in facts]
    if len(coordinates) != len(set(coordinates)):
        raise ValueError("duplicate canonical fact coordinates are forbidden.")
    observations: list[CanonicalObservation] = []
    for fact in sorted(facts, key=lambda item: item.fact_id):
        status, reasons, missing = _evaluate_canonical_fact(fact)
        invalid_nodes = _string_list(fact.payload, "invalid_node_ids") or ()
        witnesses = _string_list(fact.payload, "candidate_witness_ids")
        if witnesses is None:
            witnesses = (fact.candidate_id,) if status == "VIOLATION" else ()
        observations.append(
            CanonicalObservation(
                observation_id=fact.fact_id,
                candidate_id=fact.candidate_id,
                node_id=fact.node_id,
                obligation_id=fact.obligation_id,
                status=status,
                primary_layer=fact.primary_layer,
                affected_layers=fact.affected_layers,
                invalid_node_ids=invalid_nodes,
                candidate_witness_ids=witnesses,
                evidence_refs=(fact.evidence_ref,),
                reason_codes=reasons,
                missing_required_evidence=missing,
                declared_first_invalid_edge=fact.declared_first_invalid_edge,
            )
        )
    return tuple(observations)


def audit_scientific_facts(
    graph: PipelineGraph,
    facts: Sequence[CanonicalEvidenceFact],
    claims: Sequence[ScientificClaim],
) -> ScientificAuditReport:
    """Run native facts through canonical evaluation, audit, and claim gating."""
    return audit_scientific(graph, observe_canonical_facts(facts), claims)


__all__ = [
    "audit",
    "audit_budgeted",
    "audit_candidate",
    "audit_scientific",
    "audit_scientific_facts",
    "observe_canonical_facts",
]
