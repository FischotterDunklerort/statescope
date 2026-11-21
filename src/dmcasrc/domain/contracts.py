"""Frozen proof obligations and machine-readable audit result objects."""

from __future__ import annotations

import copy
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from dmcasrc.domain.graph import CanonicalEdge

CONTRACT_VERSION = "dmcasrc.contract.initial"
CLAIM_MANIFEST_SCHEMA_VERSION = "dmcasrc.claim-manifest.initial"
CONTRACT_OBLIGATION_IDS = frozenset(f"O{i}" for i in range(1, 13))
ALLOWED_PREDICTION_ORIGINS = frozenset(
    {
        "oof",
        "rolling_oof",
        "cross_fitted",
    }
)
AUDIT_STATUSES = frozenset(
    {
        "PASS",
        "VIOLATION",
        "UNVERIFIABLE",
        "NOT_APPLICABLE",
        "TOOL_UNSUPPORTED",
        "ERROR",
    }
)

CLAIM_DECISION_STATUSES = frozenset(
    {
        "ALLOWED",
        "RESTRICTED",
        "BLOCKED",
    }
)

CAPABILITY_STATUSES = frozenset(
    {
        "unsupported",
        "not_applicable",
    }
)


@dataclass(frozen=True)
class Obligation:
    obligation_id: str
    name: str
    evidence_key: str
    severity: str = "critical"

    def __post_init__(self) -> None:
        if not isinstance(self.obligation_id, str) or not self.obligation_id.strip():
            raise ValueError("obligation_id must be a non-empty string.")
        if self.obligation_id not in CONTRACT_OBLIGATION_IDS:
            raise ValueError(f"obligation_id must be one of O1–O12, got: {self.obligation_id!r}.")
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("obligation name must be a non-empty string.")
        if not isinstance(self.evidence_key, str) or not self.evidence_key.strip():
            raise ValueError("evidence_key must be a non-empty string.")
        if self.severity != "critical":
            raise ValueError(f"severity must be 'critical', got {self.severity!r}.")
        object.__setattr__(self, "obligation_id", self.obligation_id.strip())
        object.__setattr__(self, "name", self.name.strip())
        object.__setattr__(self, "evidence_key", self.evidence_key.strip())

    def to_dict(self) -> dict[str, Any]:
        return {
            "obligation_id": self.obligation_id,
            "name": self.name,
            "evidence_key": self.evidence_key,
            "severity": self.severity,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Obligation:
        if not isinstance(payload, Mapping):
            raise ValueError("obligation must be an object.")
        allowed = {"obligation_id", "name", "evidence_key", "severity"}
        unknown = sorted({str(k) for k in payload} - allowed)
        if unknown:
            raise ValueError(f"obligation has unknown field(s): {unknown}.")
        required = {"obligation_id", "name", "evidence_key", "severity"}
        missing = sorted(required - {str(k) for k in payload})
        if missing:
            raise ValueError(f"obligation is missing field(s): {missing}.")
        oid = payload["obligation_id"]
        if not isinstance(oid, str) or not oid.strip():
            raise ValueError("obligation_id must be a non-empty string.")
        name = payload["name"]
        if not isinstance(name, str) or not name.strip():
            raise ValueError("obligation name must be a non-empty string.")
        ek = payload["evidence_key"]
        if not isinstance(ek, str) or not ek.strip():
            raise ValueError("evidence_key must be a non-empty string.")
        sev = payload["severity"]
        if sev != "critical":
            raise ValueError(f"severity must be 'critical', got {sev!r}.")
        return cls(
            obligation_id=oid.strip(), name=name.strip(), evidence_key=ek.strip(), severity=sev
        )


_DEFAULT_OBLIGATIONS = (
    ("O1", "candidate_identity", "candidate_identity"),
    ("O2", "time_availability", "time_availability"),
    ("O3", "fold_role", "fold_role"),
    ("O4", "prediction_origin", "prediction_origin"),
    ("O5", "ancestor_closure", "ancestor_closure"),
    ("O6", "calibration_identity", "calibration_identity"),
    ("O7", "threshold_identity", "threshold_identity"),
    ("O8", "model_identity", "model_identity"),
    ("O9", "execution_identity", "execution_identity"),
    ("O10", "artifact_integrity", "artifact_integrity"),
    ("O11", "outcome_consumption", "outcome_consumption"),
    ("O12", "claim_binding", "claim_binding"),
)


@dataclass(frozen=True)
class ContractSpecification:
    version: str
    obligations: tuple[Obligation, ...]
    unknown_evidence_policy: str = "fail_closed"

    @classmethod
    def default(cls) -> ContractSpecification:
        return cls(
            version=CONTRACT_VERSION,
            obligations=tuple(Obligation(*values) for values in _DEFAULT_OBLIGATIONS),
        )

    def __post_init__(self) -> None:
        if self.version != CONTRACT_VERSION:
            raise ValueError(f"unsupported contract version: {self.version}.")
        if self.unknown_evidence_policy != "fail_closed":
            raise ValueError("unknown_evidence_policy must be 'fail_closed'.")
        oids = [o.obligation_id for o in self.obligations]
        if len(oids) != len(set(oids)):
            raise ValueError("duplicate obligation_id values are forbidden.")
        names = [o.name for o in self.obligations]
        if len(names) != len(set(names)):
            raise ValueError("duplicate obligation names are forbidden.")
        eks = [o.evidence_key for o in self.obligations]
        if len(eks) != len(set(eks)):
            raise ValueError("duplicate evidence_key values are forbidden.")
        expected = tuple(f"O{i}" for i in range(1, 13))
        if tuple(oids) != expected:
            raise ValueError(f"obligations must be O1–O12 in order, got: {oids}.")

    def by_id(self, obligation_id: str) -> Obligation:
        for obligation in self.obligations:
            if obligation.obligation_id == obligation_id:
                return obligation
        raise KeyError(obligation_id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract_version": self.version,
            "obligations": [o.to_dict() for o in self.obligations],
            "unknown_evidence_policy": self.unknown_evidence_policy,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ContractSpecification:
        if not isinstance(payload, Mapping):
            raise ValueError("contract specification must be an object.")
        allowed = {"contract_version", "obligations", "unknown_evidence_policy"}
        unknown = sorted({str(k) for k in payload} - allowed)
        if unknown:
            raise ValueError(f"contract specification has unknown field(s): {unknown}.")
        required = {"contract_version", "obligations"}
        missing = sorted(required - {str(k) for k in payload})
        if missing:
            raise ValueError(f"contract specification is missing field(s): {missing}.")
        version = payload["contract_version"]
        if not isinstance(version, str) or not version:
            raise ValueError("contract_version must be a non-empty string.")
        policy = payload.get("unknown_evidence_policy", "fail_closed")
        if not isinstance(policy, str):
            raise ValueError("unknown_evidence_policy must be a string.")
        raw_obligations = payload["obligations"]
        if not isinstance(raw_obligations, list):
            raise ValueError("obligations must be a list.")
        obligations = tuple(Obligation.from_dict(item) for item in raw_obligations)
        return cls(version=version, obligations=obligations, unknown_evidence_policy=policy)


@dataclass(frozen=True)
class ClaimScope:
    """Candidate and node scope for a claim.

    Semantics:
    - ``candidate_ids is None`` → all candidates seen by the audit
    - ``candidate_ids`` non-empty tuple → only those candidates
    - Empty tuple ``()`` → illegal (fails closed)
    Same rules apply to ``node_ids``.
    """

    candidate_ids: tuple[str, ...] | None = None
    node_ids: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        for name, value in (
            ("scope.candidate_ids", self.candidate_ids),
            ("scope.node_ids", self.node_ids),
        ):
            if value is None:
                continue
            if not isinstance(value, tuple):
                raise ValueError(f"{name} must be None or a tuple of non-empty strings.")
            if not value:
                raise ValueError(f"{name} must not be empty (use None for full coverage).")
            if any(not isinstance(item, str) or not item for item in value):
                raise ValueError(f"{name} must contain only non-empty strings.")
            if len(value) != len(set(value)):
                raise ValueError(f"{name} must not contain duplicates.")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any] | None) -> ClaimScope:
        if payload is None:
            return cls()
        if not isinstance(payload, Mapping):
            raise ValueError("claim scope must be an object.")
        allowed = {"candidate_ids", "node_ids"}
        unknown = sorted({str(key) for key in payload} - allowed)
        if unknown:
            raise ValueError(f"claim scope has unknown field(s): {unknown}.")
        candidate_ids = _optional_id_tuple("scope.candidate_ids", payload.get("candidate_ids"))
        node_ids = _optional_id_tuple("scope.node_ids", payload.get("node_ids"))
        return cls(candidate_ids=candidate_ids, node_ids=node_ids)

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {}
        if self.candidate_ids is not None:
            result["candidate_ids"] = list(self.candidate_ids)
        if self.node_ids is not None:
            result["node_ids"] = list(self.node_ids)
        return result


@dataclass(frozen=True)
class ClaimSpecification:
    """Single claim with strict scope semantics.

    - ``required_obligation_ids``: non-empty, unique, stably sorted.
    - ``scope``: see ``ClaimScope``. May be omitted for full coverage.
    - ``required_artifact_ids``: optional, non-empty, unique, stably sorted.
    """

    claim_id: str
    required_obligation_ids: tuple[str, ...]
    scope: ClaimScope
    required_artifact_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.claim_id, str) or not self.claim_id:
            raise ValueError("claim_id must be a non-empty string.")
        if not isinstance(self.required_obligation_ids, tuple):
            raise ValueError("required_obligation_ids must be a tuple of non-empty strings.")
        if not self.required_obligation_ids:
            raise ValueError("required_obligation_ids must not be empty.")
        if any(not isinstance(item, str) or not item for item in self.required_obligation_ids):
            raise ValueError("required_obligation_ids must contain only non-empty strings.")
        if len(self.required_obligation_ids) != len(set(self.required_obligation_ids)):
            raise ValueError("required_obligation_ids must not contain duplicates.")
        unknown_obligations = sorted(set(self.required_obligation_ids) - CONTRACT_OBLIGATION_IDS)
        if unknown_obligations:
            raise ValueError(
                f"required_obligation_ids must be O1–O12, unknown: {unknown_obligations}."
            )
        if not isinstance(self.scope, ClaimScope):
            raise ValueError("scope must be a ClaimScope instance.")
        if not isinstance(self.required_artifact_ids, tuple):
            raise ValueError("required_artifact_ids must be a tuple of non-empty strings.")
        if any(not isinstance(item, str) or not item for item in self.required_artifact_ids):
            raise ValueError("required_artifact_ids must contain only non-empty strings.")
        if len(self.required_artifact_ids) != len(set(self.required_artifact_ids)):
            raise ValueError("required_artifact_ids must not contain duplicates.")

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "claim_id": self.claim_id,
            "required_obligation_ids": list(self.required_obligation_ids),
            "scope": self.scope.to_dict(),
        }
        if self.required_artifact_ids:
            result["required_artifact_ids"] = list(self.required_artifact_ids)
        return result


@dataclass(frozen=True)
class ClaimManifest:
    claims: tuple[ClaimSpecification, ...]
    schema_version: str = CLAIM_MANIFEST_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != CLAIM_MANIFEST_SCHEMA_VERSION:
            raise ValueError(f"unsupported claim manifest schema: {self.schema_version}.")
        if not isinstance(self.claims, tuple):
            raise ValueError("claims must be a tuple of ClaimSpecification instances.")
        for item in self.claims:
            if not isinstance(item, ClaimSpecification):
                raise ValueError("claims must contain ClaimSpecification instances.")
        ids = [claim.claim_id for claim in self.claims]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate claim_id values are forbidden.")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ClaimManifest:
        if not isinstance(payload, Mapping):
            raise ValueError("claim manifest root must be an object.")
        if {str(key) for key in payload} != {"schema_version", "claims"}:
            raise ValueError("claim manifest has missing or unknown fields.")
        if payload["schema_version"] != CLAIM_MANIFEST_SCHEMA_VERSION:
            raise ValueError(f"unsupported claim manifest schema: {payload['schema_version']}.")
        raw_claims = payload["claims"]
        if not isinstance(raw_claims, list):
            raise ValueError("claims must be a list.")
        claims = tuple(_claim_from_dict(item) for item in raw_claims)
        ids = [claim.claim_id for claim in claims]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate claim_id values are forbidden.")
        return cls(claims=claims)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "claims": [claim.to_dict() for claim in self.claims],
        }


def _optional_id_tuple(name: str, value: Any) -> tuple[str, ...] | None:
    if value is None:
        return None
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a list of non-empty strings.")
    if not value:
        raise ValueError(f"{name} must not be empty (use field omission for full coverage).")
    if any(not isinstance(item, str) or not item for item in value):
        raise ValueError(f"{name} must contain only non-empty strings.")
    if len(value) != len(set(value)):
        raise ValueError(f"{name} must not contain duplicates.")
    return tuple(value)


def _required_id_tuple(name: str, value: Any) -> tuple[str, ...]:
    result = _optional_id_tuple(name, value)
    if result is None:
        raise ValueError(f"{name} is required.")
    return result


def _string_tuple(name: str, value: Any) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(item, str) or not item for item in value):
        raise ValueError(f"{name} must be a list of non-empty strings.")
    if len(value) != len(set(value)):
        raise ValueError(f"{name} must not contain duplicates.")
    return tuple(value)


def _claim_from_dict(payload: Mapping[str, Any]) -> ClaimSpecification:
    if not isinstance(payload, Mapping):
        raise ValueError("claim must be an object.")
    allowed = {"claim_id", "required_obligation_ids", "scope", "required_artifact_ids"}
    unknown = sorted({str(key) for key in payload} - allowed)
    if unknown:
        raise ValueError(f"claim has unknown field(s): {unknown}.")
    required = {"claim_id", "required_obligation_ids"}
    missing = sorted(required - {str(key) for key in payload})
    if missing:
        raise ValueError(f"claim is missing required field(s): {missing}.")
    claim_id = payload["claim_id"]
    if not isinstance(claim_id, str) or not claim_id:
        raise ValueError("claim_id must be a non-empty string.")
    required_obligation_ids = _required_id_tuple(
        "required_obligation_ids", payload.get("required_obligation_ids")
    )
    scope = ClaimScope.from_dict(payload.get("scope"))
    required_artifact_ids = (
        _optional_id_tuple("required_artifact_ids", payload.get("required_artifact_ids")) or ()
    )
    return ClaimSpecification(
        claim_id=claim_id,
        required_obligation_ids=required_obligation_ids,
        scope=scope,
        required_artifact_ids=required_artifact_ids,
    )


@dataclass(frozen=True)
class AuditFinding:
    candidate_key: str
    node_id: str
    layer: str
    obligation_id: str
    status: str
    error_code: str | None
    detail: str
    first_invalid_edge: str | None
    evidence_refs: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.candidate_key:
            raise ValueError("candidate_key must be non-empty.")
        if not self.node_id:
            raise ValueError("node_id must be non-empty.")
        if not self.layer:
            raise ValueError("layer must be non-empty.")
        valid_oids = tuple(f"O{i}" for i in range(1, 13))
        if self.obligation_id not in valid_oids:
            raise ValueError(f"obligation_id must be O1–O12, got: {self.obligation_id}.")
        if self.status not in AUDIT_STATUSES:
            raise ValueError(f"invalid audit status: {self.status}.")
        if not isinstance(self.detail, str):
            raise ValueError("detail must be a string.")
        if self.first_invalid_edge is not None and (
            not isinstance(self.first_invalid_edge, str) or not self.first_invalid_edge
        ):
            raise ValueError("first_invalid_edge must be None or non-empty string.")
        if any(not isinstance(r, str) or not r for r in self.evidence_refs):
            raise ValueError("evidence_refs elements must be non-empty strings.")
        if len(self.evidence_refs) != len(set(self.evidence_refs)):
            raise ValueError("evidence_refs must not contain duplicates.")
        # error_code / status contract
        if self.status in ("PASS", "NOT_APPLICABLE"):
            if self.error_code is not None:
                raise ValueError(f"{self.status} finding must have error_code=null.")
        else:
            if not self.error_code or not isinstance(self.error_code, str):
                raise ValueError(f"{self.status} finding must have a non-empty error_code.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_key": self.candidate_key,
            "node_id": self.node_id,
            "layer": self.layer,
            "obligation_id": self.obligation_id,
            "status": self.status,
            "error_code": self.error_code,
            "detail": self.detail,
            "first_invalid_edge": self.first_invalid_edge,
            "evidence_refs": list(self.evidence_refs),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> AuditFinding:
        if not isinstance(payload, Mapping):
            raise ValueError("audit finding must be an object.")
        allowed = {
            "candidate_key",
            "node_id",
            "layer",
            "obligation_id",
            "status",
            "error_code",
            "detail",
            "first_invalid_edge",
            "evidence_refs",
        }
        unknown = sorted({str(k) for k in payload} - allowed)
        if unknown:
            raise ValueError(f"audit finding has unknown field(s): {unknown}.")
        # Schema-aligned required: ALL nine fields including error_code and first_invalid_edge
        required = {
            "candidate_key",
            "node_id",
            "layer",
            "obligation_id",
            "status",
            "error_code",
            "detail",
            "first_invalid_edge",
            "evidence_refs",
        }
        missing = sorted(required - {str(k) for k in payload})
        if missing:
            raise ValueError(f"audit finding is missing field(s): {missing}.")

        def _require_str(field: str) -> str:
            value = payload[field]
            if not isinstance(value, str):
                raise ValueError(
                    f"audit finding {field} must be string, got {type(value).__name__}."
                )
            return value

        def _require_str_or_null(field: str) -> str | None:
            value = payload[field]
            if value is None:
                return None
            if not isinstance(value, str):
                raise ValueError(
                    f"audit finding {field} must be null or string, got {type(value).__name__}."
                )
            return value

        candidate_key = _require_str("candidate_key")
        node_id = _require_str("node_id")
        layer = _require_str("layer")
        obligation_id = _require_str("obligation_id")
        status = _require_str("status")
        error_code = _require_str_or_null("error_code")
        detail = _require_str("detail")
        first_invalid_edge = _require_str_or_null("first_invalid_edge")

        raw_refs = payload["evidence_refs"]
        if not isinstance(raw_refs, list):
            raise ValueError("audit finding evidence_refs must be a list of strings.")
        evidence_refs = []
        for item in raw_refs:
            if not isinstance(item, str) or not item:
                raise ValueError("audit finding evidence_refs elements must be non-empty strings.")
            evidence_refs.append(item)
        evidence_refs_tuple = tuple(evidence_refs)

        return cls(
            candidate_key=candidate_key,
            node_id=node_id,
            layer=layer,
            obligation_id=obligation_id,
            status=status,
            error_code=error_code,
            detail=detail,
            first_invalid_edge=first_invalid_edge,
            evidence_refs=evidence_refs_tuple,
        )


@dataclass(frozen=True)
class ClaimDecision:
    claim_id: str
    status: str
    reason_codes: tuple[str, ...]
    required_obligation_ids: tuple[str, ...] | None = None
    resolved_candidate_count: int | None = None
    resolved_node_count: int | None = None
    evaluated_finding_count: int | None = None
    missing_pair_count: int | None = None
    blocking_finding_ids: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        if self.status not in CLAIM_DECISION_STATUSES:
            raise ValueError(f"invalid claim decision status: {self.status}.")
        if not self.reason_codes:
            raise ValueError("reason_codes must be non-empty.")
        if len(self.reason_codes) != len(set(self.reason_codes)):
            raise ValueError("reason_codes must not contain duplicates.")
        if self.required_obligation_ids is not None and len(self.required_obligation_ids) != len(
            set(self.required_obligation_ids)
        ):
            raise ValueError("required_obligation_ids must not contain duplicates.")
        for name, value in [
            ("resolved_candidate_count", self.resolved_candidate_count),
            ("resolved_node_count", self.resolved_node_count),
            ("evaluated_finding_count", self.evaluated_finding_count),
            ("missing_pair_count", self.missing_pair_count),
        ]:
            if value is not None and (not isinstance(value, int) or value < 0):
                raise ValueError(f"{name} must be a non-negative integer.")
        if self.blocking_finding_ids is not None:
            if any(not isinstance(item, str) for item in self.blocking_finding_ids):
                raise ValueError("blocking_finding_ids must be strings.")
            if len(self.blocking_finding_ids) != len(set(self.blocking_finding_ids)):
                raise ValueError("blocking_finding_ids must not contain duplicates.")

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "claim_id": self.claim_id,
            "status": self.status,
            "reason_codes": list(self.reason_codes),
        }
        if self.required_obligation_ids is not None:
            result["required_obligation_ids"] = list(self.required_obligation_ids)
        if self.resolved_candidate_count is not None:
            result["resolved_candidate_count"] = self.resolved_candidate_count
        if self.resolved_node_count is not None:
            result["resolved_node_count"] = self.resolved_node_count
        if self.evaluated_finding_count is not None:
            result["evaluated_finding_count"] = self.evaluated_finding_count
        if self.missing_pair_count is not None:
            result["missing_pair_count"] = self.missing_pair_count
        if self.blocking_finding_ids is not None:
            result["blocking_finding_ids"] = list(self.blocking_finding_ids)
        return result

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ClaimDecision:
        if not isinstance(payload, Mapping):
            raise ValueError("claim decision must be an object.")
        allowed = {
            "claim_id",
            "status",
            "reason_codes",
            "required_obligation_ids",
            "resolved_candidate_count",
            "resolved_node_count",
            "evaluated_finding_count",
            "missing_pair_count",
            "blocking_finding_ids",
        }
        unknown = sorted({str(k) for k in payload} - allowed)
        if unknown:
            raise ValueError(f"claim decision has unknown field(s): {unknown}.")
        required = {"claim_id", "status", "reason_codes"}
        missing = sorted(required - {str(k) for k in payload})
        if missing:
            raise ValueError(f"claim decision is missing field(s): {missing}.")

        def _require_str(field: str) -> str:
            value = payload[field]
            if not isinstance(value, str) or not value:
                raise ValueError(f"claim decision {field} must be a non-empty string.")
            return value

        def _require_int_field(field: str) -> int:
            value = payload[field]
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"claim decision {field} must be a non-negative integer.")
            return value

        def _optional_str_tuple(field: str):
            value = payload[field]
            if not isinstance(value, list):
                raise ValueError(f"claim decision {field} must be a list of strings.")
            result = []
            for item in value:
                if not isinstance(item, str):
                    raise ValueError(f"claim decision {field} elements must be strings.")
                result.append(item)
            return tuple(result)

        def _optional_str_list(field: str):
            return _optional_str_tuple(field)

        return cls(
            claim_id=_require_str("claim_id"),
            status=_require_str("status"),
            reason_codes=_optional_str_tuple("reason_codes"),
            required_obligation_ids=(
                _optional_str_tuple("required_obligation_ids")
                if "required_obligation_ids" in payload
                else None
            ),
            resolved_candidate_count=(
                _require_int_field("resolved_candidate_count")
                if "resolved_candidate_count" in payload
                else None
            ),
            resolved_node_count=(
                _require_int_field("resolved_node_count")
                if "resolved_node_count" in payload
                else None
            ),
            evaluated_finding_count=(
                _require_int_field("evaluated_finding_count")
                if "evaluated_finding_count" in payload
                else None
            ),
            missing_pair_count=(
                _require_int_field("missing_pair_count")
                if "missing_pair_count" in payload
                else None
            ),
            blocking_finding_ids=(
                _optional_str_list("blocking_finding_ids")
                if "blocking_finding_ids" in payload
                else None
            ),
        )


_AUDIT_MODES = frozenset({"full", "budgeted", "single_candidate"})
_STOP_REASONS = frozenset({"budget_exhausted", "candidate_filtered", "completed"})


@dataclass(frozen=True)
class AuditExecutionSummary:
    """What was audited, what was not, and why the audit stopped.

    Carries the minimum execution metadata required so that budgeted scans and
    single-candidate traces can report their audited/partially-audited/unaudited
    candidate-node work units and stop reason. ``audit_time`` is ``None`` in
    canonical reports: non-deterministic wall-clock measurements belong in a
    separate run receipt, not in the semantic report hash.
    """

    audit_mode: str
    audited_candidate_count: int
    audited_obligation_count: int
    audited_candidate_keys: tuple[str, ...]
    audited_node_ids: tuple[str, ...]
    fully_audited_candidate_nodes: tuple[tuple[str, str], ...] = ()
    partially_audited_candidate_nodes: tuple[tuple[str, str], ...] = ()
    unaudited_candidate_nodes: tuple[tuple[str, str], ...] = ()
    unaudited_candidate_keys: tuple[str, ...] = ()
    unaudited_node_ids: tuple[str, ...] = ()
    stop_reason: str | None = None
    audit_time: str | None = None
    artifact_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.audit_mode not in _AUDIT_MODES:
            raise ValueError(
                f"audit_mode must be one of {sorted(_AUDIT_MODES)}, got: {self.audit_mode!r}."
            )
        for name, value in (
            ("audited_candidate_count", self.audited_candidate_count),
            ("audited_obligation_count", self.audited_obligation_count),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer.")
        for name, value in (
            ("audited_candidate_keys", self.audited_candidate_keys),
            ("audited_node_ids", self.audited_node_ids),
            ("unaudited_candidate_keys", self.unaudited_candidate_keys),
            ("unaudited_node_ids", self.unaudited_node_ids),
            ("artifact_ids", self.artifact_ids),
        ):
            if not isinstance(value, tuple):
                raise ValueError(f"{name} must be a tuple of strings.")
            if any(not isinstance(item, str) or not item for item in value):
                raise ValueError(f"{name} elements must be non-empty strings.")
            if len(value) != len(set(value)):
                raise ValueError(f"{name} must not contain duplicates.")
        pair_groups = (
            ("fully_audited_candidate_nodes", self.fully_audited_candidate_nodes),
            ("partially_audited_candidate_nodes", self.partially_audited_candidate_nodes),
            ("unaudited_candidate_nodes", self.unaudited_candidate_nodes),
        )
        for name, value in pair_groups:
            if not isinstance(value, tuple):
                raise ValueError(f"{name} must be a tuple of candidate-node pairs.")
            for pair in value:
                if (
                    not isinstance(pair, tuple)
                    or len(pair) != 2
                    or any(not isinstance(item, str) or not item for item in pair)
                ):
                    raise ValueError(
                        f"{name} entries must be (candidate_key, node_id) string pairs."
                    )
            if len(value) != len(set(value)):
                raise ValueError(f"{name} must not contain duplicate pairs.")
            if value != tuple(sorted(value)):
                raise ValueError(f"{name} must be sorted by candidate_key and node_id.")
        classified_pairs = (
            set(self.fully_audited_candidate_nodes)
            | set(self.partially_audited_candidate_nodes)
            | set(self.unaudited_candidate_nodes)
        )
        classified_count = sum(len(value) for _, value in pair_groups)
        if len(classified_pairs) != classified_count:
            raise ValueError("candidate-node audit classifications must be disjoint.")
        if classified_pairs:
            incomplete_pairs = set(self.partially_audited_candidate_nodes) | set(
                self.unaudited_candidate_nodes
            )
            candidates = {candidate for candidate, _node in classified_pairs}
            nodes = {node for _candidate, node in classified_pairs}
            expected_unaudited_candidates = {
                candidate
                for candidate in candidates
                if any(pair[0] == candidate for pair in incomplete_pairs)
            }
            expected_audited_candidates = candidates - expected_unaudited_candidates
            expected_unaudited_nodes = {
                node for node in nodes if any(pair[1] == node for pair in incomplete_pairs)
            }
            expected_audited_nodes = nodes - expected_unaudited_nodes
            if set(self.audited_candidate_keys) != expected_audited_candidates:
                raise ValueError(
                    "audited_candidate_keys must contain candidates whose pairs "
                    "are all fully audited."
                )
            if set(self.unaudited_candidate_keys) != expected_unaudited_candidates:
                raise ValueError(
                    "unaudited_candidate_keys must contain candidates with a "
                    "partial or unaudited pair."
                )
            if set(self.audited_node_ids) != expected_audited_nodes:
                raise ValueError(
                    "audited_node_ids must contain nodes whose pairs are all fully audited."
                )
            if set(self.unaudited_node_ids) != expected_unaudited_nodes:
                raise ValueError(
                    "unaudited_node_ids must contain nodes with a partial or unaudited pair."
                )
            if self.audited_candidate_count != len(expected_audited_candidates):
                raise ValueError("audited_candidate_count must match fully audited candidates.")
        if self.stop_reason is not None and self.stop_reason not in _STOP_REASONS:
            raise ValueError(
                f"stop_reason must be None or one of {sorted(_STOP_REASONS)}, "
                f"got: {self.stop_reason!r}."
            )
        if self.audit_time is not None and (
            not isinstance(self.audit_time, str) or not self.audit_time
        ):
            raise ValueError("audit_time must be null or a non-empty string.")
        if self.audit_mode == "full" and self.stop_reason not in (None, "completed"):
            raise ValueError("full audit stop_reason must be None or 'completed'.")
        if (
            self.audit_mode == "budgeted"
            and self.stop_reason != "budget_exhausted"
            and self.unaudited_candidate_keys
        ):
            raise ValueError(
                "budgeted audit with unaudited candidates must report 'budget_exhausted'."
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "audit_mode": self.audit_mode,
            "audited_candidate_count": self.audited_candidate_count,
            "audited_obligation_count": self.audited_obligation_count,
            "audited_candidate_keys": list(self.audited_candidate_keys),
            "audited_node_ids": list(self.audited_node_ids),
            "fully_audited_candidate_nodes": _candidate_node_pairs_to_dict(
                self.fully_audited_candidate_nodes
            ),
            "partially_audited_candidate_nodes": _candidate_node_pairs_to_dict(
                self.partially_audited_candidate_nodes
            ),
            "unaudited_candidate_nodes": _candidate_node_pairs_to_dict(
                self.unaudited_candidate_nodes
            ),
            "unaudited_candidate_keys": list(self.unaudited_candidate_keys),
            "unaudited_node_ids": list(self.unaudited_node_ids),
            "stop_reason": self.stop_reason,
            "audit_time": self.audit_time,
            "artifact_ids": list(self.artifact_ids),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> AuditExecutionSummary:
        if not isinstance(payload, Mapping):
            raise ValueError("audit execution summary must be an object.")
        allowed = {
            "audit_mode",
            "audited_candidate_count",
            "audited_obligation_count",
            "audited_candidate_keys",
            "audited_node_ids",
            "fully_audited_candidate_nodes",
            "partially_audited_candidate_nodes",
            "unaudited_candidate_nodes",
            "unaudited_candidate_keys",
            "unaudited_node_ids",
            "stop_reason",
            "audit_time",
            "artifact_ids",
        }
        unknown = sorted({str(k) for k in payload} - allowed)
        if unknown:
            raise ValueError(f"audit execution summary has unknown field(s): {unknown}.")
        required = {
            "audit_mode",
            "audited_candidate_count",
            "audited_obligation_count",
            "audited_candidate_keys",
            "audited_node_ids",
        }
        missing = sorted(required - {str(k) for k in payload})
        if missing:
            raise ValueError(f"audit execution summary is missing field(s): {missing}.")
        mode = _require_str_field(payload, "audit_mode")
        if mode not in _AUDIT_MODES:
            raise ValueError(f"audit_mode must be one of {sorted(_AUDIT_MODES)}, got: {mode!r}.")
        stop = _optional_str_field(payload, "stop_reason")
        if stop is not None and stop not in _STOP_REASONS:
            raise ValueError(
                f"stop_reason must be null or one of {sorted(_STOP_REASONS)}, got: {stop!r}."
            )
        audit_time_raw = payload.get("audit_time")
        if audit_time_raw is not None and (
            not isinstance(audit_time_raw, str) or not audit_time_raw
        ):
            raise ValueError("audit_time must be null or a non-empty string.")
        return cls(
            audit_mode=mode,
            audited_candidate_count=_require_int_field(payload, "audited_candidate_count"),
            audited_obligation_count=_require_int_field(payload, "audited_obligation_count"),
            audited_candidate_keys=_str_tuple_field(
                payload, "audited_candidate_keys", required=True
            ),
            audited_node_ids=_str_tuple_field(payload, "audited_node_ids", required=True),
            fully_audited_candidate_nodes=_candidate_node_pairs_field(
                payload, "fully_audited_candidate_nodes"
            ),
            partially_audited_candidate_nodes=_candidate_node_pairs_field(
                payload, "partially_audited_candidate_nodes"
            ),
            unaudited_candidate_nodes=_candidate_node_pairs_field(
                payload, "unaudited_candidate_nodes"
            ),
            unaudited_candidate_keys=_str_tuple_field(payload, "unaudited_candidate_keys"),
            unaudited_node_ids=_str_tuple_field(payload, "unaudited_node_ids"),
            stop_reason=stop,
            audit_time=audit_time_raw,
            artifact_ids=_str_tuple_field(payload, "artifact_ids"),
        )


@dataclass(frozen=True)
class AuditReport:
    contract_version: str
    findings: tuple[AuditFinding, ...]
    claim_decisions: tuple[ClaimDecision, ...]
    capability_not_applicable: tuple[str, ...] = ()
    execution: AuditExecutionSummary | None = None

    def __post_init__(self) -> None:
        if self.contract_version != CONTRACT_VERSION:
            raise ValueError(f"unsupported contract version: {self.contract_version}.")
        keys = [(f.candidate_key, f.node_id, f.obligation_id) for f in self.findings]
        if len(keys) != len(set(keys)):
            raise ValueError("duplicate (candidate_key, node_id, obligation_id) in findings.")
        claim_ids = [d.claim_id for d in self.claim_decisions]
        if len(claim_ids) != len(set(claim_ids)):
            raise ValueError("duplicate claim_id in claim_decisions.")
        if len(self.capability_not_applicable) != len(set(self.capability_not_applicable)):
            raise ValueError("duplicate entries in capability_not_applicable.")
        for d in self.claim_decisions:
            resolved_fields = [
                d.required_obligation_ids,
                d.resolved_candidate_count,
                d.resolved_node_count,
                d.evaluated_finding_count,
                d.missing_pair_count,
                d.blocking_finding_ids,
            ]
            if any(f is None for f in resolved_fields):
                raise ValueError(f"claim decision '{d.claim_id}' is missing resolved metrics.")
        if self.execution is not None:
            if self.execution.audited_obligation_count > len(self.findings):
                raise ValueError(
                    "execution.audited_obligation_count cannot exceed the number of findings."
                )
            if (
                self.execution.audit_mode == "full"
                and self.execution.audited_obligation_count != len(self.findings)
            ):
                raise ValueError(
                    "full audit must audit every finding (audited_obligation_count == findings)."
                )

    def to_dict(self) -> dict[str, Any]:
        decisions_payload: list[dict[str, Any]] = []
        for decision in self.claim_decisions:
            entry = decision.to_dict()
            decisions_payload.append(entry)
        result: dict[str, Any] = {
            "contract_version": self.contract_version,
            "findings": [finding.to_dict() for finding in self.findings],
            "claim_decisions": decisions_payload,
            "capability_not_applicable": list(self.capability_not_applicable),
        }
        if self.execution is not None:
            result["execution"] = self.execution.to_dict()
        return result

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> AuditReport:
        if not isinstance(payload, Mapping):
            raise ValueError("audit report must be an object.")
        allowed = {
            "contract_version",
            "findings",
            "claim_decisions",
            "capability_not_applicable",
            "execution",
        }
        unknown = sorted({str(k) for k in payload} - allowed)
        if unknown:
            raise ValueError(f"audit report has unknown field(s): {unknown}.")
        # Schema-aligned required: ALL FOUR fields including capability_not_applicable
        required = {"contract_version", "findings", "claim_decisions", "capability_not_applicable"}
        missing = sorted(required - {str(k) for k in payload})
        if missing:
            raise ValueError(f"audit report is missing field(s): {missing}.")
        version = payload["contract_version"]
        if not isinstance(version, str) or not version:
            raise ValueError("audit report contract_version must be a non-empty string.")
        raw_findings = payload["findings"]
        raw_decisions = payload["claim_decisions"]
        if not isinstance(raw_findings, list):
            raise ValueError("findings must be a list.")
        if not isinstance(raw_decisions, list):
            raise ValueError("claim_decisions must be a list.")
        findings = tuple(AuditFinding.from_dict(item) for item in raw_findings)
        claim_decisions = tuple(ClaimDecision.from_dict(item) for item in raw_decisions)
        cna_tuple = _parse_capability_not_applicable(payload["capability_not_applicable"])
        execution_obj: AuditExecutionSummary | None = None
        if "execution" in payload and payload["execution"] is not None:
            execution_obj = AuditExecutionSummary.from_dict(payload["execution"])
        return cls(
            contract_version=version,
            findings=findings,
            claim_decisions=claim_decisions,
            capability_not_applicable=cna_tuple,
            execution=execution_obj,
        )


def _parse_capability_not_applicable(raw: Any) -> tuple[str, ...]:
    """Strict parse of capability_not_applicable: non-empty strings, no coercion."""
    if not isinstance(raw, list):
        raise ValueError("capability_not_applicable must be a list of non-empty strings.")
    parsed: list[str] = []
    for item in raw:
        if not isinstance(item, str) or not item:
            raise ValueError(
                f"capability_not_applicable elements must be non-empty strings, "
                f"got {type(item).__name__}={item!r}."
            )
        parsed.append(item)
    seen: set[str] = set()
    dupes: list[str] = []
    for item in parsed:
        if item in seen:
            dupes.append(item)
        seen.add(item)
    if dupes:
        raise ValueError(f"capability_not_applicable contains duplicates: {sorted(set(dupes))}.")
    return tuple(parsed)


def _require_str_field(payload: Mapping[str, Any], field: str) -> str:
    value = payload[field]
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string.")
    return value


def _require_int_field(payload: Mapping[str, Any], field: str) -> int:
    value = payload[field]
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer.")
    return value


def _optional_str_field(payload: Mapping[str, Any], field: str) -> str | None:
    if field not in payload:
        return None
    value = payload[field]
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be null or a non-empty string.")
    return value


def _str_tuple_field(
    payload: Mapping[str, Any], field: str, *, required: bool = False
) -> tuple[str, ...]:
    if field not in payload:
        if required:
            raise ValueError(f"{field} is required.")
        return ()
    value = payload[field]
    if not isinstance(value, list):
        raise ValueError(f"{field} must be a list of strings.")
    result = []
    for item in value:
        if not isinstance(item, str) or not item:
            raise ValueError(f"{field} elements must be non-empty strings.")
        result.append(item)
    if len(result) != len(set(result)):
        raise ValueError(f"{field} must not contain duplicates.")
    return tuple(result)


def _candidate_node_pairs_to_dict(
    pairs: tuple[tuple[str, str], ...],
) -> list[dict[str, str]]:
    return [
        {"candidate_key": candidate_key, "node_id": node_id} for candidate_key, node_id in pairs
    ]


def _candidate_node_pairs_field(
    payload: Mapping[str, Any], field: str
) -> tuple[tuple[str, str], ...]:
    if field not in payload:
        return ()
    value = payload[field]
    if not isinstance(value, list):
        raise ValueError(f"{field} must be a list of candidate-node objects.")
    pairs: list[tuple[str, str]] = []
    for item in value:
        if not isinstance(item, Mapping) or set(item) != {"candidate_key", "node_id"}:
            raise ValueError(f"{field} entries must contain only candidate_key and node_id.")
        candidate_key = item["candidate_key"]
        node_id = item["node_id"]
        if (
            not isinstance(candidate_key, str)
            or not candidate_key
            or not isinstance(node_id, str)
            or not node_id
        ):
            raise ValueError(f"{field} candidate_key and node_id must be non-empty strings.")
        pairs.append((candidate_key, node_id))
    return tuple(pairs)


def validate_obligation_ids(contract: ContractSpecification, values: Iterable[str]) -> None:
    known = {item.obligation_id for item in contract.obligations}
    unknown = sorted(set(values) - known)
    if unknown:
        raise ValueError(f"claim references unknown obligations: {unknown}.")


# G5.2 canonical contract.  The O1-O12 objects above remain the frozen initial
# compatibility surface; all new scientific-capability work uses this single
# named registry.
CANONICAL_CONTRACT_VERSION = "dmcasrc.contract.subsequent"
CANONICAL_AUDIT_STATUSES = frozenset({"PASS", "VIOLATION", "UNVERIFIABLE"})
CANONICAL_CLAIM_STATUSES = frozenset({"ALLOWED", "BLOCKED", "UNVERIFIABLE"})
CANONICAL_LAYER_IDS = (
    "Data",
    "DataDistribution",
    "Fit",
    "Assessment",
    "Macro",
    "Calibration",
    "Threshold",
    "Model",
    "RL",
    "Selector",
    "Risk",
    "Timer",
    "Execution",
    "Identity",
    "Artifact",
    "Claim",
)

_CANONICAL_OBLIGATION_DEFINITIONS = (
    ("TIME_AVAILABILITY", "time_availability", "Data", ("Data",)),
    ("TIME_ALIGNMENT", "time_alignment", "Data", ("Data", "Assessment")),
    ("LABEL_TIME_ALIGNMENT", "label_time_alignment", "Data", ("Data", "Assessment")),
    ("UNIVERSE_AVAILABILITY", "universe_availability", "DataDistribution", ("DataDistribution",)),
    ("ASSESSMENT_FIT_ISOLATION", "assessment_fit_isolation", "Fit", ("Fit", "Assessment")),
    (
        "ASSESSMENT_CALIBRATION_ISOLATION",
        "assessment_calibration_isolation",
        "Calibration",
        ("Calibration", "Assessment"),
    ),
    (
        "ASSESSMENT_THRESHOLD_ISOLATION",
        "assessment_threshold_isolation",
        "Threshold",
        ("Threshold", "Assessment"),
    ),
    (
        "ASSESSMENT_SEARCH_ISOLATION",
        "assessment_search_isolation",
        "Assessment",
        ("Assessment", "Selector"),
    ),
    (
        "ASSESSMENT_SELECTION_ISOLATION",
        "assessment_selection_isolation",
        "Assessment",
        ("Assessment", "Selector"),
    ),
    ("PREDICTION_ORIGIN", "prediction_origin", "Model", ("Model", "Assessment")),
    ("ANCESTOR_CLOSURE", "ancestor_closure", "Identity", ("Identity",)),
    ("CANDIDATE_IDENTITY", "candidate_identity", "Identity", ("Identity",)),
    ("MODEL_IDENTITY", "model_identity", "Model", ("Model", "Identity")),
    ("THRESHOLD_IDENTITY", "threshold_identity", "Threshold", ("Threshold", "Identity")),
    ("EXECUTION_IDENTITY", "execution_identity", "Execution", ("Execution", "Identity")),
    ("ARTIFACT_IDENTITY", "artifact_identity", "Artifact", ("Artifact", "Identity")),
    (
        "SELECTION_EXECUTION_CONSISTENCY",
        "selection_execution_consistency",
        "Execution",
        ("Execution", "Selector"),
    ),
    (
        "OUTCOME_FEEDBACK_ISOLATION",
        "outcome_feedback_isolation",
        "Assessment",
        ("Assessment", "Selector"),
    ),
    ("CLAIM_DEPENDENCY_INTEGRITY", "claim_dependency_integrity", "Claim", ("Claim", "Artifact")),
)
CANONICAL_OBLIGATION_IDS = tuple(item[0] for item in _CANONICAL_OBLIGATION_DEFINITIONS)
CANONICAL_OBSERVATION_REASON_CODES = frozenset(
    {
        "OBLIGATION_SATISFIED",
        "EVIDENCE_INCOMPLETE",
        "MISSING_ASSESSMENT_SOURCE",
        "MISSING_FIT_SOURCE",
        "MISSING_CALIBRATION_SOURCE",
        "MISSING_THRESHOLD_SOURCE",
        "MISSING_SEARCH_SOURCE",
        "MISSING_SELECTION_SOURCE",
        "ASSESSMENT_ROWS_CONSUMED_BY_FIT",
        "ASSESSMENT_ROWS_CONSUMED_BY_CALIBRATION",
        "ASSESSMENT_ROWS_CONSUMED_BY_THRESHOLD",
        "ASSESSMENT_ROWS_CONSUMED_BY_SEARCH",
        "ASSESSMENT_ROWS_CONSUMED_BY_SELECTION",
        "MISSING_MODEL_IDENTITY",
        "MODEL_IDENTITY_MISMATCH",
        "MISSING_ARTIFACT_HASH",
        "ARTIFACT_HASH_MISMATCH",
    }
    | {f"MISSING_{item}_EVIDENCE" for item in CANONICAL_OBLIGATION_IDS}
    | {f"{item}_MISMATCH" for item in CANONICAL_OBLIGATION_IDS}
)
CANONICAL_CLAIM_REASON_CODES = frozenset(
    {
        "CLAIM_OBLIGATION_VIOLATION",
        "CLAIM_DEPENDENCY_MISSING",
        "CLAIM_DEPENDENCY_UNVERIFIABLE",
        "CLAIM_DEPENDENCIES_SATISFIED",
    }
)


def _canonical_strings(
    name: str, values: tuple[str, ...], *, empty: bool = False
) -> tuple[str, ...]:
    if not isinstance(values, tuple):
        raise ValueError(f"{name} must be a tuple.")
    if not empty and not values:
        raise ValueError(f"{name} must not be empty.")
    if any(not isinstance(item, str) or not item for item in values):
        raise ValueError(f"{name} elements must be non-empty strings.")
    if len(values) != len(set(values)):
        raise ValueError(f"{name} must not contain duplicates.")
    return values


@dataclass(frozen=True)
class CanonicalObligation:
    obligation_id: str
    evidence_key: str
    primary_layer: str
    affected_layers: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.obligation_id not in CANONICAL_OBLIGATION_IDS:
            raise ValueError(f"unknown canonical obligation: {self.obligation_id}.")
        if not isinstance(self.evidence_key, str) or not self.evidence_key:
            raise ValueError("evidence_key must be a non-empty string.")
        if self.primary_layer not in CANONICAL_LAYER_IDS:
            raise ValueError(f"unknown primary layer: {self.primary_layer}.")
        layers = _canonical_strings("affected_layers", self.affected_layers)
        if set(layers) - set(CANONICAL_LAYER_IDS):
            raise ValueError("affected_layers contains an unknown layer.")
        if self.primary_layer not in layers:
            raise ValueError("primary_layer must occur in affected_layers.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "obligation_id": self.obligation_id,
            "evidence_key": self.evidence_key,
            "primary_layer": self.primary_layer,
            "affected_layers": list(self.affected_layers),
        }


@dataclass(frozen=True)
class CanonicalContractSpecification:
    version: str
    obligations: tuple[CanonicalObligation, ...]
    unknown_evidence_policy: str = "fail_closed_with_explicit_missing_evidence"

    @classmethod
    def default(cls) -> CanonicalContractSpecification:
        return cls(
            version=CANONICAL_CONTRACT_VERSION,
            obligations=tuple(
                CanonicalObligation(*definition) for definition in _CANONICAL_OBLIGATION_DEFINITIONS
            ),
        )

    def __post_init__(self) -> None:
        if self.version != CANONICAL_CONTRACT_VERSION:
            raise ValueError(f"unsupported canonical contract version: {self.version}.")
        if self.unknown_evidence_policy != "fail_closed_with_explicit_missing_evidence":
            raise ValueError("canonical contract must fail closed with explicit missing evidence.")
        if tuple(item.obligation_id for item in self.obligations) != CANONICAL_OBLIGATION_IDS:
            raise ValueError("canonical obligations must exactly match the ordered registry.")
        keys = tuple(item.evidence_key for item in self.obligations)
        if len(keys) != len(set(keys)):
            raise ValueError("canonical evidence keys must be unique.")

    def by_id(self, obligation_id: str) -> CanonicalObligation:
        for obligation in self.obligations:
            if obligation.obligation_id == obligation_id:
                return obligation
        raise KeyError(obligation_id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract_version": self.version,
            "obligations": [item.to_dict() for item in self.obligations],
            "unknown_evidence_policy": self.unknown_evidence_policy,
        }


@dataclass(frozen=True)
class CanonicalObservation:
    observation_id: str
    candidate_id: str
    node_id: str
    obligation_id: str
    status: str
    primary_layer: str
    affected_layers: tuple[str, ...]
    invalid_node_ids: tuple[str, ...]
    candidate_witness_ids: tuple[str, ...]
    evidence_refs: tuple[str, ...]
    reason_codes: tuple[str, ...]
    missing_required_evidence: tuple[str, ...]
    declared_first_invalid_edge: CanonicalEdge | None = None

    def __post_init__(self) -> None:
        for name in ("observation_id", "candidate_id", "node_id"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"{name} must be a non-empty string.")
        if self.obligation_id not in CANONICAL_OBLIGATION_IDS:
            raise ValueError(f"unknown canonical obligation: {self.obligation_id}.")
        if self.status not in CANONICAL_AUDIT_STATUSES:
            raise ValueError(f"invalid canonical audit status: {self.status}.")
        if self.primary_layer not in CANONICAL_LAYER_IDS:
            raise ValueError(f"unknown primary layer: {self.primary_layer}.")
        for name in (
            "affected_layers",
            "invalid_node_ids",
            "candidate_witness_ids",
            "evidence_refs",
            "reason_codes",
            "missing_required_evidence",
        ):
            _canonical_strings(
                name,
                getattr(self, name),
                empty=name not in {"affected_layers", "reason_codes"},
            )
        if set(self.reason_codes) - CANONICAL_OBSERVATION_REASON_CODES:
            raise ValueError("reason_codes contains an unknown observation reason.")
        if self.primary_layer not in self.affected_layers:
            raise ValueError("primary_layer must occur in affected_layers.")
        if self.status == "UNVERIFIABLE" and not self.missing_required_evidence:
            raise ValueError("UNVERIFIABLE requires missing_required_evidence.")
        if self.status != "UNVERIFIABLE" and self.missing_required_evidence:
            raise ValueError("missing_required_evidence is reserved for UNVERIFIABLE.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "observation_id": self.observation_id,
            "candidate_id": self.candidate_id,
            "node_id": self.node_id,
            "obligation_id": self.obligation_id,
            "status": self.status,
            "primary_layer": self.primary_layer,
            "affected_layers": list(self.affected_layers),
            "invalid_node_ids": list(self.invalid_node_ids),
            "candidate_witness_ids": list(self.candidate_witness_ids),
            "evidence_refs": list(self.evidence_refs),
            "reason_codes": list(self.reason_codes),
            "missing_required_evidence": list(self.missing_required_evidence),
            "declared_first_invalid_edge": (
                None
                if self.declared_first_invalid_edge is None
                else self.declared_first_invalid_edge.to_dict()
            ),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> CanonicalObservation:
        expected = {
            "observation_id",
            "candidate_id",
            "node_id",
            "obligation_id",
            "status",
            "primary_layer",
            "affected_layers",
            "invalid_node_ids",
            "candidate_witness_ids",
            "evidence_refs",
            "reason_codes",
            "missing_required_evidence",
            "declared_first_invalid_edge",
        }
        if not isinstance(payload, Mapping) or set(payload) != expected:
            raise ValueError("canonical observation fields differ from the schema.")

        def values(name: str) -> tuple[str, ...]:
            raw = payload[name]
            if not isinstance(raw, list):
                raise ValueError(f"{name} must be a list.")
            return tuple(raw)

        return cls(
            observation_id=str(payload["observation_id"]),
            candidate_id=str(payload["candidate_id"]),
            node_id=str(payload["node_id"]),
            obligation_id=str(payload["obligation_id"]),
            status=str(payload["status"]),
            primary_layer=str(payload["primary_layer"]),
            affected_layers=values("affected_layers"),
            invalid_node_ids=values("invalid_node_ids"),
            candidate_witness_ids=values("candidate_witness_ids"),
            evidence_refs=values("evidence_refs"),
            reason_codes=values("reason_codes"),
            missing_required_evidence=values("missing_required_evidence"),
            declared_first_invalid_edge=(
                None
                if payload["declared_first_invalid_edge"] is None
                else CanonicalEdge.from_dict(payload["declared_first_invalid_edge"])
            ),
        )


@dataclass(frozen=True)
class CanonicalEvidenceFact:
    """Traceable adapter output consumed by canonical obligation evaluators."""

    fact_id: str
    candidate_id: str
    node_id: str
    obligation_id: str
    primary_layer: str
    affected_layers: tuple[str, ...]
    source_profile: str
    native_object_ref: str
    native_source_path: str
    native_fact_type: str
    canonical_identity: str
    evidence_sha256: str
    evidence_ref: str
    payload: Mapping[str, Any]
    declared_first_invalid_edge: CanonicalEdge | None = None

    def __post_init__(self) -> None:
        for name in (
            "fact_id",
            "candidate_id",
            "node_id",
            "source_profile",
            "native_object_ref",
            "native_source_path",
            "native_fact_type",
            "canonical_identity",
            "evidence_ref",
        ):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"{name} must be a non-empty string.")
        if self.obligation_id not in CANONICAL_OBLIGATION_IDS:
            raise ValueError(f"unknown canonical obligation: {self.obligation_id}.")
        if self.primary_layer not in CANONICAL_LAYER_IDS:
            raise ValueError(f"unknown primary layer: {self.primary_layer}.")
        layers = _canonical_strings("affected_layers", self.affected_layers)
        if self.primary_layer not in layers:
            raise ValueError("primary_layer must occur in affected_layers.")
        digest = self.evidence_sha256
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError("evidence_sha256 must be 64 lowercase hexadecimal characters.")
        if not isinstance(self.payload, Mapping):
            raise ValueError("payload must be an object.")
        object.__setattr__(self, "payload", copy.deepcopy(dict(self.payload)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "fact_id": self.fact_id,
            "candidate_id": self.candidate_id,
            "node_id": self.node_id,
            "obligation_id": self.obligation_id,
            "primary_layer": self.primary_layer,
            "affected_layers": list(self.affected_layers),
            "source_profile": self.source_profile,
            "native_object_ref": self.native_object_ref,
            "native_source_path": self.native_source_path,
            "native_fact_type": self.native_fact_type,
            "canonical_identity": self.canonical_identity,
            "evidence_sha256": self.evidence_sha256,
            "evidence_ref": self.evidence_ref,
            "payload": copy.deepcopy(dict(self.payload)),
            "declared_first_invalid_edge": (
                None
                if self.declared_first_invalid_edge is None
                else self.declared_first_invalid_edge.to_dict()
            ),
        }


@dataclass(frozen=True)
class ScientificClaim:
    claim_id: str
    required_obligation_ids: tuple[str, ...]
    candidate_ids: tuple[str, ...]
    node_ids: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.claim_id, str) or not self.claim_id:
            raise ValueError("claim_id must be a non-empty string.")
        obligations = _canonical_strings("required_obligation_ids", self.required_obligation_ids)
        if set(obligations) - set(CANONICAL_OBLIGATION_IDS):
            raise ValueError("claim references an unknown canonical obligation.")
        _canonical_strings("candidate_ids", self.candidate_ids)
        if self.node_ids is not None:
            _canonical_strings("node_ids", self.node_ids)


@dataclass(frozen=True)
class ScientificClaimDecision:
    claim_id: str
    status: str
    reason_codes: tuple[str, ...]
    blocking_finding_ids: tuple[str, ...] = ()
    missing_dependencies: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.claim_id, str) or not self.claim_id:
            raise ValueError("claim_id must be a non-empty string.")
        if self.status not in CANONICAL_CLAIM_STATUSES:
            raise ValueError(f"invalid canonical claim status: {self.status}.")
        _canonical_strings("reason_codes", self.reason_codes)
        if set(self.reason_codes) - CANONICAL_CLAIM_REASON_CODES:
            raise ValueError("reason_codes contains an unknown claim reason.")
        _canonical_strings("blocking_finding_ids", self.blocking_finding_ids, empty=True)
        _canonical_strings("missing_dependencies", self.missing_dependencies, empty=True)

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "status": self.status,
            "reason_codes": list(self.reason_codes),
            "blocking_finding_ids": list(self.blocking_finding_ids),
            "missing_dependencies": list(self.missing_dependencies),
        }


@dataclass(frozen=True)
class ClaimAction:
    claim_id: str
    status: str
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.claim_id, str) or not self.claim_id:
            raise ValueError("claim_id must be a non-empty string.")
        if self.status not in CANONICAL_CLAIM_STATUSES:
            raise ValueError(f"invalid canonical claim status: {self.status}.")
        _canonical_strings("reason_codes", self.reason_codes)
        if set(self.reason_codes) - CANONICAL_CLAIM_REASON_CODES:
            raise ValueError("reason_codes contains an unknown claim reason.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "status": self.status,
            "reason_codes": list(self.reason_codes),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ClaimAction:
        if not isinstance(payload, Mapping) or set(payload) != {
            "claim_id",
            "status",
            "reason_codes",
        }:
            raise ValueError("claim action fields differ from the canonical schema.")
        reasons = payload["reason_codes"]
        if not isinstance(reasons, list):
            raise ValueError("claim action reason_codes must be a list.")
        return cls(str(payload["claim_id"]), str(payload["status"]), tuple(reasons))


@dataclass(frozen=True)
class ScientificAuditFinding:
    finding_id: str
    status: str
    obligation_id: str
    primary_layer: str
    affected_layers: tuple[str, ...]
    first_invalid_edge: CanonicalEdge | None
    candidate_witness_ids: tuple[str, ...]
    affected_claim_ids: tuple[str, ...]
    claim_actions: tuple[ClaimAction, ...]
    evidence_refs: tuple[str, ...]
    reason_codes: tuple[str, ...]
    missing_required_evidence: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.finding_id, str) or not self.finding_id:
            raise ValueError("finding_id must be a non-empty string.")
        if self.status not in CANONICAL_AUDIT_STATUSES:
            raise ValueError(f"invalid canonical audit status: {self.status}.")
        if self.obligation_id not in CANONICAL_OBLIGATION_IDS:
            raise ValueError(f"unknown canonical obligation: {self.obligation_id}.")
        if self.primary_layer not in CANONICAL_LAYER_IDS:
            raise ValueError(f"unknown primary layer: {self.primary_layer}.")
        for name in (
            "affected_layers",
            "candidate_witness_ids",
            "affected_claim_ids",
            "evidence_refs",
            "reason_codes",
            "missing_required_evidence",
        ):
            _canonical_strings(
                name,
                getattr(self, name),
                empty=name not in {"affected_layers", "reason_codes"},
            )
        if set(self.reason_codes) - CANONICAL_OBSERVATION_REASON_CODES:
            raise ValueError("reason_codes contains an unknown observation reason.")
        if self.primary_layer not in self.affected_layers:
            raise ValueError("primary_layer must occur in affected_layers.")
        if not isinstance(self.claim_actions, tuple) or any(
            not isinstance(item, ClaimAction) for item in self.claim_actions
        ):
            raise ValueError("claim_actions must be a tuple of ClaimAction instances.")
        if self.status == "UNVERIFIABLE" and not self.missing_required_evidence:
            raise ValueError("UNVERIFIABLE requires missing_required_evidence.")
        if self.status != "UNVERIFIABLE" and self.missing_required_evidence:
            raise ValueError("missing_required_evidence is reserved for UNVERIFIABLE.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "finding_id": self.finding_id,
            "status": self.status,
            "obligation_id": self.obligation_id,
            "primary_layer": self.primary_layer,
            "affected_layers": list(self.affected_layers),
            "first_invalid_edge": (
                None if self.first_invalid_edge is None else self.first_invalid_edge.to_dict()
            ),
            "candidate_witness_ids": list(self.candidate_witness_ids),
            "affected_claim_ids": list(self.affected_claim_ids),
            "claim_actions": [item.to_dict() for item in self.claim_actions],
            "evidence_refs": list(self.evidence_refs),
            "reason_codes": list(self.reason_codes),
            "missing_required_evidence": list(self.missing_required_evidence),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ScientificAuditFinding:
        expected = {
            "finding_id",
            "status",
            "obligation_id",
            "primary_layer",
            "affected_layers",
            "first_invalid_edge",
            "candidate_witness_ids",
            "affected_claim_ids",
            "claim_actions",
            "evidence_refs",
            "reason_codes",
            "missing_required_evidence",
        }
        if not isinstance(payload, Mapping) or set(payload) != expected:
            raise ValueError("scientific audit finding fields differ from the canonical schema.")
        edge_payload = payload["first_invalid_edge"]
        edge = None if edge_payload is None else CanonicalEdge.from_dict(edge_payload)

        def values(name: str) -> tuple[str, ...]:
            raw = payload[name]
            if not isinstance(raw, list):
                raise ValueError(f"{name} must be a list.")
            return tuple(raw)

        raw_actions = payload["claim_actions"]
        if not isinstance(raw_actions, list):
            raise ValueError("claim_actions must be a list.")
        return cls(
            finding_id=str(payload["finding_id"]),
            status=str(payload["status"]),
            obligation_id=str(payload["obligation_id"]),
            primary_layer=str(payload["primary_layer"]),
            affected_layers=values("affected_layers"),
            first_invalid_edge=edge,
            candidate_witness_ids=values("candidate_witness_ids"),
            affected_claim_ids=values("affected_claim_ids"),
            claim_actions=tuple(ClaimAction.from_dict(item) for item in raw_actions),
            evidence_refs=values("evidence_refs"),
            reason_codes=values("reason_codes"),
            missing_required_evidence=values("missing_required_evidence"),
        )


@dataclass(frozen=True)
class ScientificAuditReport:
    status: str
    findings: tuple[ScientificAuditFinding, ...]
    claim_decisions: tuple[ScientificClaimDecision, ...]
    contract_version: str = CANONICAL_CONTRACT_VERSION
    schema_version: str = "dmcasrc.scientific-audit-report.subsequent"

    def __post_init__(self) -> None:
        if self.status not in CANONICAL_AUDIT_STATUSES:
            raise ValueError(f"invalid canonical audit status: {self.status}.")
        if self.contract_version != CANONICAL_CONTRACT_VERSION:
            raise ValueError("scientific audit report must use the canonical contract.")
        if self.schema_version != "dmcasrc.scientific-audit-report.subsequent":
            raise ValueError("unsupported scientific audit report schema.")
        if not isinstance(self.findings, tuple) or any(
            not isinstance(item, ScientificAuditFinding) for item in self.findings
        ):
            raise ValueError("findings must be a tuple of ScientificAuditFinding instances.")
        if not isinstance(self.claim_decisions, tuple) or any(
            not isinstance(item, ScientificClaimDecision) for item in self.claim_decisions
        ):
            raise ValueError("claim_decisions must be canonical decisions.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "contract_version": self.contract_version,
            "status": self.status,
            "findings": [item.to_dict() for item in self.findings],
            "claim_decisions": [item.to_dict() for item in self.claim_decisions],
        }
