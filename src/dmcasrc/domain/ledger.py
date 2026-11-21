"""Candidate-level ledger used as the audit unit."""

from __future__ import annotations

import copy
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

CANDIDATE_LEDGER_SCHEMA_VERSION = "dmcasrc.candidate-ledger.initial"
_EVIDENCE_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class LedgerValidationError(ValueError):
    pass


def _text(name: str, value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise LedgerValidationError(f"{name} must be a non-empty string.")
    return value.strip()


def parse_timestamp(name: str, value: Any) -> datetime:
    raw = _text(name, value)
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError as exc:
        raise LedgerValidationError(f"{name} must be an ISO-8601 timestamp.") from exc
    if parsed.tzinfo is None:
        raise LedgerValidationError(f"{name} must include a timezone.")
    return parsed


# ── Shared evidence structure validator (mirrors candidate-ledger.schema.json) ──
# The schema's evidence object is additionalProperties:false with 12 known
# namespaces; each namespace is a oneOf of a capability_status branch and a
# substantive-evidence branch (both additionalProperties:false). This validator
# is the single source of truth used by both CandidateRecord.__post_init__ and
# CandidateRecord.from_dict so direct construction cannot emit a Schema-illegal
# payload.
_EVIDENCE_NAMESPACES = frozenset(
    {
        "candidate_identity",
        "time_availability",
        "fold_role",
        "prediction_origin",
        "ancestor_closure",
        "calibration_identity",
        "threshold_identity",
        "model_identity",
        "execution_identity",
        "artifact_integrity",
        "outcome_consumption",
        "claim_binding",
    }
)
_CAPABILITY_STATUSES = frozenset({"unsupported", "not_applicable"})

# Per-namespace substantive fields: field -> (is_array, unique_items).
_EVIDENCE_SUBSTANTIVE_FIELDS: dict[str, dict[str, tuple[bool, bool]]] = {
    "candidate_identity": {"source_keys": (True, False)},
    "time_availability": {"available_ts": (True, False)},
    "fold_role": {
        "assessment_keys": (True, True),
        "fit_keys": (True, True),
        "calibration_keys": (True, True),
        "threshold_keys": (True, True),
    },
    "prediction_origin": {"origin": (False, False)},
    "ancestor_closure": {"observed_node_ids": (True, True)},
    "calibration_identity": {
        "artifact_id": (False, False),
        "fit_candidate_keys": (True, True),
        "applies_to_candidate_keys": (True, True),
    },
    "threshold_identity": {
        "artifact_id": (False, False),
        "fit_candidate_keys": (True, True),
        "applies_to_candidate_keys": (True, True),
    },
    "model_identity": {
        "artifact_id": (False, False),
        "semantic_id": (False, False),
        "training_data_refs": (True, True),
        "config_artifact_id": (False, False),
        "config_hash": (False, False),
    },
    "execution_identity": {
        "decision_execution_id": (False, False),
        "action_execution_id": (False, False),
        "evaluation_execution_id": (False, False),
    },
    "artifact_integrity": {"artifact_ids": (True, True)},
    "outcome_consumption": {"consumer_roles": (True, True)},
    "claim_binding": {
        "claim_ids": (True, True),
        "artifact_ids": (True, True),
    },
}


def _validate_evidence_structure(evidence: Any) -> dict[str, Any]:
    """Validate candidate evidence against the candidate-ledger initial schema.

    Mirrors the schema's ``additionalProperties: false`` evidence object and the
    per-namespace ``oneOf`` (capability_status branch vs. substantive branch).
    Returns a normalized, deep-copied dict with string keys. Any violation —
    unknown namespace, unknown field, ``first_invalid_edge`` forgery, wrong
    types, duplicate items where ``uniqueItems`` is declared, or a malformed
    ``config_hash`` — raises ``LedgerValidationError``.
    """
    if not isinstance(evidence, Mapping):
        raise LedgerValidationError("candidate evidence must be an object.")
    normalized: dict[str, Any] = {}
    for raw_key, raw_value in evidence.items():
        key = str(raw_key)
        if not key:
            raise LedgerValidationError("evidence namespace key must be non-empty.")
        if key not in _EVIDENCE_NAMESPACES:
            raise LedgerValidationError(f"unknown evidence namespace: {key!r}.")
        if not isinstance(raw_value, Mapping):
            raise LedgerValidationError(f"evidence for {key!r} must be an object.")
        value = {str(k): copy.deepcopy(v) for k, v in raw_value.items()}
        if any(not isinstance(k, str) or not k for k in value):
            raise LedgerValidationError(f"evidence {key!r} field names must be non-empty strings.")
        substantive = _EVIDENCE_SUBSTANTIVE_FIELDS[key]
        keys = set(value.keys())
        if "capability_status" in keys:
            # Capability branch: capability_status is required and the ONLY
            # allowed field (additionalProperties:false on this branch).
            extra = sorted(keys - {"capability_status"})
            if extra:
                raise LedgerValidationError(
                    f"evidence {key!r}: capability_status cannot coexist with {extra}."
                )
            cap = value["capability_status"]
            if not isinstance(cap, str) or cap not in _CAPABILITY_STATUSES:
                raise LedgerValidationError(
                    f"evidence {key!r}: capability_status must be 'unsupported' or 'not_applicable'."
                )
        else:
            # Substantive branch: only declared substantive fields are allowed.
            unknown = sorted(keys - set(substantive.keys()))
            if unknown:
                raise LedgerValidationError(
                    f"evidence {key!r}: unknown field(s): {unknown}."
                )
            for fname, (is_array, unique) in substantive.items():
                if fname not in value:
                    continue
                fval = value[fname]
                if is_array:
                    if not isinstance(fval, list):
                        raise LedgerValidationError(
                            f"evidence {key!r}.{fname} must be an array of non-empty strings."
                        )
                    if any(not isinstance(item, str) or not item for item in fval):
                        raise LedgerValidationError(
                            f"evidence {key!r}.{fname} must contain only non-empty strings."
                        )
                    if unique and len(fval) != len(set(fval)):
                        raise LedgerValidationError(
                            f"evidence {key!r}.{fname} must not contain duplicates."
                        )
                    if key == "model_identity" and fname == "training_data_refs" and not fval:
                        raise LedgerValidationError(
                            f"evidence {key!r}.{fname} must contain at least one reference."
                        )
                else:
                    if not isinstance(fval, str) or not fval.strip():
                        raise LedgerValidationError(
                            f"evidence {key!r}.{fname} must be a non-empty string."
                        )
                    if (
                        key == "model_identity"
                        and fname == "config_hash"
                        and not _EVIDENCE_SHA256_PATTERN.fullmatch(fval)
                    ):
                        raise LedgerValidationError(
                            f"evidence {key!r}.config_hash must be 64 lowercase hex characters."
                        )
        normalized[key] = value
    return normalized


@dataclass(frozen=True)
class CandidateRecord:
    candidate_key: str
    layer: str
    node_id: str
    decision_ts: str
    fold: str
    action: str
    outcome_ref: str
    evidence: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "candidate_key", _text("candidate_key", self.candidate_key))
        object.__setattr__(self, "layer", _text("layer", self.layer))
        object.__setattr__(self, "node_id", _text("node_id", self.node_id))
        decision_ts = _text("decision_ts", self.decision_ts)
        parse_timestamp("decision_ts", decision_ts)
        object.__setattr__(self, "decision_ts", decision_ts)
        object.__setattr__(self, "fold", _text("fold", self.fold))
        object.__setattr__(self, "action", _text("action", self.action))
        object.__setattr__(self, "outcome_ref", _text("outcome_ref", self.outcome_ref))
        object.__setattr__(
            self, "evidence", _validate_evidence_structure(self.evidence)
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> CandidateRecord:
        expected = {
            "candidate_key",
            "layer",
            "node_id",
            "decision_ts",
            "fold",
            "action",
            "outcome_ref",
            "evidence",
        }
        if not isinstance(payload, Mapping) or {str(key) for key in payload} != expected:
            raise LedgerValidationError("candidate record has missing or unknown fields.")
        decision_ts = _text("decision_ts", payload["decision_ts"])
        parse_timestamp("decision_ts", decision_ts)
        return cls(
            candidate_key=_text("candidate_key", payload["candidate_key"]),
            layer=_text("layer", payload["layer"]),
            node_id=_text("node_id", payload["node_id"]),
            decision_ts=decision_ts,
            fold=_text("fold", payload["fold"]),
            action=_text("action", payload["action"]),
            outcome_ref=_text("outcome_ref", payload["outcome_ref"]),
            evidence=payload["evidence"],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_key": self.candidate_key,
            "layer": self.layer,
            "node_id": self.node_id,
            "decision_ts": self.decision_ts,
            "fold": self.fold,
            "action": self.action,
            "outcome_ref": self.outcome_ref,
            "evidence": copy.deepcopy(self.evidence),
        }


@dataclass(frozen=True)
class CandidateLedger:
    records: tuple[CandidateRecord, ...]
    schema_version: str = CANDIDATE_LEDGER_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.records, tuple):
            raise LedgerValidationError("candidate ledger records must be a tuple.")
        for item in self.records:
            if not isinstance(item, CandidateRecord):
                raise LedgerValidationError("candidate ledger records must be CandidateRecord instances.")
        if self.schema_version != CANDIDATE_LEDGER_SCHEMA_VERSION:
            raise LedgerValidationError(f"unsupported ledger schema: {self.schema_version}.")
        identities = [(item.candidate_key, item.layer, item.node_id) for item in self.records]
        if len(identities) != len(set(identities)):
            raise LedgerValidationError("duplicate candidate/layer/node records are forbidden.")
        candidate_node_pairs = [(item.candidate_key, item.node_id) for item in self.records]
        if len(candidate_node_pairs) != len(set(candidate_node_pairs)):
            raise LedgerValidationError("duplicate candidate/node records are forbidden.")

    def records_for_candidate(self, candidate_key: str) -> tuple[CandidateRecord, ...]:
        """Return all records for *candidate_key* sorted by ``(layer, node_id)``."""
        matches = [r for r in self.records if r.candidate_key == candidate_key]
        matches.sort(key=lambda r: (r.layer, r.node_id))
        return tuple(matches)

    def by_candidate_node(
        self, candidate_key: str, node_id: str
    ) -> CandidateRecord | None:
        """Return the record matching *candidate_key* and *node_id*.

        Raises ``LedgerValidationError`` when more than one record matches
        the same (candidate_key, node_id) pair — this should never happen
        because ``__post_init__`` forbids it.
        """
        matches = [
            r
            for r in self.records
            if r.candidate_key == candidate_key and r.node_id == node_id
        ]
        if not matches:
            return None
        if len(matches) == 1:
            return matches[0]
        raise LedgerValidationError("duplicate candidate/node records are forbidden.")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> CandidateLedger:
        if not isinstance(payload, Mapping):
            raise LedgerValidationError("candidate ledger root must be an object.")
        if {str(key) for key in payload} != {"schema_version", "records"}:
            raise LedgerValidationError("candidate ledger has missing or unknown fields.")
        if payload["schema_version"] != CANDIDATE_LEDGER_SCHEMA_VERSION:
            raise LedgerValidationError(f"unsupported ledger schema: {payload['schema_version']}.")
        if not isinstance(payload["records"], list):
            raise LedgerValidationError("candidate ledger records must be a list.")
        return cls(records=tuple(CandidateRecord.from_dict(item) for item in payload["records"]))

    def to_dict(self) -> dict[str, Any]:
        ordered = sorted(
            self.records,
            key=lambda item: (item.candidate_key, item.layer, item.node_id),
        )
        return {
            "schema_version": self.schema_version,
            "records": [record.to_dict() for record in ordered],
        }
