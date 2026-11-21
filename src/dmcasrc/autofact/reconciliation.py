"""Evidence-source labelling and deterministic, conflict-visible reconciliation."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from .events import EvidenceRequirement, EvidenceSource


class ReconciliationStatus(StrEnum):
    RESOLVED = "RESOLVED"
    CONFLICT = "CONFLICT"
    UNKNOWN_NONBLOCKING = "UNKNOWN_NONBLOCKING"
    UNVERIFIABLE = "UNVERIFIABLE"


_SOURCE_PRECEDENCE = {
    EvidenceSource.AUTO_OBSERVED: 4,
    EvidenceSource.EXTERNAL: 3,
    EvidenceSource.DERIVED: 2,
    EvidenceSource.DECLARED: 1,
    EvidenceSource.UNKNOWN: 0,
}


def _canonical(value: Any) -> str:
    """Produce a stable comparison key without treating the key as evidence."""

    try:
        encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=repr)
    except (TypeError, ValueError):
        encoded = repr(value)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class EvidenceValue:
    """One asserted value and the runtime event(s) supporting it."""

    key: str
    value: Any
    source: EvidenceSource
    provenance_event_ids: tuple[str, ...]
    requirement: EvidenceRequirement = EvidenceRequirement.REQUIRED_FOR_THIS_CONTRACT

    def __post_init__(self) -> None:
        if not isinstance(self.key, str) or not self.key:
            raise ValueError("key must be a non-empty string.")
        if not isinstance(self.source, EvidenceSource):
            raise ValueError("source must be an EvidenceSource.")
        if not isinstance(self.requirement, EvidenceRequirement):
            raise ValueError("requirement must be an EvidenceRequirement.")
        if (
            not isinstance(self.provenance_event_ids, tuple)
            or any(not isinstance(item, str) or not item for item in self.provenance_event_ids)
            or len(self.provenance_event_ids) != len(set(self.provenance_event_ids))
        ):
            raise ValueError("provenance_event_ids must be unique non-empty strings.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "value": self.value,
            "source": self.source.value,
            "provenance_event_ids": list(self.provenance_event_ids),
            "requirement": self.requirement.value,
        }


@dataclass(frozen=True)
class ReconciledEvidence:
    """Resolution result retaining every supplied source rather than hiding conflicts."""

    key: str
    status: ReconciliationStatus
    selected: EvidenceValue | None
    inputs: tuple[EvidenceValue, ...]
    conflict_sources: tuple[EvidenceSource, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.key, str) or not self.key:
            raise ValueError("key must be a non-empty string.")
        if not isinstance(self.status, ReconciliationStatus):
            raise ValueError("status must be a ReconciliationStatus.")
        if not isinstance(self.inputs, tuple) or not self.inputs:
            raise ValueError("inputs must be a non-empty tuple.")
        if any(not isinstance(item, EvidenceValue) for item in self.inputs):
            raise ValueError("inputs must contain EvidenceValue values.")
        if {item.key for item in self.inputs} != {self.key}:
            raise ValueError("all inputs must have the reconciled key.")
        if self.selected is not None and self.selected not in self.inputs:
            raise ValueError("selected value must be one of inputs.")
        if self.status in {ReconciliationStatus.CONFLICT, ReconciliationStatus.UNVERIFIABLE} and self.selected is not None:
            raise ValueError("conflict and unverifiable results may not select a value.")
        if self.status == ReconciliationStatus.RESOLVED and self.selected is None:
            raise ValueError("resolved evidence requires a selected value.")
        if any(not isinstance(item, EvidenceSource) for item in self.conflict_sources):
            raise ValueError("conflict_sources must contain EvidenceSource values.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "status": self.status.value,
            "selected": None if self.selected is None else self.selected.to_dict(),
            "inputs": [item.to_dict() for item in self.inputs],
            "conflict_sources": [item.value for item in self.conflict_sources],
        }


def reconcile_evidence(values: Sequence[EvidenceValue]) -> tuple[ReconciledEvidence, ...]:
    """Reconcile same-key values with explicit conflict and missing-data semantics.

    Precedence only chooses between *equal* known values.  Different known
    values always remain a conflict, even when an automatically observed
    value has higher precedence than a declaration.  This avoids converting a
    convenient value into silently trusted evidence.
    """

    if not values:
        return ()
    if any(not isinstance(item, EvidenceValue) for item in values):
        raise TypeError("values must contain EvidenceValue values.")
    by_key: dict[str, list[EvidenceValue]] = {}
    for item in values:
        by_key.setdefault(item.key, []).append(item)

    resolved: list[ReconciledEvidence] = []
    for key in sorted(by_key):
        inputs = tuple(
            sorted(
                by_key[key],
                key=lambda item: (
                    -_SOURCE_PRECEDENCE[item.source],
                    _canonical(item.value),
                    item.provenance_event_ids,
                ),
            )
        )
        known = tuple(item for item in inputs if item.source != EvidenceSource.UNKNOWN)
        requirements = {item.requirement for item in inputs}
        if known:
            distinct = {_canonical(item.value) for item in known}
            if len(distinct) > 1:
                resolved.append(
                    ReconciledEvidence(
                        key=key,
                        status=ReconciliationStatus.CONFLICT,
                        selected=None,
                        inputs=inputs,
                        conflict_sources=tuple(sorted({item.source for item in known}, key=str)),
                    )
                )
                continue
            resolved.append(
                ReconciledEvidence(
                    key=key,
                    status=ReconciliationStatus.RESOLVED,
                    selected=known[0],
                    inputs=inputs,
                )
            )
            continue

        nonblocking = requirements <= {
            EvidenceRequirement.OPTIONAL_METADATA,
            EvidenceRequirement.UNKNOWN_BUT_NONBLOCKING,
        }
        resolved.append(
            ReconciledEvidence(
                key=key,
                status=(
                    ReconciliationStatus.UNKNOWN_NONBLOCKING
                    if nonblocking
                    else ReconciliationStatus.UNVERIFIABLE
                ),
                selected=None,
                inputs=inputs,
            )
        )
    return tuple(resolved)


def has_blocking_reconciliation(result: ReconciledEvidence) -> bool:
    """Return whether this resolution cannot supply valid required evidence."""

    if not isinstance(result, ReconciledEvidence):
        raise TypeError("result must be ReconciledEvidence.")
    return result.status in {ReconciliationStatus.CONFLICT, ReconciliationStatus.UNVERIFIABLE}


__all__ = [
    "EvidenceValue",
    "ReconciledEvidence",
    "ReconciliationStatus",
    "has_blocking_reconciliation",
    "reconcile_evidence",
]
