"""Normalized runtime-event and provenance objects for AutoFact.

This module deliberately has no pandas, NumPy, scikit-learn, or joblib import.
The runtime observer imports those packages only when a caller explicitly
starts instrumentation.  Keeping the event vocabulary framework-neutral lets
the existing DM-CASRC core remain import-isolated from vendor dependencies.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any


class EventType(StrEnum):
    """Canonical operations observed or declared by the AutoFact layer."""

    DATA_LOAD = "DATA_LOAD"
    DATA_CREATE = "DATA_CREATE"
    DATA_SPLIT = "DATA_SPLIT"
    TRANSFORM_FIT = "TRANSFORM_FIT"
    TRANSFORM_APPLY = "TRANSFORM_APPLY"
    MODEL_FIT = "MODEL_FIT"
    CANDIDATE_CREATE = "CANDIDATE_CREATE"
    PREDICT = "PREDICT"
    EVALUATE = "EVALUATE"
    CANDIDATE_SELECT = "CANDIDATE_SELECT"
    ARTIFACT_READ = "ARTIFACT_READ"
    ARTIFACT_WRITE = "ARTIFACT_WRITE"
    CLAIM_REQUEST = "CLAIM_REQUEST"
    CLAIM_RELEASE = "CLAIM_RELEASE"


class EvidenceSource(StrEnum):
    """Origin class for evidence rather than an audit verdict."""

    AUTO_OBSERVED = "AUTO_OBSERVED"
    DECLARED = "DECLARED"
    DERIVED = "DERIVED"
    EXTERNAL = "EXTERNAL"
    UNKNOWN = "UNKNOWN"


class EvidenceRequirement(StrEnum):
    """How a missing field participates in a specific contract."""

    REQUIRED_FOR_THIS_CONTRACT = "REQUIRED_FOR_THIS_CONTRACT"
    OPTIONAL_METADATA = "OPTIONAL_METADATA"
    UNKNOWN_BUT_NONBLOCKING = "UNKNOWN_BUT_NONBLOCKING"
    UNKNOWN_AND_UNVERIFIABLE = "UNKNOWN_AND_UNVERIFIABLE"


class CaptureMode(StrEnum):
    """Evidence-collection configuration, not an audit status."""

    MANUAL = "MANUAL"
    AUTO = "AUTO"
    HYBRID = "HYBRID"


_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_MEMORY_ADDRESS = re.compile(r"0x[0-9A-Fa-f]+")


def _canonical_value(value: Any, active: set[int] | None = None) -> Any:
    """Return a deterministic, bounded JSON-compatible representation.

    This is a provenance fingerprint helper, not a serializer for scientific
    model state.  Large arrays and framework objects are represented by type,
    shape, and a content or state fingerprint in :func:`object_reference`.
    """

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if value != value:
            return "NaN"
        if value in {float("inf"), float("-inf")}:
            return str(value)
        return value
    if isinstance(value, Path):
        return {"path": value.as_posix()}
    if isinstance(value, bytes):
        return {"bytes_sha256": hashlib.sha256(value).hexdigest(), "size": len(value)}

    stack = set() if active is None else active
    object_id = id(value)
    if object_id in stack:
        return {"cycle": f"{type(value).__module__}.{type(value).__qualname__}"}

    if isinstance(value, Mapping):
        stack.add(object_id)
        try:
            return {
                str(key): _canonical_value(item, stack)
                for key, item in sorted(value.items(), key=lambda item: str(item[0]))
            }
        finally:
            stack.remove(object_id)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        stack.add(object_id)
        try:
            return [_canonical_value(item, stack) for item in value]
        finally:
            stack.remove(object_id)
    if isinstance(value, (set, frozenset)):
        stack.add(object_id)
        try:
            values = [_canonical_value(item, stack) for item in value]
            return sorted(values, key=lambda item: _stable_hash(item))
        finally:
            stack.remove(object_id)

    return {
        "type": f"{type(value).__module__}.{type(value).__qualname__}",
        "repr": _MEMORY_ADDRESS.sub("0x<address>", repr(value))[:512],
    }


def _stable_hash(value: Any) -> str:
    encoded = json.dumps(
        _canonical_value(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class ObjectReference:
    """Run-local identity plus a stable, content-aware provenance fingerprint."""

    object_id: str
    type_name: str
    fingerprint: str
    fingerprint_mode: str
    shape: tuple[int, ...] | None = None
    size_bytes: int | None = None

    def __post_init__(self) -> None:
        for name in ("object_id", "type_name", "fingerprint", "fingerprint_mode"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty string.")
        if not _HEX64.fullmatch(self.fingerprint):
            raise ValueError("fingerprint must be a lowercase SHA-256 digest.")
        if self.shape is not None and (
            not isinstance(self.shape, tuple)
            or any(isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in self.shape)
        ):
            raise ValueError("shape must be a tuple of non-negative integers or None.")
        if self.size_bytes is not None and (
            isinstance(self.size_bytes, bool) or not isinstance(self.size_bytes, int) or self.size_bytes < 0
        ):
            raise ValueError("size_bytes must be a non-negative integer or None.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "object_id": self.object_id,
            "type_name": self.type_name,
            "fingerprint": self.fingerprint,
            "fingerprint_mode": self.fingerprint_mode,
            "shape": None if self.shape is None else list(self.shape),
            "size_bytes": self.size_bytes,
        }


def _array_bytes(value: Any) -> bytes | None:
    tobytes = getattr(value, "tobytes", None)
    if not callable(tobytes):
        return None
    try:
        return tobytes(order="C")
    except (TypeError, ValueError):
        try:
            return tobytes()
        except (TypeError, ValueError):
            return None


def object_reference(value: Any, *, sample_bytes: int = 1024 * 1024) -> ObjectReference:
    """Create a provenance reference without importing optional frameworks.

    Full content hashes are used for reasonably sized array-like values.
    Larger buffers use a clearly labelled first/last-byte sample; consumers
    must not treat a sampled fingerprint as a byte-identity artifact hash.
    """

    if sample_bytes <= 0:
        raise ValueError("sample_bytes must be positive.")
    type_name = f"{type(value).__module__}.{type(value).__qualname__}"
    shape_value = getattr(value, "shape", None)
    shape: tuple[int, ...] | None = None
    if shape_value is not None:
        try:
            shape = tuple(int(item) for item in shape_value)
        except (TypeError, ValueError):
            shape = None
    raw = _array_bytes(value)
    if raw is not None:
        size = len(raw)
        prefix = {
            "type": type_name,
            "shape": shape,
            "dtype": str(getattr(value, "dtype", "unknown")),
            "size_bytes": size,
        }
        if size <= sample_bytes:
            digest = hashlib.sha256(raw).hexdigest()
            mode = "array_full_content"
        else:
            head = raw[: sample_bytes // 2]
            tail = raw[-(sample_bytes // 2) :]
            digest = hashlib.sha256(
                json.dumps(prefix, sort_keys=True).encode("utf-8") + head + tail
            ).hexdigest()
            mode = "array_sampled_content"
        fingerprint = _stable_hash({**prefix, "content": digest, "mode": mode})
        return ObjectReference(
            object_id=f"py:{id(value):x}",
            type_name=type_name,
            fingerprint=fingerprint,
            fingerprint_mode=mode,
            shape=shape,
            size_bytes=size,
        )

    if isinstance(value, Path):
        path_payload: dict[str, Any] = {"path": value.as_posix(), "exists": value.exists()}
        if value.is_file():
            path_payload["size_bytes"] = value.stat().st_size
        return ObjectReference(
            object_id=f"py:{id(value):x}",
            type_name=type_name,
            fingerprint=_stable_hash(path_payload),
            fingerprint_mode="path_metadata",
            shape=None,
            size_bytes=path_payload.get("size_bytes"),
        )

    get_params = getattr(value, "get_params", None)
    if callable(get_params):
        try:
            parameters = get_params(deep=False)
        except (AttributeError, TypeError, ValueError):
            parameters = None
        if isinstance(parameters, Mapping):
            return ObjectReference(
                object_id=f"py:{id(value):x}",
                type_name=type_name,
                fingerprint=_stable_hash({"type": type_name, "params": parameters}),
                fingerprint_mode="estimator_parameters",
                shape=shape,
                size_bytes=None,
            )

    return ObjectReference(
        object_id=f"py:{id(value):x}",
        type_name=type_name,
        fingerprint=_stable_hash(value),
        fingerprint_mode="structural_summary",
        shape=shape,
        size_bytes=None,
    )


@dataclass(frozen=True)
class RuntimeEvent:
    """Normalized event retained as inspectable runtime evidence."""

    event_id: str
    event_type: EventType
    logical_time: int
    operator: str
    inputs: tuple[ObjectReference, ...] = ()
    outputs: tuple[ObjectReference, ...] = ()
    source: EvidenceSource = EvidenceSource.AUTO_OBSERVED
    wall_time: str | None = None
    candidate_id: str | None = None
    fold_role: str | None = None
    dataset_role: str | None = None
    temporal_range: tuple[str, str] | None = None
    parent_event_ids: tuple[str, ...] = ()
    artifact_path: str | None = None
    artifact_sha256: str | None = None
    claim_id: str | None = None
    attributes: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.event_id, str) or not self.event_id:
            raise ValueError("event_id must be a non-empty string.")
        if not isinstance(self.event_type, EventType):
            raise ValueError("event_type must be an EventType.")
        if isinstance(self.logical_time, bool) or not isinstance(self.logical_time, int) or self.logical_time < 1:
            raise ValueError("logical_time must be a positive integer.")
        if not isinstance(self.operator, str) or not self.operator:
            raise ValueError("operator must be a non-empty string.")
        for name in ("inputs", "outputs"):
            values = getattr(self, name)
            if not isinstance(values, tuple) or any(not isinstance(item, ObjectReference) for item in values):
                raise ValueError(f"{name} must be a tuple of ObjectReference values.")
        if not isinstance(self.source, EvidenceSource):
            raise ValueError("source must be an EvidenceSource.")
        if self.wall_time is not None:
            if not isinstance(self.wall_time, str) or not self.wall_time:
                raise ValueError("wall_time must be an ISO-8601 string or None.")
            try:
                parsed = datetime.fromisoformat(self.wall_time.replace("Z", "+00:00"))
            except ValueError as exc:
                raise ValueError("wall_time must be ISO-8601.") from exc
            if parsed.tzinfo is None:
                raise ValueError("wall_time must include a timezone.")
        for name in ("candidate_id", "fold_role", "dataset_role", "artifact_path", "claim_id"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value):
                raise ValueError(f"{name} must be a non-empty string or None.")
        if self.temporal_range is not None:
            if not isinstance(self.temporal_range, tuple) or len(self.temporal_range) != 2:
                raise ValueError("temporal_range must be a two-item tuple or None.")
            for item in self.temporal_range:
                if not isinstance(item, str) or not item:
                    raise ValueError("temporal_range values must be non-empty strings.")
        if (
            not isinstance(self.parent_event_ids, tuple)
            or any(not isinstance(item, str) or not item for item in self.parent_event_ids)
            or len(self.parent_event_ids) != len(set(self.parent_event_ids))
        ):
            raise ValueError("parent_event_ids must be unique non-empty strings.")
        if self.artifact_sha256 is not None and (
            not isinstance(self.artifact_sha256, str) or not _HEX64.fullmatch(self.artifact_sha256)
        ):
            raise ValueError("artifact_sha256 must be a lowercase SHA-256 digest or None.")
        if self.attributes is not None and not isinstance(self.attributes, Mapping):
            raise ValueError("attributes must be a mapping or None.")
        object.__setattr__(self, "attributes", dict(self.attributes or {}))

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "logical_time": self.logical_time,
            "operator": self.operator,
            "inputs": [item.to_dict() for item in self.inputs],
            "outputs": [item.to_dict() for item in self.outputs],
            "source": self.source.value,
            "wall_time": self.wall_time,
            "candidate_id": self.candidate_id,
            "fold_role": self.fold_role,
            "dataset_role": self.dataset_role,
            "temporal_range": None if self.temporal_range is None else list(self.temporal_range),
            "parent_event_ids": list(self.parent_event_ids),
            "artifact_path": self.artifact_path,
            "artifact_sha256": self.artifact_sha256,
            "claim_id": self.claim_id,
            "attributes": _canonical_value(self.attributes),
        }


def trace_sha256(events: Sequence[RuntimeEvent]) -> str:
    """Hash normalized trace content, excluding wall-clock timing by design."""

    if any(not isinstance(item, RuntimeEvent) for item in events):
        raise TypeError("events must contain RuntimeEvent values.")
    payload = []
    for event in sorted(events, key=lambda item: (item.logical_time, item.event_id)):
        item = event.to_dict()
        item["wall_time"] = None
        payload.append(item)
    # RuntimeEvent.to_dict already canonicalizes its only arbitrary-valued
    # field (attributes); all remaining fields are validated JSON primitives.
    # Avoid a second recursive walk, retaining the exact canonical digest.
    encoded=json.dumps(payload,ensure_ascii=False,sort_keys=True,separators=(",", ":")).encode('utf8')
    return hashlib.sha256(encoded).hexdigest()


__all__ = [
    "CaptureMode",
    "EvidenceRequirement",
    "EvidenceSource",
    "EventType",
    "ObjectReference",
    "RuntimeEvent",
    "object_reference",
    "trace_sha256",
]
