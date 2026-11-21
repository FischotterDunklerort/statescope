"""Deterministic identities for project-neutral adapter evidence."""

from __future__ import annotations

import hashlib
import json
import math
import unicodedata
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import PurePath
from typing import Any


class IdentityError(ValueError):
    pass


def normalize_timestamp(value: str | datetime) -> str:
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise IdentityError("timestamp must be ISO-8601") from exc
    elif isinstance(value, datetime):
        parsed = value
    else:
        raise IdentityError("timestamp must be a string or datetime")
    if parsed.tzinfo is None:
        raise IdentityError("timestamp must include a timezone")
    return parsed.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _normalize(value: Any) -> Any:
    if value is None or isinstance(value, bool | int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise IdentityError("non-finite floats are not stable identities")
        return {"$float": format(value, ".17g")}
    if isinstance(value, str):
        return unicodedata.normalize("NFC", value)
    if isinstance(value, datetime):
        return {"$timestamp": normalize_timestamp(value)}
    if isinstance(value, PurePath):
        return {"$path": value.as_posix()}
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise IdentityError("identity mappings require string keys")
        return {key: _normalize(value[key]) for key in sorted(value)}
    if isinstance(value, Sequence) and not isinstance(value, bytes | bytearray):
        return [_normalize(item) for item in value]
    raise IdentityError(f"unsupported identity value: {type(value).__name__}")


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        _normalize(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def semantic_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def stable_id(namespace: str, value: Any) -> str:
    if not isinstance(namespace, str) or not namespace.strip():
        raise IdentityError("namespace must be a non-empty string")
    return f"{namespace}:{semantic_hash(value)[:24]}"


def candidate_key(project_id: str, parts: Mapping[str, Any]) -> str:
    return stable_id(f"{project_id}.candidate", parts)


def node_key(project_id: str, parts: Mapping[str, Any]) -> str:
    """Return a stable project-scoped canonical node identity."""
    return stable_id(f"{project_id}.node", parts)


def finding_key(
    *,
    candidate_id: str,
    node_id: str,
    obligation_id: str,
    reason_codes: Sequence[str],
) -> str:
    """Return a deterministic finding identity from semantic coordinates."""
    return stable_id(
        "dmcasrc.finding",
        {
            "candidate_id": candidate_id,
            "node_id": node_id,
            "obligation_id": obligation_id,
            "reason_codes": sorted(reason_codes),
        },
    )


__all__ = [
    "IdentityError",
    "candidate_key",
    "canonical_json",
    "finding_key",
    "node_key",
    "normalize_timestamp",
    "semantic_hash",
    "stable_id",
]
