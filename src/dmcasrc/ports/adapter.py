"""Project adapter port.  Audit rules intentionally do not belong here."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class Adapter(Protocol):
    adapter_id: str
    adapter_version: str

    def discover_graph(self, source: Any) -> Any: ...

    def build_ledger(self, source: Any) -> Any: ...

    def register_artifacts(self, source: Any) -> Any: ...


def validate_adapter(value: Any) -> Adapter:
    required = (
        "adapter_id",
        "adapter_version",
        "discover_graph",
        "build_ledger",
        "register_artifacts",
    )
    missing = [name for name in required if not hasattr(value, name)]
    if missing:
        raise TypeError(f"adapter is missing required member(s): {missing}.")
    if not isinstance(value.adapter_id, str) or not value.adapter_id:
        raise TypeError("adapter_id must be a non-empty string.")
    if not isinstance(value.adapter_version, str) or not value.adapter_version:
        raise TypeError("adapter_version must be a non-empty string.")
    return value


__all__ = ["Adapter", "validate_adapter"]
