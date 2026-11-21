"""Explicit project adapter registry."""

from __future__ import annotations

from typing import Any

from .base import ProjectAdapter, validate_project_adapter


class AdapterRegistry:
    def __init__(self) -> None:
        self._items: dict[str, ProjectAdapter] = {}

    def register(self, adapter: Any) -> None:
        validated = validate_project_adapter(adapter)
        if validated.project_id in self._items:
            raise ValueError(f"duplicate project adapter: {validated.project_id}")
        self._items[validated.project_id] = validated

    def get(self, project_id: str) -> ProjectAdapter:
        try:
            return self._items[project_id]
        except KeyError as exc:
            raise KeyError(f"unknown project adapter: {project_id}") from exc

    def ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._items))


def adapter_registry() -> AdapterRegistry:
    from .finrl.adapter import FinrlAdapter
    from .freqai.adapter import FreqaiAdapter
    from .freqtrade.adapter import FreqtradeAdapter
    from .lean.adapter import LeanAdapter
    from .qlib.adapter import QlibAdapter
    from .sklearn_pipeline.adapter import SklearnPipelineAdapter

    registry = AdapterRegistry()
    for adapter in (
        QlibAdapter(),
        FinrlAdapter(),
        SklearnPipelineAdapter(),
        FreqtradeAdapter(),
        FreqaiAdapter(),
        LeanAdapter(),
    ):
        registry.register(adapter)
    return registry


__all__ = ["AdapterRegistry", "adapter_registry"]
