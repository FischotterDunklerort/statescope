"""Project evidence adapters.

The package is deliberately vendor-neutral: importing it does not import any
external project runtime.
"""

from .base import ProjectAdapter, validate_project_adapter
from .models import (
    AdapterEvidenceBundle,
    AdapterRunContext,
    AdapterRunSummary,
    AdapterScope,
    EvidenceReference,
)
from .registry import adapter_registry

__all__ = [
    "AdapterEvidenceBundle",
    "AdapterRunContext",
    "AdapterRunSummary",
    "AdapterScope",
    "EvidenceReference",
    "ProjectAdapter",
    "adapter_registry",
    "validate_project_adapter",
]
