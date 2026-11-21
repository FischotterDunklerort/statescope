"""Low-intrusion runtime fact capture for DM-CASRC supplementary experiments.

Importing this package does not import optional ML frameworks or alter global
instrumentation.  Framework hooks are installed only by an explicit
``AutoFactCapture.start()`` / context-manager call.
"""

from .capture import AutoFactCapture, InstrumentationStatus
from .events import (
    CaptureMode,
    EventType,
    EvidenceRequirement,
    EvidenceSource,
    ObjectReference,
    RuntimeEvent,
    object_reference,
    trace_sha256,
)
from .reconciliation import (
    EvidenceValue,
    ReconciledEvidence,
    ReconciliationStatus,
    has_blocking_reconciliation,
    reconcile_evidence,
)

__all__ = [
    "AutoFactCapture",
    "CaptureMode",
    "EvidenceRequirement",
    "EvidenceSource",
    "EvidenceValue",
    "EventType",
    "InstrumentationStatus",
    "ObjectReference",
    "ReconciledEvidence",
    "ReconciliationStatus",
    "RuntimeEvent",
    "has_blocking_reconciliation",
    "object_reference",
    "reconcile_evidence",
    "trace_sha256",
]
