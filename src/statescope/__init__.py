"""Public StateScope API.

Legacy implementation module names remain importable for archived objects.
The facade exports the original classes rather than creating wrapper types.
"""
import importlib
import sys
from pathlib import Path

_base = Path(__file__).resolve().parent.parent
_implementation = _base / "audit.py"
if not _implementation.is_file():
    # Source-checkout support; an installed wheel uses its adjacent audit.py.
    _implementation = _base / "outputs/statescope-sqj-lifecycle-revision/runtime/audit.py"
if not _implementation.is_file():
    raise ImportError("StateScope implementation is missing; reinstall the complete package")
sys.path.insert(0, str(_implementation.parent))
_audit = importlib.import_module("audit")
if Path(_audit.__file__).resolve() != _implementation.resolve():
    raise ImportError("A different legacy audit module is already loaded; use a fresh interpreter")
_runtime = importlib.import_module("runtime")
if Path(_runtime.__file__).resolve() != (_implementation.parent / "runtime.py").resolve():
    raise ImportError("A different StateScope runtime is already loaded; use a fresh interpreter")
Audit = _audit.Audit
from dmcasrc.autofact.provenance_continuity import EvidenceArchive, EvidenceHandle, ContinuityError
from dmcasrc.autofact.persistence import ProvenanceStore

__version__ = "0.1.2"
__all__ = ["Audit", "EvidenceArchive", "EvidenceHandle", "ContinuityError", "ProvenanceStore"]

