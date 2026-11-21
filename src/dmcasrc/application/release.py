"""Explicit local artifact release bound to the captured evidence and policy.

The caller requests a real release at a real output boundary. Unknown or violated
preconditions retain a withheld receipt and never emit CLAIM_RELEASE. These
content hashes bind local records; they do not authenticate a malicious producer.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from collections.abc import Sequence

from dmcasrc.autofact.events import EventType, EvidenceSource, trace_sha256
from dmcasrc.application.formal_contracts import FormalContract, evaluate_formal_contracts
from dmcasrc.autofact.capture import model_state_sha256


def _read_artifact_snapshot(path: Path) -> bytes | None:
    return path.read_bytes() if path.is_file() else None


def record_artifact_write(capture, path: Path, *, artifact_id: str, candidate_id: str):
    """Record bytes after the caller's completed serialization operation."""
    path = Path(path)
    if not artifact_id or not candidate_id:
        raise ValueError("artifact and candidate identifiers must be nonempty")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    observed = [e for e in capture.events if e.event_type == EventType.ARTIFACT_WRITE
                and e.operator == "joblib.dump" and e.artifact_path is not None
                and Path(e.artifact_path).resolve() == path.resolve() and e.artifact_sha256 == digest
                and e.attributes.get("serialization_completed")]
    binding = observed[-1] if observed else None
    return capture.emit(EventType.ARTIFACT_WRITE, operator="dmcasrc.record_artifact_write",
                        source=EvidenceSource.DERIVED, artifact_path=str(path), artifact_sha256=digest,
                        candidate_id=candidate_id,
                        attributes={"artifact_id": artifact_id, "identity_origin": "actual_serialized_bytes",
                                    "serialization_event_id":None if binding is None else binding.event_id,
                                    "serialized_object_id":None if binding is None else binding.attributes.get("serialized_object_id"),
                                    "serialized_state_sha256":None if binding is None else binding.attributes.get("serialized_state_sha256")})


def release_artifact_claim(capture, path: Path, receipt_path: Path, *, artifact_id: str,
                           candidate_id: str, claim_id: str,
                           contracts: Sequence[FormalContract], model=None,
                           require_model_binding: bool = True,
                           require_complete_observation: bool = False) -> dict:
    """Persist a release or withheld receipt using actual bytes and prerequisites."""
    path, receipt_path = Path(path), Path(receipt_path)
    if receipt_path.exists():
        raise FileExistsError("release receipts are immutable; choose a new path")
    if not artifact_id or not candidate_id or not claim_id or not contracts:
        raise ValueError("release requires identifiers and nonempty prerequisite contracts")
    scoped = capture.events if model is None else capture.trace_for_model(model)
    evaluated = evaluate_formal_contracts(scoped, contracts=contracts,
                                         candidate_ids=(candidate_id,))
    writes = [e for e in capture.events if e.event_type == EventType.ARTIFACT_WRITE
              and e.attributes.get("artifact_id") == artifact_id and e.candidate_id == candidate_id]
    snapshot = _read_artifact_snapshot(path)
    actual = hashlib.sha256(snapshot).hexdigest() if snapshot is not None else None
    binding = writes[-1].artifact_sha256 if writes else None
    reasons = [r.contract_id + ":" + r.status.value for r in evaluated.results if r.status.value != "PASS"]
    boundary = capture.observation_boundary_for(model) if model is not None else None
    if require_complete_observation and boundary is None:
        reasons.append("OBSERVATION_BOUNDARY_NOT_ESTABLISHED")
    if actual is None:
        reasons.append("ARTIFACT_MISSING")
    elif binding is None:
        reasons.append("ARTIFACT_WRITE_NOT_OBSERVED")
    elif actual != binding:
        reasons.append("ARTIFACT_BYTES_CHANGED_AFTER_WRITE")
    object_id = None if model is None else f"py:{id(model):x}"
    current_state = None if model is None else model_state_sha256(model)
    observed_binding = writes[-1].attributes if writes else {}
    if require_model_binding:
        if model is None:
            reasons.append("MODEL_NOT_SELECTED")
        elif not observed_binding.get("serialization_event_id"):
            reasons.append("SERIALIZATION_OBJECT_NOT_OBSERVED")
        elif object_id != observed_binding.get("serialized_object_id"):
            reasons.append("SERIALIZED_OBJECT_DIFFERS_FROM_SELECTED_MODEL")
        elif current_state is None or current_state != observed_binding.get("serialized_state_sha256"):
            reasons.append("MODEL_STATE_CHANGED_AFTER_SERIALIZATION")
    published_path = receipt_path.with_name(receipt_path.stem + ".artifact")
    receipt = {"schema": "dmcasrc.artifact-release.subsequent", "claim_id": claim_id,
               "candidate_id": candidate_id, "artifact_id": artifact_id,
               "status": "WITHHELD" if reasons else "RELEASED", "reasons": reasons,
               "assurance_scope":"OBSERVED_OPERATION_CONTRACTS_ONLY" if boundary is None else boundary['scope'],
               "complete_observation_requested":bool(require_complete_observation),
               "complete_observation_established":boundary is not None,
               "observation_boundary":boundary,
               "artifact_sha256": actual, "written_sha256": binding,
               "binding_level":"OBSERVED_MODEL_STATE_AND_FILE_BYTES" if require_model_binding else "FILE_BYTES_ONLY",
               "selected_object_id":object_id,"model_state_sha256":current_state,
               "serialization_event_id":observed_binding.get("serialization_event_id"),
               "published_artifact_path":None,
               "prerequisites": evaluated.to_dict(),
               "fit_scope": "whole captured trace" if model is None else "explicit selected output and captured producers",
               "policy_sha256": hashlib.sha256(json.dumps([c.to_dict() for c in contracts],
                                          sort_keys=True).encode()).hexdigest()}
    if not reasons:
        # Publish the exact inspected snapshot, not the mutable source pathname.
        # Exclusive creation prevents overwriting another local publication.
        import os
        published_path.parent.mkdir(parents=True, exist_ok=True)
        with published_path.open("xb") as handle:
            handle.write(snapshot)
            handle.flush()
            os.fsync(handle.fileno())
        receipt["published_artifact_path"] = str(published_path)
        event = capture.emit(EventType.CLAIM_RELEASE, operator="dmcasrc.release_artifact_claim",
                             source=EvidenceSource.DECLARED, candidate_id=candidate_id, claim_id=claim_id,
                             attributes={"required_artifact_ids": [artifact_id],
                                         "artifact_bindings": {artifact_id: actual},
                                         "required_contract_ids": [c.contract_id for c in contracts],
                                         "published_artifact_path":str(published_path),
                                         "selected_object_id":object_id,
                                         "serialization_event_id":observed_binding.get("serialization_event_id"),
                                         "release_origin": "explicit_caller_requested_output_boundary"})
        receipt["release_event_id"] = event.event_id
    receipt["trace_sha256"] = trace_sha256(capture.events)
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    with receipt_path.open("x", encoding="utf8") as handle:
        json.dump(receipt, handle, sort_keys=True, indent=2)
    return receipt
