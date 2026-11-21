"""Stable JSON, CSV, and Markdown rendering for audit reports."""

from __future__ import annotations

import csv
import io
import json
from collections.abc import Mapping
from typing import Any

from dmcasrc.domain.contracts import AuditReport

_FINDING_FIELDS = (
    "candidate_key",
    "node_id",
    "layer",
    "obligation_id",
    "status",
    "error_code",
    "detail",
    "first_invalid_edge",
    "evidence_refs",
)


def _report(payload: Mapping[str, Any]) -> dict[str, Any]:
    try:
        report = AuditReport.from_dict(payload)
        return report.to_dict()
    except (ValueError, KeyError, TypeError) as exc:
        raise ValueError(f"invalid audit report: {exc}") from exc


def _csv_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return ";".join(str(item) for item in value)
    return str(value)


def _render_csv(payload: dict[str, Any]) -> str:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=_FINDING_FIELDS, lineterminator="\n")
    writer.writeheader()
    for finding in payload["findings"]:
        if not isinstance(finding, Mapping):
            raise ValueError("each finding must be an object.")
        writer.writerow({field: _csv_value(finding.get(field)) for field in _FINDING_FIELDS})
    return output.getvalue()


def _cell(value: Any) -> str:
    return _csv_value(value).replace("|", "\\|").replace("\n", " ")


def _candidate_node_labels(value: Any) -> str:
    if not isinstance(value, list):
        return "none"
    labels = [
        f"{item.get('candidate_key')}/{item.get('node_id')}"
        for item in value
        if isinstance(item, Mapping)
    ]
    return ", ".join(labels) or "none"


def _render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# DM-CASRC Audit Report",
        "",
        f"Contract: `{payload['contract_version']}`",
        "",
    ]
    execution = payload.get("execution")
    if isinstance(execution, Mapping):
        lines.append(f"Audit mode: `{execution.get('audit_mode')}`")
        stop = execution.get("stop_reason")
        if stop:
            lines.append(f"Stop reason: `{stop}`")
        lines.append(
            f"Audited: {execution.get('audited_obligation_count')} obligation(s) "
            f"across {execution.get('audited_candidate_count')} candidate(s)."
        )
        unaudited = execution.get("unaudited_candidate_keys") or []
        if unaudited:
            lines.append(f"Unaudited candidates: {', '.join(unaudited)}")
        lines.append(
            "Fully audited candidate-nodes: "
            + _candidate_node_labels(execution.get("fully_audited_candidate_nodes"))
        )
        lines.append(
            "Partially audited candidate-nodes: "
            + _candidate_node_labels(execution.get("partially_audited_candidate_nodes"))
        )
        lines.append(
            "Unaudited candidate-nodes: "
            + _candidate_node_labels(execution.get("unaudited_candidate_nodes"))
        )
        lines.append("")
    lines.extend([
        "| Candidate | Node | Layer | Obligation | Status | Error code | Detail | First invalid edge | Evidence refs |",
        "|---|---|---|---|---|---|---|---|---|",
    ])
    for item in payload["findings"]:
        lines.append(
            "| "
            + " | ".join(
                _cell(item.get(field))
                for field in _FINDING_FIELDS
            )
            + " |"
        )
    lines.extend(["", "## Claim decisions", ""])
    for decision in payload["claim_decisions"]:
        reasons = ", ".join(decision.get("reason_codes", [])) or "none"
        lines.append(f"- `{decision.get('claim_id')}`: **{decision.get('status')}** ({reasons})")
    return "\n".join(lines) + "\n"


def render_audit_report(payload: Mapping[str, Any], output_format: str) -> str:
    normalized = _report(payload)
    if output_format == "json":
        return json.dumps(normalized, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if output_format == "csv":
        return _render_csv(normalized)
    if output_format == "markdown":
        return _render_markdown(normalized)
    raise ValueError(f"unsupported report format: {output_format}.")


__all__ = ["render_audit_report"]
