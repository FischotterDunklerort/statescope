"""Command-line entry point for validation, audit, reporting, and reproduction."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from dmcasrc.application.audit import audit, audit_budgeted, audit_candidate
from dmcasrc.cli.exit_codes import CliFailure, ExitCode
from dmcasrc.domain.contracts import AuditReport, ClaimManifest, ContractSpecification
from dmcasrc.domain.graph import PipelineGraph
from dmcasrc.domain.ledger import CandidateLedger
from dmcasrc.infrastructure.artifact_writer import ArtifactWriter
from dmcasrc.infrastructure.deployment_manifest_loader import (
    load_artifact_manifest,
    verify_artifact_manifest,
)
from dmcasrc.infrastructure.schema_validation import validate_payload
from dmcasrc.reporting.outputs import render_audit_report


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON field: {key}.")
        result[key] = value
    return result


def _load_object(path: str | Path, *, schema_name: str = "") -> Mapping[str, Any]:
    source = Path(path)
    if not source.is_file():
        raise CliFailure(
            "INPUT_FILE_NOT_FOUND",
            f"input not found: {source}",
            ExitCode.INPUT_OR_CONTRACT_ERROR,
        )
    try:
        raw = source.read_text(encoding="utf-8")
    except OSError as exc:
        raise CliFailure(
            "INPUT_FILE_READ_FAILED",
            f"cannot read {source}: {exc}",
            ExitCode.INPUT_OR_CONTRACT_ERROR,
        ) from exc
    try:
        payload = json.loads(raw, object_pairs_hook=_unique_object)
    except json.JSONDecodeError as exc:
        raise CliFailure(
            "JSON_DECODE_ERROR",
            f"invalid JSON in {source}: {exc}",
            ExitCode.INPUT_OR_CONTRACT_ERROR,
        ) from exc
    except ValueError as exc:
        if "duplicate" in str(exc).lower():
            raise CliFailure(
                "JSON_DUPLICATE_FIELD",
                f"duplicate JSON field in {source}: {exc}",
                ExitCode.INPUT_OR_CONTRACT_ERROR,
            ) from exc
        raise
    if not isinstance(payload, Mapping):
        raise CliFailure(
            "JSON_ROOT_INVALID",
            f"JSON root must be an object: {source}",
            ExitCode.INPUT_OR_CONTRACT_ERROR,
        )
    if schema_name:
        version_key = None
        if schema_name == "pipeline-graph":
            version_key = "schema_version"
            expected = "dmcasrc.pipeline-graph.initial"
        elif schema_name == "candidate-ledger":
            version_key = "schema_version"
            expected = "dmcasrc.candidate-ledger.initial"
        elif schema_name == "artifact-registry":
            version_key = "schema_version"
            expected = "dmcasrc.artifact-registry.initial"
        elif schema_name == "claim-manifest":
            version_key = "schema_version"
            expected = "dmcasrc.claim-manifest.initial"
        elif schema_name == "audit-report":
            version_key = "contract_version"
            expected = "dmcasrc.contract.initial"
        elif schema_name == "run-config":
            version_key = "schema_version"
            expected = "dmcasrc.run-config.initial"
        if version_key and payload.get(version_key) != expected:
            raise CliFailure(
                "SCHEMA_VERSION_UNSUPPORTED",
                f"{schema_name}: unsupported {version_key}: {payload.get(version_key)}",
                ExitCode.INPUT_OR_CONTRACT_ERROR,
            )
        try:
            validate_payload(payload, schema_name)
        except Exception as exc:
            raise CliFailure(
                "SCHEMA_VALIDATION_FAILED",
                f"{schema_name} validation failed: {exc}",
                ExitCode.INPUT_OR_CONTRACT_ERROR,
            ) from exc
    return payload


def _write_output(path: str | Path, content: str, *, kind: str) -> None:
    output = Path(path)
    writer = ArtifactWriter(output.parent)
    try:
        writer.write_text(
            output.name,
            content,
            artifact_id=f"{kind}-output",
            kind=kind,
            semantic_id=f"{kind}:dmcasrc.contract.initial",
            media_type="application/json" if output.suffix == ".json" else "text/plain",
        )
    except OSError as exc:
        raise CliFailure(
            "OUTPUT_WRITE_FAILED",
            f"cannot write output to {path}: {exc}",
            ExitCode.INTERNAL_ERROR,
        ) from exc


def _load_inputs(args: argparse.Namespace):
    try:
        graph = PipelineGraph.from_dict(_load_object(args.graph, schema_name="pipeline-graph"))
        ledger = CandidateLedger.from_dict(_load_object(args.ledger, schema_name="candidate-ledger"))
        registry = load_artifact_manifest(args.registry)
        claims = ClaimManifest.from_dict(_load_object(args.claims, schema_name="claim-manifest"))
        verifications = verify_artifact_manifest(registry, root=args.artifact_root)
    except (ValueError, KeyError, TypeError, OSError) as exc:
        raise CliFailure(
            "CROSS_OBJECT_REFERENCE_INVALID",
            str(exc),
            ExitCode.INPUT_OR_CONTRACT_ERROR,
        ) from exc
    return graph, ledger, registry, claims, verifications


def _validate(args: argparse.Namespace) -> int:
    graph, ledger, registry, claims, verifications = _load_inputs(args)
    if any(item.status != "PASS" for item in verifications):
        failures = ", ".join(
            f"{item.artifact_id}:{item.error_code}"
            for item in verifications
            if item.status != "PASS"
        )
        raise CliFailure(
            "ARTIFACT_VERIFICATION_BLOCKED",
            f"artifact verification did not pass: {failures}",
            ExitCode.INPUT_OR_CONTRACT_ERROR,
        )
    known_nodes = {node.node_id for node in graph.nodes}
    unknown = sorted({record.node_id for record in ledger.records} - known_nodes)
    if unknown:
        raise CliFailure(
            "CROSS_OBJECT_REFERENCE_INVALID",
            f"ledger references unknown nodes: {unknown}",
            ExitCode.INPUT_OR_CONTRACT_ERROR,
        )
    known_obligations = {item.obligation_id for item in ContractSpecification.default().obligations}
    claimed = {item for claim in claims.claims for item in claim.required_obligation_ids}
    if claimed - known_obligations:
        raise CliFailure(
            "CROSS_OBJECT_REFERENCE_INVALID",
            f"claims reference unknown obligations: {sorted(claimed - known_obligations)}",
            ExitCode.INPUT_OR_CONTRACT_ERROR,
        )
    known_candidates = {record.candidate_key for record in ledger.records}
    for claim in claims.claims:
        if claim.scope.candidate_ids is not None:
            unknown_candidates = sorted(
                set(claim.scope.candidate_ids) - known_candidates
            )
            if unknown_candidates:
                raise CliFailure(
                    "CROSS_OBJECT_REFERENCE_INVALID",
                    f"claim '{claim.claim_id}' references unknown candidates: {unknown_candidates}",
                    ExitCode.INPUT_OR_CONTRACT_ERROR,
                )
        if claim.scope.node_ids is not None:
            unknown_scope_nodes = sorted(
                set(claim.scope.node_ids) - known_nodes
            )
            if unknown_scope_nodes:
                raise CliFailure(
                    "CROSS_OBJECT_REFERENCE_INVALID",
                    f"claim '{claim.claim_id}' references unknown nodes: {unknown_scope_nodes}",
                    ExitCode.INPUT_OR_CONTRACT_ERROR,
                )
    print(
        json.dumps(
            {
                "status": "PASS",
                "nodes": len(graph.nodes),
                "candidates": len(ledger.records),
                "artifacts": len(registry.artifacts),
                "claims": len(claims.claims),
            },
            sort_keys=True,
        )
    )
    return 0


def _audit(args: argparse.Namespace) -> int:
    graph, ledger, registry, claims, verifications = _load_inputs(args)
    report = audit(
        graph,
        ledger,
        registry,
        ContractSpecification.default(),
        claim_manifest=claims,
        artifact_verifications=verifications,
    )
    report_json = render_audit_report(report.to_dict(), "json")
    _write_output(args.output, report_json, kind="audit-report")
    if any(
        decision.status == "BLOCKED"
        for decision in report.claim_decisions
    ):
        # Report has been written above; surface a stable, machine-readable code.
        print(
            f"dmcasrc[CLAIM_GATE_BLOCKED]: one or more claims were blocked; "
            f"audit report written to {args.output}",
            file=sys.stderr,
        )
        return int(ExitCode.CLAIM_BLOCKED)
    return 0


def _report(args: argparse.Namespace) -> int:
    payload = _load_object(args.input, schema_name="audit-report")
    try:
        report = AuditReport.from_dict(payload)
        rendered = render_audit_report(report.to_dict(), args.format)
    except (ValueError, KeyError, TypeError) as exc:
        raise CliFailure(
            "REPORT_INPUT_INVALID",
            str(exc),
            ExitCode.INPUT_OR_CONTRACT_ERROR,
        ) from exc
    _write_output(args.output, rendered, kind=f"audit-report-{args.format}")
    return 0


def _scan(args: argparse.Namespace) -> int:
    if args.budget < 0:
        raise CliFailure(
            code="AUDIT_BUDGET_INVALID",
            detail="budget must be a non-negative integer",
            exit_code=ExitCode.INPUT_OR_CONTRACT_ERROR,
        )
    graph, ledger, registry, claims, verifications = _load_inputs(args)
    report = audit_budgeted(
        graph,
        ledger,
        registry,
        ContractSpecification.default(),
        claim_manifest=claims,
        artifact_verifications=verifications,
        budget=args.budget,
    )
    report_json = render_audit_report(report.to_dict(), "json")
    _write_output(args.output, report_json, kind="audit-report")
    if any(
        decision.status == "BLOCKED"
        for decision in report.claim_decisions
    ):
        print(
            f"dmcasrc[CLAIM_GATE_BLOCKED]: one or more claims were blocked; "
            f"audit report written to {args.output}",
            file=sys.stderr,
        )
        return int(ExitCode.CLAIM_BLOCKED)
    return 0


def _trace(args: argparse.Namespace) -> int:
    graph, ledger, registry, claims, verifications = _load_inputs(args)
    known_candidates = {record.candidate_key for record in ledger.records}
    if args.candidate not in known_candidates:
        raise CliFailure(
            code="TRACE_CANDIDATE_UNKNOWN",
            detail=f"candidate_key not found in ledger: {args.candidate!r}",
            exit_code=ExitCode.INPUT_OR_CONTRACT_ERROR,
        )
    report = audit_candidate(
        graph,
        ledger,
        registry,
        ContractSpecification.default(),
        claim_manifest=claims,
        artifact_verifications=verifications,
        candidate_key=args.candidate,
    )
    report_json = render_audit_report(report.to_dict(), "json")
    _write_output(args.output, report_json, kind="audit-report")
    if any(
        decision.status == "BLOCKED"
        for decision in report.claim_decisions
    ):
        print(
            f"dmcasrc[CLAIM_GATE_BLOCKED]: one or more claims were blocked; "
            f"audit report written to {args.output}",
            file=sys.stderr,
        )
        return int(ExitCode.CLAIM_BLOCKED)
    return 0


def _config_command(args: argparse.Namespace) -> int:
    config_path = Path(args.config).resolve()
    config = _load_object(config_path, schema_name="run-config")
    if config.get("schema_version") != "dmcasrc.run-config.initial":
        raise CliFailure(
            "RUN_CONFIG_INVALID",
            f"run config has unsupported schema_version: "
            f"{config.get('schema_version')!r}",
            ExitCode.INPUT_OR_CONTRACT_ERROR,
        )
    base = config_path.parent
    forwarded = argparse.Namespace(
        graph=base / config["graph"],
        ledger=base / config["ledger"],
        registry=base / config["registry"],
        claims=base / config["claims"],
        artifact_root=base / config["artifact_root"],
        output=args.output,
    )
    return _audit(forwarded)


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--graph", required=True)
    parser.add_argument("--ledger", required=True)
    parser.add_argument("--registry", required=True)
    parser.add_argument("--claims", required=True)
    parser.add_argument("--artifact-root", required=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dmcasrc",
        description="Candidate-level evidence-contract audit system.",
    )
    parser.add_argument("--version", action="version", version="dmcasrc 0.1.0")
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate", help="validate graph, ledger, claims, and artifacts")
    _common(validate)
    validate.set_defaults(handler=_validate)
    audit_parser = commands.add_parser("audit", help="run the full candidate-level audit")
    _common(audit_parser)
    audit_parser.add_argument("--output", required=True)
    audit_parser.set_defaults(handler=_audit)
    report = commands.add_parser("report", help="render an audit report")
    report.add_argument("--input", required=True)
    report.add_argument("--format", choices=("json", "csv", "markdown"), required=True)
    report.add_argument("--output", required=True)
    report.set_defaults(handler=_report)
    scan_parser = commands.add_parser("scan", help="run a budgeted (resource-limited) audit")
    _common(scan_parser)
    scan_parser.add_argument("--budget", type=int, required=True)
    scan_parser.add_argument("--output", required=True)
    scan_parser.set_defaults(handler=_scan)
    trace_parser = commands.add_parser("trace", help="trace a single candidate's obligation chain")
    _common(trace_parser)
    trace_parser.add_argument("--candidate", required=True)
    trace_parser.add_argument("--output", required=True)
    trace_parser.set_defaults(handler=_trace)
    for name in ("benchmark", "reproduce"):
        command = commands.add_parser(name, help=f"run a frozen {name} configuration")
        command.add_argument("--config", required=True)
        command.add_argument("--output", required=True)
        command.set_defaults(handler=_config_command)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    raw_argv = list(argv) if argv is not None else sys.argv[1:]
    if raw_argv[:3] == ["benchmark", "fault", "run"]:
        from benchmarks.projects.faults.run import main as fault_main

        try:
            return int(fault_main(raw_argv[3:]))
        except SystemExit as exc:
            return int(exc.code) if exc.code is not None else 1
    try:
        try:
            args = build_parser().parse_args(raw_argv)
        except SystemExit as exc:
            code = int(exc.code) if exc.code is not None else 1
            if code == 2:
                # argparse rejected argv → CLI_USAGE_ERROR
                raise CliFailure(
                    "CLI_USAGE_ERROR",
                    "command-line arguments invalid",
                    ExitCode.INPUT_OR_CONTRACT_ERROR,
                ) from exc
            raise
        return int(args.handler(args))
    except CliFailure as exc:
        print(
            f"dmcasrc[{exc.code}]: {exc.detail}",
            file=sys.stderr,
        )
        return int(exc.exit_code)
    except SystemExit as exc:
        return int(exc.code) if exc.code is not None else 1
    except Exception as exc:
        print(
            f"dmcasrc[INTERNAL_ERROR]: "
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 4


__all__ = ["build_parser", "main"]


if __name__ == "__main__":
    raise SystemExit(main())
