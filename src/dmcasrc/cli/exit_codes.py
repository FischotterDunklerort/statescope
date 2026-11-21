"""Stable CLI exit codes for use across the application."""

from __future__ import annotations

from enum import IntEnum


class ExitCode(IntEnum):
    SUCCESS = 0
    INPUT_OR_CONTRACT_ERROR = 2
    CLAIM_BLOCKED = 3
    INTERNAL_ERROR = 4


# ── Registered CLI stable codes (used by static error-code scanner) ──
# Each code must appear as a quoted string literal so the conformance
# scanner in tools/validation/check_contract_initial.py can discover it.
_CLI_CODES: tuple[str, ...] = (
    "CLI_USAGE_ERROR",
    "INPUT_FILE_NOT_FOUND",
    "INPUT_FILE_READ_FAILED",
    "JSON_DECODE_ERROR",
    "JSON_DUPLICATE_FIELD",
    "JSON_ROOT_INVALID",
    "SCHEMA_VALIDATION_FAILED",
    "SCHEMA_VERSION_UNSUPPORTED",
    "CROSS_OBJECT_REFERENCE_INVALID",
    "ARTIFACT_VERIFICATION_BLOCKED",
    "AUDIT_BUDGET_INVALID",
    "TRACE_CANDIDATE_UNKNOWN",
    "RUN_CONFIG_INVALID",
    "REPORT_INPUT_INVALID",
    "CLAIM_GATE_BLOCKED",
    "OUTPUT_WRITE_FAILED",
    "INTERNAL_ERROR",
)


class CliFailure(Exception):
    """Uniform CLI error with stable code and exit contract."""

    def __init__(
        self,
        code: str,
        detail: str,
        exit_code: ExitCode,
    ) -> None:
        super().__init__(detail)
        self.code = code
        self.detail = detail
        self.exit_code = exit_code


__all__ = ["ExitCode", "CliFailure"]
