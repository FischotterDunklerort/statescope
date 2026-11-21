"""Runtime JSON Schema validation using Draft 2020-12."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import jsonschema
from jsonschema import Draft202012Validator, FormatChecker, ValidationError

_SCHEMA_DIR = Path(__file__).resolve().parent.parent / "resources" / "schemas" / "initial"
# Fallback to repo-root spec directory during development
_REPO_StudyC_DIR = None
for parent in Path(__file__).resolve().parents:
    candidate = parent / "spec" / "schemas" / "initial"
    if candidate.is_dir():
        _REPO_StudyC_DIR = candidate
        break


def _schema_path(schema_name: str) -> Path:
    """Resolve a schema file by name (without .schema.json suffix)."""
    filename = f"{schema_name}.schema.json" if not schema_name.endswith(".schema.json") else schema_name
    for base in (_SCHEMA_DIR, _REPO_StudyC_DIR):
        if base is not None:
            candidate = base / filename
            if candidate.is_file():
                return candidate
    raise FileNotFoundError(f"schema file not found: {schema_name}")


def load_schema(schema_name: str) -> Mapping[str, Any]:
    """Load a JSON Schema document by name."""
    path = _schema_path(schema_name)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"schema root must be an object: {path}")
    return payload


def validator_for(schema_name: str) -> Draft202012Validator:
    """Return a Draft 2020-12 validator with FormatChecker for the given schema."""
    schema = load_schema(schema_name)
    return Draft202012Validator(schema, format_checker=FormatChecker())


def validate_payload(
    payload: Mapping[str, Any],
    schema_name: str,
) -> None:
    """Validate *payload* against the named schema.

    Raises :class:`jsonschema.ValidationError` on failure.
    """
    validator = validator_for(schema_name)
    errors = sorted(validator.iter_errors(payload), key=str)
    if errors:
        messages = []
        for err in errors:
            pointer = err.json_path if hasattr(err, "json_path") else ".".join(str(p) for p in err.absolute_path)
            messages.append(
                f"{schema_name} {pointer}: {err.message}"
            )
        raise ValidationError(
            "\n".join(messages),
            validator="jsonschema",
            instance=dict(payload),
        )


def validate_all_schema_documents() -> None:
    """Run ``check_schema`` on every known initial schema document.

    Raises :class:`jsonschema.SchemaError` on the first invalid schema.
    """
    schemas = [
        "pipeline-graph",
        "candidate-ledger",
        "artifact-registry",
        "claim-manifest",
        "audit-report",
        "contract-specification",
        "run-config",
        "error-code-catalog",
    ]
    for name in schemas:
        schema = load_schema(name)
        jsonschema.validators.validator_for(schema, default=Draft202012Validator).check_schema(schema)


__all__ = [
    "load_schema",
    "validator_for",
    "validate_payload",
    "validate_all_schema_documents",
]
