"""Scenario-neutral native-fact semantics shared by project adapters."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from dmcasrc.adapters.identity import semantic_hash, stable_id
from dmcasrc.domain.contracts import (
    CANONICAL_LAYER_IDS,
    CANONICAL_OBLIGATION_IDS,
    CanonicalEvidenceFact,
)
from dmcasrc.domain.graph import CanonicalEdge


@dataclass(frozen=True)
class NativeSemanticRule:
    source_profile: str
    native_fact_type: str
    obligation_id: str
    primary_layer: str
    affected_layers: tuple[str, ...]
    first_invalid_edge: CanonicalEdge | None
    missing_required_evidence: tuple[str, ...] = ()
    witness_policy: str = "DEPENDENT_CANDIDATES"
    observation_node_id: str | None = None

    def __post_init__(self) -> None:
        for name in ("source_profile", "native_fact_type"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"{name} must be a non-empty string.")
        if self.obligation_id not in CANONICAL_OBLIGATION_IDS:
            raise ValueError(f"unknown canonical obligation: {self.obligation_id}.")
        if self.primary_layer not in CANONICAL_LAYER_IDS:
            raise ValueError(f"unknown canonical layer: {self.primary_layer}.")
        if (
            not isinstance(self.affected_layers, tuple)
            or not self.affected_layers
            or self.primary_layer not in self.affected_layers
            or set(self.affected_layers) - set(CANONICAL_LAYER_IDS)
        ):
            raise ValueError("affected_layers must include the canonical primary layer.")
        if not isinstance(self.missing_required_evidence, tuple) or any(
            not isinstance(item, str) or not item for item in self.missing_required_evidence
        ):
            raise ValueError("missing_required_evidence must be a tuple of strings.")
        if self.missing_required_evidence and self.first_invalid_edge is not None:
            raise ValueError("capability-boundary rules cannot assert an invalid edge.")
        if self.witness_policy not in {
            "DEPENDENT_CANDIDATES",
            "CLEAN_MINUS_OBSERVED_CANDIDATES",
        }:
            raise ValueError("unknown native-fact witness policy.")
        if self.observation_node_id is not None and (
            not isinstance(self.observation_node_id, str)
            or not self.observation_node_id
        ):
            raise ValueError("observation_node_id must be a non-empty string when set.")
        if self.first_invalid_edge is not None and self.observation_node_id is not None:
            raise ValueError("observation_node_id is reserved for edge-free rules.")


@dataclass(frozen=True)
class NativeFactPacket:
    """Raw native delta packet; it intentionally carries no audit status."""

    source_profile: str
    native_object_ref: str
    native_source_path: str
    native_fact_type: str
    evidence_sha256: str
    evidence_ref: str
    clean_value: Any
    observed_value: Any
    candidate_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in (
            "source_profile",
            "native_object_ref",
            "native_source_path",
            "native_fact_type",
            "evidence_ref",
        ):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"{name} must be a non-empty string.")
        if (
            not isinstance(self.evidence_sha256, str)
            or len(self.evidence_sha256) != 64
            or any(item not in "0123456789abcdef" for item in self.evidence_sha256)
        ):
            raise ValueError("evidence_sha256 must be lowercase SHA-256.")
        if (
            not isinstance(self.candidate_ids, tuple)
            or not self.candidate_ids
            or any(not isinstance(item, str) or not item for item in self.candidate_ids)
            or len(self.candidate_ids) != len(set(self.candidate_ids))
        ):
            raise ValueError("candidate_ids must be a unique non-empty tuple.")
        semantic_hash(self.clean_value)
        semantic_hash(self.observed_value)


def native_rule(
    source_profile: str,
    native_fact_type: str,
    obligation_id: str,
    primary_layer: str,
    affected_layers: tuple[str, ...],
    edge: tuple[str, str, str] | None,
    *,
    missing_required_evidence: tuple[str, ...] = (),
    witness_policy: str = "DEPENDENT_CANDIDATES",
    observation_node_id: str | None = None,
) -> NativeSemanticRule:
    return NativeSemanticRule(
        source_profile=source_profile,
        native_fact_type=native_fact_type,
        obligation_id=obligation_id,
        primary_layer=primary_layer,
        affected_layers=affected_layers,
        first_invalid_edge=None if edge is None else CanonicalEdge(*edge),
        missing_required_evidence=missing_required_evidence,
        witness_policy=witness_policy,
        observation_node_id=observation_node_id,
    )


def project_native_fact(
    packet: NativeFactPacket,
    rules: Sequence[NativeSemanticRule],
) -> tuple[CanonicalEvidenceFact, ...]:
    """Map one native delta packet into traceable candidate-level facts."""
    matches = [item for item in rules if item.native_fact_type == packet.native_fact_type]
    if len(matches) != 1:
        raise ValueError(
            f"native fact type must resolve to exactly one rule: {packet.native_fact_type}"
        )
    rule = matches[0]
    if rule.source_profile != packet.source_profile:
        raise ValueError("native fact packet source profile does not match its rule.")
    changed = packet.clean_value != packet.observed_value
    node_id = (
        rule.first_invalid_edge.target_id
        if rule.first_invalid_edge is not None
        else rule.observation_node_id or f"capability.{rule.primary_layer.lower()}"
    )
    facts = []
    for candidate_id in sorted(packet.candidate_ids):
        payload: dict[str, Any] = {
            "expected": packet.clean_value,
            "observed": packet.observed_value,
        }
        if changed and rule.missing_required_evidence:
            payload["missing_required_evidence"] = list(rule.missing_required_evidence)
        canonical_identity = stable_id(
            f"{packet.source_profile}.canonical-fact",
            {
                "candidate_id": candidate_id,
                "native_fact_type": packet.native_fact_type,
                "obligation_id": rule.obligation_id,
                "node_id": node_id,
            },
        )
        facts.append(
            CanonicalEvidenceFact(
                fact_id=stable_id(
                    "dmcasrc.native-fact",
                    {"canonical_identity": canonical_identity},
                ),
                candidate_id=candidate_id,
                node_id=node_id,
                obligation_id=rule.obligation_id,
                primary_layer=rule.primary_layer,
                affected_layers=rule.affected_layers,
                source_profile=packet.source_profile,
                native_object_ref=packet.native_object_ref,
                native_source_path=packet.native_source_path,
                native_fact_type=packet.native_fact_type,
                canonical_identity=canonical_identity,
                evidence_sha256=packet.evidence_sha256,
                evidence_ref=packet.evidence_ref,
                payload=payload,
                declared_first_invalid_edge=rule.first_invalid_edge,
            )
        )
    return tuple(facts)


__all__ = [
    "NativeFactPacket",
    "NativeSemanticRule",
    "native_rule",
    "project_native_fact",
]
