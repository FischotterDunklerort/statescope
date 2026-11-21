"""Result-neutral closure checks for evidence adapters.

The scientific auditor intentionally fails closed when an observation points
outside a pipeline graph.  This module makes that invariant available as an
explicit *pre-injection* adapter qualification check.  It verifies wiring and
references only; it neither assigns a scientific verdict nor changes the
canonical audit or claim-gate semantics.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from dmcasrc.domain.contracts import CanonicalEvidenceFact, ScientificClaim

from .models import AdapterEvidenceBundle


_APPLICABILITY = frozenset({"SUPPORTED", "PARTIAL", "NOT_APPLICABLE"})


@dataclass(frozen=True)
class AdapterContractClosure:
    """Machine-readable result of structural adapter qualification."""

    passed: bool
    checks: Mapping[str, bool]
    errors: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "passed": self.passed,
            "checks": dict(sorted(self.checks.items())),
            "errors": list(self.errors),
        }


def _edge_nodes_valid(facts: Sequence[CanonicalEvidenceFact], node_ids: set[str]) -> bool:
    for fact in facts:
        edge = fact.declared_first_invalid_edge
        if edge is not None and (
            edge.source_id not in node_ids or edge.target_id not in node_ids
        ):
            return False
    return True


def _artifact_references_valid(bundle: AdapterEvidenceBundle) -> bool:
    artifact_ids = {item.artifact_id for item in bundle.artifacts.artifacts}
    for artifact in bundle.artifacts.artifacts:
        metadata = artifact.metadata
        config = metadata.get("config_artifact_id")
        if config is not None and config not in artifact_ids:
            return False
        for reference in metadata.get("training_data_refs", []):
            if reference not in artifact_ids:
                return False
    for record in bundle.ledger.records:
        evidence = record.evidence
        if record.outcome_ref.startswith("artifact:") and record.outcome_ref not in artifact_ids:
            return False
        for namespace, field in (
            ("model_identity", "artifact_id"),
            ("model_identity", "config_artifact_id"),
            ("threshold_identity", "artifact_id"),
        ):
            value = evidence.get(namespace, {}).get(field)
            if value is not None and value not in artifact_ids:
                return False
        for namespace, field in (
            ("model_identity", "training_data_refs"),
            ("artifact_integrity", "artifact_ids"),
            ("claim_binding", "artifact_ids"),
        ):
            for value in evidence.get(namespace, {}).get(field, []):
                if value not in artifact_ids:
                    return False
    return True


def validate_adapter_contract_closure(
    bundle: AdapterEvidenceBundle,
    facts: Sequence[CanonicalEvidenceFact],
    claims: Sequence[ScientificClaim],
    *,
    required_fault_ids: Iterable[str],
    declared_fault_applicability: Mapping[str, str],
    claim_artifact_bindings: Mapping[str, Sequence[str]],
    artifact_binding_semantic_ids: Mapping[str, str],
) -> AdapterContractClosure:
    """Validate graph, evidence, claim, artifact, and scope closure.

    ``artifact_binding_semantic_ids`` deliberately represents the identity of
    each *artifact binding* (artifact role + semantic content), not merely a
    content hash.  Thus a checkpoint may be compared across roles through its
    content identity while the registry still rejects duplicate binding IDs.
    """

    errors: list[str] = []
    graph_nodes = {node.node_id for node in bundle.graph.nodes}
    ledger_candidates = {record.candidate_key for record in bundle.ledger.records}
    ledger_nodes = {record.node_id for record in bundle.ledger.records}
    fact_nodes = {fact.node_id for fact in facts}
    # Canonical observation construction deliberately preserves the fact's
    # candidate/node/obligation coordinate.  This preflight verifies those
    # coordinates without importing or invoking the audit application layer.
    fact_coordinates = {
        (fact.candidate_id, fact.node_id, fact.obligation_id)
        for fact in facts
    }
    observation_nodes = fact_nodes
    artifact_ids = {artifact.artifact_id for artifact in bundle.artifacts.artifacts}

    checks: dict[str, bool] = {}
    checks["ledger_node_ids_exist_in_graph"] = ledger_nodes <= graph_nodes
    checks["fact_node_ids_exist_in_graph"] = fact_nodes <= graph_nodes
    checks["observation_node_ids_exist_in_graph"] = observation_nodes <= graph_nodes
    checks["fact_edges_reference_valid_nodes"] = _edge_nodes_valid(facts, graph_nodes)
    checks["artifact_references_exist"] = _artifact_references_valid(bundle)

    declared = {str(key): str(value) for key, value in declared_fault_applicability.items()}
    required = tuple(sorted({str(item) for item in required_fault_ids}))
    checks["required_faults_have_declared_applicability"] = (
        set(declared) == set(required)
        and all(value in _APPLICABILITY for value in declared.values())
    )

    claim_candidates_ok = True
    claim_nodes_ok = True
    claim_observations_ok = True
    claim_bindings_ok = set(claim_artifact_bindings) == {claim.claim_id for claim in claims}
    for claim in claims:
        if not set(claim.candidate_ids) <= ledger_candidates:
            claim_candidates_ok = False
        if claim.node_ids is not None and not set(claim.node_ids) <= graph_nodes:
            claim_nodes_ok = False
        for candidate_id in claim.candidate_ids:
            for obligation_id in claim.required_obligation_ids:
                if claim.node_ids is None:
                    present = any(
                        coordinate_candidate_id == candidate_id
                        and coordinate_obligation_id == obligation_id
                        for coordinate_candidate_id, _coordinate_node_id, coordinate_obligation_id in fact_coordinates
                    )
                    if not present:
                        claim_observations_ok = False
                else:
                    for node_id in claim.node_ids:
                        present = (candidate_id, node_id, obligation_id) in fact_coordinates
                        if not present:
                            claim_observations_ok = False
        for artifact_id in claim_artifact_bindings.get(claim.claim_id, ()):
            if artifact_id not in artifact_ids:
                claim_bindings_ok = False
    checks["claim_candidate_ids_exist"] = claim_candidates_ok
    checks["claim_node_ids_exist"] = claim_nodes_ok
    checks["claim_scope_has_observations"] = claim_observations_ok
    checks["claim_artifact_bindings_exist"] = claim_bindings_ok

    semantic_values = tuple(artifact_binding_semantic_ids.values())
    checks["artifact_binding_semantic_identities_unique"] = (
        bool(artifact_binding_semantic_ids)
        and all(isinstance(value, str) and value for value in semantic_values)
        and len(semantic_values) == len(set(semantic_values))
        and set(artifact_binding_semantic_ids) <= artifact_ids
    )

    for name, passed in checks.items():
        if not passed:
            errors.append(name)
    return AdapterContractClosure(
        passed=not errors,
        checks=checks,
        errors=tuple(sorted(errors)),
    )


__all__ = ["AdapterContractClosure", "validate_adapter_contract_closure"]
