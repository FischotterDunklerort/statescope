"""Outcome-blind violation-causal primary finding resolution.

G6.4-T2 resolves a *private graph of violation findings*, rather than treating
the canonical pipeline/evidence graph as a causal graph.  A directed edge
``A -> B`` means that runtime-visible structure identifies ``B`` as a
downstream or derived consequence of ``A``.  It is derived only from finding
nodes, finding kinds, pipeline ancestry, and declared first-invalid-edge
relations; it never reads reference labels, results, scenario identifiers, or
fault metadata.

``resolve_primary`` remains a compatibility resolver.  The FULL
mechanism path uses ``resolve_minimal_causal_roots`` to retain every
independent root and choose a canonical root only after building this
violation-causal graph.  The no-resolver ablation remains independent in
``mechanism_policy._topological_resolution``.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from ..domain.contracts import _CANONICAL_OBLIGATION_DEFINITIONS, ScientificAuditFinding
from ..domain.graph import PipelineGraph

EDGE_NOT_APPLICABLE = "EDGE_NOT_APPLICABLE_TO_IDENTITY_VIOLATION"


class FindingKind(StrEnum):
    DIRECT_NATIVE_VIOLATION = "DIRECT_NATIVE_VIOLATION"
    PROPAGATED_CONSEQUENCE = "PROPAGATED_CONSEQUENCE"
    DIRECT_IDENTITY_VIOLATION = "DIRECT_IDENTITY_VIOLATION"
    DERIVED_CONTRACT_FAILURE = "DERIVED_CONTRACT_FAILURE"


# obligation semantic families derived from the canonical contract only
_IDENTITY_OBLIGATIONS = frozenset(
    item[0]
    for item in _CANONICAL_OBLIGATION_DEFINITIONS
    if item[0].endswith("_IDENTITY") or item[0] in ("ANCESTOR_CLOSURE", "CANDIDATE_IDENTITY")
)
_CONTRACT_OBLIGATIONS = frozenset(
    item[0]
    for item in _CANONICAL_OBLIGATION_DEFINITIONS
    if item[0] in (
        "CLAIM_DEPENDENCY_INTEGRITY",
        "OUTCOME_FEEDBACK_ISOLATION",
        "SELECTION_EXECUTION_CONSISTENCY",
        "PREDICTION_ORIGIN",
        "ASSESSMENT_SELECTION_ISOLATION",
    )
)


@dataclass(frozen=True)
class FindingClassification:
    finding_id: str
    kind: FindingKind
    node_id: str | None
    invalid_edge: Any
    evidence_count: int


@dataclass(frozen=True)
class PrimaryResolution:
    primary_finding_id: str | None
    primary_kind: FindingKind | None
    primary_layer: str | None
    primary_obligation: str | None
    first_invalid_edge: Any
    edge_not_applicable_reason: str | None
    affected_finding_ids: tuple[str, ...]
    all_invalid_edges: tuple[Any, ...]
    affected_layers: tuple[str, ...]
    affected_obligations: tuple[str, ...]
    classifications: tuple[FindingClassification, ...] = field(default_factory=tuple)
    # G6.4-T2 runtime instrumentation.  ``None`` denotes a legacy resolution
    # path for which the causal resolver was intentionally not evaluated.
    finding_count: int | None = None
    violating_finding_count: int | None = None
    direct_native_candidate_count: int | None = None
    direct_identity_candidate_count: int | None = None
    derived_candidate_count: int | None = None
    propagated_candidate_count: int | None = None
    resolver_opportunity: bool | None = None
    default_primary_finding: str | None = None
    resolved_primary_finding: str | None = None
    resolver_changed_choice: bool | None = None
    resolver_tie_count: int | None = None
    causal_edge_count: int | None = None
    causal_edges: tuple[tuple[str, str], ...] = ()
    causal_root_candidate_count: int | None = None
    minimal_root_set: tuple[str, ...] = ()
    minimal_root_set_size: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "primary_finding_id": self.primary_finding_id,
            "primary_kind": self.primary_kind.value if self.primary_kind else None,
            "primary_layer": self.primary_layer,
            "primary_obligation": self.primary_obligation,
            "first_invalid_edge": (
                self.first_invalid_edge.to_dict()
                if hasattr(self.first_invalid_edge, "to_dict")
                else (self.first_invalid_edge if self.first_invalid_edge != EDGE_NOT_APPLICABLE else None)
            ),
            "edge_not_applicable_reason": self.edge_not_applicable_reason,
            "affected_finding_ids": list(self.affected_finding_ids),
            "all_invalid_edges": [
                e.to_dict() if hasattr(e, "to_dict") else e for e in self.all_invalid_edges
            ],
            "affected_layers": list(self.affected_layers),
            "affected_obligations": list(self.affected_obligations),
            "finding_count": self.finding_count,
            "violating_finding_count": self.violating_finding_count,
            "direct_native_candidate_count": self.direct_native_candidate_count,
            "direct_identity_candidate_count": self.direct_identity_candidate_count,
            "derived_candidate_count": self.derived_candidate_count,
            "propagated_candidate_count": self.propagated_candidate_count,
            "resolver_opportunity": self.resolver_opportunity,
            "default_primary_finding": self.default_primary_finding,
            "resolved_primary_finding": self.resolved_primary_finding,
            "resolver_changed_choice": self.resolver_changed_choice,
            "resolver_tie_count": self.resolver_tie_count,
            "causal_edge_count": self.causal_edge_count,
            "causal_edges": [
                {"parent": parent, "child": child} for parent, child in self.causal_edges
            ],
            "causal_root_candidate_count": self.causal_root_candidate_count,
            "minimal_root_set": list(self.minimal_root_set),
            "minimal_root_set_size": self.minimal_root_set_size,
        }


def _edge_source(edge: Any) -> str | None:
    if edge is None:
        return None
    if isinstance(edge, Mapping):
        return edge.get("source_id")
    return getattr(edge, "source_id", None)


def _edge_target(edge: Any) -> str | None:
    if edge is None:
        return None
    if isinstance(edge, Mapping):
        return edge.get("target_id")
    return getattr(edge, "target_id", None)


def _finding_node_id(finding: ScientificAuditFinding) -> str | None:
    """Derive the node id from finding fields (observation id encodes node)."""
    # observation/finding ids follow "<candidate>:<node>:<obligation>" in this
    # contract; fall back to edge source if present.
    fid = finding.finding_id or ""
    parts = fid.split(":")
    if len(parts) >= 3:
        return parts[1] if len(parts) >= 3 else None
    return _edge_source(finding.first_invalid_edge)


def _violated_node_ids(findings: Sequence[ScientificAuditFinding]) -> set[str]:
    out = set()
    for f in findings:
        if f.status != "VIOLATION":
            continue
        n = _finding_node_id(f)
        if n:
            out.add(n)
    return out


def classify_finding(
    finding: ScientificAuditFinding,
    graph: PipelineGraph,
    violated_nodes: set[str],
) -> FindingClassification:
    obligation = finding.obligation_id
    edge = finding.first_invalid_edge
    node = _finding_node_id(finding)
    evidence_count = len(finding.evidence_refs)

    if obligation in _IDENTITY_OBLIGATIONS:
        kind = FindingKind.DIRECT_IDENTITY_VIOLATION
    elif obligation in _CONTRACT_OBLIGATIONS:
        kind = FindingKind.DERIVED_CONTRACT_FAILURE
    else:
        node = _finding_node_id(finding)
        other_violated = violated_nodes - {node} if node else violated_nodes
        propagated = False
        if node and other_violated:
            try:
                ancestors = set(graph.ancestors(node))
            except KeyError:
                ancestors = set()
            propagated = bool(ancestors & other_violated)
        edge_source = _edge_source(edge)
        edge_target = _edge_target(edge)
        # The declared first-invalid edge is runtime-visible dependency
        # structure.  It can expose a causal relation not represented by the
        # pipeline graph's provenance/topology alone.
        if (
            not propagated
            and node is not None
            and edge_source in other_violated
            and edge_target == node
        ):
            propagated = True
        kind = (
            FindingKind.PROPAGATED_CONSEQUENCE if propagated else FindingKind.DIRECT_NATIVE_VIOLATION
        )
    return FindingClassification(
        finding_id=finding.finding_id,
        kind=kind,
        node_id=node,
        invalid_edge=edge,
        evidence_count=evidence_count,
    )


def _graph_position(graph: PipelineGraph, node_id: str | None) -> int:
    if node_id is None:
        return 0
    try:
        order = graph.topological_order()
    except Exception:  # noqa: BLE001
        return 0
    if node_id in order:
        return order.index(node_id)
    return 0


@dataclass(frozen=True)
class ViolationCausalGraph:
    """Private causal graph over violation findings.

    Nodes are violation findings.  An edge ``parent -> child`` is present only
    when a finding is structurally known to be upstream of a derived or
    propagated finding.  The pipeline graph is evidence for an edge; it is not
    itself this causal graph.
    """

    findings: tuple[ScientificAuditFinding, ...]
    classifications: tuple[FindingClassification, ...]
    edges: tuple[tuple[str, str], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "dmcasrc.violation-causal-graph.initial",
            "nodes": [
                {
                    "finding_id": item.finding_id,
                    "kind": item.kind.value,
                    "node_id": item.node_id,
                }
                for item in self.classifications
            ],
            "edges": [
                {"parent": parent, "child": child} for parent, child in self.edges
            ],
        }

    def root_finding_ids(self) -> tuple[str, ...]:
        children = {child for _, child in self.edges}
        return tuple(
            item.finding_id
            for item in self.classifications
            if item.finding_id not in children
        )


def _is_known_upstream_cause(
    parent: FindingClassification,
    child: FindingClassification,
    graph: PipelineGraph,
) -> bool:
    """Return whether known runtime structure supports ``parent -> child``.

    A relation must be explicit in one of the permitted structural sources:
    same-node direct-to-derived/propagated semantics, pipeline ancestry, or a
    declared first-invalid edge.  No relation is invented for merely adjacent
    or independently failing findings.
    """
    parent_node = parent.node_id
    child_node = child.node_id
    if parent_node is None:
        return False
    if child_node == parent_node:
        return True

    child_edge_source = _edge_source(child.invalid_edge)
    child_edge_target = _edge_target(child.invalid_edge)
    if child_node is not None and (
        child_edge_source == parent_node and child_edge_target == child_node
    ):
        return True

    parent_edge_source = _edge_source(parent.invalid_edge)
    parent_edge_target = _edge_target(parent.invalid_edge)
    if child_node is not None and (
        parent_edge_source == parent_node and parent_edge_target == child_node
    ):
        return True

    if child_node is None:
        return False
    try:
        return parent_node in set(graph.ancestors(child_node))
    except KeyError:
        return False


def build_violation_causal_graph(
    findings: Sequence[ScientificAuditFinding],
    graph: PipelineGraph,
    violated_nodes: set[str] | None = None,
) -> ViolationCausalGraph:
    """Build the private violation-causal graph from runtime structure."""
    violations = tuple(
        sorted(
            (item for item in findings if item.status == "VIOLATION"),
            key=lambda item: item.finding_id,
        )
    )
    if violated_nodes is None:
        violated_nodes = {
            item for item in (_finding_node_id(f) for f in violations) if item
        }
    classified = [
        classify_finding(item, graph, set(violated_nodes)) for item in violations
    ]
    edges: list[tuple[str, str]] = []
    for parent in classified:
        for child in classified:
            if child.finding_id == parent.finding_id:
                continue
            # Direct findings can be independent roots.  Only known derived
            # or propagated consequences may be suppressed by an upstream
            # direct cause.
            if child.kind in {
                FindingKind.DIRECT_NATIVE_VIOLATION,
                FindingKind.DIRECT_IDENTITY_VIOLATION,
            }:
                continue
            if _is_known_upstream_cause(parent, child, graph):
                edges.append((parent.finding_id, child.finding_id))
    return ViolationCausalGraph(
        findings=tuple(violations),
        classifications=tuple(classified),
        edges=tuple(sorted(set(edges))),
    )


def _canonical_graph_position(graph: PipelineGraph, node_id: str | None) -> int:
    """Return a deterministic rank, placing unknown nodes after known ones."""
    try:
        order = graph.topological_order()
    except Exception:  # noqa: BLE001
        return 0
    return order.index(node_id) if node_id in order else len(order)


def _canonical_root_sort_key(
    item: ScientificAuditFinding,
    classifications: Mapping[str, FindingClassification],
    graph: PipelineGraph,
) -> tuple[int, str, str]:
    """Outcome-blind canonical tie-break for independent causal roots.

    Direct-native and direct-identity findings intentionally receive no
    hard-coded relative priority.  A known causal relation determines root
    membership; remaining independent roots are ordered only by structural
    rank and canonical identifiers.
    """
    classification = classifications[item.finding_id]
    return (
        _canonical_graph_position(graph, classification.node_id),
        item.obligation_id,
        item.finding_id,
    )


def _legacy_topological_primary_finding_id(
    violations: Sequence[ScientificAuditFinding],
    graph: PipelineGraph,
) -> str | None:
    """Compatibility ordering used only when a caller did not supply A2's ID.

    ``mechanism_policy._topological_resolution`` remains the actual A2 path.
    This small, resolver-free helper exists solely so direct callers of
    the FULL resolver receive complete diagnostic fields.
    """
    if not violations:
        return None
    try:
        node_rank = {node: index for index, node in enumerate(graph.topological_order())}
    except Exception:  # noqa: BLE001
        node_rank = {}

    def sort_key(item: ScientificAuditFinding) -> tuple[int, str, str]:
        return (
            node_rank.get(_finding_node_id(item), len(node_rank)),
            item.obligation_id,
            item.finding_id,
        )

    return min(violations, key=sort_key).finding_id


def _resolution_from_causal_primary(
    violations: Sequence[ScientificAuditFinding],
    classifications: Sequence[FindingClassification],
    primary: ScientificAuditFinding | None,
    **instrumentation: Any,
) -> PrimaryResolution:
    """Build a primary-resolution record without discarding any finding."""
    if primary is None:
        return PrimaryResolution(
            primary_finding_id=None,
            primary_kind=None,
            primary_layer=None,
            primary_obligation=None,
            first_invalid_edge=None,
            edge_not_applicable_reason=None,
            affected_finding_ids=tuple(item.finding_id for item in violations),
            all_invalid_edges=(),
            affected_layers=tuple(sorted({item.primary_layer for item in violations})),
            affected_obligations=tuple(sorted({item.obligation_id for item in violations})),
            classifications=tuple(classifications),
            **instrumentation,
        )

    by_id = {item.finding_id: item for item in classifications}
    primary_classification = by_id[primary.finding_id]
    edge = primary.first_invalid_edge
    edge_not_applicable = None
    if edge is None and primary_classification.kind in {
        FindingKind.DIRECT_IDENTITY_VIOLATION,
        FindingKind.DERIVED_CONTRACT_FAILURE,
    }:
        edge_not_applicable = EDGE_NOT_APPLICABLE
    return PrimaryResolution(
        primary_finding_id=primary.finding_id,
        primary_kind=primary_classification.kind,
        primary_layer=primary.primary_layer,
        primary_obligation=primary.obligation_id,
        first_invalid_edge=edge,
        edge_not_applicable_reason=edge_not_applicable,
        affected_finding_ids=tuple(item.finding_id for item in violations),
        all_invalid_edges=tuple(
            item.first_invalid_edge
            for item in violations
            if item.first_invalid_edge is not None
        ),
        affected_layers=tuple(sorted({item.primary_layer for item in violations})),
        affected_obligations=tuple(sorted({item.obligation_id for item in violations})),
        classifications=tuple(classifications),
        **instrumentation,
    )


def resolve_minimal_causal_roots(
    findings: Sequence[ScientificAuditFinding],
    graph: PipelineGraph,
    violated_nodes: set[str] | None = None,
    *,
    default_primary_finding_id: str | None = None,
) -> PrimaryResolution:
    """Resolve a minimal root set and canonical primary from causal structure.

    Every violating finding is retained in the minimal set unless another
    violating finding is a known upstream cause.  The canonical primary is a
    deterministic, outcome-blind tie-break over that complete root set.
    ``default_primary_finding_id`` is the separately computed legacy fallback
    primary and is used only for diagnostics, never to select the FULL result.
    """
    all_findings = tuple(findings)
    violations = tuple(
        sorted(
            (item for item in all_findings if item.status == "VIOLATION"),
            key=lambda item: item.finding_id,
        )
    )
    causal = build_violation_causal_graph(all_findings, graph, violated_nodes)
    classifications = {item.finding_id: item for item in causal.classifications}
    root_ids = set(causal.root_finding_ids())
    root_findings = tuple(
        sorted(
            (item for item in causal.findings if item.finding_id in root_ids),
            key=lambda item: _canonical_root_sort_key(item, classifications, graph),
        )
    )
    primary = root_findings[0] if root_findings else None

    default_primary = default_primary_finding_id
    if default_primary not in classifications:
        default_primary = _legacy_topological_primary_finding_id(violations, graph)

    primary_rank = (
        _canonical_graph_position(graph, classifications[primary.finding_id].node_id)
        if primary is not None
        else None
    )
    resolver_tie_count = (
        sum(
            _canonical_graph_position(graph, classifications[item.finding_id].node_id)
            == primary_rank
            for item in root_findings
        )
        - 1
        if primary_rank is not None
        else 0
    )
    resolver_opportunity = (
        len(violations) > 1
        and bool(causal.edges)
        and default_primary is not None
        and default_primary not in root_ids
    )
    resolved_primary = primary.finding_id if primary is not None else None
    instrumentation = {
        "finding_count": len(all_findings),
        "violating_finding_count": len(violations),
        "direct_native_candidate_count": sum(
            item.kind == FindingKind.DIRECT_NATIVE_VIOLATION
            for item in causal.classifications
        ),
        "direct_identity_candidate_count": sum(
            item.kind == FindingKind.DIRECT_IDENTITY_VIOLATION
            for item in causal.classifications
        ),
        "derived_candidate_count": sum(
            item.kind == FindingKind.DERIVED_CONTRACT_FAILURE
            for item in causal.classifications
        ),
        "propagated_candidate_count": sum(
            item.kind == FindingKind.PROPAGATED_CONSEQUENCE
            for item in causal.classifications
        ),
        "resolver_opportunity": resolver_opportunity,
        "default_primary_finding": default_primary,
        "resolved_primary_finding": resolved_primary,
        "resolver_changed_choice": default_primary != resolved_primary,
        "resolver_tie_count": resolver_tie_count,
        "causal_edge_count": len(causal.edges),
        "causal_edges": causal.edges,
        "causal_root_candidate_count": len(root_findings),
        "minimal_root_set": tuple(item.finding_id for item in root_findings),
        "minimal_root_set_size": len(root_findings),
    }
    return _resolution_from_causal_primary(
        violations,
        causal.classifications,
        primary,
        **instrumentation,
    )


def resolve_primary(
    findings: Sequence[ScientificAuditFinding],
    graph: PipelineGraph,
) -> PrimaryResolution:
    violations = [f for f in findings if f.status == "VIOLATION"]
    if not violations:
        return PrimaryResolution(
            primary_finding_id=None,
            primary_kind=None,
            primary_layer=None,
            primary_obligation=None,
            first_invalid_edge=None,
            edge_not_applicable_reason=None,
            affected_finding_ids=(),
            all_invalid_edges=(),
            affected_layers=(),
            affected_obligations=(),
        )
    violated_nodes = _violated_node_ids(findings)
    classified = [
        classify_finding(f, graph, violated_nodes) for f in violations
    ]
    # selection order
    priority = {
        FindingKind.DIRECT_IDENTITY_VIOLATION: 0,
        FindingKind.DIRECT_NATIVE_VIOLATION: 1,
        FindingKind.DERIVED_CONTRACT_FAILURE: 2,
        FindingKind.PROPAGATED_CONSEQUENCE: 3,
    }

    def sort_key(item: tuple[FindingClassification, ScientificAuditFinding]):
        cls, finding = item
        return (
            priority[cls.kind],
            _graph_position(graph, cls.node_id),
            -cls.evidence_count,
            finding.obligation_id,
            finding.finding_id,
        )

    ranked = sorted(zip(classified, violations, strict=False), key=sort_key)
    primary_cls, primary = ranked[0]
    edge = primary.first_invalid_edge
    edge_not_applicable = None
    if edge is None and primary_cls.kind in (
        FindingKind.DIRECT_IDENTITY_VIOLATION,
        FindingKind.DERIVED_CONTRACT_FAILURE,
    ):
        edge_not_applicable = EDGE_NOT_APPLICABLE
    elif edge is None:
        # non-identity direct violation without edge: leave as None
        pass

    all_edges = tuple(
        f.first_invalid_edge
        for f in violations
        if f.first_invalid_edge is not None
    )
    layers = tuple(sorted({f.primary_layer for f in violations}))
    obligations = tuple(sorted({f.obligation_id for f in violations}))
    return PrimaryResolution(
        primary_finding_id=primary.finding_id,
        primary_kind=primary_cls.kind,
        primary_layer=primary.primary_layer,
        primary_obligation=primary.obligation_id,
        first_invalid_edge=edge,
        edge_not_applicable_reason=edge_not_applicable,
        affected_finding_ids=tuple(f.finding_id for f in violations),
        all_invalid_edges=all_edges,
        affected_layers=layers,
        affected_obligations=obligations,
        classifications=tuple(classified),
    )


__all__ = [
    "EDGE_NOT_APPLICABLE",
    "FindingKind",
    "FindingClassification",
    "PrimaryResolution",
    "ViolationCausalGraph",
    "classify_finding",
    "build_violation_causal_graph",
    "resolve_minimal_causal_roots",
    "resolve_primary",
]
