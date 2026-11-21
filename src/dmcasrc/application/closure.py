"""Ancestor closure and outcome-blind witness-propagation closure.

The witness input on a :class:`CanonicalObservation` is runtime/native
evidence.  It is therefore the direct-witness input; reachability is never
used to manufacture direct support.  Propagated witnesses are created only
from candidate dependency edges.  In the native ``PipelineGraph`` adapter,
those edges are graph edges whose target node has ``kind == "candidate"``;
the node id is the candidate id.  This keeps ordinary pipeline-stage nodes
from being mistaken for candidate witnesses.

``WitnessDependencyEdge`` is a small internal adapter for candidate or
provenance relations supplied by another runtime.  It intentionally accepts
cycles: traversal below is bounded by deterministic best-path state.
"""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from ..domain.contracts import CanonicalObservation
from ..domain.graph import PipelineGraph


def _digest(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _canonical_ids(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(sorted({value for value in values if isinstance(value, str) and value}))


@dataclass(frozen=True)
class WitnessDependencyEdge:
    """A runtime-visible candidate/provenance edge used for propagation."""

    source_candidate_id: str
    target_candidate_id: str
    relation: str

    def __post_init__(self) -> None:
        for name in ("source_candidate_id", "target_candidate_id", "relation"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty string")

    def to_dict(self) -> dict[str, str]:
        return {
            "source_candidate_id": self.source_candidate_id,
            "target_candidate_id": self.target_candidate_id,
            "relation": self.relation,
        }


@dataclass(frozen=True)
class WitnessPropagationTrace:
    """The canonical structural reason one witness was propagated."""

    source_witness_id: str
    source_candidate_id: str
    target_candidate_id: str
    propagation_depth: int
    propagation_path: tuple[str, ...]
    propagation_path_digest: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_witness_id": self.source_witness_id,
            "source_candidate_id": self.source_candidate_id,
            "target_candidate_id": self.target_candidate_id,
            "propagation_depth": self.propagation_depth,
            "propagation_path": list(self.propagation_path),
            "propagation_path_digest": self.propagation_path_digest,
        }


@dataclass(frozen=True)
class WitnessClosure:
    """Explicit direct/propagated/derived witness closure for one observation."""

    observation_id: str
    candidate_id: str
    direct_witness_ids: tuple[str, ...]
    propagated_witness_ids: tuple[str, ...]
    derived_witness_ids: tuple[str, ...]
    complete_witness_ids: tuple[str, ...]
    propagation_opportunity: bool
    propagation_invoked: bool
    propagation_changed_complete_set: bool
    max_propagation_depth: int
    propagation_edge_traversals: int
    propagation_fanout: int
    propagation_path_digest: str
    propagation_paths: tuple[WitnessPropagationTrace, ...]

    def __post_init__(self) -> None:
        direct = _canonical_ids(self.direct_witness_ids)
        propagated = tuple(
            value for value in _canonical_ids(self.propagated_witness_ids) if value not in direct
        )
        derived = tuple(
            value
            for value in _canonical_ids(self.derived_witness_ids)
            if value not in direct and value not in propagated
        )
        complete = _canonical_ids((*direct, *propagated, *derived))
        if self.complete_witness_ids != complete:
            raise ValueError("complete witness ids must equal direct union propagated union derived")
        if self.propagation_changed_complete_set != bool(propagated):
            raise ValueError("propagation_changed_complete_set must match propagated witnesses")
        if self.propagation_invoked and not self.propagation_opportunity:
            raise ValueError("propagation cannot be invoked without a structural opportunity")
        object.__setattr__(self, "direct_witness_ids", direct)
        object.__setattr__(self, "propagated_witness_ids", propagated)
        object.__setattr__(self, "derived_witness_ids", derived)
        object.__setattr__(self, "complete_witness_ids", complete)

    # G6.1 names remain readable for code that only needs the tier sets.
    @property
    def direct(self) -> tuple[str, ...]:
        return self.direct_witness_ids

    @property
    def propagated(self) -> tuple[str, ...]:
        return self.propagated_witness_ids

    @property
    def derived(self) -> tuple[str, ...]:
        return self.derived_witness_ids

    @property
    def complete(self) -> tuple[str, ...]:
        return self.complete_witness_ids

    def to_dict(self) -> dict[str, Any]:
        return {
            "observation_id": self.observation_id,
            "candidate_id": self.candidate_id,
            "direct_witness_count": len(self.direct_witness_ids),
            "propagated_witness_count": len(self.propagated_witness_ids),
            "derived_witness_count": len(self.derived_witness_ids),
            "complete_witness_count": len(self.complete_witness_ids),
            "direct_witness_ids": list(self.direct_witness_ids),
            "propagated_witness_ids": list(self.propagated_witness_ids),
            "derived_witness_ids": list(self.derived_witness_ids),
            "complete_witness_ids": list(self.complete_witness_ids),
            "propagation_opportunity": self.propagation_opportunity,
            "propagation_invoked": self.propagation_invoked,
            "propagation_changed_complete_set": self.propagation_changed_complete_set,
            "max_propagation_depth": self.max_propagation_depth,
            "propagation_edge_traversals": self.propagation_edge_traversals,
            "propagation_fanout": self.propagation_fanout,
            "propagation_path_digest": self.propagation_path_digest,
            "propagation_paths": [item.to_dict() for item in self.propagation_paths],
        }


# Backwards-compatible import name.  The object now additionally exposes the
# derived tier and propagation instrumentation required by G6.4-T3.
WitnessTiers = WitnessClosure


def ancestor_closure_expand(
    observation: CanonicalObservation,
    graph: PipelineGraph,
) -> CanonicalObservation:
    """Expand affected layers / invalid nodes from ancestor violations."""
    node_id = observation.node_id
    try:
        ancestors = graph.ancestors(node_id)
    except KeyError:
        ancestors = ()
    layers = set(observation.affected_layers)
    invalid = set(observation.invalid_node_ids)
    for ancestor in ancestors:
        for node in graph.nodes:
            if node.node_id == ancestor:
                layers.add(node.layer)
    invalid.update(ancestors)
    return CanonicalObservation(
        observation_id=observation.observation_id,
        candidate_id=observation.candidate_id,
        node_id=observation.node_id,
        obligation_id=observation.obligation_id,
        status=observation.status,
        primary_layer=observation.primary_layer,
        affected_layers=tuple(sorted(layers)),
        invalid_node_ids=tuple(sorted(invalid)),
        candidate_witness_ids=observation.candidate_witness_ids,
        evidence_refs=observation.evidence_refs,
        reason_codes=observation.reason_codes,
        missing_required_evidence=observation.missing_required_evidence,
        declared_first_invalid_edge=observation.declared_first_invalid_edge,
    )


def _native_dependency_structure(
    observation: CanonicalObservation,
    graph: PipelineGraph,
) -> tuple[tuple[WitnessDependencyEdge, ...], frozenset[str], dict[str, str]]:
    """Adapt candidate graph nodes from the native runtime graph.

    ``GraphNode.kind == 'candidate'`` is the explicit runtime marker.  This
    leaves stage nodes such as ``fit`` and ``model`` out of the candidate
    witness universe even when they are connected by pipeline edges.
    """
    candidate_nodes = {
        node.node_id for node in graph.nodes if node.kind.casefold() == "candidate"
    }
    source_nodes: dict[str, str] = {}
    # Native evidence binds the observation's own candidate to the observed
    # runtime node.  Other declared direct witnesses can be sources only when
    # the graph independently exposes a candidate node with that exact id.
    if observation.candidate_id in observation.candidate_witness_ids:
        source_nodes[observation.candidate_id] = observation.node_id
    for witness_id in observation.candidate_witness_ids:
        if witness_id in candidate_nodes:
            source_nodes.setdefault(witness_id, witness_id)
    edges = tuple(
        WitnessDependencyEdge(edge.source, edge.target, edge.relation)
        for edge in graph.edges
    )
    return edges, frozenset(candidate_nodes), source_nodes


def _canonical_trace(
    source: str,
    target: str,
    path: tuple[str, ...],
) -> WitnessPropagationTrace:
    return WitnessPropagationTrace(
        source_witness_id=source,
        source_candidate_id=source,
        target_candidate_id=target,
        propagation_depth=len(path) - 1,
        propagation_path=path,
        propagation_path_digest=_digest(list(path)),
    )


def _trace_key(trace: WitnessPropagationTrace) -> tuple[Any, ...]:
    return (
        trace.propagation_depth,
        trace.propagation_path,
        trace.source_witness_id,
        trace.target_candidate_id,
    )


def _eligible_propagation_exists(
    direct: tuple[str, ...],
    derived: tuple[str, ...],
    edges: tuple[WitnessDependencyEdge, ...],
    candidate_nodes: frozenset[str],
    source_nodes: dict[str, str],
) -> bool:
    """Check structural opportunity without recording a propagation run."""
    adjacency: dict[str, list[WitnessDependencyEdge]] = defaultdict(list)
    for edge in edges:
        adjacency[edge.source_candidate_id].append(edge)
    for values in adjacency.values():
        values.sort(key=lambda edge: (edge.target_candidate_id, edge.relation))
    excluded = set(direct) | set(derived)
    for source in direct:
        start = source_nodes.get(source)
        if start is None:
            continue
        visited = {start}
        queue = [start]
        while queue:
            node = queue.pop(0)
            for edge in adjacency[node]:
                target = edge.target_candidate_id
                if target in candidate_nodes and target not in excluded:
                    return True
                if target not in visited:
                    visited.add(target)
                    queue.append(target)
    return False


def witness_closure(
    observation: CanonicalObservation,
    graph: PipelineGraph | None = None,
    *,
    propagation_enabled: bool = True,
    derived_witness_ids: Sequence[str] = (),
    dependency_edges: Sequence[WitnessDependencyEdge] | None = None,
) -> WitnessClosure:
    """Build a deterministic, cycle-safe witness closure.

    ``dependency_edges`` is only an adapter for a runtime that already has a
    candidate/provenance relation.  When omitted, the existing ``PipelineGraph``
    candidate edges are used.  No scenario, outcome, truth, or historical data
    is accepted by this function.
    """
    direct = _canonical_ids(observation.candidate_witness_ids)
    derived = _canonical_ids(derived_witness_ids)
    if dependency_edges is None:
        if graph is None:
            raise ValueError("graph is required when dependency_edges are not supplied")
        edges, candidate_nodes, source_nodes = _native_dependency_structure(observation, graph)
    else:
        edges = tuple(
            sorted(
                dependency_edges,
                key=lambda edge: (
                    edge.source_candidate_id,
                    edge.target_candidate_id,
                    edge.relation,
                ),
            )
        )
        candidate_nodes = frozenset(
            (*direct, *(edge.source_candidate_id for edge in edges), *(edge.target_candidate_id for edge in edges))
        )
        source_nodes = {source: source for source in direct}

    opportunity = _eligible_propagation_exists(
        direct, derived, edges, candidate_nodes, source_nodes
    )
    if not propagation_enabled or not opportunity:
        complete = _canonical_ids((*direct, *derived))
        return WitnessClosure(
            observation_id=observation.observation_id,
            candidate_id=observation.candidate_id,
            direct_witness_ids=direct,
            propagated_witness_ids=(),
            derived_witness_ids=derived,
            complete_witness_ids=complete,
            propagation_opportunity=opportunity,
            propagation_invoked=False,
            propagation_changed_complete_set=False,
            max_propagation_depth=0,
            propagation_edge_traversals=0,
            propagation_fanout=0,
            propagation_path_digest=_digest([]),
            propagation_paths=(),
        )

    adjacency: dict[str, list[WitnessDependencyEdge]] = defaultdict(list)
    for edge in edges:
        adjacency[edge.source_candidate_id].append(edge)
    for values in adjacency.values():
        values.sort(key=lambda edge: (edge.target_candidate_id, edge.relation))

    excluded = set(direct) | set(derived)
    traces: dict[str, WitnessPropagationTrace] = {}
    traversals = 0
    fanout = 0
    for source in direct:
        start = source_nodes.get(source)
        if start is None:
            continue
        best_state: dict[str, tuple[int, tuple[str, ...]]] = {start: (0, (start,))}
        queue: list[tuple[str, ...]] = [(start,)]
        while queue:
            path = queue.pop(0)
            node = path[-1]
            state = (len(path) - 1, path)
            if best_state.get(node) != state:
                continue
            outgoing = adjacency[node]
            candidate_targets = {
                edge.target_candidate_id
                for edge in outgoing
                if edge.target_candidate_id in candidate_nodes
                and edge.target_candidate_id not in excluded
            }
            fanout = max(fanout, len(candidate_targets))
            for edge in outgoing:
                traversals += 1
                next_path = (*path, edge.target_candidate_id)
                target = edge.target_candidate_id
                next_state = (len(next_path) - 1, next_path)
                previous = best_state.get(target)
                if previous is None or next_state < previous:
                    best_state[target] = next_state
                    queue.append(next_path)
                    queue.sort(key=lambda value: (len(value), value))
                if target in candidate_nodes and target not in excluded:
                    trace = _canonical_trace(source, target, next_path)
                    previous_trace = traces.get(target)
                    if previous_trace is None or _trace_key(trace) < _trace_key(previous_trace):
                        traces[target] = trace

    propagation_paths = tuple(sorted(traces.values(), key=lambda item: item.target_candidate_id))
    propagated = tuple(item.target_candidate_id for item in propagation_paths)
    complete = _canonical_ids((*direct, *propagated, *derived))
    return WitnessClosure(
        observation_id=observation.observation_id,
        candidate_id=observation.candidate_id,
        direct_witness_ids=direct,
        propagated_witness_ids=propagated,
        derived_witness_ids=derived,
        complete_witness_ids=complete,
        propagation_opportunity=True,
        propagation_invoked=True,
        propagation_changed_complete_set=bool(propagated),
        max_propagation_depth=max((item.propagation_depth for item in propagation_paths), default=0),
        propagation_edge_traversals=traversals,
        propagation_fanout=fanout,
        propagation_path_digest=_digest([item.to_dict() for item in propagation_paths]),
        propagation_paths=propagation_paths,
    )


def witness_tiers(
    observation: CanonicalObservation,
    graph: PipelineGraph,
) -> WitnessClosure:
    """Compatibility view over the explicit witness closure."""
    return witness_closure(observation, graph)


def witness_closure_direct_only(observation: CanonicalObservation) -> CanonicalObservation:
    """A5: retain all independently native direct witnesses, never propagation."""
    return CanonicalObservation(
        observation_id=observation.observation_id,
        candidate_id=observation.candidate_id,
        node_id=observation.node_id,
        obligation_id=observation.obligation_id,
        status=observation.status,
        primary_layer=observation.primary_layer,
        affected_layers=observation.affected_layers,
        invalid_node_ids=observation.invalid_node_ids,
        candidate_witness_ids=_canonical_ids(observation.candidate_witness_ids),
        evidence_refs=observation.evidence_refs,
        reason_codes=observation.reason_codes,
        missing_required_evidence=observation.missing_required_evidence,
        declared_first_invalid_edge=observation.declared_first_invalid_edge,
    )


def witness_tiers_batch(
    observations: Sequence[CanonicalObservation],
    graph: PipelineGraph,
) -> dict[str, Any]:
    """Witness-tier report including explicit propagation instrumentation."""
    closures = [witness_closure(item, graph) for item in observations]
    return {
        "schema": "dmcasrc.witness-closure.subsequent",
        "checked": len(closures),
        "has_direct": sum(1 for item in closures if item.direct_witness_ids),
        "has_propagated": sum(1 for item in closures if item.propagated_witness_ids),
        "has_complete_gt_direct": sum(
            1 for item in closures if len(item.complete_witness_ids) > len(item.direct_witness_ids)
        ),
        "results": [item.to_dict() for item in closures],
    }


__all__ = [
    "WitnessClosure",
    "WitnessDependencyEdge",
    "WitnessPropagationTrace",
    "WitnessTiers",
    "ancestor_closure_expand",
    "witness_closure",
    "witness_closure_direct_only",
    "witness_tiers",
    "witness_tiers_batch",
]
