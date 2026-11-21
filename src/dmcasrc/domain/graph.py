"""Deterministic pipeline graph with cycle and ancestor checks."""

from __future__ import annotations

import copy
import heapq
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

PIPELINE_GRAPH_SCHEMA_VERSION = "dmcasrc.pipeline-graph.initial"


class GraphValidationError(ValueError):
    pass


class GraphCycleError(GraphValidationError):
    pass


@dataclass(frozen=True)
class CanonicalEdge:
    """Scenario-neutral identity of a dependency edge.

    G5.2 localization deliberately excludes implementation-specific ``edge_id``
    values.  The source role, target role, and relation are the complete
    machine-readable comparison object.
    """

    source_id: str
    target_id: str
    relation: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_id", _text("source_id", self.source_id))
        object.__setattr__(self, "target_id", _text("target_id", self.target_id))
        object.__setattr__(self, "relation", _text("relation", self.relation))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> CanonicalEdge:
        if not isinstance(payload, Mapping):
            raise GraphValidationError("canonical edge must be an object.")
        _keys("canonical edge", payload, {"source_id", "target_id", "relation"})
        return cls(
            source_id=payload["source_id"],
            target_id=payload["target_id"],
            relation=payload["relation"],
        )

    def to_dict(self) -> dict[str, str]:
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "relation": self.relation,
        }


def _text(name: str, value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise GraphValidationError(f"{name} must be a non-empty string.")
    return value.strip()


def _attributes(name: str, value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise GraphValidationError(f"{name} must be an object.")
    if len(value) > 0:
        raise GraphValidationError(f"{name} must be empty (initial does not support attributes).")
    return {str(key): copy.deepcopy(item) for key, item in value.items()}


def _keys(name: str, value: Mapping[str, Any], expected: set[str]) -> None:
    actual = {str(key) for key in value}
    if actual != expected:
        raise GraphValidationError(
            f"{name} fields differ: missing={sorted(expected - actual)}, "
            f"unknown={sorted(actual - expected)}."
        )


@dataclass(frozen=True)
class GraphNode:
    node_id: str
    layer: str
    kind: str
    attributes: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "node_id", _text("node_id", self.node_id))
        object.__setattr__(self, "layer", _text("layer", self.layer))
        object.__setattr__(self, "kind", _text("kind", self.kind))
        object.__setattr__(self, "attributes", _attributes("node.attributes", self.attributes))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> GraphNode:
        if not isinstance(payload, Mapping):
            raise GraphValidationError("node must be an object.")
        _keys("node", payload, {"node_id", "layer", "kind", "attributes"})
        return cls(
            node_id=_text("node_id", payload["node_id"]),
            layer=_text("layer", payload["layer"]),
            kind=_text("kind", payload["kind"]),
            attributes=_attributes("node.attributes", payload["attributes"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "layer": self.layer,
            "kind": self.kind,
            "attributes": copy.deepcopy(self.attributes),
        }


@dataclass(frozen=True)
class GraphEdge:
    edge_id: str
    source: str
    target: str
    relation: str
    attributes: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "edge_id", _text("edge_id", self.edge_id))
        object.__setattr__(self, "source", _text("edge.source", self.source))
        object.__setattr__(self, "target", _text("edge.target", self.target))
        object.__setattr__(self, "relation", _text("edge.relation", self.relation))
        object.__setattr__(self, "attributes", _attributes("edge.attributes", self.attributes))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> GraphEdge:
        if not isinstance(payload, Mapping):
            raise GraphValidationError("edge must be an object.")
        _keys("edge", payload, {"edge_id", "source", "target", "relation", "attributes"})
        return cls(
            edge_id=_text("edge_id", payload["edge_id"]),
            source=_text("edge.source", payload["source"]),
            target=_text("edge.target", payload["target"]),
            relation=_text("edge.relation", payload["relation"]),
            attributes=_attributes("edge.attributes", payload["attributes"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "edge_id": self.edge_id,
            "source": self.source,
            "target": self.target,
            "relation": self.relation,
            "attributes": copy.deepcopy(self.attributes),
        }


@dataclass(frozen=True)
class PipelineGraph:
    nodes: tuple[GraphNode, ...]
    edges: tuple[GraphEdge, ...]
    schema_version: str = PIPELINE_GRAPH_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != PIPELINE_GRAPH_SCHEMA_VERSION:
            raise GraphValidationError(
                f"unsupported graph schema: {self.schema_version}."
            )
        if not isinstance(self.nodes, tuple):
            raise GraphValidationError("nodes must be a tuple of GraphNode instances.")
        for item in self.nodes:
            if not isinstance(item, GraphNode):
                raise GraphValidationError("nodes must contain GraphNode instances.")
        if not isinstance(self.edges, tuple):
            raise GraphValidationError("edges must be a tuple of GraphEdge instances.")
        for item in self.edges:
            if not isinstance(item, GraphEdge):
                raise GraphValidationError("edges must contain GraphEdge instances.")
        node_ids = [node.node_id for node in self.nodes]
        edge_ids = [edge.edge_id for edge in self.edges]
        if len(node_ids) != len(set(node_ids)):
            raise GraphValidationError("duplicate node_id values are forbidden.")
        if len(edge_ids) != len(set(edge_ids)):
            raise GraphValidationError("duplicate edge_id values are forbidden.")
        known = set(node_ids)
        dangling = [
            edge.edge_id
            for edge in self.edges
            if edge.source not in known or edge.target not in known
        ]
        if dangling:
            raise GraphValidationError(f"edges reference unknown nodes: {sorted(dangling)}.")
        self.topological_order()

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> PipelineGraph:
        if not isinstance(payload, Mapping):
            raise GraphValidationError("pipeline graph root must be an object.")
        _keys("pipeline graph", payload, {"schema_version", "nodes", "edges"})
        if payload["schema_version"] != PIPELINE_GRAPH_SCHEMA_VERSION:
            raise GraphValidationError(f"unsupported graph schema: {payload['schema_version']}.")
        if not isinstance(payload["nodes"], list) or not isinstance(payload["edges"], list):
            raise GraphValidationError("nodes and edges must be lists.")
        return cls(
            nodes=tuple(GraphNode.from_dict(item) for item in payload["nodes"]),
            edges=tuple(GraphEdge.from_dict(item) for item in payload["edges"]),
        )

    def topological_order(self) -> tuple[str, ...]:
        indegree = {node.node_id: 0 for node in self.nodes}
        children = {node.node_id: [] for node in self.nodes}
        for edge in self.edges:
            indegree[edge.target] += 1
            children[edge.source].append(edge.target)
        queue = [node_id for node_id, degree in indegree.items() if degree == 0]
        heapq.heapify(queue)
        order: list[str] = []
        while queue:
            node_id = heapq.heappop(queue)
            order.append(node_id)
            for child in sorted(children[node_id]):
                indegree[child] -= 1
                if indegree[child] == 0:
                    heapq.heappush(queue, child)
        if len(order) != len(self.nodes):
            raise GraphCycleError("pipeline graph contains a directed cycle.")
        return tuple(order)

    def ancestors(self, node_id: str) -> tuple[str, ...]:
        if node_id not in {node.node_id for node in self.nodes}:
            raise KeyError(node_id)
        parents: dict[str, set[str]] = {node.node_id: set() for node in self.nodes}
        for edge in self.edges:
            parents[edge.target].add(edge.source)
        found: set[str] = set()
        stack = list(parents[node_id])
        while stack:
            parent = stack.pop()
            if parent not in found:
                found.add(parent)
                stack.extend(parents[parent])
        return tuple(item for item in self.topological_order() if item in found)

    def relevant_node_ids(self, node_id: str) -> tuple[str, ...]:
        """Return all ancestors plus *node_id* in topological order.

        Raises ``KeyError`` when *node_id* is unknown.
        """
        if node_id not in {node.node_id for node in self.nodes}:
            raise KeyError(node_id)
        return self.ancestors(node_id) + (node_id,)

    def first_invalid_edge(
        self,
        node_id: str,
        invalid_node_ids: set[str],
    ) -> str | None:
        """Return the deterministically-first edge that connects an invalid
        node to a valid node on the path toward *node_id*.

        * Only ancestors of *node_id* plus *node_id* itself are considered.
        * ``invalid_node_ids`` outside that sub-graph are silently ignored.
        * Ties are broken by ``(topological_index[edge.target],
          topological_index[edge.source], edge.edge_id)``.
        * Returns ``None`` when no eligible edge exists.
        * Raises ``KeyError`` when *node_id* is unknown.
        """
        if node_id not in {node.node_id for node in self.nodes}:
            raise KeyError(node_id)
        topo = self.topological_order()
        topo_index = {nid: idx for idx, nid in enumerate(topo)}
        relevant = set(self.relevant_node_ids(node_id))
        candidates: list[tuple[int, int, str, str]] = []
        for edge in self.edges:
            if edge.source not in relevant or edge.target not in relevant:
                continue
            if edge.source not in invalid_node_ids and edge.target not in invalid_node_ids:
                continue
            candidates.append(
                (topo_index[edge.target], topo_index[edge.source], edge.edge_id, edge.edge_id)
            )
        if not candidates:
            return None
        candidates.sort(key=lambda item: (item[0], item[1], item[2]))
        return candidates[0][3]

    def first_invalid_edge_object(
        self,
        node_id: str,
        invalid_node_ids: set[str],
    ) -> CanonicalEdge | None:
        """Return the topologically-first invalid edge as a canonical object."""
        if node_id not in {node.node_id for node in self.nodes}:
            raise KeyError(node_id)
        topo_index = {
            candidate: index for index, candidate in enumerate(self.topological_order())
        }
        relevant = set(self.relevant_node_ids(node_id))
        candidates = [
            edge
            for edge in self.edges
            if edge.source in relevant
            and edge.target in relevant
            and (
                edge.source in invalid_node_ids
                or edge.target in invalid_node_ids
            )
        ]
        if not candidates:
            return None
        selected = min(
            candidates,
            key=lambda edge: (
                topo_index[edge.target],
                topo_index[edge.source],
                edge.relation,
                edge.source,
                edge.target,
                edge.edge_id,
            ),
        )
        return CanonicalEdge(selected.source, selected.target, selected.relation)

    def first_edge_into(self, node_id: str, source_ids: set[str]) -> str | None:
        matches = sorted(
            edge.edge_id
            for edge in self.edges
            if edge.target == node_id and edge.source in source_ids
        )
        return matches[0] if matches else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "nodes": [node.to_dict() for node in sorted(self.nodes, key=lambda item: item.node_id)],
            "edges": [edge.to_dict() for edge in sorted(self.edges, key=lambda item: item.edge_id)],
        }
