"""Qlib graph discovery."""

from dmcasrc.adapters._normalized import read_graph
from dmcasrc.adapters.models import AdapterRunContext
from dmcasrc.domain.graph import PipelineGraph


def discover_graph(context: AdapterRunContext) -> PipelineGraph:
    return read_graph(context)


__all__ = ["discover_graph"]
