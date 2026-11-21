"""Qlib candidate-ledger construction."""

from dmcasrc.adapters._normalized import read_ledger
from dmcasrc.adapters.models import AdapterRunContext
from dmcasrc.domain.graph import PipelineGraph
from dmcasrc.domain.ledger import CandidateLedger


def build_ledger(
    context: AdapterRunContext,
    graph: PipelineGraph,
) -> CandidateLedger:
    return read_ledger(context, graph)


__all__ = ["build_ledger"]
