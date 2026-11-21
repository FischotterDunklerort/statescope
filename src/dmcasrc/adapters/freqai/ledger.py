"""FreqAI candidate-ledger construction."""

from dmcasrc.adapters._normalized import read_ledger


def build_ledger(context, graph):
    return read_ledger(context, graph)


__all__ = ["build_ledger"]
