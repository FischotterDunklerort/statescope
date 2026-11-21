"""Freqtrade artifact registry."""

from dmcasrc.adapters._normalized import read_artifacts


def register_artifacts(context, graph, ledger):
    del graph, ledger
    return read_artifacts(context)
