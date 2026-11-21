"""Ultralytics artifact registry loading."""

from dmcasrc.adapters._normalized import read_artifacts


def register_artifacts(context, graph, ledger):
    del graph, ledger
    return read_artifacts(context)


__all__ = ["register_artifacts"]
