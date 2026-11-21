"""Ultralytics graph discovery from normalized evidence."""

from dmcasrc.adapters._normalized import read_graph


def discover_graph(context):
    return read_graph(context)


__all__ = ["discover_graph"]
