"""LEAN graph discovery."""

from dmcasrc.adapters._normalized import read_graph


def discover_graph(context):
    return read_graph(context)
