"""FinRL graph discovery."""

from dmcasrc.adapters._normalized import read_graph
from dmcasrc.adapters.models import AdapterRunContext


def discover_graph(context: AdapterRunContext):
    return read_graph(context)
