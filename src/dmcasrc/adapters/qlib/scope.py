"""Qlib adapter scope."""

from dmcasrc.adapters._normalized import read_scope
from dmcasrc.adapters.models import AdapterRunContext, AdapterScope


def scope(context: AdapterRunContext) -> AdapterScope:
    return read_scope(context, "qlib")


__all__ = ["scope"]
