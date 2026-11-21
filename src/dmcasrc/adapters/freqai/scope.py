"""FreqAI adapter scope."""

from dmcasrc.adapters._normalized import read_scope


def scope(context):
    return read_scope(context, "freqai")


__all__ = ["scope"]
