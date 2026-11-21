"""Frozen capability scope for the supplementary CV adapter."""

from dmcasrc.adapters._normalized import read_scope


def scope(context):
    return read_scope(context, "ultralytics_yolo")


__all__ = ["scope"]
