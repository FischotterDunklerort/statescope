"""Shared domain exceptions for DM-CASRC."""

from __future__ import annotations


class StrictContractError(ValueError):
    """Raised when required contract evidence cannot be proven."""


__all__ = ["StrictContractError"]
