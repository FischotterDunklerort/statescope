"""subsequent supplementary evidence adapter for an external Ultralytics YOLO run.

The package is intentionally not imported by the frozen global adapter registry.
The supplementary experiment registers it through its own registry surface so
that this non-financial validation cannot change the primary benchmark surface.
"""

from .adapter import UltralyticsYoloV2Adapter

__all__ = ["UltralyticsYoloV2Adapter"]
