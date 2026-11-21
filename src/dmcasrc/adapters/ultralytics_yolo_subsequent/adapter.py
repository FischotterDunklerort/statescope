"""Ultralytics YOLO evidence orchestration.

This adapter only parses normalized evidence emitted by the external wrapper.
It deliberately contains no fault verdicts or scientific decision rules.
"""

from dmcasrc.adapters._normalized import read_raw_references, read_summary
from dmcasrc.adapters.models import AdapterEvidenceBundle

from .artifacts import register_artifacts
from .graph import discover_graph
from .ledger import build_ledger
from .native_facts import native_fact_rules
from .scope import scope


class UltralyticsYoloV2Adapter:
    """Evidence-only adapter for the independently frozen subsequent cohort."""

    project_id = "ultralytics_yolo_subsequent"

    def version(self) -> str:
        return "2.0.0"

    scope = staticmethod(scope)
    discover_graph = staticmethod(discover_graph)
    build_ledger = staticmethod(build_ledger)
    register_artifacts = staticmethod(register_artifacts)
    native_fact_rules = staticmethod(native_fact_rules)

    def collect(self, context):
        graph = self.discover_graph(context)
        ledger = self.build_ledger(context, graph)
        return AdapterEvidenceBundle(
            graph=graph,
            ledger=ledger,
            artifacts=self.register_artifacts(context, graph, ledger),
            scope=self.scope(context),
            raw_evidence_refs=read_raw_references(context),
            run_summary=read_summary(context, self.project_id),
        )


__all__ = ["UltralyticsYoloV2Adapter"]
