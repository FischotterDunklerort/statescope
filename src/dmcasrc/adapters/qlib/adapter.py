"""Qlib evidence orchestration without audit decisions."""

from dmcasrc.adapters._normalized import read_raw_references, read_summary
from dmcasrc.adapters.models import (
    AdapterEvidenceBundle,
    AdapterRunContext,
    AdapterScope,
)
from dmcasrc.domain.graph import PipelineGraph
from dmcasrc.domain.ledger import CandidateLedger
from dmcasrc.infrastructure.deployment_manifest_schema import ArtifactManifest

from .artifacts import register_artifacts
from .graph import discover_graph
from .ledger import build_ledger
from .native_facts import native_fact_rules
from .scope import scope


class QlibAdapter:
    project_id = "qlib"

    def version(self) -> str:
        return "1.0.0"

    native_fact_rules = staticmethod(native_fact_rules)

    def scope(self, context: AdapterRunContext) -> AdapterScope:
        return scope(context)

    def discover_graph(self, context: AdapterRunContext) -> PipelineGraph:
        return discover_graph(context)

    def build_ledger(
        self,
        context: AdapterRunContext,
        graph: PipelineGraph,
    ) -> CandidateLedger:
        return build_ledger(context, graph)

    def register_artifacts(
        self,
        context: AdapterRunContext,
        graph: PipelineGraph,
        ledger: CandidateLedger,
    ) -> ArtifactManifest:
        return register_artifacts(context, graph, ledger)

    def collect(self, context: AdapterRunContext) -> AdapterEvidenceBundle:
        graph = self.discover_graph(context)
        ledger = self.build_ledger(context, graph)
        artifacts = self.register_artifacts(context, graph, ledger)
        return AdapterEvidenceBundle(
            graph=graph,
            ledger=ledger,
            artifacts=artifacts,
            scope=self.scope(context),
            raw_evidence_refs=read_raw_references(context),
            run_summary=read_summary(context, self.project_id),
        )


__all__ = ["QlibAdapter"]
