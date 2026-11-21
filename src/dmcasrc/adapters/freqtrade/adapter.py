"""Freqtrade evidence orchestration."""

from dmcasrc.adapters._normalized import read_raw_references, read_summary
from dmcasrc.adapters.models import AdapterEvidenceBundle

from .artifacts import register_artifacts
from .graph import discover_graph
from .ledger import build_ledger
from .native_facts import native_fact_rules
from .scope import scope


class FreqtradeAdapter:
    project_id = "freqtrade"

    def version(self):
        return "1.0.0"

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
