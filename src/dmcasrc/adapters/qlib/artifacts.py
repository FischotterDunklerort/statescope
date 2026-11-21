"""Qlib artifact registration."""

from dmcasrc.adapters._normalized import read_artifacts
from dmcasrc.adapters.models import AdapterRunContext
from dmcasrc.domain.graph import PipelineGraph
from dmcasrc.domain.ledger import CandidateLedger
from dmcasrc.infrastructure.deployment_manifest_schema import ArtifactManifest


def register_artifacts(
    context: AdapterRunContext,
    graph: PipelineGraph,
    ledger: CandidateLedger,
) -> ArtifactManifest:
    del graph, ledger
    return read_artifacts(context)


__all__ = ["register_artifacts"]
