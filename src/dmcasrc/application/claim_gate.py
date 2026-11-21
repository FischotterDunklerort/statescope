"""Two-phase claim gating: scope resolution, then fail-closed decision.

G1-C semantics:
- ``resolve_claim_scope`` produces a fully-determined view of which candidates,
  nodes, obligations and findings participate in a claim.
- ``evaluate_resolved_claim`` runs deterministic decision logic on that view.

This module never silently drops missing evidence; expected candidate ×
obligation pairs that have no finding are reported as ``missing_pairs`` and
cause ``CLAIM_EVIDENCE_INCOMPLETE``.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
from dataclasses import asdict

from dmcasrc.domain.contracts import (
    AuditFinding,
    CanonicalObservation,
    ClaimDecision,
    ClaimManifest,
    ClaimSpecification,
    ScientificClaim,
    ScientificClaimDecision,
)
from dmcasrc.infrastructure.deployment_manifest_schema import ArtifactManifest

_BLOCKED = {"VIOLATION", "UNVERIFIABLE", "ERROR"}


def finding_stable_id(finding: AuditFinding) -> str:
    """Stable identifier for a finding independent of process memory layout."""
    return (
        f"{finding.candidate_key}:"
        f"{finding.node_id}:"
        f"{finding.obligation_id}"
    )


# ── Phase 1: scope resolution ──


@dataclass(frozen=True)
class ResolvedClaimScope:
    """All scope information resolved against actual audit findings."""

    claim_id: str
    candidate_ids: tuple[str, ...]
    node_ids: tuple[str, ...]
    obligation_ids: tuple[str, ...]
    findings: tuple[AuditFinding, ...]
    missing_pairs: tuple[tuple[str, str, str], ...]
    blocking_finding_ids: tuple[str, ...]
    unknown_candidate_ids: tuple[str, ...] = ()
    unknown_node_ids: tuple[str, ...] = ()
    scope_error: str | None = None


def _resolve_scope_candidates(
    claim: ClaimSpecification,
    findings_candidates: set[str],
    known_candidate_ids: Iterable[str] | None,
) -> tuple[list[str], list[str]]:
    """Return (scope_candidates, unknown_candidates)."""
    if claim.scope.candidate_ids is not None:
        scope_candidates = sorted(claim.scope.candidate_ids)
        reference_pool = (
            set(known_candidate_ids) if known_candidate_ids is not None else findings_candidates
        )
        unknown = [c for c in scope_candidates if c not in reference_pool]
    elif known_candidate_ids is not None:
        scope_candidates = sorted(set(known_candidate_ids))
        unknown = []
    else:
        scope_candidates = sorted(findings_candidates)
        unknown = []
    return scope_candidates, unknown


def _resolve_scope_nodes(
    claim: ClaimSpecification,
    findings_nodes: set[str],
    known_node_ids: Iterable[str] | None,
) -> tuple[list[str], list[str]]:
    """Return (scope_nodes, unknown_nodes)."""
    if claim.scope.node_ids is not None:
        scope_nodes = sorted(claim.scope.node_ids)
        reference_pool = (
            set(known_node_ids) if known_node_ids is not None else findings_nodes
        )
        unknown = [n for n in scope_nodes if n not in reference_pool]
    elif known_node_ids is not None:
        scope_nodes = sorted(set(known_node_ids))
        unknown = []
    else:
        scope_nodes = sorted(findings_nodes)
        unknown = []
    return scope_nodes, unknown


def _resolve_obligation_pool(
    claim: ClaimSpecification,
    obligation_order: Mapping[str, int] | None,
) -> tuple[str, ...]:
    if obligation_order is None:
        return tuple(sorted(claim.required_obligation_ids))
    return tuple(
        sorted(claim.required_obligation_ids, key=lambda oid: obligation_order.get(oid, 0))
    )


def _detect_duplicates(
    relevant: list[AuditFinding],
) -> set[tuple[str, str, str]]:
    seen: set[tuple[str, str, str]] = set()
    duplicates: set[tuple[str, str, str]] = set()
    for finding in relevant:
        key = (finding.candidate_key, finding.node_id, finding.obligation_id)
        if key in seen:
            duplicates.add(key)
        seen.add(key)
    return duplicates


def _missing_pairs(
    scope_candidates: list[str],
    scope_nodes: list[str],
    obligation_pool: tuple[str, ...],
    relevant: list[AuditFinding],
) -> tuple[tuple[str, str, str], ...]:
    expected = {
        (c, n, o)
        for c in scope_candidates
        for n in scope_nodes
        for o in obligation_pool
    }
    actual = {
        (f.candidate_key, f.node_id, f.obligation_id) for f in relevant
    }
    return tuple(sorted(expected - actual))


def resolve_claim_scope(
    claim: ClaimSpecification,
    findings: Sequence[AuditFinding],
    *,
    known_candidate_ids: Iterable[str] | None = None,
    known_node_ids: Iterable[str] | None = None,
    obligation_order: Mapping[str, int] | None = None,
) -> ResolvedClaimScope:
    """Resolve a single claim against the audit findings.

    When scope fields are ``None``, the resolver prefers
    ``known_candidate_ids`` / ``known_node_ids`` (audit's authoritative
    pools) and only falls back to the findings pool if those are
    absent. Completeness is always evaluated as
    ``candidate × node × obligation``.
    """
    findings_candidates = {f.candidate_key for f in findings}
    findings_nodes = {f.node_id for f in findings}
    scope_candidates, unknown_candidates = _resolve_scope_candidates(
        claim, findings_candidates, known_candidate_ids
    )
    scope_nodes, unknown_nodes = _resolve_scope_nodes(
        claim, findings_nodes, known_node_ids
    )
    obligation_pool = _resolve_obligation_pool(claim, obligation_order)

    scope_error: str | None = None
    if unknown_candidates or unknown_nodes:
        scope_error = "CLAIM_SCOPE_UNKNOWN_REFERENCE"

    relevant = [
        finding for finding in findings
        if finding.candidate_key in scope_candidates
        and finding.node_id in scope_nodes
        and finding.obligation_id in obligation_pool
    ]

    if _detect_duplicates(relevant):
        scope_error = "CLAIM_DUPLICATE_FINDING"

    missing_pairs = _missing_pairs(scope_candidates, scope_nodes, obligation_pool, relevant)
    blocking = tuple(
        sorted(
            finding_stable_id(finding)
            for finding in relevant
            if finding.status in _BLOCKED
        )
    )

    return ResolvedClaimScope(
        claim_id=claim.claim_id,
        candidate_ids=tuple(scope_candidates),
        node_ids=tuple(scope_nodes),
        obligation_ids=obligation_pool,
        findings=tuple(relevant),
        missing_pairs=missing_pairs,
        blocking_finding_ids=blocking,
        unknown_candidate_ids=tuple(unknown_candidates),
        unknown_node_ids=tuple(unknown_nodes),
        scope_error=scope_error,
    )


# ── Phase 2: decision ──


def _decision_from_resolved(
    resolved: ResolvedClaimScope,
    status: str,
    reason_codes: tuple[str, ...],
) -> ClaimDecision:
    """Construct a ClaimDecision with all resolved metrics attached."""
    return ClaimDecision(
        claim_id=resolved.claim_id,
        status=status,
        reason_codes=reason_codes,
        required_obligation_ids=resolved.obligation_ids,
        resolved_candidate_count=len(resolved.candidate_ids),
        resolved_node_count=len(resolved.node_ids),
        evaluated_finding_count=len(resolved.findings),
        missing_pair_count=len(resolved.missing_pairs),
        blocking_finding_ids=resolved.blocking_finding_ids,
    )


def evaluate_resolved_claim(resolved: ResolvedClaimScope) -> ClaimDecision:
    """Apply the deterministic decision priority on a resolved scope."""
    reasons: list[str] = []

    if resolved.scope_error == "CLAIM_SCOPE_UNKNOWN_REFERENCE":
        reasons.append("CLAIM_SCOPE_UNKNOWN_REFERENCE")
        return _decision_from_resolved(resolved, "BLOCKED", ("CLAIM_SCOPE_UNKNOWN_REFERENCE",))

    if resolved.scope_error == "CLAIM_DUPLICATE_FINDING":
        reasons.append("CLAIM_DUPLICATE_FINDING")
        reasons.append("CLAIM_EVALUATION_ERROR")
        return _decision_from_resolved(resolved, "BLOCKED", ("CLAIM_DUPLICATE_FINDING", "CLAIM_EVALUATION_ERROR"))

    if not resolved.candidate_ids or not resolved.node_ids:
        reasons.append("CLAIM_SCOPE_EMPTY")
        return _decision_from_resolved(resolved, "BLOCKED", ("CLAIM_SCOPE_EMPTY",))

    if not resolved.findings:
        reasons.append("CLAIM_SCOPE_EMPTY")
        return _decision_from_resolved(resolved, "BLOCKED", ("CLAIM_SCOPE_EMPTY",))

    if resolved.missing_pairs:
        reasons.append("CLAIM_EVIDENCE_INCOMPLETE")

    error_findings = [f for f in resolved.findings if f.status == "ERROR"]
    if error_findings:
        reasons.append("CLAIM_EVALUATION_ERROR")

    violation_findings = [f for f in resolved.findings if f.status == "VIOLATION"]
    if violation_findings:
        reasons.append("CLAIM_OBLIGATION_BLOCKED")

    unverifiable_findings = [f for f in resolved.findings if f.status == "UNVERIFIABLE"]
    if unverifiable_findings:
        reasons.append("CLAIM_EVIDENCE_UNVERIFIABLE")

    statuses = {f.status for f in resolved.findings}
    if statuses and statuses == {"NOT_APPLICABLE"}:
        reasons.append("CLAIM_APPLICABLE_EVIDENCE_MISSING")

    capability_limited = any(f.status == "TOOL_UNSUPPORTED" for f in resolved.findings)

    if reasons:
        return _decision_from_resolved(resolved, "BLOCKED", tuple(reasons))

    if capability_limited:
        return _decision_from_resolved(resolved, "RESTRICTED", ("CLAIM_CAPABILITY_LIMITED",))

    return _decision_from_resolved(resolved, "ALLOWED", ("CLAIM_REQUIREMENTS_SATISFIED",))


# ── Public entry point ──


def evaluate_claims(
    claim_manifest: ClaimManifest,
    findings: Sequence[AuditFinding],
    registry: ArtifactManifest,
    *,
    obligation_order: Mapping[str, int] | None = None,
    known_candidate_ids: Iterable[str] | None = None,
    known_node_ids: Iterable[str] | None = None,
) -> tuple[tuple[ClaimDecision, ...], tuple[ResolvedClaimScope, ...]]:
    """Resolve each claim, then return (decisions, resolved scopes)."""
    artifact_ids = {artifact.artifact_id for artifact in registry.artifacts}
    decisions: list[ClaimDecision] = []
    scopes: list[ResolvedClaimScope] = []

    for claim in sorted(claim_manifest.claims, key=lambda c: c.claim_id):
        # Artifact-level claim binding still applies (artifact identity gate).
        missing_artifacts = sorted(set(claim.required_artifact_ids) - artifact_ids)
        if missing_artifacts:
            obligations = tuple(sorted(claim.required_obligation_ids))
            resolved = ResolvedClaimScope(
                claim_id=claim.claim_id,
                candidate_ids=(),
                node_ids=(),
                obligation_ids=obligations,
                findings=(),
                missing_pairs=(),
                blocking_finding_ids=(),
                unknown_candidate_ids=(),
                unknown_node_ids=(),
                scope_error="CLAIM_ARTIFACT_MISSING",
            )
            scopes.append(resolved)
            decisions.append(
                _decision_from_resolved(resolved, "BLOCKED", ("CLAIM_ARTIFACT_MISSING",))
            )
            continue

        resolved = resolve_claim_scope(
            claim,
            findings,
            known_candidate_ids=known_candidate_ids,
            known_node_ids=known_node_ids,
            obligation_order=obligation_order,
        )
        scopes.append(resolved)
        decisions.append(evaluate_resolved_claim(resolved))

    return tuple(decisions), tuple(scopes)


def evaluate_scientific_claims(
    claims: Sequence[ScientificClaim],
    observations: Sequence[CanonicalObservation],
) -> tuple[ScientificClaimDecision, ...]:
    """Gate G5.2 claims against only their declared candidate dependencies."""
    claim_ids = [claim.claim_id for claim in claims]
    if len(claim_ids) != len(set(claim_ids)):
        raise ValueError("duplicate scientific claim IDs are forbidden.")
    coordinates = [(item.candidate_id, item.node_id, item.obligation_id) for item in observations]
    if len(coordinates) != len(set(coordinates)):
        raise ValueError("duplicate canonical observation coordinates are forbidden.")
    observation_ids = [item.observation_id for item in observations]
    if len(observation_ids) != len(set(observation_ids)):
        raise ValueError("duplicate canonical observation IDs are forbidden.")
    decisions: list[ScientificClaimDecision] = []
    for claim in sorted(claims, key=lambda item: item.claim_id):
        relevant = [
            observation
            for observation in observations
            if observation.candidate_id in claim.candidate_ids
            and observation.obligation_id in claim.required_obligation_ids
            and (claim.node_ids is None or observation.node_id in claim.node_ids)
        ]
        if claim.node_ids is None:
            expected = {
                (candidate_id, obligation_id)
                for candidate_id in claim.candidate_ids
                for obligation_id in claim.required_obligation_ids
            }
            actual = {
                (item.candidate_id, item.obligation_id) for item in relevant
            }
        else:
            expected = {
                (candidate_id, node_id, obligation_id)
                for candidate_id in claim.candidate_ids
                for node_id in claim.node_ids
                for obligation_id in claim.required_obligation_ids
            }
            actual = {
                (item.candidate_id, item.node_id, item.obligation_id)
                for item in relevant
            }
        missing = tuple(
            "|".join(dependency) for dependency in sorted(expected - actual)
        )
        violating = tuple(
            sorted(
                item.observation_id
                for item in relevant
                if item.status == "VIOLATION"
            )
        )
        unverifiable = tuple(
            sorted(
                item.observation_id
                for item in relevant
                if item.status == "UNVERIFIABLE"
            )
        )
        if violating:
            decisions.append(
                ScientificClaimDecision(
                    claim.claim_id,
                    "BLOCKED",
                    ("CLAIM_OBLIGATION_VIOLATION",),
                    blocking_finding_ids=violating,
                )
            )
        elif missing or unverifiable:
            reasons = []
            if missing:
                reasons.append("CLAIM_DEPENDENCY_MISSING")
            if unverifiable:
                reasons.append("CLAIM_DEPENDENCY_UNVERIFIABLE")
            decisions.append(
                ScientificClaimDecision(
                    claim.claim_id,
                    "UNVERIFIABLE",
                    tuple(reasons),
                    blocking_finding_ids=unverifiable,
                    missing_dependencies=missing,
                )
            )
        else:
            decisions.append(
                ScientificClaimDecision(
                    claim.claim_id,
                    "ALLOWED",
                    ("CLAIM_DEPENDENCIES_SATISFIED",),
                )
            )
    return tuple(decisions)


def _receipt_hash(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False).encode("utf-8")).hexdigest()


class ScientificClaimSession:
    """Incremental reevaluation of explicit scientific claim dependencies.

    This reuses the existing gate, including its wildcard-node semantics. A
    transaction invalidates claims whose required coordinates changed, including
    deletions and changes in reasons/sources without a status change. A policy
    version change conservatively invalidates all claims. Receipts bind snapshots
    and their predecessor; they provide traceability, not source authenticity.
    """

    def __init__(self, claims: Sequence[ScientificClaim],
                 observations: Sequence[CanonicalObservation], *, policy_version: str = "initial") -> None:
        self._check_types(claims, observations, policy_version)
        initial = evaluate_scientific_claims(claims, observations)
        self._claims = {c.claim_id: c for c in claims}
        self._observations = {self._coordinate(o): o for o in observations}
        self._decisions = {d.claim_id: d for d in initial}
        self._decision_leaves = {d.claim_id: _receipt_hash(d.to_dict()) for d in initial}
        self._policy_sha256 = _receipt_hash([asdict(self._claims[k]) for k in sorted(self._claims)])
        self._leaves = {key: _receipt_hash(o.to_dict()) for key, o in self._observations.items()}
        self._policy_version = policy_version
        self._revision = 0
        self._previous_receipt = None
        self._rebuild_index()
        self.receipt = self._make_receipt(tuple(sorted(self._claims)), (), ())

    @staticmethod
    def _coordinate(o: CanonicalObservation) -> tuple[str, str, str]:
        return (o.candidate_id, o.node_id, o.obligation_id)

    @staticmethod
    def _check_types(claims, observations, version):
        if not isinstance(version, str) or not version:
            raise ValueError("policy_version must be a nonempty string")
        if any(not isinstance(c, ScientificClaim) for c in claims):
            raise TypeError("claims must contain ScientificClaim values")
        if any(not isinstance(o, CanonicalObservation) for o in observations):
            raise TypeError("observations must contain CanonicalObservation values")

    def _rebuild_index(self) -> None:
        self._dependents: dict[tuple[str, str], set[str]] = {}
        self._by_pair: dict[tuple[str, str], set[tuple[str, str, str]]] = {}
        for c in self._claims.values():
            for candidate in c.candidate_ids:
                for obligation in c.required_obligation_ids:
                    self._dependents.setdefault((candidate, obligation), set()).add(c.claim_id)
        for coordinate in self._observations:
            self._by_pair.setdefault((coordinate[0], coordinate[2]), set()).add(coordinate)

    def _relevant(self, c: ScientificClaim) -> tuple[CanonicalObservation, ...]:
        if c.node_ids is not None:
            keys = ((candidate, node, obligation) for candidate in c.candidate_ids
                    for node in c.node_ids for obligation in c.required_obligation_ids)
            return tuple(self._observations[k] for k in keys if k in self._observations)
        keys = {k for candidate in c.candidate_ids for obligation in c.required_obligation_ids
                for k in self._by_pair.get((candidate, obligation), ())}
        return tuple(self._observations[k] for k in sorted(keys))

    @property
    def decisions(self) -> tuple[ScientificClaimDecision, ...]:
        return tuple(self._decisions[k] for k in sorted(self._decisions))

    @property
    def observations(self) -> tuple[CanonicalObservation, ...]:
        return tuple(self._observations[k] for k in sorted(self._observations))

    def _make_receipt(self, invalidated, changed_coordinates, removed_claims) -> dict:
        payload = {
            "schema": "dmcasrc.scientific-claim-revision.initial", "revision": self._revision,
            "policy_version": self._policy_version,
            "policy_sha256": self._policy_sha256,
            "observation_root_sha256": _receipt_hash([self._leaves[k] for k in sorted(self._leaves)]),
            "previous_receipt_sha256": self._previous_receipt,
            "changed_coordinates": [list(k) for k in sorted(changed_coordinates)],
            "invalidated_claim_ids": list(sorted(invalidated)),
            "removed_claim_ids": list(sorted(removed_claims)),
            "recomputed_claim_count": len(invalidated),
            "reused_claim_count": len(self._claims) - len(invalidated),
            "decision_root_sha256": _receipt_hash([self._decision_leaves[k] for k in sorted(self._decision_leaves)]),
        }
        return {**payload, "receipt_sha256": _receipt_hash(payload)}

    def apply(self, *, upserts: Sequence[CanonicalObservation] = (),
              remove_coordinates: Sequence[tuple[str, str, str]] = (),
              claims: Sequence[ScientificClaim] | None = None,
              policy_version: str | None = None) -> tuple[ScientificClaimDecision, ...]:
        """Atomically apply evidence/policy changes; invalid input preserves state."""
        proposed_claims = tuple(self._claims.values()) if claims is None else tuple(claims)
        version = self._policy_version if policy_version is None else policy_version
        self._check_types(proposed_claims, upserts, version)
        ids = [c.claim_id for c in proposed_claims]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate scientific claim IDs are forbidden")
        removed = list(remove_coordinates)
        if any(not isinstance(k, tuple) or len(k) != 3 or
               any(not isinstance(v, str) or not v for v in k) for k in removed):
            raise ValueError("removals require explicit candidate/node/obligation coordinates")
        if len(removed) != len(set(removed)) or any(k not in self._observations for k in removed):
            raise ValueError("duplicate or nonexistent removal coordinate")
        keys = [self._coordinate(o) for o in upserts]
        if len(keys) != len(set(keys)) or set(keys) & set(removed):
            raise ValueError("duplicate/conflicting update coordinates")
        proposed = dict(self._observations)
        for key in removed:
            del proposed[key]
        proposed.update(zip(keys, upserts))
        observation_ids = [o.observation_id for o in proposed.values()]
        if len(observation_ids) != len(set(observation_ids)):
            raise ValueError("duplicate canonical observation IDs are forbidden")
        changed = set(removed) | {k for k in keys if proposed[k] != self._observations.get(k)}
        new_claims = {c.claim_id: c for c in proposed_claims}
        invalidated = {cid for cid, c in new_claims.items() if self._claims.get(cid) != c}
        for key in changed:
            for cid in self._dependents.get((key[0], key[2]), ()):
                c = new_claims.get(cid)
                if c is not None and (c.node_ids is None or key[1] in c.node_ids):
                    invalidated.add(cid)
        if version != self._policy_version:
            invalidated.update(new_claims)
        removed_claims = set(self._claims) - set(new_claims)
        self._previous_receipt = self.receipt["receipt_sha256"]
        self._revision += 1
        policy_changed = new_claims != self._claims
        self._claims, self._observations, self._policy_version = new_claims, proposed, version
        for key in removed:
            self._leaves.pop(key)
        for key in keys:
            self._leaves[key] = _receipt_hash(proposed[key].to_dict())
        if policy_changed:
            self._policy_sha256 = _receipt_hash([asdict(self._claims[k]) for k in sorted(self._claims)])
            self._rebuild_index()
        else:
            for key in removed:
                self._by_pair.get((key[0], key[2]), set()).discard(key)
            for key in keys:
                self._by_pair.setdefault((key[0], key[2]), set()).add(key)
        for cid in removed_claims:
            self._decisions.pop(cid)
            self._decision_leaves.pop(cid)
        for cid in invalidated:
            c = self._claims[cid]
            self._decisions[cid] = evaluate_scientific_claims((c,), self._relevant(c))[0]
            self._decision_leaves[cid] = _receipt_hash(self._decisions[cid].to_dict())
        self.receipt = self._make_receipt(invalidated, changed, removed_claims)
        return self.decisions

    def decision_basis(self, claim_id: str) -> dict:
        c = self._claims[claim_id]
        observations = self._relevant(c)
        return {"claim": asdict(c), "decision": self._decisions[claim_id].to_dict(),
                "required_observations": [o.to_dict() for o in observations],
                "basis_sha256": _receipt_hash([o.to_dict() for o in observations]),
                "revision_receipt_sha256": self.receipt["receipt_sha256"]}


__all__ = [
    "ResolvedClaimScope",
    "resolve_claim_scope",
    "evaluate_resolved_claim",
    "evaluate_claims",
    "evaluate_scientific_claims",
    "ScientificClaimSession",
]
