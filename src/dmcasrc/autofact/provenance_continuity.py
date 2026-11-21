"""Observed transfers and JSON-only recovery of previously issued evidence.

Trust starts at an observed operation, not at content equality. Archive handles
are capabilities issued by this live store instance. Their authenticated records
can cross capture contexts in this process; reopening a directory does not grant
trust. No model is deserialized by this module. Cooperative Python observation
and the existing operation/state contracts remain assumptions.
"""
from dataclasses import dataclass
from pathlib import Path
import copy
import hashlib
import hmac
import json
import secrets
import time

from .capture import model_state_sha256
from .events import EvidenceSource, EventType
from .identity_evidence import IdentityEvidenceMap
from .persistence import contract, supported_contract, environment, inventory, encoded, content_digest
from .state_lineage import FITS, StateVersionGraph
from .state_identity import IdentityRegistry, LIVE_STATES


class ContinuityError(ValueError):
    pass


@dataclass(frozen=True)
class EvidenceHandle:
    ticket: str
    object_uid: str


def _snapshot(capture, model):
    if getattr(capture, '_unsupported_context_operations', ()):
        raise ContinuityError('OBSERVATION_BOUNDARY_NOT_CLOSED')
    trace = capture.trace_for_model(model, purpose='strict')
    if (not capture.lineage_receipt['root_state_versions'] or any(
            e.event_type in FITS and e.attributes.get('members_are_training') is not False
            and 'source_ids' not in e.attributes for e in trace)):
        raise ContinuityError('SOURCE_UPPER_BOUND_NOT_CLOSED')
    gaps = capture.lineage_receipt['gaps']
    if gaps:
        raise ContinuityError('SOURCE_ASSOCIATION_OR_CHAIN_INCOMPLETE:' + ';'.join(gaps))
    objects, aliases = inventory(model)
    paths = {id(v): p for p, v in objects.items()}
    entries = {}
    for path, value in objects.items():
        event = capture._state_events.get(id(value))
        cert = capture._component_certificates.get(id(value))
        saved = None
        if cert is not None:
            if id(cert['owner']) not in paths:
                raise ContinuityError('COMPONENT_OWNER_OUTSIDE_SELECTED_INVENTORY')
            saved = {k: v for k, v in cert.items() if k not in {'object', 'owner'}}
            saved['owner_path'] = paths[id(cert['owner'])]
        entries[path] = dict(event_id=event.event_id if event else None,
            type=type(value).__module__ + '.' + type(value).__qualname__,
            fingerprint=model_state_sha256(value), content=content_digest(value), certificate=saved)
    roots = capture.lineage_receipt['root_state_versions']
    graph = StateVersionGraph(capture.events)
    keep, missing = graph.history(roots)
    # Component construction certificates may also anchor historical fits.
    for entry in entries.values():
        cert = entry['certificate']
        if cert:
            more, absent = graph.history(cert.get('source_state_event_ids', ()))
            keep.update(more); missing.update(absent)
    if missing:
        raise ContinuityError('SOURCE_CHAIN_INCOMPLETE')
    copies={r['copy_event_id'] for e in capture.events if e.event_id in keep
        for r in e.attributes.get('consumed_parameter_transfers',())}
    return dict(contract=contract(), environment=environment(), source_coverage=capture.source_coverage,
        aliases=aliases, objects=entries,
        events=[e.to_dict() for e in capture.events if (e.event_id in keep and e.event_type in FITS) or e.event_id in copies])


def _attach(capture, model, record, ticket, operation):
    """Build a private evidence view; callers commit it only after rechecking."""
    from dmcasrc.application.managed import _read_event
    from .state_certificates import reviewed_purpose_state, purpose_fingerprint, continuation_input_fingerprint
    if capture.source_coverage == 'strict' and record.get('source_coverage') != 'strict':
        raise ContinuityError('ARCHIVED_PRODUCER_COVERAGE_NOT_VERIFIED')
    objects, aliases = inventory(model)
    if set(objects) != set(record['objects']) or aliases != record['aliases']:
        raise ContinuityError('SELECTED_COMPONENT_STRUCTURE_CHANGED')
    for path, value in objects.items():
        item = record['objects'][path]
        try: current_content = content_digest(value)
        except (ValueError, TypeError) as exc:
            raise ContinuityError('STATE_BINDING_UNHASHABLE') from exc
        if (item['type'] != type(value).__module__ + '.' + type(value).__qualname__
                or item['content'] != current_content):
            raise ContinuityError('STATE_BINDING_CHANGED')
    # Events from independent captures often share default event identifiers.
    # Namespace every archived reference, preserving the original immutable JSON.
    mapping = {e['event_id']: 'continuity:' + ticket + ':' + str(i) for i, e in enumerate(record['events'])}
    reference_keys = {'event_id', 'parent_event_ids', 'previous_state_event_id',
        'import_state_event_ids', 'input_state_event_ids', 'component_state_event_ids',
        'producer_state_event_id', 'latest_owner_event_id', 'source_state_event_ids','source_state','copy_event_id'}
    def remap(value, key=None):
        if key in reference_keys:
            if isinstance(value, str): return mapping.get(value, value)
            if isinstance(value, (list, tuple)): return [mapping.get(v, v) for v in value]
        if isinstance(value, dict): return {k: remap(v, k) for k, v in value.items()}
        if isinstance(value, list): return [remap(v) for v in value]
        return value
    existing = {e.event_id: e.to_dict() for e in capture.events}
    for raw in record['events']:
        value = remap(copy.deepcopy(raw))
        value['attributes']['operation_object_id'] = 'archived:' + ticket + ':' + str(raw['attributes'].get('operation_object_id'))
        if value['event_id'] in existing:
            if existing[value['event_id']] != value: raise ContinuityError('ARCHIVE_EVENT_COLLISION')
        else: capture._events.append(_read_event(value))
    events = {e.event_id: e for e in capture.events}
    added = []
    for path, value in objects.items():
        item = record['objects'][path]
        parent = mapping.get(item['event_id'])
        if parent is not None:
            attrs = dict(events[parent].attributes)
            attrs.update(operation_object_id=f'py:{id(value):x}', fit_semantics='RESET',
                previous_state_event_id=None, import_state_event_ids=[parent], input_state_event_ids=[],
                component_state_event_ids=[], state_dependency_gaps=[], restored_gaps_by_purpose={},
                fitted_state_sha256=model_state_sha256(value), source_ids=[], members_are_training=False,
                learned_state_sha256=model_state_sha256({k: v for k, v in vars(value).items()
                    if k.endswith('_') and not k.startswith('_')}), continuity_record=ticket)
            if reviewed_purpose_state(value):
                attrs.update(prediction_state_sha256=purpose_fingerprint(value, 'prediction'),
                    continuation_input_sha256=continuation_input_fingerprint(value))
            event = capture.emit(events[parent].event_type, operator=operation,
                source=EvidenceSource.DERIVED, attributes=attrs)
            capture._state_events[id(value)] = event
            capture._operation_objects[attrs['operation_object_id']] = value
            learned = {k: v for k, v in vars(value).items() if k.endswith('_') and hasattr(v, 'shape')}
            capture._receiver_learned_objects[id(value)] = learned
            for v in learned.values():
                capture._learned_state_sources[id(v)] = (v, model_state_sha256(v), event.event_id)
            added.append(dict(source_state=parent, target_state=event.event_id, operation=operation, path=path))
        cert = item['certificate']
        if cert is not None:
            cert = remap(copy.deepcopy(cert)); owner = objects[cert.pop('owner_path')]
            cert.update(object=value, owner=owner, fingerprint=model_state_sha256(value))
            capture._component_certificates[id(value)] = cert
    # Rebind reviewed owned-tree purpose anchors after all owners exist.
    from .state_certificates import tree_prediction_fingerprint, CERTIFICATE_VERSION
    for value in objects.values():
        cert = capture._component_certificates.get(id(value))
        if cert is not None and cert.get('purpose_contract') == CERTIFICATE_VERSION:
            owner = capture._state_events.get(id(cert['owner']))
            if owner is not None:
                cert.update(prediction_fingerprint=tree_prediction_fingerprint(value),
                    continuation_fingerprint=model_state_sha256(value), latest_owner_event_id=owner.event_id)
    capture.trace_for_model(model, purpose='strict')
    if capture.lineage_receipt['gaps']:
        raise ContinuityError('RECOVERED_VIEW_NOT_CLOSED:' + ';'.join(capture.lineage_receipt['gaps']))
    return added


def candidate_view(capture):
    view = copy.copy(capture)
    view._events = list(capture.events)
    for name in ('_state_events', '_operation_objects', '_receiver_learned_objects', '_component_certificates'):
        setattr(view, name, dict(getattr(capture, name)))
    # Exact learned array bindings are weakly held, independently of the old view.
    view._learned_state_sources = IdentityEvidenceMap()
    for k, v in capture._learned_state_sources.items(): view._learned_state_sources[k] = v
    return view


def commit_view(audit, view):
    # Keep the instrumented capture instance: hooks close over this object.
    for name in ('_events', '_sequence', '_state_events', '_operation_objects',
                 '_receiver_learned_objects', '_component_certificates', '_learned_state_sources'):
        setattr(audit.capture, name, getattr(view, name))
    audit.invalidate('verified provenance continuity')


class EvidenceArchive:
    """JSON evidence store anchored by live-issued capabilities and a private MAC.

    A newly constructed store has no authority over old files. Issued handles
    bind the original live object, not any equal-valued object. This deliberately
    does not implement cross-process identity or a public trust infrastructure.
    """
    def __init__(self, directory, *, storage_format='json'):
        if storage_format not in ('json','dictionary'):
            raise ValueError('unsupported evidence storage format')
        self.storage_format = storage_format
        self.directory = Path(directory); self.directory.mkdir(parents=True, exist_ok=True)
        self.identity = LIVE_STATES; self._key = secrets.token_bytes(32); self._issued = {}

    def export(self, audit, model):
        record = _snapshot(audit.capture, model)
        uid = self.identity.uid(model); ticket = secrets.token_hex(16)
        record.update(ticket=ticket, object_uid=uid, generation=self.identity.state(model)[1])
        for path,value in inventory(model)[0].items():
            member_uid,generation=self.identity.state(value)
            record['objects'][path].update(object_uid=member_uid,generation=generation)
        if self.storage_format == 'dictionary':
            from .member_storage import pack
            record = pack(record)
        data = encoded(record); mac = hmac.new(self._key, data, hashlib.sha256).hexdigest()
        self._issued[ticket] = (uid, mac)
        (self.directory / (ticket + '.json')).write_bytes(data)
        return EvidenceHandle(ticket, uid)

    def _verified(self, model, handle):
        if type(handle) is not EvidenceHandle or handle.ticket not in self._issued:
            raise ContinuityError('UNISSUED_EVIDENCE_HANDLE')
        uid, mac = self._issued[handle.ticket]
        if handle.object_uid != uid or self.identity.uid(model) != uid:
            raise ContinuityError('LIVE_OBJECT_IDENTITY_MISMATCH')
        try: data = (self.directory / (handle.ticket + '.json')).read_bytes()
        except OSError as exc: raise ContinuityError('EVIDENCE_RECORD_UNAVAILABLE') from exc
        if not hmac.compare_digest(mac, hmac.new(self._key, data, hashlib.sha256).hexdigest()):
            raise ContinuityError('EVIDENCE_AUTHENTICATION_FAILED')
        from .member_storage import unpack
        record = unpack(json.loads(data))
        if record['generation'] != self.identity.state(model)[1]:
            raise ContinuityError('OBSERVED_STATE_GENERATION_CHANGED')
        objects,aliases=inventory(model)
        if set(objects)!=set(record['objects']) or aliases!=record['aliases']:
            raise ContinuityError('SELECTED_COMPONENT_STRUCTURE_CHANGED')
        for path,value in objects.items():
            item=record['objects'][path]
            if self.identity.state(value)!=(item['object_uid'],item['generation']):
                raise ContinuityError('SELECTED_COMPONENT_IDENTITY_OR_GENERATION_CHANGED')
        if not supported_contract(record['contract']) or record['environment'] != environment():
            raise ContinuityError('EVIDENCE_CONTRACT_OR_ENVIRONMENT_CHANGED')
        if record['ticket'] != handle.ticket or record['object_uid'] != uid:
            raise ContinuityError('EVIDENCE_RECORD_IDENTITY_MISMATCH')
        return record, len(data)

    def recover(self, audit, model, handle):
        start = time.perf_counter(); record, size = self._verified(model, handle)
        if any(id(value) in audit.capture._state_events for value in inventory(model)[0].values()):
            # Never roll a known current state backwards, even to equal values.
            raise ContinuityError('CURRENT_STATE_ALREADY_BOUND')
        view = candidate_view(audit.capture)
        edges = _attach(view, model, record, handle.ticket, 'observed.evidence.recover')
        commit_view(audit, view)
        return dict(kind='verified_history_recovery', record=handle.ticket, added_edges=edges,
            archive_bytes=size, verification_seconds=time.perf_counter() - start)


def copy_fitted_state(audit, donor):
    start = time.perf_counter(); record = _snapshot(audit.capture, donor)
    clone = copy.deepcopy(donor)  # Actual copy occurs inside the observed boundary.
    LIVE_STATES.advance(clone)
    ticket = secrets.token_hex(16); view = candidate_view(audit.capture)
    edges = _attach(view, clone, record, ticket, 'observed.fitted_state.copy')
    commit_view(audit, view)
    return clone, dict(kind='observed_state_copy', added_edges=edges,
        verification_seconds=time.perf_counter() - start, archive_bytes=0)


def copy_fitted_parameter(audit, donor, field):
    import numpy as np
    from sklearn.linear_model import SGDClassifier, SGDRegressor
    if type(donor) not in {SGDClassifier, SGDRegressor} or field not in {'coef_', 'intercept_'}:
        raise ContinuityError('PARAMETER_TRANSFER_PROFILE_NOT_REVIEWED')
    _snapshot(audit.capture, donor)
    value = getattr(donor, field); copied = np.copy(value)
    event = audit.capture._state_events[id(donor)]
    audit.capture._learned_state_sources[id(copied)] = (copied, model_state_sha256(copied), event.event_id)
    receipt=dict(source_state=event.event_id,field=field,target_digest=model_state_sha256(copied),
        target_uid=LIVE_STATES.uid(copied))
    observed=audit.capture.emit(EventType.DATA_CREATE,operator='observed.fitted_parameter.copy',attributes=receipt)
    receipt['copy_event_id']=observed.event_id
    if not hasattr(audit.capture,'_tracked_parameter_copies'):
        audit.capture._tracked_parameter_copies=IdentityEvidenceMap()
    audit.capture._tracked_parameter_copies[id(copied)]=(copied,receipt)
    audit.last_continuity_receipt=dict(kind='observed_parameter_copy',**receipt)
    audit.invalidate('observed parameter copy')
    # No recipient edge is emitted: the existing native fit hook records actual
    # consumption, and rejects a copied array changed before that entry.
    return copied


def transfer_fitted_parameters(audit, receiver, donor):
    """Reviewed lbfgs logistic coefficient assignment, then native warm fit.

    This operation really writes both learned arrays. The retained receiver
    fields and donor origin both remain predecessors. Actual later native fit
    consumption is checked by the existing continuation entry guard.
    """
    import numpy as np
    import sklearn
    from sklearn.linear_model import LogisticRegression
    if (sklearn.__version__ != '1.8.0' or type(receiver) is not LogisticRegression or type(donor) is not LogisticRegression
            or receiver.solver != 'lbfgs' or donor.solver != 'lbfgs'
            or len(receiver.classes_) < 2 or not np.array_equal(receiver.classes_, donor.classes_)
            or receiver.coef_.shape != donor.coef_.shape or receiver.intercept_.shape != donor.intercept_.shape):
        raise ContinuityError('FITTED_PARAMETER_ASSIGNMENT_PROFILE_NOT_REVIEWED')
    _snapshot(audit.capture, receiver); _snapshot(audit.capture, donor)
    old = audit.capture._state_events[id(receiver)]; parent = audit.capture._state_events[id(donor)]
    receiver.coef_ = copy.deepcopy(donor.coef_); receiver.intercept_ = copy.deepcopy(donor.intercept_)
    receiver.warm_start = True
    LIVE_STATES.advance(receiver)
    attrs = dict(old.attributes)
    attrs.update(fit_semantics='RESET', previous_state_event_id=None, source_ids=[], members_are_training=False,
        import_state_event_ids=[old.event_id, parent.event_id], input_state_event_ids=[], component_state_event_ids=[],
        fitted_state_sha256=model_state_sha256(receiver),
        learned_state_sha256=model_state_sha256({k:v for k,v in vars(receiver).items() if k.endswith('_') and not k.startswith('_')}))
    event = audit.capture.emit(old.event_type, operator='observed.fitted_parameters.assign',
        source=EvidenceSource.DERIVED, attributes=attrs)
    audit.capture._state_events[id(receiver)] = event
    audit.capture._receiver_learned_objects[id(receiver)] = {k:v for k,v in vars(receiver).items() if k.endswith('_') and hasattr(v,'shape')}
    audit.invalidate('observed fitted parameter assignment')
    return dict(kind='observed_parameter_assignment', added_edges=[dict(source_state=p, target_state=event.event_id,
        operation='observed.fitted_parameters.assign') for p in (old.event_id, parent.event_id)])
