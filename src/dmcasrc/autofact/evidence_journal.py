"""Reinstall selected evidence from an already recorded trusted local journal.

This is a bounded evidence-cache repair, not provenance reconstruction from
values. No external attestation/import API. Current assessment declarations
are never replayed; unknown objects and changed versions cannot be adopted.
"""
import copy
import hashlib
import json
from pathlib import Path

from .capture import model_state_sha256
from .state_lineage import FITS, StateVersionGraph, predictive_components


def implementation_digest():
    root=Path(__file__).parent
    names=['capture.py','state_lineage.py','component_contracts.py','component_registry.py',
           'component_adapters.py','evidence_journal.py','row_sources.py','external_row_adapters.py','text_row_adapter.py',
           'evidence_planning.py']
    return hashlib.sha256(b''.join(hashlib.sha256((root/name).read_bytes()).digest() for name in names)).hexdigest()


class EvidenceJournal:
    def __init__(self,capture):
        self._capture=capture
        self._contract=implementation_digest()
        self._events={event.event_id:copy.deepcopy(event) for event in capture.events if event.event_type in FITS}
        self._fingerprints={id(obj):(obj,model_state_sha256(obj),capture._state_events.get(id(obj)))
                            for obj in capture._operation_objects.values()}
        self._certificates={key:dict(cert) for key,cert in capture._component_certificates.items()}
        self._event_digest=self._digest()

    def _digest(self):
        return hashlib.sha256(json.dumps([e.to_dict() for e in self._events.values()],sort_keys=True).encode()).hexdigest()

    def _valid(self,capture):
        if capture is not self._capture:raise ValueError('journal belongs to another capture session')
        if implementation_digest()!=self._contract:raise ValueError('journal implementation contract changed')
        if self._digest()!=self._event_digest:raise ValueError('journal event record changed')

    def _objects(self,selected):
        objects=[];seen=set()
        def visit(obj):
            if id(obj) in seen:return
            seen.add(id(obj));objects.append(obj)
            for _,child in predictive_components(obj):visit(child)
        visit(selected);return objects

    def plan(self,capture,selected,*,collect_all=False):
        self._valid(capture)
        objects=[entry[0] for entry in self._fingerprints.values()] if collect_all else self._objects(selected)
        roots=[];blocked=[];valid_objects=set()
        for obj in objects:
            binding=self._fingerprints.get(id(obj))
            current=capture._state_events.get(id(obj))
            if binding is None:
                if current is None and not predictive_components(obj):
                    # A saved, validated creation certificate is an acquisition
                    # path for a child with no separately observed root fit.
                    from .component_registry import check_certificate
                    from .component_contracts import ownership_contract
                    assessment={member for e in capture.events if e.fold_role=='assessment'
                                for member in e.attributes.get('member_ids',())}
                    cert=self._certificates.get(id(obj))
                    checked=check_certificate(obj,objects,assessment,cert,contract_for=ownership_contract,
                        fingerprint=model_state_sha256,events=self._events)
                    if not checked.valid:blocked.append('OBJECT_NOT_REGISTERED:'+type(obj).__name__)
                continue
            if (binding[0] is not obj or binding[1]!=model_state_sha256(obj)
                    or current is None or binding[2] is None or current.event_id!=binding[2].event_id):
                blocked.append('STATE_OR_VERSION_CHANGED:'+type(obj).__name__);continue
            valid_objects.add(id(obj));roots.append(current.event_id)
        graph=StateVersionGraph(tuple(self._events.values()));needed,gaps=graph.history(roots)
        present={event.event_id for event in capture.events}
        events=sorted(needed-present)
        certificates=[key for key,cert in self._certificates.items()
                      if key not in capture._component_certificates and id(cert['owner']) in valid_objects
                      and cert['event_id'] in needed]
        return {'event_ids':events,'certificate_ids':certificates,'blocked':sorted(set(blocked)|set(gaps)),
                'strategy':'all' if collect_all else 'selected','current_assessment_replayed':False}

    def repair(self,capture,selected,*,collect_all=False):
        plan=self.plan(capture,selected,collect_all=collect_all)
        # Replay original versions, not a new fit or an imported PASS decision.
        for event_id in plan['event_ids']:capture._events.append(copy.deepcopy(self._events[event_id]))
        capture._events.sort(key=lambda event:(event.logical_time,event.event_id))
        from .component_registry import check_certificate
        from .component_contracts import ownership_contract
        objects=self._objects(selected) if not collect_all else [binding[0] for binding in self._fingerprints.values()]
        assessment={member for event in capture.events if event.fold_role=='assessment'
                    for member in event.attributes.get('member_ids',())}
        accepted=[]
        for key in plan['certificate_ids']:
            cert=self._certificates[key]
            checked=check_certificate(cert['object'],objects,assessment,cert,contract_for=ownership_contract,
                fingerprint=model_state_sha256,events={e.event_id:e for e in capture.events})
            if checked.valid:capture._component_certificates[key]=dict(cert);accepted.append(key)
        return {**plan,'events_added':len(plan['event_ids']),'certificates_added':len(accepted),
                'unverified_external_evidence_accepted':False}

    def _planning_context(self,capture,selections):
        self._valid(capture)
        # Repaired event caches are intentionally excluded: acquisition changes
        # evidence, whereas target state/version, policy, and contract stay fixed.
        payload={'session':id(capture),'contract':self._contract,
            'policy':[(e.event_id,e.attributes.get('member_ids',())) for e in capture.events
                      if e.fold_role=='assessment' or e.attributes.get('split_role')=='assessment'],
            'outputs':[(name,[(id(obj),model_state_sha256(obj),
                       getattr(capture._state_events.get(id(obj)),'event_id',None))
                       for obj in self._objects(value)]) for name,value in selections.items()]}
        return hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()

    def joint_problem(self,capture,selections,*,event_cost=1.,certificate_cost=1.,bundle_cost=None):
        """Adapter from actual trusted journal obligations to finite planning.

        Costs are caller-declared units, not measured developer effort. Optional
        bundle is an alternative receipt for the SAME recorded events. Initial
        collection was already complete: this only measures evidence-cache repair.
        """
        from .evidence_planning import EvidenceKey,Acquisition,EvidenceProblem
        if not selections:raise ValueError('no selected outputs')
        context=self._planning_context(capture,selections)
        key=lambda kind,identity:EvidenceKey(kind,str(identity),self._contract,'recorded-training-evidence')
        goals=[];blocked=[];needed_events=set();needed_certs=set()
        for name,value in selections.items():
            plan=self.plan(capture,value)
            items={key('event',e) for e in plan['event_ids']}|{key('certificate',c) for c in plan['certificate_ids']}
            # Gaps cannot disappear merely because no acquisition is available.
            items.update(key('unavailable',name+':'+gap) for gap in plan['blocked'])
            goals.append((name,frozenset(items)));blocked.extend((name,gap) for gap in plan['blocked'])
            needed_events.update(plan['event_ids']);needed_certs.update(plan['certificate_ids'])
        actions=[]
        for event_id in sorted(needed_events):
            actions.append(Acquisition('event:'+event_id,frozenset(),frozenset({key('event',event_id)}),event_cost))
        for certificate_id in sorted(needed_certs):
            origin=self._certificates[certificate_id]['event_id']
            req=frozenset({key('event',origin)}) if origin in needed_events else frozenset()
            actions.append(Acquisition('certificate:'+str(certificate_id),req,
                           frozenset({key('certificate',certificate_id)}),certificate_cost))
        if bundle_cost is not None and needed_events:
            actions.append(Acquisition('recorded-event-bundle',frozenset(),
                           frozenset(key('event',e) for e in needed_events),bundle_cost))
        return EvidenceProblem(context,frozenset(),tuple(goals),tuple(actions),tuple(blocked))

    def execute_joint(self,capture,selections,problem,plan):
        from .evidence_planning import execute_plan
        from .component_registry import check_certificate
        from .component_contracts import ownership_contract
        import time
        started=time.perf_counter()
        def acquire(action):
            self._valid(capture)
            return {item:copy.deepcopy(self._events[item.identity]) if item.kind=='event'
                    else dict(self._certificates[int(item.identity)]) for item in action.provides}
        def validate(action,receipt):
            accepted=set()
            for item,record in receipt.items():
                if item.kind=='event':
                    if record.to_dict()!=self._events[item.identity].to_dict():continue
                    if not any(e.event_id==item.identity for e in capture.events):capture._events.append(record)
                    accepted.add(item)
                elif item.kind=='certificate':
                    owners=[obj for value in selections.values() for obj in self._objects(value)]
                    assessment={member for e in capture.events if e.fold_role=='assessment'
                                for member in e.attributes.get('member_ids',())}
                    checked=check_certificate(record['object'],owners,assessment,record,
                        contract_for=ownership_contract,fingerprint=model_state_sha256,
                        events={e.event_id:e for e in capture.events})
                    if checked.valid:capture._component_certificates[int(item.identity)]=record;accepted.add(item)
            capture._events.sort(key=lambda e:(e.logical_time,e.event_id))
            return accepted
        result=execute_plan(problem,plan,current_context=lambda:self._planning_context(capture,selections),
                            acquire=acquire,validate=validate)
        result['validation_seconds']=time.perf_counter()-started
        result['initial_collection_cost_reduced']=False
        return result
