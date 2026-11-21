"""StateScope: the paper-facing interface over versioned learned-state evidence."""
import runtime
from dmcasrc.autofact import AutoFactCapture
from dmcasrc.autofact.persistence import ProvenanceStore
from dmcasrc.application.formal_contracts import evaluate_formal_contracts,learned_fit_contract
from pathlib import Path
from coverage_guard import CoverageGuard
import copy,hashlib,pickle,re
from dmcasrc.autofact.capture import model_state_sha256
from dmcasrc.autofact.state_lineage import predictive_components
from dmcasrc.autofact.events import EventType, EvidenceSource

def _check_capture_legacy(cap,model,*,purpose='strict'):
    trace=cap.trace_for_model(model,purpose=purpose)
    result=evaluate_formal_contracts(trace,contracts=(learned_fit_contract(),)).results[0]
    reasons=list(result.reason_codes)
    gaps=[g for ev in trace for g in ev.attributes.get('lineage_gaps',[])]
    invalid=[g for g in gaps if any(code in g for code in (
        'CHANGED_OR_UNHASHABLE','STATE_SEMANTICS_NOT_ESTABLISHED',
        'SELECTED_MODEL_FIT_NOT_OBSERVED','FITTED_COMPONENT_NOT_OBSERVED'))]
    unsupported=[g for g in gaps if 'NOT_SUPPORTED' in g or 'STATE_SEMANTICS_NOT_ESTABLISHED' in g]
    # Invalid selected roots cannot certify either a clean or exposed current state.
    decision='UNKNOWN' if result.status.value=='UNVERIFIABLE' or invalid else result.status.value
    if unsupported:reasons.append('UNSUPPORTED_OPERATION')
    if any('CHANGED_OR_UNHASHABLE' in g for g in invalid):reasons.append('STATE_VALIDATION_FAILED')
    reasons.extend(gaps)
    uncovered=getattr(cap,'_unsupported_context_operations',[])
    if uncovered:
        decision='UNKNOWN';reasons.extend(['UNSUPPORTED_OPERATION','UNMODELED_PREPROCESSING_IN_CONTEXT'])
        reasons.extend(r['operation'] for r in uncovered)
    if decision=='UNKNOWN':
        reasons=[r for r in reasons if not r.endswith('_SATISFIED') and r!='ASSESSMENT_MEMBER_USED_FOR_TRANSFORM_FIT']
    raw=list(dict.fromkeys(reasons))
    codes=list(dict.fromkeys(r.split(':',1)[0] for r in raw if re.match(r'^[A-Z][A-Z0-9_]+(?::.*)?$',r)))
    from dmcasrc.autofact.source_bounds import source_bounds
    excluded={str(m) for event in cap.events if event.event_type==EventType.DATA_SPLIT
              and (event.fold_role=='assessment' or event.attributes.get('split_role')=='assessment')
              for m in event.attributes.get('member_ids',())}
    bounds=source_bounds(trace,cap.lineage_receipt,association_valid=not invalid and not uncovered,
        assessment=excluded,formally_complete=result.status.value!='UNVERIFIABLE')
    # A mismatch is reported conservatively and retained for regression review.
    if decision == 'PASS' and bounds.get('external_bound_exclusions'):
        decision = 'UNKNOWN'
        # A possible member in a trusted bound does not supply positive evidence.
        codes.append('EXTERNAL_SOURCE_BOUND_INTERSECTS_EXCLUSION')
        raw.append('EXTERNAL_SOURCE_BOUND_INTERSECTS_EXCLUSION')
    mismatch=decision!=bounds['decision']
    if mismatch:
        decision='UNKNOWN';codes.append('SOURCE_BOUND_INTERPRETATION_MISMATCH')
        raw.append('SOURCE_BOUND_INTERPRETATION_MISMATCH')
    coverage = [{'event': e.event_id, **e.attributes['input_producer_evidence']}
                for e in trace if 'input_producer_evidence' in e.attributes]
    return {'decision':decision,'sources':cap.lineage_receipt,'reasons':codes,'raw_reasons':raw,
        'input_producer_coverage':{'mode':cap.source_coverage, 'fit_inputs':coverage,
            'external_bounds':[{'event':e.event_id,'basis':e.attributes['external_source_basis'],
                'possible_members':e.attributes['external_source_upper_bound']}
                for e in trace if 'external_source_upper_bound' in e.attributes],
            'unresolved':[item for item in bounds['missing_obligations']
                          if item.startswith(('INPUT_PRODUCER_', 'EXTERNAL_FEATURE_', 'ARCHIVED_PRODUCER_'))]},
        'source_bounds':bounds,'bound_interpretation_matches_legacy':not mismatch}

def check_capture(cap, model, *, purpose='strict'):
    result = _check_capture_legacy(cap, model, purpose=purpose)
    from dmcasrc.autofact.witnesses import known_binding_failed
    invalid_current = known_binding_failed(cap,model,purpose)
    replaced_entry = any('CONTINUATION_LEARNED_OBJECT_REPLACED' in reason
                         for reason in result['raw_reasons'])
    if (replaced_entry or invalid_current) and result['decision'] in {'PASS','VIOLATION'}:
        result['legacy_decision'] = result['decision']
        result['legacy_source_bounds'] = result['source_bounds']
        result['decision'] = 'UNKNOWN'
        result['source_bounds'] = dict(result['source_bounds'], association_valid=False,
            confirmed_members=[], confirmed_excluded_members=[], possible_members=None,
            upper_bound_closed=False, decision='UNKNOWN')
        result['reasons'].append('ENTRY_OBJECT_ASSOCIATION_INVALID')
        result['raw_reasons'].append('ENTRY_OBJECT_ASSOCIATION_INVALID')
        result['bound_interpretation_matches_legacy'] = False
    if result['decision'] == 'VIOLATION':
        return result
    from dmcasrc.autofact.witnesses import find_witness
    excluded = {str(m) for event in cap.events if event.event_type==EventType.DATA_SPLIT
                and (event.fold_role=='assessment' or event.attributes.get('split_role')=='assessment')
                for m in event.attributes.get('member_ids',())}
    witness = find_witness(cap, model, excluded, purpose)
    if witness is None:
        return result
    result.setdefault('legacy_decision',result['decision'])
    result.setdefault('legacy_source_bounds',result['source_bounds'])
    result['decision'] = 'VIOLATION'
    result['authorization_kind'] = 'witness'
    result['valid_witness'] = witness
    result['source_bounds'] = dict(association_valid=False, witness_association_valid=True,
        confirmed_members=[witness['member']], confirmed_excluded_members=[witness['member']],
        possible_members=None, upper_bound_closed=False,
        missing_obligations=result['legacy_source_bounds']['missing_obligations'],
        decision='VIOLATION', interpretation='witness-specific positive lower bound')
    result['bound_interpretation_matches_legacy'] = False
    result['reasons'] = list(dict.fromkeys(result['reasons'] + ['VALID_SOURCE_WITNESS']))
    result['raw_reasons'] = list(dict.fromkeys(result['raw_reasons'] + ['VALID_SOURCE_WITNESS']))
    return result


class Audit:
    @property
    def witness_authorization(self):
        return True
    def __init__(self,*,boundary=None,cache=False,source_coverage='strict'):
        if source_coverage not in {'strict', 'legacy'}:
            raise ValueError('source_coverage must be strict or legacy')
        self.source_coverage=source_coverage
        self.boundary=boundary;self.cache=bool(cache);self._cache={};self.cache_hits=0;self.cache_misses=0;self._epoch=0
    def __enter__(self):
        self.capture=AutoFactCapture(track_row_membership=True,test_partition_role='assessment')
        self.capture.source_coverage=self.source_coverage
        self.capture._witness_enabled=self.witness_authorization
        self.capture.__enter__();self.coverage=CoverageGuard(self.capture);self.coverage.__enter__();return self
    def __exit__(self,*args):
        self.coverage.__exit__(*args);return self.capture.__exit__(*args)
    def declare_assessment(self,rows):
        members=self.capture._known_members(rows)
        if members is None:raise ValueError('Assessment rows need declared identities or an observed split')
        self.capture.emit(EventType.DATA_SPLIT,operator='audit.declare_assessment',fold_role='assessment',attributes={'member_ids':list(members)})
        self.invalidate('assessment declaration')
    def bind_members(self,rows,members,*,source=None):
        """Declare row identities, optionally attesting trusted raw or unknown derived input."""
        members=tuple(members)
        count=rows.shape[0] if hasattr(rows,'shape') and len(rows.shape)>0 else len(rows)
        if count!=len(members) or any(not isinstance(m,str) or not m for m in members):
            raise ValueError('Supply one nonempty string identity per row')
        previous=self.capture._known_members(rows)
        if previous is not None and tuple(previous)!=members:
            raise ValueError('An observed row mapping cannot be replaced')
        if source is not None:
            if source not in {'trusted_raw', 'unknown_derived'}:
                raise ValueError('Use trusted_raw, unknown_derived, or omit source')
            self.capture.declare_input_source(rows, source)
        self.capture._row_membership[id(rows)]=(rows,members)
        self.invalidate('member declaration')

    def declare_external_features(self, rows, *, source_upper_bound=None, basis=None):
        """Record unresolved features, or attest a conservative producer bound.

        The caller supplies a trustworthy upper bound and identifies its basis.
        A bound is possible ancestry and can never confirm a training member.
        """
        if self.capture._known_members(rows) is None:
            raise ValueError('Bind the feature row identities before declaring the producer')
        if source_upper_bound is None:
            if basis is not None:
                raise ValueError('A basis accompanies a trusted upper bound')
            self.capture.declare_input_source(rows, 'unknown_derived')
        else:
            if not isinstance(basis, str) or not basis.strip() or isinstance(source_upper_bound, (str, bytes)):
                raise ValueError('A trusted source bound needs member identities and a nonempty basis')
            members = tuple(source_upper_bound)
            if any(not isinstance(member, str) or not member for member in members):
                raise ValueError('Supply nonempty string identities in the source bound')
            self.capture.declare_input_source(rows, 'trusted_external_bound', tuple(sorted(set(members))), basis)
        self.invalidate('external producer declaration')

    def invalidate(self,reason='explicit change'):
        self._cache.clear();self._epoch+=1
    def _key(self,model):
        # Rehash evidence on every call: changing an old record also invalidates.
        evidence=hashlib.sha256(pickle.dumps((self.capture.events,self.capture._component_certificates),protocol=5)).hexdigest()
        nodes=[];seen=set()
        def visit(v):
            if id(v) in seen:return
            seen.add(id(v));nodes.append((id(v),model_state_sha256(v)))
            for name,child in predictive_components(v):visit(child)
        visit(model)
        if any(d is None for _,d in nodes):return None
        return (self._epoch,id(model),tuple(nodes),evidence,repr(self.boundary),repr(getattr(self.capture,'_unsupported_context_operations',[])))
    def check(self,model,*,assessment=None,purpose='strict',validation_mode=None):
        # Preserve the supplied software's strict alias for replay. Public query
        # purposes are prediction/continuation; strict selects association mode.
        if purpose=='strict':
            if validation_mode not in (None,'strict'):raise ValueError('Conflicting strict mode')
            query_purpose='prediction';validation_mode='strict';engine_profile='strict'
        else:
            if purpose not in {'prediction','continuation'}:raise ValueError('Unknown query purpose')
            query_purpose=purpose;validation_mode=validation_mode or 'purpose'
            if validation_mode not in {'strict','purpose'}:raise ValueError('Unknown validation mode')
            engine_profile='strict' if validation_mode=='strict' else purpose
        members=None
        cap=self.capture
        if assessment is not None:
            known=cap._known_members(assessment)
            if known is None:raise ValueError('Query assessment rows need an observed or declared identity mapping')
            members=tuple(sorted(set(known)))
            # A query view changes the policy, never the historical evidence.
            cap=copy.copy(self.capture)
            cap._events=[e for e in self.capture.events if not (e.event_type==EventType.DATA_SPLIT and (e.fold_role=='assessment' or e.attributes.get('split_role')=='assessment'))]
            cap.emit(EventType.DATA_SPLIT,operator='audit.query_policy',source=EvidenceSource.DECLARED,fold_role='assessment',attributes={'member_ids':list(members)})
        key=self._key(model) if self.cache else None
        if self.witness_authorization:
            from dmcasrc.autofact.witnesses import known_binding_failed
            if known_binding_failed(cap,model,engine_profile):
                key=None
        if key is not None:key=(key,members,query_purpose,validation_mode,self.witness_authorization)
        if key is not None and key in self._cache:
            self.cache_hits+=1;return copy.deepcopy(self._cache[key])
        self.cache_misses+=1
        decide=check_capture if self.witness_authorization else _check_capture_legacy
        result=decide(cap,model,purpose=engine_profile)
        result['state_validity_purpose']=query_purpose
        result['validation_mode']=validation_mode
        result['policy']={'mode':'query' if members is not None else 'recorded assessment union','member_ids':list(members) if members is not None else sorted({str(m) for e in cap.events if e.event_type==EventType.DATA_SPLIT and (e.fold_role=='assessment' or e.attributes.get('split_role')=='assessment') for m in e.attributes.get('member_ids',[])})}
        result['boundary']={'declaration':self.boundary,'declared':bool(self.boundary),'kind':'cooperative runtime observation','unchecked_entries':['unobserved array/dataframe writes','unobserved native or extension state changes','equal-content origin transfers']}
        result['release_condition_eligible']=bool(self.boundary) and self.source_coverage=='strict' and result['decision']=='PASS'
        result['release_condition_role']='necessary evidence check only; boundary adequacy and other release checks are external'
        if key is not None and result.get('authorization_kind') != 'witness':
            self._cache.clear();self._cache[key]=copy.deepcopy(result)
        return result
    @staticmethod
    def support(model,method='fit'):
        from dmcasrc.autofact.state_lineage import receiver_operation_contract
        semantics,covered=receiver_operation_contract(model,method)
        return {'class':type(model).__name__,'method':method,
                'operation_supported':semantics in {'RESET','CONTINUE'},
                'members_are_training':covered,'retention':semantics,'full_workflow_support':False}
    def save(self,model,directory,*,storage_format='json'):
        return ProvenanceStore(Path(directory),storage_format=storage_format).save(self.capture,model)
    def load(self,key,directory):return ProvenanceStore(Path(directory)).load(self.capture,key)
    def copy_fitted_state(self,donor):
        from dmcasrc.autofact.provenance_continuity import copy_fitted_state
        model,receipt=copy_fitted_state(self,donor)
        self.last_continuity_receipt=receipt
        return model
    def copy_fitted_parameter(self,donor,*,field):
        from dmcasrc.autofact.provenance_continuity import copy_fitted_parameter
        return copy_fitted_parameter(self,donor,field)
    def transfer_fitted_parameters(self,receiver,donor):
        from dmcasrc.autofact.provenance_continuity import transfer_fitted_parameters
        return transfer_fitted_parameters(self,receiver,donor)
    def recover_state(self,model,*,archive,handle):
        return archive.recover(self,model,handle)
    def check_with_recovery(self,model,*,assessment=None,purpose='strict',validation_mode=None,archive,handle):
        from dmcasrc.autofact.provenance_continuity import ContinuityError
        initial=self.check(model,assessment=assessment,purpose=purpose,validation_mode=validation_mode)
        if initial['decision']!='UNKNOWN':return {**initial,'recovery':{'attempted':False}}
        try:receipt=archive.recover(self,model,handle)
        except ContinuityError as exc:
            return {**initial,'recovery':{'attempted':True,'accepted':False,'reason':str(exc)}}
        final=self.check(model,assessment=assessment,purpose=purpose,validation_mode=validation_mode)
        return {**final,'recovery':{'attempted':True,'accepted':True,'initial_decision':initial['decision'],**receipt}}
