"""Executable interpretation of the existing selected-trace evidence contract.

None as an upper set means an open bound over the application identity universe.
Owned-component upper bounds never become confirmed members. Hash and identity
associations are conditional on the cooperative observation boundary.
"""
from .state_lineage import FITS

def source_bounds(trace, receipt, *, association_valid, assessment, formally_complete):
    lower=set();upper=set();external=set();missing=[]
    for event in trace:
        if event.event_type not in FITS:continue
        attrs=event.attributes
        external.update(attrs.get('external_source_upper_bound', ()))
        if attrs.get('members_are_training') is False:continue
        if 'source_ids' not in attrs:
            missing.append('membership:'+event.event_id);continue
        members={str(x) for x in attrs['source_ids']}
        upper.update(members)
        if attrs.get('membership_origin')!='certified_owned_component_upper_bound':
            lower.update(members)
    for item in receipt.get('component_evidence',()):
        upper.update(str(x) for x in item.get('member_upper_bound',()))
    gaps=sorted(set(receipt.get('gaps',()))|set(missing)
        |({'selected_state_association'} if not association_valid else set())
        |({'formal_source_or_policy_obligation'} if not formally_complete else set()))
    roots=bool(receipt.get('root_state_versions') or receipt.get('component_evidence'))
    closed=bool(association_valid and formally_complete and roots and not gaps)
    if not association_valid:lower.clear()
    excluded=set(assessment)
    upper.update(external)
    decision=('UNKNOWN' if not association_valid else
              'VIOLATION' if lower & excluded else
              'PASS' if closed and not upper & excluded else 'UNKNOWN')
    return dict(association_valid=bool(association_valid),confirmed_members=sorted(lower),
        possible_members=sorted(upper) if closed else None,upper_bound_closed=closed,
        open_bound='all application member identities' if not closed else None,
        missing_obligations=gaps,confirmed_excluded_members=sorted(lower & excluded),
        external_bound_exclusions=sorted(external & excluded),
        decision=decision,interpretation='existing exact-or-open selected-trace semantics; no finite partial upper-bound inference')
