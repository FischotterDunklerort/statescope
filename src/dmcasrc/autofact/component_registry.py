"""Finite, exact-class component specifications and shared evidence checker.

Specifications supply operation semantics, not audit verdicts. The checker
requires an observed creation event, current contract, owner scope, unchanged
finalized child state and a disjoint membership bound. An intersecting upper
bound is insufficient to prove a violation.
"""
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class ComponentSpec:
    identifier: str
    classes: tuple
    library_version: str
    guard: Callable
    template_fields: tuple = ()
    partial_fit: bool = False
    fit_continues: Callable = lambda value: bool(getattr(value, 'warm_start', False))

    def accepts(self, value, version):
        return version == self.library_version and type(value) in self.classes and bool(self.guard(value))

    def receiver(self, value, method):
        if method == 'partial_fit':
            return ('CONTINUE', True) if self.partial_fit else ('UNESTABLISHED', False)
        if method != 'fit': return 'UNESTABLISHED', False
        return ('CONTINUE' if self.fit_continues(value) else 'RESET'), True


@dataclass(frozen=True)
class CertificateCheck:
    valid: bool
    reason: str


def builtin_specs():
    from sklearn.ensemble import (RandomForestClassifier, RandomForestRegressor,
        ExtraTreesClassifier, ExtraTreesRegressor, GradientBoostingClassifier, GradientBoostingRegressor)
    from sklearn.preprocessing import KBinsDiscretizer
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.linear_model import LogisticRegression
    prefix = 'owned-construction-v1/sklearn-1.8.0'
    return (
        ComponentSpec(prefix + '/forest', (RandomForestClassifier, RandomForestRegressor, ExtraTreesClassifier, ExtraTreesRegressor),
                      '1.8.0', lambda v: True, ('estimator_',)),
        ComponentSpec(prefix + '/boosting-default-init', (GradientBoostingClassifier, GradientBoostingRegressor),
                      '1.8.0', lambda v: v.init is None or v.init == 'zero'),
        ComponentSpec(prefix + '/bins-owned-encoder', (KBinsDiscretizer,), '1.8.0', lambda v: True),
        ComponentSpec(prefix + '/calibration', (CalibratedClassifierCV,), '1.8.0',
                      lambda v: type(v.estimator) is LogisticRegression and v.cv != 'prefit'),
    )


def extension_specs():
    # Separate module is added after the shared checker has been frozen. It
    # supplies only declarative applicability/receiver rules, never decisions.
    try:
        from .component_adapters import specs
    except ModuleNotFoundError as error:
        if error.name != __package__ + '.component_adapters': raise
        return ()
    return specs()


def component_spec(value):
    import sklearn
    matches = [spec for spec in (*builtin_specs(), *extension_specs()) if spec.accepts(value, sklearn.__version__)]
    if len(matches) > 1: raise ValueError('ambiguous component contracts')
    return matches[0] if matches else None


def extension_receiver(value, method):
    import sklearn
    candidates=[spec for spec in extension_specs() if type(value) in spec.classes]
    matches=[spec for spec in candidates if spec.accepts(value, sklearn.__version__)]
    if len(matches)>1:raise ValueError('ambiguous receiver contracts')
    if matches:return matches[0].receiver(value,method)
    return ('UNESTABLISHED',False) if candidates else None


def check_certificate(value, owners, assessment, certificate, *, contract_for, fingerprint, events):
    if certificate is None or certificate.get('object') is not value:
        return CertificateCheck(False, 'COMPONENT_SOURCE_MISSING')
    required = {'owner', 'event_id', 'upper_bound', 'fingerprint', 'contract', 'creation_owner_id'}
    if not required <= certificate.keys():
        return CertificateCheck(False, 'COMPONENT_EVIDENCE_INCOMPLETE')
    if not any(certificate.get('owner') is owner for owner in owners):
        return CertificateCheck(False, 'COMPONENT_OWNER_SCOPE_MISMATCH')
    current_contract = contract_for(certificate['owner'])
    if not current_contract or current_contract != certificate.get('contract'):
        return CertificateCheck(False, 'COMPONENT_CONTRACT_CHANGED')
    event = events.get(certificate['event_id'])
    if event is None:
        return CertificateCheck(False, 'COMPONENT_CREATION_EVENT_MISSING')
    if event.attributes.get('operation_object_id') != certificate['creation_owner_id']:
        return CertificateCheck(False, 'COMPONENT_CREATION_OWNER_MISMATCH')
    if set(event.attributes.get('source_ids', ())) != set(certificate['upper_bound']):
        return CertificateCheck(False, 'COMPONENT_BOUND_CREATION_MISMATCH')
    if set(certificate.get('upper_bound', ())).intersection(assessment):
        return CertificateCheck(False, 'COMPONENT_BOUND_INTERSECTS_ASSESSMENT')
    expected = certificate.get('fingerprint')
    if expected is None or fingerprint(value) != expected:
        return CertificateCheck(False, 'COMPONENT_FINAL_STATE_CHANGED')
    return CertificateCheck(True, 'OWNED_COMPONENT_UPPER_BOUND_DISJOINT')
