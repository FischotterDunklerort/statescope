"""Native participation evidence before auditing; no numerical-influence oracle."""
from pathlib import Path
import sys,json,copy,inspect,warnings
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import GradientBoostingClassifier,GradientBoostingRegressor
import sklearn.linear_model._stochastic_gradient as sg
from unittest.mock import patch
R=Path(__file__).resolve().parents[2];N=R/'outputs/statescope-sqj-submission-integration/submission-v3.0'
def snapshot(m):
    return {k:np.asarray(getattr(m,k)).tolist() for k in ['coef_','intercept_','t_','classes_','n_estimators_','train_score_'] if hasattr(m,k)}
def execute(case,audit=None):
    A=np.array([[1.,0.],[-1.,0.],[2.,1.],[-2.,-1.]])
    E=np.array([[10.,0.],[-10.,0.]])
    ya=np.array([1,0,1,0]);ye=np.array([1,0]);events=[]
    if audit:
        audit.bind_members(A,[f'A:{i}' for i in range(4)])
        audit.bind_members(E,['E:0','E:1'])
    if case.startswith('sgd'):
        m=SGDClassifier(loss='hinge',penalty=None,learning_rate='constant',eta0=.1,shuffle=False,random_state=71)
        original=sg.fit_binary
        def native(*args,**kw):
            bound=inspect.signature(original).bind(*args,**kw).arguments
            events.append({'entry':'fit_binary','X':np.asarray(bound['X']).tolist(),
                           'sample_weight':np.asarray(bound['sample_weight']).tolist(),
                           'coef_before':snapshot(args[0])})
            return original(*args,**kw)
        with patch.object(sg,'fit_binary',native):
            m.partial_fit(A,ya,classes=[0,1]);before=snapshot(m)
            margins=(2*ye-1)*m.decision_function(E)
            m.partial_fit(E,ye,sample_weight=np.zeros(2) if case=='sgd-zero-weight' else None)
        after=snapshot(m)
        facts={'solver_entries':events,'margins_before':margins.tolist(),
               'loss_gradient_zero':bool(np.all(margins>=1)),'before':before,'after':after,
               'coefficient_equal':before['coef_']==after['coef_'],'intercept_equal':before['intercept_']==after['intercept_'],
               'counter_delta':after['t_']-before['t_'],'reference_source':'EXPOSED',
               'policy_basis':'Batch traverses the native solver and advances optimizer state under whole-batch construction participation.'}
    else:
        regression=case=='boosting-regression-no-growth'
        m=(GradientBoostingRegressor if regression else GradientBoostingClassifier)(n_estimators=3,warm_start=True,random_state=71)
        m.fit(A,ya);before=snapshot(m);oldtrees=tuple(m.estimators_.flat);oldclasses=getattr(m,'classes_',None)
        labels=ye+10 if case=='boosting-new-labels-no-growth' else ye
        if case=='boosting-growth':m.n_estimators=4
        with warnings.catch_warnings(record=True) as messages:
            m.fit(E,labels)
        facts={'before':before,'after':snapshot(m),'tree_count_before':len(oldtrees),
               'tree_count_after':len(m.estimators_.flat),'retained_tree_objects':all(t is u for t,u in zip(oldtrees,m.estimators_.flat)),
               'class_object_replaced':oldclasses is not None and oldclasses is not m.classes_,
               'warnings':[str(w.message) for w in messages],
               'reference_source':'EXPOSED' if not regression else 'UNESTABLISHED',
               'policy_basis':'Classifier fit reconstructs classes_ from the supplied labels, including without tree growth.' if not regression else 'No tree growth; direct construction participation remains unestablished for this regression call.'}
    if audit:
        view=np.zeros((2,1));audit.bind_members(view,['E:0','E:1'])
        facts['audit']={p:audit.check(m,assessment=view,purpose=p) for p in ['prediction','continuation']}
    return facts
def main():
    cases=['sgd-zero-gradient','sgd-zero-weight','boosting-no-growth','boosting-new-labels-no-growth','boosting-regression-no-growth','boosting-growth']
    # Entire reference is frozen before importing the auditor.
    native={case:execute(case) for case in cases}
    dest=N/'verification/native-participation'
    dest.mkdir(exist_ok=True)
    if (dest/'native.json').exists():raise RuntimeError('Preserve prior native run')
    (dest/'native.json').write_text(json.dumps(native,indent=2),encoding='utf-8')
    sys.path[:0]=[str(R),str(R/'outputs/statescope-sqj-lifecycle-revision/runtime')]
    from statescope import Audit
    outputs={}
    for case in cases:
        with Audit(boundary="Native entry and whole-batch participation protocol") as audit:
            outputs[case]=execute(case,audit)
    (dest/'audit.json').write_text(json.dumps(outputs,indent=2),encoding='utf-8')
    print(json.dumps({k:{'reference':v['reference_source'],'actual':{p:x['decision'] for p,x in v['audit'].items()},'class_object_replaced':v.get('class_object_replaced'),'counter_delta':v.get('counter_delta')} for k,v in outputs.items()},indent=2))
if __name__=='__main__':main()

