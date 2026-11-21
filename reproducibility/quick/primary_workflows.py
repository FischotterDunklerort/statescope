"""Frozen native schedules. Audit adapters may observe them but do not set source truth."""
from pathlib import Path
import copy
import json
import sys
import warnings

import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler

NATIVE_FIT = SGDClassifier.fit

def classifier():
    return SGDClassifier(loss="log_loss", learning_rate="optimal", shuffle=False,
                         random_state=3071, max_iter=4, tol=None)

def fields(model):
    if isinstance(model, SGDClassifier):
        return {k:np.asarray(getattr(model,k)).tolist() for k in ("coef_","intercept_","classes_","t_")}
    return dict(trees=[tree.tree_.__getstate__()["values"].tolist() for tree in model.estimators_.flat],
                estimators=len(model.estimators_),classes=model.classes_.tolist())

def execute(case, adapter=None):
    x,y=make_classification(n_samples=96,n_features=4,n_redundant=0,random_state=3071)
    arrays={"A":x[:48].copy(),"E":x[:48].copy(),"B":x[48:72].copy()}
    labels={"A":y[:48],"E":y[:48],"B":y[48:72]}
    ids={k:[f"{k}:{i}" for i in range(len(v))] for k,v in arrays.items()}
    calls=[]
    def fit(model, part, partial=False, data=None, mapped=True, **kwargs):
        values=arrays[part] if data is None else data
        if adapter and mapped:
            adapter.bind(values,ids[part])
        previous=fields(model) if hasattr(model,"coef_") or hasattr(model,"estimators_") else None
        result=getattr(model,"partial_fit" if partial else "fit")(values,labels[part],**kwargs)
        if adapter:adapter.observe(model)
        calls.append(dict(method="partial_fit" if partial else "fit",actual_members=ids[part],
                          before=previous,after=fields(model)))
        return result
    m=classifier()
    excluded="E"
    true_sources=set()
    if case in {"direct_unknown","replaced_clean","post_return_copy","equal_inplace_origin"}:
        fit(m,"E")
        clean=classifier()
        clean_x=arrays["A"]+.3 if case=="post_return_copy" else arrays["A"]
        NATIVE_FIT(clean,clean_x,labels["A"])
        if case=="equal_inplace_origin":
            for key,value in vars(clean).items():
                if isinstance(value,np.ndarray) and isinstance(vars(m).get(key),np.ndarray):
                    np.copyto(vars(m)[key],value)
                else:
                    setattr(m,key,copy.deepcopy(value))
            true_sources=set(ids["A"])
        else:
            if case=="post_return_copy":
                fit(m,"B",partial=True)
            m.__dict__.clear()
            m.__dict__.update(copy.deepcopy(vars(clean)))
            true_sources=set(ids["A"])
            if case!="post_return_copy":
                if case=="direct_unknown":m.intercept_[0]+=.75
                fit(m,"B",partial=True)
                true_sources.update(ids["B"])
            if case=="direct_unknown":excluded="B"
    elif case in {"retained_counter","retained_intercept","inplace_edit","trusted_reset"}:
        fit(m,"E")
        if case=="retained_counter":m.t_+=4096
        elif case=="inplace_edit":m.coef_[0,0]+=.75
        else:m.intercept_[0]+=.75
        if case=="trusted_reset":
            fit(m,"A");true_sources=set(ids["A"])
        else:
            fit(m,"B",partial=True);true_sources=set(ids["E"]+ids["B"])
    elif case=="donor_import":
        donor=classifier();fit(donor,"E")
        copied=adapter.copy_coefficient(donor) if adapter else donor.coef_.copy()
        fit(m,"A",coef_init=copied)
        m.intercept_[0]+=.75
        fit(m,"B",partial=True)
        true_sources=set(ids["E"]+ids["A"]+ids["B"])
    elif case=="producer":
        scaler=StandardScaler()
        if adapter:adapter.bind(arrays["E"],ids["E"])
        scaler.fit(arrays["E"])
        fit(m,"A")
        m.intercept_[0]+=.75
        transformed=scaler.transform(arrays["B"])
        fit(m,"B",partial=True,data=transformed)
        calls.append(dict(producer_fit_members=ids["E"],mean=scaler.mean_.tolist(),features=transformed.tolist()))
        true_sources=set(ids["A"]+ids["B"]+ids["E"])
    elif case in {"boosting_rng","no_growth"}:
        m=GradientBoostingClassifier(n_estimators=5,warm_start=True,subsample=.65,max_features=3,random_state=3071)
        if case=="boosting_rng":
            fit(m,"E");m._rng.uniform(size=61);m.n_estimators=7
            fit(m,"B");true_sources=set(ids["E"]+ids["B"])
        else:
            fit(m,"A")
            with warnings.catch_warnings(record=True) as warning:
                fit(m,"E")
            calls.append(dict(warnings=[str(w.message) for w in warning],
                              source_policy="Classifier fit reconstructs classes_ from the supplied labels"))
            true_sources=set(ids["A"]+ids["E"])
    elif case=="missing_members":
        fit(m,"A");m.intercept_[0]+=.75
        fit(m,"E",partial=True,data=arrays["E"].copy(),mapped=False)
        true_sources=set(ids["A"]+ids["E"])
    elif case=="false_identity":
        if adapter:
            adapter.bind(arrays["A"],ids["E"])
        fit(m,"A",mapped=False)
        true_sources=set(ids["A"])
    else:
        raise ValueError(case)
    queries=adapter.query(m,ids[excluded]) if adapter else None
    return dict(case=case,actual_sources=sorted(true_sources),excluded=ids[excluded],
                source_truth="EXPOSED" if true_sources.intersection(ids[excluded]) else "CLEAN",
                native_calls=calls,state=fields(m),queries=queries)

if __name__=="__main__":
    folder=Path(__file__).resolve().parent
    protocol=json.loads((folder/"protocol.json").read_text())
    from sklearn.linear_model import _stochastic_gradient as native_sgd
    original_dataset=native_sgd.make_dataset
    original_kernel=native_sgd._get_plain_sgd_function
    rows=[]
    for case in protocol["cases"]:
        entries=[];datasets=[]
        def dataset(*args,**kwargs):
            array=args[0]
            datasets.append(dict(X=(array.toarray() if hasattr(array,"toarray") else array).tolist(),
                                 y=np.asarray(args[1]).tolist()))
            return original_dataset(*args,**kwargs)
        def kernel_factory(*args,**kwargs):
            kernel=original_kernel(*args,**kwargs)
            def kernel_call(*values,**options):
                entries.append(dict(coefficient_input=np.asarray(values[0]).copy().tolist(),
                                    intercept_input=float(values[1]),counter_input=float(values[25])))
                return kernel(*values,**options)
            return kernel_call
        native_sgd.make_dataset=dataset
        native_sgd._get_plain_sgd_function=kernel_factory
        try:
            row=execute(case["name"])
        finally:
            native_sgd.make_dataset=original_dataset
            native_sgd._get_plain_sgd_function=original_kernel
        row["native_dataset_entries"]=datasets
        row["native_solver_entries"]=entries
        rows.append(row)
    assert all(row["source_truth"]==ref["source"] for row,ref in zip(rows,protocol["cases"]))
    assert not any(k=="audit" or k.startswith("dmcasrc") for k in sys.modules)
    (folder/"native_reference.json").write_text(json.dumps(rows,indent=2),encoding="utf8")
    print("Native schedules",len(rows),"protocol source truth retained; native entry observations recorded")

