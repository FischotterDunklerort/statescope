"""Finite native contracts; these helpers do not make audit decisions."""
def tfidf_producer(m):
    from .state_lineage import observed_tfidf
    return observed_tfidf(m) and m.use_idf is True

def materialized_text(value):
    import numpy as np
    import pandas as pd
    return type(value) in {list,tuple,np.ndarray,pd.Series} and all(type(x) is str for x in value)

def csr_output(data):
    import numpy as np
    from scipy.sparse import csr_matrix
    return type(data) is csr_matrix and data.dtype == np.dtype('float64')

def reviewed_weight(m, data, w):
    if w is None:
        return True
    import numpy as np
    from sklearn.linear_model import SGDClassifier
    return (type(m) is SGDClassifier and m.loss == 'log_loss' and
            m.learning_rate == 'optimal' and type(w) is np.ndarray and
            w.dtype == np.dtype('float64') and type(getattr(data,'shape',None)) is tuple
            and w.shape == (data.shape[0],) and np.isfinite(w).all() and (w >= 0).all())

def forest_configuration(m):
    import sklearn
    import numbers
    from sklearn.ensemble import RandomForestClassifier
    return (sklearn.__version__ == '1.8.0' and type(m) is RandomForestClassifier
            and m.bootstrap is False and m.class_weight is None and m.n_jobs == 1
            and m.criterion == 'gini' and m.monotonic_cst is None and not m.oob_score
            and isinstance(m.random_state, numbers.Integral)
            and m.max_samples is None)

def forest_snapshot(m):
    from .capture import model_state_sha256
    from sklearn.tree import DecisionTreeClassifier
    import numpy as np
    if (not forest_configuration(m) or getattr(m,'n_outputs_',0) != 1
            or not isinstance(getattr(m,'classes_',None), np.ndarray)
            or len(m.classes_) != 2):
        return None
    trees = tuple((t,t.tree_,model_state_sha256(t)) for t in m.estimators_
                  if type(t) is DecisionTreeClassifier )
    if len({t.n_features_in_ for t,_,_ in trees}) != 1 or len(trees) != len(m.estimators_) or not trees or any(d is None for _,_,d in trees):
        return None
    return (tuple(m.classes_.tolist()), trees)
