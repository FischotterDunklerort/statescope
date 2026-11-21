"""Two family adapters added after freezing the shared checker.

Only exact sklearn 1.8.0 / 1.7.2 classes; synchronous execution and bounded native
estimators. No custom callbacks, prefit imports or unknown pipeline caching.
No audit decisions or case identifiers appear in these specifications.
"""
from .component_registry import ComponentSpec


def supported_base(value, classifier, *, depth=0):
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
    from sklearn.linear_model import SGDClassifier, SGDRegressor, LogisticRegression, LinearRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    leaves = (DecisionTreeClassifier, SGDClassifier, LogisticRegression) if classifier else (DecisionTreeRegressor, SGDRegressor, LinearRegression)
    if type(value) in leaves:return True
    if type(value) is Pipeline and depth < 2 and value.memory is None and value.transform_input is None:
        return (len(value.steps) >= 1 and all(type(child) in (StandardScaler, MinMaxScaler)
                for _, child in value.steps[:-1])
                and supported_base(value.steps[-1][1], classifier, depth=depth+1))
    return False


def specs():
    return _version_specs('1.8.0') + _version_specs('1.7.2')


def _version_specs(version):
    from sklearn.ensemble import BaggingClassifier, BaggingRegressor
    from sklearn.multioutput import MultiOutputClassifier, MultiOutputRegressor
    prefix='checked-components-v2/sklearn-'+version
    return (
        ComponentSpec(prefix+'/bagging-clf',(BaggingClassifier,),version,
            lambda v: v.n_jobs in (None,1) and (v.estimator is None or supported_base(v.estimator,True)),('estimator_',)),
        ComponentSpec(prefix+'/bagging-reg',(BaggingRegressor,),version,
            lambda v: v.n_jobs in (None,1) and (v.estimator is None or supported_base(v.estimator,False)),('estimator_',)),
        ComponentSpec(prefix+'/multioutput-clf',(MultiOutputClassifier,),version,
            lambda v: v.n_jobs in (None,1) and supported_base(v.estimator,True),partial_fit=True,fit_continues=lambda v:False),
        ComponentSpec(prefix+'/multioutput-reg',(MultiOutputRegressor,),version,
            lambda v: v.n_jobs in (None,1) and supported_base(v.estimator,False),partial_fit=True,fit_continues=lambda v:False),
    )
