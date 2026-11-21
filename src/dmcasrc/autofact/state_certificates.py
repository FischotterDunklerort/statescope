"""Purpose-specific association for reviewed boosting and linear profiles.

The global model fingerprint is unchanged. Boosting prediction omits reviewed
RNG fields, while linear prediction omits the update counter and alpha. Continuation
retains both kinds of update state. Only boosting permits an explicitly requested
n_estimators increase at entry; the linear profile permits no parameter delta.
"""
import pickle
import hashlib

PURPOSES = {'strict', 'prediction', 'continuation'}
CERTIFICATE_VERSION = 'boosting-purpose-state-v1/sklearn-1.8.0'


def reviewed_boosting(value):
    import sklearn
    from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
    return (sklearn.__version__ == '1.8.0'
            and type(value) in {GradientBoostingClassifier, GradientBoostingRegressor}
            and (value.init is None or value.init == 'zero'))


def reviewed_sgd(value):
    import sklearn
    import numbers
    import numpy as np
    from sklearn.linear_model import SGDClassifier, SGDRegressor
    if (sklearn.__version__ != '1.8.0'
            or type(value) not in {SGDClassifier, SGDRegressor}
            or value.average is not False or value.early_stopping is not False
            or value.warm_start is not False or value.fit_intercept is not True
            or value.penalty != 'l2'
            or not isinstance(value.random_state, numbers.Integral)
            or getattr(getattr(value, 'coef_', None), 'dtype', None) != np.dtype('float64')
            or not isinstance(getattr(value, 't_', None), numbers.Real)
            or not np.isfinite(value.t_) or value.t_ < 1):
        return False
    pair = (value.loss, value.learning_rate)
    if type(value) is SGDRegressor:
        return pair == ('squared_error', 'optimal') and value.coef_.ndim == 1
    if len(getattr(value, 'classes_', ())) != 2:
        return False
    weights = value.class_weight
    if weights is not None:
        if type(weights) is not dict or not set(weights) <= set(value.classes_):
            return False
        if any(type(v) not in (int, float) or not np.isfinite(v) or v <= 0 for v in weights.values()):
            return False
        if pair != ('log_loss', 'optimal'):
            return False
    pairs = {('hinge','optimal'), ('log_loss','optimal'), ('modified_huber','optimal'),
             ('squared_hinge','constant'), ('perceptron','invscaling'),
             ('log_loss','adaptive'), ('hinge','constant'), ('modified_huber','invscaling')}
    return (pair in pairs and isinstance(value.alpha, numbers.Real)
            and np.isfinite(value.alpha) and value.alpha > 0
            and (value.learning_rate == 'optimal' or
                 (np.isfinite(value.eta0) and value.eta0 > 0))
            and (value.learning_rate != 'invscaling' or
                 (np.isfinite(value.power_t) and value.power_t > 0)))


def reviewed_purpose_state(value):
    return reviewed_boosting(value) or reviewed_sgd(value)


def _hash(value):
    try:
        return hashlib.sha256(pickle.dumps(value, protocol=5)).hexdigest()
    except Exception:
        return None


def tree_prediction_fingerprint(value):
    import numpy as np
    from sklearn.tree import DecisionTreeRegressor
    if type(value) is not DecisionTreeRegressor:
        return None
    fields = dict(vars(value))
    if isinstance(fields.get('random_state'), np.random.RandomState):
        fields.pop('random_state')
    return _hash(('native-regression-tree-prediction', fields))


def purpose_fingerprint(value, purpose):
    if purpose not in PURPOSES:
        raise ValueError('Unknown state validity purpose')
    from .capture import model_state_sha256
    if purpose == 'prediction' and reviewed_sgd(value):
        fields = dict(vars(value))
        # Native decision_function/predict read coef_/intercept_/classes_, not
        # t_ and alpha are read by optimization, not by the reviewed binary
        # hinge/log_loss prediction path in sklearn 1.8.0. Keep all other fields.
        fields.pop('t_', None)
        fields.pop('alpha', None)
        return _hash(('native-sgd-prediction-v2', fields))
    if purpose != 'prediction' or not reviewed_boosting(value):
        return model_state_sha256(value)
    import numpy as np
    fields = dict(vars(value))
    fields.pop('_rng', None)
    if isinstance(fields.get('random_state'), np.random.RandomState):
        fields.pop('random_state')
    trees = fields.get('estimators_')
    if trees is not None:
        fingerprints = [tree_prediction_fingerprint(tree) for tree in trees.flat]
        if any(item is None for item in fingerprints):
            return None
        fields['estimators_'] = ('reviewed-tree-prediction-digests', trees.shape, fingerprints)
    return _hash(('native-boosting-prediction', fields))


def continuation_input_fingerprint(value):
    if reviewed_sgd(value):
        # partial_fit has no permitted entry-state configuration delta.
        return _hash(('native-sgd-continuation-input', dict(vars(value))))
    if not reviewed_boosting(value):
        return None
    fields = dict(vars(value))
    # Native warm_start growth is explicitly requested by changing this count.
    # RNG, learned state and every other parameter remain in the check.
    fields.pop('n_estimators', None)
    return _hash(('native-boosting-continuation-input', fields))
