"""Versioned state ancestry for supported capture adapters.

Graph reachability and adapter adequacy are separate contracts. Edges here are
established operation dependencies; speculative sources must be represented as
gaps, not as definite violating ancestors. The graph does not infer arbitrary
Python data or control dependencies.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
from .events import EventType, RuntimeEvent

FITS = {EventType.MODEL_FIT, EventType.TRANSFORM_FIT}

def observed_tabular_transformer(value):
    """Finite sklearn 1.8 row-preserving tabular preprocessing profiles."""
    import sklearn
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import OneHotEncoder
    if sklearn.__version__ != '1.8.0': return False
    if type(value) is SimpleImputer:
        return value.strategy in {'mean', 'median', 'most_frequent', 'constant'} and not value.add_indicator
    if type(value) is OneHotEncoder:
        return (value.categories == 'auto' if isinstance(value.categories, str) else False) and value.feature_name_combiner == 'concat'
    return False

def observed_column_transformer(value):
    """Reviewed serial sklearn 1.8 column composition, without selector callbacks."""
    import sklearn
    from sklearn.compose import ColumnTransformer
    from joblib import effective_n_jobs
    return (sklearn.__version__ == '1.8.0' and type(value) is ColumnTransformer
            and effective_n_jobs(value.n_jobs) == 1
            and all(not callable(columns) for _, _, columns in value.transformers)
            and not callable(value.verbose_feature_names_out))


def observed_identity_transformer(value):
    """The exact stateless identity used by ColumnTransformer's passthrough."""
    import sklearn
    from sklearn.preprocessing import FunctionTransformer
    return (sklearn.__version__ == '1.8.0' and type(value) is FunctionTransformer
            and value.func is None and value.inverse_func is None
            and value.kw_args is None and value.inv_kw_args is None
            and value.validate is False and value.accept_sparse is True
            and value.check_inverse is False and value.feature_names_out == 'one-to-one'
            and set(vars(value)) <= {'func', 'inverse_func', 'validate', 'accept_sparse',
                'check_inverse', 'feature_names_out', 'kw_args', 'inv_kw_args',
                '_sklearn_output_config', 'n_features_in_', 'feature_names_in_'})


def observed_tfidf(value):
    """Exact sklearn1.8 content/word path; CountVectorizer enumerates CSR rows.

    Vocabulary and IDF are fitted producer state, not row-identity evidence.
    Custom callbacks, supplied vocabularies and subclasses require other rules.
    """
    import sklearn
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    return (sklearn.__version__ == '1.8.0' and type(value) is TfidfVectorizer
            and value.input == 'content' and value.analyzer == 'word'
            and value.preprocessor is None and value.tokenizer is None
            and value.vocabulary is None and value.dtype is np.float64)


def receiver_operation_contract(value,method):
    """Exact supported receiver types; a hook alone does not certify semantics."""
    from .stateless_transforms import is_hashing, configuration
    if is_hashing(value):
        # fit/partial_fit validates this pure operation; it learns no data state.
        return ("RESET", False) if configuration(value) is not None and method in {"fit","partial_fit"} else ("UNESTABLISHED",False)
    from .component_registry import extension_receiver
    extra = extension_receiver(value, method)
    if extra is not None:return extra
    from sklearn.feature_extraction.text import TfidfVectorizer
    if type(value) is TfidfVectorizer:
        return ('RESET', True) if observed_tfidf(value) and method in {'fit','fit_transform'} else ('UNESTABLISHED',False)
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import OneHotEncoder
    if type(value) in {SimpleImputer, OneHotEncoder}:
        return ('RESET', True) if observed_tabular_transformer(value) and method in {'fit', 'fit_transform'} else ('UNESTABLISHED', False)
    if isinstance(value, ColumnTransformer):
        return ('RESET', True) if observed_column_transformer(value) and method in {'fit', 'fit_transform'} else ('UNESTABLISHED', False)
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression,LinearRegression,LogisticRegressionCV,SGDClassifier,SGDRegressor
    from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler,MinMaxScaler,RobustScaler
    from sklearn.feature_selection import SelectKBest,SelectPercentile,VarianceThreshold,SelectFromModel
    from sklearn.naive_bayes import GaussianNB
    from sklearn.ensemble import RandomForestClassifier,StackingClassifier,VotingClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC,LinearSVC
    from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
    from sklearn.multiclass import OneVsRestClassifier
    supported={CalibratedClassifierCV,PCA,LogisticRegression,LinearRegression,LogisticRegressionCV,SGDClassifier,SGDRegressor,
        GridSearchCV,RandomizedSearchCV,Pipeline,StandardScaler,MinMaxScaler,RobustScaler,SelectKBest,SelectPercentile,
        VarianceThreshold,SelectFromModel,GaussianNB,RandomForestClassifier,StackingClassifier,VotingClassifier,
        MLPClassifier,KNeighborsClassifier,SVC,LinearSVC,QuadraticDiscriminantAnalysis,OneVsRestClassifier}
    import sklearn
    if sklearn.__version__ == '1.7.2':
        # Version-transfer profile: only these additional leaf operations have
        # been reviewed for the new Bagging family, not the whole 1.8 census.
        from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
        supported.update({DecisionTreeClassifier,DecisionTreeRegressor})
    if sklearn.__version__ == '1.8.0':
        from sklearn.linear_model import (PassiveAggressiveClassifier,PassiveAggressiveRegressor,
            Perceptron,SGDOneClassSVM,ElasticNet,Lasso,MultiTaskElasticNet,MultiTaskLasso,
            HuberRegressor,PoissonRegressor,GammaRegressor,TweedieRegressor)
        from sklearn.naive_bayes import BernoulliNB,MultinomialNB,ComplementNB,CategoricalNB
        from sklearn.preprocessing import MaxAbsScaler,KBinsDiscretizer
        from sklearn.decomposition import IncrementalPCA
        from sklearn.neural_network import MLPRegressor
        from sklearn.ensemble import RandomForestRegressor,ExtraTreesClassifier,ExtraTreesRegressor,GradientBoostingClassifier,GradientBoostingRegressor
        from sklearn.dummy import DummyClassifier,DummyRegressor
        from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
        # Fixed-version adapters: fit resets unless the public warm_start
        # option retains state; partial_fit consumes the previous state.
        # Composite/unreviewed candidates are observed but remain unknown.
        supported.update({PassiveAggressiveClassifier,PassiveAggressiveRegressor,Perceptron,
            SGDOneClassSVM,ElasticNet,Lasso,MultiTaskElasticNet,MultiTaskLasso,HuberRegressor,
            PoissonRegressor,GammaRegressor,TweedieRegressor,BernoulliNB,MultinomialNB,
            ComplementNB,CategoricalNB,MaxAbsScaler,IncrementalPCA,MLPRegressor,
            RandomForestRegressor,ExtraTreesClassifier,ExtraTreesRegressor,
            DummyClassifier,DummyRegressor,DecisionTreeClassifier,DecisionTreeRegressor,
            GradientBoostingClassifier,GradientBoostingRegressor,KBinsDiscretizer})
        if type(value) in {GradientBoostingClassifier,GradientBoostingRegressor} and value.init not in {None,'zero'}:
            return 'UNESTABLISHED',False
    if type(value) not in supported:return 'UNESTABLISHED',False
    # The reviewed ensemble continuation rule is tree growth. A warm call
    # without new trees can validate data without learning from those members;
    # it must not be labeled a source-bearing growth operation.
    if sklearn.__version__ == '1.8.0' and bool(getattr(value,'warm_start',False)):
        from sklearn.ensemble import (RandomForestClassifier,RandomForestRegressor,
            ExtraTreesClassifier,ExtraTreesRegressor,GradientBoostingClassifier,GradientBoostingRegressor)
        if type(value) in {RandomForestClassifier,RandomForestRegressor,ExtraTreesClassifier,
                ExtraTreesRegressor,GradientBoostingClassifier,GradientBoostingRegressor}:
            previous_trees=getattr(value,'estimators_',())
            if len(previous_trees) and value.n_estimators<=len(previous_trees):
                # Classifier.fit still reconstructs classes_ from the new y.
                # Record that direct batch participation without pretending
                # that all no-growth retention/metadata behavior is reviewed.
                return 'UNESTABLISHED', type(value) is GradientBoostingClassifier
    # prefit selector fit validates/imports a supplied estimator; X is not a
    # newly learned training source. Its import path has no capture contract.
    if type(value) is SelectFromModel and value.prefit:return 'UNESTABLISHED',False
    continuation=method=='partial_fit' or bool(getattr(value,'warm_start',False))
    if type(value) is LogisticRegression and value.solver=='liblinear':continuation=False
    return ('CONTINUE' if continuation else 'RESET'),True

def observed_pipeline_container(value):
    """Exact Pipeline stores learned state in its declared fitted children.

    A separately assembled container needs complete child provenance. Custom
    subclasses and additional parent fields have no container contract here.
    """
    try:
        from sklearn.pipeline import Pipeline
        return type(value) is Pipeline and set(vars(value)) <= {
            "steps","transform_input","memory","verbose"}
    except ImportError:
        return False

def observed_calibration_container(value):
    """The fixed-version private container has no learned state of its own.

    Its estimator and calibrators remain predictive roots and require evidence.
    This contract removes a spurious container-fit requirement, not child gaps.
    """
    try:
        from sklearn.calibration import _CalibratedClassifier
        return type(value) is _CalibratedClassifier and set(vars(value)) <= {
            'estimator','calibrators','classes','method'}
    except ImportError:
        return False

def predictive_components(value):
    """Named, supported predictive children; cloning templates are excluded."""
    children = []
    from sklearn.compose import ColumnTransformer
    if isinstance(value, ColumnTransformer):
        # transformers contains construction templates; prediction consumes
        # only fitted, nonempty, non-dropped branches in transformers_.
        for name, child, columns in getattr(value, 'transformers_', ()):
            if isinstance(child, str) or (hasattr(columns, '__len__') and len(columns) == 0):
                continue
            children.append(('transformers_.' + name, child))
    from sklearn.preprocessing import KBinsDiscretizer
    from sklearn.ensemble import GradientBoostingClassifier,GradientBoostingRegressor
    if type(value) is KBinsDiscretizer and hasattr(value,'_encoder'):
        children.append(('_encoder',value._encoder))
    if type(value) in {GradientBoostingClassifier,GradientBoostingRegressor} and hasattr(value,'init_'):
        children.append(('init_',value.init_))
    if observed_calibration_container(value):
        children=[('estimator',value.estimator),('calibrators',value.calibrators)]
    for name in ("steps", "estimator", "base_estimator", "calibrated_classifiers_",
                 "best_estimator_", "estimators_", "final_estimator_", "estimator_"):
        if name in {"estimator", "base_estimator"} and any(
            getattr(value, fitted, None) is not None for fitted in
            ("best_estimator_", "estimators_", "calibrated_classifiers_", "estimator_")):
            continue
        if observed_calibration_container(value):continue
        # For a fitted forest this is a child-construction template. Prediction
        # consumes estimators_, whose source requirements are retained.
        if name=='estimator_':
            from .component_registry import component_spec
            spec = component_spec(value)
            if spec is not None and name in spec.template_fields:continue
        child = getattr(value, name, None)
        if child is not None:
            children.append((name, child))
    flattened=[]
    def walk(path, child):
        import numpy as np
        if isinstance(child,np.ndarray) and child.dtype==object:
            for index,item in enumerate(child.flat):walk(f'{path}[{index}]',item)
        elif isinstance(child, (tuple, list)):
            for i,item in enumerate(child):walk(f"{path}[{i}]",item)
        elif isinstance(child, dict):
            for key,item in child.items():walk(f"{path}.{key}",item)
        elif hasattr(child,"fit") or hasattr(child,"estimator"):
            flattened.append((path,child))
    for name,child in children:walk(name,child)
    return flattened

@dataclass
class VersionNode:
    event: RuntimeEvent
    predecessors: set[str] = field(default_factory=set)
    gaps: set[str] = field(default_factory=set)

class StateVersionGraph:
    def __init__(self, events):
        self.events = tuple(events)
        self.nodes = {}
        latest = {}
        observed={e.event_id:e for e in self.events}
        for event in self.events:
            if event.event_type not in FITS or event.attributes.get("nested_fit_call"):
                continue
            owner=event.attributes.get("operation_object_id")
            node=VersionNode(event)
            semantics=event.attributes.get("fit_semantics")
            previous=event.attributes.get("previous_state_event_id")
            if semantics=="CONTINUE":
                if previous:node.predecessors.add(previous)
                elif event.attributes.get("preexisting_state_unobserved"):
                    node.gaps.add("CONTINUATION_HISTORY_NOT_OBSERVED:"+str(owner))
            elif semantics!="RESET":
                node.gaps.add("STATE_SEMANTICS_NOT_ESTABLISHED:"+str(owner))
            for key in ("import_state_event_ids", "input_state_event_ids", "component_state_event_ids"):
                node.predecessors.update(event.attributes.get(key,()))
            node.gaps.update(event.attributes.get("state_dependency_gaps",()))
            for key in event.attributes.get("stateless_transform_event_ids",()):
                proof=observed.get(key)
                if proof is None or proof.attributes.get("stateless_transform_verified") is not True:
                    node.gaps.add("STATELESS_TRANSFORM_EVIDENCE_NOT_OBSERVED:"+key)
            self.nodes[event.event_id]=node
            latest[owner]=event.event_id
        self.latest=latest

    def history(self, roots):
        """Return all established reachable fits and separately propagated gaps."""
        selected=set();gaps=set();active=set()
        # Enter/exit frames preserve cycle checks without Python stack limits.
        for root in roots:
            stack=[(root,False)]
            while stack:
                key,leaving=stack.pop()
                if leaving:
                    active.discard(key);continue
                if key in active:
                    gaps.add('STATE_DEPENDENCY_CYCLE:'+key);continue
                if key in selected:continue
                node=self.nodes.get(key)
                if node is None:
                    gaps.add('STATE_VERSION_NOT_OBSERVED:'+key);continue
                active.add(key);selected.add(key);gaps.update(node.gaps)
                stack.append((key,True))
                stack.extend((p,False) for p in sorted(node.predecessors,reverse=True))
        return selected,gaps
