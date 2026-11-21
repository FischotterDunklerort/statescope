"""Opt-in adapter for the archived Taxi program on sklearn 1.8.0, single-threaded.
The caller owns the cooperative Python runtime. Source declarations originate at file reads.
"""
import builtins
import hashlib
import io
import sys
import types
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import sklearn
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import FunctionTransformer

from dmcasrc.autofact.capture import model_state_sha256
from dmcasrc.autofact.events import EventType
from dmcasrc.autofact.state_identity import LIVE_STATES

TRIPS = "4952f3ba0eba4907b139093b07f80849d497fa223258951a3ec63d582ad52535"
DISTANCES = "f7362a1e0a9564fa0acfdb6e7625ebfeb7597b4ebfc21a1b305a2f17fda7b6c8"
PROGRAM = "3eeb5fdbd28f7fb84dce0af43782d325ba240ab30d8299bef7a0b5e12e84c00f"

_programs = {}

class ContractError(ValueError):
    pass

def require(condition, message):
    if not condition:
        raise ContractError(message)

def read_verified(path, digest):
    data = path.read_bytes()
    require(hashlib.sha256(data).hexdigest() == digest, "unreviewed file: " + path.name)
    return data

def load_program(demo):
    require(sklearn.__version__ == "1.8.0", "sklearn version")
    demo = Path(demo)
    path = demo / "pipelines/mlflow-regression-nyctaxifare.py"
    text = read_verified(path, PROGRAM).decode().split("trips = load_file")[0]
    mod = types.ModuleType("statescope_reviewed_taxi_program")
    sys.modules[mod.__name__] = mod
    exec(compile(text.replace("sparse=False", "sparse_output=False"), str(path), "exec"), mod.__dict__)
    _programs[mod] = (mod.calculate_features, mod.calculate_features.__code__, mod.filter_trips, mod.filter_distances, demo)
    return mod

class TripSplit:
    def __init__(self, program, strict):
        require(program in _programs and (program.filter_trips, program.filter_distances) == _programs[program][2:4], "unreviewed raw filters")
        self.program = program
        base = _programs[program][4] / "datasets/nyc-taxi"
        trips = pd.read_parquet(io.BytesIO(read_verified(base / "trips.parquet", TRIPS)))
        trips["_trip_row"] = np.arange(len(trips))
        distances = pd.read_csv(io.BytesIO(read_verified(base / "distances.csv", DISTANCES)))
        # The extra column records the trip read. Key uniqueness makes the lookup functional.
        joined = program.filter_trips(trips).merge(program.filter_distances(distances),
                    on=["pickup_zip", "dropoff_zip"], validate="many_to_one")
        days = joined.tpep_dropoff_datetime.dt.normalize()
        self.parts = {}
        for role, mask in [("train", days <= "2016-02-15"),
                           ("test", days > "2016-02-15" if strict else days >= "2016-02-15")]:
            frame = joined.loc[mask].copy()
            ids = tuple(TRIPS + ":" + str(i) for i in frame.pop("_trip_row"))
            require(len(set(ids)) == len(ids), "duplicate joined trip")
            self.parts[role] = (frame, ids, frame.copy(deep=True))

    def bind(self, audit, role, frame, ids):
        original, members, snapshot = self.parts[role]
        require(original is frame and members == tuple(ids) and snapshot.equals(frame),
                "raw trip carrier, row order or identities changed")
        # The exact split and file-backed functional join above establish that
        # this matrix has no fitted feature producer.
        audit.bind_members(frame, members, source='trusted_raw')

class TaxiContract:
    """One fit, no pipeline cache, serial nested execution; fail closed outside the receipt."""
    def __init__(self, audit, split, program, model):
        self.audit, self.cap, self.split = audit, audit.capture, split
        self.program, self.model = program, model
        self.fn, self.encoder, self.regressor = model[0][0], model[0][1], model[-1]
        require(type(split) is TripSplit and not audit.cache, "exact trip-source contract and uncached query required")
        require(program is split.program and program in _programs, "unreviewed program")
        self.callback, self.code = _programs[program][:2]
        self.ledger = []
        self.events = []
        self.trained = False

    def check_callback(self):
        f = self.fn
        require(type(f) is FunctionTransformer and f.func is self.callback and f.func.__code__ is self.code
                and f.func.__closure__ is None and f.func.__globals__ is self.program.__dict__
                and all(f.func.__globals__.get(k, getattr(builtins,k)) is getattr(builtins,k)
                        and f.func.__builtins__[k] is getattr(builtins,k) for k in ("list","str"))
                and f.kw_args is None and not f.validate and f.inverse_func is None
                and f.inv_kw_args is None and f.feature_names_out is None,
                "callback or external inputs outside reviewed path")

    def fit(self):
        require(not self.trained and not self.ledger, "one fit per receipt")
        self.check_callback()
        raw, ids, snapshot = self.split.parts["train"]
        policy, test_ids, _ = self.split.parts["test"]
        self.split.bind(self.audit,"train",raw,ids)
        self.split.bind(self.audit,"test",policy,test_ids)
        require(not any(getattr(c,"memory",None) for c in (self.model,self.model[0]))
                and self.encoder.n_jobs in (None,1), "cache/parallelism outside contract")
        require(type(self.model) is Pipeline and type(self.model[0]) is Pipeline and
                len(self.model.steps) == len(self.model[0].steps) == 2 and
                type(self.regressor) is SGDRegressor and type(self.encoder) is ColumnTransformer and
                self.regressor.get_params() == self.program.estimator_fn().get_params() and
                self.encoder.get_params(deep=False).keys() == self.program.transformer_fn()[1].get_params(deep=False).keys(),
                "unsupported pipeline configuration")
        reference = self.program.transformer_fn()[1]
        expected_params = reference.get_params(deep=False)
        actual_params = self.encoder.get_params(deep=False)
        expected_steps = expected_params.pop("transformers")
        actual_steps = actual_params.pop("transformers")
        require(actual_params == expected_params and len(expected_steps)==len(actual_steps)
                and all(type(child) is type(template) and child.get_params()==template.get_params()
                        and name==template_name and columns==template_columns
                        for (name,child,columns),(template_name,template,template_columns)
                        in zip(actual_steps,expected_steps)), "unreviewed column transformations")
        fn_fit = FunctionTransformer.fit
        fitted = []
        def fit_callback(obj, X, *args, **kwargs):
            if obj is self.fn:
                require(X is raw and not fitted and snapshot.equals(X), "callback fit input")
            result = fn_fit(obj, X, *args, **kwargs)
            if obj is self.fn:
                require(result is obj, "callback fit receiver")
                fitted.append(obj)
            return result
        fn_transform = FunctionTransformer.transform
        encode = ColumnTransformer.fit_transform
        learn = SGDRegressor.fit
        targets = raw.fare_amount.copy()
        # Row-preservation follows from the reviewed callback's column-only operations
        # on this exact numeric/date schema. Value checks detect edits, never create IDs.
        expected = snapshot.drop(columns=["tpep_pickup_datetime","tpep_dropoff_datetime"]).copy()
        expected["pickup_dow"] = snapshot.tpep_pickup_datetime.dt.dayofweek
        expected["pickup_hour"] = snapshot.tpep_pickup_datetime.dt.hour
        expected["trip_duration"] = (snapshot.tpep_dropoff_datetime - snapshot.tpep_pickup_datetime).map(
            lambda delta: delta.total_seconds()/60)

        def transform(obj, X, *args, **kw):
            if obj is not self.fn:
                return fn_transform(obj,X,*args,**kw)
            self.check_callback()
            require(fitted == [obj] and X is raw and snapshot.equals(X) and not self.ledger, "unbound callback input")
            before = model_state_sha256(X)
            prior = self.cap._data_state_sources.get(id(X))
            _, gaps = self.cap.input_source_status(X)
            states = ()
            if prior is not None and prior[0] is X:
                gaps += prior[3]
                if prior[1] != before:
                    gaps += ('INPUT_PRODUCER_OUTPUT_CHANGED_BEFORE_TRANSFORM',)
                else:
                    states = prior[2]
            calls = []
            def callback_trace(frame, action, value):
                if frame.f_code is self.code and frame.f_globals is self.program.__dict__:
                    if action == "call":
                        calls.append(("entry", frame.f_locals.get("df") is X))
                    elif action == "return":
                        calls.append(("return", value is X))
            require(sys.getprofile() is None, "another profiler owns callback observation")
            sys.setprofile(callback_trace)
            try:
                result = fn_transform(obj,X,*args,**kw)
            finally:
                sys.setprofile(None)
            require(calls == [("entry", True), ("return", True)], "native callback entry/return unobserved")
            require(result is X and expected.equals(result), "callback return carrier or contents")
            state = model_state_sha256(obj)
            require(state is not None and before is not None, "unverifiable callback state")
            event = self.cap.emit(EventType.TRANSFORM_FIT,operator="taxi.reviewed_function.fit_transform",
                inputs=(X,),outputs=(result,),attributes=dict(contract_scope="assessment_fit_boundary",
                    operation_object_id=f"py:{id(obj):x}",fit_semantics="RESET",members_are_training=False,
                    source_ids=[],fitted_state_sha256=state, input_version=before,
                    output_version=model_state_sha256(result),row_contract="trip-keyed column-only callback",
                    input_state_event_ids=list(states),state_dependency_gaps=list(gaps)))
            # Register this stateless component as a selected root only inside check(),
            # after the complete receipt and its scope have been validated.
            LIVE_STATES.advance(obj)
            self.events.append(event)
            self.ledger.append("date callback returned its input")
            self.cap._data_state_sources[id(result)] = (result,model_state_sha256(result),(event.event_id,),())
            return result

        def encode_matrix(obj, X, *args, **kw):
            if obj is not self.encoder:
                return encode(obj,X,*args,**kw)
            require(self.ledger == ["date callback returned its input"] and X is raw and expected.equals(X)
                    and self.cap._known_members(X)==ids, "encoder input changed")
            matrix = encode(obj,X,*args,**kw)
            self.matrix = matrix
            self.matrix_snapshot = matrix.copy()
            self.ledger.append("encoder returned matrix")
            return matrix

        def train(obj, X, y, *args, **kw):
            if obj is not self.regressor:
                return learn(obj,X,y,*args,**kw)
            require(len(self.ledger)==2 and X is self.matrix and np.array_equal(X,self.matrix_snapshot)
                    and y is targets and snapshot.fare_amount.equals(y) and self.cap._known_members(X)==ids,
                    "actual learner input or target changed")
            result = learn(obj,X,y,*args,**kw)
            self.ledger.append("SGDRegressor.fit returned")
            return result

        with ExitStack() as stack:
            for cls, name, wrapper in [(FunctionTransformer,"fit",fit_callback),(FunctionTransformer,"transform",transform),
                                       (ColumnTransformer,"fit_transform",encode_matrix),(SGDRegressor,"fit",train)]:
                stack.enter_context(patch.object(cls,name,wrapper))
            self.model.fit(raw,targets)
        require(len(self.ledger)==3, "missing actual training entry")
        self.fit_events = tuple(e for e in self.cap.events if e.event_type in {EventType.TRANSFORM_FIT, EventType.MODEL_FIT} and e.logical_time >= self.events[0].logical_time)
        self.learned_arrays = (self.regressor.coef_, self.regressor.intercept_)
        self.children = tuple((name, child) for name,child in self.encoder.named_transformers_.items() if not isinstance(child,str))
        self.selected = (self.model[0], self.model[-1], self.fn, self.encoder)
        for obj in (self.encoder,self.regressor,*(child for _,child in self.children)):
            event = self.cap._state_events.get(id(obj))
            require(event is not None, "unobserved nested fit")
            self.events.append(event)
        self.evidence_versions = tuple(model_state_sha256(e.attributes) for e in self.fit_events)
        require(all(self.evidence_versions), "unverifiable fit evidence")
        self.trained = True
        self.generations = tuple((obj,LIVE_STATES.state(obj)) for obj in (self.fn,self.encoder,self.regressor))

    def verify(self, model):
        require(self.trained and model is self.model and (model[0],model[-1],model[0][0],model[0][1])==self.selected
                and all(self.encoder.named_transformers_.get(k) is v for k,v in self.children),
                "selected model or historical component replaced")
        self.check_callback()
        require(all(any(e is event for e in self.cap.events) for event in self.events + list(self.fit_events)),
                "observed nested history missing")
        require(all(LIVE_STATES.state(obj)==generation for obj,generation in self.generations),
                "selected version subsequently fitted")
        require(self.learned_arrays[0] is self.regressor.coef_ and self.learned_arrays[1] is self.regressor.intercept_,
                "learner arrays replaced")
        require(all(model_state_sha256(e.attributes)==v for e,v in zip(self.fit_events,self.evidence_versions)),
                "issued fit evidence has changed")
        self.split.bind(self.audit,"test",*self.split.parts["test"][:2])

    def check(self, model, *, purpose):
        self.verify(model)
        with patch.dict(self.cap._state_events, {id(self.fn): self.events[0]}):
            return self.audit.check(model, assessment=self.split.parts["test"][0], purpose=purpose)

