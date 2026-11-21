"""Positive source proofs with independent entry and current-version bindings.

The module issues no clean-source certificate. It uses the existing runtime
fingerprints and native-operation guards. Event origins and carrier references
are retained separately from unresolved whole-history requirements.
"""
from dataclasses import dataclass
import weakref

from .capture import model_state_sha256
from .state_certificates import purpose_fingerprint, reviewed_sgd, reviewed_boosting
from .state_identity import LIVE_STATES
from .contract_extension import tfidf_producer, csr_output, reviewed_weight, forest_snapshot, materialized_text
from .state_lineage import predictive_components, observed_pipeline_container

@dataclass(frozen=True)
class WitnessRecord:
    model_ref: object
    event: object
    identity: tuple
    references: tuple
    direct_members: tuple
    predecessors: tuple
    coefficient_ref: object
    coefficient_fingerprint: object
    edge_proofs: tuple = ()
    forest_state: object = None

def references(model):
    """Retain identities of learned containers and actually selected children."""
    import numpy as np
    found = []
    visited = set()
    def visit(value, path):
        if id(value) in visited:
            return
        visited.add(id(value))
        for name, item in vars(value).items():
            if name.endswith("_") and isinstance(item, np.ndarray):
                found.append((path + "." + name, item))
        for name, child in predictive_components(value):
            found.append((path + "." + name, child))
            visit(child, path + "." + name)
    visit(model, "selected")
    return tuple(found)

def same_references(expected, model):
    current = dict(references(model))
    return all(path in current and current[path] is reference() for path,reference in expected)

def _available(capture, record):
    return any(event is record.event for event in capture.events)

def _direct_input_available(capture, record):
    # A missing row-transform proof invalidates this batch's member origin,
    # while a separately verified incoming coefficient may still be followed.
    events={event.event_id:event for event in capture.events}
    return all(key in events and events[key].attributes.get("stateless_transform_verified") is True
               for key in record.event.attributes.get("stateless_transform_event_ids",()))

def _coefficient_source(capture, value, source):
    if source is None or source[0] is not value:
        return None
    records = getattr(capture, "_witness_records", {})
    record = records.get(source[2])
    if record is None or record.coefficient_fingerprint is None or not _available(capture,record):
        return None
    digest = model_state_sha256(value)
    if digest is None or digest != source[1] or digest != record.coefficient_fingerprint:
        return None
    # A distinct array must be an observed tracked parameter copy.
    copied = getattr(capture,"_tracked_parameter_copies",{}).get(id(value))
    if id(record.model_ref()) in getattr(capture,"_witness_failed",set()):
        return None
    if value is not record.coefficient_ref():
        if copied is None or copied[0] is not value:
            return None
        receipt=copied[1]
        if (receipt.get("field")!="coef_" or receipt.get("source_state")!=record.event.event_id
                or receipt.get("target_digest")!=digest or not any(
                    e.event_id==receipt.get("copy_event_id") and e.operator=="observed.fitted_parameter.copy"
                    for e in capture.events)):
            return None
    return record.event.event_id

def observed_features(capture, model, data, event, raw=None):
    from sklearn.preprocessing import StandardScaler
    from .identity_evidence import IdentityEvidenceMap
    if not ((type(model) is StandardScaler and model.with_mean and model.with_std and model.copy)
            or (tfidf_producer(model) and csr_output(data) and materialized_text(raw) and capture._known_members(data) is not None)):
        return
    if event.attributes.get("state_dependency_gaps") or id(model) in getattr(capture,"_tfidf_failed",()):
        return
    prior=capture._state_events.get(id(model))
    record=getattr(capture,"_witness_records",{}).get(prior.event_id) if prior else None
    if (record is None or event.attributes.get("producer_state_event_id")!=prior.event_id
            or not current_binding(capture,record,model,"continuation")):
        return
    if not hasattr(capture,"_witness_features"):
        capture._witness_features=IdentityEvidenceMap()
    capture._witness_features[id(data)]=(data,prior.event_id,event.event_id)

def prepare(capture, model, method, arguments):
    if not getattr(capture, "_witness_enabled", False):
        return None
    data = arguments.get("X")
    if data is None and tfidf_producer(model):
        data = arguments.get("raw_documents")
    tfidf_input = tfidf_producer(model) and materialized_text(data)
    members = capture._known_members(data) if data is not None else None
    incoming = []
    proofs = []
    weights = arguments.get("sample_weight")
    weight_digest = model_state_sha256(weights) if weights is not None else None
    weight_ok = reviewed_weight(model,data,weights)
    old_forest = None
    if method == "fit" and weights is None and getattr(model,"warm_start",False):
        prior = capture._state_events.get(id(model))
        rec = getattr(capture,"_witness_records",{}).get(prior.event_id) if prior else None
        snapshot = forest_snapshot(model) if rec and rec.forest_state else None
        if (snapshot is not None and rec.model_ref() is model and
                LIVE_STATES.state(model)==rec.identity and _available(capture,rec) and
                id(model) not in getattr(capture,"_witness_failed",()) and
                model.n_estimators > len(snapshot[1]) and snapshot==rec.forest_state):
            old_forest = rec
    if weight_ok and reviewed_sgd(model) and method == "partial_fit":
        value = getattr(model,"coef_",None)
        source = capture._learned_state_sources.get(id(value))
        origin = _coefficient_source(capture,value,source)
        previous = getattr(capture,"_witness_records",{}).get(origin)
        if (previous is not None and previous.model_ref() is model and LIVE_STATES.state(model)==previous.identity
                and capture._state_events.get(id(model)) is previous.event):
            incoming.append(("consumed_sgd_coefficient",origin))
    if method == "fit" and type(model).__name__ == "SGDClassifier":
        value = arguments.get("coef_init")
        if value is not None:
            origin = _coefficient_source(capture,value,capture._learned_state_sources.get(id(value)))
            if origin is not None:
                incoming.append(("consumed_coefficient_import",origin))
    producer = capture._data_state_sources.get(id(data))
    receipt=getattr(capture,"_witness_features",{}).get(id(data))
    if (producer is not None and producer[0] is data and not producer[3] and receipt is not None
            and receipt[0] is data and receipt[1] in producer[2] and any(e.event_id==receipt[2] for e in capture.events)):
        if producer[1] is not None and model_state_sha256(data)==producer[1]:
            for origin in (receipt[1],):
                record = getattr(capture,"_witness_records",{}).get(origin)
                if record is not None and _available(capture,record):
                    if ".TfidfVectorizer." in record.event.operator and not (method in {"fit","partial_fit"} and type(model).__name__ in {"SGDClassifier","SGDRegressor"}):
                        continue
                    incoming.append(("consumed_observed_features",origin))
                    if ".TfidfVectorizer." in record.event.operator:
                        proofs.append((origin, receipt[2]))
    return dict(method=method,prior=capture._state_events.get(id(model)),
                old_stages=len(getattr(model,"estimators_",())) if reviewed_boosting(model) else 0,
                members=None if members is None else tuple(members),
                predecessors=tuple(dict.fromkeys(incoming)), sample_weight=weights,
                weight_ok=weight_ok, weight_digest=weight_digest, old_forest=old_forest,
                proofs=tuple(proofs), tfidf_input=tfidf_input)

def mark_failed(capture, model):
    if getattr(capture,"_witness_enabled",False):
        capture.__dict__.setdefault("_witness_failed",set()).add(id(model))

def issue(capture, model, pending, result, operator):
    if pending is None:
        return
    event = capture._state_events.get(id(model))
    tfidf = tfidf_producer(model) and pending["tfidf_input"] and pending["method"] in {"fit","fit_transform"}
    returned_features = tfidf and pending["method"]=="fit_transform" and csr_output(result)
    if (event is None or event is pending["prior"] or (result is not model and not returned_features) or event.operator!=operator
            or event.attributes.get("operation_object_id")!=f"py:{id(model):x}"):
        mark_failed(capture,model)
        return
    attrs = event.attributes
    # Raw candidate membership is observational data, not a source assertion.
    attrs["witness_input_members"] = pending["members"]
    attrs["witness_completed"] = True
    capture.__dict__.setdefault("_witness_failed",set()).discard(id(model))
    from sklearn.preprocessing import StandardScaler
    scaler = type(model) is StandardScaler and model.with_mean and model.with_std and pending["method"]=="fit"
    growth=(reviewed_boosting(model) and pending["method"]=="fit" and model.n_iter_no_change is None
            and model.subsample==1 and len(model.estimators_)>0 and
            (attrs.get("fit_semantics")=="RESET" or len(model.estimators_)>pending["old_stages"]))
    forest = forest_snapshot(model) if pending["method"]=="fit" and pending["sample_weight"] is None else None
    if forest and (forest[1][0][0].n_features_in_ != model.n_features_in_ or attrs.get("preexisting_state_unobserved")):
        forest = None
    older = pending["old_forest"]
    edge = ()
    # Native warm growth appends trees. Recheck the actual prefix and class mapping.
    if forest and older:
        old_labels,old_trees=older.forest_state
        if forest[0]==old_labels and len(forest[1])>len(old_trees) and forest[1][:len(old_trees)]==old_trees:
            edge=(("retained_forest_trees",older.event.event_id),)
    # A preexisting, unproved prefix cannot be used as a local origin bridge.
    if forest and attrs.get("previous_state_event_id") and model.warm_start and not edge:
        forest = None
    weight = pending["sample_weight"]
    weighted_ok = (weight is None or (reviewed_sgd(model) and pending["weight_ok"] and
                   pending["weight_digest"] is not None and model_state_sha256(weight)==pending["weight_digest"]))
    supported = ((reviewed_sgd(model) and pending["method"] in {"fit","partial_fit"})
                 or growth or scaler or tfidf or forest is not None)
    if (not supported or attrs.get("fit_semantics") not in {"RESET","CONTINUE"}
            or attrs.get("members_are_training") is not True or not weighted_ok):
        return
    members = pending["members"]
    if members is not None and set(members)!=set(attrs.get("source_ids",())):
        return
    coefficient = model.coef_ if reviewed_sgd(model) else None
    # Existing fingerprint primitive is used only for runtime state association.
    record = WitnessRecord(model_ref=weakref.ref(model),event=event,identity=LIVE_STATES.state(model),
        references=tuple((path,weakref.ref(obj)) for path,obj in references(model)),direct_members=tuple(members or ()),
        predecessors=pending["predecessors"]+edge,
        coefficient_ref=weakref.ref(coefficient) if coefficient is not None else lambda:None,
        coefficient_fingerprint=model_state_sha256(coefficient) if coefficient is not None else None,
        edge_proofs=pending["proofs"], forest_state=forest)
    capture.__dict__.setdefault("_witness_records",{})[event.event_id] = record
    if returned_features and capture._known_members(result) is not None:
        from .identity_evidence import IdentityEvidenceMap
        if not hasattr(capture,"_witness_features"):
            capture._witness_features=IdentityEvidenceMap()
        capture._witness_features[id(result)]=(result,event.event_id,event.event_id)

def current_binding(capture, record, model, purpose):
    if record.model_ref() is not model or id(model) in getattr(capture,"_witness_failed",set()):
        return False
    if capture._state_events.get(id(model)) is not record.event or LIVE_STATES.state(model)!=record.identity:
        return False
    if not _available(capture,record) or not same_references(record.references,model):
        return False
    if record.forest_state is not None and forest_snapshot(model) != record.forest_state:
        return False
    expected = record.event.attributes.get(
        "prediction_state_sha256" if purpose=="prediction" and (reviewed_sgd(model) or reviewed_boosting(model))
        else "fitted_state_sha256")
    return expected is not None and purpose_fingerprint(model,purpose)==expected

def selected_leaves(model):
    if observed_pipeline_container(model):
        return [(name if suffix=="selected" else name+"."+suffix,leaf)
                for name,child in predictive_components(model) for suffix,leaf in selected_leaves(child)]
    return [("selected",model)]

def find_witness(capture, model, excluded, purpose):
    if getattr(capture,"_unsupported_context_operations",()):
        return None
    records = getattr(capture,"_witness_records",{})
    for path, selected in selected_leaves(model):
        event = capture._state_events.get(id(selected))
        record = records.get(event.event_id) if event else None
        if record is None or not current_binding(capture,record,selected,purpose):
            continue
        stack=[(record.event.event_id,[])]
        visited=set()
        while stack:
            key, steps=stack.pop()
            if key in visited:
                continue
            visited.add(key)
            current=records.get(key)
            if current is None or not _available(capture,current):
                continue
            matches=(sorted(set(current.direct_members).intersection(excluded))
                     if _direct_input_available(capture,current) else [])
            if matches:
                return dict(member=matches[0],selected_path=path,selected_event=record.event.event_id,
                            source_event=key,steps=steps,
                            binding="current identity, generation, learned references and purpose state",
                            source_contract="reviewed direct fitting input")
            for relation, parent in current.predecessors:
                required=tuple(proof for origin,proof in current.edge_proofs if origin==parent)
                if required and not set(required) <= {e.event_id for e in capture.events}:
                    continue
                step=dict(relation=relation,consumer=key,predecessor=parent)
                if required:
                    step["required_operation_events"]=list(required)
                stack.append((parent,steps+[step]))
    return None


def known_binding_failed(capture, model, purpose):
    records=getattr(capture,"_witness_records",{})
    for _,selected in selected_leaves(model):
        event=capture._state_events.get(id(selected))
        record=records.get(event.event_id) if event else None
        if record is not None and not current_binding(capture,record,selected,purpose):
            return True
    return False

