"""Reviewed stateless text transformation, without invented fitted origins.

A transformation receipt proves the ordered rows of the actual native result.
It does not authorize a model decision. The usual source/state checks still
apply at training and query time. Truthful raw identities and an unchanged
native library remain the cooperative observation boundary.
"""
import inspect
from .identity_evidence import IdentityEvidenceMap

CONTRACT = "sklearn-1.8-hashing-content-word-v1"

def is_hashing(value):
    from sklearn.feature_extraction.text import HashingVectorizer
    return isinstance(value, HashingVectorizer)

def configuration(value):
    import sklearn
    import numpy as np
    from sklearn.feature_extraction.text import HashingVectorizer
    if sklearn.__version__ != "1.8.0" or type(value) is not HashingVectorizer:
        return None
    params = value.get_params(deep=False)
    if set(vars(value)) - (set(params) | {"_sklearn_output_config", "_stop_words_id"}):
        return None
    # Native build_analyzer caches this validation identity; with stop_words=None
    # it carries no fitted vocabulary or document-derived state.
    if hasattr(value, "_stop_words_id") and value._stop_words_id != id(None):
        return None
    if (value.input != "content" or value.encoding != "utf-8"
            or value.decode_error not in {"strict", "ignore", "replace"}
            or value.analyzer != "word" or value.preprocessor is not None
            or value.tokenizer is not None or value.stop_words is not None
            or value.strip_accents not in {None, "ascii", "unicode"}
            or value.dtype is not np.float64
            or value.token_pattern != r"(?u)\b\w\w+\b"
            or type(value.lowercase) is not bool or type(value.binary) is not bool
            or type(value.alternate_sign) is not bool
            or type(value.n_features) is not int or value.n_features <= 0
            or value.norm not in {None, "l1", "l2"}
            or type(value.ngram_range) is not tuple or len(value.ngram_range) != 2
            or not all(type(n) is int for n in value.ngram_range)
            or not 1 <= value.ngram_range[0] <= value.ngram_range[1]):
        return None
    return {k: ("float64" if k == "dtype" else v) for k,v in params.items()}

def begin(capture, value, data, native):
    from .capture import model_state_sha256
    config = configuration(value)
    original = inspect.unwrap(native)
    native_ok = (original.__module__ == "sklearn.feature_extraction.text"
                 and original.__qualname__ == "HashingVectorizer.transform")
    members = capture._known_members(data)
    # Materialized strings only: no generators, file reads or custom iterators.
    import numpy as np
    import pandas as pd
    known_container = type(data) in {list, tuple, np.ndarray, pd.Series}
    content_ok = known_container and all(type(x) is str for x in data)
    before = model_state_sha256(data) if content_ok else None
    return dict(config=config, members=members, input=data, input_fingerprint=before,
                reviewed=bool(config is not None and native_ok and content_ok))

def finish(capture, value, result, pending):
    import numpy as np
    from scipy.sparse import csr_matrix
    from .capture import model_state_sha256
    gaps=[]
    if not pending["reviewed"]:
        gaps.append("STATELESS_TRANSFORM_CONFIGURATION_UNSUPPORTED")
    elif pending["config"] != configuration(value):
        gaps.append("STATELESS_TRANSFORM_CONFIGURATION_CHANGED_DURING_CALL")
    if pending["input_fingerprint"] is None or model_state_sha256(pending["input"]) != pending["input_fingerprint"]:
        gaps.append("STATELESS_TRANSFORM_INPUT_CHANGED_OR_UNVERIFIED")
    members = pending["members"]
    if members is None:
        gaps.append("STATELESS_TRANSFORM_INPUT_MEMBERSHIP_MISSING")
    if (type(result) is not csr_matrix or result.dtype != np.dtype("float64")
            or pending["config"] is None
            or result.shape != (len(members) if members is not None else -1,
                                pending["config"]["n_features"] if pending["config"] else -1)):
        gaps.append("STATELESS_TRANSFORM_OUTPUT_NOT_VERIFIED")
    return dict(stateless_transform_contract=CONTRACT,
                stateless_transform_verified=not gaps,
                stateless_transform_configuration=pending["config"],
                stateless_input_object=f"py:{id(pending['input']):x}",
                stateless_input_fingerprint=pending["input_fingerprint"],
                stateless_input_members=list(members) if members is not None else None,
                stateless_gaps=gaps)

def record_output(capture, result, event):
    from .capture import model_state_sha256
    if not hasattr(capture, "_stateless_outputs"):
        capture._stateless_outputs=IdentityEvidenceMap()
    attrs=event.attributes
    valid=bool(attrs.get("stateless_transform_verified"))
    members=tuple(attrs.get("stateless_input_members") or ()) if valid else None
    proofs=tuple(dict.fromkeys((*attrs.get("stateless_transform_event_ids",()),event.event_id)))
    capture._stateless_outputs[id(result)]=(result, model_state_sha256(result), members, proofs, valid)

def requirements(capture, value):
    """Return actual transform proof IDs, and any failure to use their output."""
    from .capture import model_state_sha256
    item=getattr(capture,"_stateless_outputs",{}).get(id(value))
    if item is None or item[0] is not value:return (),()
    gaps=[]
    stored=capture._row_membership.get(id(value))
    if (not item[4] or item[1] is None or model_state_sha256(value)!=item[1]):
        gaps.append("STATELESS_TRANSFORM_OUTPUT_CHANGED_OR_UNVERIFIED")
    if stored is None or stored[0] is not value or tuple(stored[1])!=item[2]:
        gaps.append("STATELESS_TRANSFORM_OUTPUT_MEMBERSHIP_MISSING_OR_CHANGED")
    events={e.event_id:e for e in capture.events}
    for key in item[3]:
        event=events.get(key)
        if event is None or event.attributes.get("stateless_transform_verified") is not True:
            gaps.append("STATELESS_TRANSFORM_EVIDENCE_NOT_OBSERVED")
    return item[3],tuple(dict.fromkeys(gaps))

def checked_members(capture, value):
    item=getattr(capture,"_stateless_outputs",{}).get(id(value))
    if item is None or item[0] is not value:return False,None
    _,gaps=requirements(capture,value)
    return True,None if gaps else item[2]

def inherit_indexed_output(capture, source, output, members):
    """Use only the existing reviewed native indexing map, never value equality."""
    from .capture import model_state_sha256
    item=getattr(capture,"_stateless_outputs",{}).get(id(source))
    if item is None or item[0] is not source:return
    proofs,gaps=requirements(capture,source)
    capture._stateless_outputs[id(output)]=(output,model_state_sha256(output),
                                            tuple(members) if not gaps else None,
                                            proofs,not gaps)

