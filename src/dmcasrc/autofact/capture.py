"""Low-intrusion framework hooks and generic runtime-return observation.

AutoFact intentionally observes operations and provenance only.  It does not
assign PASS/VIOLATION, map benchmark fault identifiers, or make claim-release
decisions.  Those remain the responsibility of the executable-contract layer
and the existing DM-CASRC audit core.
"""

from __future__ import annotations

import functools
import hashlib
import pickle
import copy
import inspect
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .events import (
    CaptureMode,
    EventType,
    EvidenceSource,
    ObjectReference,
    RuntimeEvent,
    object_reference,
)

_INHERITED_ATTRIBUTE = object()


def _sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError:
        return None
    return digest.hexdigest()


def _path_argument(args: Sequence[Any], kwargs: Mapping[str, Any]) -> Path | None:
    for value in (*args, *kwargs.values()):
        if isinstance(value, Path):
            return value
        if isinstance(value, str):
            try:
                return Path(value)
            except (OSError, ValueError):
                continue
    return None


def _member_ids(value: Any) -> tuple[str, ...] | None:
    """Normalize conventional row/member iterables without importing NumPy."""

    if isinstance(value, (str, bytes, bytearray, Mapping)):
        return None
    try:
        items = tuple(value)
    except TypeError:
        return None
    normalized: list[str] = []
    for item in items:
        if isinstance(item, bool):
            return None
        if isinstance(item, (str, int)):
            normalized.append(f"row:{item}")
        else:
            return None
    if len(normalized) != len(set(normalized)):
        return None
    return tuple(sorted(normalized))


def model_state_sha256(value: Any) -> str | None:
    """Hash trusted in-memory state without loading a serialized artifact.

    This is a run-local mutation check, not an authenticated model signature.
    Unsupported serializers remain unknown rather than falling back to params.
    """
    try:
        # sklearn 1.9 installs this temporary callback context while a parent
        # search fits its refitted child, then removes it on context exit. It
        # affects instrumentation, not prediction state. Exclude only this
        # named field; all parameters and learned fields remain covered.
        if hasattr(value, "__dict__") and "_parent_callback_ctx" in vars(value):
            value = copy.copy(value)
            vars(value).pop("_parent_callback_ctx", None)
        return hashlib.sha256(pickle.dumps(value, protocol=5)).hexdigest()
    except Exception:
        return None


@dataclass(frozen=True)
class InstrumentationStatus:
    framework: str
    enabled: bool
    observers: tuple[str, ...]
    detail: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "framework": self.framework,
            "enabled": self.enabled,
            "observers": list(self.observers),
            "detail": self.detail,
        }


class AutoFactCapture:
    """Collect a finite runtime trace through optional, reversible hooks.

    The observer is deliberately selective rather than global: it instruments
    stable public pandas, joblib, and scikit-learn entry points when installed,
    and provides explicit semantic annotations for facts that a generic Python
    runtime cannot infer safely.  ``start`` never raises merely because an
    optional framework is absent.
    """

    def __init__(
        self,
        *,
        run_id: str = "autofact",
        mode: CaptureMode = CaptureMode.AUTO,
        object_sample_bytes: int = 1024 * 1024,
        track_row_membership: bool = False,
        test_partition_role: str | None = None,
        state_fingerprints: bool = True,
    ) -> None:
        if not isinstance(run_id, str) or not run_id:
            raise ValueError("run_id must be a non-empty string.")
        if not isinstance(mode, CaptureMode):
            raise ValueError("mode must be a CaptureMode.")
        if object_sample_bytes <= 0:
            raise ValueError("object_sample_bytes must be positive.")
        self.run_id = run_id
        self.mode = mode
        self.object_sample_bytes = object_sample_bytes
        if test_partition_role not in {None, "assessment", "validation"}:
            raise ValueError("test_partition_role must be assessment, validation, or None")
        self.track_row_membership = track_row_membership
        self.source_coverage = 'legacy'
        self.test_partition_role = test_partition_role
        self.state_fingerprints = bool(state_fingerprints)
        # Live identities use weak references; expired entries cannot be reused
        # for a later object with the same Python id. Historical facts remain
        # in events. Non-weak-referenceable data retains the strong fallback.
        from .identity_evidence import IdentityEvidenceMap
        self._row_membership = IdentityEvidenceMap()
        self._row_roots = 0
        self._row_source_contracts = None
        self._events: list[RuntimeEvent] = []
        self._patches: list[tuple[Any, str, Any]] = []
        self._statuses: dict[str, InstrumentationStatus] = {}
        self._sequence = 0
        self._started = False
        self._observer_errors: list[str] = []
        self._operation_objects: dict[str, Any] = {}
        self._fit_depth: dict[int, int] = {}
        self._state_events: dict[int, RuntimeEvent] = {}
        self._learned_state_sources: dict[int, tuple[Any, str | None, str]] = {}
        self._data_state_sources = IdentityEvidenceMap()
        self._last_lineage_receipt: dict[str, Any] = {}
        # Version-bound component evidence; never infer provenance from equal values.
        self._component_certificates: dict[int, dict[str, Any]] = {}
        self._receiver_learned_objects: dict[int, dict[str, Any]] = {}

    @property
    def events(self) -> tuple[RuntimeEvent, ...]:
        return tuple(self._events)

    @property
    def observer_errors(self) -> tuple[str, ...]:
        return tuple(self._observer_errors)

    @property
    def instrumentation(self) -> tuple[InstrumentationStatus, ...]:
        return tuple(self._statuses[key] for key in sorted(self._statuses))

    def __enter__(self) -> AutoFactCapture:
        return self.start()

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.stop()

    def _reference(self, value: Any) -> ObjectReference:
        try:
            if not self.state_fingerprints:
                token=f"{type(value).__module__}.{type(value).__qualname__}:{id(value)}:{getattr(value,'shape',None)}"
                return ObjectReference(object_id=f"py:{id(value):x}",
                    type_name=f"{type(value).__module__}.{type(value).__qualname__}",
                    fingerprint=hashlib.sha256(token.encode()).hexdigest(),fingerprint_mode="identity_and_shape_only")
            return object_reference(value, sample_bytes=self.object_sample_bytes)
        except Exception as exc:  # pragma: no cover - defensive observer isolation
            self._observer_errors.append(f"reference:{type(exc).__name__}")
            fallback = hashlib.sha256(
                f"{type(value).__module__}.{type(value).__qualname__}".encode()
            ).hexdigest()
            return ObjectReference(
                object_id=f"py:{id(value):x}",
                type_name=f"{type(value).__module__}.{type(value).__qualname__}",
                fingerprint=fallback,
                fingerprint_mode="observer_fallback",
            )

    def emit(
        self,
        event_type: EventType,
        *,
        operator: str,
        inputs: Sequence[Any] = (),
        outputs: Sequence[Any] = (),
        source: EvidenceSource = EvidenceSource.AUTO_OBSERVED,
        candidate_id: str | None = None,
        fold_role: str | None = None,
        dataset_role: str | None = None,
        temporal_range: tuple[str, str] | None = None,
        parent_event_ids: Sequence[str] = (),
        artifact_path: str | None = None,
        artifact_sha256: str | None = None,
        claim_id: str | None = None,
        attributes: Mapping[str, Any] | None = None,
    ) -> RuntimeEvent:
        """Append one normalized event in logical execution order."""

        if not isinstance(event_type, EventType):
            raise ValueError("event_type must be an EventType.")
        self._sequence += 1
        event = RuntimeEvent(
            event_id=f"{self.run_id}:{self._sequence:06d}:{event_type.value}",
            event_type=event_type,
            logical_time=self._sequence,
            operator=operator,
            inputs=tuple(self._reference(item) for item in inputs),
            outputs=tuple(self._reference(item) for item in outputs),
            source=source,
            candidate_id=candidate_id,
            fold_role=fold_role,
            dataset_role=dataset_role,
            temporal_range=temporal_range,
            parent_event_ids=tuple(parent_event_ids),
            artifact_path=artifact_path,
            artifact_sha256=artifact_sha256,
            claim_id=claim_id,
            attributes=attributes,
        )
        self._events.append(event)
        return event

    def annotate(
        self,
        event_type: EventType,
        *,
        operator: str,
        candidate_id: str | None = None,
        claim_id: str | None = None,
        attributes: Mapping[str, Any] | None = None,
        parent_event_ids: Sequence[str] = (),
        artifact_path: str | None = None,
        artifact_sha256: str | None = None,
    ) -> RuntimeEvent:
        """Record the minimal explicitly declared semantic fact unavailable to hooks."""

        return self.emit(
            event_type,
            operator=operator,
            source=EvidenceSource.DECLARED,
            candidate_id=candidate_id,
            claim_id=claim_id,
            attributes=attributes,
            parent_event_ids=parent_event_ids,
            artifact_path=artifact_path,
            artifact_sha256=artifact_sha256,
        )

    def _emit_from_call(
        self,
        event_type: EventType,
        operator: str,
        args: Sequence[Any],
        kwargs: Mapping[str, Any],
        result: Any,
        *,
        drop_first_arg: bool,
        artifact: bool = False,
        call_metadata: Mapping[str, Any] | None = None,
    ) -> None:
        try:
            values = tuple(args[1:] if drop_first_arg and args else args)
            values += tuple(kwargs.values())
            path = _path_argument(values, {}) if artifact else None
            attributes: dict[str, Any] = {"observer": "framework_hook"}
            attributes.update(call_metadata or {})
            if self.track_row_membership and drop_first_arg and args:
                attributes["operation_object_id"] = f"py:{id(args[0]):x}"
                self._operation_objects[attributes["operation_object_id"]] = args[0]
            if self.track_row_membership and event_type in {EventType.TRANSFORM_FIT, EventType.MODEL_FIT}:
                attributes["contract_scope"] = "assessment_fit_boundary"
                member_ids = self._known_members(values[0]) if values else None
                if member_ids is not None and attributes.get('members_are_training',True):
                    # Positional maps preserve repeated rows for later indexing;
                    # audit membership is a set, including bootstrap/concat repeats.
                    attributes["source_ids"] = list(dict.fromkeys(member_ids))
                    attributes["membership_origin"] = "observed_positional_lineage"
                attributes["fitted_state_sha256"] = model_state_sha256(args[0]) if self.state_fingerprints else None
                attributes["learned_state_sha256"] = model_state_sha256(
                    {k:v for k,v in vars(args[0]).items() if k.endswith('_') and not k.startswith('_')}
                ) if self.state_fingerprints else None
                from .state_certificates import reviewed_purpose_state, purpose_fingerprint, continuation_input_fingerprint
                if reviewed_purpose_state(args[0]):
                    attributes['prediction_state_sha256'] = purpose_fingerprint(args[0], 'prediction') if self.state_fingerprints else None
                    attributes['continuation_input_sha256'] = continuation_input_fingerprint(args[0]) if self.state_fingerprints else None
            if event_type == EventType.ARTIFACT_WRITE and operator == "joblib.dump":
                value = args[0] if args else kwargs.get("value")
                if value is not None:
                    attributes["serialized_object_id"] = f"py:{id(value):x}"
                    attributes["serialized_state_sha256"] = model_state_sha256(value)
                    attributes["serialization_completed"] = True
                    self._operation_objects[attributes["serialized_object_id"]] = value
            if self.track_row_membership and event_type == EventType.TRANSFORM_APPLY and values:
                from .state_lineage import observed_tfidf
                from sklearn.feature_extraction.text import TfidfVectorizer
                members = self._known_members(values[0])
                row_guard = type(args[0]) is not TfidfVectorizer or observed_tfidf(args[0])
                from .stateless_transforms import is_hashing
                if is_hashing(args[0]):
                    row_guard = attributes.get("stateless_transform_verified") is True
                if row_guard and members is not None and getattr(result, "shape", (None,))[0] == len(members):
                    self._row_membership[id(result)] = (result, members)
            event = self.emit(
                event_type,
                operator=operator,
                inputs=values,
                outputs=() if result is None else (result,),
                artifact_path=None if path is None else path.as_posix(),
                artifact_sha256=None if path is None else _sha256_file(path),
                attributes=attributes,
            )
            if self.track_row_membership and event_type == EventType.TRANSFORM_APPLY and result is not None:
                from .stateless_transforms import is_hashing, record_output
                if is_hashing(args[0]):
                    record_output(self, result, event)
            if self.track_row_membership and drop_first_arg and args:
                value=args[0]
                if event_type in {EventType.MODEL_FIT, EventType.TRANSFORM_FIT} and not attributes.get("nested_fit_call"):
                    from sklearn.feature_extraction.text import TfidfVectorizer
                    if type(value) is TfidfVectorizer:
                        getattr(self, '_tfidf_failed', set()).discard(id(value))
                    from .state_identity import LIVE_STATES
                    LIVE_STATES.advance(value)
                    self._state_events[id(value)]=event
                    self._receiver_learned_objects[id(value)] = {
                        key:item for key,item in vars(value).items()
                        if key.endswith('_') and not key.startswith('_') and hasattr(item, 'shape')}
                    # Exact learned array identity plus a before-consumption
                    # fingerprint; no equality-based source guessing.
                    for name,item in vars(value).items() if self.state_fingerprints else ():
                        if name.endswith("_") and hasattr(item,"shape"):
                            self._learned_state_sources[id(item)]=(item,model_state_sha256(item),event.event_id)
                    # ColumnTransformer.fit_transform bypasses transform and
                    # returns a newly stacked matrix. Bind that actual output
                    # to its input rows and the just-completed producer version.
                    from .state_lineage import observed_column_transformer, observed_tabular_transformer, observed_tfidf
                    if operator.endswith('.fit_transform') and (observed_column_transformer(value) or observed_tabular_transformer(value) or observed_tfidf(value)):
                        members = self._known_members(values[0]) if values else None
                        if members is not None and getattr(result, 'shape', (None,))[0] == len(members):
                            self._row_membership[id(result)] = (result, members)
                        self._data_state_sources[id(result)] = (result,
                            model_state_sha256(result) if self.state_fingerprints else None,
                            (event.event_id,), ())
                elif event_type==EventType.TRANSFORM_APPLY and result is not None:
                    states=tuple(attributes.get("input_state_event_ids",()))
                    producer=attributes.get("producer_state_event_id")
                    if producer:states=tuple(dict.fromkeys((*states,producer)))
                    self._data_state_sources[id(result)]=(result,
                        model_state_sha256(result) if self.state_fingerprints else None,states,
                        tuple(attributes.get("state_dependency_gaps",())))
                    if getattr(self,"_witness_enabled",False):
                        from .witnesses import observed_features
                        observed_features(self,value,result,event,values[0] if values else None)
        except Exception as exc:  # pragma: no cover - observation must not affect vendor code
            self._observer_errors.append(f"emit:{operator}:{type(exc).__name__}")

    def _patch_callable(
        self,
        owner: Any,
        name: str,
        *,
        event_type: EventType,
        operator: str,
        drop_first_arg: bool,
        artifact: bool = False,
    ) -> bool:
        try:
            original = getattr(owner, name)
        except AttributeError:
            return False
        if not callable(original):
            return False
        if any(patched_owner is owner and patched_name == name
               for patched_owner, patched_name, _ in self._patches):
            return False
        # Class-level lookup resolves conditional descriptors. Preserve their
        # instance availability checks and the original inheritance structure.
        restore = vars(owner).get(name, _INHERITED_ATTRIBUTE)
        descriptor = inspect.getattr_static(owner, name)

        @functools.wraps(original)
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            metadata: dict[str, Any] = {}
            is_fit = self.track_row_membership and drop_first_arg and args and event_type in {
                EventType.MODEL_FIT, EventType.TRANSFORM_FIT}
            object_id = id(args[0]) if is_fit else None
            if is_fit:
                value = args[0]
                nested = self._fit_depth.get(object_id, 0) > 0
                from .state_lineage import receiver_operation_contract
                semantics,members_are_training=receiver_operation_contract(value,name)
                continuation = semantics == "CONTINUE"
                metadata["fit_semantics"] = semantics
                metadata["members_are_training"] = members_are_training
                metadata["nested_fit_call"] = nested
                owner_id = f"py:{object_id:x}"
                has_history = any(e.attributes.get("operation_object_id") == owner_id
                                  and e.event_type in {EventType.MODEL_FIT, EventType.TRANSFORM_FIT}
                                  for e in self.events)
                learned_state = any(k.endswith("_") and not k.startswith("_") for k in vars(value))
                metadata["preexisting_state_unobserved"] = continuation and learned_state and not has_history and not nested
                previous=self._state_events.get(object_id)
                if previous is not None:metadata["previous_state_event_id"]=previous.event_id
                imports=[];gaps=[]
                from .state_certificates import reviewed_boosting, reviewed_purpose_state, continuation_input_fingerprint, tree_prediction_fingerprint
                boosting = reviewed_boosting(value) and not nested
                purpose_reviewed = reviewed_purpose_state(value) and not nested
                continuation_entry_valid = True
                if purpose_reviewed and continuation and previous is not None:
                    checkpoint = previous.attributes.get('continuation_input_sha256')
                    continuation_entry_valid = checkpoint is not None and continuation_input_fingerprint(value) == checkpoint
                    inherited_checkpoint_gaps = previous.attributes.get('restored_gaps_by_purpose', {}).get('continuation', ())
                    if inherited_checkpoint_gaps:
                        continuation_entry_valid = False
                        gaps.extend(inherited_checkpoint_gaps)
                    if not continuation_entry_valid:
                        gaps.append('CONTINUATION_INPUT_STATE_CHANGED_OR_UNHASHABLE:' + owner_id)
                # A later fit fingerprint cannot certify the state that entered
                # a continuation. Check it before native code overwrites it.
                if continuation and previous is not None and not nested:
                    expected=previous.attributes.get("learned_state_sha256")
                    actual=model_state_sha256({k:v for k,v in vars(value).items() if k.endswith('_') and not k.startswith('_')})
                    if expected is None or actual!=expected:
                        gaps.append("CONTINUATION_INPUT_STATE_CHANGED_OR_UNHASHABLE:"+owner_id)
                    for field, item in self._receiver_learned_objects.get(object_id, {}).items():
                        if getattr(value, field, None) is not item:
                            gaps.append("CONTINUATION_LEARNED_OBJECT_REPLACED:"+owner_id+":"+field)
                from .component_contracts import owned_descendants, ownership_contract
                component_contract=ownership_contract(value) if not nested else None
                before_components={id(child) for child in owned_descendants(value)} if component_contract else set()
                before_prediction = {id(child): tree_prediction_fingerprint(child)
                    for child in owned_descendants(value)} if boosting else {}
                if semantics=='UNESTABLISHED':gaps.append('RECEIVER_OPERATION_NOT_SUPPORTED:'+type(value).__name__)
                # SGD's two public initializers are supported state imports.
                # Binding handles keyword and positional argument forms.
                if type(value).__name__ in {"SGDClassifier","SGDRegressor","SGDOneClassSVM",
                        "PassiveAggressiveClassifier","PassiveAggressiveRegressor","Perceptron"}:
                    bound=inspect.signature(original).bind(*args,**kwargs).arguments
                    for argument in ("coef_init","intercept_init","offset_init"):
                        imported=bound.get(argument)
                        if imported is None:continue
                        source=self._learned_state_sources.get(id(imported))
                        if source is not None and source[0] is imported and source[1] is not None and model_state_sha256(imported)==source[1]:
                            imports.append(source[2])
                            transfer=getattr(self,'_tracked_parameter_copies',{}).get(id(imported))
                            if transfer is not None and transfer[0] is imported:
                                metadata.setdefault('consumed_parameter_transfers',[]).append(
                                    {**transfer[1],'native_argument':argument})
                        else:gaps.append("IMPORTED_STATE_SOURCE_NOT_OBSERVED:"+argument)
                metadata["import_state_event_ids"]=list(dict.fromkeys(imports))
                # Learned selectors may call user code while fitting. Only the
                # declared sklearn score functions have an operation contract.
                score_function=getattr(value,"score_func",None)
                if score_function is not None:
                    from sklearn.feature_selection import f_classif,f_regression,chi2,mutual_info_classif,mutual_info_regression
                    if score_function not in {f_classif,f_regression,chi2,mutual_info_classif,mutual_info_regression}:
                        gaps.append("SELECTOR_SCORE_FUNCTION_NOT_SUPPORTED:"+type(value).__name__)
                data_values=tuple(args[1:])+tuple(kwargs.values())
                primary=args[1] if len(args)>1 else kwargs.get('X',kwargs.get('raw_documents'))
                if primary is not None and self.source_coverage == 'strict':
                    if getattr(self, '_row_source_contracts', None) is not None:
                        self._known_members(primary)  # resolve a reviewed native row operation first
                    kind, unspecified = self.input_source_status(primary)
                    gaps.extend(unspecified)
                    metadata['input_producer_evidence'] = {'kind':kind}
                input_states=[]
                for item in data_values:
                    source=self._data_state_sources.get(id(item))
                    if source is None or source[0] is not item:continue
                    gaps.extend(source[3])
                    if source[1] is None or model_state_sha256(item)!=source[1]:
                        reason = ("INPUT_PRODUCER_OUTPUT_CHANGED_BEFORE_FIT" if self.source_coverage == 'strict'
                                  else "TRANSFORM_OUTPUT_CHANGED_BEFORE_FIT")
                        gaps.append(reason + ":" + f"py:{id(item):x}")
                    else:input_states.extend(source[2])
                from .stateless_transforms import requirements
                proof_ids=[]
                for item in data_values:
                    ids, failures=requirements(self,item)
                    proof_ids.extend(ids);gaps.extend(failures)
                if proof_ids:
                    metadata["stateless_transform_event_ids"]=list(dict.fromkeys(proof_ids))
                metadata["input_state_event_ids"]=list(dict.fromkeys(input_states))
                metadata["state_dependency_gaps"]=gaps
                if 'input_producer_evidence' in metadata:
                    metadata['input_producer_evidence']['producer_versions'] = list(dict.fromkeys(input_states))
                    metadata['input_producer_evidence']['unresolved'] = [g for g in gaps if g.startswith(
                        ('INPUT_PRODUCER_', 'EXTERNAL_FEATURE_', 'TRANSFORM_PRODUCER_'))]
                self._fit_depth[object_id] = self._fit_depth.get(object_id, 0) + 1
            elif self.track_row_membership and drop_first_arg and args and event_type==EventType.TRANSFORM_APPLY:
                from .stateless_transforms import is_hashing, begin
                stateless_pending = None
                gaps=[]
                if is_hashing(args[0]):
                    data=args[1] if len(args)>1 else kwargs.get("X")
                    stateless_pending=begin(self,args[0],data,original)
                else:
                    previous=self._state_events.get(id(args[0]))
                    from sklearn.feature_extraction.text import TfidfVectorizer
                    from .state_lineage import observed_tfidf
                    if type(args[0]) is TfidfVectorizer and not observed_tfidf(args[0]):
                        gaps.append('TFIDF_OPERATION_CONTRACT_NOT_ESTABLISHED')
                    if id(args[0]) in getattr(self, '_tfidf_failed', ()):
                        gaps.append('TFIDF_PREVIOUS_FIT_FAILED')
                    if previous:
                        metadata["producer_state_event_id"]=previous.event_id
                        expected=previous.attributes.get("fitted_state_sha256")
                        if expected is None or not self.state_fingerprints or model_state_sha256(args[0])!=expected:
                            gaps.append("TRANSFORM_PRODUCER_STATE_CHANGED_OR_UNHASHABLE:"+f"py:{id(args[0]):x}")
                    else:gaps.append("TRANSFORM_PRODUCER_HISTORY_NOT_OBSERVED:"+f"py:{id(args[0]):x}")
                metadata["input_state_event_ids"]=list(dict.fromkeys(
                    state for item in tuple(args[1:])+tuple(kwargs.values())
                    for source in [self._data_state_sources.get(id(item))]
                    if source is not None and source[0] is item for state in source[2]))
                for item in tuple(args[1:])+tuple(kwargs.values()):
                    source=self._data_state_sources.get(id(item))
                    if source is not None and source[0] is item:gaps.extend(source[3])
                primary=args[1] if len(args)>1 else kwargs.get('X',kwargs.get('raw_documents'))
                if primary is not None and self.source_coverage == 'strict':
                    _, unspecified = self.input_source_status(primary)
                    gaps.extend(unspecified)
                    for item in tuple(args[1:])+tuple(kwargs.values()):
                        source=self._data_state_sources.get(id(item))
                        if source is not None and source[0] is item and (source[1] is None or source[1]!=model_state_sha256(item)):
                            gaps.append('INPUT_PRODUCER_OUTPUT_CHANGED_BEFORE_TRANSFORM')
                metadata["state_dependency_gaps"]=list(dict.fromkeys(gaps))
            if not (self.track_row_membership and drop_first_arg and args and event_type==EventType.TRANSFORM_APPLY):
                stateless_pending = None
            witness_pending = None
            if is_fit and not nested and getattr(self, "_witness_enabled", False):
                from . import witnesses
                try:
                    arguments = inspect.signature(original).bind(*args, **kwargs).arguments
                    witness_pending = witnesses.prepare(self, args[0], name, arguments)
                except Exception as exc:
                    witnesses.mark_failed(self,args[0])
                    self._observer_errors.append("witness_prepare:" + type(exc).__name__)
            try:
                result = original(*args, **kwargs)
            except BaseException:
                if is_fit:
                    from sklearn.feature_extraction.text import TfidfVectorizer
                    if type(args[0]) is TfidfVectorizer:
                        self.__dict__.setdefault('_tfidf_failed', set()).add(id(args[0]))
                if is_fit and not nested and getattr(self, "_witness_enabled", False):
                    witnesses.mark_failed(self, args[0])
                raise
            finally:
                if is_fit:
                    self._fit_depth[object_id] -= 1
            if stateless_pending is not None:
                from .stateless_transforms import finish
                stateless_result=finish(self,args[0],result,stateless_pending)
                metadata.update(stateless_result)
                metadata["state_dependency_gaps"]=list(dict.fromkeys(
                    (*metadata.get("state_dependency_gaps",()),*stateless_result["stateless_gaps"])))
            if is_fit and not metadata.get("nested_fit_call"):
                from .state_lineage import predictive_components
                metadata["component_state_event_ids"]=[self._state_events[id(child)].event_id
                    for _,child in predictive_components(args[0]) if id(child) in self._state_events]
            self._emit_from_call(
                event_type,
                operator,
                args,
                kwargs,
                result,
                drop_first_arg=drop_first_arg,
                artifact=artifact,
                call_metadata=metadata,
            )
            if is_fit and not metadata.get("nested_fit_call") and component_contract:
                event=self._state_events.get(object_id)
                members=self._known_members(args[1]) if len(args)>1 else self._known_members(kwargs.get('X'))
                if event is not None and members is not None:
                    for child in owned_descendants(args[0]):
                        if id(child) not in before_components:
                            self._component_certificates[id(child)]={
                                'object':child,'owner':args[0],'event_id':event.event_id,
                                'creation_owner_id':event.attributes.get('operation_object_id'),
                                'upper_bound':tuple(members),'fingerprint':model_state_sha256(child),
                                'contract':component_contract}
                    if boosting and continuation_entry_valid:
                        from .state_certificates import CERTIFICATE_VERSION
                        for child in owned_descendants(args[0]):
                            certificate = self._component_certificates.get(id(child))
                            predictive = tree_prediction_fingerprint(child)
                            if (certificate is None or certificate['owner'] is not args[0]
                                    or predictive is None):
                                continue
                            if id(child) in before_components and before_prediction.get(id(child)) != predictive:
                                continue
                            certificate.update(purpose_contract=CERTIFICATE_VERSION,
                                prediction_fingerprint=predictive,
                                continuation_fingerprint=model_state_sha256(child),
                                latest_owner_event_id=event.event_id)
                            child_event=self._state_events.get(id(child))
                            if 'source_state_event_ids' not in certificate and child_event is not None:
                                certificate['source_state_event_ids']=(child_event.event_id,)
            if witness_pending is not None:
                try:
                    witnesses.issue(self, args[0], witness_pending, result, operator)
                except Exception as exc:
                    witnesses.mark_failed(self, args[0])
                    self._observer_errors.append("witness_issue:" + type(exc).__name__)
            return result

        replacement = wrapped
        if isinstance(owner, type) and type(descriptor).__module__ == 'sklearn.utils._available_if':
            from sklearn.utils._available_if import _AvailableIfDescriptor
            from sklearn.utils.metaestimators import available_if
            if isinstance(descriptor, _AvailableIfDescriptor):
                replacement = available_if(descriptor.check)(wrapped)
        setattr(owner, name, replacement)
        self._patches.append((owner, name, restore))
        return True

    def _register(self, framework: str, observers: list[str], detail: str | None = None) -> None:
        self._statuses[framework] = InstrumentationStatus(
            framework=framework,
            enabled=bool(observers),
            observers=tuple(observers),
            detail=detail,
        )

    def declare_input_source(self, value, kind, bound=(), basis=None):
        current = self._data_state_sources.get(id(value))
        digest = model_state_sha256(value)
        if current is not None and current[0] is value:
            if len(current) == 5 and current[4] == (kind, bound, basis) and current[1] == digest:
                return
            raise ValueError('Existing input production evidence cannot be replaced')
        if digest is None:
            raise ValueError('The declared input cannot be fingerprinted')
        states = ()
        gaps = ()
        if kind == 'unknown_derived':
            gaps = ('EXTERNAL_FEATURE_PRODUCER_UNVERIFIED',)
        if kind == 'trusted_external_bound':
            event = self.emit(EventType.TRANSFORM_FIT, operator='audit.external_producer_bound',
                source=EvidenceSource.DECLARED, attributes={'fit_semantics':'RESET',
                    'members_are_training':False, 'source_ids':[],
                    'external_source_upper_bound':list(bound), 'external_source_basis':basis})
            states = (event.event_id,)
        self._data_state_sources[id(value)] = (value, digest, states, gaps, (kind, bound, basis))

    def input_source_status(self, value):
        entry = self._data_state_sources.get(id(value)) if value is not None else None
        if entry is None or entry[0] is not value:
            return 'unspecified', ('INPUT_PRODUCER_COVERAGE_UNSPECIFIED',) if self.source_coverage == 'strict' else ()
        if len(entry) == 5:
            return entry[4][0], ()
        return 'observed', ()

    def _known_members(self, value: Any) -> tuple[str, ...] | None:
        if getattr(self,"_stateless_outputs",None):
            from .stateless_transforms import checked_members
            found, members=checked_members(self,value)
            if found:return members
        if self._row_source_contracts is not None:
            return self._row_source_contracts.members(value)
        stored = self._row_membership.get(id(value))
        return stored[1] if stored is not None and stored[0] is value else None

    def trace_for_model(self, model: Any, *, purpose='strict') -> tuple[RuntimeEvent, ...]:
        from .state_certificates import PURPOSES
        if purpose not in PURPOSES:raise ValueError('Unknown state validity purpose')
        return self._trace_for_model_versions(model, purpose=purpose)

    def observation_boundary_for(self, model: Any) -> dict[str, Any] | None:
        """Open hooks do not certify completeness of state entry paths."""
        return None

    @property
    def lineage_receipt(self) -> dict[str, Any]:
        """The last selected graph and excluded assessment-history advisory."""
        return copy.deepcopy(self._last_lineage_receipt)

    def _trace_for_model_versions(self, model: Any, *, purpose='strict') -> tuple[RuntimeEvent, ...]:
        from .state_lineage import StateVersionGraph, predictive_components, FITS, observed_pipeline_container, observed_calibration_container, observed_identity_transformer
        from dataclasses import replace
        graph_events=tuple(replace(event,attributes={**event.attributes,
            'state_dependency_gaps':event.attributes['restored_gaps_by_purpose'][purpose]})
            if purpose in event.attributes.get('restored_gaps_by_purpose',{}) else event
            for event in self.events)
        graph=StateVersionGraph(graph_events)
        roots=[];gaps=set();live={};visited=set();component_evidence=[]
        assessment={str(member) for e in self.events if e.event_type==EventType.DATA_SPLIT
                    and (e.fold_role=="assessment" or e.attributes.get("split_role")=="assessment")
                    for member in e.attributes.get("member_ids",())}
        def visit(value,path,owners=()):
            if id(value) in visited:return
            visited.add(id(value));owner=f"py:{id(value):x}";live[owner]=value
            event=self._state_events.get(id(value))
            if id(value) in getattr(self, '_tfidf_failed', ()):
                gaps.add('TFIDF_PREVIOUS_FIT_FAILED:'+path)
            if observed_pipeline_container(value) or observed_calibration_container(value) or observed_identity_transformer(value):
                pass  # learned-state roots are the declared fitted children
            elif event is not None:roots.append(event.event_id)
            elif self._valid_component_certificate(value,owners,assessment,purpose=purpose):
                certificate=self._component_certificates[id(value)]
                component_evidence.append({'path':path,'owner_object_id':f"py:{id(certificate['owner']):x}",
                    'creation_event_id':certificate['event_id'],'contract':certificate['contract'],
                    'member_upper_bound':list(certificate['upper_bound']),
                    'rule':'OWNED_COMPONENT_UPPER_BOUND_DISJOINT'})
            elif self._valid_component_certificate(value,owners,(),purpose=purpose):
                # The owner/content association is valid even when its source
                # upper bound intersects this policy. An upper bound alone
                # remains insufficient to establish a violating member.
                certificate=self._component_certificates[id(value)]
                component_evidence.append({'path':path,'owner_object_id':f"py:{id(certificate['owner']):x}",
                    'creation_event_id':certificate['event_id'],'contract':certificate['contract'],
                    'member_upper_bound':list(certificate['upper_bound']),
                    'rule':'OWNED_COMPONENT_STATE_VALID_SOURCE_BOUND_ONLY'})
            elif value is model:gaps.add("SELECTED_MODEL_FIT_NOT_OBSERVED:"+path)
            else:gaps.add("FITTED_COMPONENT_NOT_OBSERVED:"+path)
            for name,child in predictive_components(value):visit(child,path+"."+name,owners+(value,))
        visit(model,"selected")
        keep,derived_gaps=graph.history(roots);gaps.update(derived_gaps)
        # Historical imported/consumed versions are snapshots. Only components
        # still used by the selected output require current-state consistency.
        for owner,value in live.items():
            # Structural Pipeline state is its currently selected children.
            # Recomposition must not select an obsolete whole-parent fit.
            if observed_pipeline_container(value) or observed_calibration_container(value) or observed_identity_transformer(value):continue
            event=self._state_events.get(id(value))
            if event is None:continue
            # Some parents finalize child prediction values after child.fit
            # returns (boosting leaf updates). The creation certificate binds
            # the final state after the parent operation, including that phase.
            if self._valid_component_state_certificate(value,tuple(live.values()),purpose):continue
            if self._valid_component_certificate(value,tuple(live.values()),assessment,purpose=purpose):continue
            from .state_certificates import purpose_fingerprint, reviewed_purpose_state
            expected=event.attributes.get('prediction_state_sha256' if purpose=='prediction' and reviewed_purpose_state(value) else 'fitted_state_sha256')
            if expected is None or purpose_fingerprint(value,purpose)!=expected:
                gaps.add("FITTED_STATE_CHANGED_OR_UNHASHABLE:"+owner)
        assessment={str(member) for e in self.events if e.event_type==EventType.DATA_SPLIT
                    and (e.fold_role=="assessment" or e.attributes.get("split_role")=="assessment")
                    for member in e.attributes.get("member_ids",())}
        advisories=[]
        for owner,value in live.items():
            reset=self._state_events.get(id(value))
            if reset is None or reset.attributes.get("fit_semantics")!="RESET":continue
            for event in self.events:
                overlap=assessment.intersection(event.attributes.get("source_ids",()))
                if event.event_type in FITS and event.event_id not in keep and overlap and event.attributes.get("operation_object_id")==owner and event.logical_time<reset.logical_time:
                    advisories.append({"excluded_event_id":event.event_id,"object_id":owner,
                        "assessment_members":sorted(overlap),"rule":"VERIFIED_RECEIVER_RESET",
                        "overriding_reset_event_id":reset.event_id})
        self._last_lineage_receipt={"selected_object_id":f"py:{id(model):x}",
            'state_validity_purpose':purpose,
            "root_state_versions":roots,"retained_fit_event_ids":sorted(keep),"gaps":sorted(gaps),
            "pipeline_container_contract":observed_pipeline_container(model),
            "component_evidence":component_evidence,
            "excluded_assessment_history":advisories,
            "version_predecessors":{key:sorted(graph.nodes[key].predecessors) for key in sorted(keep)}}
        selected=tuple(e for e in self.events if e.event_type not in FITS or e.event_id in keep)
        # A newly constructed child may be fitted through a private native
        # array copy without row identities. A valid owned bound can discharge
        # that missing membership, only within its parent's selected ancestry
        # and only when disjoint from assessment. Keep all other dependencies.
        from dataclasses import replace
        bound_events={}
        for owner,value in live.items():
            certificate=self._component_certificates.get(id(value))
            if certificate is None:continue
            if not self._valid_component_certificate(value,tuple(live.values()),assessment,purpose=purpose):continue
            child_event=self._state_events.get(id(value))
            candidate_ids=list(certificate.get('source_state_event_ids',())) if purpose!='strict' else []
            if child_event is not None:candidate_ids.append(child_event.event_id)
            for candidate_id in dict.fromkeys(candidate_ids):
                candidate=graph.nodes.get(candidate_id)
                if candidate is None or 'source_ids' in candidate.event.attributes:continue
                bound_events[candidate_id]=certificate
                component_evidence.append({'path':owner,'creation_event_id':certificate['event_id'],
                    'contract':certificate['contract'],'member_upper_bound':list(certificate['upper_bound']),
                    'rule':'OWNED_COMPONENT_MISSING_MEMBERSHIP_DISCHARGED',
                    'child_event_id':candidate_id})
        selected=tuple(replace(event,attributes={**event.attributes,
            'source_ids':list(bound_events[event.event_id]['upper_bound']),
            'membership_origin':'certified_owned_component_upper_bound',
            'bound_creation_event_id':bound_events[event.event_id]['event_id']})
            if event.event_id in bound_events else event for event in selected)
        if gaps:
            selected+=(RuntimeEvent(event_id=f"{self.run_id}:selected-version-gap",logical_time=self._sequence+1,
                event_type=EventType.MODEL_FIT,operator="dmcasrc.selected_lineage_requirement",source=EvidenceSource.DERIVED,
                attributes={"contract_scope":"assessment_fit_boundary","lineage_gaps":sorted(gaps)}),)
        return selected

    def _valid_component_state_certificate(self,value,owners,purpose):
        """Association is separate from whether a source bound is disjoint."""
        from .state_certificates import CERTIFICATE_VERSION, tree_prediction_fingerprint, reviewed_boosting
        from .component_contracts import ownership_contract
        from .component_registry import check_certificate
        if purpose=='strict':return False
        cert=self._component_certificates.get(id(value))
        if cert is None or cert.get('purpose_contract')!=CERTIFICATE_VERSION:return False
        owner=cert['owner']
        if not reviewed_boosting(owner):return False
        latest=self._state_events.get(id(owner))
        if latest is None or latest.event_id!=cert.get('latest_owner_event_id'):return False
        child_event=self._state_events.get(id(value))
        if child_event is None or child_event.event_id not in cert.get('source_state_event_ids',()):return False
        temporary=dict(cert)
        temporary['fingerprint']=cert.get('prediction_fingerprint' if purpose=='prediction' else 'continuation_fingerprint')
        fingerprint=tree_prediction_fingerprint if purpose=='prediction' else model_state_sha256
        return check_certificate(value,owners,(),temporary,contract_for=ownership_contract,
            fingerprint=fingerprint,events={e.event_id:e for e in self.events}).valid

    def _valid_component_certificate(self,value,owners,assessment,*,purpose='strict'):
        if self._valid_component_state_certificate(value,owners,purpose):
            return not set(self._component_certificates[id(value)]['upper_bound']).intersection(assessment)
        cert=self._component_certificates.get(id(value))
        if purpose!='strict' and cert is not None and cert.get('purpose_contract'):
            return False
        from .component_contracts import ownership_contract
        from .component_registry import check_certificate
        result=check_certificate(value,owners,assessment,self._component_certificates.get(id(value)),
            contract_for=ownership_contract,fingerprint=model_state_sha256,
            events={event.event_id:event for event in self.events})
        return result.valid

    def _trace_for_model_legacy(self, model: Any) -> tuple[RuntimeEvent, ...]:
        """Scope learned fits to an explicitly selected fitted output.

        Supported sklearn composition attributes and captured transform producers
        supply dependency edges. Continuation retains preceding state-producing
        fits through the last observed reset. Missing or changed state becomes an
        explicit unknown fit; it cannot silently disappear from the scope.
        """
        objects: set[str] = set()
        selected_objects: dict[str, Any] = {}
        visited: set[int] = set()
        def visit(value):
            if id(value) in visited:
                return
            visited.add(id(value))
            if isinstance(value, (tuple, list)):
                for item in value:visit(item)
            elif isinstance(value, dict):
                for item in value.values():visit(item)
            elif hasattr(value, "fit") or hasattr(value, "estimator"):
                key = f"py:{id(value):x}"
                objects.add(key)
                selected_objects[key] = value
                for name in ("steps", "estimator", "base_estimator", "calibrated_classifiers_",
                             "best_estimator_", "estimators_", "final_estimator_", "estimator_"):
                    if name in {"estimator", "base_estimator"} and any(
                        getattr(value, fitted, None) is not None for fitted in
                        ("best_estimator_", "estimators_", "calibrated_classifiers_", "estimator_")):
                        continue  # unfitted cloning template is not a producer
                    child = getattr(value, name, None)
                    if child is not None:visit(child)
        visit(model)
        fits = [e for e in self.events if e.event_type in {EventType.MODEL_FIT, EventType.TRANSFORM_FIT}]
        root = f"py:{id(model):x}"
        root_fits = [e for e in fits if e.attributes.get("operation_object_id") == root]
        gaps: set[str] = set()
        def gap(reason: str):
            gaps.add(reason)
        if not root_fits:
            gap("SELECTED_MODEL_FIT_NOT_OBSERVED")
            boundary = self._sequence + 1
        else:
            boundary = max(e.logical_time for e in root_fits)
        chosen: dict[str, RuntimeEvent] = {}
        def retain_history(owner: str, before: int):
            history = [e for e in fits if e.attributes.get("operation_object_id") == owner
                       and e.logical_time <= before and not e.attributes.get("nested_fit_call", False)]
            if not history:
                gap("PRODUCER_FIT_NOT_OBSERVED:" + owner)
                return
            for e in reversed(history):
                chosen[e.event_id] = e
                if e.attributes.get("preexisting_state_unobserved"):
                    gap("CONTINUATION_HISTORY_NOT_OBSERVED:" + owner)
                if e.attributes.get("fit_semantics", "RESET") == "RESET":
                    break
        for owner in objects:
            if any(e.attributes.get("operation_object_id") == owner for e in fits):
                retain_history(owner, boundary if owner == root else self._sequence)
            elif any(k.endswith("_") and not k.startswith("_") for k in vars(selected_objects[owner])):
                gap("FITTED_COMPONENT_NOT_OBSERVED:" + owner)
        pending = [(ref.object_id, e.logical_time) for e in chosen.values() for ref in e.inputs]
        seen = set()
        while pending:
            object_id, before = pending.pop()
            if (object_id, before) in seen:continue
            seen.add((object_id, before))
            producers = [e for e in self.events if e.event_type == EventType.TRANSFORM_APPLY
                         and e.logical_time < before and any(ref.object_id == object_id for ref in e.outputs)]
            if not producers:continue
            producer = max(producers, key=lambda e:e.logical_time)
            owner = producer.attributes.get("operation_object_id")
            prior = [e for e in fits if e.attributes.get("operation_object_id") == owner
                     and e.logical_time < producer.logical_time]
            if prior:
                before_ids = set(chosen)
                retain_history(owner, producer.logical_time - 1)
                for event_id in set(chosen) - before_ids:
                    event = chosen[event_id]
                    pending.extend((ref.object_id, event.logical_time) for ref in event.inputs)
            else:
                gap("TRANSFORM_PRODUCER_HISTORY_NOT_OBSERVED:" + str(owner))
            consumer_refs = [ref for e in chosen.values() if e.logical_time == before for ref in e.inputs
                             if ref.object_id == object_id]
            producer_refs = [ref for ref in producer.outputs if ref.object_id == object_id]
            if consumer_refs and producer_refs and consumer_refs[0].fingerprint != producer_refs[0].fingerprint:
                gap("TRANSFORM_OUTPUT_CHANGED_BEFORE_FIT:" + object_id)
            pending.extend((ref.object_id, producer.logical_time) for ref in producer.inputs)
        # A fitted object can be modified directly or shared and refitted after
        # an ancestor was consumed. Compare exact supported in-memory state.
        for owner in {e.attributes.get("operation_object_id") for e in chosen.values()}:
            current = self._operation_objects.get(owner)
            selected_history = [e for e in chosen.values() if e.attributes.get("operation_object_id") == owner]
            latest = max(selected_history, key=lambda e:e.logical_time)
            expected = latest.attributes.get("fitted_state_sha256")
            if current is not None and (expected is None or model_state_sha256(current) != expected):
                gap("FITTED_STATE_CHANGED_OR_UNHASHABLE:" + str(owner))
        keep = {e.event_id for e in chosen.values()}
        selected = tuple(e for e in self.events if e.event_type not in {EventType.MODEL_FIT,EventType.TRANSFORM_FIT}
                         or e.event_id in keep)
        if gaps:
            unknown = RuntimeEvent(event_id=f"{self.run_id}:selected-lineage-unknown:{root}",
                                   logical_time=self._sequence+1, event_type=EventType.MODEL_FIT,
                                   operator="dmcasrc.selected_lineage_requirement", source=EvidenceSource.DERIVED,
                                   attributes={"contract_scope":"assessment_fit_boundary",
                                               "operation_object_id":root,"lineage_gaps":sorted(gaps)})
            selected += (unknown,)
        return selected

    def _instrument_row_split(self, observers: list[str]) -> None:
        import numpy as np
        import sklearn.model_selection as selection

        original = selection.train_test_split

        @functools.wraps(original)
        def split(*arrays: Any, **kwargs: Any) -> Any:
            if not arrays:
                return original(*arrays, **kwargs)
            # Append positions to the SAME split call: the original RNG is
            # consumed once and all original arrays receive identical indices.
            shape = getattr(arrays[0], "shape", None)
            count = int(shape[0]) if shape is not None else len(arrays[0])
            result = original(*arrays, np.arange(count), **kwargs)
            train_positions, test_positions = result[-2:]
            known = self._known_members(arrays[0])
            if known is None:
                self._row_roots += 1
                known = tuple(f"{self.run_id}:dataset:{self._row_roots}:row:{i}" for i in range(count))
            for value in arrays:
                self._row_membership[id(value)] = (value, known)
            for offset, positions in enumerate((train_positions, test_positions)):
                members = tuple(known[int(i)] for i in positions)
                for index in range(len(arrays)):
                    value = result[2 * index + offset]
                    self._row_membership[id(value)] = (value, members)
                    if self.source_coverage == 'strict':
                        self.inherit_input_source(arrays[index], value)
                        from .stateless_transforms import inherit_indexed_output
                        inherit_indexed_output(self, arrays[index], value, members)
                role = "fit" if offset == 0 else (self.test_partition_role or "test")
                self.emit(
                    EventType.DATA_SPLIT,
                    operator="sklearn.model_selection.train_test_split",
                    inputs=arrays,
                    outputs=(result[offset],),
                    fold_role=role,
                    attributes={"split_role": role, "member_ids": list(members),
                                "membership_origin": "observed_positional_lineage",
                                "role_origin": "declared_api_policy" if offset else "api_training_partition"},
                )
            return result[:-2]

        selection.train_test_split = split
        self._patches.append((selection, "train_test_split", original))
        observers.append("sklearn.model_selection.train_test_split.membership")

    def inherit_input_source(self, value, result):
        source = self._data_state_sources.get(id(value))
        if source is None or source[0] is not value:
            return
        gaps = list(source[3])
        states = source[2]
        if source[1] is None or model_state_sha256(value) != source[1]:
            gaps.append('INDEXED_SOURCE_CHANGED_OR_UNHASHABLE:' + f'py:{id(value):x}')
            states = ()
        self._data_state_sources[id(result)] = (result,
            model_state_sha256(result) if self.state_fingerprints else None,
            states, tuple(dict.fromkeys(gaps)))

    def _instrument_safe_indexing(self, observers: list[str]) -> None:
        """Propagate exact native indexing, including sklearn's loaded aliases.

        This private helper is deliberately version-bound. Unknown copies and
        custom operations are not guessed from feature values.
        """
        import sys
        import numpy as np
        import sklearn.utils as utils
        original = utils._safe_indexing

        @functools.wraps(original)
        def index(value, indices, *, axis=0):
            result = original(value, indices, axis=axis)
            try:
                members = self._known_members(value)
                if members is not None:
                    selected = members if axis == 1 else tuple(
                        members[int(i)] for i in original(np.arange(len(members)), indices, axis=0))
                    if getattr(result, "shape", (len(selected),))[0] == len(selected):
                        self._row_membership[id(result)] = (result, selected)
                        from .stateless_transforms import inherit_indexed_output
                        inherit_indexed_output(self,value,result,selected)
                self.inherit_input_source(value, result)
            except Exception as exc:
                self._observer_errors.append(f"membership:index:{type(exc).__name__}")
            return result

        for module_name, module in tuple(sys.modules.items()):
            if module is None or not (module_name == "sklearn" or module_name.startswith("sklearn.")):
                continue
            for name, value in tuple(vars(module).items()):
                if value is original:
                    setattr(module, name, index)
                    self._patches.append((module, name, original))
        observers.append("sklearn.utils._safe_indexing.membership")

    def _instrument_imblearn(self) -> None:
        observers: list[str] = []
        try:
            from imblearn.under_sampling import NearMiss
        except ImportError:
            self._register("imblearn", observers, "optional dependency unavailable")
            return
        if self.track_row_membership:
            from imblearn.pipeline import Pipeline as ImbalancedPipeline
            self._patch_callable(ImbalancedPipeline, "fit", event_type=EventType.MODEL_FIT,
                                 operator="imblearn.pipeline.Pipeline.fit", drop_first_arg=True)
            observers.append("imblearn.pipeline.Pipeline.fit")
            original = NearMiss.fit_resample

            @functools.wraps(original)
            def resample(estimator: Any, X: Any, y: Any, **kwargs: Any) -> Any:
                result = original(estimator, X, y, **kwargs)
                members = self._known_members(X)
                indices = getattr(estimator, "sample_indices_", None)
                if self.source_coverage == 'strict':
                    self.inherit_input_source(X, result[0])
                if members is not None and indices is not None:
                    selected = tuple(members[int(i)] for i in indices)
                    if len(selected) == len(result[0]):
                        for value in result:
                            self._row_membership[id(value)] = (value, selected)
                        self.emit(EventType.DATA_SPLIT, operator="imblearn.NearMiss.fit_resample",
                                  inputs=(X, y), outputs=result,
                                  attributes={"split_role": "resampled_fit", "member_ids": list(selected),
                                              "membership_origin": "native_sample_indices_"})
                return result

            NearMiss.fit_resample = resample
            self._patches.append((NearMiss, "fit_resample", original))
            observers.append("imblearn.NearMiss.fit_resample.membership")
        self._register("imblearn", observers)

    def _instrument_sklearn(self) -> None:
        observers: list[str] = []
        try:
            from sklearn.calibration import CalibratedClassifierCV
            from sklearn.decomposition import PCA
            from sklearn.linear_model import LogisticRegression
            from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
            from sklearn.pipeline import Pipeline
            from sklearn.compose import ColumnTransformer
            from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
            from sklearn.feature_selection import SelectKBest,SelectPercentile,VarianceThreshold,SelectFromModel
        except ImportError:
            self._register("sklearn", observers, "optional dependency unavailable")
            return

        from sklearn.feature_extraction.text import TfidfVectorizer
        transforms = (StandardScaler, MinMaxScaler, RobustScaler, PCA,
                      SelectKBest,SelectPercentile,VarianceThreshold,SelectFromModel)
        if self.track_row_membership:
            transforms += (TfidfVectorizer,)
            self._patch_callable(TfidfVectorizer, 'fit_transform',
                event_type=EventType.TRANSFORM_FIT,
                operator='sklearn.feature_extraction.text.TfidfVectorizer.fit_transform', drop_first_arg=True)
            self._instrument_row_split(observers)
        models = (Pipeline, GridSearchCV, RandomizedSearchCV, LogisticRegression, CalibratedClassifierCV)
        if self.track_row_membership:
            from sklearn.naive_bayes import GaussianNB
            from sklearn.linear_model import LinearRegression, LogisticRegressionCV, SGDClassifier, SGDRegressor
            from sklearn.ensemble import RandomForestClassifier, StackingClassifier, VotingClassifier
            from sklearn.neural_network import MLPClassifier
            from sklearn.neighbors import KNeighborsClassifier
            from sklearn.svm import SVC
            from sklearn.svm import LinearSVC
            from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
            from sklearn.multiclass import OneVsRestClassifier
            models += (GaussianNB, LinearRegression, LogisticRegressionCV, KNeighborsClassifier,
                       SVC, OneVsRestClassifier, SGDClassifier, SGDRegressor,
                       RandomForestClassifier, StackingClassifier, VotingClassifier, MLPClassifier,
                       LinearSVC,QuadraticDiscriminantAnalysis)
            from sklearn.dummy import DummyClassifier,DummyRegressor
            from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
            from sklearn.preprocessing import KBinsDiscretizer
            models += (DummyClassifier,DummyRegressor,DecisionTreeClassifier,DecisionTreeRegressor)
            transforms += (KBinsDiscretizer,)
            self._instrument_safe_indexing(observers)
            # Enumerate the same public API candidates as the retained census.
            # Observation does not confer a receiver-operation contract.
            from sklearn.utils import all_estimators
            extra_models=[]; extra_transforms=[]
            for _, cls in all_estimators():
                try:
                    parameters=inspect.signature(cls).parameters
                    candidate=callable(getattr(cls,'partial_fit',None)) or 'warm_start' in parameters
                except (TypeError,ValueError):
                    continue
                if candidate:
                    (extra_transforms if callable(getattr(cls,'transform',None)) else extra_models).append(cls)
            transforms=tuple(dict.fromkeys((*transforms,*extra_transforms)))
            models=tuple(dict.fromkeys((*models,*extra_models)))
        if self.track_row_membership:
            from sklearn.impute import SimpleImputer
            from sklearn.preprocessing import OneHotEncoder
            transforms += (SimpleImputer, OneHotEncoder)
            for tabular in (SimpleImputer, OneHotEncoder):
                self._patch_callable(tabular, 'fit_transform',
                    event_type=EventType.TRANSFORM_FIT,
                    operator=tabular.__module__+'.'+tabular.__qualname__+'.fit_transform', drop_first_arg=True)
            transforms += (ColumnTransformer,)
            self._patch_callable(ColumnTransformer, 'fit_transform',
                event_type=EventType.TRANSFORM_FIT,
                operator='sklearn.compose.ColumnTransformer.fit_transform', drop_first_arg=True)
        for cls in transforms:
            label = f"{cls.__module__}.{cls.__qualname__}"
            if self._patch_callable(
                cls,
                "fit",
                event_type=EventType.TRANSFORM_FIT,
                operator=f"{label}.fit",
                drop_first_arg=True,
            ):
                observers.append(f"{label}.fit")
            if self.track_row_membership and self._patch_callable(
                cls,"partial_fit",event_type=EventType.TRANSFORM_FIT,
                operator=f"{label}.partial_fit",drop_first_arg=True):
                observers.append(f"{label}.partial_fit")
            if self._patch_callable(
                cls,
                "transform",
                event_type=EventType.TRANSFORM_APPLY,
                operator=f"{label}.transform",
                drop_first_arg=True,
            ):
                observers.append(f"{label}.transform")
        for cls in models:
            label = f"{cls.__module__}.{cls.__qualname__}"
            if self._patch_callable(
                cls,
                "fit",
                event_type=EventType.MODEL_FIT,
                operator=f"{label}.fit",
                drop_first_arg=True,
            ):
                observers.append(f"{label}.fit")
            if self.track_row_membership and self._patch_callable(
                cls,"partial_fit",event_type=EventType.MODEL_FIT,
                operator=f"{label}.partial_fit",drop_first_arg=True):
                observers.append(f"{label}.partial_fit")
            for method in ("predict", "predict_proba"):
                if self._patch_callable(
                    cls,
                    method,
                    event_type=EventType.PREDICT,
                    operator=f"{label}.{method}",
                    drop_first_arg=True,
                ):
                    observers.append(f"{label}.{method}")
        split_label = f"{StratifiedKFold.__module__}.{StratifiedKFold.__qualname__}"
        if self._patch_callable(
            StratifiedKFold,
            "split",
            event_type=EventType.DATA_SPLIT,
            operator=f"{split_label}.split",
            drop_first_arg=True,
        ):
            observers.append(f"{split_label}.split")
        self._register("sklearn", observers)

    def _instrument_pandas(self) -> None:
        observers: list[str] = []
        try:
            import pandas as pd
        except ImportError:
            self._register("pandas", observers, "optional dependency unavailable")
            return
        for name in ("read_csv", "read_parquet", "read_pickle"):
            if self._patch_callable(
                pd,
                name,
                event_type=EventType.DATA_LOAD,
                operator=f"pandas.{name}",
                drop_first_arg=False,
                artifact=True,
            ):
                observers.append(f"pandas.{name}")
        for name in ("to_csv", "to_parquet", "to_pickle"):
            if self._patch_callable(
                pd.DataFrame,
                name,
                event_type=EventType.ARTIFACT_WRITE,
                operator=f"pandas.DataFrame.{name}",
                drop_first_arg=True,
                artifact=True,
            ):
                observers.append(f"pandas.DataFrame.{name}")
        self._register("pandas", observers)

    def _instrument_joblib(self) -> None:
        observers: list[str] = []
        try:
            import joblib
        except ImportError:
            self._register("joblib", observers, "optional dependency unavailable")
            return
        for name, event_type in (("load", EventType.ARTIFACT_READ), ("dump", EventType.ARTIFACT_WRITE)):
            if self._patch_callable(
                joblib,
                name,
                event_type=event_type,
                operator=f"joblib.{name}",
                drop_first_arg=False,
                artifact=True,
            ):
                observers.append(f"joblib.{name}")
        self._register("joblib", observers)

    def _instrument_numpy(self) -> None:
        """Opt-in exact dense concatenation, positional members and producers.

        axis 0 appends row identities; axis 1 requires equal ordered identities.
        Unknown blocks, flattening and subclasses never obtain inferred members.
        Observed producer versions survive composition independently of members.
        """
        import numpy as np
        if not self.track_row_membership:
            self._register("numpy", [], "row tracking disabled")
            return
        original = np.concatenate

        @functools.wraps(original)
        def concatenate(arrays, *args, **kwargs):
            # Never consume an iterator twice or add support the original lacks.
            blocks = tuple(arrays) if isinstance(arrays, (list, tuple)) else None
            result = original(arrays, *args, **kwargs)
            if (blocks is None or not blocks or type(result) is not np.ndarray
                    or not all(type(b) is np.ndarray for b in blocks)):
                self._row_membership.pop(id(result), None)
                self._data_state_sources.pop(id(result), None)
                return result
            axis = kwargs.get("axis", args[0] if args else 0)
            members = [self._known_members(block) for block in blocks]
            selected = None
            if axis is not None and all(m is not None for m in members):
                effective_axis = int(axis) % result.ndim
                if effective_axis == 0 and all(len(m) == b.shape[0] for m, b in zip(members, blocks)):
                    selected = tuple(member for group in members for member in group)
                elif effective_axis == 1 and all(m == members[0] for m in members):
                    selected = members[0]
            if selected is not None and len(selected) == result.shape[0]:
                self._row_membership[id(result)] = (result, selected)
            else:
                # out= may overwrite a previously observed array. Its old row
                # evidence cannot survive an unknown or flattened operation.
                self._row_membership.pop(id(result), None)
            states, gaps = [], []
            for block in blocks:
                source = self._data_state_sources.get(id(block))
                if source is None or source[0] is not block:
                    if self.source_coverage == 'strict':
                        gaps.append('INPUT_PRODUCER_COVERAGE_UNSPECIFIED')
                    continue
                gaps.extend(source[3])
                if source[1] is None or model_state_sha256(block) != source[1]:
                    gaps.append("CONCATENATED_PRODUCER_OUTPUT_CHANGED_OR_UNHASHABLE")
                else:
                    states.extend(source[2])
            if states or gaps or self.source_coverage == 'strict':
                self._data_state_sources[id(result)] = (
                    result, model_state_sha256(result) if self.state_fingerprints else None,
                    tuple(dict.fromkeys(states)), tuple(dict.fromkeys(gaps)))
            else:
                self._data_state_sources.pop(id(result), None)
            return result

        np.concatenate = concatenate
        self._patches.append((np, "concatenate", original))
        self._register("numpy", ["numpy.concatenate.dense-positional-members-and-producers"])

    def start(self, *, frameworks: Sequence[str] = ("sklearn", "pandas", "joblib")) -> AutoFactCapture:
        """Enable requested framework hooks; missing frameworks remain recorded limits."""

        if self._started:
            return self
        self._started = True
        if self.mode == CaptureMode.MANUAL:
            self._register("runtime_hooks", [], "manual mode does not install framework hooks")
            return self
        available = {
            "sklearn": self._instrument_sklearn,
            "pandas": self._instrument_pandas,
            "joblib": self._instrument_joblib,
            "imblearn": self._instrument_imblearn,
            "numpy": self._instrument_numpy,
        }
        for framework in frameworks:
            selected = available.get(framework)
            if selected is None:
                self._register(str(framework), [], "unknown AutoFact framework observer")
            else:
                selected()
        return self

    def stop(self) -> None:
        """Restore all patched callables in reverse installation order."""

        while self._patches:
            owner, name, original = self._patches.pop()
            if original is _INHERITED_ATTRIBUTE:
                delattr(owner, name)
            else:
                setattr(owner, name, original)
        self._started = False

    def observe_return_mapping(
        self,
        value: Mapping[str, Any],
        *,
        operator: str = "runtime.return_mapping",
    ) -> tuple[RuntimeEvent, ...]:
        """Discover conventional split/candidate/artifact fields in a returned mapping.

        This is a schema-shape observation layer: it recognizes ordinary
        ``split``, ``candidates``, ``evaluation``, and ``native_artifacts``
        fields and never reads a benchmark ID, fault ID, or expected label.
        Ambiguous shapes are simply left unobserved.
        """

        if not isinstance(value, Mapping):
            raise TypeError("value must be a mapping.")
        before = len(self._events)
        split = value.get("split")
        if isinstance(split, Mapping):
            memberships: dict[str, tuple[str, ...]] = {}
            for role, keys in (
                ("fit", ("fit_indices", "fit_ids", "fit_keys")),
                ("assessment", ("assessment_indices", "assessment_ids", "assessment_keys")),
                ("calibration", ("calibration_indices", "calibration_ids", "calibration_keys")),
                ("threshold", ("threshold_indices", "threshold_ids", "threshold_keys")),
            ):
                observed = None
                for key in keys:
                    if key in split:
                        observed = _member_ids(split[key])
                        break
                if observed is None:
                    continue
                memberships[role] = observed
                self.emit(
                    EventType.DATA_SPLIT,
                    operator=f"{operator}.split.{role}",
                    source=EvidenceSource.AUTO_OBSERVED,
                    fold_role=role,
                    attributes={"split_role": role, "member_ids": list(observed)},
                )
            fit_ids = memberships.get("fit")
            if fit_ids is not None:
                self.emit(
                    EventType.TRANSFORM_FIT,
                    operator=f"{operator}.fit_membership",
                    source=EvidenceSource.DERIVED,
                    attributes={
                        "source_ids": list(fit_ids),
                        "derivation": "conventional_split_membership",
                        "contract_scope": "assessment_fit_boundary",
                    },
                )

        candidates = value.get("candidates")
        if isinstance(candidates, Sequence) and not isinstance(candidates, (str, bytes, bytearray)):
            for row in candidates:
                if not isinstance(row, Mapping):
                    continue
                identifier = next(
                    (
                        row.get(key)
                        for key in ("candidate_id", "candidate_key", "semantic_id", "id")
                        if isinstance(row.get(key), str) and row.get(key)
                    ),
                    None,
                )
                if identifier is None:
                    continue
                attributes = {
                    key: row[key]
                    for key in (
                        "timestamp",
                        "instrument",
                        "action",
                        "decision_execution_id",
                        # Stable semantic identity fields are copied when a
                        # native report exposes them.  The observer does not
                        # invent or interpret these values; the formal layer
                        # decides whether they close a lineage.
                        "candidate_id",
                        "canonical_candidate_id",
                        "canonical_identity",
                        "candidate_fingerprint",
                        "identity_fingerprint",
                        "lineage_identity",
                        "semantic_id",
                    )
                    if key in row
                }
                self.emit(
                    EventType.CANDIDATE_CREATE,
                    operator=f"{operator}.candidates",
                    source=EvidenceSource.AUTO_OBSERVED,
                    candidate_id=identifier,
                    attributes=attributes,
                )

        evaluation = value.get("evaluation")
        if isinstance(evaluation, Mapping):
            self.emit(
                EventType.EVALUATE,
                operator=f"{operator}.evaluation",
                source=EvidenceSource.AUTO_OBSERVED,
                attributes=dict(evaluation),
            )

        artifacts = value.get("native_artifacts")
        if isinstance(artifacts, Mapping):
            for artifact_id, descriptor in sorted(artifacts.items(), key=lambda item: str(item[0])):
                if not isinstance(artifact_id, str) or not artifact_id or not isinstance(descriptor, Mapping):
                    continue
                digest = descriptor.get("storage_sha256") or descriptor.get("raw_sha256")
                path = descriptor.get("path")
                if not isinstance(digest, str) or len(digest) != 64:
                    continue
                self.emit(
                    EventType.ARTIFACT_WRITE,
                    operator=f"{operator}.native_artifacts",
                    source=EvidenceSource.AUTO_OBSERVED,
                    artifact_path=path if isinstance(path, str) and path else None,
                    artifact_sha256=digest,
                    attributes={"artifact_id": artifact_id, "semantic_sha256": descriptor.get("semantic_sha256")},
                )
        return tuple(self._events[before:])


__all__ = ["AutoFactCapture", "InstrumentationStatus"]
