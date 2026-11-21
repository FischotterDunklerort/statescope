"""Explicit fixed-source data adapters; no model decisions or task labels."""
import functools
import hashlib
import inspect
from pathlib import Path

def file_digest(value):return hashlib.sha256(Path(inspect.getsourcefile(value)).read_bytes()).hexdigest()

def install_skactiveml(engine,expected_digest):
    from skactiveml.classifier import SklearnClassifier
    method=SklearnClassifier._fit
    if file_digest(method)!=expected_digest:raise ValueError('skactiveml source contract changed')
    engine.filter_frame(method.__code__,source_name='X',output_name='X_train',index_name='is_included',
                        contract='skactiveml-native-label-mask-v1/'+expected_digest,owner_class=SklearnClassifier)

def install_dask(engine,expected_digest):
    import dask.array as da
    import dask
    import dask.array.core as core
    import numpy as np
    from itertools import product
    if file_digest(core.from_array)!=expected_digest:raise ValueError('dask source contract changed')
    original=core.from_array
    @functools.wraps(original)
    def from_array(x,*args,**kwargs):
        result=original(x,*args,**kwargs)
        # Reviewed native branch eagerly copies and slices exact dense arrays.
        # Actual graph constants are bound; keys or equal content alone do not
        # identify sources. Unknown tasks, custom getter and locks are excluded.
        if (type(x) is not np.ndarray or x.ndim not in (1,2) or kwargs.get('getitem') is not None
                or kwargs.get('lock') not in (None,False) or engine.members(x) is None
                or dask.config.get('scheduler',None) not in ('sync','synchronous','single-threaded')):return result
        graph=result.__dask_graph__().to_dict()
        slices=core.slices_from_chunks(result.chunks)
        keys=product([result.name],*(range(len(axis)) for axis in result.chunks))
        for key,selection in zip(keys,slices):
            block=graph.get(key)
            if type(block) is not np.ndarray:continue
            engine.index(x,block,np.arange(len(x))[selection[0]],contract='dask-dense-graph-constant-v1/'+expected_digest)
        return result
    engine.patch(core,'from_array',lambda _:from_array)
    if da.from_array is original:engine.patch(da,'from_array',lambda _:from_array)

def install_flaml(engine,expected_digest,validation_digest,state_digest):
    import numpy as np
    from flaml.automl.data import DataTransformer
    from flaml.automl.task.generic_task import GenericTask
    from flaml.automl.task import generic_task
    import sklearn.model_selection as selection
    from flaml.automl.state import AutoMLState
    from .capture import model_state_sha256
    from .events import EventType
    if file_digest(DataTransformer.fit_transform)!=expected_digest:raise ValueError('FLAML transformer source contract changed')
    if file_digest(GenericTask.validate_data)!=validation_digest:raise ValueError('FLAML validation source contract changed')
    if file_digest(AutoMLState.prepare_sample_train_data)!=state_digest:raise ValueError('FLAML sample source contract changed')
    # Rebind the reviewed module's imported alias for each capture session.
    # A prior session's closed wrapper must not receive the new observations.
    engine.patch(generic_task,'train_test_split',lambda _:selection.train_test_split)
    pending=[]
    def validation_wrap(original):
        signature=inspect.signature(original)
        @functools.wraps(original)
        def validate(*args,**kwargs):
            bound=signature.bind(*args,**kwargs).arguments
            pending.append((bound.get('self'),bound.get('y_train_all')))
            try:return original(*args,**kwargs)
            finally:pending.pop()
        return validate
    engine.patch(GenericTask,'validate_data',validation_wrap)
    def sample_wrap(original):
        @functools.wraps(original)
        def sample(state,sample_size):
            result=original(state,sample_size)
            if type(state) is AutoMLState and type(sample_size) is int and sample_size>=0:
                smaller=sample_size<=state.data_size[0]
                sources=(state.X_train,state.y_train) if smaller else (state.X_train_all,state.y_train_all)
                for source,output in zip(sources,result[:2]):
                    if type(source) is np.ndarray and output is not source:
                        positions=np.arange(len(source))[:sample_size] if smaller else np.arange(len(source))
                        engine.index(source,output,positions,contract='flaml-native-prefix-sample-v1/'+state_digest)
            return result
        return sample
    engine.patch(AutoMLState,'prepare_sample_train_data',sample_wrap)
    def wrap(original):
        @functools.wraps(original)
        def fit_transform(value,X,y,task):
            if pending and pending[-1][0] is task:
                source_y=pending[-1][1]
                # The reviewed validation function uses ndarray.flatten before
                # invoking this method. Only the one-dimensional label branch
                # is contracted here; no equality-based adoption.
                if type(source_y) is np.ndarray and source_y.ndim==1 and type(y) is np.ndarray:
                    engine.index(source_y,y,np.arange(len(source_y)),contract='flaml-native-label-flatten-v1/'+validation_digest)
            valid=(type(value) is DataTransformer and type(X) is np.ndarray and X.ndim==2
                   and X.dtype.kind in 'fiu' and type(task) is GenericTask
                   and (task.is_classification() or task.is_regression()))
            members=engine.members(X);target_members=engine.members(y)
            if not valid or members is None or target_members is None:engine.rejections.append({'contract':'flaml-numeric-row-preserving',
                'X_type':type(X).__name__,'y_type':type(y).__name__,'task_type':type(task).__name__,
                'X_known':members is not None,'y_known':target_members is not None})
            before=model_state_sha256(X) if valid else None
            result=original(value,X,y,task)
            gaps=[]
            if not valid:gaps.append('FLAML_TRANSFORM_CONTRACT_NOT_ESTABLISHED')
            if members is None or target_members is None:gaps.append('FLAML_TRANSFORM_INPUT_MEMBERSHIP_MISSING')
            if valid and before!=model_state_sha256(X):gaps.append('FLAML_TRANSFORM_INPUT_VERSION_CHANGED')
            _, unresolved = engine.capture.input_source_status(X)
            gaps.extend(unresolved)
            ancestry=engine.capture._data_state_sources.get(id(X))
            states=ancestry[2] if ancestry is not None and ancestry[0] is X else ()
            if ancestry is not None and ancestry[0] is X:
                gaps.extend(ancestry[3])
                if ancestry[1]!=model_state_sha256(X):gaps.append('FLAML_TRANSFORM_PRODUCER_VERSION_CHANGED')
            attrs={'contract_scope':'assessment_fit_boundary','operation_object_id':f'py:{id(value):x}',
                   'fit_semantics':'RESET' if valid else 'UNESTABLISHED','members_are_training':True,
                   'input_state_event_ids':list(states),'state_dependency_gaps':gaps,
                   'fitted_state_sha256':model_state_sha256(value),'row_contract':'flaml-numeric-row-preserving-v1/'+expected_digest}
            known=tuple(dict.fromkeys((members or ())+(target_members or ())))
            if known:attrs['source_ids']=list(known)
            event=engine.capture.emit(EventType.TRANSFORM_FIT,operator='flaml.DataTransformer.fit_transform.contract',
                inputs=(X,y),outputs=result,attributes=attrs)
            for source,output in zip((X,y),result):
                if output is X and not gaps:continue
                if hasattr(output,'shape') and output.shape[0]==len(X):
                    engine.index(source,output,np.arange(len(X)),contract='flaml-numeric-row-preserving-v1/'+expected_digest,
                                 producer_versions=(event.event_id,))
                    inherited=engine.capture._data_state_sources.get(id(output))
                    engine.capture._data_state_sources[id(output)]=(output,model_state_sha256(output),
                        tuple(dict.fromkeys((event.event_id,)+(inherited[2] if inherited else ()))),
                        tuple(dict.fromkeys(tuple(gaps)+(inherited[3] if inherited else ()))))
            return result
        return fit_transform
    engine.patch(DataTransformer,'fit_transform',wrap)
