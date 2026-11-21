"""Explicit TF-IDF row contract; 1.3.2 also reviewed for common comparison."""
import functools
import numpy as np
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from .capture import model_state_sha256
from .events import EventType

def install_tfidf(engine):
    if sklearn.__version__ not in {'1.8.0','1.7.2','1.3.2'}:raise ValueError('TF-IDF row contracts require a registered sklearn version')
    contract='sklearn-'+sklearn.__version__+'/tfidf-word-row-preserving-v1'
    def accepts(value):
        return (type(value) is TfidfVectorizer and value.input=='content' and value.analyzer=='word'
                and value.preprocessor is None and value.tokenizer is None and value.vocabulary is None)
    def wrap(method):
        def decorate(original):
            @functools.wraps(original)
            def call(value,raw_documents,*args,**kwargs):
                result=original(value,raw_documents,*args,**kwargs)
                if not accepts(value):return result
                cap=engine.capture;gaps=[];states=[]
                if method=='fit_transform':
                    members=engine.members(raw_documents)
                    attrs={'contract_scope':'assessment_fit_boundary','operation_object_id':f'py:{id(value):x}',
                           'fit_semantics':'RESET','members_are_training':True,'fitted_state_sha256':model_state_sha256(value),
                           'row_contract':contract,'input_state_event_ids':[]}
                    kind, unspecified=cap.input_source_status(raw_documents)
                    gaps.extend(unspecified)
                    if cap.source_coverage=='strict':
                        attrs['input_producer_evidence']={'kind':kind}
                    prior=cap._data_state_sources.get(id(raw_documents))
                    if prior is not None and prior[0] is raw_documents:
                        if prior[1]!=model_state_sha256(raw_documents):gaps.append('TFIDF_INPUT_VERSION_CHANGED')
                        else:attrs['input_state_event_ids']=list(prior[2])
                        gaps.extend(prior[3])
                    if members is not None:attrs['source_ids']=list(dict.fromkeys(members))
                    attrs['state_dependency_gaps']=gaps
                    event=cap.emit(EventType.TRANSFORM_FIT,operator='sklearn.TfidfVectorizer.fit_transform.contract',
                                   inputs=(raw_documents,),outputs=(result,),attributes=attrs)
                    from .state_identity import LIVE_STATES
                    LIVE_STATES.advance(value)
                    cap._state_events[id(value)]=event;cap._operation_objects[f'py:{id(value):x}']=value
                else:
                    event=cap._state_events.get(id(value))
                    if event is None:gaps.append('TFIDF_FIT_NOT_OBSERVED')
                    elif event.attributes.get('fitted_state_sha256')!=model_state_sha256(value):gaps.append('TFIDF_FITTED_STATE_CHANGED')
                if event is not None:states.append(event.event_id)
                count=getattr(result,'shape',(0,))[0]
                engine.index(raw_documents,result,np.arange(count),contract=contract,producer_versions=states)
                existing=cap._data_state_sources.get(id(result))
                cap._data_state_sources[id(result)]=(result,model_state_sha256(result),
                    tuple(dict.fromkeys(states+(list(existing[2]) if existing else []))),
                    tuple(dict.fromkeys(gaps+(list(existing[3]) if existing else []))))
                return result
            return call
        return decorate
    engine.patch(TfidfVectorizer,'fit_transform',wrap('fit_transform'))
    engine.patch(TfidfVectorizer,'transform',wrap('transform'))
