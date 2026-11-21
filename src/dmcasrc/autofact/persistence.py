"""Observed pickle/joblib persistence in a trusted, single-writer local store.

Records originate at an observed save, never by matching an arbitrary object's
values. Native paths have explicit registry entries. Restore imports training
events, not assessment policy or a saved verdict. Open-capture assumptions remain.
This is not a security boundary against a user who can alter the trusted store.
"""
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import pickle
import secrets
import sys
import math

from .capture import model_state_sha256
from .events import EvidenceSource
from .state_lineage import FITS, predictive_components

SCHEMA='dmcasrc.observed-persistence.v1'
def digest(blob):return hashlib.sha256(blob).hexdigest()
def encoded(value):return json.dumps(value,sort_keys=True,separators=(',',':')).encode()

def encoded_chunks(value, chunk_chars=65536):
    """Identical canonical JSON bytes with a bounded encoding buffer."""
    buffer=[];size=0
    for fragment in json.JSONEncoder(sort_keys=True,separators=(',',':')).iterencode(value):
        buffer.append(fragment);size+=len(fragment)
        if size>=chunk_chars:
            yield ''.join(buffer).encode();buffer=[];size=0
    if buffer:yield ''.join(buffer).encode()

def encoded_digest(value):
    result=hashlib.sha256()
    for chunk in encoded_chunks(value):result.update(chunk)
    return result.hexdigest()

def decode_record(text):
    """Share repeated immutable member-ID strings within one saved record.

    Identical declared identifiers retain their exact values. No array values,
    model states, member lists or provenance relations are merged or removed.
    """
    members={}
    def object_hook(value):
        for name in ('source_ids','upper_bound'):
            ids=value.get(name)
            if isinstance(ids,list) and all(type(item) is str for item in ids):
                value[name]=[members.setdefault(item,item) for item in ids]
        return value
    return json.loads(text,object_hook=object_hook)
def environment():
    import sklearn,numpy,scipy,joblib
    return {'python':list(sys.version_info[:2]),'sklearn':sklearn.__version__,
            'numpy':numpy.__version__,'scipy':scipy.__version__,'joblib':joblib.__version__}
def contract():
    root=Path(__file__).parent
    files=sorted(root.glob('*.py'))+[root.parent/'application/formal_contracts.py']
    return digest(b''.join(hashlib.sha256(p.read_bytes()).digest() for p in files))

# Explicit migration from the immediately preceding reviewed runtime. Its
# source/capture semantics are unchanged; the new binary-SGD prediction
# comparison excludes alpha. Load still verifies environment, model content,
# all saved gaps and every source record before creating new state anchors.
# This identifier is a format-compatibility check, not evidence of provenance.
PREVIOUS_REVIEWED_CONTRACT = '97d6e1fe78554e99dccc802006a17cc1aac003e7166a548cc14614117af37cf8'

def supported_contract(identifier):
    return identifier in {contract(), PREVIOUS_REVIEWED_CONTRACT, '888f98499dc67c98396963d418c01c29b2b7a60a397d93186c29e2063c343ad5'}
def content_digest(value):
    """Content correspondence across pickle memo layouts; not origin evidence."""
    import numpy as np
    import scipy.sparse as sparse
    def normalize(v):
        if v is None or type(v) in (str,int,bool):return v
        if isinstance(v,np.generic):return normalize(v.item())
        if isinstance(v,np.dtype):return {'numpy_dtype':v.str,'descr':normalize(v.descr),'metadata':normalize(dict(v.metadata) if v.metadata else None)}
        if type(v) is float:return v if math.isfinite(v) else str(v)
        if isinstance(v,bytes):return {'bytes':v.hex()}
        if type(v) is slice:
            return {'slice':[normalize(v.start),normalize(v.stop),normalize(v.step)]}
        # MaskedArray.tobytes may lazily initialize fill_value on its receiver.
        # Normalize a copy so content verification never mutates the live model.
        if isinstance(v,np.ma.MaskedArray):v=v.copy()
        if isinstance(v,np.ndarray):
            if v.dtype.names:return {'structured':[(n,normalize(v[n])) for n in v.dtype.names],'shape':v.shape}
            if v.dtype.hasobject:return {'array':normalize(v.tolist()),'shape':v.shape,'dtype':str(v.dtype)}
            return {'array_bytes':v.tobytes().hex(),'shape':v.shape,'dtype':str(v.dtype)}
        if sparse.issparse(v):
            x=v.tocsr();return {'csr':[normalize(x.data),normalize(x.indices),normalize(x.indptr)],'shape':x.shape}
        if isinstance(v,dict):return {str(k):normalize(x) for k,x in sorted(v.items(),key=lambda x:str(x[0]))}
        if isinstance(v,(list,tuple)):return [normalize(x) for x in v]
        if isinstance(v,(set,frozenset)):return sorted([normalize(x) for x in v],key=str)
        if isinstance(v,type):return {'class':v.__module__+'.'+v.__qualname__}
        if isinstance(v,np.random.RandomState):return normalize(v.get_state())
        import pandas as pd
        if type(v) is pd.Index:
            return {'pandas_index':normalize(v.to_numpy()),'name':normalize(v.name),'dtype':str(v.dtype)}
        from sklearn.metrics import accuracy_score
        if v is accuracy_score:
            import sklearn,inspect
            if sklearn.__version__!='1.8.0':raise ValueError('unreviewed accuracy scorer version')
            return {'native_function':'sklearn-1.8.0/metrics.accuracy_score',
                'source_sha256':hashlib.sha256(inspect.getsource(inspect.unwrap(v)).encode()).hexdigest()}
        # Frozen-transfer failure: sklearn 1.8 log-loss SGD uses a stateless
        # Cython loss object. Only this exact reviewed type/default reduction is
        # accepted. This compatibility fix is not a new provenance relation.
        import sklearn
        if sklearn.__version__=='1.8.0':
            from sklearn._loss._loss import CyHalfBinomialLoss, CyHalfSquaredError
            if type(v) in {CyHalfBinomialLoss,CyHalfSquaredError}:
                reduced=v.__reduce__()
                if reduced!=(type(v),()):raise ValueError('unexpected stateless loss state')
                return {'type':'sklearn-1.8.0/'+type(v).__name__,'constructor_args':[]}
        if type(v).__module__=='sklearn.tree._tree':return normalize(v.__getstate__())
        if type(v).__module__=='sklearn.linear_model._sgd_fast':return normalize(v.__reduce__()[1])
        if hasattr(v,'__dict__') and type(v).__module__.startswith('sklearn.'):
            return {'type':type(v).__module__+'.'+type(v).__qualname__,'fields':normalize(vars(v))}
        raise ValueError('unsupported persistence content '+type(v).__name__)
    return digest(encoded(normalize(value)))
def inventory(model):
    objects={};aliases={};seen={}
    def visit(value,path):
        if id(value) in seen:aliases[path]=seen[id(value)];return
        seen[id(value)]=path;objects[path]=value
        for name,child in predictive_components(value):visit(child,path+'.'+name)
    visit(model,'selected');return objects,aliases

class ProvenanceStore:
    def __init__(self,root,*,storage_format='json'):
        if storage_format not in ('json','dictionary'):
            raise ValueError('unsupported evidence storage format')
        self.storage_format = storage_format
        self.root=Path(root).resolve();self.root.mkdir(parents=True,exist_ok=True)
        self.native_records=[]
    def _atomic(self,path,blob):
        temporary=path.with_name(path.name+'.'+secrets.token_hex(6)+'.tmp')
        with temporary.open('xb') as handle:handle.write(blob);handle.flush();os.fsync(handle.fileno())
        os.replace(temporary,path)
    def _atomic_json(self,path,value):
        temporary=path.with_name(path.name+'.'+secrets.token_hex(6)+'.tmp')
        with temporary.open('xb') as handle:
            for chunk in encoded_chunks(value):handle.write(chunk)
            handle.flush();os.fsync(handle.fileno())
        os.replace(temporary,path)
    def _registry(self):
        path=self.root/'paths.json'
        return json.loads(path.read_text()) if path.exists() else {}
    def record(self,capture,model,blob,format,*,native_path=None):
        if format not in {'pickle','joblib'}:raise ValueError('unsupported persistence format')
        gaps_by_purpose={}
        for purpose in ('strict','prediction','continuation'):
            capture.trace_for_model(model,purpose=purpose)
            gaps_by_purpose[purpose]=capture.lineage_receipt['gaps']
        gaps=gaps_by_purpose['strict']
        objects,aliases=inventory(model);paths={id(o):p for p,o in objects.items()};entries={}
        for path,value in objects.items():
            event=capture._state_events.get(id(value));cert=capture._component_certificates.get(id(value))
            saved_cert=None
            if cert is not None and id(cert['owner']) in paths:
                saved_cert={k:cert[k] for k in ('event_id','upper_bound','fingerprint','contract','creation_owner_id')}
                saved_cert.update({k:cert[k] for k in ('purpose_contract','prediction_fingerprint',
                    'continuation_fingerprint','latest_owner_event_id','source_state_event_ids') if k in cert})
                saved_cert['owner_path']=paths[id(cert['owner'])]
            entries[path]={'event_id':event.event_id if event is not None else None,
                'fingerprint':model_state_sha256(value),'type':type(value).__module__+'.'+type(value).__qualname__,
                'content_sha256':content_digest(value),
                'certificate':saved_cert}
        ticket=secrets.token_hex(16)
        record={'schema':SCHEMA,'ticket':ticket,'format':format,'blob_sha256':digest(blob),
            'contract':contract(),'environment':environment(),'capture_run':capture.run_id,
            'observation_scope':'OPEN_CAPTURE_CONDITIONAL','source_coverage':capture.source_coverage,
            'aliases':aliases,'objects':entries,
            'save_gaps':gaps,'save_gaps_by_purpose':gaps_by_purpose,
            'events':[event.to_dict() for event in capture.events if event.event_type in FITS],
            'native_path':str(Path(native_path).resolve()) if native_path else None,'committed':True}
        record['record_sha256']=encoded_digest(record)
        self._atomic(self.root/(ticket+'.bin'),blob)
        self._write_record(self.root/(ticket+'.json'),record)
        if native_path:
            registry=self._registry();registry[str(Path(native_path).resolve())]=ticket
            self._atomic(self.root/'paths.json',encoded(registry))
        self.native_records.append({'ticket':ticket,'path':record['native_path'],'format':format})
        return ticket
    def _write_record(self,path,record):
        if self.storage_format == 'dictionary':
            from .member_storage import pack
            record = pack(record)
            record['container_sha256'] = encoded_digest(record)
        self._atomic_json(path,record)

    def save(self,capture,model,*,format='pickle'):
        if format=='pickle':blob=pickle.dumps(model,protocol=5)
        elif format=='joblib':
            import joblib
            stream=io.BytesIO();joblib.dump(model,stream);blob=stream.getvalue()
        else:raise ValueError('unsupported persistence format')
        return self.record(capture,model,blob,format)
    def _read(self,ticket):
        if type(ticket) is not str or len(ticket)!=32 or any(c not in '0123456789abcdef' for c in ticket):
            raise ValueError('unknown persistence ticket')
        try:record=decode_record((self.root/(ticket+'.json')).read_text());blob=(self.root/(ticket+'.bin')).read_bytes()
        except FileNotFoundError as error:raise ValueError('unknown persistence ticket') from error
        if 'storage_format' in record:
            from .member_storage import unpack
            checksum = record.pop('container_sha256',None)
            if encoded_digest(record) != checksum:
                raise ValueError('persistence container changed')
            record = unpack(record)
        check=dict(record);expected=check.pop('record_sha256',None)
        if encoded_digest(check)!=expected:raise ValueError('persistence record changed')
        if record['schema']!=SCHEMA or record['ticket']!=ticket or not record['committed']:raise ValueError('invalid persistence commit')
        if not supported_contract(record['contract']) or record['environment']!=environment():raise ValueError('persistence contract/environment changed')
        if digest(blob)!=record['blob_sha256']:raise ValueError('persistence bytes changed')
        return record,blob
    def load(self,capture,ticket):
        from dmcasrc.application.managed import _read_event
        record,blob=self._read(ticket)
        if capture.source_coverage == 'strict' and record.get('source_coverage') != 'strict':
            raise ValueError('ARCHIVED_PRODUCER_COVERAGE_NOT_VERIFIED')
        if record['format']=='pickle':model=pickle.loads(blob)
        else:
            import joblib
            model=joblib.load(io.BytesIO(blob))
        objects,aliases=inventory(model)
        if set(objects)!=set(record['objects']) or aliases!=record['aliases']:raise ValueError('restored component/alias structure differs')
        for path,value in objects.items():
            entry=record['objects'][path]
            if entry['type']!=type(value).__module__+'.'+type(value).__qualname__:raise ValueError('restored type differs')
            if entry['content_sha256']!=content_digest(value):raise ValueError('restored state differs')
        events={event.event_id:event for event in capture.events};incoming=[]
        for raw in record['events']:
            current=events.get(raw['event_id'])
            if current is not None:
                if current.to_dict()!=raw:raise ValueError('training event identity collision')
            else:incoming.append(_read_event(raw))
        for event in incoming:
            if event.event_id not in events:
                capture._events.append(event);events[event.event_id]=event
                capture._sequence=max(capture._sequence,event.logical_time)
        for path,value in objects.items():
            entry=record['objects'][path];event=events.get(entry['event_id'])
            if event is not None:
                attrs=dict(event.attributes)
                attrs.update(operation_object_id=f'py:{id(value):x}',fit_semantics='RESET',previous_state_event_id=None,
                    import_state_event_ids=[event.event_id],input_state_event_ids=[],component_state_event_ids=[],
                    state_dependency_gaps=list(record['save_gaps']),
                    restored_gaps_by_purpose=record.get('save_gaps_by_purpose',{}),
                    fitted_state_sha256=model_state_sha256(value),
                    learned_state_sha256=model_state_sha256({k:v for k,v in vars(value).items()
                        if k.endswith('_') and not k.startswith('_')}),
                    source_ids=[],members_are_training=False,restored_origin=ticket,restored_contract=SCHEMA,
                    observation_scope='OPEN_CAPTURE_CONDITIONAL')
                from .state_certificates import reviewed_purpose_state, purpose_fingerprint, continuation_input_fingerprint
                if reviewed_purpose_state(value):
                    attrs.update(prediction_state_sha256=purpose_fingerprint(value,'prediction'),
                        continuation_input_sha256=continuation_input_fingerprint(value))
                restored=capture.emit(event.event_type,operator='observed.persistence.restore',source=EvidenceSource.DERIVED,attributes=attrs)
                capture._state_events[id(value)]=restored;capture._operation_objects[attrs['operation_object_id']]=value
                learned={k:v for k,v in vars(value).items() if k.endswith('_') and hasattr(v,'shape')}
                capture._receiver_learned_objects[id(value)]=learned
                for item in learned.values():capture._learned_state_sources[id(item)]=(item,model_state_sha256(item),restored.event_id)
            cert=entry['certificate']
            if cert is not None:
                cert=dict(cert);owner=objects[cert.pop('owner_path')]
                capture._component_certificates[id(value)]={**cert,'object':value,'owner':owner,'fingerprint':model_state_sha256(value)}
                rebound=capture._component_certificates[id(value)]
                restored_child=capture._state_events.get(id(value))
                ids=(*rebound.get('source_state_event_ids',()),entry['event_id'],
                    restored_child.event_id if restored_child is not None else None)
                rebound['source_state_event_ids']=tuple(dict.fromkeys(i for i in ids if i is not None))
        # Parents can appear after children in alias-heavy inventories. Rebind
        # purpose anchors only once every restored owner event exists.
        from .state_certificates import tree_prediction_fingerprint, CERTIFICATE_VERSION
        for value in objects.values():
            cert=capture._component_certificates.get(id(value))
            if cert is not None and cert.get('purpose_contract')==CERTIFICATE_VERSION:
                owner_event=capture._state_events.get(id(cert['owner']))
                if owner_event is not None:
                    cert.update(prediction_fingerprint=tree_prediction_fingerprint(value),
                        continuation_fingerprint=model_state_sha256(value),latest_owner_event_id=owner_event.event_id)
                    child_event=capture._state_events.get(id(value))
                    if child_event is not None:
                        cert['source_state_event_ids']=tuple(dict.fromkeys((*cert.get('source_state_event_ids',()),child_event.event_id)))
        return model
    @contextlib.contextmanager
    def observe_native(self,capture):
        """Observe named single-object files; unregistered native loads stay native."""
        import joblib
        originals=(pickle.dump,pickle.load,joblib.dump,joblib.load);busy=[False]
        def path_of(value):
            if not isinstance(value,(str,os.PathLike)):value=getattr(value,'name',value)
            return Path(value).resolve() if isinstance(value,(str,os.PathLike)) else None
        def after_save(model,value,format):
            path=path_of(value)
            if path is None:return
            if hasattr(value,'flush'):value.flush()
            self.record(capture,model,path.read_bytes(),format,native_path=path)
        def native_load(value,format,original,*args,**kwargs):
            path=path_of(value);ticket=self._registry().get(str(path)) if path else None
            if ticket is None:return original(value,*args,**kwargs)
            if args or kwargs:raise ValueError('registered native restore supports default load options only')
            record,blob=self._read(ticket)
            if record['format']!=format or digest(path.read_bytes())!=digest(blob):raise ValueError('native path changed after observed save')
            return self.load(capture,ticket)
        def p_dump(model,file,*args,**kwargs):
            start=file.tell() if hasattr(file,'tell') else None
            result=originals[0](model,file,*args,**kwargs)
            if not busy[0] and start==0:after_save(model,file,'pickle')
            return result
        def p_load(file,*args,**kwargs):
            if hasattr(file,'tell') and file.tell()!=0:return originals[1](file,*args,**kwargs)
            return native_load(file,'pickle',originals[1],*args,**kwargs)
        def j_dump(model,file,*args,**kwargs):
            result=originals[2](model,file,*args,**kwargs)
            if not busy[0] and path_of(file) is not None:
                if len(result)!=1:raise ValueError('multi-file joblib persistence not contracted')
                after_save(model,file,'joblib')
            return result
        def j_load(file,*args,**kwargs):return native_load(file,'joblib',originals[3],*args,**kwargs)
        pickle.dump,pickle.load,joblib.dump,joblib.load=p_dump,p_load,j_dump,j_load
        try:yield self
        finally:pickle.dump,pickle.load,joblib.dump,joblib.load=originals
