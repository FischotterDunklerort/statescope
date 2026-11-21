"""Limited managed execution for training-state provenance.

The trusted worker owns estimators and input snapshots. Clients send only JSON
commands plus numeric array copies, never estimators, pickle payloads, callbacks
or arbitrary attribute assignments. This is a cooperative local API boundary,
not a sandbox against a hostile operating-system user or compromised sklearn.
Save/restore tickets name immutable worker-owned snapshots with retained traces;
equal bytes at an external path are never accepted as provenance.
"""
from __future__ import annotations

import hashlib
import io
import json
import multiprocessing as mp
import os
import secrets
from dataclasses import replace
from pathlib import Path

import numpy as np

from dmcasrc.autofact.capture import AutoFactCapture,model_state_sha256
from dmcasrc.autofact.events import EventType,EvidenceSource,RuntimeEvent,ObjectReference

SCHEMA='dmcasrc.managed-worker.v2/sklearn-1.8.0'


def _contract_digest():
    from dmcasrc.autofact import capture,state_lineage,component_contracts,component_registry
    files=[Path(__file__),Path(capture.__file__),Path(state_lineage.__file__),Path(component_contracts.__file__),Path(component_registry.__file__)]
    files.extend([Path(__file__).with_name('formal_contracts.py'),Path(__file__).with_name('release.py'),
                  Path(capture.__file__).with_name('row_sources.py'),Path(capture.__file__).with_name('events.py')])
    adapter_path=Path(component_registry.__file__).with_name('component_adapters.py')
    if adapter_path.is_file():files.append(adapter_path)
    return hashlib.sha256(b''.join(hashlib.sha256(path.read_bytes()).digest() for path in files)).hexdigest()


def _read_event(value):
    value=dict(value)
    for field in ['inputs','outputs']:
        value[field]=tuple(ObjectReference(**{**reference,
            'shape':None if reference['shape'] is None else tuple(reference['shape'])}) for reference in value[field])
    value['event_type']=EventType(value['event_type']);value['source']=EvidenceSource(value['source'])
    value['parent_event_ids']=tuple(value['parent_event_ids'])
    if value['temporal_range'] is not None:value['temporal_range']=tuple(value['temporal_range'])
    return RuntimeEvent(**value)


def _factories():
    from sklearn.linear_model import SGDClassifier,SGDRegressor,LogisticRegression
    from sklearn.preprocessing import StandardScaler,MinMaxScaler
    from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor,ExtraTreesClassifier,ExtraTreesRegressor
    from sklearn.calibration import CalibratedClassifierCV
    return {cls.__name__:cls for cls in (SGDClassifier,SGDRegressor,LogisticRegression,
        StandardScaler,MinMaxScaler,RandomForestClassifier,RandomForestRegressor,
        ExtraTreesClassifier,ExtraTreesRegressor,CalibratedClassifierCV)}


def _json_value(value):
    if value is None or type(value) in {str,int,bool}:return value
    if type(value) is float and np.isfinite(value):return value
    if type(value) in {list,tuple}:return [_json_value(item) for item in value]
    if type(value) is dict and all(type(key) is str for key in value):
        return {key:_json_value(item) for key,item in value.items()}
    raise TypeError('managed commands accept JSON values, not Python objects or callbacks')


class _ManagedCapture(AutoFactCapture):
    def observation_boundary_for(self,model):
        if not self._managed_valid(model):return None
        return {'scope':'MANAGED_WORKER_ALLOWLIST','schema':SCHEMA,
                'worker_pid':os.getpid(),'session':self.run_id,
                'audit_binding':self._managed_binding(model),
                'assumptions':['trusted worker, sklearn and local snapshot store',
                               'registered member identities are accurate',
                               'no arbitrary code or raw estimator aliases in worker commands']}


class _Worker:
    def __init__(self,root):
        import sklearn
        if sklearn.__version__!='1.8.0':raise ValueError('managed contracts require sklearn 1.8.0')
        self.root=Path(root).resolve();self.root.mkdir(parents=True,exist_ok=True)
        self.models={};self.data={};self.snapshots={};self.expected={};self.invalid=set()
        self.expected_components={}
        self.audit_tickets={};self.initial_contract=_contract_digest()
        self.capture=_ManagedCapture(run_id='managed-'+secrets.token_hex(8),track_row_membership=True,
                                     test_partition_role='assessment')
        self.capture._managed_valid=self.valid
        self.capture._managed_binding=self.binding
        self.capture.start()
        from dmcasrc.autofact.row_sources import RowSourceContracts
        self.rows=RowSourceContracts(self.capture);self.rows.__enter__()

    def identifier(self):return secrets.token_hex(16)

    def valid(self,model):
        if _contract_digest()!=self.initial_contract:return False
        if not any(value is model for value in self.models.values()):return False
        # Prediction may update internal sklearn caches, so verify at the
        # operation boundary and refresh only after a permitted operation.
        from dmcasrc.autofact.state_lineage import predictive_components
        seen=set()
        def visit(value):
            if id(value) in seen:return True
            seen.add(id(value))
            if id(value) in self.invalid:return False
            if id(value) in self.expected and model_state_sha256(value)!=self.expected[id(value)]:return False
            if id(value) in self.expected_components and self.component_versions(value)!=self.expected_components[id(value)]:return False
            return all(visit(child) for _,child in predictive_components(value))
        return visit(model)

    def remember(self,model):
        self.expected[id(model)]=model_state_sha256(model)
        self.expected_components[id(model)]=self.component_versions(model)

    def component_versions(self,model):
        from dmcasrc.autofact.state_lineage import predictive_components
        versions={};seen={id(model)}
        def visit(value,path):
            for name,child in predictive_components(value):
                if id(child) in seen:continue
                seen.add(id(child));key=path+'.'+name
                event=self.capture._state_events.get(id(child))
                cert=self.capture._component_certificates.get(id(child))
                versions[key]=(id(child),None if event is None else event.event_id,
                               None if cert is None else cert['event_id'])
                visit(child,key)
        visit(model,'selected');return versions

    def binding(self,model):
        self.capture.trace_for_model(model);lineage=self.capture.lineage_receipt
        assessment=sorted({str(m) for event in self.capture.events if event.event_type==EventType.DATA_SPLIT
                           and event.fold_role=='assessment' for m in event.attributes.get('member_ids',())})
        value={'session':self.capture.run_id,'scope':SCHEMA,'contract_sha256':self.initial_contract,
               'assessment_sha256':hashlib.sha256(json.dumps(assessment).encode()).hexdigest(),
               'selected_object_id':f'py:{id(model):x}','model_state_sha256':model_state_sha256(model),
               'root_versions':lineage['root_state_versions'],'retained_versions':lineage['retained_fit_event_ids'],
               'gaps':lineage['gaps']}
        value['sha256']=hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()
        return value

    def select_data(self,key,indices):
        X,y,members=self.data[key]
        if type(indices) is not list or any(type(i) is not int or i<0 or i>=len(X) for i in indices):
            raise ValueError('selection requires valid integer row positions')
        ix=np.asarray(indices,dtype=int);z=X[ix];target=None if y is None else y[ix]
        self.rows.index(X,z,ix,contract='managed-native-row-select-v1')
        if target is not None:self.rows.index(y,target,ix,contract='managed-native-row-select-v1')
        selected=tuple(members[i] for i in indices);new=self.identifier();self.data[new]=(z,target,selected)
        return new

    def dispatch(self,command):
        op=command['op'];c=self.capture
        if op=='data':
            X=np.asarray(command['X']).copy();y=command['y']
            if X.dtype.kind not in 'biuf' or X.ndim!=2:raise ValueError('numeric two-dimensional data required')
            y=None if y is None else np.asarray(y).copy()
            if y is not None and (y.dtype.kind not in 'biuf' or len(y)!=len(X)):raise ValueError('numeric aligned targets required')
            members=tuple(command['members'])
            if len(members)!=len(X) or len(set(members))!=len(members):raise ValueError('unique aligned member identities required')
            key=self.identifier();self.data[key]=(X,y,members)
            c._row_membership[id(X)]=(X,members)
            if y is not None:c._row_membership[id(y)]=(y,members)
            if command['assessment']:
                c.emit(EventType.DATA_SPLIT,operator='managed.register_assessment',fold_role='assessment',
                       attributes={'member_ids':list(members),'split_role':'assessment'})
            return key
        if op=='select':return self.select_data(command['data'],command['indices'])
        if op=='filter':
            X,_,_=self.data[command['data']];mask=command['mask']
            if type(mask) is not list or len(mask)!=len(X) or any(type(v) is not bool for v in mask):
                raise ValueError('filter requires an aligned boolean mask')
            return self.select_data(command['data'],[i for i,v in enumerate(mask) if v])
        if op=='chunks':
            X,_,_=self.data[command['data']];size=command['rows']
            if type(size) is not int or size<=0:raise ValueError('positive integer chunk rows required')
            return [self.select_data(command['data'],list(range(start,min(start+size,len(X))))) for start in range(0,len(X),size)]
        if op=='create':
            factory=_factories().get(command['kind'])
            if factory is None:raise ValueError('constructor not in managed allowlist')
            params=command['params']
            for key,value in params.items():
                if key in {'estimator','base_estimator','memory','score_func'}:raise ValueError('object/callback constructor input requires a separate contract')
                if key=='n_jobs' and value not in {None,1}:raise ValueError('managed execution currently requires n_jobs=1')
            value=factory(**params)
            # Calibration's supported underlying estimator is constructed here.
            if command['kind']=='CalibratedClassifierCV':
                from sklearn.linear_model import LogisticRegression
                value.estimator=LogisticRegression(max_iter=200)
            key=self.identifier();self.models[key]=value;return key
        if op=='compose':
            from sklearn.pipeline import Pipeline
            steps=[(name,self.models[key]) for name,key in command['steps']]
            if not steps or any(not self.valid(value) for _,value in steps):raise ValueError('invalid component state')
            key=self.identifier();self.models[key]=Pipeline(steps);self.remember(self.models[key]);return key
        if op=='fit':
            model=self.models[command['model']]
            if not self.valid(model):raise ValueError('managed state invalidated')
            method=command['method']
            if method not in {'fit','partial_fit'}:raise ValueError('unsupported training operation')
            from dmcasrc.autofact.state_lineage import receiver_operation_contract
            semantics,_=receiver_operation_contract(model,method)
            if semantics=='UNESTABLISHED':raise ValueError('receiver operation contract not established')
            X,y,_=self.data[command['data']];kwargs=command['kwargs']
            if any(key not in {'classes'} for key in kwargs):raise ValueError('training kwargs outside managed contract')
            donor_key=command.get('donor')
            if donor_key is not None:
                if method!='fit' or type(model).__name__ not in {'SGDClassifier','SGDRegressor'}:raise ValueError('import requires supported SGD fit initializers')
                donor=self.models[donor_key]
                if not self.valid(donor):raise ValueError('donor invalidated')
                if id(donor) not in c._state_events:raise ValueError('donor has no recorded training state')
                origin=c._state_events[id(donor)].event_id
                initializers={'coef_init':donor.coef_.copy(),'intercept_init':donor.intercept_.copy()}
                for item in initializers.values():
                    c._learned_state_sources[id(item)]=(item,model_state_sha256(item),origin)
                c.emit(EventType.CANDIDATE_CREATE,operator='managed.copy_initializers',
                    source=EvidenceSource.DERIVED,parent_event_ids=(origin,),
                    attributes={'donor_state_event_id':origin,'copy_rule':'fresh array from registered donor',
                                'parameters':list(initializers)})
                kwargs={**kwargs,**initializers}
            try:
                if y is None:getattr(model,method)(X,**kwargs)
                else:getattr(model,method)(X,y,**kwargs)
            except Exception:
                # Native operations can mutate before raising. Do not reuse
                # evidence from before a failed operation.
                self.invalid.add(id(model));raise
            self.remember(model)
            return self.audit(command['model'])
        if op=='transform':
            model=self.models[command['model']]
            if not self.valid(model):raise ValueError('producer invalidated')
            if type(model).__name__ not in {'StandardScaler','MinMaxScaler'}:raise ValueError('transform contract not established')
            X,y,members=self.data[command['data']]
            z=model.transform(X);key=self.identifier();self.data[key]=(z,y,members)
            self.remember(model);return key
        if op=='predict':
            model=self.models[command['model']]
            if not self.valid(model):raise ValueError('managed state invalidated')
            X,_,_=self.data[command['data']]
            result=np.asarray(model.predict(X)).copy()
            self.remember(model);return result.tolist()
        if op=='audit':return self.audit(command['model'])
        if op=='save':return self.save(command['model'])
        if op=='restore':return self.restore(command['ticket'])
        if op=='publish':
            import joblib
            from dmcasrc.application.formal_contracts import learned_fit_contract
            from dmcasrc.application.release import record_artifact_write,release_artifact_claim
            model=self.models[command['model']]
            if not self.valid(model):raise ValueError('managed state invalidated')
            ticket=command.get('audit_ticket')
            if ticket is not None:
                previous=self.audit_tickets.get(ticket)
                if previous is None or previous['model']!=command['model'] or previous['binding']!=self.binding(model)['sha256']:
                    raise ValueError('audit ticket invalidated by model, policy or contract change')
            name=self.identifier();path=self.root/(name+'.joblib');joblib.dump(model,path)
            record_artifact_write(c,path,artifact_id=name,candidate_id=name)
            return release_artifact_claim(c,path,self.root/(name+'.receipt.json'),artifact_id=name,
                candidate_id=name,claim_id=name,contracts=(learned_fit_contract(),),model=model,
                require_complete_observation=True)
        if op=='trace':return [event.to_dict() for event in c.events]
        raise ValueError('operation not in managed allowlist: '+str(op))

    def audit(self,key):
        from dmcasrc.application.formal_contracts import evaluate_formal_contracts,learned_fit_contract
        model=self.models[key]
        result=evaluate_formal_contracts(self.capture.trace_for_model(model),contracts=(learned_fit_contract(),)).results[0]
        lineage=self.capture.lineage_receipt;binding=self.binding(model);ticket=self.identifier()
        self.audit_tickets[ticket]={'model':key,'binding':binding['sha256']}
        valid=self.valid(model);decision='UNKNOWN' if result.status.value=='UNVERIFIABLE' else result.status.value
        if not valid and decision=='PASS':decision='UNKNOWN'
        return {'decision':decision,'lineage':lineage,'boundary_valid':valid,'audit_binding':binding,'audit_ticket':ticket}

    def save(self,key):
        import joblib
        from dmcasrc.autofact.state_lineage import predictive_components
        model=self.models[key]
        if not self.valid(model):raise ValueError('managed state invalidated')
        ticket=self.identifier();path=self.root/(ticket+'.snapshot')
        stream=io.BytesIO();joblib.dump(model,stream);blob=stream.getvalue()
        with path.open('xb') as handle:handle.write(blob)
        # Keep the origin independently of the caller-visible path and bytes.
        records={};seen=set()
        def visit(value,path):
            if id(value) in seen:return
            seen.add(id(value))
            records[path]={'object':value,'event':self.capture._state_events.get(id(value)),
                           'certificate':self.capture._component_certificates.get(id(value))}
            for name,child in predictive_components(value):visit(child,path+'.'+name)
        visit(model,'selected')
        self.snapshots[ticket]={'path':path,'sha256':hashlib.sha256(blob).hexdigest(),'records':records,
                                'contract':SCHEMA,'origin':self.identifier()}
        paths={id(item['object']):name for name,item in records.items()}
        components={}
        for name,item in records.items():
            cert=item['certificate']
            if cert is not None and id(cert['owner']) in paths:
                cert={key:cert[key] for key in ['event_id','upper_bound','fingerprint','contract','creation_owner_id']} | {
                    'owner_path':paths[id(item['certificate']['owner'])]}
            else:cert=None
            components[name]={'event_id':None if item['event'] is None else item['event'].event_id,
                              'certificate':cert}
        origin={'schema':SCHEMA,'ticket':ticket,'snapshot_sha256':self.snapshots[ticket]['sha256'],
                'contract_digest':_contract_digest(),'origin':self.snapshots[ticket]['origin'],
                'components':components,
                # Assessment is a current query declaration, not training
                # provenance. A new session must supply its own assessment.
                'source_events':[event.to_dict() for event in self.capture.events
                    if event.event_type in {EventType.MODEL_FIT,EventType.TRANSFORM_FIT}]}
        origin['record_sha256']=hashlib.sha256(json.dumps(origin,sort_keys=True).encode()).hexdigest()
        with (self.root/(ticket+'.origin.json')).open('x',encoding='utf8') as handle:
            json.dump(origin,handle,indent=2,sort_keys=True);handle.flush();os.fsync(handle.fileno())
        return {'ticket':ticket,'path':str(path),'sha256':self.snapshots[ticket]['sha256'],
                'session':self.capture.run_id,'contract':SCHEMA}

    def read_origin(self,ticket):
        if type(ticket) is not str or len(ticket)!=32 or any(char not in '0123456789abcdef' for char in ticket):
            raise ValueError('unknown snapshot ticket; external objects and paths are not accepted')
        origin_path=self.root/(ticket+'.origin.json')
        if not origin_path.is_file():raise ValueError('unknown snapshot ticket; no worker-owned origin record')
        origin=json.loads(origin_path.read_text(encoding='utf8'))
        digest=origin.pop('record_sha256',None)
        if digest!=hashlib.sha256(json.dumps(origin,sort_keys=True).encode()).hexdigest():
            raise ValueError('snapshot origin record changed')
        if origin['schema']!=SCHEMA or origin['contract_digest']!=_contract_digest():
            raise ValueError('snapshot contract invalidated')
        if origin['ticket']!=ticket:raise ValueError('snapshot origin ticket mismatch')
        c=self.capture;events={event.event_id:event for event in c.events}
        for value in origin['source_events']:
            event=_read_event(value)
            if event.event_id not in events:
                c._events.append(event);events[event.event_id]=event
                c._sequence=max(c._sequence,event.logical_time)
        # Tokens represent original component identities for path rebinding;
        # no model state or provenance is inferred from token equality.
        tokens={name:object() for name in origin['components']};records={}
        for name,item in origin['components'].items():
            cert=item['certificate']
            if cert is not None:cert={**cert,'owner':tokens[cert['owner_path']]}
            records[name]={'object':tokens[name],'event':events.get(item['event_id']),
                           'certificate':cert}
        return {'path':self.root/(ticket+'.snapshot'),'sha256':origin['snapshot_sha256'],
                'records':records,'contract':SCHEMA,'origin':origin['origin']}

    def restore(self,ticket):
        import joblib
        from dmcasrc.autofact.state_lineage import predictive_components
        record=self.read_origin(ticket)
        if record['contract']!=SCHEMA:raise ValueError('snapshot contract invalidated')
        blob=record['path'].read_bytes()
        if hashlib.sha256(blob).hexdigest()!=record['sha256']:raise ValueError('snapshot bytes changed')
        # Only a worker-created snapshot selected by its registry ticket is
        # deserialized. Content hash alone never creates this registry entry.
        model=joblib.load(io.BytesIO(blob));new={};seen=set()
        def visit(value,path):
            if id(value) in seen:return
            seen.add(id(value));new[path]=value
            for name,child in predictive_components(value):visit(child,path+'.'+name)
        visit(model,'selected')
        if set(new)!=set(record['records']):raise ValueError('restored component structure differs')
        mapping={id(record['records'][path]['object']):value for path,value in new.items()}
        c=self.capture
        for path,value in new.items():
            old=record['records'][path];event=old['event']
            if event is not None:
                attrs=dict(event.attributes)
                attrs.update(operation_object_id=f'py:{id(value):x}',fit_semantics='RESET',
                    previous_state_event_id=None,import_state_event_ids=[event.event_id],
                    input_state_event_ids=[],component_state_event_ids=[],state_dependency_gaps=[],
                    fitted_state_sha256=model_state_sha256(value),
                    learned_state_sha256=model_state_sha256({k:v for k,v in vars(value).items()
                        if k.endswith('_') and not k.startswith('_')}),
                    restored_origin=record['origin'],restored_contract=SCHEMA)
                # Sources are inherited through the origin edge, not claimed
                # as a fresh fit on the latest data.
                attrs['source_ids']=[];attrs['members_are_training']=False
                restored=c.emit(event.event_type,operator='managed.restore',source=EvidenceSource.DERIVED,attributes=attrs)
                c._state_events[id(value)]=restored;c._operation_objects[attrs['operation_object_id']]=value
                c._receiver_learned_objects[id(value)]={k:v for k,v in vars(value).items()
                    if k.endswith('_') and not k.startswith('_') and hasattr(v,'shape')}
                for name,item in vars(value).items():
                    if name.endswith('_') and hasattr(item,'shape'):
                        c._learned_state_sources[id(item)]=(item,model_state_sha256(item),restored.event_id)
            cert=old['certificate']
            if cert is not None:
                c._component_certificates[id(value)]={**cert,'object':value,
                    'owner':mapping[id(cert['owner'])],'fingerprint':model_state_sha256(value)}
            self.remember(value)
        # Rebind after all restored child versions are installed. Otherwise a
        # parent would remember the temporary, incomplete child registry.
        for value in new.values():self.remember(value)
        key=self.identifier();self.models[key]=model;return key


def _serve(connection,root):
    worker=None
    try:
        worker=_Worker(root);connection.send({'ok':True,'pid':os.getpid(),'schema':SCHEMA})
        while True:
            # Raw pipe traffic is an internal protocol. The public client
            # validates JSON/array payloads before using multiprocessing IPC.
            request=connection.recv()
            if request['op']=='close':break
            try:connection.send({'ok':True,'value':worker.dispatch(request)})
            except Exception as exc:connection.send({'ok':False,'error':type(exc).__name__+': '+str(exc)})
    except Exception as exc:
        try:connection.send({'ok':False,'error':type(exc).__name__+': '+str(exc)})
        except (EOFError,BrokenPipeError):pass
    finally:
        if worker is not None:worker.rows.__exit__();worker.capture.stop()
        connection.close()


class ManagedSession:
    """A local, trusted worker with explicit model/data operations.

    Tickets can be restored by a new worker using the same trusted local store
    and exact contract digest. Cross-host/external pickle import is unsupported.
    Same-user store tampering and arbitrary IPC access are outside the trust
    model. Unknown pickle/joblib inputs are not loaded.
    """
    def __init__(self,root,*,timeout=90):
        ctx=mp.get_context('spawn');parent,child=ctx.Pipe()
        self._connection=parent;self._timeout=timeout
        self._process=ctx.Process(target=_serve,args=(child,str(Path(root).resolve())),daemon=True)
        self._process.start();child.close()
        ready=self._receive();self.worker_pid=ready['pid']

    def _receive(self):
        if not self._connection.poll(self._timeout):
            self._process.terminate();self._process.join();raise TimeoutError('managed worker timeout; session terminated')
        reply=self._connection.recv()
        if not reply['ok']:raise ValueError(reply['error'])
        return reply

    def command(self,op,**kwargs):
        request=_json_value({'op':op,**kwargs})
        self._connection.send(request);return self._receive().get('value')

    def register(self,X,y,members,*,assessment=False):
        X=np.asarray(X);y=None if y is None else np.asarray(y)
        # Validate before IPC so no object-dtype values invoke pickling hooks.
        if X.dtype.kind not in 'biuf' or (y is not None and y.dtype.kind not in 'biuf'):
            raise TypeError('only numeric arrays can enter managed execution')
        ids=[str(member) for member in members]
        return self.command('data',X=X.tolist(),y=None if y is None else y.tolist(),members=ids,assessment=bool(assessment))

    def create(self,kind,**params):return self.command('create',kind=kind,params=params)
    def fit(self,model,data,*,method='fit',donor=None,**kwargs):
        return self.command('fit',model=model,data=data,method=method,donor=donor,kwargs=kwargs)
    def transform(self,model,data):return self.command('transform',model=model,data=data)
    def select(self,data,indices):return self.command('select',data=data,indices=[int(v) if isinstance(v,np.integer) else v for v in indices])
    def filter(self,data,mask):return self.command('filter',data=data,mask=[bool(v) if isinstance(v,np.bool_) else v for v in mask])
    def chunks(self,data,rows):return self.command('chunks',data=data,rows=rows)
    def compose(self,steps):return self.command('compose',steps=steps)
    def audit(self,model):return self.command('audit',model=model)
    def predict(self,model,data):return np.asarray(self.command('predict',model=model,data=data))
    def save(self,model):return self.command('save',model=model)
    def restore(self,ticket):return self.command('restore',ticket=ticket)
    def publish(self,model,*,audit_ticket=None):return self.command('publish',model=model,audit_ticket=audit_ticket)
    def trace(self):return self.command('trace')
    def close(self):
        if self._process.is_alive():
            try:self._connection.send({'op':'close'})
            except (EOFError,BrokenPipeError):pass
            self._process.join(5)
            if self._process.is_alive():self._process.terminate();self._process.join()
        self._connection.close()
    def __enter__(self):return self
    def __exit__(self,*args):self.close()
