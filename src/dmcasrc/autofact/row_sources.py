"""Observed positional mappings and producer versions for bounded adapters.

Adapters attest actual operations in reviewed native code. Equality of values
never establishes a mapping. This cooperative capture API is not a sandbox.
"""
import functools
import sys
import threading
from .capture import model_state_sha256
from .events import EventType


class RowSourceContracts:
    def __init__(self,capture):
        self.capture=capture;self.bindings={};self.resolvers=[];self._resolving=set();self.patches=[]
        self.records=[];self.rejections=[];self._previous=None
        self._thread=threading.get_ident()

    def __enter__(self):
        self._previous=getattr(self.capture,'_row_source_contracts',None)
        if self._previous is not None:raise ValueError('row contract context already active')
        self.capture._row_source_contracts=self
        return self

    def __exit__(self,*unused):
        for owner,name,original in reversed(self.patches):setattr(owner,name,original)
        self.capture._row_source_contracts=self._previous

    def patch(self,owner,name,wrapper):
        original=getattr(owner,name);replacement=wrapper(original)
        setattr(owner,name,replacement);self.patches.append((owner,name,original))
        return replacement

    def _raw(self,value):
        item=self.capture._row_membership.get(id(value))
        return item[1] if item is not None and item[0] is value else None

    def members(self,value):
        if threading.get_ident()!=self._thread:return None
        binding=self.bindings.get(id(value))
        if binding is not None and binding['object'] is value:
            if model_state_sha256(value)!=binding['fingerprint']:
                self.capture._row_membership.pop(id(value),None)
                self.capture._data_state_sources[id(value)]=(value,model_state_sha256(value),(),('ROW_OUTPUT_VERSION_CHANGED',))
                return None
            return binding['members']
        known=self._raw(value)
        if known is not None:return known
        if id(value) in self._resolving:return None
        self._resolving.add(id(value));frame=None
        try:
            frame=sys._getframe(1)
            for _ in range(64):
                if frame is None:break
                for code,resolver in self.resolvers:
                    if frame.f_code is code and resolver(frame,value):return self._raw(value)
                frame=frame.f_back
        finally:
            self._resolving.discard(id(value));del frame
        return None

    def index(self,source,output,positions,*,contract,producer_versions=()):
        import numpy as np
        members=self.members(source)
        if members is None:return False
        if not hasattr(output,'shape') or len(output.shape)<1:return False
        positions=np.asarray(positions)
        if positions.ndim!=1 or positions.dtype.kind not in 'iu':return False
        if np.any(positions<0) or np.any(positions>=len(members)):return False
        selected=tuple(members[int(i)] for i in positions)
        if len(selected)!=output.shape[0]:return False
        source_hash=model_state_sha256(source);output_hash=model_state_sha256(output)
        if source_hash is None or output_hash is None:return False
        states=list(producer_versions);gaps=[]
        ancestry=self.capture._data_state_sources.get(id(source))
        _, unspecified=self.capture.input_source_status(source)
        gaps.extend(unspecified)
        if ancestry is not None and ancestry[0] is source:
            if ancestry[1] is None or ancestry[1]!=source_hash:gaps.append('ROW_SOURCE_PRODUCER_VERSION_CHANGED')
            else:states.extend(ancestry[2])
            gaps.extend(ancestry[3])
        self.capture._row_membership[id(output)]=(output,selected)
        self.capture._data_state_sources[id(output)]=(output,output_hash,tuple(dict.fromkeys(states)),tuple(dict.fromkeys(gaps)))
        self.bindings[id(output)]={'object':output,'fingerprint':output_hash,'members':selected}
        record={'contract':contract,'source_object':f'py:{id(source):x}','source_version':source_hash,
                'output_object':f'py:{id(output):x}','output_version':output_hash,
                'positions':positions.tolist(),'producer_versions':list(dict.fromkeys(states))}
        self.records.append(record)
        self.capture.emit(EventType.DATA_CREATE,operator='dmcasrc.row_contract',inputs=(source,),outputs=(output,),attributes=record)
        return True

    def filter_frame(self,code,*,source_name,output_name,index_name,contract,owner_class=None):
        def resolve(frame,value):
            import numpy as np
            local=frame.f_locals
            if owner_class is not None and type(local.get('self')) is not owner_class:return False
            if local.get(output_name) is not value:return False
            source=local.get(source_name);index=local.get(index_name)
            if type(source) is not np.ndarray or type(index) is not np.ndarray:return False
            if index.ndim!=1 or index.dtype.kind!='b' or len(index)!=len(source):return False
            return self.index(source,value,np.flatnonzero(index),contract=contract)
        self.resolvers.append((code,resolve))

