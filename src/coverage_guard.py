"""Detect known uncovered top-level encoding calls in a cooperative context.

The guard conservatively invalidates context coverage. It does not infer a
selected-output dependency for the unmodeled operation or certify all Python.
"""
import functools,inspect,sys
class CoverageGuard:
    def __init__(self,cap):self.cap=cap;self.patches=[]
    def __enter__(self):
        from sklearn.preprocessing import LabelEncoder
        self.cap._unsupported_context_operations=[]
        self.cap._reporting_encoder_operations=[]
        import sklearn
        from sklearn.metrics import _classification
        reporting_codes={inspect.unwrap(getattr(_classification,n)).__code__ for n in
            ('multilabel_confusion_matrix','matthews_corrcoef','hinge_loss')} if sklearn.__version__=='1.8.0' else set()
        for name in ['fit','fit_transform']:
            old=getattr(LabelEncoder,name)
            def factory(native,method):
                @functools.wraps(native)
                def call(value,*args,**kwargs):
                    outer=not any(self.cap._fit_depth.values())
                    caller=sys._getframe(1)
                    # These exact native functions instantiate a local encoder
                    # and return counts/scalars, never that fitted encoder.
                    reporting=(type(value) is LabelEncoder and method=='fit' and caller.f_code in reporting_codes
                               and any(caller.f_locals.get(n) is value for n in ('le','lb')))
                    del caller
                    result=native(value,*args,**kwargs)
                    if outer and reporting:
                        self.cap._reporting_encoder_operations.append({'operation':'LabelEncoder.'+method,'scope':'reviewed sklearn-1.8 metric-local encoder; no fitted object returned'})
                    elif outer:
                        members=self.cap._known_members(args[0]) if args else None
                        self.cap._unsupported_context_operations.append({'operation':'LabelEncoder.'+method,'object_id':hex(id(value)),'input_member_count':len(members) if members is not None else None,'scope':'context coverage; exact output dependence not inferred'})
                    return result
                return call
            setattr(LabelEncoder,name,factory(old,name));self.patches.append((LabelEncoder,name,old))
        return self
    def __exit__(self,*args):
        for owner,name,old in reversed(self.patches):setattr(owner,name,old)
