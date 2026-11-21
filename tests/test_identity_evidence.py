from pathlib import Path
import sys,gc,weakref,unittest
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import runtime
import numpy as np
from audit import Audit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from dmcasrc.autofact.identity_evidence import IdentityEvidenceMap

class IdentityEvidence(unittest.TestCase):
 def test_dead_object_removed_and_replacement_callback_cannot_remove_new_entry(self):
  m=IdentityEvidenceMap();a=np.zeros(2);b=np.ones(2)
  m[7]=(a,('old',));m[7]=(b,('new',));del a;gc.collect()
  self.assertIs(m[7][0],b);self.assertEqual(m[7][1],('new',))
  w=weakref.ref(b);del b;gc.collect();self.assertIsNone(w());self.assertNotIn(7,m)
 def test_nonweak_objects_remain_identity_bound(self):
  m=IdentityEvidenceMap();a=['text'];m[id(a)]=(a,('T',));self.assertIs(m[id(a)][0],a)
  self.assertEqual(dict(m)[id(a)][1],('T',));m.pop(id(a));self.assertEqual(len(m),0)
 def test_consumed_producer_history_survives_matrix_collection(self):
  t=np.array([[0.,1.],[1.,0.],[2.,1.],[3.,0.]])
  e=t+10;y=np.array([0,1,0,1])
  with Audit() as a:
   a.bind_members(t,['T:'+str(i) for i in range(4)]);a.bind_members(e,['E:'+str(i) for i in range(4)])
   scaler=StandardScaler().fit(e);z=scaler.transform(t)
   m=SGDClassifier(random_state=71,max_iter=5,tol=None).fit(z,y)
   w=weakref.ref(z);key=id(z);del z;gc.collect()
   self.assertIsNone(w());self.assertNotIn(key,a.capture._row_membership);self.assertNotIn(key,a.capture._data_state_sources)
   scaler.fit(t)
   self.assertEqual(a.check(m,assessment=e)['decision'],'VIOLATION')
 def test_live_data_mutation_still_invalidates_consumption(self):
  t=np.array([[0.,1.],[1.,0.],[2.,1.],[3.,0.]]);e=t+10;y=np.array([0,1,0,1])
  with Audit() as a:
   a.bind_members(t,['T:'+str(i) for i in range(4)]);a.bind_members(e,['E:'+str(i) for i in range(4)])
   z=StandardScaler().fit(e).transform(t);z[0,0]+=1
   m=SGDClassifier(random_state=71,max_iter=5,tol=None).fit(z,y)
   self.assertEqual(a.check(m,assessment=e)['decision'],'UNKNOWN')

if __name__=='__main__':unittest.main()
