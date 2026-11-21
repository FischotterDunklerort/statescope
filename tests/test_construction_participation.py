from pathlib import Path
import sys,unittest
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import runtime
from audit import Audit
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_extraction.text import HashingVectorizer

class ConstructionParticipation(unittest.TestCase):
    def test_no_growth_labels_are_observed_but_retention_stays_unestablished(self):
        x=np.array([[1.],[-1.],[2.],[-2.]])
        e=np.array([[3.],[-3.]])
        with Audit(boundary="All label and tree operations observed") as a:
            a.bind_members(x,["a0","a1","a2","a3"]);a.bind_members(e,["e0","e1"])
            m=GradientBoostingClassifier(n_estimators=3,warm_start=True,random_state=71).fit(x,[1,0,1,0])
            m.fit(e,[11,10])
            event=a.capture._state_events[id(m)]
            self.assertEqual(event.attributes["source_ids"],["e0","e1"])
            self.assertEqual(event.attributes["fit_semantics"],"UNESTABLISHED")
            self.assertFalse(a.support(m)["operation_supported"])
            for p in ["prediction","continuation"]:
                self.assertEqual(a.check(m,assessment=e,purpose=p)["decision"],"UNKNOWN")
    def test_stateless_validation_does_not_become_training_membership(self):
        with Audit(boundary="Pure native hashing") as a:
            raw=["first sample","second sample"];a.bind_members(raw,["r0","r1"])
            h=HashingVectorizer(n_features=64).fit(raw)
            support=a.support(h)
            self.assertTrue(support["operation_supported"])
            self.assertFalse(support["members_are_training"])
            event=a.capture._state_events[id(h)]
            self.assertNotIn("source_ids",event.attributes)
if __name__=="__main__":unittest.main()

