"""Check coverage separately from mapping, with both audit purposes."""
from pathlib import Path
import tempfile
import unittest

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from statescope import Audit, EvidenceArchive, ContinuityError

# Same synthetic inputs as the archived producer_replay.py fixture.
EXCLUDED = np.array([[-3., 1.], [-2., 1.2], [1., 0.5], [2., 1.4]])
PERMITTED = np.array([[-2., 1.], [-1., 0.], [1., 0.], [2., 1.]])
Y = np.array([0, 0, 1, 1])
E = [f'origin:{i}' for i in range(4)]
C = [f'permitted:{i}' for i in range(4)]

def model():
    return SGDClassifier(random_state=71,shuffle=False,loss='log_loss')

def query(a, learner, policy=E):
    assessment=np.zeros((len(policy),1))
    a.bind_members(assessment,policy)
    return [a.check(learner,assessment=assessment,purpose=p)
            for p in ('prediction','continuation')]

class StrictSources(unittest.TestCase):
    def expect(self,a,m,expected):
        results=query(a,m)
        self.assertEqual([r['decision'] for r in results],[expected,expected])
        return results

    def test_default_unspecified_and_compatibility_are_distinct(self):
        with Audit(boundary='strict') as a:
            a.bind_members(PERMITTED,C)
            m=model().fit(PERMITTED,Y)
            ans=self.expect(a,m,'UNKNOWN')
            self.assertTrue(all(r['input_producer_coverage']['unresolved'] for r in ans))
        with Audit(boundary='legacy',source_coverage='legacy') as a:
            a.bind_members(PERMITTED,C)
            m=model().fit(PERMITTED,Y)
            self.assertFalse(self.expect(a,m,'PASS')[0]['release_condition_eligible'])

    def test_rebinding_keeps_producer_and_valid_exposure(self):
        with Audit() as a:
            a.bind_members(EXCLUDED,E,source='trusted_raw')
            a.bind_members(PERMITTED,C,source='trusted_raw')
            z=StandardScaler().fit(EXCLUDED).transform(PERMITTED)
            a.bind_members(z,C)
            with self.assertRaises(ValueError):
                a.bind_members(z,C,source='trusted_raw')
            with self.assertRaises(ValueError):
                a.bind_members(z,E)
            self.expect(a,model().fit(z,Y),'VIOLATION')
        with Audit() as a:
            z=PERMITTED.copy()
            a.bind_members(z,C,source='unknown_derived')
            a.bind_members(z,C)
            with self.assertRaises(ValueError):
                a.bind_members(z,C,source='trusted_raw')
            ans=self.expect(a,model().fit(z,Y),'UNKNOWN')
            self.assertIn('EXTERNAL_FEATURE_PRODUCER_UNVERIFIED',ans[0]['input_producer_coverage']['unresolved'])

    def test_plain_copy_slice_and_reviewed_composition(self):
        import sklearn.utils as sk
        for operation in ('copy','slice','index','split','concat','transform'):
            with self.subTest(operation=operation), Audit() as a:
                from sklearn.model_selection import train_test_split
                x=PERMITTED.copy()
                a.bind_members(x,C,source='unknown_derived')
                labels=Y
                ids=C
                if operation=='copy': new=x.copy()
                if operation=='slice': new=x[:, :]
                if operation=='index': new=sk._safe_indexing(x,np.arange(len(x)))
                if operation=='split':
                    new,_,labels,_=train_test_split(x,Y,test_size=.25,shuffle=False)
                    ids=C[:3]
                if operation=='concat': new=np.concatenate([x,x],axis=1)
                if operation=='transform': new=StandardScaler().fit(x).transform(x)
                a.bind_members(new,ids)
                self.expect(a,model().fit(new,labels),'UNKNOWN')

    def test_covered_index_split_and_hashing_are_usable(self):
        import sklearn.utils as sk
        docs=['one bright','two dark','three warm','four cold']
        with Audit() as a:
            from sklearn.model_selection import train_test_split
            a.bind_members(docs,C,source='trusted_raw')
            data,_,labels,_=train_test_split(docs,Y,test_size=.25,shuffle=False)
            x=HashingVectorizer(n_features=32).transform(data)
            indexed=sk._safe_indexing(x,np.arange(3))
            self.expect(a,model().fit(indexed,labels),'PASS')

    def test_caller_cannot_relabel_existing_fit_with_new_declaration(self):
        with Audit() as a:
            z=PERMITTED.copy()
            a.bind_members(z,C)
            learner=model().fit(z,Y)
            self.expect(a,learner,'UNKNOWN')
            a.bind_members(z,C,source='trusted_raw')
            self.expect(a,learner,'UNKNOWN')
            self.expect(a,learner.fit(z,Y),'PASS')

    def test_archive_preserves_requirements_and_trust(self):
        with tempfile.TemporaryDirectory() as d:
            store=EvidenceArchive(d)
            with Audit() as a:
                a.bind_members(PERMITTED,C)
                m=model().fit(PERMITTED,Y)
                with self.assertRaises(ContinuityError):
                    store.export(a,m)
                key=a.save(m,d)
            with Audit() as a:
                loaded=a.load(key,d)
                self.expect(a,loaded,'UNKNOWN')
            with Audit(source_coverage='legacy') as a:
                a.bind_members(PERMITTED,C)
                m=model().fit(PERMITTED,Y)
                old=store.export(a,m)
            with Audit() as a:
                q=np.zeros((4,1))
                a.bind_members(q,E)
                result=a.check_with_recovery(m,archive=store,handle=old,assessment=q)
                self.assertEqual(result['decision'],'UNKNOWN')
                self.assertFalse(result['recovery']['accepted'])
            with Audit() as a:
                a.bind_members(PERMITTED,C,source='trusted_raw')
                m=model().fit(PERMITTED,Y)
                new=store.export(a,m)
            with Audit() as a:
                a.recover_state(m,archive=store,handle=new)
                self.expect(a,m,'PASS')

    def test_bound_overlap_is_not_confirmed(self):
        with Audit() as a:
            q=PERMITTED.copy()
            a.bind_members(q,C)
            a.declare_external_features(q,source_upper_bound=C+E,basis='Conservative fixture provenance')
            m=model().fit(q,Y)
            answers=self.expect(a,m,'UNKNOWN')
            self.assertTrue(all(r['source_bounds']['upper_bound_closed'] for r in answers))
            self.assertTrue(all(not r['source_bounds']['confirmed_excluded_members'] for r in answers))

if __name__=='__main__':
    unittest.main(verbosity=2)
