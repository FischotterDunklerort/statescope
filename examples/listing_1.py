from tempfile import TemporaryDirectory
from sklearn.datasets import make_classification
from sklearn.linear_model import SGDClassifier
from statescope import Audit, EvidenceArchive

X, y = make_classification(60, 4, n_redundant=0, random_state=7)
t, e, yt, ye = X[:40], X[40:], y[:40], y[40:]
excluded = e
# The v0.1.2 API uses 'assessment' for the excluded rows.
query = dict(assessment=excluded, purpose='prediction')

with TemporaryDirectory(dir='.') as path:
    archive = EvidenceArchive(path)
    with Audit(boundary='All fitting is observed') as a:
        for name, rows in [('t', t), ('e', e)]:
            ids = [f'{name}:{i}' for i in range(len(rows))]
            a.bind_members(rows, ids, source='trusted_raw')
        model = SGDClassifier(random_state=7)
        model.partial_fit(e, ye, classes=[0, 1])
        model.partial_fit(t, yt)
        retained = a.check(model, **query)['decision']
        model.fit(t, yt)
        replaced = a.check(model, **query)['decision']
        handle = archive.export(a, model)

    with Audit(boundary='Same observed object') as a:
        a.bind_members(e, [f'e:{i}' for i in range(20)])
        missing = a.check(model, **query)['decision']
        recovered = a.check_with_recovery(
            model, archive=archive, handle=handle, **query
        )['decision']

print(retained, replaced, missing, recovered)
