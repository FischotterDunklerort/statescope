# StateScope

StateScope v0.1.2 audits whether excluded rows participated in the construction of a fitted scikit-learn model. An update on permitted rows can still use coefficients, transformed features, or trees learned earlier. StateScope observes reviewed operations and checks whether that evidence applies to the model selected for prediction or continued training.

A verified, disjoint source upper bound supports `PASS`. A verified excluded dependency supports `VIOLATION`. Insufficient evidence leads to `UNKNOWN`. These decisions depend on authentic row identities and source declarations, accurate operation contracts, and a valid association with the current model.

## Install and run

Use Python 3.13.9 in an isolated environment. On Windows, run the following commands from this directory.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r reproducibility/requirements-quick.txt .
.\.venv\Scripts\python.exe reproducibility/quick/run.py
.\.venv\Scripts\python.exe reproducibility/quick/verify.py
```

The default command re-executes four synthetic cases, including [Listing 1](examples/listing_1.py). It uses the installed package and has no evidence-archive dependency. The [offline installation instructions](REPRODUCIBILITY.md) explain how to prepare a local wheelhouse. The [artifact evaluation guide](ARTIFACT_EVALUATION.md) explains what each check establishes.

## Research evidence

[Supported operations](docs/supported_operations.md) and [limitations](docs/limitations.md) describe the reviewed profiles. The [paper-to-artifact map](reproducibility/evidence/PAPER_TO_ARTIFACT_MAP.md) indexes the full evaluation. Table 8 and Table 16 can be recomputed from authorized local records using `--archive PATH`. The public source package includes the verification tools and manuscript expectations. The original experiment records and data have a separate distribution status.

## Citation and license

Cite StateScope v0.1.2 using [CITATION.cff](CITATION.cff). SIQI LI licenses the StateScope software under the [Apache License 2.0](LICENSE). See [NOTICE](NOTICE) for the copyright notice. Third-party data, raw records, and derived evidence have separate permissions.
