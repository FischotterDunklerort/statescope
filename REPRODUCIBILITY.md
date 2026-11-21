# Reproducibility

## Install

Use Python 3.13.9 with the [pinned reviewer dependencies](reproducibility/requirements-quick.txt). Install from this source directory with `python -m pip install -r reproducibility/requirements-quick.txt .` in a fresh virtual environment. For an offline installation, obtain the platform-compatible dependency wheels in advance and use the reviewed wheel.

```text
python -m pip download --only-binary=:all: -r reproducibility/requirements-quick.txt -d wheelhouse
python -m pip install --no-index --find-links wheelhouse -r reproducibility/requirements-quick.txt PATH_TO_REVIEWED_WHEEL
python -m pip check
```

The first command prepares the wheelhouse and can run on a network-enabled machine. Building the source offline also requires setuptools 77.0.3 or later and wheel. The validation uses setuptools 80.9.0 and wheel 0.45.1.

## Execute the reviewer checks

Run `python reproducibility/quick/run.py` and then `python reproducibility/quick/verify.py` from the repository root. The default exercises four synthetic cases. Table 8 and Table 16 are reported as `NOT RUN` in this mode. Run `python -m unittest discover -s tests -v` for the separate regression suite.

Use `--output PATH` for a fresh output directory on a repeat and pass it to the verifier. To also recompute the archived results, use `python reproducibility/quick/run.py --archive PATH --output quick-with-archive` and then `python reproducibility/quick/verify.py quick-with-archive`. The archive must be separately available and authorized for local use. A required raw record that is absent produces `NOT AVAILABLE` and a nonzero exit code.

The [full instructions](reproducibility/full/run_instructions.md) and [paper map](reproducibility/evidence/PAPER_TO_ARTIFACT_MAP.md) describe the archived experiments. The [readiness report](RELEASE_READINESS_REPORT.md) summarizes the staged verification.
