# Artifact Evaluation

## Artifact Overview

StateScope audits which data and fitted dependencies entered a model's construction. This package provides the v0.1.2 implementation, tests, synthetic reviewer cases, and an index to the manuscript evidence.

## Scope of Evaluation

Parts A and B re-execute selected native training and audit calls. Part C is an optional recomputation and evidence inspection using authorized local records. The broader study also examines controlled evidence omission, public workflows, and resource costs.

## System Requirements

The validated configuration uses Windows, Python 3.13.9, scikit-learn 1.8.0, and a CPU. The synthetic cases run offline after the dependencies are installed. They require neither an external service nor an evidence archive.

## Installation

Follow the [installation instructions](REPRODUCIBILITY.md) and run `python -m pip install -r reproducibility/requirements-quick.txt .` using a fresh virtual environment. The offline instructions use a wheelhouse built for the intended platform.

## Quick Reproduction

From the source directory, run these commands with the installed environment.

```text
python reproducibility/quick/run.py
python reproducibility/quick/verify.py
```

The default checks cover Listing 1, zero-weight solver participation, the counter intervention, and the H2 history. The command writes case evidence and actual outcomes under `quick-output/`. A required mismatch returns a nonzero exit code. Use a new output path on a repeat.

## Expected Results

Listing 1 gives `VIOLATION`, `PASS`, `UNKNOWN`, and `PASS` at its four checkpoints. `PASS` requires a valid association and verified source coverage disjoint from the excluded rows. `VIOLATION` requires verified excluded participation. `UNKNOWN` means that the available evidence supports neither conclusion. The checks rely on truthful inputs and source declarations and reviewed operation contracts. An expected abstention can pass the test while remaining an abstention in the research result.

## Reproducing the Full Evaluation

Use [the full run instructions](reproducibility/full/run_instructions.md) when the archived records and dependencies are separately available. Pass an explicit `--archive PATH` to recompute Table 8 and Table 16. Read the reported methods to distinguish recomputation from re-execution. Distributed and external-tool campaigns need their original environments.

## Mapping to the Manuscript

The [paper-to-artifact map](reproducibility/evidence/PAPER_TO_ARTIFACT_MAP.md) uses the 32-page Revised manuscript and records the verification scope of each experiment.

## Known Limitations

The quick cases verify selected fixtures. The graph replay checks inference over supplied facts. Controlled omissions and field changes generate related queries. Their abstention rates describe the intervention conditions, while normal execution has its own coverage denominators. Distributed Dask and cross-script STREAMLINE leave fitted-model associations unresolved in the evaluated workflows.

## Artifact Contents

`src/` contains the unchanged runtime. `examples/`, `tests/`, and `reproducibility/` contain the example, regression suite, and reproduction tools. `artifacts/manifests/` describes the reviewed source package. The companion evidence has a separate distribution status and remains outside this package.
