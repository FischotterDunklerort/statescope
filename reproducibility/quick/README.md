# Quick Reproduction

From the repository root, run `python reproducibility/quick/run.py` and `python reproducibility/quick/verify.py` using the installed environment. The default runs Parts A and B offline and reports Part C as `NOT RUN`. An explicit `--archive PATH` adds the archived checks. The archive is outside the public source package. Choose a fresh output directory on a repeat and pass that directory to the verifier.

Part A runs Listing 1. Part B examines the zero-weight native call, a purpose-specific state check, and the H2 dependency. In H2 the fit is held fixed while origin evidence is withdrawn. Part C reads archived records to recompute the Table 8 results and Table 16 comparisons. It checks the saved arrays and evidence with the recorded outputs. It does not repeat the performance campaign.

The runner saves `quick_results.json`, `quick_report.md`, `execution_environment.json`, `verification.log`, and per-case JSON evidence. The expected outcomes in `expected/manuscript.json` support the checks. A failure retains the observed result.
