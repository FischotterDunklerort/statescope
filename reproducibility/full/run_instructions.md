# Full evaluation run instructions

Use the 32-page Revised manuscript to interpret the experiment results. The [map](../evidence/PAPER_TO_ARTIFACT_MAP.md) identifies each experiment's data, runner, archive paths, expected outcome, and verification status. Check the companion checksums before using the files. Run each historical campaign into a fresh copy of its study tree and preserve the recorded outputs.

## Native execution and graph checks

The quick procedure exercises Listing 1, the native zero-weight case, the purpose-specific counter edit, and the H2 schedule. With separately authorized evidence, the command below re-executes the frozen graph plan against the installed wheel. The 660 checks include 640 without an injected wrong-version edge and 20 with that deliberate injection. Replay verifies both counts and every archived record.

```text
python reproducibility/full/replay_graph.py --archive PATH_TO_AUTHORIZED_ARCHIVE --output graph-replay.json
```

The authorized local replay directory contains `harness/validation/graph_checks.py`, `designs/plans.json`, `published-results/`, and `table16/`. It supports graph replay and the optional Part C checks. The broader historical archive remains separate. The larger native-reference campaign used `reproduce_experiments.py science` with the frozen plan, `native_checks.py`, and the independent explicit adapter. The script expects its original project layout and the retained contract and reference fixtures. The original operator, bounded graph, parameter and purpose controls use `supplemental.py PROJECT inventory SCHEDULE PURPOSE`, `supplemental.py PROJECT dag`, `supplemental.py PROJECT purpose`, and `imports.py PROJECT`. Set the original workspace paths in a local copy of the runner configuration. Use the release wheel and the pinned numerical dependencies. Restoring every original input and rerunning these campaigns remains pending.

## Public workflows and external tools

Restore the input snapshots identified by the archived provenance records and tool commit list. The research runner used `science.py PROJECT` for workflow and controlled arms, `supplemental.py PROJECT taxi` for the Taxi adapter and negative controls, `public_matrix.py PROJECT` for the 36 common controlled cases, and `fixed_summary.py` for the matched omissions. The public snapshots have separate rights and input requirements. Distribute only cleared materials. Dask's distributed worker trace and synchronous timing correspond to different executions.

The external comparisons require the tools and their pinned compatible environments. See `public/external-tools/commits.csv` for commits and the archived JSON for the exact shared tasks. The noWorkflow script adaptations and NBSafety ordered-cell cases report different types of evidence. The static Yang analyzer was outside the executed Windows comparison. These tools were not rerun in release staging.

## Performance

The End-to-End Benchmark used the `table13` phase of the validation runner. Its 210 archived accepted process durations sum to 8,502.64 seconds, excluding between-run waits, setup and rejected attempts. The encoding-only scale sweep used the `scaling` phase. Its 150 accepted durations sum to 886.83 seconds. The buffered file-persistence results use the archived `runtime_measure.py` protocol. Use the prescribed thread settings, matched process ordering, and hardware-specific working-set metric.

Cold-start timing uses the experimental discovery variant in the separate archive. Its `benchmark.py run OUTPUT 15` command needs the original workflow tree, adapters and experimental code. The release wheel has full discovery. Part C checks the archived mode comparison without repeating the Windows performance campaigns. Keep the Table 13 post-startup window separate from the Table 16 launch-to-completion window.
