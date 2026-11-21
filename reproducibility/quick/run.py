"""Execute reviewer cases and recompute the archived manuscript results."""
import argparse
import importlib.metadata as metadata
import json
from pathlib import Path
import os
import platform
import sys
from time import perf_counter
import traceback

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
THREADS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS")
CASES = {
    "A-listing": ("Listing 1, p. 3", "Re-execution", "Retention, replacement, and trusted restoration",
                  "60 synthetic rows, fixed seed 7, two updates then a fresh allowed fit"),
    "B-zero-weight": ("Table 4, p. 12", "Re-execution", "Native participation with zero sample weights",
                      "Binary hinge SGD receives an excluded batch with zero sample weights"),
    "B-purpose": ("Table 5, p. 13", "Re-execution", "Purpose-specific association",
                  "Change the optimizer counter after a permitted fit and probe the next native update"),
    "B-H2": ("Table 7, p. 15", "Re-execution", "Verified historical dependency",
              "Run H2 with its counter transition; then withhold the observed origin while the model remains fixed"),
    "C-table8": ("Table 8, p. 16", "Recomputation", "Rule votes, coverage subsets, and history-level comparisons",
                  "Join archived rule votes and independently recorded source labels by query ID"),
    "C-table16": ("Table 16, p. 24", "Recomputation and evidence inspection",
                   "Cold context entry, matched runtime ratios, and archived evidence equality",
                   "Read the saved process blocks, fitted arrays, predictions, queries, and evidence"),
}


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False)+"\n", encoding="utf8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "quick-output")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--archive", type=Path, help="Authorized local evidence directory for Part C.")
    mode.add_argument("--core-only", action="store_true", help="Run the synthetic cases (default).")
    args = parser.parse_args()
    core_only = args.archive is None
    args.output.mkdir(parents=True, exist_ok=False)
    evidence = args.output / "evidence"
    evidence.mkdir()
    start = perf_counter()
    os.environ.update({key: "1" for key in THREADS})
    os.environ["OMP_DYNAMIC"] = "FALSE"
    os.environ["MKL_DYNAMIC"] = "FALSE"
    import statescope
    import sklearn
    from threadpoolctl import threadpool_info
    import cases
    import recompute
    if statescope.__version__ != "0.1.2" or sklearn.__version__ != "1.8.0":
        parser.error("The quick procedure requires StateScope 0.1.2 and scikit-learn 1.8.0.")
    expected = recompute.read(HERE / "expected" / "manuscript.json")
    actions = {
        "A-listing": cases.listing, "B-zero-weight": cases.zero_weight, "B-purpose": cases.purpose,
        "B-H2": cases.h2,
        "C-table8": lambda: (recompute.table8(args.archive),
                             {"inputs": ["published-results/native_reference_queries.csv",
                                         "published-results/query_set_membership.csv"]}),
        "C-table16": lambda: (recompute.table16(args.archive),
                             {"inputs": ["table16/registered_only_benchmark.csv",
                                         "table16/<workload>/<block>/<mode> process records and arrays"]}),
    }
    results = []
    for case, (reference, method, mechanism, inputs) in CASES.items():
        tick = perf_counter()
        item = dict(case_id=case, manuscript_reference=reference, method=method,
                    mechanism=mechanism, input_condition=inputs, expected=expected[case])
        payload = {}
        if core_only and case.startswith("C-"):
            item.update(observed=None, status="NOT RUN")
        else:
            try:
                observed, payload = actions[case]()
                item.update(observed=observed, status="PASS" if observed == expected[case] else "FAIL")
            except FileNotFoundError:
                item.update(observed=None, status="NOT AVAILABLE")
                payload = {"diagnostic": traceback.format_exc()}
            except Exception:
                item.update(observed=None, status="FAIL")
                payload = {"diagnostic": traceback.format_exc()}
        item["elapsed_seconds"] = round(perf_counter()-tick, 6)
        item["evidence_file"] = f"evidence/{case}.json"
        write(args.output / item["evidence_file"], payload)
        results.append(item)

    pools = threadpool_info()
    thread_ok = bool(pools) and all(item["num_threads"] == 1 for item in pools)
    write(args.output / "execution_environment.json", dict(
        python=sys.version, os=platform.platform(), machine=platform.machine(),
        versions={n: metadata.version(n) for n in ("statescope", "scikit-learn", "numpy", "scipy",
                                                     "joblib", "pandas", "threadpoolctl", "psutil")},
        threadpools=[{k: pool[k] for k in ("user_api", "internal_api", "num_threads", "version")} for pool in pools],
        one_thread_verified=thread_ok))
    ok = thread_ok and all(r["status"] == "PASS" or (core_only and r["status"] == "NOT RUN") for r in results)
    elapsed = round(perf_counter()-start, 6)
    write(args.output / "quick_results.json", dict(schema_version=1, mode="core" if core_only else "all",
        elapsed_seconds=elapsed, status="PASS" if ok else "FAIL", cases=results))
    lines = [
        "# StateScope v0.1.2 quick reproduction", "",
        f"The procedure completed in {elapsed:.2f} seconds on the recorded host. The run status is **{'PASS' if ok else 'FAIL'}**.",
        "Parts A and B execute synthetic native training and audit calls. Part C recomputes saved results when an authorized archive is supplied explicitly.",
        "", "| Case | Manuscript | Method | Result | Seconds | Evidence |",
        "|---|---|---|---|---:|---|",
    ]
    lines += [f"| {r['case_id']} | {r['manuscript_reference']} | {r['method']} | {r['status']} | "
              f"{r['elapsed_seconds']:.3f} | [JSON]({r['evidence_file']}) |" for r in results]
    lines += ["", "The JSON report includes the specified and observed outcomes. An expected UNKNOWN passes a check when the corresponding evidence is intentionally absent. The check results describe these fixtures.", ""]
    (args.output / "quick_report.md").write_text("\n".join(lines), encoding="utf8")
    (args.output / "verification.log").write_text("\n".join(f"{r['case_id']} {r['status']}" for r in results)+"\n", encoding="utf8")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
