"""Replay frozen graph fixtures against the installed StateScope wheel."""
import argparse
from importlib.metadata import version
import json
from pathlib import Path
from runpy import run_path
import sys
from time import perf_counter
from types import ModuleType
from unittest.mock import patch

import statescope

def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf8") as stream:
        json.dump(data, stream, indent=2)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--archive", type=Path, required=True, help="Authorized local evidence directory.")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if version("statescope") != "0.1.2":
        p.error("Install StateScope 0.1.2 before replaying the frozen plan.")
    common = ModuleType("common")
    common.OUT = a.archive
    common.SITE = Path(statescope.__file__).resolve().parent.parent
    common.read, common.write = read, write
    started = perf_counter()
    with patch.dict(sys.modules, common=common):
        archived = run_path(str(a.archive / "harness/validation/graph_checks.py"))
    plans = read(a.archive / "designs/plans.json")
    actual = []
    comparisons = {}
    for group in ("structured_graph", "generated_graph"):
        executions = [archived["run_one"](plan) for plan in plans[group]]
        serializable = json.loads(json.dumps(executions))
        original = read(a.archive / "published-results" / f"{group}.json")
        comparisons[group] = {"cases": len(executions), "archived_records_equal": serializable == original}
        actual.extend(executions)

    injected = [r for r in actual if r["malformed"] == "incorrect_attested_edge"]
    other = [r for r in actual if r["malformed"] != "incorrect_attested_edge"]
    observed = {
        "total_checks": len(actual),
        "checks_without_wrong_version_injection": len(other),
        "checks_with_wrong_version_injection": len(injected),
        "wrong_version_lower_bound_breaches": sum(not r["lower_sound"] for r in injected),
        "wrong_version_false_violations": sum(r["decision"] == "VIOLATION" and
                                            r["reference_truth"] == "PASS" for r in injected),
        "false_determinate_without_wrong_version_injection": sum(
            r["decision"] != r["reference_truth"] and r["decision"] != "UNKNOWN" for r in other),
    }
    # The 20 deliberate injections are included in the corpus of 660.
    expected = {
        "total_checks": 660, "checks_without_wrong_version_injection": 640,
        "checks_with_wrong_version_injection": 20, "wrong_version_lower_bound_breaches": 20,
        "wrong_version_false_violations": 5, "false_determinate_without_wrong_version_injection": 0,
    }
    unique_ids = len({r["id"] for r in actual}) == len(actual)
    ok = observed == expected and unique_ids and all(r["archived_records_equal"] for r in comparisons.values())
    write(a.output, {"status": "PASS" if ok else "FAIL", "observed": observed,
        "expected": expected, "unique_ids": unique_ids, "elapsed_seconds": perf_counter()-started,
        "comparison": comparisons, "results": actual})
    print(f"{'PASS' if ok else 'FAIL'}: {observed}, expected {expected}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
