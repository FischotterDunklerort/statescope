"""Read-only recomputation from the archived query and process records."""
import csv
from collections import Counter, defaultdict
import gzip
import json
import re
from statistics import median

import numpy as np

RULES = ("whole", "field", "direct", "full", "explicit")
DECISIONS = ("PASS", "VIOLATION", "UNKNOWN")


def rows(path):
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def table8(archive):
    frozen = archive / "published-results"
    measurements = rows(frozen / "native_reference_queries.csv")
    views = defaultdict(dict)
    for row in measurements:
        query, rule = row["query_id"], row["rule"]
        if (rule in views[query] or row["decision"] not in DECISIONS or
                row["ground_truth"] not in DECISIONS[:2]):
            raise ValueError(f"Repeated or invalid rule output for {query}/{rule}")
        views[query][rule] = row
    for query, bundle in views.items():
        if set(bundle) != set(RULES) or len({(v["history"], v["ground_truth"]) for v in bundle.values()}) != 1:
            raise ValueError(f"Unpaired policy or source label for {query}")

    outcomes = {}
    for rule in RULES:
        votes = [view[rule] for view in views.values()]
        counts = Counter(row["decision"] for row in votes)
        outcomes[rule] = {status: counts[status] for status in DECISIONS}
        outcomes[rule]["false_determinate"] = sum(
            row["decision"] != "UNKNOWN" and row["decision"] != row["ground_truth"] for row in votes
        )

    subsets = {}
    flags = rows(frozen / "query_set_membership.csv")
    if len({r["query_id"] for r in flags}) != len(flags):
        raise ValueError("Duplicate membership query IDs")
    for row in flags:
        src = views[row["query_id"]]["full"]
        if any(src[k] != row[k] for k in ("decision", "history", "purpose")):
            raise ValueError(f"Membership disagrees with the query record for {row['query_id']}")
    previous = set(views)
    for flag in ("Q354", "Q152", "Q146"):
        subset = {r["query_id"] for r in flags if r[flag] == "True"}
        if not subset < previous:
            raise ValueError(f"{flag} fails strict nesting")
        previous = subset
        subsets[flag] = {
            "determinate": sum(views[k]["full"]["decision"] != "UNKNOWN" for k in subset),
            "queries": len(subset), "histories": len({views[k]["full"]["history"] for k in subset}),
        }

    by_history = defaultdict(set)
    exclusive = []
    for bundle in views.values():
        full = bundle["full"]
        for rule, check in bundle.items():
            if rule != "explicit" and check["decision"] == "VIOLATION":
                by_history[full["history"]].add(rule)
        if full["decision"] == "VIOLATION" and all(bundle[k]["decision"] == "UNKNOWN"
                                                    for k in RULES[:3]):
            exclusive.append(full)
    return {
        "histories": len({x["history"] for x in measurements}), "derived_queries": len(views),
        "rule_outcomes": outcomes, "subsets": subsets,
        "explicit_agreement": sum(v["full"]["decision"] == v["explicit"]["decision"] for v in views.values()),
        "full_vs_whole_extra_exposure": sum(v["full"]["decision"] == "VIOLATION" and
                                           v["whole"]["decision"] != "VIOLATION" for v in views.values()),
        "full_vs_direct_extra_exposure": sum(v["full"]["decision"] == "VIOLATION" and
                                            v["direct"]["decision"] != "VIOLATION" for v in views.values()),
        "full_only_queries": len(exclusive),
        "histories_with_full_only_queries": len({q["history"] for q in exclusive}),
        "histories_exposed_only_by_full_across_views": sum(rules == {"full"} for rules in by_history.values()),
        "field_pass_full_unknown": sum(v["field"]["decision"] == "PASS" and
                                      v["full"]["decision"] == "UNKNOWN" for v in views.values()),
    }


def canonicalize(value):
    """Rename process-local addresses while preserving reference relationships."""
    pattern = re.compile(r"py:[a-f0-9]+")
    seen = {}
    def visit(item):
        if isinstance(item, dict):
            return {key: visit(val) for key, val in sorted(item.items()) if key != "wall_time"}
        if isinstance(item, list):
            return [visit(val) for val in item]
        if isinstance(item, str):
            return pattern.sub(lambda m: seen.setdefault(m[0], f"object:{len(seen)+1}"), item)
        return item
    return visit(value)


def archived_run(archive, row):
    path = archive / row["app"] / f"{int(row['block']):02}" / row["mode"]
    raw = read(path / "attempt.json")
    for key in ("app", "block", "mode", "status", "attempt", "position", "exit_code",
                "launch_to_task_end_s", "context_entry_s", "peak_working_set_mib"):
        if row[key] and str(raw.get(key)) != row[key]:
            raise ValueError(f"Index disagrees with process record for {path.name}/{key}")
    return path, raw


def native_arrays(a, b):
    with np.load(a[0] / a[1]["numeric_report"], allow_pickle=False) as left, \
         np.load(b[0] / b[1]["numeric_report"], allow_pickle=False) as right:
        if set(left.files) != set(right.files) or not any("prediction" in x for x in left.files):
            raise ValueError("Expected comparable saved predictions and fitted arrays")
        return all(np.array_equal(left[k], right[k]) for k in left.files)


def evidence(run, app):
    path, row = run
    raw = read(path / row["source_report"])
    if app == "reviews":
        with gzip.open(path / "observed.json.gz", "rt", encoding="utf8") as f:
            events = json.load(f)
        answers = raw["queries"]
    else:
        events = read(path / "audit_evidence.json")
        answers = [raw[k] for k in ("prediction", "continuation")]
    return canonicalize({"events": events, "answers": answers})


def table16(archive):
    archive = archive / "table16"
    measurements = rows(archive / "registered_only_benchmark.csv")
    grouped = defaultdict(dict)
    for row in measurements:
        key = (row["app"], int(row["block"]))
        if row["mode"] in grouped[key] or row["status"] != "valid" or row["attempt"] != "1" or row["exit_code"] != "0":
            raise ValueError(f"Invalid or duplicate formal attempt for {key}")
        grouped[key][row["mode"]] = archived_run(archive, row)
    launch_ratio = {"full": [], "registered_only": []}
    cold = {"full": [], "registered_only": []}
    q_ok, q_n, e_ok, e_n = (0, 0, 0, 0)
    numeric_ok = []
    for (app, block), modes in grouped.items():
        if set(modes) != ({"full", "registered_only", "bare"} if app == "reviews" else {"full", "registered_only"}):
            raise ValueError(f"Incomplete block for {app}/{block}")
        full, opt = modes["full"], modes["registered_only"]
        if full[1]["scan_calls"] != 1 or opt[1]["scan_calls"] != 0:
            raise ValueError(f"Wrong discovery mode in {app}/{block}")
        numeric_ok.append(native_arrays(full, opt))
        if app == "reviews":
            numeric_ok.append(native_arrays(full, modes["bare"]))
            bare = modes["bare"][1]["launch_to_task_end_s"]
            for mode in launch_ratio:
                launch_ratio[mode].append(modes[mode][1]["launch_to_task_end_s"] / bare)
        else:
            for mode in cold:
                cold[mode].append(modes[mode][1]["context_entry_s"])
        original, candidate = evidence(full, app), evidence(opt, app)
        qa, qb = original["answers"], candidate["answers"]
        ea, eb = original["events"], candidate["events"]
        q_ok += sum(x == y for x, y in zip(qa, qb))
        e_ok += sum(x == y for x, y in zip(ea, eb))
        q_n += max(len(qa), len(qb))
        e_n += max(len(ea), len(eb))

    return {
        "blocks": dict(Counter(k[0] for k in grouped)),
        "executions": len(measurements),
        "cold_context_seconds": {k: round(median(v), 3) for k, v in cold.items()},
        "reviews_runtime_ratio": {k: round(median(v), 2) for k, v in launch_ratio.items()},
        "audit_queries": [q_ok, q_n], "evidence_entries": [e_ok, e_n],
        "prediction_and_fitted_arrays": [sum(numeric_ok), len(numeric_ok)],
    }
