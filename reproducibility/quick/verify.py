"""Check a recorded Quick Reproduction result against the frozen expectations."""
import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text(encoding="utf8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", nargs="?", type=Path, default=HERE.parents[1] / "quick-output")
    args = parser.parse_args()
    record = read(args.output / "quick_results.json")
    expected = read(HERE / "expected/manuscript.json")
    environment = read(args.output / "execution_environment.json")
    cases = {x["case_id"]: x for x in record["cases"]}
    if set(cases) != set(expected) or len(record["cases"]) != len(expected) or record["schema_version"] != 1:
        print("FAIL: unexpected case inventory or report format")
        return 1
    valid = record["status"] == "PASS" and environment["one_thread_verified"]
    for key, target in expected.items():
        item = cases[key]
        core_skip = record["mode"] == "core" and key.startswith("C-")
        result = item["status"] == "NOT RUN" if core_skip else (
            item["status"] == "PASS" and item["observed"] == target)
        payload = args.output / "evidence" / f"{key}.json"
        result = result and item["expected"] == target and payload.is_file()
        print(key, "SKIPPED (core mode)" if result and core_skip else ("PASS" if result else "FAIL"))
        valid &= result
    return 0 if valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
