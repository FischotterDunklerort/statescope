"""Small re-executions of manuscript fixtures on the installed v0.1.2 runtime."""
from copy import deepcopy
from pathlib import Path
from runpy import run_path
from unittest.mock import patch

import numpy as np
from sklearn.linear_model import SGDClassifier
from statescope import Audit
from dmcasrc.autofact import witnesses

from native_boundaries import execute as boundary
from primary_workflows import execute as primary

ROOT = Path(__file__).resolve().parents[2]
USES = ("prediction", "continuation")


def listing():
    result = run_path(str(ROOT / "examples/listing_1.py"))
    names = ("retained", "replaced", "missing", "recovered")
    return {name: result[name] for name in names}, {
        "rows": 60, "permitted_rows": 40, "excluded_rows": 20, "seed": 7,
        "restored_same_live_estimator": True,
    }


class StrictInputs:
    """Supply the authentic raw declarations required by the frozen wheel."""
    def __init__(self, audit):
        self.audit = audit

    def bind_members(self, values, members):
        self.audit.bind_members(values, members, source="trusted_raw")

    def check(self, *args, **kwargs):
        return self.audit.check(*args, **kwargs)


def zero_weight():
    native = boundary("sgd-zero-weight")
    # Use the fixture's same declarations for the training arrays and the separate
    # identity-only declaration for its abstract assessment view.
    with Audit(boundary="All synthetic fitting is observed") as audit:
        class Adapter(StrictInputs):
            def bind_members(self, values, members):
                if values.shape == (2, 1):
                    self.audit.bind_members(values, members)
                else:
                    super().bind_members(values, members)

        observed = boundary("sgd-zero-weight", Adapter(audit))
    checked = {k: value for k, value in observed.items() if k != "audit"} == native
    entries = native["solver_entries"]
    native_facts = {
        "identical_native_trace": checked,
        "excluded_batch_reaches_binary_solver": len(entries) == 2 and entries[-1]["X"] == [[10., 0.], [-10., 0.]],
        "weights_zero": entries[-1]["sample_weight"] == [0., 0.],
        "coefficients_and_intercept_equal": native["coefficient_equal"] and native["intercept_equal"],
        "counter_delta": native["counter_delta"],
    }
    decisions = {use: row["decision"] for use, row in observed["audit"].items()}
    return {**decisions, **native_facts}, {"native": native, "observed": observed}


def purpose():
    # The paper's intervention changes t_ after fitting. Compare the next native
    # update as well as scores, independently of the audit verdict.
    x = np.array([[1., 0.], [-1., 0.], [2., 1.], [-2., -1.]])
    y = np.array([1, 0, 1, 0])
    assess = np.zeros((2, 2))
    model = SGDClassifier(random_state=71, loss="log_loss", learning_rate="optimal", shuffle=False)
    with Audit(boundary="Observed synthetic SGD") as audit:
        audit.bind_members(x, ["clean:0", "clean:1", "clean:2", "clean:3"], source="trusted_raw")
        audit.bind_members(assess, ["excluded:0", "excluded:1"])
        model.partial_fit(x, y, classes=[0, 1])
        before = {u: audit.check(model, assessment=assess, purpose=u) for u in USES}
        clean_copy = deepcopy(model)
        reference = model.decision_function(x)
        model.t_ += 4096
        unchanged = bool(np.array_equal(reference, model.decision_function(x)))
        after = {u: audit.check(model, assessment=assess, purpose=u) for u in USES}
    clean_copy.partial_fit(x, y)
    model.partial_fit(x, y)
    changed = not (np.array_equal(clean_copy.coef_, model.coef_) and
                   np.array_equal(clean_copy.intercept_, model.intercept_))
    return {
        "before": {k: v["decision"] for k, v in before.items()},
        "after": {k: v["decision"] for k, v in after.items()},
        "scoring_unchanged": unchanged, "next_update_changes_parameters": changed,
    }, {"baseline": before, "counter_edit": after, "seed": 71}


def h2():
    native = primary("retained_counter")
    with Audit(boundary="H2 synthetic fit boundary") as audit:
        class Adapter(StrictInputs):
            bind = StrictInputs.bind_members

            def observe(self, model):
                pass  # The frozen schedule gives the observer the same native calls.

            def query(self, model, excluded):
                assessment = np.zeros((len(excluded), 1))
                audit.bind_members(assessment, excluded)
                def decide():
                    return {u: audit.check(model, assessment=assessment, purpose=u) for u in USES}

                full = decide()
                with patch.object(witnesses, "find_witness", return_value=None):
                    whole = decide()
                origin = min(full["prediction"]["valid_witness"]["source_event"],
                             full["continuation"]["valid_witness"]["source_event"])
                events = list(audit.capture.events)
                assert any(event.event_id == origin for event in events)
                # Withdraw only the observed origin after fitting. The live state
                # and source truth are unchanged.
                with patch.object(audit.capture, "_events",
                        [event for event in events if event.event_id != origin]):
                    missing = decide()
                return {"full": full, "whole": whole, "origin_omitted": missing,
                        "omitted_event": origin}

        observed = primary("retained_counter", Adapter(audit))
    queries = observed.pop("queries")
    same = all(native[k] == observed[k] for k in ("native_calls", "state", "excluded", "actual_sources"))
    decisions = {rule: {u: queries[rule][u]["decision"] for u in USES}
                 for rule in ("full", "whole", "origin_omitted")}
    return {**decisions, "native_fit_and_state_match": same, "native_source": native["source_truth"]}, {
        "native": native, "audited": observed, "queries": queries,
    }
