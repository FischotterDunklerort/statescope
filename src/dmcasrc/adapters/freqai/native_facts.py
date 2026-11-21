"""FreqAI native objects mapped to canonical evidence semantics."""

from dmcasrc.adapters._scientific import native_rule

_RULES = (
    native_rule(
        "freqai",
        "train_backtest_window_overlap",
        "ASSESSMENT_FIT_ISOLATION",
        "Fit",
        ("Fit", "Assessment"),
        ("fit.training", "data.assessment", "sample_overlap"),
    ),
    native_rule(
        "freqai",
        "label_horizon_shift",
        "LABEL_TIME_ALIGNMENT",
        "Data",
        ("Data", "Macro"),
        ("data.input", "fit.training", "label_shift"),
    ),
    native_rule(
        "freqai",
        "add_future_feature",
        "TIME_AVAILABILITY",
        "Data",
        ("Data", "Model", "Selector"),
        ("data.future_feature", "preprocessing.feature_engine", "consumed_by"),
    ),
    native_rule(
        "freqai",
        "assessment_in_hyperparameter_selection",
        "ASSESSMENT_SEARCH_ISOLATION",
        "Model",
        ("Model",),
        ("model.predict", "data.assessment", "hyperparameter_tuning"),
    ),
    native_rule(
        "freqai",
        "threshold_from_assessment_outcome",
        "ASSESSMENT_THRESHOLD_ISOLATION",
        "Threshold",
        ("Threshold",),
        ("data.assessment", "execution.order", "threshold_selection"),
    ),
    native_rule(
        "freqai",
        "join_nearest_future",
        "CANDIDATE_IDENTITY",
        "Identity",
        ("Identity",),
        ("identity.unstable_join", "candidate.ledger", "consumed_by"),
    ),
)


def native_fact_rules():
    return _RULES


__all__ = ["native_fact_rules"]
