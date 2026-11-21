"""Qlib native objects mapped to canonical evidence semantics."""

from dmcasrc.adapters._scientific import native_rule

_RULES = (
    native_rule(
        "qlib",
        "fit_window_overlaps_test",
        "ASSESSMENT_FIT_ISOLATION",
        "Fit",
        ("Fit", "Assessment"),
        ("fit.training", "data.assessment", "sample_overlap"),
    ),
    native_rule(
        "qlib",
        "label_horizon_shift",
        "LABEL_TIME_ALIGNMENT",
        "Data",
        ("Data", "Macro"),
        ("data.input", "fit.training", "label_shift"),
    ),
    native_rule(
        "qlib",
        "expand_handler_fit_scope",
        "ASSESSMENT_FIT_ISOLATION",
        "Data",
        ("Data",),
        ("data.assessment", "preprocessing.fit", "consumed_by"),
    ),
    native_rule(
        "qlib",
        "future_universe_members",
        "UNIVERSE_AVAILABILITY",
        "Data",
        ("Data", "Selector"),
        ("data.input", "selector.decision", "universe_leakage"),
    ),
    native_rule(
        "qlib",
        "factor_clock_future_nearest",
        "TIME_ALIGNMENT",
        "Data",
        ("Data",),
        ("data.input", "model.predict", "time_alignment"),
    ),
    native_rule(
        "qlib",
        "assessment_in_hyperparameter_search",
        "ASSESSMENT_SEARCH_ISOLATION",
        "Model",
        ("Model",),
        ("model.predict", "data.assessment", "hyperparameter_tuning"),
    ),
    native_rule(
        "qlib",
        "threshold_from_assessment_outcome",
        "ASSESSMENT_THRESHOLD_ISOLATION",
        "Threshold",
        ("Threshold",),
        ("data.assessment", "execution.order", "threshold_selection"),
    ),
    native_rule(
        "qlib",
        "rank_on_future_return",
        "TIME_AVAILABILITY",
        "Selector",
        ("Selector",),
        ("data.future_return_label", "selector.signal_ranking", "consumed_by"),
    ),
    native_rule(
        "qlib",
        "result_feedback_into_filter",
        "OUTCOME_FEEDBACK_ISOLATION",
        "Risk",
        ("Risk", "Claim"),
        ("risk.rule", "execution.order", "feedback_rule"),
    ),
    native_rule(
        "qlib",
        "selection_execution_population_mismatch",
        "SELECTION_EXECUTION_CONSISTENCY",
        "Selector",
        ("Selector", "Timer"),
        ("selector.decision", "execution.order", "population_mismatch"),
    ),
    native_rule(
        "qlib",
        "split_by_market_regime",
        "UNIVERSE_AVAILABILITY",
        "DataDistribution",
        ("DataDistribution",),
        None,
        missing_required_evidence=("distributional_test_capability",),
    ),
    native_rule(
        "qlib",
        "claim_wrong_model_artifact",
        "CLAIM_DEPENDENCY_INTEGRITY",
        "Claim",
        ("Claim",),
        ("claim.performance", "execution.order", "artifact_binding_missing"),
    ),
)


def native_fact_rules():
    return _RULES


__all__ = ["native_fact_rules"]
