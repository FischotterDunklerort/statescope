"""scikit-learn Pipeline native objects mapped to canonical evidence semantics."""

from dmcasrc.adapters._scientific import native_rule

_RULES = (
    native_rule(
        "sklearn-pipeline",
        "row_overlap_include_assessment",
        "ASSESSMENT_FIT_ISOLATION",
        "Fit",
        ("Fit", "Assessment"),
        ("fit.training", "data.assessment", "sample_overlap"),
    ),
    native_rule(
        "sklearn-pipeline",
        "label_horizon_shift",
        "LABEL_TIME_ALIGNMENT",
        "Data",
        ("Data", "Macro"),
        ("data.input", "fit.training", "label_shift"),
    ),
    native_rule(
        "sklearn-pipeline",
        "post_decision_feature",
        "TIME_AVAILABILITY",
        "Data",
        ("Data", "Model", "Selector", "Execution"),
        ("data.input", "model.predict", "time_availability"),
    ),
    native_rule(
        "sklearn-pipeline",
        "include_assessment_in_preprocess_fit",
        "ASSESSMENT_FIT_ISOLATION",
        "Data",
        ("Data",),
        ("data.assessment", "preprocessing.fit", "consumed_by"),
    ),
    native_rule(
        "sklearn-pipeline",
        "fit_hidden_transformer_on_all_rows",
        "ASSESSMENT_FIT_ISOLATION",
        "Data",
        ("Data",),
        None,
        missing_required_evidence=("operator_internal_lineage", "closure_or_cache_dependency_path"),
    ),
    native_rule(
        "sklearn-pipeline",
        "feature_clock_future_nearest",
        "TIME_ALIGNMENT",
        "Data",
        ("Data",),
        ("data.input", "model.predict", "time_alignment"),
    ),
    native_rule(
        "sklearn-pipeline",
        "assessment_in_search_folds",
        "ASSESSMENT_SEARCH_ISOLATION",
        "Model",
        ("Model",),
        ("model.predict", "data.assessment", "hyperparameter_tuning"),
    ),
    native_rule(
        "sklearn-pipeline",
        "include_assessment_in_calibration_fit",
        "ASSESSMENT_CALIBRATION_ISOLATION",
        "Calibration",
        ("Calibration",),
        ("data.assessment", "calibration.fit", "consumed_by"),
    ),
    native_rule(
        "sklearn-pipeline",
        "threshold_from_assessment_outcome",
        "ASSESSMENT_THRESHOLD_ISOLATION",
        "Threshold",
        ("Threshold",),
        ("data.assessment", "execution.order", "threshold_selection"),
    ),
    native_rule(
        "sklearn-pipeline",
        "bind_stale_model_bundle",
        "ARTIFACT_IDENTITY",
        "Identity",
        ("Identity",),
        ("identity.binding", "execution.order", "artifact_binding"),
    ),
    native_rule(
        "sklearn-pipeline",
        "split_by_label_regime",
        "UNIVERSE_AVAILABILITY",
        "DataDistribution",
        ("DataDistribution",),
        None,
        missing_required_evidence=("distributional_test_capability",),
    ),
    native_rule(
        "sklearn-pipeline",
        "claim_stale_artifact",
        "CLAIM_DEPENDENCY_INTEGRITY",
        "Claim",
        ("Claim",),
        ("claim.performance", "execution.order", "artifact_binding_missing"),
    ),
)


def native_fact_rules():
    return _RULES


__all__ = ["native_fact_rules"]
