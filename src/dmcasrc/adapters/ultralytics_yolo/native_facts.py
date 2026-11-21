"""Evidence-only mappings from Ultralytics-native facts to frozen semantics.

The rules identify canonical obligation/layer mappings. They never inspect an
outcome and never assert PASS/VIOLATION/claim status; the DM-CASRC core owns
that evaluation.
"""

from dmcasrc.adapters._scientific import native_rule


_RULES = (
    native_rule(
        "ultralytics_yolo",
        "assessment_image_in_fit_manifest",
        "ASSESSMENT_FIT_ISOLATION",
        "Fit",
        ("Fit", "Assessment"),
        ("data.assessment", "model.train", "assessment_consumed_by_fit"),
    ),
    native_rule(
        "ultralytics_yolo",
        "assessment_used_for_model_selection",
        "ASSESSMENT_SEARCH_ISOLATION",
        "Assessment",
        ("Assessment", "Selector"),
        ("data.assessment", "model.selection", "assessment_consumed_by_selection"),
    ),
    native_rule(
        "ultralytics_yolo",
        "assessment_used_for_confidence_threshold",
        "ASSESSMENT_THRESHOLD_ISOLATION",
        "Threshold",
        ("Threshold", "Assessment"),
        ("data.assessment", "threshold.selection", "assessment_consumed_by_threshold"),
    ),
    native_rule(
        "ultralytics_yolo",
        "checkpoint_binding_mismatch",
        "MODEL_IDENTITY",
        "Model",
        ("Model", "Identity"),
        ("model.checkpoint", "prediction.inference", "model_binding"),
    ),
    native_rule(
        "ultralytics_yolo",
        "image_candidate_identity_drift",
        "CANDIDATE_IDENTITY",
        "Identity",
        ("Identity", "Data"),
        ("data.assessment", "prediction.inference", "candidate_identity"),
    ),
    native_rule(
        "ultralytics_yolo",
        "train_serving_distribution_skew",
        "UNIVERSE_AVAILABILITY",
        "DataDistribution",
        ("DataDistribution",),
        None,
        missing_required_evidence=("distributional_test_capability",),
    ),
    native_rule(
        "ultralytics_yolo",
        "claim_prediction_artifact_mismatch",
        "CLAIM_DEPENDENCY_INTEGRITY",
        "Claim",
        ("Claim", "Artifact"),
        ("artifact.registry", "claim.report", "claim_artifact_binding"),
    ),
)


def native_fact_rules():
    return _RULES


__all__ = ["native_fact_rules"]
