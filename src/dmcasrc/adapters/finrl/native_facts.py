"""FinRL native objects mapped to canonical evidence semantics."""

from dmcasrc.adapters._scientific import native_rule

_RULES = (
    native_rule(
        "finrl",
        "episode_window_overlap",
        "ASSESSMENT_FIT_ISOLATION",
        "Fit",
        ("Fit", "Assessment"),
        ("fit.training", "data.assessment", "sample_overlap"),
    ),
    native_rule(
        "finrl",
        "reward_target_shift",
        "LABEL_TIME_ALIGNMENT",
        "Data",
        ("Data", "Macro"),
        ("data.input", "fit.training", "label_shift"),
    ),
    native_rule(
        "finrl",
        "future_observation_mix",
        "TIME_AVAILABILITY",
        "Data",
        ("Data", "Model", "Selector", "Execution"),
        ("data.input", "model.predict", "time_availability"),
    ),
    native_rule(
        "finrl",
        "market_clock_future_nearest",
        "TIME_ALIGNMENT",
        "Data",
        ("Data",),
        ("data.input", "model.predict", "time_alignment"),
    ),
    native_rule(
        "finrl",
        "test_env_in_hyperparameter_search",
        "ASSESSMENT_SEARCH_ISOLATION",
        "Model",
        ("Model",),
        ("model.predict", "data.assessment", "hyperparameter_tuning"),
    ),
    native_rule(
        "finrl",
        "select_checkpoint_on_assessment",
        "ASSESSMENT_SELECTION_ISOLATION",
        "Model",
        ("Model", "Claim"),
        None,
        missing_required_evidence=(
            "callback_cache_access_path",
            "indirect_assessment_consumption_lineage",
        ),
    ),
    native_rule(
        "finrl",
        "evaluation_feedback_into_filter",
        "OUTCOME_FEEDBACK_ISOLATION",
        "Risk",
        ("Risk", "Claim"),
        ("risk.rule", "execution.order", "feedback_rule"),
    ),
    native_rule(
        "finrl",
        "use_future_price_in_reward",
        "TIME_AVAILABILITY",
        "RL",
        ("RL", "Timer"),
        ("data.future_price_outcome", "model.reward_fit", "consumed_by"),
    ),
    native_rule(
        "finrl",
        "shift_terminal_alignment",
        "EXECUTION_IDENTITY",
        "RL",
        ("RL", "Execution"),
        ("execution.terminal_boundary", "execution.evaluation", "consumed_by"),
        witness_policy="CLEAN_MINUS_OBSERVED_CANDIDATES",
    ),
    native_rule(
        "finrl",
        "load_stale_checkpoint",
        "MODEL_IDENTITY",
        "Identity",
        ("Identity",),
        ("artifact.wrong_policy_checkpoint", "model.action_producer", "consumed_by"),
    ),
    native_rule(
        "finrl",
        "checkpoint_join_nearest_time",
        "CANDIDATE_IDENTITY",
        "Identity",
        ("Identity",),
        ("identity.binding", "execution.order", "candidate_drift"),
    ),
    native_rule(
        "finrl",
        "claim_wrong_checkpoint",
        "CLAIM_DEPENDENCY_INTEGRITY",
        "Claim",
        ("Claim",),
        ("claim.performance", "execution.order", "artifact_binding_missing"),
    ),
)


def native_fact_rules():
    return _RULES


__all__ = ["native_fact_rules"]
