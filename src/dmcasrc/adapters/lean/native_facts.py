"""LEAN native objects mapped to canonical evidence semantics."""

from dmcasrc.adapters._scientific import native_rule

_RULES = (
    native_rule(
        "lean",
        "future_consolidated_bar",
        "TIME_AVAILABILITY",
        "Data",
        ("Data", "Model", "Selector", "Execution"),
        ("data.input", "execution.order", "time_availability"),
    ),
    native_rule(
        "lean",
        "expose_next_bar",
        "TIME_AVAILABILITY",
        "Data",
        ("Data", "Model", "Selector", "Execution"),
        ("data.future_bar", "execution.on_data_decision", "consumed_by"),
    ),
    native_rule(
        "lean",
        "future_universe_membership",
        "UNIVERSE_AVAILABILITY",
        "Data",
        ("Data", "Selector"),
        ("data.input", "selector.decision", "universe_leakage"),
    ),
    native_rule(
        "lean",
        "corporate_action_consumed_early",
        "TIME_ALIGNMENT",
        "Data",
        ("Data",),
        ("data.input", "execution.order", "time_alignment"),
    ),
    native_rule(
        "lean",
        "resolution_feed_future_nearest",
        "TIME_ALIGNMENT",
        "Data",
        ("Data",),
        ("data.input", "execution.order", "time_alignment"),
    ),
    native_rule(
        "lean",
        "backtest_feedback_into_universe_rule",
        "OUTCOME_FEEDBACK_ISOLATION",
        "Risk",
        ("Risk", "Claim"),
        ("risk.rule", "execution.order", "feedback_rule"),
    ),
    native_rule(
        "lean",
        "midpoint_as_fill_price",
        "EXECUTION_IDENTITY",
        "Execution",
        ("Execution",),
        ("execution.order", "execution.order", "price_semantics"),
    ),
    native_rule(
        "lean",
        "signal_close_as_fill_price",
        "EXECUTION_IDENTITY",
        "Execution",
        ("Execution",),
        ("execution.order", "execution.order", "price_semantics"),
    ),
    native_rule(
        "lean",
        "selection_execution_population_mismatch",
        "SELECTION_EXECUTION_CONSISTENCY",
        "Selector",
        ("Selector", "Timer"),
        ("selector.decision", "execution.order", "population_mismatch"),
    ),
    native_rule(
        "lean",
        "load_alternate_config",
        "MODEL_IDENTITY",
        "Identity",
        ("Identity",),
        ("config.alternate", "execution.engine", "consumed_by"),
    ),
    native_rule(
        "lean",
        "security_identity_nearest_time",
        "CANDIDATE_IDENTITY",
        "Identity",
        ("Identity",),
        ("identity.binding", "execution.order", "candidate_drift"),
    ),
    native_rule(
        "lean",
        "claim_wrong_result_packet",
        "CLAIM_DEPENDENCY_INTEGRITY",
        "Claim",
        ("Claim",),
        ("claim.performance", "execution.order", "artifact_binding_missing"),
    ),
)


def native_fact_rules():
    return _RULES


__all__ = ["native_fact_rules"]
