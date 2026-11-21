"""Freqtrade classic native objects mapped to canonical evidence semantics."""

from dmcasrc.adapters._scientific import native_rule

_RULES = (
    native_rule(
        "freqtrade-classic",
        "inject_lookahead_indicator",
        "TIME_AVAILABILITY",
        "Data",
        ("Data", "Model", "Selector"),
        ("data.future_candle", "selector.entry_signal", "consumed_by"),
    ),
    native_rule(
        "freqtrade-classic",
        "future_pair_universe",
        "UNIVERSE_AVAILABILITY",
        "Data",
        ("Data", "Selector"),
        ("data.input", "selector.decision", "universe_leakage"),
    ),
    native_rule(
        "freqtrade-classic",
        "multitimeframe_future_nearest",
        "TIME_ALIGNMENT",
        "Data",
        ("Data",),
        ("data.input", "model.predict", "time_alignment"),
    ),
    native_rule(
        "freqtrade-classic",
        "backtest_feedback_into_filter",
        "OUTCOME_FEEDBACK_ISOLATION",
        "Risk",
        ("Risk", "Claim"),
        ("risk.rule", "execution.order", "feedback_rule"),
    ),
    native_rule(
        "freqtrade-classic",
        "bind_close_to_open_window",
        "EXECUTION_IDENTITY",
        "Execution",
        ("Execution",),
        ("execution.window", "evaluation.trade", "consumed_by"),
    ),
    native_rule(
        "freqtrade-classic",
        "selection_execution_window_mismatch",
        "SELECTION_EXECUTION_CONSISTENCY",
        "Selector",
        ("Selector", "Timer"),
        ("selector.decision", "execution.order", "population_mismatch"),
    ),
)


def native_fact_rules():
    return _RULES


__all__ = ["native_fact_rules"]
