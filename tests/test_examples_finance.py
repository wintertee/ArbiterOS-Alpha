"""Unit tests for the finance trading example."""

from langgraph.types import Command

from examples.finance import trading_graph


def _unwrap(result):
    if isinstance(result, Command):
        return result.update or {}
    return result


def _base_state(**overrides):
    state = {
        "ticker": "NVDA",
        "current_date": "2024-05-10",
        "config": {
            "max_debate_rounds": 2,
            "data_vendors": {"fundamentals": "yfinance", "technical": "alpha_vantage"},
            "max_position": 0.25,
        },
        "history": [],
        "reports": {},
        "vendor_prices": {},
    }
    state.update(overrides)
    return state


def test_trading_graph_propagate_returns_decision():
    """TradingAgentsGraph.propagate should emit a final order payload."""
    app = trading_graph.TradingAgentsGraph(
        {
            "max_debate_rounds": 2,
            "data_vendors": {
                "fundamentals": "yfinance",
                "technical": "alpha_vantage",
                "sentiment": "newsapi",
            },
            "max_position": 0.2,
        }
    )

    decision = app.propagate("NVDA", "2024-05-10")

    assert decision["ticker"] == "NVDA"
    assert decision["action"] in {"BUY", "HOLD"}
    assert 0 <= decision["confidence"] <= 1


def test_synchronize_data_detects_drift():
    """Vendor mismatch should raise drift and require refresh."""
    state = _base_state(
        vendor_prices={"alpha": 100.0, "beta": 101.8},
        history=[],
    )

    result = _unwrap(trading_graph.synchronize_data(state))

    assert result["price_drift_bps"] > 12.0
    assert result["data_refresh_required"] is True


def test_research_debate_progresses_rounds():
    """Research debate should increment rounds and eventually stop."""
    state = _base_state(
        reports={
            "fundamentals": {"pe_ratio": 25, "eps_trend": "up"},
            "technical": {"macd": 0.2, "rsi": 55},
            "sentiment": {"score": 0.6},
        },
        debate_round=1,
        max_debate_rounds=2,
        bull_case=["Round 1 bull"],
        bear_case=["Round 1 bear"],
        history=[],
    )

    result = _unwrap(trading_graph.research_debate(state))

    assert result["debate_round"] == 2
    assert result["needs_additional_research"] is False
    assert "research_summary" in result


def test_validate_trade_plan_flags_missing_fields():
    """Validator should route back when required plan keys are absent."""
    state = _base_state(
        trader_plan={"ticker": "NVDA"},
        history=[],
    )

    result = _unwrap(trading_graph.validate_trade_plan(state))

    assert result["plan_valid"] is False
    assert result["needs_additional_research"] is True


