"""Finance trading agent example with governance-aware LangGraph wiring."""

from __future__ import annotations

import logging
import operator
from pathlib import Path
from typing import Annotated, Any, Dict, List, TypedDict

from langgraph.graph import END, START, StateGraph
from rich.logging import RichHandler

import arbiteros_alpha.instructions as Instr
from arbiteros_alpha import ArbiterOSAlpha, print_history

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    handlers=[RichHandler()],
)

FINANCE_POLICY_PATH = Path(__file__).with_name("custom_policy_list.yaml")
FINANCE_POLICY_PY_PATH = Path(__file__).with_name("custom_policy.py")

# 1) Setup OS
os = ArbiterOSAlpha(validate_schemas=True)
os.load_policies(
    custom_policy_yaml_path=str(FINANCE_POLICY_PATH),
    custom_policy_python_path=str(FINANCE_POLICY_PY_PATH),
)


def _merge_nested_dicts(
    existing: Dict[str, Dict[str, Any]] | None,
    updates: Dict[str, Dict[str, Any]] | None,
) -> Dict[str, Dict[str, Any]]:
    merged: Dict[str, Dict[str, Any]] = {}
    if existing:
        merged.update(existing)
    if updates:
        merged.update(updates)
    return merged


def _merge_scalar_dict(
    existing: Dict[str, float] | None, updates: Dict[str, float] | None
) -> Dict[str, float]:
    merged: Dict[str, float] = {}
    if existing:
        merged.update(existing)
    if updates:
        merged.update(updates)
    return merged


def _latest_bool(existing: bool | None, update: bool | None) -> bool:
    if update is None:
        return bool(existing)
    return bool(update)


def _merge_plan_dict(
    existing: Dict[str, Any] | None, updates: Dict[str, Any] | None
) -> Dict[str, Any]:
    """Return a shallow merge of plan dictionaries to satisfy LangGraph aggregations."""
    merged: Dict[str, Any] = {}
    if existing:
        merged.update(existing)
    if updates:
        merged.update(updates)
    return merged


class TradingState(TypedDict, total=False):
    """State exchanged across the finance agent graph."""

    ticker: str
    current_date: str
    config: Dict[str, Any]
    history: Annotated[List[str], operator.add]
    reports: Annotated[Dict[str, Dict[str, Any]], _merge_nested_dicts]
    vendor_prices: Annotated[Dict[str, float], _merge_scalar_dict]
    price_drift_bps: float
    data_latency_seconds: float
    data_freshness_score: float
    data_refresh_required: bool
    bull_case: List[str]
    bear_case: List[str]
    debate_round: int
    max_debate_rounds: int
    research_summary: Dict[str, Any]
    research_summary_quality: float
    needs_additional_research: Annotated[bool, _latest_bool]
    trader_plan: Annotated[Dict[str, Any], _merge_plan_dict]
    plan_validity_score: float
    plan_valid: bool
    risk_warnings: List[str]
    final_order: Dict[str, Any]
    decision_ready: bool


def _base_state(state: TradingState) -> TradingState:
    config = state.get("config", {})
    return {
        "ticker": state.get("ticker", "UNKNOWN"),
        "current_date": state.get("current_date", "1970-01-01"),
        "config": config,
        "history": state.get("history", []),
        "reports": state.get("reports", {}),
        "vendor_prices": state.get("vendor_prices", {}),
        "price_drift_bps": state.get("price_drift_bps", 0.0),
        "data_latency_seconds": state.get("data_latency_seconds", 0.0),
        "data_freshness_score": state.get("data_freshness_score", 1.0),
        "data_refresh_required": state.get("data_refresh_required", False),
        "bull_case": state.get("bull_case", []),
        "bear_case": state.get("bear_case", []),
        "debate_round": state.get("debate_round", 0),
        "max_debate_rounds": config.get("max_debate_rounds", 2),
        "research_summary": state.get("research_summary", {}),
        "research_summary_quality": state.get("research_summary_quality", 0.0),
        "needs_additional_research": state.get("needs_additional_research", False),
        "trader_plan": state.get("trader_plan", {}),
        "plan_validity_score": state.get("plan_validity_score", 1.0),
        "plan_valid": state.get("plan_valid", True),
        "risk_warnings": state.get("risk_warnings", []),
        "final_order": state.get("final_order", {}),
        "decision_ready": state.get("decision_ready", False),
    }


def _summarize_state_for_logging(state: TradingState) -> Dict[str, Any]:
    """Return a trimmed view of the state for readable streaming logs."""
    summarized: Dict[str, Any] = dict(state)
    history = summarized.get("history")
    if isinstance(history, list) and len(history) > 5:
        summarized["history_tail"] = history[-5:]
        summarized["history_len"] = len(history)
        summarized.pop("history", None)
    return summarized


def _deterministic_metric(ticker: str, salt: str) -> float:
    """Return a deterministic pseudo metric in [0, 1]."""
    raw = sum(ord(c) for c in ticker + salt)
    return (raw % 100) / 100.0


def _price_from_metric(metric: float) -> float:
    return round(50 + metric * 150, 2)


@os.instruction(Instr.LOAD)
def bootstrap_state(state: TradingState) -> TradingState:
    """Ensure downstream nodes receive fully hydrated defaults."""
    base = _base_state(state)
    note = (
        f"Bootstrap state for {base['ticker']} on {base['current_date']} "
        f"with config keys: {list(base['config'].keys())}"
    )
    base["history"] = base["history"] + [note]
    return base


@os.instruction(Instr.TOOL_CALL)
def fetch_fundamentals(state: TradingState) -> TradingState:
    """Simulate fundamentals ingest from configured vendor."""
    vendor = state["config"].get("data_vendors", {}).get("fundamentals", "yfinance")
    metric = _deterministic_metric(state["ticker"], vendor)
    price = _price_from_metric(metric)
    report = {
        "pe_ratio": round(15 + metric * 20, 2),
        "eps_trend": "up" if metric > 0.4 else "flat",
        "vendor": vendor,
        "price_snapshot": price,
    }

    latency = 30 if vendor == "yfinance" else 120
    note = f"Fundamentals from {vendor} (latency={latency}s)"

    freshness = max(0.0, 1.0 - latency / 240.0)
    return {
        "reports": {**state["reports"], "fundamentals": report},
        "vendor_prices": {**state["vendor_prices"], vendor: price},
        "data_latency_seconds": max(state.get("data_latency_seconds", 0.0), float(latency)),
        "data_freshness_score": min(state.get("data_freshness_score", 1.0), freshness),
        "history": state["history"] + [note],
    }


@os.instruction(Instr.LOAD)
def fetch_technical(state: TradingState) -> TradingState:
    """Simulate technical indicator ingestion."""
    vendor = state["config"].get("data_vendors", {}).get("technical", "alpha_vantage")
    metric = _deterministic_metric(state["ticker"], vendor + "tech")
    price = _price_from_metric(metric + 0.05)

    report = {
        "macd": round(metric - 0.5, 2),
        "rsi": round(30 + metric * 40, 1),
        "bollinger_signal": "breakout" if metric > 0.6 else "range",
        "vendor": vendor,
        "price_snapshot": price,
    }

    latency = 10 if vendor == "intraday" else 45
    note = f"Technical data from {vendor} (latency={latency}s)"

    freshness = max(0.0, 1.0 - latency / 240.0)
    return {
        "reports": {**state["reports"], "technical": report},
        "vendor_prices": {**state["vendor_prices"], f"{vendor}-technical": price},
        "data_latency_seconds": max(state.get("data_latency_seconds", 0.0), float(latency)),
        "data_freshness_score": min(state.get("data_freshness_score", 1.0), freshness),
        "history": state["history"] + [note],
    }


@os.instruction(Instr.LOAD)
def fetch_sentiment(state: TradingState) -> TradingState:
    """Simulate sentiment analysis feed."""
    vendor = state["config"].get("data_vendors", {}).get("sentiment", "newsapi")
    metric = _deterministic_metric(state["ticker"], vendor + "sent")
    polarity = "bullish" if metric > 0.55 else "mixed"
    report = {
        "headline": f"{state['ticker']} sentiment {polarity}",
        "score": round(metric, 2),
        "vendor": vendor,
    }
    latency = 60 if vendor == "rss" else 20
    note = f"Sentiment feed {vendor} (latency={latency}s)"

    freshness = max(0.0, 1.0 - latency / 240.0)
    return {
        "reports": {**state["reports"], "sentiment": report},
        "data_latency_seconds": max(state.get("data_latency_seconds", 0.0), float(latency)),
        "data_freshness_score": min(state.get("data_freshness_score", 1.0), freshness),
        "history": state["history"] + [note],
    }


@os.instruction(Instr.GENERATE)
def synchronize_data(state: TradingState) -> TradingState:
    """Cross-check vendor prices and flag drift."""
    prices = list(state.get("vendor_prices", {}).values())
    if len(prices) < 2:
        drift_bps = 0.0
    else:
        min_price, max_price = min(prices), max(prices)
        drift_bps = ((max_price - min_price) / max_price) * 10_000
    drift_bps = round(drift_bps, 2)
    refresh_required = drift_bps >= 12.0

    note = (
        f"Price drift {drift_bps} bps across vendors; "
        f"{'refresh required' if refresh_required else 'within tolerance'}."
    )
    return {
        "price_drift_bps": drift_bps,
        "data_refresh_required": refresh_required,
        "history": state["history"] + [note],
    }


@os.instruction(Instr.GENERATE)
def research_debate(state: TradingState) -> TradingState:
    """Run one round of the bullish vs bearish debate."""
    round_idx = state.get("debate_round", 0) + 1
    max_rounds = state.get("max_debate_rounds", 2)
    fundamentals = state["reports"].get("fundamentals", {})
    technical = state["reports"].get("technical", {})
    sentiment = state["reports"].get("sentiment", {})

    bull_point = (
        f"Round {round_idx}: EPS trend {fundamentals.get('eps_trend', 'flat')} "
        f"with RSI {technical.get('rsi', 50)}"
    )
    bear_point = (
        f"Round {round_idx}: MACD {technical.get('macd', 0.0)} "
        f"and sentiment {sentiment.get('score', 0.0)}"
    )

    bull_case = state.get("bull_case", []) + [bull_point]
    bear_case = state.get("bear_case", []) + [bear_point]

    summary = {
        "ticker": state["ticker"],
        "bull_case": bull_case,
        "bear_case": bear_case,
        "consensus": "bull" if fundamentals.get("pe_ratio", 0) < 30 else "neutral",
    }
    summary_quality = 1.0 if isinstance(summary, dict) else 0.0
    needs_more = round_idx < max_rounds

    note = (
        f"Debate round {round_idx}/{max_rounds} completed; "
        f"{'continuing' if needs_more else 'ready to synthesize'}."
    )
    return {
        "debate_round": round_idx,
        "bull_case": bull_case,
        "bear_case": bear_case,
        "research_summary": summary,
        "research_summary_quality": summary_quality,
        "needs_additional_research": needs_more,
        "history": state["history"] + [note],
    }


@os.instruction(Instr.GENERATE)
def trader_node(state: TradingState) -> TradingState:
    """Convert research summary into a tentative trade plan."""
    summary = state.get("research_summary", {})
    consensus = summary.get("consensus", "neutral")
    signal = "BUY" if consensus == "bull" else "HOLD"
    confidence = 0.55 if signal == "BUY" else 0.35

    entry_price = max(state.get("vendor_prices", {}).values(), default=100.0)
    size = 0.15 if confidence >= 0.5 else 0.05
    plan = {
        "ticker": state["ticker"],
        "signal": signal,
        "confidence": confidence,
        "entry_price": round(entry_price, 2),
        "position_size": size,
        "risk_limit": state["config"].get("risk_limit", 0.2),
    }
    note = f"Trader proposes {signal} with {confidence:.2f} confidence."

    return {
        "trader_plan": plan,
        "history": state["history"] + [note],
    }


@os.instruction(Instr.VERIFY)
def validate_trade_plan(state: TradingState) -> TradingState:
    """Sanity check trader plan fields to avoid schema drift."""
    plan = state.get("trader_plan", {})
    required = {"signal", "entry_price", "position_size"}
    missing = sorted(required - plan.keys())
    plan_valid = not missing
    validity_score = 1.0 if plan_valid else 0.0
    needs_research = not plan_valid
    note = (
        "Trade plan validation passed."
        if plan_valid
        else f"Trade plan missing fields: {missing}"
    )
    return {
        "plan_valid": plan_valid,
        "plan_validity_score": validity_score,
        "needs_additional_research": needs_research,
        "history": state["history"] + [note],
    }


@os.instruction(Instr.CONSTRAIN)
def risk_manager(state: TradingState) -> TradingState:
    """Check proposed plan against risk controls."""
    plan = state.get("trader_plan", {})
    risk_warnings: List[str] = []
    exposure_limit = state["config"].get("max_position", 0.25)
    drawdown_limit = state["config"].get("max_drawdown", 0.15)

    if plan.get("position_size", 0.0) > exposure_limit:
        risk_warnings.append(
            f"Position size {plan.get('position_size')} exceeds limit {exposure_limit}"
        )
    if state.get("price_drift_bps", 0.0) >= 12.0:
        risk_warnings.append("Vendor price drift too high for execution.")
    if state.get("data_latency_seconds", 0.0) > 180:
        risk_warnings.append("Market data stale beyond tolerance.")

    approved = not risk_warnings
    if not approved:
        adjusted_size = min(plan.get("position_size", 0.0), exposure_limit)
        plan["position_size"] = adjusted_size

    note = (
        "Risk manager approved plan."
        if approved
        else f"Risk manager adjustments: {risk_warnings}"
    )

    return {
        "trader_plan": plan,
        "risk_warnings": risk_warnings,
        "history": state["history"] + [note],
    }


@os.instruction(Instr.RESPOND)
def finalize_decision(state: TradingState) -> TradingState:
    """Produce the final decision payload."""
    plan = state.get("trader_plan", {})
    warnings = state.get("risk_warnings", [])
    decision_ready = not warnings
    final_order = {
        "ticker": plan.get("ticker", state["ticker"]),
        "action": plan.get("signal", "HOLD"),
        "size": plan.get("position_size", 0.0),
        "entry_price": plan.get("entry_price", 0.0),
        "confidence": plan.get("confidence", 0.0),
        "warnings": warnings,
    }
    note = (
        "Final order ready for execution."
        if decision_ready
        else "Final order contains warnings; manual review required."
    )
    return {
        "final_order": final_order,
        "decision_ready": decision_ready,
        "history": state["history"] + [note],
    }


# 3) Graph wiring
builder = StateGraph(TradingState)
builder.add_node(bootstrap_state)
builder.add_node(fetch_fundamentals)
builder.add_node(fetch_technical)
builder.add_node(fetch_sentiment)
builder.add_node(synchronize_data)
builder.add_node(research_debate)
builder.add_node(trader_node)
builder.add_node(validate_trade_plan)
builder.add_node(risk_manager)
builder.add_node(finalize_decision)

builder.add_edge(START, "bootstrap_state")
builder.add_edge("bootstrap_state", "fetch_fundamentals")
builder.add_edge("fetch_fundamentals", "fetch_technical")
builder.add_edge("fetch_technical", "fetch_sentiment")
builder.add_edge("fetch_sentiment", "synchronize_data")
builder.add_edge("synchronize_data", "research_debate")


def route_after_research(state: TradingState) -> str:
    return "research_debate" if state.get("needs_additional_research") else "trader_node"


builder.add_conditional_edges("research_debate", route_after_research, path_map=None)
builder.add_edge("trader_node", "validate_trade_plan")
builder.add_edge("validate_trade_plan", "risk_manager")
builder.add_edge("risk_manager", "finalize_decision")
builder.add_edge("finalize_decision", END)

try:
    os.validate_graph_structure(
        builder, visualize=True, visualization_file="finance_workflow.mmd"
    )
    logger.info("Graph structure validation passed.")
except RuntimeError as exc:  # pragma: no cover - demo logging path
    logger.error("Graph structure validation failed: %s", exc)

print("Finished validating graph structure\n\n")

graph = builder.compile()


class TradingAgentsGraph:
    """Entry point for running the governed finance example."""

    def __init__(self, config: Dict[str, Any] | None = None):
        self.config = config or {}

    def propagate(self, ticker: str, current_date: str) -> Dict[str, Any]:
        """Execute the trading graph and return the final decision.

        Args:
            ticker: Symbol under analysis (for example, "NVDA").
            current_date: Trading date context (ISO-8601 string).

        Returns:
            Final decision dictionary containing the execution plan and warnings.
        """
        initial_state: TradingState = {
            "ticker": ticker,
            "current_date": current_date,
            "config": self.config,
            "history": [],
        }
        result = graph.invoke(initial_state)
        return result.get("final_order", {})


if __name__ == "__main__":
    sample_config = {
        "max_debate_rounds": 3,
        "data_vendors": {
            "fundamentals": "yfinance",
            "technical": "alpha_vantage",
            "sentiment": "newsapi",
        },
        "max_position": 0.2,
        "max_drawdown": 0.1,
    }

    initial_state: TradingState = {
        "ticker": "NVDA",
        "current_date": "2024-05-10",
        "config": sample_config,
        "history": [],
    }

    os.history.clear()
    logger.info("Running finance agent with streaming updates.")

    final_state: TradingState | None = None
    for chunk in graph.stream(initial_state, stream_mode="values", debug=False):
        logger.info("State update: %s", _summarize_state_for_logging(chunk))
        final_state = chunk

    # print_history(os.history)

    final_order = final_state.get("final_order", {}) if final_state else {}
    logger.info("Final decision: %s", final_order)

