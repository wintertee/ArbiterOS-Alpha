"""Tests for the GUI example governance hooks."""

from langgraph.types import Command

from examples.GUI_agent import gui


def _make_state(**overrides):
    state = {
        "goal": "",
        "subtasks": [],
        "current_subtask": "",
        "gui_state": "",
        "action": "",
        "executed": False,
        "confidence": 0.0,
        "done": False,
        "history": [],
        "plan_confidence": 0.0,
        "needs_replan": False,
        "gui_ready_score": 1.0,
        "ui_blockers": [],
        "perception_latency": 0.0,
        "reason_consistency": 1.0,
        "execution_alignment": 1.0,
        "retry_count": 0,
        "max_retries": 3,
    }
    state.update(overrides)
    return state


def _unwrap(result):
    if isinstance(result, Command):
        return result.update or {}
    return result


def test_plan_sets_confidence_and_flag():
    """Planner should output confidence and replan flag derived from heuristics."""
    state = _make_state(goal="open settings and enable dark mode", history=[])

    result = _unwrap(gui.plan(state))

    assert result["subtasks"] == ["open settings", "enable dark mode"]
    assert result["plan_confidence"] >= 0.65
    assert result["needs_replan"] is False


def test_plan_requests_replan_when_goal_missing():
    """Planner should request replanning when goal is empty."""
    state = _make_state(goal="", history=[])

    result = _unwrap(gui.plan(state))

    assert result["plan_confidence"] < 0.65
    assert result["needs_replan"] is True


def test_perceive_detects_blockers_when_action_pending():
    """Perception should surface blockers and lower readiness when action is pending."""
    state = _make_state(action="perform test", executed=False, retry_count=1, history=[])

    result = _unwrap(gui.perceive(state))

    assert result["gui_ready_score"] < 0.75
    assert "pending_transition" in result["ui_blockers"]


def test_execute_increments_retry_when_gui_unstable():
    """Execution must defer when GUI readiness is low."""
    state = _make_state(
        action="perform test",
        gui_ready_score=0.5,
        current_subtask="test",
        subtasks=["test"],
        history=[],
    )

    result = _unwrap(gui.execute(state))

    assert result["executed"] is False
    assert result["retry_count"] == 1
    # Subtasks remain untouched in retry scenario
    assert "subtasks" not in result


def test_resolve_blockers_requests_replan_for_login_issue():
    """Blocker resolution should set needs_replan when login issues occur."""
    state = _make_state(ui_blockers=["login_expired"], history=[])

    result = _unwrap(gui.resolve_blockers(state))

    assert result["ui_blockers"] == []
    assert result["needs_replan"] is True

