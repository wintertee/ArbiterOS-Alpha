import logging
from pathlib import Path
from typing import List, Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from rich.logging import RichHandler

import arbiteros_alpha.instructions as Instr
from arbiteros_alpha import ArbiterOSAlpha, print_history
# Policies are now loaded from YAML files via os.load_policies()
# No need to import policy classes unless defining custom policies programmatically

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    handlers=[RichHandler()],
)

GUI_POLICY_PATH = Path(__file__).with_name("custom_policy_list.yaml")
GUI_POLICY_PY_PATH = Path(__file__).with_name("custom_policy.py")
# 1) Setup OS
os = ArbiterOSAlpha(validate_schemas=True)

# Load policies from YAML files
# This loads both kernel policies (read-only) and custom policies (developer-defined)
# Policies are defined in:
# - arbiteros_alpha/kernel_policy_list.yaml (kernel-defined, read-only)
# - examples/custom_policy_list.yaml (developer-defined, can be modified)
os.load_policies(
    custom_policy_yaml_path=str(GUI_POLICY_PATH),
    custom_policy_python_path=str(GUI_POLICY_PY_PATH),
)


class GUIState(TypedDict):
    """End-to-end GUI agent demo state with governance hooks.

    Fields:
        goal: The user's high-level objective.
        subtasks: Planner-produced list of subtasks.
        current_subtask: The subtask currently being worked on.
        gui_state: Perception snapshot of the current GUI.
        action: The action command produced by the reasoner.
        executed: Whether the last action has been executed.
        confidence: Continuous score in [0, 1] estimating readiness/quality.
        done: Whether all subtasks are completed.
        history: Human-readable log of steps for demo output.
        plan_confidence: Heuristic confidence score for the current plan.
        needs_replan: Flag set by governance or execution to request replanning.
        gui_ready_score: Proxy for perception readiness (1.0 means stable UI).
        ui_blockers: List of blockers detected by perception (e.g., popups).
        perception_latency: Simulated latency used to avoid stale screenshots.
        reason_consistency: Score indicating whether reasoning aligns with plan.
        execution_alignment: Alignment score between action intent and GUI state.
        retry_count: Number of consecutive retries for the current subtask.
        max_retries: Upper bound before escalation.
    """

    goal: str
    subtasks: List[str]
    current_subtask: str
    gui_state: str
    action: str
    executed: bool
    confidence: float
    done: bool
    history: List[str]
    plan_confidence: float
    needs_replan: bool
    gui_ready_score: float
    ui_blockers: List[str]
    perception_latency: float
    reason_consistency: float
    execution_alignment: float
    retry_count: int
    max_retries: int


# 2) Nodes (Planner → Orchestrator → Perception → Reasoner → Decision/Execute loop → Report)


@os.instruction(Instr.GENERATE)
def plan(state: GUIState) -> GUIState:
    """Planner module that surfaces plan confidence for governance."""
    goal = (state.get("goal") or "").strip()
    if not goal:
        subtasks = []
    else:
        raw_parts = [p.strip() for p in goal.replace("&", "and").split("and") if p.strip()]
        subtasks = raw_parts if raw_parts else [goal]

    unique_ratio = (len(set(subtasks)) / len(subtasks)) if subtasks else 0.0
    coverage_bonus = 0.2 if (" and " in goal.lower() and len(subtasks) > 1) else 0.0
    plan_confidence = min(0.95, 0.2 + 0.5 * bool(subtasks) + 0.2 * unique_ratio + coverage_bonus)
    needs_replan = plan_confidence < 0.65

    note = (
        f"Plan created with {len(subtasks)} subtask(s): {subtasks} "
        f"(confidence={plan_confidence:.2f}, needs_replan={needs_replan})"
    )
    return {
        "subtasks": subtasks,
        "plan_confidence": plan_confidence,
        "needs_replan": needs_replan,
        "history": state["history"] + [note],
    }


@os.instruction(Instr.GENERATE)
def orchestrate(state: GUIState) -> GUIState:
    """Orchestrator module with retry and replanning awareness."""
    remaining = [s for s in state["subtasks"] if s != state.get("current_subtask")]
    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", 3)
    history = state["history"]

    if retry_count >= max_retries:
        note = (
            f"Retry limit reached ({retry_count}/{max_retries}). "
            "Escalating to planner."
        )
        return {
            "needs_replan": True,
            "retry_count": 0,
            "current_subtask": "",
            "history": history + [note],
        }

    if not remaining and not state.get("current_subtask"):
        note = "No subtasks available. Nothing to do."
        return {"done": True, "history": history + [note]}

    next_subtask = state.get("current_subtask") or (remaining[0] if remaining else "")
    if not next_subtask:
        note = "All subtasks completed."
        return {"done": True, "history": history + [note]}

    if state.get("needs_replan") and state.get("plan_confidence", 1.0) >= 0.65:
        # Replanning request satisfied; allow orchestration to continue.
        note = "Plan validated after replanning. Resuming execution."
        return {
            "needs_replan": False,
            "history": history + [note],
        }

    note = f"Selected subtask: {next_subtask}"
    return {
        "current_subtask": next_subtask,
        "done": False,
        "history": history + [note],
    }


@os.instruction(Instr.TOOL_CALL)
def perceive(state: GUIState) -> GUIState:
    """Perception module with blocker detection and readiness score."""
    action_pending = bool(state.get("action"))
    executed = state.get("executed", False)
    retry_count = state.get("retry_count", 0)

    if not executed and action_pending:
        simulated_gui: Literal["ready", "busy", "error"] = "busy"
        gui_ready_score = 0.6 - 0.05 * retry_count
        ui_blockers = ["pending_transition"]
    else:
        simulated_gui = "ready"
        gui_ready_score = 0.9
        ui_blockers = ["login_expired"] if retry_count > 1 else []

    gui_ready_score = max(0.2, min(gui_ready_score, 1.0))
    perception_latency = 0.2 if gui_ready_score >= 0.8 else 0.8

    note = (
        f"Perceived GUI state: {simulated_gui} "
        f"(ready_score={gui_ready_score:.2f}, blockers={ui_blockers})"
    )
    return {
        "gui_state": simulated_gui,
        "gui_ready_score": gui_ready_score,
        "ui_blockers": ui_blockers,
        "perception_latency": perception_latency,
        "history": state["history"] + [note],
    }


@os.instruction(Instr.GENERATE)
def reason(state: GUIState) -> GUIState:
    """Reasoner module that tracks consistency with planner and GUI state."""
    subtask = (state.get("current_subtask") or "").strip()
    gui = state.get("gui_state") or "unknown"
    if not subtask:
        action = "noop"
    else:
        action = f"perform '{subtask}' on GUI (state={gui})"
    consistency = 0.9 if (subtask and gui == "ready") else 0.6 if subtask else 0.3
    if state.get("ui_blockers"):
        consistency -= 0.2
    consistency = max(0.0, consistency)

    note = f"Reasoned action: {action} (consistency={consistency:.2f})"
    return {
        "action": action,
        "reason_consistency": consistency,
        "history": state["history"] + [note],
    }


@os.instruction(Instr.EVALUATE_PROGRESS)
def decide_finish(state: GUIState) -> GUIState:
    """Finish subtask decision node with governance-friendly signals."""
    subtask = (state.get("current_subtask") or "").strip()
    action = state.get("action") or ""
    gui_ready_score = state.get("gui_ready_score", 1.0)
    reason_consistency = state.get("reason_consistency", 1.0)

    is_format_ok = bool(subtask) and subtask in action and action.startswith("perform")
    confidence = max(
        0.1,
        min(
            0.95,
            0.4 * is_format_ok + 0.3 * reason_consistency + 0.3 * gui_ready_score,
        ),
    )

    needs_replan = reason_consistency < 0.5 or gui_ready_score < 0.5
    note = (
        f"Decision: format_ok={is_format_ok}, gui_ready={gui_ready_score:.2f}, "
        f"consistency={reason_consistency:.2f}, confidence={confidence:.2f}, "
        f"needs_replan={needs_replan}"
    )
    return {
        "confidence": confidence,
        "needs_replan": needs_replan,
        "history": state["history"] + [note],
    }


@os.instruction(Instr.TOOL_CALL)
def execute(state: GUIState) -> GUIState:
    """Execution module that validates GUI readiness before acting."""
    action = state.get("action") or "noop"
    gui_ready_score = state.get("gui_ready_score", 1.0)
    history = state["history"]

    execution_alignment = 0.95 if gui_ready_score >= 0.75 else 0.5
    succeeded = execution_alignment >= 0.8

    if succeeded:
        completed = state.get("current_subtask", "")
        remaining = [s for s in state["subtasks"] if s != completed]
        note = f"Executed action: {action}"
        return {
            "executed": True,
            "subtasks": remaining,
            "current_subtask": "",
            "retry_count": 0,
            "execution_alignment": execution_alignment,
            "needs_replan": False,
            "history": history + [note],
        }

    # Execution deferred due to unstable GUI.
    note = (
        "Execution delayed: GUI unstable "
        f"(alignment={execution_alignment:.2f}). Retrying after perception."
    )
    return {
        "executed": False,
        "retry_count": state.get("retry_count", 0) + 1,
        "execution_alignment": execution_alignment,
        "history": history + [note],
    }


@os.instruction(Instr.FALLBACK)
def resolve_blockers(state: GUIState) -> GUIState:
    """Resolve blockers such as popups or login expiration."""
    blockers = state.get("ui_blockers", [])
    if not blockers:
        note = "No blockers detected during resolution step."
        return {"history": state["history"] + [note]}

    needs_replan = any("login" in blocker for blocker in blockers)
    note = f"Resolved blockers: {blockers} (needs_replan={needs_replan})"
    return {
        "ui_blockers": [],
        "needs_replan": needs_replan,
        "history": state["history"] + [note],
    }


@os.instruction(Instr.GENERATE)
def report(state: GUIState) -> GUIState:
    """Report node summarizing governance signals."""
    summary = (
        f"Completed subtasks. Final history length: {len(state['history'])}. "
        f"Plan confidence={state.get('plan_confidence', 0.0):.2f}"
    )
    return {"history": state["history"] + [summary]}


# 3) Graph wiring (approximates the provided GUI agent flow)
builder = StateGraph(GUIState)

builder.add_node(plan)
builder.add_node(orchestrate)
builder.add_node(perceive)
builder.add_node(resolve_blockers)
builder.add_node(reason)
builder.add_node(decide_finish)
builder.add_node(execute)
builder.add_node(report)

# START → Planner → Orchestrator
builder.add_edge(START, "plan")
builder.add_edge("plan", "orchestrate")

# If done at orchestrator, go to report then END
# Otherwise inner loop: Perception → Reason → Decision
def route_after_orchestrate(state: GUIState) -> str:
    if state.get("done"):
        return "report"
    if state.get("needs_replan"):
        return "plan"
    return "perceive"


builder.add_conditional_edges("orchestrate", route_after_orchestrate, path_map=None)


def route_after_perceive(state: GUIState) -> str:
    if state.get("ui_blockers"):
        return "resolve_blockers"
    return "reason"


builder.add_conditional_edges("perceive", route_after_perceive, path_map=None)
builder.add_edge("resolve_blockers", "reason")
builder.add_edge("reason", "decide_finish")

# Decision:
# - If confidence high, execute then go back to orchestrate for next subtask
# - If low, iterate via perception to refine
def route_after_decide(state: GUIState) -> str:
    if state.get("needs_replan"):
        return "plan"
    if state.get("confidence", 0.0) >= 0.7:
        return "execute"
    if state.get("retry_count", 0) >= state.get("max_retries", 3):
        return "plan"
    return "perceive"


builder.add_conditional_edges("decide_finish", route_after_decide, path_map=None)
builder.add_edge("execute", "orchestrate")

# Report and end
builder.add_edge("report", END)

# 4) Validate graph structure before execution
try:
    # Validate and visualize workflow
    os.validate_graph_structure(builder, visualize=True, visualization_file="workflow.mmd")
    logger.info("Graph structure validation passed.")
except RuntimeError as e:
    logger.error("Graph structure validation failed: %s", e)
    # For demo purposes, continue to show the run behavior

graph = builder.compile()

print(f"Finished validating graph structure\n\n")

# 5) Run demo
if __name__ == "__main__":
    initial_state: GUIState = {
        "goal": "open settings and enable dark mode",
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

    for chunk in graph.stream(initial_state, stream_mode="values", debug=False):
        logger.info("State update: %s", chunk)

    print_history(os.history)


