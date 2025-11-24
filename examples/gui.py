import logging
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


# 1) Setup OS
os = ArbiterOSAlpha(validate_schemas=True)

# Load policies from YAML files
# This loads both kernel policies (read-only) and custom policies (developer-defined)
# Policies are defined in:
# - arbiteros_alpha/kernel_policy_list.yaml (kernel-defined, read-only)
# - examples/custom_policy_list.yaml (developer-defined, can be modified)
os.load_policies()


class GUIState(TypedDict):
    """End-to-end GUI agent demo state.

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


# 2) Nodes (Planner → Orchestrator → Perception → Reasoner → Decision/Execute loop → Report)


@os.instruction(Instr.GENERATE)
def plan(state: GUIState) -> GUIState:
    """Planner Module

    Converts a user goal into a list of subtasks. This is intentionally simple:
    - Splits on 'and' to simulate multi-step planning
    - Falls back to a single-step plan if nothing to split
    """
    goal = (state.get("goal") or "").strip()
    if not goal:
        subtasks = []
    else:
        raw_parts = [p.strip() for p in goal.split("and") if p.strip()]
        subtasks = raw_parts if raw_parts else [goal]
    note = f"Plan created with {len(subtasks)} subtask(s): {subtasks}"
    return {"subtasks": subtasks, "history": state["history"] + [note]}


@os.instruction(Instr.GENERATE)
def orchestrate(state: GUIState) -> GUIState:
    """Orchestrator Module

    Picks the next subtask to work on. Sets done=True if none remain.
    """
    remaining = [s for s in state["subtasks"] if s != state.get("current_subtask")]
    # If current_subtask was completed earlier, drop it
    if state.get("current_subtask") and state["current_subtask"] in remaining:
        # keep it as current; otherwise choose a new one below
        pass

    next_subtask = ""
    if not remaining and not state.get("current_subtask"):
        # No tasks at all
        note = "No subtasks available. Nothing to do."
        return {"done": True, "history": state["history"] + [note]}

    if not state.get("current_subtask"):
        next_subtask = remaining[0] if remaining else ""
    else:
        next_subtask = state["current_subtask"]

    if not next_subtask:
        note = "All subtasks completed."
        return {"done": True, "history": state["history"] + [note]}

    note = f"Selected subtask: {next_subtask}"
    return {
        "current_subtask": next_subtask,
        "done": False,
        "history": state["history"] + [note],
    }


@os.instruction(Instr.TOOL_CALL)
def perceive(state: GUIState) -> GUIState:
    """Perception Module

    Captures the 'current GUI state'. This demo simulates a stable GUI that is
    always ready after execution settles.
    """
    simulated_gui: Literal["ready", "busy", "error"] = "ready"
    note = f"Perceived GUI state: {simulated_gui}"
    return {"gui_state": simulated_gui, "history": state["history"] + [note]}


@os.instruction(Instr.GENERATE)
def reason(state: GUIState) -> GUIState:
    """Reasoner Module

    Produces an action command based on the current subtask and perceived GUI
    state. In this demo, the action is deterministic and formatted as a
    single-line instruction.
    """
    subtask = (state.get("current_subtask") or "").strip()
    gui = state.get("gui_state") or "unknown"
    if not subtask:
        action = "noop"
    else:
        action = f"perform '{subtask}' on GUI (state={gui})"
    note = f"Reasoned action: {action}"
    return {"action": action, "history": state["history"] + [note]}


@os.instruction(Instr.EVALUATE_PROGRESS)
def decide_finish(state: GUIState) -> GUIState:
    """Finish Subtask? (Decision)

    Heuristic:
    - If action contains the subtask phrase and GUI is 'ready', confidence is high
    - Otherwise confidence is lower and agent will iterate
    """
    subtask = (state.get("current_subtask") or "").strip()
    action = state.get("action") or ""
    gui = state.get("gui_state") or ""

    is_format_ok = bool(subtask) and subtask in action and action.startswith("perform")
    is_gui_ready = gui == "ready"

    confidence = 0.9 if (is_format_ok and is_gui_ready) else 0.4
    note = (
        f"Decision: format_ok={is_format_ok}, gui_ready={is_gui_ready}, "
        f"confidence={confidence:.2f}"
    )
    return {"confidence": confidence, "history": state["history"] + [note]}


@os.instruction(Instr.TOOL_CALL)
def execute(state: GUIState) -> GUIState:
    """Execution Module

    Executes the action. This is a no-op tool call for the demo.
    """
    action = state.get("action") or "noop"
    note = f"Executed action: {action}"

    # Mark current_subtask as completed on successful execution to progress.
    completed = state["current_subtask"]
    remaining = [s for s in state["subtasks"] if s != completed]
    return {
        "executed": True,
        "subtasks": remaining,
        "current_subtask": "",
        "history": state["history"] + [note],
    }


@os.instruction(Instr.GENERATE)
def report(state: GUIState) -> GUIState:
    """Report to User

    Summarizes what happened across the run for the demonstration.
    """
    summary = f"Completed subtasks. Final history length: {len(state['history'])}"
    return {"history": state["history"] + [summary]}


# 3) Graph wiring (approximates the provided GUI agent flow)
builder = StateGraph(GUIState)

builder.add_node(plan)
builder.add_node(orchestrate)
builder.add_node(perceive)
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
    return "report" if state.get("done") else "perceive"


builder.add_conditional_edges("orchestrate", route_after_orchestrate, path_map=None)
builder.add_edge("perceive", "reason")
builder.add_edge("reason", "decide_finish")

# Decision:
# - If confidence high, execute then go back to orchestrate for next subtask
# - If low, iterate via perception to refine
def route_after_decide(state: GUIState) -> str:
    return "execute" if (state.get("confidence", 0.0) >= 0.7) else "perceive"


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
    }

    for chunk in graph.stream(initial_state, stream_mode="values", debug=False):
        logger.info("State update: %s", chunk)

    print_history(os.history)


