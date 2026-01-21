import logging
import operator
from typing import Annotated, List, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send
from rich.logging import RichHandler

from arbiteros_alpha import ArbiterOSAlpha
from arbiteros_alpha import instructions as Instr
from arbiteros_alpha.policy import HistoryPolicyChecker

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.DEBUG,
    handlers=[RichHandler()],
)


# Schema for structured output to use in planning
class Section(TypedDict):
    name: str
    description: str


class Sections(TypedDict):
    sections: List[Section]


# Graph state
class State(TypedDict):
    topic: str  # Report topic
    sections: list[Section]  # List of report sections
    completed_sections: Annotated[
        list, operator.add
    ]  # All workers write to this key in parallel
    final_report: str  # Final report


# Worker state
class WorkerState(TypedDict):
    section: Section
    completed_sections: Annotated[list, operator.add]


# OS
arbiter_os = ArbiterOSAlpha(backend="langgraph")
arbiter_os.add_policy_checker(
    HistoryPolicyChecker(
        name="no_direct_synthesizer",
        bad_sequence=[Instr.GENERATE, Instr.RESPOND],
    )
)


# Nodes
@arbiter_os.instruction(Instr.DECOMPOSE)
def orchestrator(state: State):
    """Orchestrator that generates a plan for the report"""

    return {
        "sections": [
            {
                "name": "Introduction",
                "description": "An introduction to the topic.",
            },
            {
                "name": "Main Content",
                "description": "Detailed information about the topic.",
            },
            {
                "name": "Conclusion",
                "description": "A summary and conclusion of the topic.",
            },
        ]
    }


@arbiter_os.instruction(Instr.GENERATE)
def llm_call(state: WorkerState):
    """Worker writes a section of the report"""

    # Write the updated section to completed sections
    return {
        "completed_sections": [
            f"## {state['section']['name']}\n\nThis is the content for {state['section']['description']}"
        ]
    }


@arbiter_os.instruction(Instr.RESPOND)
def synthesizer(state: State):
    """Synthesize full report from sections"""

    # List of completed sections
    completed_sections = state["completed_sections"]

    # Format completed section to str to use as context for final sections
    completed_report_sections = "\n\n---\n\n".join(completed_sections)

    return {"final_report": completed_report_sections}


# Conditional edge function to create llm_call workers that each write a section of the report
def assign_workers(state: State):
    """Assign a worker to each section in the plan"""

    # Kick off section writing in parallel via Send() API
    return [Send("llm_call", {"section": s}) for s in state["sections"]]


# Build workflow
orchestrator_worker_builder = StateGraph(State)

# Add the nodes
orchestrator_worker_builder.add_node("orchestrator", orchestrator)
orchestrator_worker_builder.add_node("llm_call", llm_call)
orchestrator_worker_builder.add_node("synthesizer", synthesizer)

# Add edges to connect nodes
orchestrator_worker_builder.add_edge(START, "orchestrator")
orchestrator_worker_builder.add_conditional_edges(
    "orchestrator", assign_workers, ["llm_call"]
)
orchestrator_worker_builder.add_edge("llm_call", "synthesizer")
orchestrator_worker_builder.add_edge("synthesizer", END)

# Compile the workflow
orchestrator_worker = orchestrator_worker_builder.compile()

arbiter_os.register_compiled_graph(orchestrator_worker)


@arbiter_os.rollout()
def main():
    # Invoke
    state = orchestrator_worker.invoke({"topic": "Create a report on LLM scaling laws"})
    return state


if __name__ == "__main__":
    main()
    arbiter_os.history.pprint()
