"""Plan-and-Execute agent example with governance-aware LangGraph wiring.

This agent demonstrates a "plan-and-execute" style agent inspired by the
Plan-and-Solve paper and Baby-AGI project. The core idea is to first come up
with a multi-step plan, and then go through that plan one item at a time.
After accomplishing a particular task, the agent can revisit the plan and
modify as appropriate.
"""

from __future__ import annotations

import logging
import operator
import os as os_module
from pathlib import Path
from typing import Annotated, List, Tuple, TypedDict

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field
from rich.logging import RichHandler

import arbiteros_alpha.instructions as Instr
from arbiteros_alpha import ArbiterOSAlpha, print_history

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    handlers=[RichHandler()],
)

# Set API keys and base URL directly
os_module.environ["OPENAI_API_KEY"] = "sk-zk298f767a7a34739457e8df6f3afd50e5869417dab53465"
os_module.environ["OPENAI_BASE_URL"] = "https://api.zhizengzeng.com/v1"
os_module.environ["TAVILY_API_KEY"] = "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"

PLAN_EXECUTE_POLICY_PATH = Path(__file__).with_name("custom_policy_list.yaml")
PLAN_EXECUTE_POLICY_PY_PATH = Path(__file__).with_name("custom_policy.py")

# 1) Setup OS
os_instance = ArbiterOSAlpha(validate_schemas=True)
os_instance.load_policies(
    custom_policy_yaml_path=str(PLAN_EXECUTE_POLICY_PATH),
    custom_policy_python_path=str(PLAN_EXECUTE_POLICY_PY_PATH),
)

# 2) Define Tools
tools = [TavilySearchResults(max_results=3)]

# 3) Define Execution Agent
llm = ChatOpenAI(model="gpt-4o")
prompt = "You are a helpful assistant."
agent_executor = create_react_agent(llm, tools, prompt=prompt)

# 4) Define State
def _latest_int(existing: int | None, update: int | None) -> int:
    """Return the latest integer value (non-None update takes precedence)."""
    if update is not None:
        return int(update)
    return int(existing) if existing is not None else 0


def _latest_float(existing: float | None, update: float | None) -> float:
    """Return the latest float value (non-None update takes precedence)."""
    if update is not None:
        return float(update)
    return float(existing) if existing is not None else 0.0


def _latest_bool(existing: bool | None, update: bool | None) -> bool:
    """Return the latest boolean value (non-None update takes precedence)."""
    if update is not None:
        return bool(update)
    return bool(existing) if existing is not None else False


def _latest_list(existing: List[str] | None, update: List[str] | None) -> List[str]:
    """Return the latest list value (non-None update takes precedence)."""
    if update is not None:
        return update
    return existing if existing is not None else []


def _latest_str(existing: str | None, update: str | None) -> str:
    """Return the latest string value (non-None update takes precedence)."""
    if update is not None:
        return update
    return existing if existing is not None else ""


class PlanExecute(TypedDict, total=False):
    """State for plan-and-execute agent."""

    input: str
    plan: Annotated[List[str], _latest_list]
    past_steps: Annotated[List[Tuple[str, str]], operator.add]
    response: Annotated[str, _latest_str]
    # Quality metrics for policy enforcement
    plan_quality_score: Annotated[float, _latest_float]  # Quality of the current plan (0.0-1.0)
    execution_success_score: Annotated[float, _latest_float]  # Success rate of executed steps (0.0-1.0)
    step_count: Annotated[int, _latest_int]  # Number of steps executed
    max_steps: Annotated[int, _latest_int]  # Maximum allowed steps before replanning
    response_quality_score: Annotated[float, _latest_float]  # Quality of the final response (0.0-1.0)
    error_count: Annotated[int, _latest_int]  # Number of errors encountered
    execution_failed: Annotated[bool, _latest_bool]  # Flag indicating if last execution failed


# 5) Define Pydantic Models for Structured Output
class Plan(BaseModel):
    """Plan to follow in future."""

    steps: List[str] = Field(
        description="different steps to follow, should be in sorted order"
    )


class Response(BaseModel):
    """Response to user."""

    response: str


class Act(BaseModel):
    """Action to perform."""

    action: Response | Plan = Field(
        description="Action to perform. If you want to respond to user, use Response. "
        "If you need to further use tools to get the answer, use Plan."
    )


# 6) Define Planner and Replanner Chains
planner_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """For the given objective, come up with a simple step by step plan. \
This plan should involve individual tasks, that if executed correctly will yield the correct answer. Do not add any superfluous steps. \
The result of the final step should be the final answer. Make sure that each step has all the information needed - do not skip steps.""",
        ),
        ("placeholder", "{messages}"),
    ]
)
planner = planner_prompt | ChatOpenAI(
    model="gpt-4o", temperature=0
).with_structured_output(Plan)

replanner_prompt = ChatPromptTemplate.from_template(
    """For the given objective, come up with a simple step by step plan. \
This plan should involve individual tasks, that if executed correctly will yield the correct answer. Do not add any superfluous steps. \
The result of the final step should be the final answer. Make sure that each step has all the information needed - do not skip steps.

Your objective was this:
{input}

Your original plan was this:
{plan}

You have currently done the follow steps:
{past_steps}

Update your plan accordingly. If no more steps are needed and you can return to the user, then respond with that. Otherwise, fill out the plan. Only add steps to the plan that still NEED to be done. Do not return previously done steps as part of the plan."""
)

replanner = replanner_prompt | ChatOpenAI(
    model="gpt-4o", temperature=0
).with_structured_output(Act)

# 7) Define Graph Nodes


@os_instance.instruction(Instr.GENERATE)
def plan_step(state: PlanExecute) -> PlanExecute:
    """Create an initial plan based on the user's input."""
    plan_result = planner.invoke({"messages": [("user", state["input"])]})
    steps = plan_result.steps
    
    # Calculate plan quality score
    # Factors: number of steps (optimal 3-7), step clarity, completeness
    step_count = len(steps)
    if step_count == 0:
        plan_quality = 0.0
    elif step_count > 10:
        # Too many steps suggests poor decomposition
        plan_quality = 0.3
    elif step_count < 2:
        # Too few steps suggests incomplete planning
        plan_quality = 0.5
    else:
        # Optimal range: 3-7 steps
        plan_quality = min(1.0, 0.7 + 0.1 * (7 - abs(step_count - 5)))
    
    # Check step clarity (simple heuristic: longer steps might be clearer)
    avg_step_length = sum(len(step) for step in steps) / max(len(steps), 1)
    clarity_bonus = min(0.2, avg_step_length / 100.0)
    plan_quality = min(1.0, plan_quality + clarity_bonus)
    
    return {
        "plan": steps,
        "plan_quality_score": plan_quality,
        "step_count": 0,  # Reset step count for new plan
        "error_count": 0,  # Reset error count
        "execution_failed": False,
    }


@os_instance.instruction(Instr.TOOL_CALL)
def execute_step(state: PlanExecute) -> PlanExecute:
    """Execute the first step in the current plan."""
    plan = state["plan"]
    if not plan:
        error_count = state.get("error_count", 0) + 1
        return {
            "past_steps": [("", "No plan available")],
            "error_count": error_count,
            "execution_failed": True,
        }

    plan_str = "\n".join(f"{i + 1}. {step}" for i, step in enumerate(plan))
    task = plan[0]
    task_formatted = f"""For the following plan:
{plan_str}\n\nYou are tasked with executing step 1, {task}."""
    
    try:
        agent_response = agent_executor.invoke(
            {"messages": [("user", task_formatted)]}
        )
        result_content = agent_response["messages"][-1].content
        
        # Check if execution was successful
        # Simple heuristic: check for error indicators
        error_indicators = [
            "unable to",
            "error",
            "failed",
            "cannot",
            "connection issue",
            "not available",
        ]
        execution_failed = any(
            indicator.lower() in result_content.lower() for indicator in error_indicators
        )
        
        step_count = state.get("step_count", 0) + 1
        error_count = state.get("error_count", 0)
        if execution_failed:
            error_count += 1
        
        # Calculate execution success score
        total_steps = len(state.get("past_steps", [])) + 1
        success_steps = total_steps - error_count
        execution_success_score = success_steps / max(total_steps, 1)
        
        return {
            "past_steps": [(task, result_content)],
            "step_count": step_count,
            "error_count": error_count,
            "execution_failed": execution_failed,
            "execution_success_score": execution_success_score,
        }
    except Exception as e:
        # Handle execution exceptions
        error_count = state.get("error_count", 0) + 1
        step_count = state.get("step_count", 0) + 1
        logger.error(f"Execution error: {e}")
        return {
            "past_steps": [(task, f"Execution error: {str(e)}")],
            "step_count": step_count,
            "error_count": error_count,
            "execution_failed": True,
            "execution_success_score": max(0.0, 1.0 - (error_count / max(step_count, 1))),
        }


@os_instance.instruction(Instr.GENERATE)
def replan_step(state: PlanExecute) -> PlanExecute:
    """Re-plan based on the results of executed steps."""
    # Format past steps for the replanner
    past_steps_str = "\n".join(
        f"- {step}: {result}" for step, result in state.get("past_steps", [])
    )
    plan_str = "\n".join(state.get("plan", []))

    replanner_input = {
        "input": state["input"],
        "plan": plan_str,
        "past_steps": past_steps_str,
    }

    try:
        output = replanner.invoke(replanner_input)
        if isinstance(output.action, Response):
            response = output.action.response
            # Calculate response quality score
            # Simple heuristic: longer responses with key information are better
            response_length = len(response)
            has_answer_indicators = any(
                word in response.lower()
                for word in ["is", "are", "was", "were", "the", "answer", "result"]
            )
            response_quality = min(1.0, 0.5 + 0.3 * (response_length > 50) + 0.2 * has_answer_indicators)
            
            return {
                "response": response,
                "response_quality_score": response_quality,
            }
        else:
            # New plan generated
            new_steps = output.action.steps
            # Calculate plan quality for new plan
            step_count = len(new_steps)
            if step_count == 0:
                plan_quality = 0.0
            elif step_count > 10:
                plan_quality = 0.3
            elif step_count < 2:
                plan_quality = 0.5
            else:
                plan_quality = min(1.0, 0.7 + 0.1 * (7 - abs(step_count - 5)))
            
            return {
                "plan": new_steps,
                "plan_quality_score": plan_quality,
            }
    except Exception as e:
        logger.error(f"Replanning error: {e}")
        # On error, try to continue with existing plan or generate error response
        error_count = state.get("error_count", 0) + 1
        if error_count > 3:
            # Too many errors, generate error response
            return {
                "response": f"Unable to complete task due to multiple errors. Last error: {str(e)}",
                "response_quality_score": 0.2,
                "error_count": error_count,
            }
        # Return existing plan with lower quality score
        return {
            "plan": state.get("plan", []),
            "plan_quality_score": 0.3,
            "error_count": error_count,
        }


# 8) Graph Wiring
builder = StateGraph(PlanExecute)

# Add nodes
builder.add_node("planner", plan_step)
builder.add_node("agent", execute_step)
builder.add_node("replan", replan_step)

# Add edges
builder.add_edge(START, "planner")
builder.add_edge("planner", "agent")
builder.add_edge("agent", "replan")


def should_end(state: PlanExecute) -> str:
    """Determine if we should end or continue executing."""
    if "response" in state and state["response"]:
        return END
    else:
        return "agent"


builder.add_conditional_edges(
    "replan",
    should_end,
    ["agent", END],
)

# 9) Validate graph structure
try:
    os_instance.validate_graph_structure(
        builder, visualize=True, visualization_file="plan_execute_workflow.mmd"
    )
    logger.info("Graph structure validation passed.")
except RuntimeError as exc:
    logger.error("Graph structure validation failed: %s", exc)

print("Finished validating graph structure\n\n")

graph = builder.compile()


class PlanExecuteAgent:
    """Entry point for running the plan-and-execute agent."""

    def __init__(self):
        """Initialize the plan-and-execute agent."""
        pass

    def run(self, user_input: str) -> dict:
        """Execute the plan-and-execute graph and return the final result.

        Args:
            user_input: The user's query or task.

        Returns:
            Final state dictionary containing the response or execution results.
        """
        initial_state: PlanExecute = {
            "input": user_input,
            "plan": [],
            "past_steps": [],
            "response": "",
            "plan_quality_score": 0.0,
            "execution_success_score": 1.0,
            "step_count": 0,
            "max_steps": 10,  # Maximum steps before forcing replan
            "response_quality_score": 0.0,
            "error_count": 0,
            "execution_failed": False,
        }
        result = graph.invoke(initial_state)
        return result


if __name__ == "__main__":
    os_instance.history.clear()
    logger.info("Running plan-and-execute agent with streaming updates.")

    initial_state: PlanExecute = {
        "input": "what is the hometown of the mens 2024 Australia open winner?",
        "plan": [],
        "past_steps": [],
        "response": "",
        "plan_quality_score": 0.0,
        "execution_success_score": 1.0,
        "step_count": 0,
        "max_steps": 10,
        "response_quality_score": 0.0,
        "error_count": 0,
        "execution_failed": False,
    }

    final_state: PlanExecute | None = None
    for chunk in graph.stream(initial_state, stream_mode="values", debug=False):
        logger.info("State update: %s", chunk)
        final_state = chunk

    print_history(os_instance.history)

    if final_state:
        logger.info("Final response: %s", final_state.get("response", "No response generated"))

