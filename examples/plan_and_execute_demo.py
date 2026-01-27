"""Plan-and-Execute Agent Demo.

This file is extracted from plan-and-execute.ipynb for migration to ArbiterOS.
It implements a "plan-and-execute" style agent inspired by Plan-and-Solve and Baby-AGI.

The core idea is to:
1. First come up with a multi-step plan
2. Go through that plan one item at a time
3. After accomplishing a task, revisit and modify the plan as appropriate
"""

import operator
from typing import Annotated, List, Literal, Tuple, Union

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field
from typing_extensions import TypedDict
from arbiteros_alpha import ArbiterOSAlpha
import arbiteros_alpha.instructions as Instr


# =============================================================================
# State Definition
# =============================================================================


class PlanExecute(TypedDict):
    """State for the plan-and-execute agent.

    Attributes:
        input: The original user input/question.
        plan: List of steps in the current plan.
        past_steps: Accumulated list of (step, result) tuples from execution.
        response: Final response to return to the user.
    """

    input: str
    plan: List[str]
    past_steps: Annotated[List[Tuple], operator.add]
    response: str


# =============================================================================
# Pydantic Models for Structured Output
# =============================================================================


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

    action: Union[Response, Plan] = Field(
        description="Action to perform. If you want to respond to user, use Response. "
        "If you need to further use tools to get the answer, use Plan."
    )


# =============================================================================
# LLM and Prompt Setup
# =============================================================================

# Will be initialized with custom base_url and api_key
llm: ChatOpenAI = None
agent_executor = None
planner = None
replanner = None


os = ArbiterOSAlpha(backend="langgraph")


def initialize_llm(api_key: str, base_url: str, model: str = "gpt-4o"):
    """Initialize the LLM components with custom API settings.

    Args:
        api_key: The API key for authentication.
        base_url: The base URL for the API endpoint.
        model: The model name to use.
    """
    global llm, agent_executor, planner, replanner

    llm = ChatOpenAI(model=model, temperature=0, api_key=api_key, base_url=base_url)

    # Create the execution agent (using a simple prompt, no tools for this demo)
    agent_prompt = "You are a helpful assistant that executes tasks step by step."
    agent_executor = create_react_agent(llm, [], prompt=agent_prompt)

    # Create the planner
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
    planner = planner_prompt | llm.with_structured_output(Plan)

    # Create the replanner
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
    replanner = replanner_prompt | llm.with_structured_output(Act)


# =============================================================================
# Node Functions
# =============================================================================


@os.instruction(Instr.TOOL_CALL)
async def execute_step(state: PlanExecute):
    """Execute the current step of the plan.

    Takes the first step from the plan and asks the agent to execute it.

    Args:
        state: Current agent state containing the plan.

    Returns:
        Updated state with the executed step added to past_steps.
    """
    plan = state["plan"]
    plan_str = "\n".join(f"{i + 1}. {step}" for i, step in enumerate(plan))
    task = plan[0]
    task_formatted = f"""For the following plan:
{plan_str}\n\nYou are tasked with executing step {1}, {task}."""
    agent_response = await agent_executor.ainvoke(
        {"messages": [("user", task_formatted)]}
    )
    return {
        "past_steps": [(task, agent_response["messages"][-1].content)],
    }


@os.instruction(Instr.DECOMPOSE)
async def plan_step(state: PlanExecute):
    """Create an initial plan based on the user input.

    Args:
        state: Current agent state containing the input.

    Returns:
        Updated state with the generated plan.
    """
    plan = await planner.ainvoke({"messages": [("user", state["input"])]})
    return {"plan": plan.steps}


@os.instruction(Instr.GENERATE)
async def replan_step(state: PlanExecute):
    """Replan based on execution results.

    Either generates a new plan or produces the final response.

    Args:
        state: Current agent state with input, plan, and past_steps.

    Returns:
        Updated state with either a new plan or final response.
    """
    output = await replanner.ainvoke(state)
    if isinstance(output.action, Response):
        return {"response": output.action.response}
    else:
        return {"plan": output.action.steps}


def should_end(state: PlanExecute) -> Literal["agent", "__end__"]:
    """Determine if execution should end.

    Args:
        state: Current agent state.

    Returns:
        Either END to finish or "agent" to continue execution.
    """
    if "response" in state and state["response"]:
        return END
    else:
        return "agent"


# =============================================================================
# Graph Construction
# =============================================================================


def build_graph():
    """Build and compile the plan-and-execute graph.

    Returns:
        Compiled LangGraph application.
    """
    workflow = StateGraph(PlanExecute)

    # Add the plan node
    workflow.add_node("planner", plan_step)

    # Add the execution step
    workflow.add_node("agent", execute_step)

    # Add a replan node
    workflow.add_node("replan", replan_step)

    workflow.add_edge(START, "planner")

    # From plan we go to agent
    workflow.add_edge("planner", "agent")

    # From agent, we replan
    workflow.add_edge("agent", "replan")

    workflow.add_conditional_edges(
        "replan",
        should_end,
        ["agent", END],
    )

    compiled = workflow.compile()
    os.register_compiled_graph(compiled)
    return compiled


# =============================================================================
# Main Entry Point
# =============================================================================


async def main():
    """Run the plan-and-execute agent demo."""
    import os

    # Get API configuration from environment or use defaults
    api_key = os.environ.get(
        "OPENAI_API_KEY", "sk-SWslLpubFhsK9zXTNqZTjdvQI1r3AFLT3Nsc4Y6DVSkpfSv7"
    )
    base_url = os.environ.get("OPENAI_BASE_URL", "https://a.fe8.cn/v1")
    model = os.environ.get("OPENAI_MODEL", "gpt-4o")

    # Initialize LLM components
    initialize_llm(api_key=api_key, base_url=base_url, model=model)

    # Build the graph
    app = build_graph()

    # Run a test query
    config = {"recursion_limit": 50}
    inputs = {"input": "What is 2 + 2? Then multiply the result by 3."}

    print("=" * 60)
    print("Plan-and-Execute Agent Demo")
    print("=" * 60)
    print(f"Input: {inputs['input']}")
    print("-" * 60)

    async for event in app.astream(inputs, config=config):
        for k, v in event.items():
            if k != "__end__":
                print(f"[{k}] {v}")
    print("=" * 60)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
