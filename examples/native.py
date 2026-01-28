import logging

from pydantic import BaseModel
from rich.logging import RichHandler

import arbiteros_alpha.instructions as Instr
from arbiteros_alpha import ArbiterOSAlpha
from arbiteros_alpha.policy import HistoryPolicyChecker

logger = logging.getLogger(__name__)


logging.basicConfig(
    level=logging.DEBUG,
    handlers=[RichHandler()],
)

# 1. Setup OS

arbiter_os = ArbiterOSAlpha(backend="native")

# Policy: Prevent direct generate->toolcall without proper flow
history_checker = HistoryPolicyChecker(
    name="no_direct_toolcall",
    bad_sequence=[Instr.GENERATE, Instr.TOOL_CALL],
)


# if you add this checker, intended error will be raised
arbiter_os.add_policy_checker(history_checker)

# 2. basic modules


class GenerateInput(BaseModel):
    query: str
    tools: list[str]


class GenerateOutput(BaseModel):
    tool: str
    args: dict


@arbiter_os.instruction(
    Instr.GENERATE, input_schema=GenerateInput, output_schema=GenerateOutput
)
def generate(query: str, tools: list[str]) -> dict:
    """Simulate an LLM deciding which tool to call."""
    print(f"Agent thinking about query: '{query}' with tools: {tools}...")
    # Simulate picking the first available tool
    selected_tool = tools[0] if tools else "none"
    return {"tool": selected_tool, "args": {"q": query}}


class ToolCallInput(BaseModel):
    tool_name: str
    tool_args: dict


class ToolCallOutput(BaseModel):
    result: str


@arbiter_os.instruction(
    Instr.TOOL_CALL, input_schema=ToolCallInput, output_schema=ToolCallOutput
)
def tool_call(tool_name: str, tool_args: dict) -> dict:
    """Execute the tool call suggested by the agent."""
    print(f"Executing tool '{tool_name}' with args {tool_args}...")
    result = f"Result from {tool_name} is 'AI is a field of computer science...'"
    return {"result": result}


class EvaluateInput(BaseModel):
    response: str
    criteria: list[str]


class EvaluateOutput(BaseModel):
    confidence: float


@arbiter_os.instruction(
    Instr.EVALUATE_PROGRESS, input_schema=EvaluateInput, output_schema=EvaluateOutput
)
def evaluate(response: str, criteria: list[str]) -> dict:
    """Evaluate confidence in the response quality."""
    # Heuristic: response quality based on length
    response_length = len(response)
    confidence = min(response_length / 100.0, 1.0)
    print(f"Evaluating against {criteria}...")
    return {"confidence": confidence}


@arbiter_os.rollout()
def main():
    # 3. Run instructions
    query = "What is AI?"
    available_tools = ["web_search", "calculator"]

    # 1. Agent decides what to do
    # Note: For native backend, we pass arguments that match the schema fields
    decision = generate(query=query, tools=available_tools)
    print(f"{decision=}\n")

    # 2. Execute the tool
    func_name = decision["tool"]
    func_args = decision["args"]
    tool_result = tool_call(tool_name=func_name, tool_args=func_args)
    print(f"{tool_result=}\n")

    # 3. Evaluate the result (using tool_result as the 'response' to evaluate)
    eval_result = evaluate(
        response=tool_result["result"], criteria=["informativeness", "accuracy"]
    )
    print(f"{eval_result=}\n")


if __name__ == "__main__":
    main()
    arbiter_os.history.pprint()
