"""Example demonstrating instruction schema validation.

This example shows how to use the schema validation feature to ensure
instruction functions follow defined input/output formats.
"""

import logging
from typing import TypedDict

from langgraph.graph import END, START, StateGraph
from rich.logging import RichHandler

import arbiteros_alpha.instructions as Instr
from arbiteros_alpha import ArbiterOSAlpha

logging.basicConfig(
    level=logging.INFO,
    handlers=[RichHandler()],
)

logger = logging.getLogger(__name__)

# 1. Setup OS with schema validation enabled
# Set validate_schemas=True to enable schema validation
# Set strict_schema_validation=True to require all schema fields
os = ArbiterOSAlpha(validate_schemas=True, strict_schema_validation=False)

# 2. Define state schema
class State(TypedDict):
    """State for a simple text generation workflow."""

    prompt: str
    content: str
    reasoning: str


# 3. Define instruction functions with proper schemas
@os.instruction(Instr.GENERATE)
def generate(state: State) -> State:
    """Generate content based on prompt.

    This function should return a dict matching GenerateOutputSchema:
    - content: str (required)
    - reasoning: str (required)
    """
    prompt = state.get("prompt", "")
    return {
        "content": f"Generated response to: {prompt}",
        "reasoning": "Used standard generation approach",
    }


@os.instruction(Instr.DECOMPOSE)
def decompose(state: State) -> State:
    """Decompose a task into subtasks.

    This function should return a dict matching DecomposeOutputSchema:
    - subtasks: list[str] (required)
    - plan: dict[str, Any] (required)
    """
    task = state.get("prompt", "")
    return {
        "subtasks": [f"Step 1: {task}", "Step 2: Review", "Step 3: Finalize"],
        "plan": {"steps": 3, "estimated_time": "10 minutes"},
    }


# 4. Query schemas for analysis
print("=== Schema Information ===")
input_schema, output_schema = os.get_instruction_schema(Instr.GENERATE)
print(f"GENERATE input schema: {input_schema.__name__}")
print(f"GENERATE output schema: {output_schema.__name__}")
print(f"GENERATE output required fields: {getattr(output_schema, '__required_keys__', 'N/A')}")

input_schema, output_schema = os.get_instruction_schema(Instr.DECOMPOSE)
print(f"\nDECOMPOSE input schema: {input_schema.__name__}")
print(f"DECOMPOSE output schema: {output_schema.__name__}")
print(f"DECOMPOSE output required fields: {getattr(output_schema, '__required_keys__', 'N/A')}")

# 5. Build and run graph
builder = StateGraph(State)
builder.add_node("generate", generate)
builder.add_node("decompose", decompose)

builder.add_edge(START, "generate")
builder.add_edge("generate", "decompose")
builder.add_edge("decompose", END)

graph = builder.compile()

# 6. Run with proper input format
initial_state: State = {
    "prompt": "Write a blog post about AI",
    "content": "",
    "reasoning": "",
}

print("\n=== Execution ===")
for chunk in graph.stream(initial_state, stream_mode="values"):
    logger.info(f"State: {chunk}")

print("\n=== History ===")
for entry in os.history:
    print(f"{entry.instruction.name}: {entry.input_state} -> {entry.output_state}")

