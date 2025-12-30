import logging
from typing import TypedDict

from langgraph.graph import END, START, StateGraph
from rich.logging import RichHandler

logger = logging.getLogger(__name__)


logging.basicConfig(
    level=logging.DEBUG,
    handlers=[RichHandler()],
)


class State(TypedDict):
    """State for a simple AI assistant with tool usage and self-evaluation."""

    query: str
    response: str
    tool_result: str
    confidence: float


def generate(state: State) -> State:
    """Generate a response to the user query."""

    # Check if this is a retry (response already exists)
    is_retry = bool(state.get("response"))

    if is_retry:
        # On retry, generate a longer, better response
        response = "Here is my comprehensive and detailed response with much more content and explanation."
    else:
        # First attempt: short response (will have low confidence)
        response = "Short reply."

    return {"response": response}


def tool_call(state: State) -> State:
    """Call external tools to enhance the response."""
    return {"tool_result": "ok"}


def evaluate(state: State) -> State:
    """Evaluate confidence in the response quality."""
    # Heuristic: response quality based on length
    # Short response (<60 chars) = low confidence (<0.6)
    # Longer response (>=60 chars) = high confidence (>=0.6)
    response_length = len(state["response"])
    confidence = min(response_length / 100.0, 1.0)
    return {"confidence": confidence}


builder = StateGraph(State)
builder.add_node(generate)
builder.add_node(tool_call)
builder.add_node(evaluate)

builder.add_edge(START, "generate")
builder.add_edge("generate", "tool_call")
builder.add_edge("tool_call", "evaluate")
builder.add_edge("evaluate", END)

graph = builder.compile()


def main():
    initial_state: State = {
        "query": "What is AI?",
        "response": "",
        "tool_result": "",
        "confidence": 0.0,
    }
    for chunk in graph.stream(initial_state, stream_mode="values", debug=False):
        logger.info(f"Current state: {chunk}\n")


if __name__ == "__main__":
    main()
