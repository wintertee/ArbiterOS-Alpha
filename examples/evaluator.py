"""Example: Using Node Evaluators for Quality Assessment.

This example demonstrates how to use NodeEvaluators to assess the quality
of node executions. Unlike PolicyCheckers which enforce constraints,
Evaluators provide quality scores and feedback without blocking execution.

Use cases:
- RL training: Collect reward signals
- Quality monitoring: Track execution quality over time
- A/B testing: Compare different implementations
- Self-improvement: Identify areas for refinement
"""

import logging
from typing import TypedDict

from rich.logging import RichHandler

from arbiteros_alpha import ArbiterOSAlpha
from arbiteros_alpha.evaluation import (
    EvaluationResult,
    NodeEvaluator,
    ThresholdEvaluator,
)
from arbiteros_alpha.history import History
from arbiteros_alpha.instructions import CognitiveCore, NormativeCore

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    handlers=[RichHandler()],
)


# 1. Setup OS
arbiter_os = ArbiterOSAlpha(backend="vanilla")


# 2. Define custom evaluators


class ResponseQualityEvaluator(NodeEvaluator):
    """Evaluates response quality based on length and content."""

    def __init__(self):
        super().__init__(
            name="response_quality",
            target_instructions=[CognitiveCore.GENERATE],  # Only evaluate GENERATE
        )

    def evaluate(self, history: History) -> EvaluationResult:
        """Assess response quality using multiple criteria."""
        current = history.entries[-1][-1]
        response = current.output_state.get("response", "")

        # Multiple quality metrics
        length_score = min(len(response) / 100, 1.0)  # Target: 100+ chars
        has_greeting = any(word in response.lower() for word in ["hello", "hi", "hey"])
        has_content = len(response.split()) > 5  # More than 5 words

        # Combined score
        score = (
            length_score * 0.5
            + (0.25 if has_greeting else 0)
            + (0.25 if has_content else 0)
        )
        passed = score >= 0.6

        feedback_parts = [
            f"length={len(response)} chars (score: {length_score:.2f})",
            f"greeting={'yes' if has_greeting else 'no'}",
            f"content={'sufficient' if has_content else 'insufficient'}",
        ]

        return EvaluationResult(
            score=score,
            passed=passed,
            feedback=f"Quality assessment: {', '.join(feedback_parts)}",
            metadata={
                "length": len(response),
                "has_greeting": has_greeting,
                "has_content": has_content,
            },
        )


class ConsistencyEvaluator(NodeEvaluator):
    """Evaluates consistency with previous responses."""

    def __init__(self):
        super().__init__(
            name="consistency_check",
            target_instructions=[CognitiveCore.GENERATE],  # Only evaluate GENERATE
        )

    def evaluate(self, history: History) -> EvaluationResult:
        """Check if current response is consistent with history."""
        # Get all previous GENERATE nodes
        all_items = [item for superstep in history.entries for item in superstep]
        generate_items = [
            item for item in all_items if item.instruction == CognitiveCore.GENERATE
        ]

        if len(generate_items) <= 1:
            return EvaluationResult(
                score=1.0,
                passed=True,
                feedback="First response, nothing to compare",
                metadata={"response_count": 1},
            )

        current_response = generate_items[-1].output_state.get("response", "")
        previous_responses = [
            item.output_state.get("response", "") for item in generate_items[:-1]
        ]

        # Simple consistency check: ensure tone is similar (both have/lack greeting)
        current_has_greeting = any(
            word in current_response.lower() for word in ["hello", "hi", "hey"]
        )
        previous_had_greeting = any(
            any(word in resp.lower() for word in ["hello", "hi", "hey"])
            for resp in previous_responses
        )

        consistent = current_has_greeting == previous_had_greeting
        score = 1.0 if consistent else 0.5

        return EvaluationResult(
            score=score,
            passed=consistent,
            feedback=f"Tone consistency: {'consistent' if consistent else 'inconsistent'} with previous responses",
            metadata={
                "response_count": len(generate_items),
                "current_has_greeting": current_has_greeting,
                "previous_had_greeting": previous_had_greeting,
            },
        )


# 3. Add evaluators to OS

# Built-in threshold evaluator for confidence (only for GENERATE nodes)
confidence_evaluator = ThresholdEvaluator(
    name="confidence_check",
    key="confidence",
    threshold=0.7,
    target_instructions=[CognitiveCore.GENERATE],  # Only evaluate GENERATE
)
arbiter_os.add_evaluator(confidence_evaluator)

# Custom evaluators
arbiter_os.add_evaluator(ResponseQualityEvaluator())
arbiter_os.add_evaluator(ConsistencyEvaluator())


# 4. Define state and functions


class State(TypedDict):
    """State for a simple AI assistant."""

    query: str
    response: str
    confidence: float


@arbiter_os.instruction(CognitiveCore.GENERATE)
def generate(state: State) -> State:
    """Generate a response to the user query."""
    query = state["query"]

    # Simulate different quality responses based on query
    if "formal" in query.lower():
        response = "I appreciate your inquiry. Let me provide a comprehensive answer to your question."
        confidence = 0.9
    elif "casual" in query.lower():
        response = "Hey! That's a great question!"
        confidence = 0.6
    else:
        response = "Sure, I can help with that. Here's what you need to know."
        confidence = 0.8

    return {"response": response, "confidence": confidence}


@arbiter_os.instruction(NormativeCore.VERIFY)
def verify(state: State) -> State:
    """Verify the response quality."""
    response = state["response"]
    # Simple verification: check if response is not empty
    verified = len(response) > 0
    logger.info(f"Verification result: {verified}")
    return {}


# 5. Run workflow with different scenarios


def main():
    """Run the workflow with different query types."""
    scenarios = [
        {
            "query": "Tell me about AI in a formal way",
            "response": "",
            "confidence": 0.0,
        },
        {
            "query": "Explain machine learning casually",
            "response": "",
            "confidence": 0.0,
        },
        {"query": "What is deep learning?", "response": "", "confidence": 0.0},
    ]

    for idx, initial_state in enumerate(scenarios, 1):
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Scenario {idx}: {initial_state['query']}")
        logger.info(f"{'=' * 60}\n")

        state = initial_state.copy()
        state.update(generate(state))

        state.update(verify(state))

    logger.info("\n" + "=" * 60)
    logger.info("Execution complete. Evaluation summary:")
    logger.info("=" * 60)

    # Print summary of evaluations
    for superstep_idx, superstep in enumerate(arbiter_os.history.entries, 1):
        for item_idx, item in enumerate(superstep, 1):
            if item.evaluation_results:
                logger.info(
                    f"\nSuperstep {superstep_idx}.{item_idx} - {item.instruction.name}:"
                )
                for eval_name, result in item.evaluation_results.items():
                    status = "✓" if result.passed else "✗"
                    logger.info(
                        f"  {status} {eval_name}: score={result.score:.2f}, {result.feedback}"
                    )


if __name__ == "__main__":
    main()
    print("\n")
    arbiter_os.history.pprint()
