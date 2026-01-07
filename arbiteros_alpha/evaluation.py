"""Evaluation framework for assessing node execution quality.

This module provides a framework for evaluating the quality of node executions
after they complete. Unlike PolicyCheckers which validate constraints,
Evaluators provide quality scores and feedback for improvement.

Typical use cases:
- RL training: Collect reward signals for agent optimization
- Quality monitoring: Track execution quality in production
- A/B testing: Compare different implementations
- Self-improvement: Identify areas for refinement
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .history import History

__all__ = ["EvaluationResult", "NodeEvaluator", "ThresholdEvaluator"]


@dataclass
class EvaluationResult:
    """Result of evaluating a node execution.

    Attributes:
        score: Quality score in range [0.0, 1.0], where higher is better.
        passed: Whether the execution passed the evaluation criteria.
        feedback: Human-readable explanation of the evaluation.
        metadata: Additional data for logging, RL training, etc.
    """

    score: float
    passed: bool
    feedback: str
    metadata: dict[str, Any] = field(default_factory=dict)


class NodeEvaluator(ABC):
    """Base class for node execution evaluators.

    Evaluators assess the quality of node executions after they complete.
    Unlike PolicyCheckers, evaluators do not block execution - they provide
    feedback and scores that can be used for monitoring, RL training, or
    self-improvement.

    The evaluate() method receives the complete execution history, with the
    current node as the last entry (`history.entries[-1][-1]`).

    Attributes:
        name: Unique identifier for this evaluator.
        target_instructions: Optional list of instruction types to evaluate.
            If None, evaluates all nodes. If specified, only evaluates nodes
            matching the listed instruction types.

    Example:
        ```python
        from arbiteros_alpha.instructions import CognitiveCore

        class ResponseLengthEvaluator(NodeEvaluator):
            def __init__(self):
                super().__init__(
                    name="response_length",
                    target_instructions=[CognitiveCore.GENERATE]
                )

            def evaluate(self, history):
                current = history.entries[-1][-1]  # Get most recent node
                response = current.output_state.get("response", "")
                score = min(len(response) / 100, 1.0)
                return EvaluationResult(
                    score=score,
                    passed=score > 0.5,
                    feedback=f"Response length: {len(response)} chars"
                )
        ```
    """

    def __init__(self, name: str, target_instructions: list = None):
        """Initialize the evaluator.

        Args:
            name: Unique identifier for this evaluator.
            target_instructions: Optional list of InstructionType enums to evaluate.
                If None (default), evaluates all node types.
                If provided, only evaluates nodes with matching instruction types.
        """
        self.name = name
        self.target_instructions = target_instructions

    @abstractmethod
    def evaluate(self, history: "History") -> EvaluationResult:
        """Evaluate the most recent node execution.

        This method is called after a node completes execution. The node's
        HistoryItem (including output_state) has already been added to history
        and can be accessed via `history.entries[-1][-1]`.

        Args:
            history: Complete execution history. The current node is the last entry.

        Returns:
            EvaluationResult containing score, pass/fail status, and feedback.

        Note:
            - Simple evaluators can just examine the current node's input/output
            - Complex evaluators can traverse the full history for context
            - Evaluators should not raise exceptions; return low scores instead
        """
        pass

    def __repr__(self) -> str:
        """Return string representation of the evaluator."""
        return f"{self.__class__.__name__}(name={self.name!r})"


class ThresholdEvaluator(NodeEvaluator):
    """Simple evaluator that checks if a metric exceeds a threshold.

    This evaluator examines a specific key in the output state and compares
    its value against a threshold. Useful for basic quality checks.

    Attributes:
        name: Inherited from NodeEvaluator.
        key: The state key to evaluate.
        threshold: Minimum value for the evaluation to pass.

    Example:
        ```python
        evaluator = ThresholdEvaluator(
            name="confidence_check",
            key="confidence",
            threshold=0.7
        )
        # Will check if output_state["confidence"] >= 0.7
        ```
    """

    def __init__(
        self, name: str, key: str, threshold: float, target_instructions: list = None
    ):
        """Initialize the threshold evaluator.

        Args:
            name: Unique identifier for this evaluator.
            key: Key in output_state to evaluate.
            threshold: Minimum value for passing evaluation.
            target_instructions: Optional list of instruction types to evaluate.
        """
        super().__init__(name, target_instructions)
        self.key = key
        self.threshold = threshold

    def evaluate(self, history: "History") -> EvaluationResult:
        """Evaluate if the metric exceeds the threshold.

        Args:
            history: Complete execution history.

        Returns:
            EvaluationResult with score equal to the metric value,
            passing if value >= threshold.
        """
        current_item = history.entries[-1][-1]
        value = current_item.output_state.get(self.key, 0.0)
        passed = value >= self.threshold

        return EvaluationResult(
            score=value,
            passed=passed,
            feedback=f"{self.key}={value:.2f} ({'✓' if passed else '✗'} threshold={self.threshold})",
            metadata={"key": self.key, "threshold": self.threshold, "value": value},
        )

    def __repr__(self) -> str:
        """Return string representation of the evaluator."""
        return f"ThresholdEvaluator(name={self.name!r}, key={self.key!r}, threshold={self.threshold})"
