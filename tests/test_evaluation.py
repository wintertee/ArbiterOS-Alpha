"""Tests for the evaluation module."""

import pytest

from arbiteros_alpha import ArbiterOSAlpha
from arbiteros_alpha.evaluation import (
    EvaluationResult,
    NodeEvaluator,
    ThresholdEvaluator,
)
from arbiteros_alpha.history import History
from arbiteros_alpha.instructions import CognitiveCore, NormativeCore


class TestEvaluationResult:
    """Tests for EvaluationResult dataclass."""

    def test_evaluation_result_creation(self):
        """Test creating an EvaluationResult with all fields."""
        result = EvaluationResult(
            score=0.8,
            passed=True,
            feedback="Good quality",
            metadata={"key": "value"},
        )

        assert result.score == 0.8
        assert result.passed is True
        assert result.feedback == "Good quality"
        assert result.metadata == {"key": "value"}

    def test_evaluation_result_with_defaults(self):
        """Test creating an EvaluationResult with default metadata."""
        result = EvaluationResult(score=0.5, passed=False, feedback="Needs improvement")

        assert result.score == 0.5
        assert result.passed is False
        assert result.feedback == "Needs improvement"
        assert result.metadata == {}


class TestNodeEvaluator:
    """Tests for the NodeEvaluator base class."""

    def test_evaluator_must_be_subclassed(self):
        """Test that NodeEvaluator cannot be instantiated directly."""
        with pytest.raises(TypeError):
            NodeEvaluator("test")  # Abstract class

    def test_custom_evaluator(self):
        """Test creating a custom evaluator."""

        class CustomEvaluator(NodeEvaluator):
            def evaluate(self, history: History) -> EvaluationResult:
                current = history.entries[-1][-1]
                response_len = len(current.output_state.get("response", ""))
                score = min(response_len / 100, 1.0)
                return EvaluationResult(
                    score=score,
                    passed=score > 0.5,
                    feedback=f"Length: {response_len}",
                )

        evaluator = CustomEvaluator(name="custom")
        assert evaluator.name == "custom"
        assert "CustomEvaluator" in repr(evaluator)


class TestThresholdEvaluator:
    """Tests for the ThresholdEvaluator implementation."""

    def test_threshold_evaluator_init(self):
        """Test ThresholdEvaluator initialization."""
        evaluator = ThresholdEvaluator(name="test", key="confidence", threshold=0.7)

        assert evaluator.name == "test"
        assert evaluator.key == "confidence"
        assert evaluator.threshold == 0.7

    def test_threshold_evaluator_passes_when_above_threshold(self):
        """Test that evaluator passes when value exceeds threshold."""
        evaluator = ThresholdEvaluator(
            name="confidence_check", key="confidence", threshold=0.7
        )

        arbiter_os = ArbiterOSAlpha(backend="native")

        @arbiter_os.instruction(CognitiveCore.GENERATE)
        def generate(state):
            return {"confidence": 0.9}

        @arbiter_os.rollout()
        def run():
            arbiter_os.history.enter_next_superstep(["generate"])
            generate({"query": "test"})

        run()

        result = evaluator.evaluate(arbiter_os.history)

        assert result.score == 0.9
        assert result.passed is True
        assert "✓" in result.feedback
        assert result.metadata["value"] == 0.9

    def test_threshold_evaluator_fails_when_below_threshold(self):
        """Test that evaluator fails when value is below threshold."""
        evaluator = ThresholdEvaluator(
            name="confidence_check", key="confidence", threshold=0.7
        )

        arbiter_os = ArbiterOSAlpha(backend="native")

        @arbiter_os.instruction(CognitiveCore.GENERATE)
        def generate(state):
            return {"confidence": 0.5}

        @arbiter_os.rollout()
        def run():
            arbiter_os.history.enter_next_superstep(["generate"])
            generate({"query": "test"})

        run()

        result = evaluator.evaluate(arbiter_os.history)

        assert result.score == 0.5
        assert result.passed is False
        assert "✗" in result.feedback

    def test_threshold_evaluator_defaults_to_zero_when_key_missing(self):
        """Test that evaluator uses 0.0 when the key is missing."""
        evaluator = ThresholdEvaluator(
            name="confidence_check", key="confidence", threshold=0.7
        )

        arbiter_os = ArbiterOSAlpha(backend="native")

        @arbiter_os.instruction(CognitiveCore.GENERATE)
        def generate(state):
            return {}  # No confidence key

        @arbiter_os.rollout()
        def run():
            arbiter_os.history.enter_next_superstep(["generate"])
            generate({"query": "test"})

        run()

        result = evaluator.evaluate(arbiter_os.history)

        assert result.score == 0.0
        assert result.passed is False


class TestArbiterOSEvaluatorIntegration:
    """Integration tests for evaluators with ArbiterOSAlpha."""

    def test_add_evaluator(self):
        """Test adding an evaluator to ArbiterOSAlpha."""
        arbiter_os = ArbiterOSAlpha(backend="native")
        evaluator = ThresholdEvaluator(name="test", key="score", threshold=0.5)

        arbiter_os.add_evaluator(evaluator)

        assert len(arbiter_os.evaluators) == 1
        assert arbiter_os.evaluators[0] == evaluator

    def test_evaluator_runs_on_instruction_execution(self):
        """Test that evaluators run when instructions execute."""
        arbiter_os = ArbiterOSAlpha(backend="native")
        evaluator = ThresholdEvaluator(
            name="quality_check", key="quality", threshold=0.6
        )
        arbiter_os.add_evaluator(evaluator)

        @arbiter_os.instruction(CognitiveCore.GENERATE)
        def generate(state):
            return {"quality": 0.8}

        @arbiter_os.rollout()
        def run():
            arbiter_os.history.enter_next_superstep(["generate"])
            generate({"query": "test"})

        run()

        # Check that evaluation results were recorded in history
        history_item = arbiter_os.history.entries[-1][-1]
        assert "quality_check" in history_item.evaluation_results
        eval_result = history_item.evaluation_results["quality_check"]
        assert eval_result.score == 0.8
        assert eval_result.passed is True

    def test_multiple_evaluators_run_independently(self):
        """Test that multiple evaluators run independently."""
        arbiter_os = ArbiterOSAlpha(backend="native")

        eval1 = ThresholdEvaluator(name="eval1", key="metric1", threshold=0.5)
        eval2 = ThresholdEvaluator(name="eval2", key="metric2", threshold=0.7)

        arbiter_os.add_evaluator(eval1)
        arbiter_os.add_evaluator(eval2)

        @arbiter_os.instruction(CognitiveCore.GENERATE)
        def generate(state):
            return {"metric1": 0.6, "metric2": 0.9}

        @arbiter_os.rollout()
        def run():
            arbiter_os.history.enter_next_superstep(["generate"])
            generate({"query": "test"})

        run()

        history_item = arbiter_os.history.entries[-1][-1]
        assert len(history_item.evaluation_results) == 2
        assert history_item.evaluation_results["eval1"].score == 0.6
        assert history_item.evaluation_results["eval2"].score == 0.9

    def test_evaluator_failure_does_not_crash_execution(self):
        """Test that evaluator exceptions don't crash execution."""

        class BrokenEvaluator(NodeEvaluator):
            def evaluate(self, history: History) -> EvaluationResult:
                raise RuntimeError("Evaluator failed!")

        arbiter_os = ArbiterOSAlpha(backend="native")
        arbiter_os.add_evaluator(BrokenEvaluator(name="broken"))

        @arbiter_os.instruction(CognitiveCore.GENERATE)
        def generate(state):
            return {"result": "success"}

        @arbiter_os.rollout()
        def run():
            arbiter_os.history.enter_next_superstep(["generate"])
            return generate({"query": "test"})

        result = run()

        # Execution should complete despite evaluator failure
        assert result == {"result": "success"}
        history_item = arbiter_os.history.entries[-1][-1]
        # Broken evaluator should not have results recorded
        assert "broken" not in history_item.evaluation_results

    def test_evaluator_accesses_full_history(self):
        """Test that evaluators can access complete history."""

        class HistoryAwareEvaluator(NodeEvaluator):
            def evaluate(self, history: History) -> EvaluationResult:
                # Count total executions
                total_items = sum(len(superstep) for superstep in history.entries)
                return EvaluationResult(
                    score=1.0 if total_items > 1 else 0.0,
                    passed=total_items > 1,
                    feedback=f"Total executions: {total_items}",
                )

        arbiter_os = ArbiterOSAlpha(backend="native")
        arbiter_os.add_evaluator(HistoryAwareEvaluator(name="history_check"))

        @arbiter_os.instruction(CognitiveCore.GENERATE)
        def generate(state):
            return {"result": "ok"}

        @arbiter_os.rollout()
        def run_session():
            # First execution
            arbiter_os.history.enter_next_superstep(["generate"])
            generate({"query": "test1"})
            eval1 = arbiter_os.history.entries[-1][-1].evaluation_results[
                "history_check"
            ]
            assert eval1.passed is False  # Only 1 execution

            # Second execution
            arbiter_os.history.enter_next_superstep(["generate"])
            generate({"query": "test2"})
            eval2 = arbiter_os.history.entries[-1][-1].evaluation_results[
                "history_check"
            ]
            assert eval2.passed is True  # Now 2 executions

        run_session()


class TestEvaluatorFiltering:
    """Tests for instruction type filtering in evaluators."""

    def test_evaluator_with_target_instructions(self):
        """Test that evaluators can specify target instruction types."""

        class SelectiveEvaluator(NodeEvaluator):
            def evaluate(self, history: History) -> EvaluationResult:
                return EvaluationResult(score=1.0, passed=True, feedback="evaluated")

        # Create evaluator targeting only GENERATE
        evaluator = SelectiveEvaluator(
            name="selective", target_instructions=[CognitiveCore.GENERATE]
        )
        assert evaluator.target_instructions == [CognitiveCore.GENERATE]

    def test_evaluator_skips_non_target_instructions(self):
        """Test that evaluators skip nodes not in target_instructions."""
        evaluated_instructions = []

        class TrackingEvaluator(NodeEvaluator):
            def evaluate(self, history: History) -> EvaluationResult:
                current = history.entries[-1][-1]
                evaluated_instructions.append(current.instruction)
                return EvaluationResult(score=1.0, passed=True, feedback="evaluated")

        arbiter_os = ArbiterOSAlpha(backend="native")
        # Only evaluate GENERATE nodes
        arbiter_os.add_evaluator(
            TrackingEvaluator(
                name="tracking", target_instructions=[CognitiveCore.GENERATE]
            )
        )

        @arbiter_os.instruction(CognitiveCore.GENERATE)
        def generate(state):
            return {"response": "test"}

        @arbiter_os.instruction(CognitiveCore.REFLECT)
        def reflect(state):
            return {"critique": "good"}

        # Execute both nodes
        @arbiter_os.rollout()
        def run_test():
            arbiter_os.history.enter_next_superstep(["generate"])
            generate({})
            arbiter_os.history.enter_next_superstep(["reflect"])
            reflect({})

        run_test()

        # Only GENERATE should have been evaluated
        assert len(evaluated_instructions) == 1
        assert evaluated_instructions[0] == CognitiveCore.GENERATE

    def test_evaluator_without_filter_evaluates_all(self):
        """Test that evaluators without target_instructions evaluate all nodes."""
        evaluated_count = [0]

        class UnfilteredEvaluator(NodeEvaluator):
            def evaluate(self, history: History) -> EvaluationResult:
                evaluated_count[0] += 1
                return EvaluationResult(score=1.0, passed=True, feedback="evaluated")

        arbiter_os = ArbiterOSAlpha(backend="native")
        # No target_instructions = evaluate everything
        arbiter_os.add_evaluator(UnfilteredEvaluator(name="unfiltered"))

        @arbiter_os.instruction(CognitiveCore.GENERATE)
        def generate(state):
            return {"response": "test"}

        @arbiter_os.instruction(CognitiveCore.REFLECT)
        def reflect(state):
            return {"critique": "good"}

        # Execute both nodes
        @arbiter_os.rollout()
        def run_test():
            arbiter_os.history.enter_next_superstep(["generate"])
            generate({})
            arbiter_os.history.enter_next_superstep(["reflect"])
            reflect({})

        run_test()

        # Both should have been evaluated
        assert evaluated_count[0] == 2

    def test_evaluator_multiple_target_instructions(self):
        """Test that evaluators can target multiple instruction types."""
        evaluated_instructions = []

        class MultiTargetEvaluator(NodeEvaluator):
            def evaluate(self, history: History) -> EvaluationResult:
                current = history.entries[-1][-1]
                evaluated_instructions.append(current.instruction)
                return EvaluationResult(score=1.0, passed=True, feedback="evaluated")

        arbiter_os = ArbiterOSAlpha(backend="native")
        # Target both GENERATE and REFLECT
        arbiter_os.add_evaluator(
            MultiTargetEvaluator(
                name="multi",
                target_instructions=[CognitiveCore.GENERATE, CognitiveCore.REFLECT],
            )
        )

        @arbiter_os.instruction(CognitiveCore.GENERATE)
        def generate(state):
            return {"response": "test"}

        @arbiter_os.instruction(CognitiveCore.REFLECT)
        def reflect(state):
            return {"critique": "good"}

        @arbiter_os.instruction(NormativeCore.VERIFY)
        def verify(state):
            return {}

        # Execute all three nodes
        @arbiter_os.rollout()
        def run_test():
            arbiter_os.history.enter_next_superstep(["generate"])
            generate({})
            arbiter_os.history.enter_next_superstep(["reflect"])
            reflect({})
            arbiter_os.history.enter_next_superstep(["verify"])
            verify({})

        run_test()

        # Only GENERATE and REFLECT should be evaluated
        assert len(evaluated_instructions) == 2
        assert CognitiveCore.GENERATE in evaluated_instructions
        assert CognitiveCore.REFLECT in evaluated_instructions
        assert NormativeCore.VERIFY not in evaluated_instructions
