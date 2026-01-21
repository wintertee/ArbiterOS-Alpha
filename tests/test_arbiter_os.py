"""Unit tests for ArbiterOSAlpha core functionality.

Tests cover:
- History tracking
- Policy checker registration and execution
- Policy router registration and execution
- Instruction decorator behavior
"""

import datetime
from typing import Any

import pytest

from arbiteros_alpha import ArbiterOSAlpha, HistoryItem
from arbiteros_alpha.instructions import CognitiveCore, MetacognitiveCore
from arbiteros_alpha.policy import (
    HistoryPolicyChecker,
    MetricThresholdPolicyRouter,
)


class TestArbiterOSAlpha:
    """Test cases for ArbiterOSAlpha class."""

    def test_init_creates_empty_state(self):
        """Test that initialization creates empty history and policy lists."""
        # Arrange & Act
        os = ArbiterOSAlpha()

        # Assert
        assert os.history.entries == []
        assert os.policy_checkers == []
        assert os.policy_routers == []

    def test_add_policy_checker(self):
        """Test adding a policy checker to the OS."""
        # Arrange
        os = ArbiterOSAlpha()
        checker = HistoryPolicyChecker(
            name="test", bad_sequence=[CognitiveCore.GENERATE, CognitiveCore.DECOMPOSE]
        )

        # Act
        os.add_policy_checker(checker)

        # Assert
        assert len(os.policy_checkers) == 1
        assert os.policy_checkers[0] == checker

    def test_add_multiple_policy_checkers(self):
        """Test adding multiple policy checkers."""
        # Arrange
        os = ArbiterOSAlpha()
        checker1 = HistoryPolicyChecker(
            name="checker1",
            bad_sequence=[CognitiveCore.GENERATE, CognitiveCore.DECOMPOSE],
        )
        checker2 = HistoryPolicyChecker(
            name="checker2",
            bad_sequence=[CognitiveCore.REFLECT, CognitiveCore.GENERATE],
        )

        # Act
        os.add_policy_checker(checker1)
        os.add_policy_checker(checker2)

        # Assert
        assert len(os.policy_checkers) == 2
        assert os.policy_checkers[0] == checker1
        assert os.policy_checkers[1] == checker2

    def test_add_policy_router(self):
        """Test adding a policy router to the OS."""
        # Arrange
        os = ArbiterOSAlpha()
        router = MetricThresholdPolicyRouter(
            name="test", key="confidence", threshold=0.5, target="retry"
        )

        # Act
        os.add_policy_router(router)

        # Assert
        assert len(os.policy_routers) == 1
        assert os.policy_routers[0] == router

    def test_add_multiple_policy_routers(self):
        """Test adding multiple policy routers."""
        # Arrange
        os = ArbiterOSAlpha()
        router1 = MetricThresholdPolicyRouter(
            name="router1", key="confidence", threshold=0.5, target="retry"
        )
        router2 = MetricThresholdPolicyRouter(
            name="router2", key="quality", threshold=0.7, target="improve"
        )

        # Act
        os.add_policy_router(router1)
        os.add_policy_router(router2)

        # Assert
        assert len(os.policy_routers) == 2
        assert os.policy_routers[0] == router1
        assert os.policy_routers[1] == router2

    def test_check_before_with_no_checkers(self):
        """Test check_before returns empty results when no checkers are registered."""
        # Arrange
        os = ArbiterOSAlpha()

        # Act
        results, all_passed = os._check_before()

        # Assert
        assert results == {}
        assert all_passed is True

    def test_check_before_with_passing_checker(self):
        """Test check_before with a checker that passes."""
        # Arrange
        os = ArbiterOSAlpha()
        checker = HistoryPolicyChecker(
            name="test", bad_sequence=[CognitiveCore.GENERATE, CognitiveCore.DECOMPOSE]
        )
        os.add_policy_checker(checker)
        os.history.enter_next_superstep(["reflect"])
        os.history.add_entry(
            HistoryItem(
                timestamp=datetime.datetime.now(),
                instruction=CognitiveCore.REFLECT,
                input_state={},
                output_state={},
            )
        )

        # Act
        results, all_passed = os._check_before()

        # Assert
        assert results == {}  # Only failed checks are recorded
        assert all_passed is True

    def test_check_before_with_failing_checker(self):
        """Test check_before with a checker that fails."""
        # Arrange
        os = ArbiterOSAlpha()
        checker = HistoryPolicyChecker(
            name="no_ab", bad_sequence=[CognitiveCore.GENERATE, CognitiveCore.DECOMPOSE]
        )
        os.add_policy_checker(checker)
        # First superstep: generate
        os.history.enter_next_superstep(["generate"])
        os.history.add_entry(
            HistoryItem(
                timestamp=datetime.datetime.now(),
                instruction=CognitiveCore.GENERATE,
                input_state={},
                output_state={},
            )
        )
        # Second superstep: decompose
        os.history.enter_next_superstep(["decompose"])
        os.history.add_entry(
            HistoryItem(
                timestamp=datetime.datetime.now(),
                instruction=CognitiveCore.DECOMPOSE,
                input_state={},
                output_state={},
            )
        )

        # Act
        results, all_passed = os._check_before()

        # Assert
        assert "no_ab" in results
        assert results["no_ab"] is False
        assert all_passed is False

    def test_route_after_with_no_routers(self):
        """Test route_after returns empty results when no routers are registered."""
        # Arrange
        os = ArbiterOSAlpha()

        # Act
        results, destination = os._route_after()

        # Assert
        assert results == {}
        assert destination is None

    def test_route_after_with_router_that_does_not_trigger(self):
        """Test route_after when router doesn't trigger."""
        # Arrange
        os = ArbiterOSAlpha()
        router = MetricThresholdPolicyRouter(
            name="test", key="confidence", threshold=0.5, target="retry"
        )
        os.add_policy_router(router)
        os.history.enter_next_superstep(["evaluate"])
        os.history.add_entry(
            HistoryItem(
                timestamp=datetime.datetime.now(),
                instruction="evaluate",
                input_state={},
                output_state={"confidence": 0.8},
            )
        )

        # Act
        results, destination = os._route_after()

        # Assert
        assert results == {}
        assert destination is None

    def test_route_after_with_router_that_triggers(self):
        """Test route_after when router triggers."""
        # Arrange
        os = ArbiterOSAlpha()
        router = MetricThresholdPolicyRouter(
            name="regenerate", key="confidence", threshold=0.5, target="retry"
        )
        os.add_policy_router(router)
        os.history.enter_next_superstep(["evaluate"])
        os.history.add_entry(
            HistoryItem(
                timestamp=datetime.datetime.now(),
                instruction="evaluate",
                input_state={},
                output_state={"confidence": 0.3},
            )
        )

        # Act
        results, destination = os._route_after()

        # Assert
        assert "regenerate" in results
        assert results["regenerate"] == "retry"
        assert destination == "retry"

    def test_instruction_decorator_creates_history_entry(self):
        """Test that instruction decorator creates a history entry."""
        # Arrange
        os = ArbiterOSAlpha()

        @os.instruction(CognitiveCore.GENERATE)
        def test_func(state: dict[str, Any]) -> dict[str, Any]:
            return {"result": "success"}

        # Act
        @os.rollout()
        def run_test():
            os.history.enter_next_superstep(["test_func"])
            return test_func({"input": "data"})

        result = run_test()

        # Assert
        assert len(os.history.entries) == 1
        assert len(os.history.entries[0]) == 1
        assert os.history.entries[0][0].instruction == CognitiveCore.GENERATE
        assert os.history.entries[0][0].input_state == {"input": "data"}
        assert os.history.entries[0][0].output_state == {"result": "success"}
        assert result == {"result": "success"}

    def test_instruction_decorator_preserves_function_name(self):
        """Test that decorator preserves original function name."""
        # Arrange
        os = ArbiterOSAlpha()

        @os.instruction(CognitiveCore.GENERATE)
        def my_function(state: dict[str, Any]) -> dict[str, Any]:
            return state

        # Assert
        assert my_function.__name__ == "my_function"

    def test_instruction_decorator_with_policy_check_failure(self):
        """Test instruction decorator behavior when policy check fails."""
        # Arrange
        os = ArbiterOSAlpha()
        checker = HistoryPolicyChecker(
            name="no_ab", bad_sequence=[CognitiveCore.GENERATE, CognitiveCore.REFLECT]
        )
        os.add_policy_checker(checker)

        @os.instruction(CognitiveCore.GENERATE)
        def func_a(state: dict[str, Any]) -> dict[str, Any]:
            return {"step": "a"}

        @os.instruction(CognitiveCore.REFLECT)
        def func_b(state: dict[str, Any]) -> dict[str, Any]:
            return {"step": "b"}

        # Act
        @os.rollout()
        def run_test():
            os.history.enter_next_superstep(["func_a"])
            func_a({})
            os.history.enter_next_superstep(["func_b"])
            func_b({})

        run_test()

        # Assert
        assert len(os.history.entries) == 2
        assert os.history.entries[1][0].check_policy_results["no_ab"] is False

    def test_instruction_decorator_with_router_trigger(self):
        """Test instruction decorator with routing."""
        # Arrange
        os = ArbiterOSAlpha()
        router = MetricThresholdPolicyRouter(
            name="retry_router", key="confidence", threshold=0.5, target="retry"
        )
        os.add_policy_router(router)

        @os.instruction(MetacognitiveCore.EVALUATE_PROGRESS)
        def evaluate(state: dict[str, Any]) -> dict[str, Any]:
            return {"confidence": 0.3}

        # Act
        @os.rollout()
        def run_test():
            os.history.enter_next_superstep(["evaluate"])
            return evaluate({})

        result = run_test()

        # Assert
        from langgraph.types import Command

        assert isinstance(result, Command)
        assert result.update == {"confidence": 0.3}
        assert result.goto == "retry"

    def test_instruction_decorator_without_router_trigger(self):
        """Test instruction decorator returns normal result when no routing."""
        # Arrange
        os = ArbiterOSAlpha()
        router = MetricThresholdPolicyRouter(
            name="retry_router", key="confidence", threshold=0.5, target="retry"
        )
        os.add_policy_router(router)

        @os.instruction(MetacognitiveCore.EVALUATE_PROGRESS)
        def evaluate(state: dict[str, Any]) -> dict[str, Any]:
            return {"confidence": 0.8}

        # Act
        @os.rollout()
        def run_test():
            os.history.enter_next_superstep(["evaluate"])
            return evaluate({})

        result = run_test()

        # Assert
        assert result == {"confidence": 0.8}
        assert not hasattr(result, "goto")

    def test_history_timestamps_are_recorded(self):
        """Test that each history entry has a timestamp."""
        # Arrange
        os = ArbiterOSAlpha()

        @os.instruction(CognitiveCore.GENERATE)
        def test_func(state: dict[str, Any]) -> dict[str, Any]:
            return {}

        before_time = datetime.datetime.now()

        # Act
        @os.rollout()
        def run_test():
            os.history.enter_next_superstep(["test_func"])
            test_func({})

        run_test()

        after_time = datetime.datetime.now()

        # Assert
        assert len(os.history.entries) == 1
        assert len(os.history.entries[0]) == 1
        assert before_time <= os.history.entries[0][0].timestamp <= after_time

    def test_multiple_instructions_create_ordered_history(self):
        """Test that multiple instructions create ordered history entries."""
        # Arrange
        os = ArbiterOSAlpha()

        @os.instruction(CognitiveCore.GENERATE)
        def first(state: dict[str, Any]) -> dict[str, Any]:
            return {"step": 1}

        @os.instruction(CognitiveCore.DECOMPOSE)
        def second(state: dict[str, Any]) -> dict[str, Any]:
            return {"step": 2}

        @os.instruction(CognitiveCore.REFLECT)
        def third(state: dict[str, Any]) -> dict[str, Any]:
            return {"step": 3}

        # Act
        @os.rollout()
        def run_test():
            os.history.enter_next_superstep(["first"])
            first({})
            os.history.enter_next_superstep(["second"])
            second({})
            os.history.enter_next_superstep(["third"])
            third({})

        run_test()

        # Assert
        assert len(os.history.entries) == 3
        assert os.history.entries[0][0].instruction == CognitiveCore.GENERATE
        assert os.history.entries[1][0].instruction == CognitiveCore.DECOMPOSE
        assert os.history.entries[2][0].instruction == CognitiveCore.REFLECT
        assert (
            os.history.entries[0][0].timestamp
            <= os.history.entries[1][0].timestamp
            <= os.history.entries[2][0].timestamp
        )

    def test_instruction_decorator_rejects_invalid_type(self):
        """Test that instruction decorator rejects non-InstructionType arguments."""
        # Arrange
        os = ArbiterOSAlpha()

        # Act & Assert
        with pytest.raises(
            TypeError, match="must be an instance of one of the Core enums"
        ):

            @os.instruction("invalid_string")  # type: ignore
            def invalid_func(state: dict[str, Any]) -> dict[str, Any]:
                return state

    def test_instruction_decorator_rejects_wrong_enum(self):
        """Test that instruction decorator rejects enums that are not from instructions.py."""
        # Arrange
        os = ArbiterOSAlpha()
        from enum import Enum, auto

        class WrongEnum(Enum):
            WRONG = auto()

        # Act & Assert
        with pytest.raises(
            TypeError, match="must be an instance of one of the Core enums"
        ):

            @os.instruction(WrongEnum.WRONG)  # type: ignore
            def invalid_func(state: dict[str, Any]) -> dict[str, Any]:
                return state


class TestHistory:
    """Test cases for HistoryItem dataclass."""

    def test_history_creation_with_required_fields(self):
        """Test creating HistoryItem with only required fields."""
        # Arrange & Act
        timestamp = datetime.datetime.now()
        history = HistoryItem(
            timestamp=timestamp,
            instruction="test",
            input_state={"key": "value"},
        )

        # Assert
        assert history.timestamp == timestamp
        assert history.instruction == "test"
        assert history.input_state == {"key": "value"}
        assert history.output_state == {}
        assert history.check_policy_results == {}
        assert history.route_policy_results == {}

    def test_history_creation_with_all_fields(self):
        """Test creating HistoryItem with all fields."""
        # Arrange & Act
        timestamp = datetime.datetime.now()
        history = HistoryItem(
            timestamp=timestamp,
            instruction="test",
            input_state={"in": "data"},
            output_state={"out": "result"},
            check_policy_results={"checker1": True},
            route_policy_results={"router1": "target"},
        )

        # Assert
        assert history.timestamp == timestamp
        assert history.instruction == "test"
        assert history.input_state == {"in": "data"}
        assert history.output_state == {"out": "result"}
        assert history.check_policy_results == {"checker1": True}
        assert history.route_policy_results == {"router1": "target"}
