"""Unit tests for the rollout decorator functionality.

Tests cover:
- History reset on rollout start
- Execution lifecycle
- Exception handling
- Nested instruction tracking
"""

import pytest

from arbiteros_alpha import ArbiterOSAlpha
from arbiteros_alpha.instructions import CognitiveCore


class TestRollout:
    """Test cases for the rollout decorator."""

    def test_rollout_resets_history(self):
        """Test that rollout resets history on each invocation."""
        # Arrange
        os = ArbiterOSAlpha()

        # Manually pollute history
        os.history.enter_next_superstep(["previous_node"])

        @os.rollout()
        def my_workflow(input_data):
            # Assert inside: history should be empty (or reset) at start of execution
            # Note: history.entries is initialized as [] in History.__init__
            return len(os.history.entries) == 0

        # Act
        is_history_empty = my_workflow("test")

        # Assert
        assert is_history_empty is True
        # And ensure it works on second call too
        os.history.enter_next_superstep(["another_pollute"])
        is_history_empty_2 = my_workflow("test2")
        assert is_history_empty_2 is True

    def test_rollout_execution_flow(self):
        """Test that the decorated function executes and returns correctly."""
        # Arrange
        os = ArbiterOSAlpha()

        @os.rollout()
        def simple_workflow(x, y):
            return x + y

        # Act
        result = simple_workflow(10, 20)

        # Assert
        assert result == 30

    def test_rollout_propagates_exceptions(self):
        """Test that exceptions within rollout are logged and re-raised."""
        # Arrange
        os = ArbiterOSAlpha()

        @os.rollout()
        def failing_workflow():
            raise ValueError("Something went wrong")

        # Act & Assert
        with pytest.raises(ValueError, match="Something went wrong"):
            failing_workflow()

    def test_rollout_with_instructions(self):
        """Test that instructions inside a rollout are recorded in the new history."""
        # Arrange
        os = ArbiterOSAlpha(backend="native")

        @os.instruction(CognitiveCore.GENERATE)
        def generate_step(state):
            return {"result": "generated"}

        @os.rollout()
        def agent_workflow(start_state):
            # native backend requires explicit superstep declaration or similar mechanism
            # or relies on the instruction decorator to add to current history.
            # In native mode, instruction decorator calls enter_next_superstep if needed
            # or we need to manage it if we want strict supersteps.
            # However, core.py instruction wrapper calls enter_next_superstep automatically for native backend?
            # Let's check core.py:
            # if self.backend == "native": self.history.enter_next_superstep([instruction_type.name])
            return generate_step(start_state)

        # Act
        # First run
        agent_workflow({"input": 1})

        # Assert - First Run
        assert len(os.history.entries) == 1
        assert os.history.entries[0][0].instruction == CognitiveCore.GENERATE
        assert os.history.entries[0][0].input_state == {"state": {"input": 1}}

        # Act - Second Run (should clear history)
        agent_workflow({"input": 2})

        # Assert - Second Run
        assert (
            len(os.history.entries) == 1
        )  # Should be 1, not 2, because history was reset
        assert os.history.entries[0][0].input_state == {"state": {"input": 2}}
