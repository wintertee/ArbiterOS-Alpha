"""Unit tests for the strict rollout context enforcement.

Tests cover:
- Requirement of rollout context for instructions
- Prohibition of nested rollouts
"""

import pytest
from arbiteros_alpha import ArbiterOSAlpha
from arbiteros_alpha.instructions import CognitiveCore


class TestRolloutStrictness:
    """Test cases for strict rollout context enforcement."""

    def test_instruction_fails_without_rollout(self):
        """Test that executing an instruction without an active rollout raises RuntimeError."""
        # Arrange
        os = ArbiterOSAlpha(backend="native")

        @os.instruction(CognitiveCore.GENERATE)
        def standalone_function(state: dict):
            return {"result": "ok"}

        # Act & Assert
        with pytest.raises(
            RuntimeError,
            match="Instructions must be executed within a @arbiter_os.rollout context",
        ):
            standalone_function({"input": "test"})

    def test_nested_rollout_fails(self):
        """Test that nesting rollouts raises RuntimeError."""
        # Arrange
        os = ArbiterOSAlpha()

        @os.rollout()
        def inner_workflow():
            return "inner"

        @os.rollout()
        def outer_workflow():
            return inner_workflow()

        # Act & Assert
        with pytest.raises(RuntimeError, match="Nested rollouts are not allowed"):
            outer_workflow()

    def test_rollout_state_resets_after_exception(self):
        """Test that the _in_rollout state is reset even if the rollout fails."""
        # Arrange
        os = ArbiterOSAlpha()

        @os.rollout()
        def failing_workflow():
            raise ValueError("Intentional failure")

        # Act
        try:
            failing_workflow()
        except ValueError:
            pass  # Ignore the error, we want to check the state

        # Assert
        assert os._in_rollout is False

        # Should be able to run a new rollout
        @os.rollout()
        def success_workflow():
            return "success"

        assert success_workflow() == "success"
