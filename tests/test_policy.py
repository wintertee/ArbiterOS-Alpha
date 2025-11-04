"""Unit tests for policy.py module.

Tests cover:
- HistoryPolicyChecker: Sequence validation logic
- MetricThresholdPolicyRouter: Threshold-based routing
"""

import datetime

from arbiteros_alpha import History
from arbiteros_alpha.policy import HistoryPolicyChecker, MetricThresholdPolicyRouter


class TestHistoryPolicyChecker:
    """Test cases for HistoryPolicyChecker."""

    def test_init_converts_sequence_to_string(self):
        """Test that __post_init__ joins sequence list into string."""
        # Arrange & Act
        checker = HistoryPolicyChecker(
            name="test_checker", bad_sequence=["generate", "toolcall"]
        )

        # Assert
        assert checker.bad_sequence == "generate->toolcall"

    def test_check_before_passes_when_no_blacklisted_sequence(self):
        """Test that check_before returns True when sequence is not blacklisted."""
        # Arrange
        checker = HistoryPolicyChecker(
            name="no_direct_toolcall", bad_sequence=["generate", "toolcall"]
        )
        history = [
            History(
                timestamp=datetime.datetime.now(),
                instruction="generate",
                input_state={},
                output_state={},
            ),
            History(
                timestamp=datetime.datetime.now(),
                instruction="evaluate",
                input_state={},
                output_state={},
            ),
        ]

        # Act
        result = checker.check_before(history)

        # Assert
        assert result is True

    def test_check_before_fails_when_blacklisted_sequence_detected(self):
        """Test that check_before returns False when blacklisted sequence is found."""
        # Arrange
        checker = HistoryPolicyChecker(
            name="no_direct_toolcall", bad_sequence=["generate", "toolcall"]
        )
        history = [
            History(
                timestamp=datetime.datetime.now(),
                instruction="generate",
                input_state={},
                output_state={},
            ),
            History(
                timestamp=datetime.datetime.now(),
                instruction="toolcall",
                input_state={},
                output_state={},
            ),
        ]

        # Act
        result = checker.check_before(history)

        # Assert
        assert result is False

    def test_check_before_detects_sequence_in_middle(self):
        """Test that blacklisted sequences are detected in the middle of history."""
        # Arrange
        checker = HistoryPolicyChecker(
            name="test", bad_sequence=["generate", "toolcall"]
        )
        history = [
            History(
                timestamp=datetime.datetime.now(),
                instruction="start",
                input_state={},
                output_state={},
            ),
            History(
                timestamp=datetime.datetime.now(),
                instruction="generate",
                input_state={},
                output_state={},
            ),
            History(
                timestamp=datetime.datetime.now(),
                instruction="toolcall",
                input_state={},
                output_state={},
            ),
            History(
                timestamp=datetime.datetime.now(),
                instruction="end",
                input_state={},
                output_state={},
            ),
        ]

        # Act
        result = checker.check_before(history)

        # Assert
        assert result is False

    def test_check_before_with_empty_history(self):
        """Test check_before with empty history returns True."""
        # Arrange
        checker = HistoryPolicyChecker(
            name="test", bad_sequence=["generate", "toolcall"]
        )
        history = []

        # Act
        result = checker.check_before(history)

        # Assert
        assert result is True

    def test_check_before_with_three_step_sequence(self):
        """Test blacklisting sequences with more than two steps."""
        # Arrange
        checker = HistoryPolicyChecker(name="test", bad_sequence=["a", "b", "c"])
        history = [
            History(
                timestamp=datetime.datetime.now(),
                instruction="a",
                input_state={},
                output_state={},
            ),
            History(
                timestamp=datetime.datetime.now(),
                instruction="b",
                input_state={},
                output_state={},
            ),
            History(
                timestamp=datetime.datetime.now(),
                instruction="c",
                input_state={},
                output_state={},
            ),
        ]

        # Act
        result = checker.check_before(history)

        # Assert
        assert result is False


class TestMetricThresholdPolicyRouter:
    """Test cases for MetricThresholdPolicyRouter."""

    def test_route_after_returns_target_when_below_threshold(self):
        """Test that router returns target when metric is below threshold."""
        # Arrange
        router = MetricThresholdPolicyRouter(
            name="regenerate_on_low_confidence",
            key="confidence",
            threshold=0.6,
            target="generate",
        )
        history = [
            History(
                timestamp=datetime.datetime.now(),
                instruction="evaluate",
                input_state={},
                output_state={"confidence": 0.4},
            )
        ]

        # Act
        result = router.route_after(history)

        # Assert
        assert result == "generate"

    def test_route_after_returns_none_when_above_threshold(self):
        """Test that router returns None when metric meets threshold."""
        # Arrange
        router = MetricThresholdPolicyRouter(
            name="regenerate_on_low_confidence",
            key="confidence",
            threshold=0.6,
            target="generate",
        )
        history = [
            History(
                timestamp=datetime.datetime.now(),
                instruction="evaluate",
                input_state={},
                output_state={"confidence": 0.8},
            )
        ]

        # Act
        result = router.route_after(history)

        # Assert
        assert result is None

    def test_route_after_returns_none_when_exactly_at_threshold(self):
        """Test that router returns None when metric equals threshold."""
        # Arrange
        router = MetricThresholdPolicyRouter(
            name="test", key="confidence", threshold=0.6, target="generate"
        )
        history = [
            History(
                timestamp=datetime.datetime.now(),
                instruction="evaluate",
                input_state={},
                output_state={"confidence": 0.6},
            )
        ]

        # Act
        result = router.route_after(history)

        # Assert
        assert result is None

    def test_route_after_defaults_to_1_when_key_missing(self):
        """Test that missing metric key defaults to 1.0 (passing threshold)."""
        # Arrange
        router = MetricThresholdPolicyRouter(
            name="test", key="confidence", threshold=0.6, target="generate"
        )
        history = [
            History(
                timestamp=datetime.datetime.now(),
                instruction="evaluate",
                input_state={},
                output_state={},  # No confidence key
            )
        ]

        # Act
        result = router.route_after(history)

        # Assert
        assert result is None  # 1.0 >= 0.6, so no routing

    def test_route_after_with_different_metric_key(self):
        """Test router works with different metric keys."""
        # Arrange
        router = MetricThresholdPolicyRouter(
            name="quality_check", key="quality_score", threshold=0.7, target="improve"
        )
        history = [
            History(
                timestamp=datetime.datetime.now(),
                instruction="check",
                input_state={},
                output_state={"quality_score": 0.5},
            )
        ]

        # Act
        result = router.route_after(history)

        # Assert
        assert result == "improve"

    def test_route_after_checks_last_entry_only(self):
        """Test that router only checks the most recent history entry."""
        # Arrange
        router = MetricThresholdPolicyRouter(
            name="test", key="confidence", threshold=0.6, target="generate"
        )
        history = [
            History(
                timestamp=datetime.datetime.now(),
                instruction="first",
                input_state={},
                output_state={"confidence": 0.3},  # Below threshold
            ),
            History(
                timestamp=datetime.datetime.now(),
                instruction="second",
                input_state={},
                output_state={"confidence": 0.9},  # Above threshold
            ),
        ]

        # Act
        result = router.route_after(history)

        # Assert
        assert result is None  # Should use last entry (0.9)
