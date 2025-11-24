"""Pytest configuration and shared fixtures for ArbiterOS tests."""

import datetime

import pytest

from arbiteros_alpha import History


@pytest.fixture
def sample_history_entry():
    """Provide a sample History entry for testing.

    Returns:
        A History instance with basic test data.
    """
    return History(
        timestamp=datetime.datetime(2025, 11, 4, 12, 0, 0),
        instruction="test_instruction",
        node_name="test_function",
        input_state={"input_key": "input_value"},
        output_state={"output_key": "output_value"},
    )


@pytest.fixture
def sample_history_list():
    """Provide a list of History entries for testing.

    Returns:
        A list of 3 History instances representing a simple execution flow.
    """
    return [
        History(
            timestamp=datetime.datetime(2025, 11, 4, 12, 0, 0),
            instruction="generate",
            node_name="generate",
            input_state={"query": "test"},
            output_state={"response": "answer"},
        ),
        History(
            timestamp=datetime.datetime(2025, 11, 4, 12, 0, 1),
            instruction="evaluate",
            node_name="evaluate",
            input_state={"response": "answer"},
            output_state={"confidence": 0.8},
        ),
        History(
            timestamp=datetime.datetime(2025, 11, 4, 12, 0, 2),
            instruction="finish",
            node_name="finish",
            input_state={"confidence": 0.8},
            output_state={"status": "complete"},
        ),
    ]
