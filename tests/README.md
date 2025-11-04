# ArbiterOS-alpha Test Suite

This directory contains comprehensive unit tests for the ArbiterOS-alpha project.

## Test Structure

- **`test_policy.py`**: Tests for policy classes
  - `HistoryPolicyChecker`: Sequence validation logic
  - `MetricThresholdPolicyRouter`: Threshold-based routing

- **`test_arbiter_os.py`**: Tests for core ArbiterOS functionality
  - `ArbiterOSAlpha`: Main coordinator class
  - `History`: Execution history dataclass
  - Instruction decorator behavior
  - Policy checker and router integration

- **`conftest.py`**: Shared pytest fixtures for test setup

## Running Tests

Run all tests:
```bash
uv run pytest tests/
```

Run with verbose output:
```bash
uv run pytest tests/ -v
```

Run specific test file:
```bash
uv run pytest tests/test_policy.py
```

Run specific test class:
```bash
uv run pytest tests/test_policy.py::TestHistoryPolicyChecker
```

Run specific test method:
```bash
uv run pytest tests/test_policy.py::TestHistoryPolicyChecker::test_init_converts_sequence_to_string
```

Run with coverage:
```bash
uv run pytest tests/ --cov=arbiteros_alpha
```

## Test Coverage

Current test coverage includes:

### Policy Module (100%)
- ✅ HistoryPolicyChecker initialization
- ✅ Sequence blacklist validation
- ✅ Edge cases (empty history, multi-step sequences)
- ✅ MetricThresholdPolicyRouter threshold logic
- ✅ Router behavior with missing keys
- ✅ Different metric keys

### Core Module (100%)
- ✅ ArbiterOSAlpha initialization
- ✅ Policy registration (checkers and routers)
- ✅ Policy execution (check_before, route_after)
- ✅ Instruction decorator functionality
- ✅ History tracking
- ✅ LangGraph Command integration for routing
- ✅ History dataclass creation

## Test Guidelines

All tests follow the **Arrange-Act-Assert** pattern:

```python
def test_example(self):
    """Test description following Google style."""
    # Arrange - Set up test conditions
    os = ArbiterOSAlpha()
    
    # Act - Execute the code being tested
    result = os.some_method()
    
    # Assert - Verify expected outcomes
    assert result == expected_value
```

### Writing New Tests

1. Place tests in the appropriate test file
2. Use descriptive test method names: `test_<feature>_<scenario>`
3. Add comprehensive docstrings
4. Follow Arrange-Act-Assert pattern
5. Test both success and failure cases
6. Use fixtures from `conftest.py` when appropriate

### Example

```python
def test_policy_checker_detects_violation(self):
    """Test that policy checker correctly detects blacklisted sequences."""
    # Arrange
    checker = HistoryPolicyChecker(name="test", bad_sequence=["a", "b"])
    history = [create_history("a"), create_history("b")]
    
    # Act
    result = checker.check_before(history)
    
    # Assert
    assert result is False
```
