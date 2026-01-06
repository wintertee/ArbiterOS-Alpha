"""Integration tests for examples.

This module tests all example files to ensure they run without errors.
These tests verify end-to-end functionality by executing complete workflows,
but do not check specific output values due to LLM non-determinism.
"""

import importlib.util
import logging
from pathlib import Path

import pytest

# Get the examples directory
EXAMPLES_DIR = Path(__file__).parent.parent / "examples"


def get_example_files() -> list[str]:
    """Get all Python example files from the examples directory.

    Returns:
        List of example file names without .py extension.
    """
    return [
        f.stem
        for f in EXAMPLES_DIR.glob("*.py")
        if f.is_file() and not f.name.startswith("_")
    ]


def run_example_module(example_name: str) -> None:
    """Run an example module by importing and executing its main function.

    Args:
        example_name: Name of the example file without .py extension.

    Raises:
        Any exception raised by the example module.
    """
    module_path = EXAMPLES_DIR / f"{example_name}.py"

    # Load the module dynamically with a unique name to avoid conflicts
    spec = importlib.util.spec_from_file_location(
        f"example_{example_name}", module_path
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load spec for {example_name}")

    module = importlib.util.module_from_spec(spec)

    # Suppress logging output during tests
    logging.disable(logging.CRITICAL)

    try:
        # Execute the module
        spec.loader.exec_module(module)

        # Call main() if it exists
        if hasattr(module, "main"):
            module.main()
    finally:
        # Re-enable logging
        logging.disable(logging.NOTSET)


@pytest.mark.parametrize("example_name", get_example_files())
def test_example_runs_without_error(example_name: str) -> None:
    """Test that an example file runs without raising exceptions.

    This is an integration test that verifies end-to-end functionality.
    Success means the example executed completely without errors.

    Args:
        example_name: Name of the example file (parametrized by pytest).
    """
    run_example_module(example_name)
