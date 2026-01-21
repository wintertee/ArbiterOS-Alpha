This PR introduces two major changes:1.  **@os.rollout() Decorator:**
    *   Adds a new decorator to manage the lifecycle of an execution 'rollout'.
    *   Automatically resets the execution history at the start of each call.
    *   Logs the start, completion, and failure of rollouts.
    *   Eliminates the need for manual history management or session IDs in simple use cases.

2.  **'Native' Backend:**
    *   Renames the 'vanilla' backend to 'native' to better reflect its role as a framework-less, pure Python implementation.
    *   Maintains backward compatibility by keeping 'vanilla' as a valid option but raising a `DeprecationWarning`.

**Changes:**
*   Modified `arbiteros_alpha/core.py` to implement the decorator and rename logic.
*   Renamed `examples/vanilla.py` to `examples/native.py` and updated usages.
*   Updated `examples/evaluator.py`, `examples/langgraph.py`, and `examples/orchestrator.py` to use `@os.rollout()`.
*   Added `tests/test_rollout.py` to verify new functionality.
*   Updated existing tests to use the 'native' backend.\" --base main
Directory: (root)
