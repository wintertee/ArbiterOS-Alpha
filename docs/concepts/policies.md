# Policy Architecture

## Overview

In ArbiterOS, all governance policies can be classified into two fundamental implementation categories: **PolicyChecker** and **PolicyRouter**. This classification is derived from the timing and nature of policy enforcement in the agent execution lifecycle.

## The Two Policy Categories

### 1. PolicyChecker: Pre-Execution Validation

**PolicyChecker** enforces constraints **before** a node executes. It answers the question: *"Should this transition be allowed?"*

- **Timing**: Runs before the instruction node
- **Purpose**: Validate preconditions, enforce architectural constraints, prevent forbidden transitions
- **Output**: Binary decision (PASS/FAIL) with optional violation messages
- **Action**: If FAIL, execution is blocked and the violation is logged

### 2. PolicyRouter: Post-Execution Routing

**PolicyRouter** makes decisions **after** a node executes. It answers the question: *"Where should we go next based on the results?"*

- **Timing**: Runs after the instruction node
- **Purpose**: Dynamic flow control based on execution results (confidence scores, verification outcomes, performance metrics)
- **Output**: Target node ID or `None` (continue normal flow)
- **Action**: If a target is returned, deterministically route execution to that node

## Mapping Academic Policies to Implementation

Based on the ArbiterKernel paper's policy taxonomy, all policies map to these two categories:

### Static Verification (Linter) → **TODO**

---

### Dynamic Transition Verification → PolicyChecker

**Description**: At runtime, the policy engine intercepts impending state transitions, enforcing that a probabilistic step (e.g., GENERATE) must go through a deterministic verification step (e.g., VERIFY) before allowing a high-risk action (e.g., TOOL CALL).

**Implementation**: PolicyChecker  
**Why**: This enforces preconditions at runtime **before** execution, ensuring verification occurs before high-risk actions.

**Research Examples**:

1. **"Think then Verify" Workflow**  
   Policy `enforce_verify_before_action`: If the previous instruction is from Cognitive Core, prohibit the next instruction from being Execution Core unless a Normative Core instruction is inserted between them.

2. **Stateful Policies**  
   Policies based on the agent's "Managed State" data determine whether to allow an operation.  
   Example: "If `history[-1].input_state.has_flag('high_risk_user')` is true, prohibit TOOL CALL from invoking payment APIs."

3. **Temporal Policies (TODO)**  
   Enforce time-related constraints, such as rate limiting.  
   Example: "Allow TOOL CALL to invoke a specific API no more than once every 5 seconds."

4. **Resource-Based Policies (TODO)**  
   Decide whether to allow high-cost operations based on remaining "Reliability Budget".  
   Example: "Only allow expensive DECOMPOSE steps when remaining reliability budget is above 50%."

5. **Domain-Specific Policy (TODO)**  
   For custom cores or instructions, enforce specific preconditions.  
   Example: "Policy can require that compute-intensive EXECUTE BACKTEST instructions must be preceded by MONITOR RESOURCES check."

**Implementation Pattern**:
```python
@dataclass
class StateBasedPolicyChecker(PolicyChecker):
    """Checks state conditions before allowing execution."""
    
    def check(self, history: list[History], current_instruction: str) -> bool:
        latest_state = history[-1].output_state if history else {}
        
        # Example: Check reliability budget
        budget = latest_state.get("reliability_budget", 100)
        if current_instruction == "decompose" and budget < 50:
            return False
        
        # Example: Check rate limiting
        if self._exceeds_rate_limit(history, current_instruction):
            return False
            
        return True, ""
```

---

### Dynamic Result Routing → PolicyRouter

**Description**: Policy checks the output results of probabilistic verification (e.g., LLM-as-judge confidence score) and determines the next routing step based on that score.

**Implementation**: PolicyRouter  

**Why**: These policies inspect the **results** of execution (confidence scores, verification outcomes) and dynamically route flow based on those results.

**Research Examples**:

1. **Confidence-Based Escalation**  
   Policy checks the output results of probabilistic verification (e.g., LLM-as-judge confidence score) and determines the next routing step.  
   Example: "If LLM-as-judge returns a low confidence score (e.g., p<0.8), the policy deterministically triggers an INTERRUPT instruction to pause execution for human review."

2. **Fallback Routing**  
   Policy checks if an instruction (e.g., VERIFY) output result is "failure"; if so, Arbiter Loop deterministically routes execution to a predefined FALLBACK plan.  
   Example: After API call failure causes `verify_api_response` instruction to return `result: 'FAIL'`, Arbiter Loop intervenes and routes execution to `FALLBACK (get_cached_sales_data)` instruction.

3. **Strategic Self-Correction**  
   Policy checks EVALUATE_PROGRESS instruction output results; if the result indicates current path is invalid (e.g., returns 'FAIL' signal), deterministically route execution to REPLAN step.  
   Example: `EVALUATE_PROGRESS` check finds current progress is irrelevant to goal (`is_productive: false`), Arbiter Loop intervenes and routes execution to `REPLAN` node to correct strategy.

4. **Domain-Specific Validation Routing**  
   Policy checks if a custom instruction's output passes specific validation checks before allowing subsequent operations.  
   Example: "CALCULATE ALPHA output must pass a custom VERIFY BOUNDS check before it can affect (route to) trading."

**Implementation Example**:
```python
from arbiteros_alpha import PolicyRouter

@dataclass
class MetricThresholdPolicyRouter(PolicyRouter):
    """Routes based on output metrics (e.g., confidence scores)."""
    key: str  # e.g., "confidence"
    threshold: float  # e.g., 0.6
    target: str  # e.g., "generate" (retry node)
    
    def route(self, history: list[History]) -> str | None:
        if not history:
            return None
            
        latest = history[-1]
        metric_value = latest.output_state.get(self.key)
        
        if metric_value is not None and metric_value < self.threshold:
            return self.target  # Route to retry/fallback node
        
        return None  # Continue normal flow
```

---

## Summary Table

| Academic Policy Category | Implementation | Timing | Purpose | Example |
|----------------------|----------------|---------|---------|---------|
| **Static Verification (Linter)** | TODO | Before-started | Validate graph structure | Reject `GENERATE→TOOL_CALL` pattern |
| **Dynamic Transition Verification (Think then Verify)** | `PolicyChecker` | Pre-execution | Enforce preconditions | Require verification before tool calls |
| **Stateful Policies** | `PolicyChecker` | Pre-execution | Check state conditions | Block actions for high-risk users |
| **Temporal Policies** | `PolicyChecker` | Pre-execution | Rate limiting | Limit API calls to 1 per 5 seconds |
| **Resource-Based Policies** | `PolicyChecker` | Pre-execution | Budget management | Block expensive ops when budget low |
| **Domain-Specific Policies** | `PolicyChecker` | Pre-execution | Custom preconditions | Require resource monitoring before backtest |
| **Confidence Escalation** | `PolicyRouter` | Post-execution | Route based on confidence | Retry generation if confidence < 0.8 |
| **Fallback Routing** | `PolicyRouter` | Post-execution | Handle failures | Route to cached data on API failure |
| **Strategic Self-Correction** | `PolicyRouter` | Post-execution | Escape bad paths | Route to REPLAN when unproductive |
| **Validation Routing** | `PolicyRouter` | Post-execution | Conditional continuation | Block trading if bounds check fails |

## Extending the Framework

To implement custom policies:

**For Pre-Execution Constraints**: Inherit from `PolicyChecker`

   - Implement the `check()` method
   - Return `(False, reason)` to block execution
   - Use `history` to inspect past state and transitions

**For Post-Execution Routing**: Inherit from `PolicyRouter`

   - Implement the `route()` method
   - Return a `target` node ID to redirect flow
   - Return `None` to continue normal execution
   - Inspect the latest `history[-1]` for output state and metrics

## Example: Building a Complete Policy

```python
from arbiteros_alpha import ArbiterOSAlpha, PolicyChecker, PolicyRouter

os = ArbiterOSAlpha()

# Pre-execution: Prevent direct GENERATE→TOOL_CALL
os.add_policy_checker(
    HistoryPolicyChecker(
        name="think_then_verify",
        bad_sequence=[GENERATE, TOOL_CALL]
    )
)

# Post-execution: Retry on low confidence
os.add_policy_router(
    MetricThresholdPolicyRouter(
        name="retry_on_low_confidence",
        key="confidence",
        threshold=0.7,
        target="generate"  # Retry the generation step
    )
)
```

This dual-category architecture ensures:

- ✅ **Separation of Concerns**: Validation logic is separate from routing logic
- ✅ **Composability**: Mix and match checkers and routers for complex policies
- ✅ **Auditability**: Each policy has a clear name and documented purpose
- ✅ **Extensibility**: Easy to add custom policies by inheriting from base classes
