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

## Mapping Paper Policies to Implementation

Based on the ArbiterKernel paper's policy taxonomy, all policies map to these two categories:

### Static Verification (Linter) → PolicyChecker

**Chinese Description (静态检查/设计时)**:  
在代理执行之前，静态分析其执行图（Execution Graph），检查所有非条件路径是否违反了声明式的策略。

**Implementation**: PolicyChecker  
**Why**: Static verification happens before any execution. The "linter" inspects the graph structure and rejects forbidden patterns before the agent runs.

**Example from Paper**:  
静态分析器会标记出架构中的违规行为，例如一个 GENERATE 指令（来自Cognitive Core）的输出被直接连接到了一个 TOOL CALL 指令（来自Execution Core），这违反了 "think then verify" 策略。

**Implementation Example**:
```python
from arbiteros_alpha import PolicyChecker

@dataclass
class HistoryPolicyChecker(PolicyChecker):
    """Prevents forbidden instruction sequences like 'generate->toolcall'."""
    bad_sequence: str  # e.g., "generate,toolcall"
    
    def check(self, history: list[History], current_instruction: str) -> tuple[bool, str]:
        # Extract last instruction from history
        recent = [h.instruction for h in history[-len(sequence):]]
        recent.append(current_instruction)
        
        if matches_bad_sequence(recent, self.bad_sequence):
            return False, f"Forbidden sequence: {self.bad_sequence}"
        return True, ""
```

---

### Dynamic Transition Verification → PolicyChecker

**Chinese Description (运行节点前)**:  
在运行时，策略引擎会拦截即将发生的状态转换，强制要求一个概率性步骤（如 GENERATE）必须经过一个确定性验证步骤（如 VERIFY），然后才允许执行一个高风险动作（如 TOOL CALL）。

**Implementation**: PolicyChecker  
**Why**: This enforces preconditions at runtime **before** execution, ensuring verification occurs before high-risk actions.

**Paper Examples**:

1. **"Think then Verify" Workflow**  
   策略 `enforce_verify_before_action`：如果上一个指令来自 Cognitive Core，则禁止下一个指令是 Execution Core，除非两者之间插入了一个 Normative Core 的指令。

2. **Stateful Policies**  
   策略基于代理的"受管状态"（Managed State）中的数据来决定是否允许执行某个操作。  
   Example: "如果 `user_memory.has_flag('high_risk_user')` 标志为真，则禁止 TOOL CALL 调用支付API"。

3. **Temporal Policies**  
   强制执行与时间相关的约束，例如速率限制。  
   Example: "允许 TOOL CALL 调用特定API的频率不得超过每5秒一次"。

4. **Resource-Based Policies**  
   根据剩余的"可靠性预算"（Reliability Budget）来决定是否允许执行高成本操作。  
   Example: "仅当剩余的可靠性预算高于50%时，才允许执行高成本的 DECOMPOSE 步骤"。

5. **Domain-Specific Policy**  
   针对自定义的核心（Cores）或指令，强制执行特定的前置条件。  
   Example: "策略可以强制要求高计算量的 EXECUTE BACKTEST 指令必须由 MONITOR RESOURCES 检查作为前导"。

**Implementation Pattern**:
```python
@dataclass
class StateBasedPolicyChecker(PolicyChecker):
    """Checks state conditions before allowing execution."""
    
    def check(self, history: list[History], current_instruction: str) -> tuple[bool, str]:
        latest_state = history[-1].output_state if history else {}
        
        # Example: Check reliability budget
        budget = latest_state.get("reliability_budget", 100)
        if current_instruction == "decompose" and budget < 50:
            return False, f"Insufficient reliability budget: {budget}%"
        
        # Example: Check rate limiting
        if self._exceeds_rate_limit(history, current_instruction):
            return False, "Rate limit exceeded"
            
        return True, ""
```

---

### Dynamic Result Routing → PolicyRouter

**Chinese Description (运行节点后)**:  
策略检查一个概率性验证（如 LLM-as-judge）的输出结果（置信度分数），并根据该分数决定下一步的路由。

**Implementation**: PolicyRouter  
**Why**: These policies inspect the **results** of execution (confidence scores, verification outcomes) and dynamically route flow based on those results.

**Paper Examples**:

1. **Confidence-Based Escalation**  
   策略检查一个概率性验证（如 LLM-as-judge）的输出结果（置信度分数），并根据该分数决定下一步的路由。  
   Example: "如果 LLM-as-judge 返回的置信度分数很低（例如 p<0.8），策略将确定性地触发一个 INTERRUPT 指令，暂停执行以进行人工审查"。

2. **Fallback Routing**  
   策略检查一个指令（如 VERIFY）的输出结果是否为"失败"，如果是，Arbiter Loop 会确定性地将执行路由到预定义的 FALLBACK 计划。  
   Example: 在API调用失败导致 `verify_api_response` 指令返回 `result: 'FAIL'` 后，Arbiter Loop 介入，并将执行路由到 `FALLBACK (get_cached_sales_data)` 指令。

3. **Strategic Self-Correction**  
   策略检查 EVALUATE_PROGRESS 指令的输出结果，如果该结果指示当前路径无效（例如返回 'FAIL' 信号），则确定性地将执行路由到 REPLAN 步骤。  
   Example: `EVALUATE_PROGRESS` 检查发现当前进展与目标无关 (`is_productive: false`)，Arbiter Loop 介入并将执行路由到 `REPLAN` 节点以纠正策略。

4. **Domain-Specific Validation Routing**  
   策略检查一个自定义指令的输出结果是否通过了特定的验证检查，然后才允许后续操作。  
   Example: "CALCULATE ALPHA 的输出必须通过一个自定义的 VERIFY BOUNDS 检查，然后才能影响（路由到）交易"。

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

| Paper Policy Category | Implementation | Timing | Purpose | Example |
|----------------------|----------------|---------|---------|---------|
| **静态验证 (Linter)** | `PolicyChecker` | Pre-execution | Validate graph structure | Reject `generate→toolcall` pattern |
| **动态转换验证 (Think then Verify)** | `PolicyChecker` | Pre-execution | Enforce preconditions | Require verification before tool calls |
| **有状态策略 (Stateful)** | `PolicyChecker` | Pre-execution | Check state conditions | Block actions for high-risk users |
| **时间策略 (Temporal)** | `PolicyChecker` | Pre-execution | Rate limiting | Limit API calls to 1 per 5 seconds |
| **资源策略 (Resource-Based)** | `PolicyChecker` | Pre-execution | Budget management | Block expensive ops when budget low |
| **领域特定策略 (Domain-Specific)** | `PolicyChecker` | Pre-execution | Custom preconditions | Require resource monitoring before backtest |
| **置信度升级 (Confidence Escalation)** | `PolicyRouter` | Post-execution | Route based on confidence | Retry generation if confidence < 0.8 |
| **故障回退 (Fallback Routing)** | `PolicyRouter` | Post-execution | Handle failures | Route to cached data on API failure |
| **战略自我纠正 (Self-Correction)** | `PolicyRouter` | Post-execution | Escape bad paths | Route to REPLAN when unproductive |
| **领域验证路由 (Validation Routing)** | `PolicyRouter` | Post-execution | Conditional continuation | Block trading if bounds check fails |

## The Contract

### PolicyChecker Contract

```python
@dataclass
class PolicyChecker:
    """Abstract base class for pre-execution validation policies."""
    name: str
    
    def check(self, history: list[History], current_instruction: str) -> tuple[bool, str]:
        """
        Validate if the next instruction should execute.
        
        Args:
            history: Execution history up to this point
            current_instruction: The instruction about to execute
            
        Returns:
            (is_valid, message): (True, "") if allowed, (False, reason) if blocked
        """
        raise NotImplementedError
```

### PolicyRouter Contract

```python
@dataclass
class PolicyRouter:
    """Abstract base class for post-execution routing policies."""
    name: str
    
    def route(self, history: list[History]) -> str | None:
        """
        Determine dynamic routing based on execution results.
        
        Args:
            history: Complete execution history including the just-executed node
            
        Returns:
            target_node_id: Node to route to, or None to continue normal flow
        """
        raise NotImplementedError
```

## Extending the Framework

To implement custom policies:

1. **For Pre-Execution Constraints**: Inherit from `PolicyChecker`
   - Implement the `check()` method
   - Return `(False, reason)` to block execution
   - Use `history` to inspect past state and transitions
   - Access `current_instruction` to see what's about to run

2. **For Post-Execution Routing**: Inherit from `PolicyRouter`
   - Implement the `route()` method
   - Return a `target` node ID to redirect flow
   - Return `None` to continue normal execution
   - Inspect the latest `history[-1]` for output state and metrics

## Example: Building a Complete Policy

```python
from arbiteros_alpha import ArbiterOSAlpha, PolicyChecker, PolicyRouter

os = ArbiterOSAlpha()

# Pre-execution: Prevent direct generate→toolcall
os.add_policy_checker(
    HistoryPolicyChecker(
        name="think_then_verify",
        bad_sequence="generate,toolcall"
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
