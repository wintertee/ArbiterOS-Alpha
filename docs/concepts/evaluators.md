# Evaluators: Quality Assessment and Feedback

## Overview

**Evaluators** are non-blocking quality assessment mechanisms that provide feedback on node execution quality. Unlike PolicyCheckers (which block execution) and PolicyRouters (which control flow), evaluators run after node execution to provide scores, feedback, and quality metrics without interrupting the workflow.

Evaluators are inspired by **Reinforcement Learning (RL)** reward functions, where each node execution receives a quality score that can be used for:

- **Monitoring**: Track agent performance over time
- **Self-improvement**: Provide feedback for reflection and refinement
- **Training**: Generate reward signals for RL-based optimization
- **Debugging**: Identify low-quality outputs for analysis

## Key Characteristics

### Non-Blocking Execution

Evaluators **never** interrupt or block execution, even when they detect low-quality outputs:

```python
# Even if evaluator fails, execution continues
result = node_function(state)
evaluations = run_evaluators(history)  # Never raises exceptions
# Execution proceeds regardless of evaluation results
```

### Post-Execution Timing

Evaluators run **after** a node completes:

```
1. Pre-execution: PolicyChecker validates preconditions
2. Execution: Node function runs
3. Post-execution: Evaluators assess quality ← HERE
4. Routing: PolicyRouter determines next step
```

### Instruction Type Filtering

Evaluators can target specific instruction types using `target_instructions`:

```python
# Only evaluate GENERATE nodes
evaluator = ThresholdEvaluator(
    name="confidence_check",
    key="confidence",
    threshold=0.7,
    target_instructions=[CognitiveCore.GENERATE]
)

# Evaluate multiple instruction types
evaluator = QualityEvaluator(
    name="quality",
    target_instructions=[CognitiveCore.GENERATE, CognitiveCore.REFLECT]
)

# Evaluate ALL nodes (default)
evaluator = UniversalEvaluator(name="universal")  # target_instructions=None
```

## Comparison with Policy Components

| Component | Timing | Blocks Execution? | Purpose | Output |
|-----------|--------|-------------------|---------|--------|
| **PolicyChecker** | Pre-execution | ✓ Yes | Enforce constraints | Pass/Fail + violation message |
| **PolicyRouter** | Post-execution | ✗ No (but controls flow) | Dynamic routing | Target node ID or None |
| **Evaluator** | Post-execution | ✗ No | Quality assessment | Score + feedback |

## Built-in Evaluators

### ThresholdEvaluator

Checks if a numeric value in the output state meets a threshold:

```python
from arbiteros_alpha import ThresholdEvaluator
from arbiteros_alpha.instructions import CognitiveCore

evaluator = ThresholdEvaluator(
    name="confidence_check",
    key="confidence",           # Key in output_state to check
    threshold=0.7,              # Minimum acceptable value
    target_instructions=[CognitiveCore.GENERATE]
)

arbiter_os.add_evaluator(evaluator)
```

**Behavior**:
- If `output_state["confidence"] >= 0.7`: `passed=True`, `score=confidence`
- If `output_state["confidence"] < 0.7`: `passed=False`, `score=confidence`
- If key missing: `score=0.0`, `passed=False`

## Creating Custom Evaluators

Extend `NodeEvaluator` and implement the `evaluate()` method:

```python
from arbiteros_alpha.evaluation import NodeEvaluator, EvaluationResult
from arbiteros_alpha.instructions import CognitiveCore

class ResponseLengthEvaluator(NodeEvaluator):
    """Evaluates response quality based on length."""
    
    def __init__(self, min_length: int = 50):
        super().__init__(
            name="response_length",
            target_instructions=[CognitiveCore.GENERATE]  # Only GENERATE
        )
        self.min_length = min_length
    
    def evaluate(self, history) -> EvaluationResult:
        # Access the most recent node execution
        current_item = history.entries[-1][-1]
        
        # Extract output
        response = current_item.output_state.get("response", "")
        length = len(response)
        
        # Calculate score (0.0 to 1.0)
        score = min(length / 100, 1.0)
        
        # Determine pass/fail
        passed = length >= self.min_length
        
        # Provide feedback
        feedback = f"Response length: {length} chars"
        
        return EvaluationResult(
            score=score,
            passed=passed,
            feedback=feedback
        )
```

### Advanced: History-Aware Evaluators

Evaluators have access to the full execution history:

```python
class ConsistencyEvaluator(NodeEvaluator):
    """Checks consistency across multiple responses."""
    
    def __init__(self):
        super().__init__(
            name="consistency_check",
            target_instructions=[CognitiveCore.GENERATE]
        )
    
    def evaluate(self, history) -> EvaluationResult:
        # Access all previous executions
        all_entries = [
            item for superstep in history.entries
            for item in superstep
        ]
        
        # Find all previous GENERATE nodes
        previous_generates = [
            item for item in all_entries[:-1]  # Exclude current
            if item.instruction == CognitiveCore.GENERATE
        ]
        
        if not previous_generates:
            return EvaluationResult(
                score=1.0,
                passed=True,
                feedback="First response, nothing to compare"
            )
        
        # Compare with previous responses
        current = all_entries[-1].output_state
        previous = previous_generates[-1].output_state
        
        # Check tone consistency
        current_tone = current.get("tone", "")
        previous_tone = previous.get("tone", "")
        
        consistent = current_tone == previous_tone
        score = 1.0 if consistent else 0.5
        
        return EvaluationResult(
            score=score,
            passed=consistent,
            feedback=f"Tone consistency: {'consistent' if consistent else 'inconsistent'}"
        )
```

## Integration with History

Evaluation results are automatically stored in `HistoryItem.evaluation_results`:

```python
# Execute node with evaluators
arbiter_os.add_evaluator(confidence_evaluator)
arbiter_os.add_evaluator(quality_evaluator)

state = {"query": "test"}
result = arbiter_os.execute(state, generate_fn)

# Access evaluation results from history
last_item = arbiter_os.history.entries[-1][-1]
evaluations = last_item.evaluation_results

for evaluator_name, eval_result in evaluations.items():
    print(f"{evaluator_name}: score={eval_result.score:.2f}, "
          f"passed={eval_result.passed}")
```

### Pretty-Printed History

Use `History.pprint()` to display evaluations:

```python
arbiter_os.history.pprint()
```

**Output**:
```
╔═══ SuperStep 1 ═══╗
  [1.1] GENERATE
    Evaluations:
      ✓ confidence_check: score=0.90 - confidence=0.90 (✓ threshold=0.7)
      ✓ response_quality: score=0.66 - Quality assessment: length=82 chars
      ✗ consistency_check: score=0.50 - Tone inconsistent with previous
```

## Use Cases

### 1. Monitoring Agent Performance

Track quality metrics across conversations:

```python
class PerformanceTracker(NodeEvaluator):
    def __init__(self):
        super().__init__(name="performance")
        self.scores = []
    
    def evaluate(self, history):
        score = calculate_quality(history.entries[-1][-1])
        self.scores.append(score)
        
        avg_score = sum(self.scores) / len(self.scores)
        
        return EvaluationResult(
            score=score,
            passed=score > 0.6,
            feedback=f"Current: {score:.2f}, Average: {avg_score:.2f}"
        )
```

### 2. Self-Reflection Triggers

Combine evaluators with routers for automatic reflection:

```python
# Evaluator assesses quality
quality_evaluator = ThresholdEvaluator(
    name="quality_check",
    key="quality_score",
    threshold=0.7,
    target_instructions=[CognitiveCore.GENERATE]
)

# Router triggers reflection on low quality
class ReflectionRouter(PolicyRouter):
    def route_after(self, history, current_output):
        last_item = history.entries[-1][-1]
        quality_eval = last_item.evaluation_results.get("quality_check")
        
        if quality_eval and not quality_eval.passed:
            return "reflect_node"  # Trigger reflection
        return None  # Continue normal flow

arbiter_os.add_evaluator(quality_evaluator)
arbiter_os.add_policy_router(ReflectionRouter(name="auto_reflect"))
```

### 3. RL Training Signal Generation

Generate reward signals for reinforcement learning:

```python
class RewardEvaluator(NodeEvaluator):
    """Generates RL rewards for training."""
    
    def __init__(self):
        super().__init__(name="rl_reward")
        self.episode_rewards = []
    
    def evaluate(self, history) -> EvaluationResult:
        current = history.entries[-1][-1]
        
        # Multi-factor reward calculation
        factors = {
            "correctness": self._check_correctness(current),
            "efficiency": self._check_efficiency(current),
            "safety": self._check_safety(current)
        }
        
        # Weighted reward
        reward = (
            0.5 * factors["correctness"] +
            0.3 * factors["efficiency"] +
            0.2 * factors["safety"]
        )
        
        self.episode_rewards.append(reward)
        
        return EvaluationResult(
            score=reward,
            passed=reward > 0.5,
            feedback=f"Reward: {reward:.2f} | Factors: {factors}"
        )
    
    def get_episode_reward(self) -> float:
        """Get cumulative reward for training."""
        return sum(self.episode_rewards)
```

## Best Practices

### 1. Fail Gracefully

Always handle missing keys and exceptions:

```python
def evaluate(self, history) -> EvaluationResult:
    try:
        current = history.entries[-1][-1]
        value = current.output_state.get("key", 0.0)  # Default value
        
        # ... evaluation logic ...
        
    except Exception as e:
        # Never crash - return neutral evaluation
        return EvaluationResult(
            score=0.0,
            passed=False,
            feedback=f"Evaluation error: {str(e)}"
        )
```

### 2. Use Instruction Filtering

Only evaluate nodes that produce relevant outputs:

```python
# ✗ Bad: Evaluates all nodes, including VERIFY (may not have "response")
class ResponseEvaluator(NodeEvaluator):
    def __init__(self):
        super().__init__(name="response_eval")  # No filtering!

# ✓ Good: Only evaluates GENERATE nodes
class ResponseEvaluator(NodeEvaluator):
    def __init__(self):
        super().__init__(
            name="response_eval",
            target_instructions=[CognitiveCore.GENERATE]  # Filtered!
        )
```

### 3. Provide Actionable Feedback

Make feedback useful for debugging and improvement:

```python
# ✗ Bad: Vague feedback
feedback = "Low quality"

# ✓ Good: Specific, actionable feedback
feedback = (
    f"Quality issues: length={len(response)} chars (min=50), "
    f"confidence={conf:.2f} (threshold=0.7), "
    f"tone={tone} (expected='formal')"
)
```

### 4. Normalize Scores

Keep scores in the 0.0-1.0 range for consistency:

```python
def evaluate(self, history) -> EvaluationResult:
    raw_value = current.output_state.get("metric", 0)
    
    # Normalize to 0.0-1.0
    score = max(0.0, min(1.0, raw_value / 100))
    
    return EvaluationResult(score=score, ...)
```

## API Reference

For detailed API documentation, see:

- [Core API - ArbiterOSAlpha.add_evaluator()](../api/core.md)
- [Evaluation API](../api/evaluation.md)
- [Complete Example](../examples/evaluator-tutorial.md)
