# Evaluator Tutorial

This tutorial demonstrates how to use evaluators to assess node execution quality in ArbiterOS.

## Overview

Evaluators provide non-blocking quality assessment for agent nodes. They run after node execution and generate scores, feedback, and pass/fail indicators without interrupting the workflow.

## Basic Example

Let's create a simple agent with evaluators:

```python
from arbiteros_alpha import ArbiterOSAlpha, ThresholdEvaluator
from arbiteros_alpha.instructions import CognitiveCore, NormativeCore

# Initialize ArbiterOS
arbiter_os = ArbiterOSAlpha(backend="native")

# Create a GENERATE node
@arbiter_os.instruction(CognitiveCore.GENERATE)
def generate(state):
    query = state.get("query", "")
    return {
        "response": f"Generated response for: {query}",
        "confidence": 0.85
    }

# Add a confidence threshold evaluator
confidence_evaluator = ThresholdEvaluator(
    name="confidence_check",
    key="confidence",
    threshold=0.7,
    target_instructions=[CognitiveCore.GENERATE]  # Only evaluate GENERATE
)
arbiter_os.add_evaluator(confidence_evaluator)

# Execute
state = {"query": "What is AI?"}
state = generate(state)

# View evaluation results
arbiter_os.history.pprint()
```

**Output**:
```
╔═══ SuperStep 1 ═══╗
  [1.1] GENERATE
    Evaluations:
      ✓ confidence_check: score=0.85 - confidence=0.85 (✓ threshold=0.7)
```

## Custom Evaluator: Response Quality

Create a custom evaluator to assess response quality:

```python
from arbiteros_alpha.evaluation import NodeEvaluator, EvaluationResult
from arbiteros_alpha.instructions import CognitiveCore

class ResponseQualityEvaluator(NodeEvaluator):
    """Evaluates response quality based on length and content."""
    
    def __init__(self):
        super().__init__(
            name="response_quality",
            target_instructions=[CognitiveCore.GENERATE]  # Only GENERATE
        )
    
    def evaluate(self, history) -> EvaluationResult:
        # Get current node output
        current_item = history.entries[-1][-1]
        response = current_item.output_state.get("response", "")
        
        # Calculate quality score
        length = len(response)
        has_greeting = any(word in response.lower() 
                          for word in ["hello", "hi", "hey"])
        has_content = length > 50
        
        # Length score (0-1 based on 100 chars target)
        length_score = min(length / 100, 1.0)
        
        # Quality score
        score = length_score
        if has_greeting:
            score *= 0.8  # Penalize greetings
        if not has_content:
            score *= 0.5  # Penalize short responses
        
        passed = score > 0.5
        
        feedback = (
            f"Quality assessment: length={length} chars (score: {length_score:.2f}), "
            f"greeting={'yes' if has_greeting else 'no'}, "
            f"content={'sufficient' if has_content else 'insufficient'}"
        )
        
        return EvaluationResult(
            score=score,
            passed=passed,
            feedback=feedback
        )

# Add to ArbiterOS
arbiter_os.add_evaluator(ResponseQualityEvaluator())
```

## History-Aware Evaluator: Consistency

Check consistency across multiple executions:

```python
class ConsistencyEvaluator(NodeEvaluator):
    """Evaluates consistency with previous responses."""
    
    def __init__(self):
        super().__init__(
            name="consistency_check",
            target_instructions=[CognitiveCore.GENERATE]
        )
    
    def evaluate(self, history) -> EvaluationResult:
        # Get all GENERATE entries
        all_entries = [
            item for superstep in history.entries
            for item in superstep
        ]
        
        generate_entries = [
            item for item in all_entries
            if item.instruction == CognitiveCore.GENERATE
        ]
        
        # First response? Nothing to compare
        if len(generate_entries) <= 1:
            return EvaluationResult(
                score=1.0,
                passed=True,
                feedback="First response, nothing to compare"
            )
        
        # Compare tone with previous response
        current = generate_entries[-1].output_state
        previous = generate_entries[-2].output_state
        
        current_tone = current.get("tone", "neutral")
        previous_tone = previous.get("tone", "neutral")
        
        consistent = current_tone == previous_tone
        score = 1.0 if consistent else 0.5
        
        feedback = f"Tone consistency: {'consistent' if consistent else 'inconsistent'} with previous responses"
        
        return EvaluationResult(
            score=score,
            passed=consistent,
            feedback=feedback
        )

# Add to ArbiterOS
arbiter_os.add_evaluator(ConsistencyEvaluator())
```

## Multi-Node Example with Filtering

Demonstrate evaluator filtering with GENERATE and VERIFY nodes:

```python
from arbiteros_alpha import ArbiterOSAlpha, ThresholdEvaluator
from arbiteros_alpha.instructions import CognitiveCore, NormativeCore

arbiter_os = ArbiterOSAlpha(backend="native")

# GENERATE node: produces response and confidence
@arbiter_os.instruction(CognitiveCore.GENERATE)
def generate(state):
    return {
        "response": "AI is artificial intelligence.",
        "confidence": 0.85,
        "tone": "formal"
    }

# VERIFY node: validates the response (returns empty dict)
@arbiter_os.instruction(NormativeCore.VERIFY)
def verify(state):
    # Verification logic...
    return {}  # No output state

# Add evaluators that only target GENERATE nodes
arbiter_os.add_evaluator(ThresholdEvaluator(
    name="confidence_check",
    key="confidence",
    threshold=0.7,
    target_instructions=[CognitiveCore.GENERATE]  # Skip VERIFY
))

arbiter_os.add_evaluator(ResponseQualityEvaluator())  # Also skips VERIFY
arbiter_os.add_evaluator(ConsistencyEvaluator())      # Also skips VERIFY

# Run scenario
state = {"query": "What is AI?"}
state = generate(state)
state = verify(state)

# View history
arbiter_os.history.pprint()
```

**Output**:
```
╔═══ SuperStep 1 ═══╗
  [1.1] GENERATE
    Evaluations:
      ✓ confidence_check: score=0.85 - confidence=0.85 (✓ threshold=0.7)
      ✓ response_quality: score=0.66 - Quality assessment: length=33 chars...
      ✓ consistency_check: score=1.00 - First response, nothing to compare

╔═══ SuperStep 2 ═══╗
  [2.1] VERIFY
    Evaluations:
      (no evaluations - VERIFY nodes skipped by evaluators)
```

## Combining Evaluators with Routers

Use evaluator results to trigger dynamic routing:

```python
from arbiteros_alpha.policy import PolicyRouter

class QualityBasedRouter(PolicyRouter):
    """Routes to reflection if quality is low."""
    
    def route_after(self, history, current_output):
        # Get the most recent evaluation results
        last_item = history.entries[-1][-1]
        evaluations = last_item.evaluation_results
        
        # Check if quality evaluator failed
        quality_eval = evaluations.get("response_quality")
        if quality_eval and not quality_eval.passed:
            return "reflect_node"  # Trigger reflection
        
        return None  # Continue normal flow

# Add router
arbiter_os.add_policy_router(QualityBasedRouter(name="quality_router"))

# Now low-quality responses automatically trigger reflection
```

## Complete Scenario: Multiple Queries

```python
# Setup (same as above with all evaluators)

scenarios = [
    {"query": "Tell me about AI in a formal way", "tone": "formal"},
    {"query": "Explain machine learning casually", "tone": "casual"},
    {"query": "What is deep learning?", "tone": "formal"}
]

for i, scenario in enumerate(scenarios, 1):
    print(f"\n{'='*60}")
    print(f"Scenario {i}: {scenario['query']}")
    print('='*60)
    
    state = scenario.copy()
    state = generate(state)
    state = verify(state)

# View complete history with all evaluations
print("\n" + "="*60)
print("COMPLETE HISTORY")
print("="*60)
arbiter_os.history.pprint()
```

## Accessing Evaluation Results Programmatically

```python
# Get evaluation results from history
for superstep_idx, superstep in enumerate(arbiter_os.history.entries, 1):
    for item_idx, item in enumerate(superstep, 1):
        print(f"SuperStep {superstep_idx}.{item_idx} - {item.instruction.name}")
        
        # Iterate through evaluations
        for eval_name, eval_result in item.evaluation_results.items():
            status = "✓" if eval_result.passed else "✗"
            print(f"  {status} {eval_name}: score={eval_result.score:.2f}")
            print(f"     {eval_result.feedback}")
```

## Best Practices

### 1. Always Use Instruction Filtering

Evaluators should target specific node types:

```python
# ✓ Good: Filtered to appropriate node types
evaluator = ThresholdEvaluator(
    name="confidence",
    key="confidence",
    threshold=0.7,
    target_instructions=[CognitiveCore.GENERATE]  # Only GENERATE
)

# ✗ Bad: No filtering - may evaluate inappropriate nodes
evaluator = ThresholdEvaluator(
    name="confidence",
    key="confidence",
    threshold=0.7
)
```

### 2. Handle Missing Keys Gracefully

```python
def evaluate(self, history) -> EvaluationResult:
    current = history.entries[-1][-1]
    
    # ✓ Good: Use .get() with defaults
    value = current.output_state.get("key", 0.0)
    
    # ✗ Bad: Direct access - may raise KeyError
    value = current.output_state["key"]
```

### 3. Provide Descriptive Feedback

```python
# ✓ Good: Detailed, actionable feedback
feedback = f"Response length: {length} chars (min={self.min_length}), " \
          f"confidence: {conf:.2f} (threshold={self.threshold})"

# ✗ Bad: Vague feedback
feedback = "Failed"
```

### 4. Use Normalized Scores

```python
# ✓ Good: Normalized to 0.0-1.0
score = min(length / 100, 1.0)

# ✗ Bad: Unbounded score
score = length  # Could be 0-10000
```

## Running the Full Example

The complete evaluator example is available in `examples/evaluator.py`:

```bash
uv run python -m examples.evaluator
```

This demonstrates:
- ✓ Three evaluators (confidence, quality, consistency)
- ✓ Instruction type filtering
- ✓ Multiple scenarios with different tones
- ✓ History-aware consistency checking
- ✓ Pretty-printed evaluation results

## Next Steps

- Learn about [Evaluator Concepts](../concepts/evaluators.md)
- See [Evaluation API Reference](../api/evaluation.md)
- Explore [Policy Integration](../concepts/policies.md)
