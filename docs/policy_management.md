# Policy Management Architecture

This document explains how policies are managed in ArbiterOS, including the separation between policy classes and rule instances.

## Overview

ArbiterOS separates policy management into two layers:

1. **Policy Classes** (in code): Reusable policy implementations like `HistoryPolicyChecker`, `MetricThresholdPolicyRouter`, and `GraphStructurePolicyChecker`
2. **Policy Rules** (in YAML): Specific rule instances that configure policy classes with parameters

This separation allows:
- **Kernel policies**: Read-only policies defined by the kernel in `kernel_policy_list.yaml`
- **Custom policies**: Developer-defined policies in `custom_policy_list.yaml` that extend kernel policies

## Policy Files

### Kernel Policy File
- **Location**: `arbiteros_alpha/kernel_policy_list.yaml`
- **Purpose**: Core governance rules defined by the kernel
- **Modification**: Read-only, cannot be modified by developers
- **Loaded first**: Kernel policies are loaded before custom policies

### Custom Policy File
- **Location**: `examples/custom_policy_list.yaml` (or custom path)
- **Purpose**: Developer-defined policies that extend kernel policies
- **Modification**: Can be freely modified by developers
- **Loaded second**: Custom policies are loaded after kernel policies

## YAML Schema

### Policy Checkers

Policy checkers validate execution constraints before instruction execution:

```yaml
policy_checkers:
  - type: HistoryPolicyChecker
    name: no_skip_to_execute
    bad_sequence: [GENERATE, TOOL_CALL]
```

**Fields:**
- `type`: Policy class name (must be registered in `PolicyLoader.POLICY_CLASS_REGISTRY`)
- `name`: Human-readable name for the policy
- `bad_sequence`: List of instruction names (e.g., `GENERATE`, `TOOL_CALL`)

### Policy Routers

Policy routers dynamically route execution flow based on conditions:

```yaml
policy_routers:
  - type: MetricThresholdPolicyRouter
    name: revisit_reason_when_low_confidence
    key: confidence
    threshold: 0.7
    target: reason
```

**Fields:**
- `type`: Policy class name
- `name`: Human-readable name for the policy
- `key`: State key to monitor (e.g., `confidence`)
- `threshold`: Minimum acceptable value
- `target`: Node name to route to when threshold is not met

### Graph Structure Checkers

Graph structure checkers validate graph structure before execution:

```yaml
graph_structure_checkers:
  - type: GraphStructurePolicyChecker
    blacklists:
      - name: no_direct_execute_without_reason
        sequence: [TOOL_CALL, TOOL_CALL]
        level: error
      - name: no_skip_to_execute
        sequence: [GENERATE, TOOL_CALL]
        level: error
      - name: multiple_generate_in_a_row
        sequence: [GENERATE, GENERATE]
        level: warning
```

**Fields:**
- `type`: Must be `GraphStructurePolicyChecker`
- `blacklists`: List of blacklist rules
  - `name`: Human-readable name for the rule
  - `sequence`: List of instruction names or node patterns
  - `level`: Severity level (`error` or `warning`)

## Usage

### Basic Usage

Load policies from default locations:

```python
from arbiteros_alpha import ArbiterOSAlpha

os = ArbiterOSAlpha()
os.load_policies()  # Loads from default kernel and custom policy files
```

### Custom Paths

Specify custom policy file paths:

```python
os.load_policies(
    kernel_policy_path="path/to/kernel_policies.yaml",
    custom_policy_path="path/to/custom_policies.yaml"
)
```

### Programmatic Policy Definition

You can still define policies programmatically if needed:

```python
from arbiteros_alpha.policy import HistoryPolicyChecker, MetricThresholdPolicyRouter

# Define policies in code
checker = HistoryPolicyChecker(
    name="custom_checker",
    bad_sequence=[Instr.GENERATE, Instr.TOOL_CALL]
)
os.add_policy_checker(checker)
```

## Architecture: Policy Classes vs Rules

### Policy Classes (Code)

Policy classes are reusable implementations defined in `arbiteros_alpha/policy.py`:

- `HistoryPolicyChecker`: Validates execution history against blacklisted sequences
- `MetricThresholdPolicyRouter`: Routes based on metric thresholds
- `GraphStructurePolicyChecker`: Validates graph structure against blacklists

These classes are **immutable** and defined by the kernel. Developers cannot modify them.

### Policy Rules (YAML)

Policy rules are specific instances that configure policy classes:

- **Kernel rules**: Defined in `kernel_policy_list.yaml` (read-only)
- **Custom rules**: Defined in `custom_policy_list.yaml` (developer-modifiable)

Rules are **instances** of policy classes with specific parameters. Developers can add new rules but cannot override kernel rules.

## Policy Loading Process

1. **Load kernel policies** from `kernel_policy_list.yaml`
   - Parse YAML configuration
   - Instantiate policy classes with rule parameters
   - Register with `ArbiterOSAlpha`

2. **Load custom policies** from `custom_policy_list.yaml`
   - Parse YAML configuration
   - Instantiate policy classes with rule parameters
   - Register with `ArbiterOSAlpha` (extends kernel policies)

3. **Apply all policies** during graph execution
   - Policy checkers run before instruction execution
   - Policy routers run after instruction execution
   - Graph structure checkers run during graph validation

## Extending Policies

### Adding New Policy Classes

To add a new policy class:

1. Define the class in `arbiteros_alpha/policy.py`:
   ```python
   @dataclass
   class CustomPolicyChecker(PolicyChecker):
       name: str
       custom_param: str
       
       def check_before(self, history: list[History]) -> bool:
           # Implementation
           pass
   ```

2. Register it in `PolicyLoader.POLICY_CLASS_REGISTRY`:
   ```python
   POLICY_CLASS_REGISTRY = {
       "HistoryPolicyChecker": HistoryPolicyChecker,
       "CustomPolicyChecker": CustomPolicyChecker,  # Add here
   }
   ```

3. Add instantiation logic in `PolicyLoader._instantiate_checker()` or similar method

### Adding New Rules

To add a new rule, simply add it to `custom_policy_list.yaml`:

```yaml
policy_checkers:
  - type: HistoryPolicyChecker
    name: my_custom_rule
    bad_sequence: [GENERATE, VERIFY, TOOL_CALL]
```

No code changes needed!

## Best Practices

1. **Kernel policies**: Keep minimal and focused on core safety rules
2. **Custom policies**: Add application-specific rules in custom policy files
3. **Naming**: Use descriptive names for policies (e.g., `no_skip_to_execute`)
4. **Levels**: Use `error` for critical violations, `warning` for non-critical issues
5. **Documentation**: Comment your YAML files to explain policy purposes

## Example: Complete Policy File

```yaml
# Custom Policy List
# Developer-defined policies that extend kernel policies

policy_checkers:
  - type: HistoryPolicyChecker
    name: no_skip_to_execute
    bad_sequence: [GENERATE, TOOL_CALL]

policy_routers:
  - type: MetricThresholdPolicyRouter
    name: revisit_reason_when_low_confidence
    key: confidence
    threshold: 0.7
    target: reason

graph_structure_checkers:
  - type: GraphStructurePolicyChecker
    blacklists:
      - name: no_direct_execute_without_reason
        sequence: [TOOL_CALL, TOOL_CALL]
        level: error
      - name: no_skip_to_execute
        sequence: [GENERATE, TOOL_CALL]
        level: error
      - name: multiple_generate_in_a_row
        sequence: [GENERATE, GENERATE]
        level: warning
```

