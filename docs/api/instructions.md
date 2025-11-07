# Instructions API Reference

This page contains the auto-generated API documentation for the ArbiterOS instruction types.

The instruction types define the fundamental operations that can be governed by ArbiterOS. Each instruction belongs to one of eight core categories, representing different aspects of agent behavior.

## Instruction Type Union

::: arbiteros_alpha.instructions.InstructionType
    options:
      show_root_heading: true
      show_source: true

## CognitiveCore

Governs probabilistic reasoning. Its outputs are always treated as unverified until subjected to explicit checks.

::: arbiteros_alpha.instructions.CognitiveCore
    options:
      show_root_heading: true
      show_source: true
      members: true

## MemoryCore

Manages the LLM's limited context window and connections to persistent memory.

::: arbiteros_alpha.instructions.MemoryCore
    options:
      show_root_heading: true
      show_source: true
      members: true

## ExecutionCore

Interfaces with deterministic external systems. These are high-stakes actions requiring strict controls.

::: arbiteros_alpha.instructions.ExecutionCore
    options:
      show_root_heading: true
      show_source: true
      members: true

## NormativeCore

Enforces human-defined rules, checks, and fallback strategies. This domain anchors ARBITEROS's claim to systematic reliability.

::: arbiteros_alpha.instructions.NormativeCore
    options:
      show_root_heading: true
      show_source: true
      members: true

## MetacognitiveCore

Enables heuristic self-assessment and resource tracking, supporting adaptive routing in the Arbiter Loop.

::: arbiteros_alpha.instructions.MetacognitiveCore
    options:
      show_root_heading: true
      show_source: true
      members: true

## AdaptiveCore

Governing autonomous learning and self-improvement within the ArbiterOS paradigm.

::: arbiteros_alpha.instructions.AdaptiveCore
    options:
      show_root_heading: true
      show_source: true
      members: true

## SocialCore

Enabling governable inter-agent collaboration in multi-agent systems.

::: arbiteros_alpha.instructions.SocialCore
    options:
      show_root_heading: true
      show_source: true
      members: true

## AffectiveCore

Enabling governed socio-emotional reasoning for human-agent teaming.

::: arbiteros_alpha.instructions.AffectiveCore
    options:
      show_root_heading: true
      show_source: true
      members: true

## Short Aliases

For convenience, all instruction types are available as module-level constants:

```python
import arbiteros_alpha.instructions as Instr

# Use short aliases instead of full paths
@os.instruction(Instr.GENERATE)  # instead of Instr.CognitiveCore.GENERATE
def generate(state): ...

@os.instruction(Instr.TOOL_CALL)  # instead of Instr.ExecutionCore.TOOL_CALL
def tool_call(state): ...
```
