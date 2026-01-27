"""LLM-powered policy designer for migration pipeline.

This module provides the PolicyDesigner class that uses LLM with structured
output to design domain-specific policy checkers and routers based on
repository analysis and node classifications.
"""

import logging
import os
from dataclasses import dataclass

from pydantic import BaseModel

from .schemas import (
    NodeClassificationBatch,
    PolicyDesignOutput,
    RepoAnalysisOutput,
)

logger = logging.getLogger(__name__)


@dataclass
class PolicyDesignerConfig:
    """Configuration for the policy designer.

    Attributes:
        api_key: OpenAI API key. Defaults to OPENAI_API_KEY env var.
        base_url: API base URL. Defaults to OPENAI_BASE_URL env var.
        model: Model name to use. Defaults to "gpt-4o".
        temperature: Sampling temperature. Lower = more deterministic.
    """

    api_key: str | None = None
    base_url: str | None = None
    model: str = "gpt-4o"
    temperature: float = 0.2  # Slightly higher for creative policy design

    def __post_init__(self) -> None:
        """Load defaults from environment variables."""
        if self.api_key is None:
            self.api_key = os.environ.get("OPENAI_API_KEY")
        if self.base_url is None:
            self.base_url = os.environ.get("OPENAI_BASE_URL")


class PolicyDesigner:
    """LLM-powered policy designer.

    Designs domain-specific policy checkers and routers based on:
    - Repository analysis (domain, workflow stages, constraints)
    - Node classifications (instruction types, roles)
    - Best practices for governance

    Example:
        >>> designer = PolicyDesigner()
        >>> policies = designer.design(analysis, classifications)
        >>> for checker in policies.checkers:
        ...     print(f"Checker: {checker.class_name}")
        >>> for router in policies.routers:
        ...     print(f"Router: {router.class_name}")
    """

    def __init__(self, config: PolicyDesignerConfig | None = None) -> None:
        """Initialize the PolicyDesigner.

        Args:
            config: Optional configuration. Uses defaults if not provided.
        """
        self.config = config or PolicyDesignerConfig()
        self._llm = None

    def _get_llm(self):
        """Lazy-load the LLM client."""
        if self._llm is None:
            try:
                from langchain_openai import ChatOpenAI
            except ImportError:
                raise ImportError(
                    "langchain-openai is required for policy design. "
                    "Install it with: pip install langchain-openai"
                )

            kwargs = {
                "model": self.config.model,
                "temperature": self.config.temperature,
            }
            if self.config.api_key:
                kwargs["api_key"] = self.config.api_key
            if self.config.base_url:
                kwargs["base_url"] = self.config.base_url

            self._llm = ChatOpenAI(**kwargs)

        return self._llm

    def _invoke_structured(
        self, prompt: str, output_schema: type[BaseModel]
    ) -> BaseModel:
        """Invoke LLM with structured JSON output.

        Args:
            prompt: The prompt to send to the LLM.
            output_schema: Pydantic model for structured output.

        Returns:
            Parsed and validated Pydantic model instance.
        """
        llm = self._get_llm()
        # Use function_calling method instead of strict JSON schema
        # to support more flexible schemas (e.g., dict[str, Any] parameters)
        structured_llm = llm.with_structured_output(
            output_schema, method="function_calling"
        )
        result = structured_llm.invoke(prompt)
        return result

    def design(
        self,
        analysis: RepoAnalysisOutput,
        classifications: NodeClassificationBatch,
    ) -> PolicyDesignOutput:
        """Design policies based on analysis and classifications.

        Args:
            analysis: Repository analysis output.
            classifications: Node classification results.

        Returns:
            PolicyDesignOutput with checker and router specifications.
        """
        logger.info(f"Designing policies for domain: {analysis.domain}")

        prompt = self._build_design_prompt(analysis, classifications)
        result = self._invoke_structured(prompt, PolicyDesignOutput)

        logger.info(
            f"Designed {len(result.checkers)} checkers and {len(result.routers)} routers"
        )

        return result

    def _build_design_prompt(
        self,
        analysis: RepoAnalysisOutput,
        classifications: NodeClassificationBatch,
    ) -> str:
        """Build the policy design prompt.

        Args:
            analysis: Repository analysis output.
            classifications: Node classification results.

        Returns:
            Formatted prompt string.
        """
        # Format domain context
        domain_section = f"""## Domain Context

**Domain:** {analysis.domain}
**Description:** {analysis.domain_description}
**Framework:** {analysis.framework.value}

### Agent Roles
"""
        for role in analysis.agent_roles:
            domain_section += f"- **{role.role_name}**: {role.description}\n"
            domain_section += f"  - Instruction: {role.suggested_instruction.value}\n"

        # Format workflow stages
        workflow_section = "\n### Workflow Stages\n"
        for stage in analysis.workflow_stages:
            workflow_section += f"\n**{stage.stage_name}**\n"
            workflow_section += f"- Description: {stage.description}\n"
            workflow_section += f"- Roles: {', '.join(stage.agent_roles)}\n"
            if stage.preconditions:
                workflow_section += (
                    f"- Preconditions: {', '.join(stage.preconditions)}\n"
                )
            if stage.postconditions:
                workflow_section += (
                    f"- Postconditions: {', '.join(stage.postconditions)}\n"
                )

        # Format key constraints
        constraints_section = "\n### Key Constraints\n"
        for constraint in analysis.key_constraints:
            constraints_section += f"- {constraint}\n"

        # Format classifications with risk assessment
        classifications_section = "\n### Node Classifications\n"
        high_risk_nodes = []
        for c in classifications.classifications:
            risk_level = self._assess_node_risk(c)
            risk_marker = " [HIGH-RISK]" if risk_level == "high" else ""
            classifications_section += f"- {c.function_name}: {c.instruction_type.value} ({c.core.value}){risk_marker}\n"
            if risk_level == "high":
                high_risk_nodes.append(c.function_name)

        # Add high-risk node section
        if high_risk_nodes:
            classifications_section += f"\n**HIGH-RISK NODES REQUIRING VERIFICATION:** {', '.join(high_risk_nodes)}\n"

        prompt = f"""You are an expert at designing SAFETY-CRITICAL governance policies for LLM agent systems. Your task is to design PolicyCheckers and PolicyRouters that ACTUALLY ENFORCE safety constraints for the ArbiterOS governance framework.

**CRITICAL SAFETY REQUIREMENTS:**
- PolicyCheckers must BLOCK unsafe operations, not just log warnings
- High-risk actions (trading, external API calls, data modification) MUST require prior verification
- The system must fail-safe: when in doubt, BLOCK the operation
- All policies must have meaningful parameters with sensible thresholds

{domain_section}
{workflow_section}
{constraints_section}
{classifications_section}

## ArbiterOS Policy Framework

### PolicyChecker - ENFORCEMENT (not just monitoring)
PolicyCheckers validate execution constraints BEFORE instruction execution.
They MUST return False to BLOCK execution when constraints are violated.

```python
@dataclass
class PolicyChecker(ABC):
    name: str
    strict_mode: bool = True  # Raise error on failure
    # Custom validation parameters...
    
    def check_before(self, history: History) -> bool:
        # Return True ONLY if validation passes
        # Return False to BLOCK the instruction
        # In strict_mode, raise PolicyViolationError on failure
        pass
```

### PolicyRouter - Dynamic Safety Routing
PolicyRouters dynamically route execution AFTER instruction execution.
Use for quality gates, confidence checks, and risk-based escalation.

```python
@dataclass
class PolicyRouter(ABC):
    name: str
    target: str  # Node to route to
    threshold: float  # Threshold for triggering
    key: str  # State key to check
    
    def route_after(self, history: History) -> str | None:
        # Check output_state for the key
        # Return target node if threshold condition met
        # Return None for normal flow
        pass
```

### History Structure
- `history.entries`: List of supersteps (each is a list of HistoryItems)
- `HistoryItem.instruction`: Instruction type (e.g., CognitiveCore.GENERATE)
- `HistoryItem.input_state`: Input state dict
- `HistoryItem.output_state`: Output state dict (contains confidence, risk_level, etc.)

## MANDATORY SAFETY PATTERNS

### 1. Verification Before High-Risk Actions
ANY high-risk node (TOOL_CALL, external execution) MUST have a VerificationRequirementChecker:
- Checks that a VERIFY instruction has been executed
- Blocks execution if verification is missing
- Parameters: min_verifications (default: 1)

### 2. Risk Threshold Enforcement
For domains with risk (trading, financial, safety-critical):
- Create a RiskThresholdChecker that blocks operations exceeding max_risk_threshold
- Default max_risk_threshold: 0.8 (block if risk_level > 0.8)
- Can also require explicit 'approved' flag for high-risk operations

### 3. Data Availability Enforcement
Create DataAvailabilityChecker for operations that require specific data:
- Parameters: required_fields (list of state keys that must be present)
- Blocks if any required field is missing or None

### 4. Confidence-Based Quality Gates
Create QualityGateRouter to reroute low-quality outputs:
- Parameters: threshold (default: 0.7), key (default: "confidence"), target (retry/review node)
- Routes to target if output_state[key] < threshold

### 5. Human Escalation for Critical Decisions
For irreversible or high-stakes decisions:
- Create HumanEscalationRouter that routes to human review node
- Triggers on: risk_level > 0.9, confidence < 0.5, or requires_human_review flag

## Task

Design PolicyCheckers and PolicyRouters that ENFORCE safety for this system.

**For PolicyCheckers (MUST include):**

1. **VerificationRequirementChecker**: Ensures VERIFY instruction executed before high-risk actions
   - Parameters: min_verifications=1, strict_mode=True
   - instructions_to_track: [VERIFY]
   - BLOCKS execution if verification count < min_verifications

2. **DataAvailabilityChecker**: Ensures required data exists before processing
   - Parameters: required_fields=["field1", "field2"], strict_mode=True
   - BLOCKS if required fields missing

3. **Domain-specific completion checkers**: Ensure workflow stages complete
   - Parameters: min_required=N (number of required completions)
   - BLOCKS if completion count < min_required

4. **ComplianceChecker** (for regulated domains): Validates regulatory requirements
   - Parameters: max_risk_threshold=0.8, requires_approval=False
   - BLOCKS high-risk operations or those lacking approval

**For PolicyRouters (MUST include):**

1. **LowConfidenceRouter**: Reroutes when output confidence is low
   - Parameters: threshold=0.7, key="confidence", target="retry_node"
   - Routes to target when confidence < threshold

2. **HighRiskRouter**: Escalates high-risk outputs
   - Parameters: threshold=0.8, key="risk_level", target="safe_node"
   - Routes to safer path when risk > threshold

3. **HumanEscalationRouter**: Routes to human review
   - Parameters: confidence_threshold=0.5, risk_threshold=0.9, target="human_review"
   - Routes when confidence too low OR risk too high

**Output Requirements:**

1. **checkers**: List of PolicyCheckerSpec with:
   - class_name, description, check_logic_description
   - parameters (with MEANINGFUL defaults), parameter_descriptions
   - instructions_to_track (which instruction types to monitor)
   - error_message_template (specific error for violations)

2. **routers**: List of PolicyRouterSpec with:
   - class_name, description, route_logic_description
   - parameters (with thresholds and targets), parameter_descriptions
   - target_node (node to route to), trigger_condition (when to trigger)

3. **yaml_checkers**: Policy configurations for policies.yaml
4. **yaml_routers**: Router configurations for policies.yaml
5. **design_rationale**: Explain your SAFETY-FOCUSED design decisions

**Domain-Specific Examples:**

- **Trading Domain**: 
  - VerificationRequirementChecker (require VERIFY before TOOL_CALL for trades)
  - RiskThresholdChecker (block trades with risk_level > 0.8)
  - AnalystCompletionChecker (require all analysts to complete)
  - HighRiskTradeRouter (escalate to safe_debator if risk high)
  - LowConfidenceRouter (retry research if confidence < 0.7)

- **Customer Service**:
  - SentimentEscalationRouter (human review if sentiment negative)
  - ComplianceChecker (ensure responses meet policy)
  - QualityGateRouter (retry if quality_score < 0.6)

CRITICAL: Design policies that ACTUALLY BLOCK unsafe operations, not just log warnings!
"""
        return prompt

    def _assess_node_risk(self, classification) -> str:
        """Assess the risk level of a node based on its instruction type.

        Args:
            classification: Node classification result.

        Returns:
            Risk level: "high", "medium", or "low".
        """
        from .schemas import InstructionTypeEnum, CoreType

        # High-risk instruction types that require verification
        high_risk_instructions = {
            InstructionTypeEnum.TOOL_CALL,  # External system interaction
            InstructionTypeEnum.TOOL_BUILD,  # Code generation
            InstructionTypeEnum.RESPOND,  # Final user-facing output
            InstructionTypeEnum.DELEGATE,  # Multi-agent delegation
        }

        # High-risk cores
        high_risk_cores = {
            CoreType.EXECUTION,  # External system interfaces
        }

        if classification.instruction_type in high_risk_instructions:
            return "high"
        if classification.core in high_risk_cores:
            return "high"
        return "low"


def design_policies(
    analysis: RepoAnalysisOutput,
    classifications: NodeClassificationBatch,
    config: PolicyDesignerConfig | None = None,
) -> PolicyDesignOutput:
    """Convenience function to design policies.

    Args:
        analysis: Repository analysis output.
        classifications: Node classification results.
        config: Optional designer configuration.

    Returns:
        PolicyDesignOutput with checker and router specifications.
    """
    designer = PolicyDesigner(config=config)
    return designer.design(analysis, classifications)
