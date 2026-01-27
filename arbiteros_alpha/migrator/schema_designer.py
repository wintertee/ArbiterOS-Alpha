"""LLM-powered schema designer for migration pipeline.

This module provides the SchemaDesigner class that uses LLM with structured
output to design Pydantic schemas for LLM I/O based on repository analysis
and function signatures.
"""

import logging
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

from pydantic import BaseModel

from .schemas import (
    FunctionInfo,
    LLMSchemaDesignOutput,
    PydanticFieldSpec,
    RepoAnalysisOutput,
)

if TYPE_CHECKING:
    from .schemas import PolicyDesignOutput

logger = logging.getLogger(__name__)


@dataclass
class SchemaDesignerConfig:
    """Configuration for the schema designer.

    Attributes:
        api_key: OpenAI API key. Defaults to OPENAI_API_KEY env var.
        base_url: API base URL. Defaults to OPENAI_BASE_URL env var.
        model: Model name to use. Defaults to "gpt-4o".
        temperature: Sampling temperature. Lower = more deterministic.
    """

    api_key: str | None = None
    base_url: str | None = None
    model: str = "gpt-4o"
    temperature: float = 0.1

    def __post_init__(self) -> None:
        """Load defaults from environment variables."""
        if self.api_key is None:
            self.api_key = os.environ.get("OPENAI_API_KEY")
        if self.base_url is None:
            self.base_url = os.environ.get("OPENAI_BASE_URL")


class SchemaDesigner:
    """LLM-powered schema designer.

    Designs Pydantic schemas for LLM I/O based on:
    - Repository analysis (domain, agent roles, state fields)
    - Function signatures and return types
    - Best practices for structured LLM output

    Example:
        >>> designer = SchemaDesigner()
        >>> schemas = designer.design(analysis, functions)
        >>> for schema in schemas.schemas:
        ...     print(f"Schema: {schema.class_name}")
        ...     for field in schema.fields:
        ...         print(f"  - {field.field_name}: {field.field_type}")
    """

    def __init__(self, config: SchemaDesignerConfig | None = None) -> None:
        """Initialize the SchemaDesigner.

        Args:
            config: Optional configuration. Uses defaults if not provided.
        """
        self.config = config or SchemaDesignerConfig()
        self._llm = None

    def _get_llm(self):
        """Lazy-load the LLM client."""
        if self._llm is None:
            try:
                from langchain_openai import ChatOpenAI
            except ImportError:
                raise ImportError(
                    "langchain-openai is required for schema design. "
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
    ) -> BaseModel | None:
        """Invoke LLM with structured JSON output.

        Args:
            prompt: The prompt to send to the LLM.
            output_schema: Pydantic model for structured output.

        Returns:
            Parsed and validated Pydantic model instance, or None if failed.
        """
        try:
            llm = self._get_llm()
            # Use function_calling method for more flexible schemas
            structured_llm = llm.with_structured_output(
                output_schema, method="function_calling"
            )
            result = structured_llm.invoke(prompt)
            return result
        except Exception as e:
            logger.error(f"LLM invocation failed: {e}")
            return None

    def _ensure_enums_defined(
        self, result: LLMSchemaDesignOutput
    ) -> LLMSchemaDesignOutput:
        """Ensure all referenced enum types are defined in the schemas list.

        Scans all schema fields for enum types and automatically generates missing enum definitions.

        Args:
            result: The schema design output from the LLM.

        Returns:
            Updated result with all required enums defined.
        """
        import re

        # Extract all existing enum names
        existing_enums = {
            schema.class_name for schema in result.schemas if schema.is_enum
        }

        # Find all referenced enum types in field_type
        referenced_enums = set()
        for schema in result.schemas:
            if schema.is_enum:
                continue
            for field in schema.fields:
                # Look for enum-like types (capitalized words that look like enum names)
                # Common patterns: TradeDecision, RiskLevel, Priority, Status
                matches = re.findall(
                    r"\b([A-Z][a-zA-Z]*(?:Decision|Level|Priority|Status|Type|Mode|State))\b",
                    field.field_type,
                )
                referenced_enums.update(matches)

        # Find missing enums
        missing_enums = referenced_enums - existing_enums

        if missing_enums:
            logger.warning(
                f"Found {len(missing_enums)} missing enum definitions: {missing_enums}"
            )

            # Generate enum definitions for missing enums
            new_enums = []
            for enum_name in sorted(missing_enums):
                # Try to infer enum values based on common patterns
                enum_values = self._infer_enum_values(enum_name)

                if enum_values:
                    logger.info(
                        f"Auto-generating enum {enum_name} with values: {enum_values}"
                    )
                    from arbiteros_alpha.migrator.schemas import PydanticSchemaSpec

                    enum_schema = PydanticSchemaSpec(
                        class_name=enum_name,
                        description=f"{enum_name} enum for type safety.",
                        fields=[],
                        is_enum=True,
                        enum_values=enum_values,
                        base_class="str, Enum",
                        function_names=[],
                    )
                    new_enums.append(enum_schema)

            # Prepend enums to schemas list (enums must be defined first)
            result.schemas = new_enums + result.schemas
            logger.info(f"Added {len(new_enums)} auto-generated enum definitions")

        return result

    def _infer_enum_values(self, enum_name: str) -> list[str]:
        """Infer likely enum values based on enum name patterns.

        Args:
            enum_name: Name of the enum (e.g., "TradeDecision", "RiskLevel")

        Returns:
            List of likely enum values.
        """
        # Common enum patterns
        patterns = {
            "Decision": ["BUY", "SELL", "HOLD"],
            "TradeDecision": ["BUY", "SELL", "HOLD"],
            "Level": ["LOW", "MEDIUM", "HIGH"],
            "RiskLevel": ["LOW", "MEDIUM", "HIGH"],
            "Priority": ["LOW", "MEDIUM", "HIGH", "URGENT"],
            "Status": ["PENDING", "IN_PROGRESS", "COMPLETED", "FAILED"],
            "Sentiment": ["POSITIVE", "NEUTRAL", "NEGATIVE"],
            "Type": ["TYPE_A", "TYPE_B", "TYPE_C"],
            "Mode": ["AUTO", "MANUAL"],
            "State": ["ACTIVE", "INACTIVE"],
        }

        # Try exact match first
        if enum_name in patterns:
            return patterns[enum_name]

        # Try suffix match
        for suffix, values in patterns.items():
            if enum_name.endswith(suffix):
                return values

        # Default: return empty and let user define
        logger.warning(
            f"Could not infer values for enum {enum_name}, skipping auto-generation"
        )
        return []

    def design(
        self,
        analysis: RepoAnalysisOutput,
        functions: list[FunctionInfo],
        policy_design: "PolicyDesignOutput | None" = None,
    ) -> LLMSchemaDesignOutput:
        """Design LLM I/O schemas based on analysis and functions.

        Args:
            analysis: Repository analysis output.
            functions: List of functions that interact with LLMs.
            policy_design: Optional policy design to align schemas with policy requirements.

        Returns:
            LLMSchemaDesignOutput with schema specifications.
        """
        logger.info(f"Designing LLM schemas for domain: {analysis.domain}")

        prompt = self._build_design_prompt(analysis, functions, policy_design)
        result = self._invoke_structured(prompt, LLMSchemaDesignOutput)

        if result is None:
            logger.error("Schema design failed: LLM returned None")
            # Return empty schema design as fallback
            return LLMSchemaDesignOutput(
                schemas=[],
                imports_needed=[],
                reasoning="Schema design failed - LLM returned None",
            )

        logger.info(f"Designed {len(result.schemas)} schemas")

        # Post-process to ensure all referenced enums are defined
        result = self._ensure_enums_defined(result)

        # Post-process to enrich schemas with policy-required fields
        if policy_design:
            result = self._enrich_with_policy_fields(result, policy_design)

        return result

    def _enrich_with_policy_fields(
        self,
        result: LLMSchemaDesignOutput,
        policy_design: "PolicyDesignOutput",
    ) -> LLMSchemaDesignOutput:
        """Enrich schemas with fields required by policies.

        Analyzes policy routers to determine what fields they expect in output_state
        and adds those fields to relevant schemas.

        Args:
            result: The schema design output from the LLM.
            policy_design: The policy design with checkers and routers.

        Returns:
            Updated result with enriched schemas.
        """
        # Extract field requirements from routers
        required_fields = self._extract_policy_field_requirements(policy_design)

        if not required_fields:
            logger.info("No policy field requirements found")
            return result

        logger.info(
            f"Found {len(required_fields)} policy field requirements: {list(required_fields.keys())}"
        )

        # Add missing fields to all non-enum schemas
        enriched_count = 0
        for schema in result.schemas:
            if schema.is_enum:
                continue

            existing_field_names = {f.field_name for f in schema.fields}

            # Add missing fields
            for field_name, field_spec in required_fields.items():
                if field_name not in existing_field_names:
                    logger.info(
                        f"Adding field '{field_name}' to schema {schema.class_name}"
                    )
                    schema.fields.append(field_spec)
                    enriched_count += 1

        logger.info(
            f"Enriched {enriched_count} schema fields based on policy requirements"
        )
        return result

    def _extract_policy_field_requirements(
        self, policy_design: "PolicyDesignOutput"
    ) -> dict[str, "PydanticFieldSpec"]:
        """Extract field requirements from policy routers.

        Analyzes router trigger conditions to determine what fields are expected
        in output_state.

        Args:
            policy_design: The policy design output.

        Returns:
            Dictionary mapping field names to their PydanticFieldSpec.
        """
        required_fields: dict[str, PydanticFieldSpec] = {}

        # Common field patterns based on router trigger conditions
        field_patterns = {
            "confidence": PydanticFieldSpec(
                field_name="confidence",
                field_type="float",
                description="Confidence score of the analysis or decision (0.0-1.0)",
                is_required=False,
                default_value="None",
                validators=["ge=0.0", "le=1.0"],
            ),
            "quality_score": PydanticFieldSpec(
                field_name="quality_score",
                field_type="float",
                description="Quality assessment score of the output (0.0-1.0)",
                is_required=False,
                default_value="None",
                validators=["ge=0.0", "le=1.0"],
            ),
            "risk_level": PydanticFieldSpec(
                field_name="risk_level",
                field_type="float",
                description="Risk level assessment (0.0-1.0, where 1.0 is highest risk)",
                is_required=False,
                default_value="None",
                validators=["ge=0.0", "le=1.0"],
            ),
        }

        # Analyze routers to find required fields
        for router in policy_design.routers:
            trigger = router.trigger_condition.lower()

            # Check for field mentions in trigger conditions
            for field_name in field_patterns:
                if field_name in trigger or field_name in str(router.parameters):
                    if field_name not in required_fields:
                        required_fields[field_name] = field_patterns[field_name]
                        logger.info(
                            f"Router '{router.class_name}' requires field: {field_name}"
                        )

        return required_fields

    def _build_design_prompt(
        self,
        analysis: RepoAnalysisOutput,
        functions: list[FunctionInfo],
        policy_design: "PolicyDesignOutput | None" = None,
    ) -> str:
        """Build the schema design prompt.

        Args:
            analysis: Repository analysis output.
            functions: Functions that interact with LLMs.
            policy_design: Optional policy design to align schemas with.

        Returns:
            Formatted prompt string.
        """
        # Format domain context
        domain_section = f"""## Domain Context

**Domain:** {analysis.domain}
**Description:** {analysis.domain_description}

### Agent Roles
"""
        for role in analysis.agent_roles:
            domain_section += f"- **{role.role_name}**: {role.description}\n"

        # Format state fields
        state_section = "\n### State Fields\n"
        for field in analysis.state_fields:
            state_section += (
                f"- **{field.field_name}** ({field.field_type}): {field.description}\n"
            )
            if field.produced_by:
                state_section += f"  - Produced by: {', '.join(field.produced_by)}\n"
            if field.consumed_by:
                state_section += f"  - Consumed by: {', '.join(field.consumed_by)}\n"

        # Format functions
        functions_section = "\n### Functions Requiring Schemas\n"
        for func in functions[:30]:  # Limit to 30 functions
            functions_section += f"\n**{func.name}** ({func.file_path})\n"
            if func.docstring:
                functions_section += f"- Docstring: {func.docstring[:200]}...\n"
            # Look for LLM calls in source
            if "invoke" in func.source_code or "llm" in func.source_code.lower():
                functions_section += "- Contains LLM invocation\n"
            # Show truncated source
            functions_section += f"```python\n{func.source_code[:500]}{'...' if len(func.source_code) > 500 else ''}\n```\n"

        # Format policy requirements (if provided)
        policy_section = ""
        if policy_design:
            policy_section = "\n### Policy Requirements\n\n"
            policy_section += "The following policies will be applied to this system. Schemas MUST include fields that these policies expect:\n\n"

            if policy_design.routers:
                policy_section += (
                    "**Policy Routers (require fields in output_state):**\n"
                )
                for router in policy_design.routers:
                    policy_section += (
                        f"- **{router.class_name}**: {router.trigger_condition}\n"
                    )
                    if router.parameters:
                        params_str = ", ".join(
                            f"{k}={v}" for k, v in router.parameters.items()
                        )
                        policy_section += f"  - Parameters: {params_str}\n"
                policy_section += "\n"

            if policy_design.checkers:
                policy_section += "**Policy Checkers (validate constraints):**\n"
                for checker in policy_design.checkers:
                    policy_section += (
                        f"- **{checker.class_name}**: {checker.description}\n"
                    )
                policy_section += "\n"

            policy_section += """**CRITICAL SCHEMA REQUIREMENTS:**
- All agent output schemas MUST include a `confidence: float` field (0.0-1.0) for confidence-based routing
- Analysis/decision schemas SHOULD include `quality_score: float` for quality gates
- Trading/risk schemas SHOULD include `risk_level: float` for risk-based routing
- These fields enable governance policies to function correctly
"""

        prompt = f"""You are an expert at designing Pydantic schemas for structured LLM outputs. Your task is to design schemas that enforce consistent, validated JSON I/O for an agent system.

{domain_section}
{state_section}
{functions_section}
{policy_section}

## Schema Design Principles

1. **Strict Typing**: Use specific types (e.g., Literal, Enum) instead of generic str where possible
2. **Validation**: Add Pydantic validators for constraints (min_length, ge, le, etc.)
3. **Documentation**: Include Field descriptions for all fields
4. **Metadata Fields**: Include confidence, quality_score, and other governance fields
5. **Policy Alignment**: Ensure schemas provide fields that policies expect
6. **Backward Compatibility**: Add render methods to convert to legacy string formats

### Example Schemas

```python
from pydantic import BaseModel, Field
from enum import Enum
from typing import List, Literal, Optional

class TradeDecision(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"

class AnalystReportOutput(BaseModel):
    report_markdown: str = Field(
        ...,
        min_length=1,
        description="The full analyst report in Markdown format."
    )
    confidence: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Confidence score of the analysis (0.0-1.0)"
    )

class ResearchManagerOutput(BaseModel):
    recommendation: TradeDecision = Field(
        ..., description="Decisive stance (BUY/SELL/HOLD)."
    )
    summary_bull: str = Field(
        ..., min_length=1, description="Summary of bull-side points."
    )
    summary_bear: str = Field(
        ..., min_length=1, description="Summary of bear-side points."
    )
    rationale: str = Field(
        ..., min_length=1, description="Explanation for the recommendation."
    )
    strategic_actions: List[str] = Field(
        default_factory=list, description="Steps to implement."
    )
    confidence: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Confidence in this recommendation (0.0-1.0)"
    )
    quality_score: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Quality assessment of the analysis (0.0-1.0)"
    )
    
    def render_plan_text(self) -> str:
        actions = "\\n".join([f"- {{a}}" for a in self.strategic_actions])
        return (
            f"Recommendation: {{self.recommendation.value}}\\n"
            f"Bull Summary: {{self.summary_bull}}\\n"
            f"Bear Summary: {{self.summary_bear}}\\n"
            f"Rationale: {{self.rationale}}\\n"
            f"Confidence: {{self.confidence or 'N/A'}}\\n"
            f"Strategic Actions:\\n{{actions}}"
        )
```

## Task

Design Pydantic schemas for the LLM outputs in this agent system.

For each agent role, consider:
1. What structured output should the LLM produce?
2. What fields are required vs optional?
3. What validation is needed?
4. What metadata fields are needed for governance (confidence, quality_score, risk_level)?
5. What render methods are needed for backward compatibility?

**CRITICAL: Include Governance Metadata Fields**

Every agent output schema MUST include appropriate metadata fields for governance:

1. **confidence** (float, 0.0-1.0, optional): How confident the agent is in its output
   - Required for: All GENERATE, EVALUATE_PROGRESS, RESPOND instructions
   - Used by: LowConfidenceRouter, QualityGateRouter, HumanEscalationRouter

2. **quality_score** (float, 0.0-1.0, optional): Self-assessed quality of the output
   - Required for: Analysis, research, and decision-making agents
   - Used by: QualityGateRouter

3. **risk_level** (float, 0.0-1.0, optional): Assessed risk level
   - Required for: Trading, financial, or high-stakes decision agents
   - Used by: HighRiskOverrideRouter

Example - Always include these patterns:
```python
confidence: Optional[float] = Field(
    None, ge=0.0, le=1.0,
    description="Confidence score (0.0-1.0)"
)
quality_score: Optional[float] = Field(
    None, ge=0.0, le=1.0,
    description="Quality assessment (0.0-1.0)"
)
```

**IMPORTANT: Generate Enums First!**
Before creating BaseModel schemas, identify all enum types needed (e.g., TradeDecision, Priority, Status).
For domains involving decisions or categorical values, always create enums to ensure type safety.

Common patterns:
- Trading: TradeDecision (BUY/SELL/HOLD), RiskLevel (LOW/MEDIUM/HIGH)
- Tasks: Priority (LOW/MEDIUM/HIGH/URGENT), Status (PENDING/IN_PROGRESS/COMPLETED)
- Analysis: Sentiment (POSITIVE/NEUTRAL/NEGATIVE)

Provide:
1. **schemas**: List of PydanticSchemaSpec, starting with ENUMS, then BaseModel schemas:
   
   For Enums:
   - class_name (e.g., "TradeDecision")
   - description (docstring)
   - is_enum: true
   - enum_values: ["BUY", "SELL", "HOLD"]
   - base_class: "str, Enum"
   - fields: [] (leave empty for enums)
   - function_names: [] (enums are referenced by other schemas)
   
   For BaseModel schemas:
   - class_name (e.g., "AnalystReportOutput")
   - description (docstring)
   - fields (list of field specs with name, type, description, validation)
   - render_methods (for backward compatibility) - use {{self.field_name}} syntax (will be wrapped in f-string)
   - is_enum: false
   - base_class: "BaseModel"
   - **function_names** (CRITICAL: list of function names that should use this schema)

2. **imports_needed**: Additional imports (e.g., "from typing import List")

3. **design_rationale**: Explain your schema design decisions, especially which enums you created and why

CRITICAL: 
- Generate ALL enums FIRST in the schemas list (is_enum=true)
- Then generate BaseModel schemas that reference those enums
- For each BaseModel schema, specify which agent functions use it via `function_names`
- This enables automatic wiring of schemas into LLM calls during transformation

Example mapping:
- BullResearcherOutput -> function_names: ["bull_node", "create_bull_researcher"]
- TraderDecisionOutput -> function_names: ["trader_node", "create_trader"]

Focus on schemas that:
- First define all enums for categorical/decision values
- Capture the domain-specific outputs (e.g., decisions, reports, analyses)
- Enforce consistency across agent outputs through strong typing
- Enable validation and error detection
- Support both structured processing and human-readable rendering
- Have clear function_names mappings for automatic wiring
"""
        return prompt


def design_schemas(
    analysis: RepoAnalysisOutput,
    functions: list[FunctionInfo],
    config: SchemaDesignerConfig | None = None,
    policy_design: "PolicyDesignOutput | None" = None,
) -> LLMSchemaDesignOutput:
    """Convenience function to design schemas.

    Args:
        analysis: Repository analysis output.
        functions: Functions that interact with LLMs.
        config: Optional designer configuration.
        policy_design: Optional policy design to align schemas with policy requirements.

    Returns:
        LLMSchemaDesignOutput with schema specifications.
    """
    designer = SchemaDesigner(config=config)
    return designer.design(analysis, functions, policy_design)
