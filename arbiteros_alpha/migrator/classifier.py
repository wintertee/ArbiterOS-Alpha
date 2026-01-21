"""LLM-based instruction type classifier using LangChain.

This module provides automatic classification of function nodes into
ArbiterOS instruction types using LLM inference with structured output.

Enhanced features:
- Domain-aware classification using analyzer context
- Batch classification with JSON schema validation
- Support for factory function patterns
- Wrapper name generation
"""

import logging
import os
from dataclasses import dataclass

from pydantic import BaseModel, Field

from .parser import ParsedFunction
from .schemas import (
    CoreType,
    FunctionInfo,
    InstructionTypeEnum,
    NodeClassificationBatch,
    NodeClassificationResult,
    RepoAnalysisOutput,
)

logger = logging.getLogger(__name__)


# All instruction types organized by core
INSTRUCTION_TYPES = {
    "CognitiveCore": {
        "GENERATE": "Content generation and reasoning. Creates novel content, performs reasoning, generates responses.",
        "DECOMPOSE": "Task decomposition and planning. Breaks down complex goals into structured, actionable plans.",
        "REFLECT": "Self-critique and quality improvement. Performs self-assessment on generated content.",
    },
    "MemoryCore": {
        "LOAD": "Retrieves information from external knowledge base to ground the agent.",
        "STORE": "Writes or updates information in long-term memory.",
        "COMPRESS": "Reduces token count using summarization or keyword extraction.",
        "FILTER": "Selectively prunes context to keep only relevant information.",
        "STRUCTURE": "Transforms unstructured text into structured format (e.g., JSON).",
        "RENDER": "Transforms structured data into coherent natural language.",
    },
    "ExecutionCore": {
        "TOOL_CALL": "External API and tool interactions. Executes predefined external functions.",
        "TOOL_BUILD": "Writes new code to create novel tools on-the-fly.",
        "DELEGATE": "Passes sub-tasks to specialized agents in multi-agent systems.",
        "RESPOND": "Yields final user-facing output and signals task completion.",
    },
    "NormativeCore": {
        "VERIFY": "Correctness validation against schemas, specifications, and factual sources.",
        "CONSTRAIN": "Policy compliance enforcement. Applies safety constraints and guidelines.",
        "FALLBACK": "Resilient recovery execution when preceding instructions fail.",
        "INTERRUPT": "Human-in-the-loop governance. Pauses execution for human review.",
    },
    "MetacognitiveCore": {
        "PREDICT_SUCCESS": "Estimates probability of successfully completing current task.",
        "EVALUATE_PROGRESS": "Strategic assessment of reasoning path viability.",
        "MONITOR_RESOURCES": "Tracks token usage, computational cost, and latency.",
    },
    "AdaptiveCore": {
        "UPDATE_KNOWLEDGE": "Integrates new information into knowledge base.",
        "REFINE_SKILL": "Improves existing capabilities through self-generated testing.",
        "LEARN_PREFERENCE": "Internalizes feedback from human interaction or rewards.",
        "FORMULATE_EXPERIMENT": "Designs experiments for active learning loops.",
    },
    "SocialCore": {
        "COMMUNICATE": "Sends structured message to another agent.",
        "NEGOTIATE": "Multi-turn dialogue to reach agreement with another agent.",
        "PROPOSE_VOTE": "Submits formal proposal and initiates consensus protocol.",
        "FORM_COALITION": "Dynamically forms temporary group of agents.",
    },
    "AffectiveCore": {
        "INFER_INTENT": "Analyzes communication to infer underlying goals and preferences.",
        "MODEL_USER_STATE": "Constructs model of user's cognitive or emotional state.",
        "ADAPT_RESPONSE": "Modifies response to align with user's inferred state.",
        "MANAGE_TRUST": "Evaluates and manages trust level with user.",
    },
}

# Flatten for easy lookup
ALL_INSTRUCTION_TYPES: list[str] = []
INSTRUCTION_TO_CORE: dict[str, str] = {}
for core, instructions in INSTRUCTION_TYPES.items():
    for instr in instructions:
        ALL_INSTRUCTION_TYPES.append(instr)
        INSTRUCTION_TO_CORE[instr] = core


class NodeClassification(BaseModel):
    """Classification result for a single node function.

    This is used as a structured output schema for LLM classification.
    Legacy schema for backward compatibility with single-function classification.
    """

    instruction_type: str = Field(
        description="The ACF instruction type (e.g., GENERATE, VERIFY, TOOL_CALL)"
    )
    core: str = Field(
        description="The core this instruction belongs to (e.g., CognitiveCore, NormativeCore)"
    )
    confidence: float = Field(
        ge=0.0, le=1.0, description="Confidence score from 0.0 to 1.0"
    )
    reasoning: str = Field(
        description="Brief explanation for why this classification was chosen"
    )


@dataclass
class ClassificationConfig:
    """Configuration for the instruction classifier.

    Attributes:
        api_key: OpenAI API key. Defaults to OPENAI_API_KEY env var.
        base_url: API base URL. Defaults to OPENAI_BASE_URL env var or OpenAI default.
        model: Model name to use. Defaults to "gpt-4o".
        temperature: Sampling temperature. Lower = more deterministic.
    """

    api_key: str | None = None
    base_url: str | None = None
    model: str = "gpt-4o"
    temperature: float = 0.0

    def __post_init__(self) -> None:
        """Load defaults from environment variables."""
        if self.api_key is None:
            self.api_key = os.environ.get("OPENAI_API_KEY")
        if self.base_url is None:
            self.base_url = os.environ.get("OPENAI_BASE_URL")


class InstructionClassifier:
    """Classifies function nodes into ArbiterOS instruction types.

    Uses LangChain with structured output to classify functions based on
    their name, docstring, and source code.

    Enhanced with:
    - Domain context awareness from RepoAnalysisOutput
    - Batch classification with validated JSON output
    - Factory function pattern support
    - Automatic wrapper name generation

    Example:
        >>> classifier = InstructionClassifier()
        >>> result = classifier.classify(parsed_function)
        >>> print(f"{result.instruction_type} ({result.confidence})")

        # With domain context
        >>> batch_result = classifier.classify_batch_with_context(
        ...     functions, analysis_output
        ... )
        >>> for r in batch_result.classifications:
        ...     print(f"{r.function_name} -> {r.instruction_type}")
    """

    def __init__(self, config: ClassificationConfig | None = None) -> None:
        """Initialize the classifier.

        Args:
            config: Optional configuration. Uses defaults if not provided.
        """
        self.config = config or ClassificationConfig()
        self._llm = None

    def _get_llm(self):
        """Lazy-load the LLM client."""
        if self._llm is None:
            try:
                from langchain_openai import ChatOpenAI
            except ImportError:
                raise ImportError(
                    "langchain-openai is required for LLM classification. "
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
        # Use function_calling method for more flexible schemas
        structured_llm = llm.with_structured_output(
            output_schema, method="function_calling"
        )
        result = structured_llm.invoke(prompt)
        return result

    def classify(self, function: ParsedFunction) -> NodeClassification:
        """Classify a single function into an instruction type.

        Args:
            function: The parsed function to classify.

        Returns:
            NodeClassification with the predicted type and confidence.
        """
        llm = self._get_llm()

        # Build the prompt
        prompt = self._build_classification_prompt(function)

        # Use structured output with function_calling method
        structured_llm = llm.with_structured_output(
            NodeClassification, method="function_calling"
        )

        result = structured_llm.invoke(prompt)
        return result

    def classify_batch(
        self, functions: list[ParsedFunction]
    ) -> list[NodeClassification]:
        """Classify multiple functions (sequential).

        Args:
            functions: List of parsed functions to classify.

        Returns:
            List of classifications in the same order as input.
        """
        return [self.classify(func) for func in functions]

    def classify_batch_with_context(
        self,
        functions: list[FunctionInfo],
        analysis: RepoAnalysisOutput | None = None,
    ) -> NodeClassificationBatch:
        """Classify multiple functions with domain context in a single LLM call.

        This method uses the domain analysis to provide better context for
        classification, resulting in more accurate instruction type assignments.
        All functions are classified in a single batch call for efficiency.

        Args:
            functions: List of FunctionInfo objects from repo scan.
            analysis: Optional RepoAnalysisOutput providing domain context.

        Returns:
            NodeClassificationBatch with all classifications.
        """
        logger.info(f"Batch classifying {len(functions)} functions")

        prompt = self._build_batch_classification_prompt(functions, analysis)
        result = self._invoke_structured(prompt, NodeClassificationBatch)

        logger.info(f"Classified {len(result.classifications)} functions")
        return result

    def classify_manual(
        self, function: ParsedFunction, instruction_type: str
    ) -> NodeClassification:
        """Create a classification with manual input.

        Args:
            function: The parsed function.
            instruction_type: The manually selected instruction type.

        Returns:
            NodeClassification with the manual selection.
        """
        core = INSTRUCTION_TO_CORE.get(instruction_type, "CognitiveCore")
        return NodeClassification(
            instruction_type=instruction_type,
            core=core,
            confidence=1.0,
            reasoning="Manually selected by user",
        )

    def classify_manual_batch(
        self,
        function_name: str,
        file_path: str,
        instruction_type: InstructionTypeEnum,
        is_factory: bool = False,
    ) -> NodeClassificationResult:
        """Create a batch classification result with manual input.

        Args:
            function_name: Name of the function.
            file_path: Path to the file containing the function.
            instruction_type: The manually selected instruction type.
            is_factory: Whether this is a factory function.

        Returns:
            NodeClassificationResult with the manual selection.
        """
        from .schemas import INSTRUCTION_TO_CORE as SCHEMA_INSTRUCTION_TO_CORE

        core = SCHEMA_INSTRUCTION_TO_CORE.get(instruction_type, CoreType.COGNITIVE)
        wrapper_name = self._generate_wrapper_name(function_name, instruction_type)

        return NodeClassificationResult(
            function_name=function_name,
            file_path=file_path,
            instruction_type=instruction_type,
            core=core,
            confidence=1.0,
            reasoning="Manually selected by user",
            wrapper_name=wrapper_name,
            is_factory=is_factory,
        )

    def _build_classification_prompt(self, function: ParsedFunction) -> str:
        """Build the classification prompt for a function.

        Args:
            function: The parsed function to classify.

        Returns:
            The prompt string for the LLM.
        """
        # Build instruction type reference
        types_reference = []
        for core, instructions in INSTRUCTION_TYPES.items():
            types_reference.append(f"\n{core}:")
            for instr, desc in instructions.items():
                types_reference.append(f"  - {instr}: {desc}")

        types_str = "\n".join(types_reference)

        prompt = f"""You are an expert at classifying LLM agent functions into the Agent Constitution Framework (ACF) instruction types.

## Available Instruction Types
{types_str}

## Function to Classify

**Name:** {function.name}

**Docstring:**
{function.docstring or "(no docstring)"}

**Source Code:**
```python
{function.source_code}
```

## Task

Analyze this function and classify it into ONE of the ACF instruction types listed above.

Consider:
1. What is the primary purpose of this function?
2. Does it generate content (Cognitive), validate/check things (Normative), interact with external systems (Execution), or manage memory (Memory)?
3. What specific operation does it perform?

Provide your classification with:
- The instruction_type (e.g., "GENERATE", "VERIFY", "TOOL_CALL")
- The core it belongs to (e.g., "CognitiveCore", "NormativeCore")
- A confidence score (0.0 to 1.0)
- Brief reasoning for your choice
"""
        return prompt

    def _build_batch_classification_prompt(
        self,
        functions: list[FunctionInfo],
        analysis: RepoAnalysisOutput | None = None,
    ) -> str:
        """Build the batch classification prompt with domain context.

        Args:
            functions: List of functions to classify.
            analysis: Optional domain analysis for context.

        Returns:
            The prompt string for the LLM.
        """
        # Build instruction type reference
        types_reference = []
        for core, instructions in INSTRUCTION_TYPES.items():
            types_reference.append(f"\n{core}:")
            for instr, desc in instructions.items():
                types_reference.append(f"  - {instr}: {desc}")

        types_str = "\n".join(types_reference)

        # Build domain context section
        domain_context = ""
        if analysis:
            domain_context = f"""
## Domain Context

**Domain:** {analysis.domain}
**Description:** {analysis.domain_description}
**Framework:** {analysis.framework.value}

### Identified Agent Roles
"""
            for role in analysis.agent_roles:
                domain_context += f"- **{role.role_name}**: {role.description}\n"
                domain_context += f"  - Functions: {', '.join(role.functions)}\n"
                domain_context += (
                    f"  - Suggested instruction: {role.suggested_instruction.value}\n"
                )

            domain_context += "\n### Workflow Stages\n"
            for stage in analysis.workflow_stages:
                domain_context += f"- **{stage.stage_name}**: {stage.description}\n"
                domain_context += f"  - Roles: {', '.join(stage.agent_roles)}\n"

        # Build functions section
        functions_section = "## Functions to Classify\n\n"
        for i, func in enumerate(functions, 1):
            factory_marker = " [FACTORY FUNCTION]" if func.is_factory else ""
            functions_section += f"""### {i}. {func.name}{factory_marker}

**File:** {func.file_path}
**Parameters:** {", ".join(func.parameters)}

**Docstring:**
{func.docstring or "(no docstring)"}

**Source Code:**
```python
{func.source_code[:2000]}{"...(truncated)" if len(func.source_code) > 2000 else ""}
```

---
"""

        prompt = f"""You are an expert at classifying LLM agent functions into the Agent Constitution Framework (ACF) instruction types.

## Available Instruction Types
{types_str}
{domain_context}
{functions_section}

## Task

Analyze ALL the functions listed above and classify each one into the most appropriate ACF instruction type.

For each function, provide:
1. **instruction_type**: The ACF instruction type (e.g., GENERATE, VERIFY, TOOL_CALL)
2. **core**: The core it belongs to (e.g., CognitiveCore, NormativeCore)  
3. **confidence**: A confidence score from 0.0 to 1.0
4. **reasoning**: Brief explanation for your choice
5. **wrapper_name**: A suggested wrapper function name (e.g., "govern_analyst" for an analyst function)
6. **is_factory**: Whether this is a factory function that returns a node function

Consider the domain context when making classifications. For example:
- In a trading system, analysts generate reports (GENERATE), researchers debate (REFLECT/NEGOTIATE), managers make decisions (EVALUATE_PROGRESS/VERIFY)
- Factory functions that return node functions should be classified based on what the INNER function does

Be consistent in your classifications for similar functions.
"""
        return prompt

    def _generate_wrapper_name(
        self, function_name: str, instruction_type: InstructionTypeEnum
    ) -> str:
        """Generate a wrapper function name based on function and instruction type.

        Args:
            function_name: Original function name.
            instruction_type: The classified instruction type.

        Returns:
            Suggested wrapper function name.
        """
        # Extract role from function name
        name_lower = function_name.lower()

        # Common patterns
        if "analyst" in name_lower:
            return "govern_analyst"
        if "researcher" in name_lower:
            return "govern_researcher"
        if "trader" in name_lower:
            return "govern_trader"
        if "manager" in name_lower or "judge" in name_lower:
            return "govern_manager"
        if "risk" in name_lower:
            return "govern_risk_analyst"
        if "tool" in name_lower:
            return "govern_tool_call"

        # Fallback based on instruction type
        type_lower = instruction_type.value.lower()
        return f"govern_{type_lower}"

    @staticmethod
    def get_all_instruction_types() -> list[str]:
        """Get list of all available instruction types.

        Returns:
            List of instruction type names.
        """
        return ALL_INSTRUCTION_TYPES.copy()

    @staticmethod
    def get_instruction_core(instruction_type: str) -> str:
        """Get the core for an instruction type.

        Args:
            instruction_type: The instruction type name.

        Returns:
            The core name, or "CognitiveCore" as default.
        """
        return INSTRUCTION_TO_CORE.get(instruction_type, "CognitiveCore")


def classify_functions_with_context(
    functions: list[FunctionInfo],
    analysis: RepoAnalysisOutput | None = None,
    config: ClassificationConfig | None = None,
) -> NodeClassificationBatch:
    """Convenience function to classify functions with domain context.

    Args:
        functions: List of FunctionInfo objects to classify.
        analysis: Optional RepoAnalysisOutput for domain context.
        config: Optional classifier configuration.

    Returns:
        NodeClassificationBatch with all classifications.
    """
    classifier = InstructionClassifier(config=config)
    return classifier.classify_batch_with_context(functions, analysis)
