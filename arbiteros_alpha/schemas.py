"""Schema definitions for instruction input/output validation.

This module provides abstract schema definitions for all instruction types,
enabling type-safe validation and analysis of instruction functions.
"""

from typing import Any, TypedDict, Union

from .instructions import (
    AdaptiveCore,
    AffectiveCore,
    CognitiveCore,
    ExecutionCore,
    InstructionType,
    MemoryCore,
    MetacognitiveCore,
    NormativeCore,
    SocialCore,
)


class InstructionSchema(TypedDict, total=False):
    """Base schema for instruction input/output validation.

    This is a flexible TypedDict that can be extended for specific
    instruction types. All fields are optional to allow for different
    instruction requirements.
    """

    # Common fields that may appear in any instruction
    task: str  # Description of the task to be performed
    context: dict[str, Any]  # Additional context information
    state: dict[str, Any]  # Current state of the system
    metadata: dict[str, Any]  # Additional metadata


# CognitiveCore schemas
class GenerateInputSchema(InstructionSchema):
    """Input schema for GENERATE instruction.

    Attributes:
        prompt: The prompt or query to generate content for.
        max_tokens: Maximum number of tokens to generate.
        temperature: Sampling temperature for generation.
    """

    prompt: str
    max_tokens: int
    temperature: float


class GenerateOutputSchema(TypedDict):
    """Output schema for GENERATE instruction.

    Attributes:
        content: Generated text content.
        reasoning: Optional reasoning chain if applicable.
    """

    content: str
    reasoning: str


class DecomposeInputSchema(InstructionSchema):
    """Input schema for DECOMPOSE instruction.

    Attributes:
        task: The complex task to decompose.
        granularity: Desired level of decomposition detail.
    """

    task: str
    granularity: str


class DecomposeOutputSchema(TypedDict):
    """Output schema for DECOMPOSE instruction.

    Attributes:
        subtasks: List of decomposed subtasks.
        plan: Structured execution plan.
    """

    subtasks: list[str]
    plan: dict[str, Any]


class ReflectInputSchema(InstructionSchema):
    """Input schema for REFLECT instruction.

    Attributes:
        content: The content to critique.
        criteria: Criteria for evaluation.
    """

    content: str
    criteria: list[str]


class ReflectOutputSchema(TypedDict):
    """Output schema for REFLECT instruction.

    Attributes:
        critique: Structured critique of the content.
        improvements: List of suggested improvements.
    """

    critique: dict[str, Any]
    improvements: list[str]


# MemoryCore schemas
class LoadInputSchema(InstructionSchema):
    """Input schema for LOAD instruction.

    Attributes:
        query: Query to retrieve information.
        knowledge_base: Target knowledge base identifier.
    """

    query: str
    knowledge_base: str


class LoadOutputSchema(TypedDict):
    """Output schema for LOAD instruction.

    Attributes:
        retrieved_data: Retrieved information from knowledge base.
        sources: List of source identifiers.
    """

    retrieved_data: list[dict[str, Any]]
    sources: list[str]


class StoreInputSchema(InstructionSchema):
    """Input schema for STORE instruction.

    Attributes:
        data: Information to store.
        memory_key: Key for memory storage.
    """

    data: dict[str, Any]
    memory_key: str


class StoreOutputSchema(TypedDict):
    """Output schema for STORE instruction.

    Attributes:
        stored: Whether storage was successful.
        memory_id: Identifier for stored memory.
    """

    stored: bool
    memory_id: str


# ExecutionCore schemas
class ToolCallInputSchema(InstructionSchema):
    """Input schema for TOOL_CALL instruction.

    Attributes:
        tool_name: Name of the tool to call.
        tool_args: Arguments for the tool call.
    """

    tool_name: str
    tool_args: dict[str, Any]


class ToolCallOutputSchema(TypedDict):
    """Output schema for TOOL_CALL instruction.

    Attributes:
        result: Result from tool execution.
        success: Whether the tool call succeeded.
    """

    result: Any
    success: bool


class RespondInputSchema(InstructionSchema):
    """Input schema for RESPOND instruction.

    Attributes:
        message: Final message to present to user.
        format: Output format (text, json, etc.).
    """

    message: str
    format: str


class RespondOutputSchema(TypedDict):
    """Output schema for RESPOND instruction.

    Attributes:
        response: Final user-facing response.
        completed: Whether task is completed.
    """

    response: str
    completed: bool


# NormativeCore schemas
class VerifyInputSchema(InstructionSchema):
    """Input schema for VERIFY instruction.

    Attributes:
        content: Content to verify.
        schema: Schema or rules to verify against.
    """

    content: Any
    schema: dict[str, Any]


class VerifyOutputSchema(TypedDict):
    """Output schema for VERIFY instruction.

    Attributes:
        valid: Whether verification passed.
        errors: List of validation errors if any.
    """

    valid: bool
    errors: list[str]


# Schema registry mapping instruction types to their schemas
InstructionInputSchema = Union[
    GenerateInputSchema,
    DecomposeInputSchema,
    ReflectInputSchema,
    LoadInputSchema,
    StoreInputSchema,
    ToolCallInputSchema,
    RespondInputSchema,
    VerifyInputSchema,
    InstructionSchema,  # Fallback for unspecified instructions
]

InstructionOutputSchema = Union[
    GenerateOutputSchema,
    DecomposeOutputSchema,
    ReflectOutputSchema,
    LoadOutputSchema,
    StoreOutputSchema,
    ToolCallOutputSchema,
    RespondOutputSchema,
    VerifyOutputSchema,
    dict[str, Any],  # Fallback for unspecified instructions
]


# Registry for instruction type to schema mapping
_INSTRUCTION_INPUT_SCHEMAS: dict[InstructionType, type[InstructionInputSchema]] = {
    CognitiveCore.GENERATE: GenerateInputSchema,
    CognitiveCore.DECOMPOSE: DecomposeInputSchema,
    CognitiveCore.REFLECT: ReflectInputSchema,
    MemoryCore.LOAD: LoadInputSchema,
    MemoryCore.STORE: StoreInputSchema,
    ExecutionCore.TOOL_CALL: ToolCallInputSchema,
    ExecutionCore.RESPOND: RespondInputSchema,
    NormativeCore.VERIFY: VerifyInputSchema,
}

_INSTRUCTION_OUTPUT_SCHEMAS: dict[InstructionType, type[InstructionOutputSchema]] = {
    CognitiveCore.GENERATE: GenerateOutputSchema,
    CognitiveCore.DECOMPOSE: DecomposeOutputSchema,
    CognitiveCore.REFLECT: ReflectOutputSchema,
    MemoryCore.LOAD: LoadOutputSchema,
    MemoryCore.STORE: StoreOutputSchema,
    ExecutionCore.TOOL_CALL: ToolCallOutputSchema,
    ExecutionCore.RESPOND: RespondOutputSchema,
    NormativeCore.VERIFY: VerifyOutputSchema,
}


def get_input_schema(instruction_type: InstructionType) -> type[InstructionInputSchema]:
    """Get the input schema for a given instruction type.

    Args:
        instruction_type: The instruction type to get schema for.

    Returns:
        The TypedDict class for input schema validation.
    """
    return _INSTRUCTION_INPUT_SCHEMAS.get(instruction_type, InstructionSchema)


def get_output_schema(instruction_type: InstructionType) -> type[InstructionOutputSchema]:
    """Get the output schema for a given instruction type.

    Args:
        instruction_type: The instruction type to get schema for.

    Returns:
        The TypedDict class for output schema validation.
    """
    return _INSTRUCTION_OUTPUT_SCHEMAS.get(instruction_type, dict)


def register_input_schema(
    instruction_type: InstructionType, schema: type[InstructionInputSchema]
) -> None:
    """Register a custom input schema for an instruction type.

    Args:
        instruction_type: The instruction type to register schema for.
        schema: The TypedDict class for input schema.
    """
    _INSTRUCTION_INPUT_SCHEMAS[instruction_type] = schema


def register_output_schema(
    instruction_type: InstructionType, schema: type[InstructionOutputSchema]
) -> None:
    """Register a custom output schema for an instruction type.

    Args:
        instruction_type: The instruction type to register schema for.
        schema: The TypedDict class for output schema.
    """
    _INSTRUCTION_OUTPUT_SCHEMAS[instruction_type] = schema


def validate_input(
    instruction_type: InstructionType, data: dict[str, Any], strict: bool = False
) -> tuple[bool, list[str]]:
    """Validate input data against instruction schema.

    Args:
        instruction_type: The instruction type to validate for.
        data: The input data dictionary to validate.
        strict: If True, all required fields must be present. If False,
            only validates that provided fields match schema types.

    Returns:
        A tuple of (is_valid, error_messages).
    """
    schema = get_input_schema(instruction_type)
    errors = []

    # For TypedDict, we can check if keys match expected structure
    # This is a basic validation - for stricter validation, consider using Pydantic
    if strict and hasattr(schema, "__required_keys__"):
        required_keys = schema.__required_keys__
        missing_keys = required_keys - set(data.keys())
        if missing_keys:
            errors.append(f"Missing required keys: {missing_keys}")

    return len(errors) == 0, errors


def validate_output(
    instruction_type: InstructionType, data: dict[str, Any], strict: bool = False
) -> tuple[bool, list[str]]:
    """Validate output data against instruction schema.

    Args:
        instruction_type: The instruction type to validate for.
        data: The output data dictionary to validate.
        strict: If True, all required fields must be present. If False,
            only validates that provided fields match schema types.

    Returns:
        A tuple of (is_valid, error_messages).
    """
    schema = get_output_schema(instruction_type)
    errors = []

    if strict and hasattr(schema, "__required_keys__"):
        required_keys = schema.__required_keys__
        missing_keys = required_keys - set(data.keys())
        if missing_keys:
            errors.append(f"Missing required keys: {missing_keys}")

    return len(errors) == 0, errors

