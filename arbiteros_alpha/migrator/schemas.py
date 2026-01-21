"""Pydantic schemas for LLM I/O in the migration pipeline.

This module defines all structured input/output schemas used for LLM interactions
during the transformation process. All LLM calls use these schemas to ensure
validated JSON I/O.
"""

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# =============================================================================
# Enums
# =============================================================================


class AgentFramework(str, Enum):
    """Supported agent frameworks for transformation."""

    LANGGRAPH = "langgraph"
    CREWAI = "crewai"
    AUTOGEN = "autogen"
    VANILLA = "vanilla"
    UNKNOWN = "unknown"


class InstructionTypeEnum(str, Enum):
    """All ACF instruction types."""

    # CognitiveCore
    GENERATE = "GENERATE"
    DECOMPOSE = "DECOMPOSE"
    REFLECT = "REFLECT"

    # MemoryCore
    LOAD = "LOAD"
    STORE = "STORE"
    COMPRESS = "COMPRESS"
    FILTER = "FILTER"
    STRUCTURE = "STRUCTURE"
    RENDER = "RENDER"

    # ExecutionCore
    TOOL_CALL = "TOOL_CALL"
    TOOL_BUILD = "TOOL_BUILD"
    DELEGATE = "DELEGATE"
    RESPOND = "RESPOND"

    # NormativeCore
    VERIFY = "VERIFY"
    CONSTRAIN = "CONSTRAIN"
    FALLBACK = "FALLBACK"
    INTERRUPT = "INTERRUPT"

    # MetacognitiveCore
    PREDICT_SUCCESS = "PREDICT_SUCCESS"
    EVALUATE_PROGRESS = "EVALUATE_PROGRESS"
    MONITOR_RESOURCES = "MONITOR_RESOURCES"

    # AdaptiveCore
    UPDATE_KNOWLEDGE = "UPDATE_KNOWLEDGE"
    REFINE_SKILL = "REFINE_SKILL"
    LEARN_PREFERENCE = "LEARN_PREFERENCE"
    FORMULATE_EXPERIMENT = "FORMULATE_EXPERIMENT"

    # SocialCore
    COMMUNICATE = "COMMUNICATE"
    NEGOTIATE = "NEGOTIATE"
    PROPOSE_VOTE = "PROPOSE_VOTE"
    FORM_COALITION = "FORM_COALITION"

    # AffectiveCore
    INFER_INTENT = "INFER_INTENT"
    MODEL_USER_STATE = "MODEL_USER_STATE"
    ADAPT_RESPONSE = "ADAPT_RESPONSE"
    MANAGE_TRUST = "MANAGE_TRUST"


class CoreType(str, Enum):
    """ACF Core types."""

    COGNITIVE = "CognitiveCore"
    MEMORY = "MemoryCore"
    EXECUTION = "ExecutionCore"
    NORMATIVE = "NormativeCore"
    METACOGNITIVE = "MetacognitiveCore"
    ADAPTIVE = "AdaptiveCore"
    SOCIAL = "SocialCore"
    AFFECTIVE = "AffectiveCore"


# Mapping from instruction to core
INSTRUCTION_TO_CORE: dict[InstructionTypeEnum, CoreType] = {
    # CognitiveCore
    InstructionTypeEnum.GENERATE: CoreType.COGNITIVE,
    InstructionTypeEnum.DECOMPOSE: CoreType.COGNITIVE,
    InstructionTypeEnum.REFLECT: CoreType.COGNITIVE,
    # MemoryCore
    InstructionTypeEnum.LOAD: CoreType.MEMORY,
    InstructionTypeEnum.STORE: CoreType.MEMORY,
    InstructionTypeEnum.COMPRESS: CoreType.MEMORY,
    InstructionTypeEnum.FILTER: CoreType.MEMORY,
    InstructionTypeEnum.STRUCTURE: CoreType.MEMORY,
    InstructionTypeEnum.RENDER: CoreType.MEMORY,
    # ExecutionCore
    InstructionTypeEnum.TOOL_CALL: CoreType.EXECUTION,
    InstructionTypeEnum.TOOL_BUILD: CoreType.EXECUTION,
    InstructionTypeEnum.DELEGATE: CoreType.EXECUTION,
    InstructionTypeEnum.RESPOND: CoreType.EXECUTION,
    # NormativeCore
    InstructionTypeEnum.VERIFY: CoreType.NORMATIVE,
    InstructionTypeEnum.CONSTRAIN: CoreType.NORMATIVE,
    InstructionTypeEnum.FALLBACK: CoreType.NORMATIVE,
    InstructionTypeEnum.INTERRUPT: CoreType.NORMATIVE,
    # MetacognitiveCore
    InstructionTypeEnum.PREDICT_SUCCESS: CoreType.METACOGNITIVE,
    InstructionTypeEnum.EVALUATE_PROGRESS: CoreType.METACOGNITIVE,
    InstructionTypeEnum.MONITOR_RESOURCES: CoreType.METACOGNITIVE,
    # AdaptiveCore
    InstructionTypeEnum.UPDATE_KNOWLEDGE: CoreType.ADAPTIVE,
    InstructionTypeEnum.REFINE_SKILL: CoreType.ADAPTIVE,
    InstructionTypeEnum.LEARN_PREFERENCE: CoreType.ADAPTIVE,
    InstructionTypeEnum.FORMULATE_EXPERIMENT: CoreType.ADAPTIVE,
    # SocialCore
    InstructionTypeEnum.COMMUNICATE: CoreType.SOCIAL,
    InstructionTypeEnum.NEGOTIATE: CoreType.SOCIAL,
    InstructionTypeEnum.PROPOSE_VOTE: CoreType.SOCIAL,
    InstructionTypeEnum.FORM_COALITION: CoreType.SOCIAL,
    # AffectiveCore
    InstructionTypeEnum.INFER_INTENT: CoreType.AFFECTIVE,
    InstructionTypeEnum.MODEL_USER_STATE: CoreType.AFFECTIVE,
    InstructionTypeEnum.ADAPT_RESPONSE: CoreType.AFFECTIVE,
    InstructionTypeEnum.MANAGE_TRUST: CoreType.AFFECTIVE,
}


# =============================================================================
# Repository Scanner Schemas
# =============================================================================


class FileInfo(BaseModel):
    """Information about a scanned file."""

    path: str = Field(..., description="Relative path from repo root")
    size_bytes: int = Field(..., description="File size in bytes")
    line_count: int = Field(..., description="Number of lines in the file")


class FunctionInfo(BaseModel):
    """Information about a function found in source code."""

    name: str = Field(..., description="Function name")
    file_path: str = Field(..., description="File containing the function")
    lineno: int = Field(..., description="Starting line number")
    end_lineno: int = Field(..., description="Ending line number")
    docstring: str | None = Field(None, description="Function docstring")
    source_code: str = Field(..., description="Full source code of the function")
    is_async: bool = Field(False, description="Whether the function is async")
    is_factory: bool = Field(
        False, description="Whether this is a factory function returning a node"
    )
    parameters: list[str] = Field(
        default_factory=list, description="Function parameter names"
    )
    decorators: list[str] = Field(
        default_factory=list, description="Decorator names on the function"
    )


class ImportInfo(BaseModel):
    """Information about imports in a file."""

    module: str = Field(..., description="Module being imported")
    names: list[str] = Field(
        default_factory=list, description="Names imported from module"
    )
    is_from_import: bool = Field(..., description="Whether this is a 'from' import")


class GraphNodeInfo(BaseModel):
    """Information about a node added to a graph."""

    node_name: str = Field(..., description="Name of the node in the graph")
    function_name: str = Field(..., description="Function/callable used for the node")
    file_path: str = Field(..., description="File where add_node is called")
    lineno: int = Field(..., description="Line number of add_node call")


class GraphEdgeInfo(BaseModel):
    """Information about an edge in the graph."""

    source: str = Field(..., description="Source node name")
    target: str = Field(..., description="Target node name")
    is_conditional: bool = Field(
        False, description="Whether this is a conditional edge"
    )
    condition_function: str | None = Field(
        None, description="Condition function name if conditional"
    )


class RepoScanResult(BaseModel):
    """Complete result of scanning a repository."""

    repo_path: str = Field(..., description="Absolute path to the repository")
    python_files: list[FileInfo] = Field(
        default_factory=list, description="All Python files found"
    )
    functions: list[FunctionInfo] = Field(
        default_factory=list, description="All functions found"
    )
    imports: dict[str, list[ImportInfo]] = Field(
        default_factory=dict, description="Imports by file path"
    )
    graph_nodes: list[GraphNodeInfo] = Field(
        default_factory=list, description="Graph nodes found"
    )
    graph_edges: list[GraphEdgeInfo] = Field(
        default_factory=list, description="Graph edges found"
    )
    state_classes: list[str] = Field(
        default_factory=list, description="State class names found"
    )
    detected_framework: AgentFramework = Field(
        AgentFramework.UNKNOWN, description="Detected agent framework"
    )


# =============================================================================
# Repository Analysis Schemas (LLM Output)
# =============================================================================


class AgentRoleInfo(BaseModel):
    """Information about an agent role identified in the system."""

    role_name: str = Field(..., description="Name of the agent role (e.g., 'analyst')")
    description: str = Field(..., description="Description of what this role does")
    functions: list[str] = Field(
        default_factory=list, description="Function names implementing this role"
    )
    suggested_instruction: InstructionTypeEnum = Field(
        ..., description="Suggested ACF instruction type for this role"
    )


class WorkflowStageInfo(BaseModel):
    """Information about a stage in the workflow."""

    stage_name: str = Field(..., description="Name of the workflow stage")
    description: str = Field(..., description="What happens in this stage")
    agent_roles: list[str] = Field(
        default_factory=list, description="Agent roles involved in this stage"
    )
    preconditions: list[str] = Field(
        default_factory=list, description="What must be true before this stage"
    )
    postconditions: list[str] = Field(
        default_factory=list, description="What should be true after this stage"
    )


class StateFieldInfo(BaseModel):
    """Information about a field in the agent state."""

    field_name: str = Field(..., description="Name of the state field")
    field_type: str = Field(..., description="Python type annotation")
    description: str = Field(..., description="Description of the field's purpose")
    produced_by: list[str] = Field(
        default_factory=list, description="Agent roles that produce this field"
    )
    consumed_by: list[str] = Field(
        default_factory=list, description="Agent roles that consume this field"
    )


class RepoAnalysisOutput(BaseModel):
    """LLM-generated analysis of a repository.

    This is the structured output schema for the repository analyzer LLM call.
    """

    framework: AgentFramework = Field(..., description="The agent framework being used")
    domain: str = Field(
        ...,
        description="The domain/task of this agent system (e.g., 'trading', 'customer_service')",
    )
    domain_description: str = Field(
        ..., description="Detailed description of what this agent system does"
    )
    agent_roles: list[AgentRoleInfo] = Field(
        default_factory=list, description="Identified agent roles in the system"
    )
    workflow_stages: list[WorkflowStageInfo] = Field(
        default_factory=list, description="Stages in the workflow"
    )
    state_fields: list[StateFieldInfo] = Field(
        default_factory=list, description="Fields in the agent state"
    )
    key_constraints: list[str] = Field(
        default_factory=list,
        description="Key constraints/rules identified in the workflow",
    )
    entry_point_file: str | None = Field(
        None, description="Main entry point file for the agent system"
    )
    graph_setup_file: str | None = Field(
        None, description="File where the graph is set up and compiled"
    )


# =============================================================================
# Node Classification Schemas (LLM Output)
# =============================================================================


class NodeClassificationResult(BaseModel):
    """Classification result for a single function/node."""

    function_name: str = Field(..., description="Name of the function")
    file_path: str = Field(..., description="File containing the function")
    instruction_type: InstructionTypeEnum = Field(
        ..., description="Classified ACF instruction type"
    )
    core: CoreType = Field(..., description="The core this instruction belongs to")
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Confidence score from 0.0 to 1.0"
    )
    reasoning: str = Field(
        ..., description="Explanation for why this classification was chosen"
    )
    wrapper_name: str = Field(
        ..., description="Suggested wrapper function name (e.g., 'govern_analyst')"
    )
    is_factory: bool = Field(False, description="Whether this is a factory function")


class NodeClassificationBatch(BaseModel):
    """Batch classification results for multiple functions.

    This is the structured output schema for the classifier LLM call.
    """

    classifications: list[NodeClassificationResult] = Field(
        default_factory=list, description="Classification results for each function"
    )
    domain_context: str = Field(
        ..., description="Brief summary of the domain context used for classification"
    )


# =============================================================================
# Policy Design Schemas (LLM Output)
# =============================================================================


class PolicyCheckerSpec(BaseModel):
    """Specification for a policy checker to be generated."""

    class_name: str = Field(
        ...,
        description="Name for the PolicyChecker class (e.g., 'AnalystCompletionChecker')",
    )
    description: str = Field(
        ..., description="Docstring describing what this checker validates"
    )
    check_logic_description: str = Field(
        ..., description="Description of the check_before logic"
    )
    parameters: dict[str, Any] = Field(
        default_factory=dict,
        description="Parameters for the checker (name -> default value)",
    )
    parameter_descriptions: dict[str, str] = Field(
        default_factory=dict, description="Descriptions for each parameter"
    )
    instructions_to_track: list[InstructionTypeEnum] = Field(
        default_factory=list, description="Instruction types this checker monitors"
    )
    error_message_template: str = Field(
        ..., description="Template for error message on failure"
    )


class PolicyRouterSpec(BaseModel):
    """Specification for a policy router to be generated."""

    class_name: str = Field(
        ..., description="Name for the PolicyRouter class (e.g., 'ConfidenceRouter')"
    )
    description: str = Field(
        ..., description="Docstring describing what this router does"
    )
    route_logic_description: str = Field(
        ..., description="Description of the route_after logic"
    )
    parameters: dict[str, Any] = Field(
        default_factory=dict,
        description="Parameters for the router (name -> default value)",
    )
    parameter_descriptions: dict[str, str] = Field(
        default_factory=dict, description="Descriptions for each parameter"
    )
    target_node: str = Field(..., description="Default target node to route to")
    trigger_condition: str = Field(..., description="Condition that triggers routing")


class PolicyYAMLSection(BaseModel):
    """A section of the policies.yaml configuration."""

    name: str = Field(..., description="Name of the policy")
    type: str = Field(..., description="Class name of the checker/router")
    enabled: bool = Field(True, description="Whether this policy is enabled")
    parameters: dict[str, Any] = Field(
        default_factory=dict, description="Parameter values"
    )


class PolicyDesignOutput(BaseModel):
    """LLM-generated policy design.

    This is the structured output schema for the policy designer LLM call.
    """

    checkers: list[PolicyCheckerSpec] = Field(
        default_factory=list, description="Policy checkers to generate"
    )
    routers: list[PolicyRouterSpec] = Field(
        default_factory=list, description="Policy routers to generate"
    )
    yaml_checkers: list[PolicyYAMLSection] = Field(
        default_factory=list, description="Checker configurations for policies.yaml"
    )
    yaml_routers: list[PolicyYAMLSection] = Field(
        default_factory=list, description="Router configurations for policies.yaml"
    )
    design_rationale: str = Field(
        ..., description="Explanation of the policy design decisions"
    )


# =============================================================================
# Schema Design Schemas (LLM Output)
# =============================================================================


class PydanticFieldSpec(BaseModel):
    """Specification for a field in a Pydantic model."""

    field_name: str = Field(..., description="Name of the field")
    field_type: str = Field(..., description="Python type annotation as string")
    description: str = Field(..., description="Field description for docstring")
    is_required: bool = Field(True, description="Whether the field is required")
    default_value: str | None = Field(
        None, description="Default value as Python code string"
    )
    validators: list[str] = Field(
        default_factory=list, description="Pydantic validators to apply"
    )


class RenderMethodSpec(BaseModel):
    """Specification for a render method on a schema."""

    method_name: str = Field(
        ..., description="Name of the render method (e.g., 'render_plan_text')"
    )
    return_type: str = Field("str", description="Return type annotation")
    description: str = Field(..., description="Method docstring")
    template: str = Field(
        ..., description="f-string template for rendering (using self.field_name)"
    )


class PydanticSchemaSpec(BaseModel):
    """Specification for a Pydantic schema to be generated."""

    class_name: str = Field(
        ..., description="Name for the Pydantic model (e.g., 'TradeDecision')"
    )
    description: str = Field(..., description="Class docstring describing the schema")
    fields: list[PydanticFieldSpec] = Field(
        default_factory=list, description="Fields in the schema"
    )
    render_methods: list[RenderMethodSpec] = Field(
        default_factory=list, description="Render methods for backward compatibility"
    )
    is_enum: bool = Field(
        False, description="Whether this should be an Enum instead of BaseModel"
    )
    enum_values: list[str] = Field(
        default_factory=list, description="Enum values if is_enum is True"
    )
    base_class: str = Field("BaseModel", description="Base class to inherit from")
    function_names: list[str] = Field(
        default_factory=list,
        description="Agent function names that should use this schema for LLM output",
    )


class LLMSchemaDesignOutput(BaseModel):
    """LLM-generated schema design.

    This is the structured output schema for the schema designer LLM call.
    """

    schemas: list[PydanticSchemaSpec] = Field(
        default_factory=list, description="Pydantic schemas to generate"
    )
    imports_needed: list[str] = Field(
        default_factory=list, description="Additional imports needed"
    )
    design_rationale: str = Field(
        ..., description="Explanation of the schema design decisions"
    )


# =============================================================================
# Code Generation Schemas
# =============================================================================


class GovernanceWrapperSpec(BaseModel):
    """Specification for a governance wrapper function."""

    wrapper_name: str = Field(
        ..., description="Name of the wrapper function (e.g., 'govern_analyst')"
    )
    instruction_type: InstructionTypeEnum = Field(
        ..., description="Instruction type to apply"
    )
    core: CoreType = Field(..., description="Core the instruction belongs to")
    description: str = Field(..., description="Docstring for the wrapper function")
    functions_to_wrap: list[str] = Field(
        default_factory=list, description="Function names that use this wrapper"
    )


class GeneratedFile(BaseModel):
    """A file generated by the code generator."""

    path: str = Field(..., description="Relative path for the generated file")
    content: str = Field(..., description="Content of the generated file")
    description: str = Field(..., description="Description of what this file contains")


class TransformationResult(BaseModel):
    """Complete result of a repository transformation."""

    success: bool = Field(..., description="Whether transformation succeeded")
    generated_files: list[GeneratedFile] = Field(
        default_factory=list, description="Files generated by the transformation"
    )
    modified_files: list[str] = Field(
        default_factory=list, description="Existing files that were modified"
    )
    backup_files: list[str] = Field(
        default_factory=list, description="Backup files created"
    )
    errors: list[str] = Field(default_factory=list, description="Error messages if any")
    warnings: list[str] = Field(
        default_factory=list, description="Warning messages if any"
    )
    summary: str = Field(..., description="Summary of the transformation")


# =============================================================================
# Configuration Schemas
# =============================================================================


class TransformConfig(BaseModel):
    """Configuration for the transformation process."""

    # LLM Configuration
    api_key: str | None = Field(None, description="OpenAI API key")
    base_url: str | None = Field(None, description="API base URL")
    model: str = Field("gpt-4o", description="Model to use for LLM calls")
    temperature: float = Field(0.0, description="LLM temperature")

    # Transformation Options
    skip_policies: bool = Field(False, description="Skip policy generation")
    skip_schemas: bool = Field(False, description="Skip schema generation")
    dry_run: bool = Field(False, description="Preview without writing files")
    interactive: bool = Field(False, description="Interactive mode with confirmations")

    # Output Configuration
    output_dir: str | None = Field(
        None, description="Output directory (defaults to repo policies/ dir)"
    )
    backup: bool = Field(True, description="Create backups of modified files")

    # File Patterns
    ignore_patterns: list[str] = Field(
        default_factory=lambda: [
            "__pycache__",
            ".git",
            ".venv",
            "venv",
            "node_modules",
            "*.pyc",
            "*.pyo",
        ],
        description="Patterns to ignore when scanning",
    )
    include_patterns: list[str] = Field(
        default_factory=lambda: ["*.py"],
        description="Patterns to include when scanning",
    )
