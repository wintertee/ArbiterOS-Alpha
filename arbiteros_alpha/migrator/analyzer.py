"""LLM-powered repository analyzer for migration pipeline.

This module provides the RepoAnalyzer class that uses LLM with structured
output to understand repository structure, domain context, agent roles,
and workflow patterns.
"""

import logging
import os
from dataclasses import dataclass

from pydantic import BaseModel

from .schemas import (
    RepoAnalysisOutput,
    RepoScanResult,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Instruction Type Reference for LLM
# =============================================================================

INSTRUCTION_TYPES_REFERENCE = """
## Available ACF Instruction Types

### CognitiveCore (Probabilistic reasoning)
- GENERATE: Content generation, reasoning, formulating queries. Most general-purpose cognitive instruction.
- DECOMPOSE: Breaks complex tasks into smaller sub-tasks or creates execution plans.
- REFLECT: Self-critique on generated output to identify flaws and improvements.

### MemoryCore (Context and memory management)
- LOAD: Retrieves information from external knowledge base.
- STORE: Writes/updates information in long-term memory.
- COMPRESS: Reduces token count via summarization or keyword extraction.
- FILTER: Selectively prunes context to keep relevant information.
- STRUCTURE: Transforms unstructured text into structured format (e.g., JSON).
- RENDER: Transforms structured data into natural language.

### ExecutionCore (External system interfaces)
- TOOL_CALL: Executes predefined external functions (API calls, database queries).
- TOOL_BUILD: Writes new code to create novel tools on-the-fly.
- DELEGATE: Passes sub-tasks to specialized agents in multi-agent systems.
- RESPOND: Yields final user-facing output, signals task completion.

### NormativeCore (Rules and constraints)
- VERIFY: Objective correctness checks against verifiable sources.
- CONSTRAIN: Applies compliance rules (safety, style, ethics).
- FALLBACK: Executes recovery strategies when instructions fail.
- INTERRUPT: Pauses execution to request human input.

### MetacognitiveCore (Self-assessment and resources)
- PREDICT_SUCCESS: Estimates probability of successfully completing task.
- EVALUATE_PROGRESS: Strategic assessment of reasoning path viability.
- MONITOR_RESOURCES: Tracks token usage, cost, latency.

### AdaptiveCore (Learning and improvement)
- UPDATE_KNOWLEDGE: Integrates new information into knowledge base.
- REFINE_SKILL: Improves capabilities through testing/fine-tuning.
- LEARN_PREFERENCE: Internalizes feedback from human interaction.
- FORMULATE_EXPERIMENT: Designs experiments for active learning.

### SocialCore (Multi-agent collaboration)
- COMMUNICATE: Sends structured message to another agent.
- NEGOTIATE: Multi-turn dialogue to reach agreement with another agent.
- PROPOSE_VOTE: Submits proposal and initiates consensus protocol.
- FORM_COALITION: Dynamically forms temporary agent groups.

### AffectiveCore (Socio-emotional reasoning)
- INFER_INTENT: Analyzes communication to infer underlying goals.
- MODEL_USER_STATE: Constructs model of user's cognitive/emotional state.
- ADAPT_RESPONSE: Modifies response to align with user's inferred state.
- MANAGE_TRUST: Evaluates and manages trust level with user.
"""


@dataclass
class AnalyzerConfig:
    """Configuration for the repository analyzer.

    Attributes:
        api_key: OpenAI API key. Defaults to OPENAI_API_KEY env var.
        base_url: API base URL. Defaults to OPENAI_BASE_URL env var.
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


class RepoAnalyzer:
    """LLM-powered repository analyzer.

    Analyzes repository scan results using LLM to understand:
    - Agent framework being used
    - Domain/task context
    - Agent roles and their responsibilities
    - Workflow stages and dependencies
    - State schema design
    - Key constraints and rules

    Example:
        >>> from arbiteros_alpha.migrator.repo_scanner import scan_repository
        >>> scan_result = scan_repository("/path/to/repo")
        >>> analyzer = RepoAnalyzer()
        >>> analysis = analyzer.analyze(scan_result)
        >>> print(f"Domain: {analysis.domain}")
        >>> for role in analysis.agent_roles:
        ...     print(f"Role: {role.role_name} -> {role.suggested_instruction}")
    """

    def __init__(self, config: AnalyzerConfig | None = None) -> None:
        """Initialize the RepoAnalyzer.

        Args:
            config: Optional configuration. Uses defaults if not provided.
        """
        self.config = config or AnalyzerConfig()
        self._llm = None

    def _get_llm(self):
        """Lazy-load the LLM client."""
        if self._llm is None:
            try:
                from langchain_openai import ChatOpenAI
            except ImportError:
                raise ImportError(
                    "langchain-openai is required for LLM analysis. "
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

    def analyze(self, scan_result: RepoScanResult) -> RepoAnalysisOutput:
        """Analyze a repository scan result.

        Args:
            scan_result: The result from RepoScanner.

        Returns:
            RepoAnalysisOutput with LLM-generated analysis.
        """
        logger.info("Starting LLM analysis of repository structure")

        prompt = self._build_analysis_prompt(scan_result)
        analysis = self._invoke_structured(prompt, RepoAnalysisOutput)

        logger.info(
            f"Analysis complete: domain={analysis.domain}, "
            f"roles={len(analysis.agent_roles)}, "
            f"stages={len(analysis.workflow_stages)}"
        )

        return analysis

    def _build_analysis_prompt(self, scan_result: RepoScanResult) -> str:
        """Build the analysis prompt from scan results.

        Args:
            scan_result: Repository scan result.

        Returns:
            Formatted prompt string.
        """
        # Format file list
        files_summary = self._format_files_summary(scan_result)

        # Format functions summary
        functions_summary = self._format_functions_summary(scan_result)

        # Format graph structure
        graph_summary = self._format_graph_summary(scan_result)

        # Format imports
        imports_summary = self._format_imports_summary(scan_result)

        # Format state classes
        state_summary = self._format_state_summary(scan_result)

        prompt = f"""You are an expert at analyzing LLM agent codebases. Analyze the following repository structure and extract key information about its agent system.

## Repository Information

**Path:** {scan_result.repo_path}
**Detected Framework:** {scan_result.detected_framework.value}

## Files Structure

{files_summary}

## Functions Found

{functions_summary}

## Graph Structure (Nodes and Edges)

{graph_summary}

## Import Patterns

{imports_summary}

## State Classes

{state_summary}

{INSTRUCTION_TYPES_REFERENCE}

## Task

Analyze this agent system and provide:

1. **Framework identification**: Confirm or correct the detected framework.

2. **Domain identification**: What domain/task is this agent system designed for? (e.g., trading, customer_service, code_generation, research)

3. **Agent roles**: Identify distinct agent roles in the system. For each role:
   - Name the role (e.g., "analyst", "researcher", "trader")
   - Describe what it does
   - List functions that implement this role
   - Suggest the most appropriate ACF instruction type

4. **Workflow stages**: Identify the stages/phases in the workflow:
   - Name each stage
   - Describe what happens
   - List which agent roles are involved
   - Note preconditions and postconditions

5. **State fields**: Identify key fields in the agent state:
   - Field name and type
   - What it represents
   - Which roles produce/consume it

6. **Key constraints**: What rules or constraints are important for this workflow?

7. **Entry points**: Identify the main entry point file and graph setup file.

Be specific and detailed. Use the function names and structure you observe to make accurate classifications.
"""
        return prompt

    def _format_files_summary(self, scan_result: RepoScanResult) -> str:
        """Format file information for the prompt."""
        if not scan_result.python_files:
            return "(no Python files found)"

        lines = []
        for f in scan_result.python_files[:50]:  # Limit to 50 files
            lines.append(f"- {f.path} ({f.line_count} lines)")

        if len(scan_result.python_files) > 50:
            lines.append(f"... and {len(scan_result.python_files) - 50} more files")

        return "\n".join(lines)

    def _format_functions_summary(self, scan_result: RepoScanResult) -> str:
        """Format function information for the prompt."""
        if not scan_result.functions:
            return "(no functions found)"

        lines = []
        for func in scan_result.functions[:100]:  # Limit to 100 functions
            factory_marker = " [FACTORY]" if func.is_factory else ""
            async_marker = " [ASYNC]" if func.is_async else ""
            decorators = f" @{', @'.join(func.decorators)}" if func.decorators else ""

            # Include brief docstring if available
            doc_preview = ""
            if func.docstring:
                doc_preview = f" - {func.docstring[:100]}..."

            lines.append(
                f"- {func.name}({', '.join(func.parameters)}){async_marker}{factory_marker}{decorators}"
                f"\n  File: {func.file_path}:{func.lineno}{doc_preview}"
            )

        if len(scan_result.functions) > 100:
            lines.append(f"\n... and {len(scan_result.functions) - 100} more functions")

        return "\n".join(lines)

    def _format_graph_summary(self, scan_result: RepoScanResult) -> str:
        """Format graph structure for the prompt."""
        lines = []

        if scan_result.graph_nodes:
            lines.append("### Nodes:")
            for node in scan_result.graph_nodes:
                lines.append(
                    f"- Node '{node.node_name}' -> function: {node.function_name} "
                    f"(in {node.file_path}:{node.lineno})"
                )
        else:
            lines.append("### Nodes: (none found)")

        lines.append("")

        if scan_result.graph_edges:
            lines.append("### Edges:")
            for edge in scan_result.graph_edges:
                cond = (
                    f" [conditional: {edge.condition_function}]"
                    if edge.is_conditional
                    else ""
                )
                lines.append(f"- {edge.source} -> {edge.target}{cond}")
        else:
            lines.append("### Edges: (none found)")

        return "\n".join(lines)

    def _format_imports_summary(self, scan_result: RepoScanResult) -> str:
        """Format import patterns for the prompt."""
        # Aggregate imports across files
        import_counts: dict[str, int] = {}
        for file_path, imports in scan_result.imports.items():
            for imp in imports:
                key = imp.module
                import_counts[key] = import_counts.get(key, 0) + 1

        # Sort by frequency
        sorted_imports = sorted(import_counts.items(), key=lambda x: -x[1])

        lines = ["Key imports (by frequency):"]
        for module, count in sorted_imports[:30]:
            lines.append(f"- {module}: {count} files")

        return "\n".join(lines)

    def _format_state_summary(self, scan_result: RepoScanResult) -> str:
        """Format state class information for the prompt."""
        if not scan_result.state_classes:
            return "(no state classes found)"

        return "\n".join(f"- {name}" for name in scan_result.state_classes)


def analyze_repository(
    scan_result: RepoScanResult,
    config: AnalyzerConfig | None = None,
) -> RepoAnalysisOutput:
    """Convenience function to analyze a scanned repository.

    Args:
        scan_result: Result from RepoScanner.
        config: Optional analyzer configuration.

    Returns:
        RepoAnalysisOutput with LLM-generated analysis.
    """
    analyzer = RepoAnalyzer(config=config)
    return analyzer.analyze(scan_result)
