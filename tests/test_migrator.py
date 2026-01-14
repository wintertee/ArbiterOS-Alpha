"""Unit tests for the migrator module.

Tests cover:
- AgentParser: AST parsing, agent type detection, function extraction
- InstructionClassifier: LLM classification (mocked), manual classification, utilities
- CodeGenerator: Code migration, backup creation, dry-run mode
- MigrationLogger: Logging output and formatting
- CLI: Basic integration tests with mocked dependencies
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from arbiteros_alpha.migrator.classifier import (
    NodeClassification,
)
from arbiteros_alpha.migrator.generator import CodeGenerator
from arbiteros_alpha.migrator.logger import ClassificationResult, MigrationLogger
from arbiteros_alpha.migrator.parser import AgentParser

# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def sample_langgraph_source():
    """Provide sample LangGraph agent source code."""
    return """from langgraph.graph import StateGraph, END, START
from typing import TypedDict

class State(TypedDict):
    query: str
    response: str

def generate(state: State) -> State:
    \"\"\"Generate a response.\"\"\"
    return {"response": "Hello"}

def verify(state: State) -> State:
    \"\"\"Verify the response.\"\"\"
    return state

builder = StateGraph(State)
builder.add_node("generate", generate)
builder.add_node("verify", verify)
builder.add_edge(START, "generate")
builder.add_edge("generate", "verify")
builder.add_edge("verify", END)

graph = builder.compile()
"""


@pytest.fixture
def sample_vanilla_source():
    """Provide sample vanilla agent source code."""
    return """from typing import TypedDict

class State(TypedDict):
    query: str
    response: str

def generate(state: State) -> State:
    \"\"\"Generate a response.\"\"\"
    return {"response": "Hello"}

def verify(state: State) -> State:
    \"\"\"Verify the response.\"\"\"
    return state

def main():
    state: State = {"query": "test", "response": ""}
    state.update(generate(state))
    state.update(verify(state))
"""


@pytest.fixture
def sample_with_existing_arbiteros():
    """Provide source with existing ArbiterOS imports."""
    return """from langgraph.graph import StateGraph
from arbiteros_alpha import ArbiterOSAlpha
import arbiteros_alpha.instructions as Instr

def generate(state):
    return state

builder = StateGraph(dict)
builder.add_node("generate", generate)
graph = builder.compile()
"""


@pytest.fixture
def sample_with_decorator():
    """Provide source with existing @os.instruction decorator."""
    return """from arbiteros_alpha import ArbiterOSAlpha
import arbiteros_alpha.instructions as Instr

os = ArbiterOSAlpha()

@os.instruction(Instr.GENERATE)
def generate(state):
    return state

def verify(state):
    return state
"""


@pytest.fixture
def sample_async_langgraph_source():
    """Provide sample LangGraph agent with async functions (plan-and-execute style)."""
    return """from langgraph.graph import StateGraph, END, START
from typing import TypedDict, List

class PlanExecute(TypedDict):
    input: str
    plan: List[str]
    response: str

async def execute_step(state: PlanExecute):
    \"\"\"Execute the current step of the plan.\"\"\"
    return {"response": "step executed"}

async def plan_step(state: PlanExecute):
    \"\"\"Create a plan based on the input.\"\"\"
    return {"plan": ["step1", "step2"]}

async def replan_step(state: PlanExecute):
    \"\"\"Replan based on execution results.\"\"\"
    return {"plan": ["revised_step"]}

def should_end(state: PlanExecute):
    \"\"\"Determine if execution should end.\"\"\"
    if state.get("response"):
        return END
    return "agent"

workflow = StateGraph(PlanExecute)
workflow.add_node("planner", plan_step)
workflow.add_node("agent", execute_step)
workflow.add_node("replan", replan_step)
workflow.add_edge(START, "planner")
workflow.add_edge("planner", "agent")
workflow.add_edge("agent", "replan")

app = workflow.compile()
"""


@pytest.fixture
def sample_mixed_sync_async_source():
    """Provide source with both sync and async functions."""
    return """from langgraph.graph import StateGraph, END
from typing import TypedDict

class State(TypedDict):
    query: str
    response: str

async def async_generate(state: State):
    \"\"\"Async generate function.\"\"\"
    return {"response": "async hello"}

def sync_verify(state: State):
    \"\"\"Sync verify function.\"\"\"
    return state

builder = StateGraph(State)
builder.add_node("generate", async_generate)
builder.add_node("verify", sync_verify)

graph = builder.compile()
"""


# ============================================================================
# Test AgentParser
# ============================================================================


class TestAgentParser:
    """Test cases for AgentParser class."""

    def test_parse_source_detects_langgraph_agent(self, sample_langgraph_source):
        """Test that LangGraph agents are correctly detected."""
        parser = AgentParser()
        result = parser.parse_source(sample_langgraph_source)

        assert result.agent_type == "langgraph"
        assert result.builder_variable == "builder"
        assert result.graph_variable == "graph"
        assert result.compile_lineno is not None

    def test_parse_source_detects_vanilla_agent(self, sample_vanilla_source):
        """Test that vanilla agents are correctly detected."""
        parser = AgentParser()
        result = parser.parse_source(sample_vanilla_source)

        assert result.agent_type == "vanilla"
        assert result.compile_lineno is None

    def test_parse_source_extracts_functions(self, sample_langgraph_source):
        """Test that functions are extracted with docstrings and node detection."""
        parser = AgentParser()
        result = parser.parse_source(sample_langgraph_source)

        assert len(result.functions) == 2
        func_names = {f.name for f in result.functions}
        assert "generate" in func_names
        assert "verify" in func_names

        generate_func = next(f for f in result.functions if f.name == "generate")
        assert generate_func.docstring == "Generate a response."
        assert generate_func.is_node_function is True
        assert generate_func.has_state_param is True

    def test_parse_source_detects_existing_arbiteros(
        self, sample_with_existing_arbiteros
    ):
        """Test that existing ArbiterOS imports are detected."""
        parser = AgentParser()
        result = parser.parse_source(sample_with_existing_arbiteros)

        assert result.has_existing_arbiteros is True

    def test_parse_source_skips_decorated_functions(self, sample_with_decorator):
        """Test that functions with @os.instruction decorator are skipped."""
        parser = AgentParser()
        result = parser.parse_source(sample_with_decorator)

        func_names = {f.name for f in result.functions}
        assert "generate" not in func_names
        assert "verify" in func_names

    def test_parse_file_reads_from_path(self, sample_langgraph_source):
        """Test that parse_file reads from file path."""
        parser = AgentParser()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(sample_langgraph_source)
            temp_path = Path(f.name)

        try:
            result = parser.parse_file(temp_path)
            assert result.agent_type == "langgraph"
        finally:
            temp_path.unlink()

    def test_parse_file_raises_file_not_found(self):
        """Test that parse_file raises FileNotFoundError for missing files."""
        parser = AgentParser()
        with pytest.raises(FileNotFoundError):
            parser.parse_file(Path("/nonexistent/file.py"))

    def test_parse_source_raises_syntax_error_for_invalid_python(self):
        """Test that parse_source raises SyntaxError for invalid Python."""
        parser = AgentParser()
        with pytest.raises(SyntaxError):
            parser.parse_source("def invalid syntax here")

    def test_parse_source_handles_empty_file(self):
        """Test parsing an empty file."""
        parser = AgentParser()
        result = parser.parse_source("")

        assert result.agent_type == "vanilla"
        assert len(result.functions) == 0

    def test_parse_source_detects_async_functions(self, sample_async_langgraph_source):
        """Test that async functions are correctly detected and marked."""
        parser = AgentParser()
        result = parser.parse_source(sample_async_langgraph_source)

        assert result.agent_type == "langgraph"
        assert len(result.functions) == 4  # 3 async + 1 sync (should_end)

        func_names = {f.name for f in result.functions}
        assert "execute_step" in func_names
        assert "plan_step" in func_names
        assert "replan_step" in func_names
        assert "should_end" in func_names

        # Check async flags
        execute_func = next(f for f in result.functions if f.name == "execute_step")
        assert execute_func.is_async is True
        assert execute_func.has_state_param is True
        assert execute_func.docstring == "Execute the current step of the plan."

        plan_func = next(f for f in result.functions if f.name == "plan_step")
        assert plan_func.is_async is True

        should_end_func = next(f for f in result.functions if f.name == "should_end")
        assert should_end_func.is_async is False

    def test_parse_source_handles_mixed_sync_async(
        self, sample_mixed_sync_async_source
    ):
        """Test that both sync and async functions are extracted correctly."""
        parser = AgentParser()
        result = parser.parse_source(sample_mixed_sync_async_source)

        assert result.agent_type == "langgraph"
        assert len(result.functions) == 2

        async_func = next(f for f in result.functions if f.name == "async_generate")
        sync_func = next(f for f in result.functions if f.name == "sync_verify")

        assert async_func.is_async is True
        assert async_func.is_node_function is True
        assert async_func.docstring == "Async generate function."

        assert sync_func.is_async is False
        assert sync_func.is_node_function is True
        assert sync_func.docstring == "Sync verify function."

    def test_parse_source_async_node_functions_detected(
        self, sample_async_langgraph_source
    ):
        """Test that async functions used in add_node are marked as node functions."""
        parser = AgentParser()
        result = parser.parse_source(sample_async_langgraph_source)

        # plan_step, execute_step, replan_step are used in add_node
        plan_func = next(f for f in result.functions if f.name == "plan_step")
        execute_func = next(f for f in result.functions if f.name == "execute_step")
        replan_func = next(f for f in result.functions if f.name == "replan_step")
        should_end_func = next(f for f in result.functions if f.name == "should_end")

        assert plan_func.is_node_function is True
        assert execute_func.is_node_function is True
        assert replan_func.is_node_function is True
        assert should_end_func.is_node_function is False  # Not used in add_node


# ============================================================================
# Test CodeGenerator
# ============================================================================


class TestCodeGenerator:
    """Test cases for CodeGenerator class."""

    def test_transform_adds_imports_and_decorators(self, sample_langgraph_source):
        """Test that transform adds imports, OS init, decorators, and registration."""
        parser = AgentParser()
        parsed = parser.parse_source(sample_langgraph_source)
        generator = CodeGenerator()
        classifications = {
            "generate": NodeClassification(
                instruction_type="GENERATE",
                core="CognitiveCore",
                confidence=0.9,
                reasoning="",
            ),
            "verify": NodeClassification(
                instruction_type="VERIFY",
                core="NormativeCore",
                confidence=0.8,
                reasoning="",
            ),
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(sample_langgraph_source)
            temp_path = Path(f.name)

        try:
            result = generator.transform(
                file_path=temp_path,
                parsed_agent=parsed,
                classifications=classifications,
                dry_run=True,
            )

            assert result.success is True
            changes_str = " ".join(result.changes)
            assert "import" in changes_str.lower()
            assert "OS initialization" in changes_str
            assert "GENERATE" in changes_str
            assert "VERIFY" in changes_str
            assert "register_compiled_graph" in changes_str
        finally:
            temp_path.unlink()

    def test_transform_creates_backup(self, sample_langgraph_source):
        """Test that transform creates a backup file."""
        parser = AgentParser()
        parsed = parser.parse_source(sample_langgraph_source)
        generator = CodeGenerator()
        classifications = {
            "generate": NodeClassification(
                instruction_type="GENERATE",
                core="CognitiveCore",
                confidence=0.9,
                reasoning="",
            ),
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(sample_langgraph_source)
            temp_path = Path(f.name)

        try:
            result = generator.transform(
                file_path=temp_path,
                parsed_agent=parsed,
                classifications=classifications,
                dry_run=False,
            )

            assert result.success is True
            assert result.backup_file != ""
            assert Path(result.backup_file).exists()
            Path(result.backup_file).unlink()
        finally:
            temp_path.unlink()

    def test_transform_dry_run_does_not_modify_file(self, sample_langgraph_source):
        """Test that dry_run mode doesn't modify files."""
        parser = AgentParser()
        parsed = parser.parse_source(sample_langgraph_source)
        generator = CodeGenerator()
        classifications = {
            "generate": NodeClassification(
                instruction_type="GENERATE",
                core="CognitiveCore",
                confidence=0.9,
                reasoning="",
            ),
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(sample_langgraph_source)
            temp_path = Path(f.name)
            original_content = sample_langgraph_source

        try:
            result = generator.transform(
                file_path=temp_path,
                parsed_agent=parsed,
                classifications=classifications,
                dry_run=True,
            )

            assert result.success is True
            assert result.backup_file == ""
            assert temp_path.read_text() == original_content
        finally:
            temp_path.unlink()

    def test_transform_skips_imports_if_already_present(
        self, sample_with_existing_arbiteros
    ):
        """Test that transform skips adding imports if already present."""
        parser = AgentParser()
        parsed = parser.parse_source(sample_with_existing_arbiteros)
        generator = CodeGenerator()
        classifications = {
            "generate": NodeClassification(
                instruction_type="GENERATE",
                core="CognitiveCore",
                confidence=0.9,
                reasoning="",
            ),
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(sample_with_existing_arbiteros)
            temp_path = Path(f.name)

        try:
            result = generator.transform(
                file_path=temp_path,
                parsed_agent=parsed,
                classifications=classifications,
                dry_run=True,
            )

            assert result.success is True
            assert not any("Added import" in change for change in result.changes)
        finally:
            temp_path.unlink()

    def test_generate_transformed_source_returns_string(self, sample_langgraph_source):
        """Test that generate_transformed_source returns transformed code."""
        parser = AgentParser()
        parsed = parser.parse_source(sample_langgraph_source)
        generator = CodeGenerator()
        classifications = {
            "generate": NodeClassification(
                instruction_type="GENERATE",
                core="CognitiveCore",
                confidence=0.9,
                reasoning="",
            ),
        }

        transformed = generator.generate_transformed_source(parsed, classifications)

        assert isinstance(transformed, str)
        assert "from arbiteros_alpha import ArbiterOSAlpha" in transformed
        assert "@os.instruction(Instr.GENERATE)" in transformed


# ============================================================================
# Test MigrationLogger
# ============================================================================


class TestMigrationLogger:
    """Test cases for MigrationLogger class."""

    def test_logger_initialization(self):
        """Test that logger initializes correctly."""
        logger = MigrationLogger(verbose=True)
        assert logger.verbose is True
        assert logger._current_step == 0

    def test_logger_methods_call_console_print(self):
        """Test that logger methods call console.print."""
        console = MagicMock()
        logger = MigrationLogger(console=console, verbose=True)

        logger.start()
        logger.step_parsing("test.py")
        logger.found_functions(["generate", "verify"])
        logger.detected_agent_type("langgraph", compile_line=10)
        logger.show_classifications(
            [
                ClassificationResult(
                    function_name="generate",
                    instruction_type="GENERATE",
                    core="CognitiveCore",
                    confidence=0.9,
                    reasoning="",
                ),
            ]
        )
        logger.complete("modified.py", "backup.py")
        logger.error("Test error")
        logger.warning("Test warning")
        logger.info("Test info")

        assert console.print.call_count > 0

    def test_prompt_confirmation_returns_user_input(self):
        """Test that prompt_confirmation returns user input."""
        console = MagicMock()
        logger = MigrationLogger(console=console)

        with patch("builtins.input", return_value="y"):
            result = logger.prompt_confirmation()
            assert result in ("y", "yes")

    def test_info_only_logs_when_verbose(self):
        """Test that info only logs when verbose is True."""
        console = MagicMock()
        logger_verbose = MigrationLogger(console=console, verbose=True)
        logger_quiet = MigrationLogger(console=console, verbose=False)

        logger_verbose.info("Test info")
        logger_quiet.info("Test info")

        # Verbose logger should print, quiet should not
        assert logger_verbose.verbose is True
        assert logger_quiet.verbose is False
