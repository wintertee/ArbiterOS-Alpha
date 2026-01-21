"""AST-based parser for detecting agent types and extracting function information.

This module provides tools to parse Python source files and identify:
- LangGraph-based agents (StateGraph, add_node, add_edge, compile patterns)
- native agents (regular Python functions)
- Function definitions with their docstrings and source code
"""

import ast
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


@dataclass
class ParsedFunction:
    """Extracted function information from source code.

    Attributes:
        name: The function name.
        docstring: The function's docstring, if present.
        source_code: The full source code of the function.
        lineno: The line number where the function starts.
        end_lineno: The line number where the function ends.
        is_node_function: True if this function is used in add_node().
        has_state_param: True if the function accepts a state parameter.
    """

    name: str
    docstring: str | None
    source_code: str
    lineno: int
    end_lineno: int
    is_node_function: bool = False
    has_state_param: bool = False


@dataclass
class ParsedAgent:
    """Parsed agent structure from source code.

    Attributes:
        agent_type: Either "langgraph" or "native".
        functions: List of parsed functions found in the file.
        compile_lineno: Line number where compile() is called (LangGraph only).
        graph_variable: Variable name holding the compiled graph.
        builder_variable: Variable name of the StateGraph builder.
        source_lines: The original source code split into lines.
        imports_end_lineno: Line number where imports section ends.
        has_existing_arbiteros: True if file already has ArbiterOS imports.
    """

    agent_type: Literal["langgraph", "native"]
    functions: list[ParsedFunction]
    compile_lineno: int | None = None
    graph_variable: str | None = None
    builder_variable: str | None = None
    source_lines: list[str] = field(default_factory=list)
    imports_end_lineno: int = 0
    has_existing_arbiteros: bool = False


class AgentParser:
    """Parser for extracting agent information from Python source files.

    This parser uses Python's AST module to analyze source files and detect:
    - Whether the agent uses LangGraph or is a native Python agent
    - All function definitions that could be node functions
    - The location of the compile() call for LangGraph agents
    - Import statements and their locations

    Example:
        >>> parser = AgentParser()
        >>> result = parser.parse_file("my_agent.py")
        >>> print(result.agent_type)  # "langgraph" or "native"
        >>> for func in result.functions:
        ...     print(f"{func.name}: {func.docstring}")
    """

    def __init__(self) -> None:
        """Initialize the AgentParser."""
        self._node_function_names: set[str] = set()
        self._source_lines: list[str] = []

    def parse_file(self, file_path: str | Path) -> ParsedAgent:
        """Parse a Python source file and extract agent information.

        Args:
            file_path: Path to the Python source file.

        Returns:
            ParsedAgent containing all extracted information.

        Raises:
            FileNotFoundError: If the file doesn't exist.
            SyntaxError: If the file contains invalid Python syntax.
        """
        file_path = Path(file_path)
        source_code = file_path.read_text(encoding="utf-8")
        return self.parse_source(source_code)

    def parse_source(self, source_code: str) -> ParsedAgent:
        """Parse Python source code and extract agent information.

        Args:
            source_code: The Python source code as a string.

        Returns:
            ParsedAgent containing all extracted information.

        Raises:
            SyntaxError: If the source contains invalid Python syntax.
        """
        self._source_lines = source_code.splitlines()
        self._node_function_names = set()

        tree = ast.parse(source_code)

        # Detect agent type and collect metadata
        agent_type: Literal["langgraph", "native"] = "native"
        compile_lineno: int | None = None
        graph_variable: str | None = None
        builder_variable: str | None = None
        imports_end_lineno = 0
        has_existing_arbiteros = False

        # First pass: detect LangGraph patterns and find node functions
        for node in ast.walk(tree):
            # Check for LangGraph imports
            if isinstance(node, ast.ImportFrom):
                if node.module and "langgraph" in node.module:
                    agent_type = "langgraph"
                if node.module and "arbiteros_alpha" in node.module:
                    has_existing_arbiteros = True

            # Check for StateGraph instantiation
            if isinstance(node, ast.Assign):
                if isinstance(node.value, ast.Call):
                    if self._is_state_graph_call(node.value):
                        agent_type = "langgraph"
                        if node.targets and isinstance(node.targets[0], ast.Name):
                            builder_variable = node.targets[0].id

            # Check for add_node calls to find node function names
            if isinstance(node, ast.Call):
                if self._is_add_node_call(node):
                    func_name = self._extract_node_function_name(node)
                    if func_name:
                        self._node_function_names.add(func_name)

                # Check for compile() call
                if self._is_compile_call(node):
                    compile_lineno = node.lineno

        # Find compile assignment (graph = builder.compile())
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                if isinstance(node.value, ast.Call) and self._is_compile_call(
                    node.value
                ):
                    if node.targets and isinstance(node.targets[0], ast.Name):
                        graph_variable = node.targets[0].id
                        compile_lineno = node.lineno

        # Find where imports end
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                imports_end_lineno = max(
                    imports_end_lineno, node.end_lineno or node.lineno
                )

        # Second pass: extract all function definitions
        functions: list[ParsedFunction] = []
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.FunctionDef):
                # Skip if already has @os.instruction decorator
                if self._has_instruction_decorator(node):
                    continue

                func = self._extract_function(node, source_code)
                func.is_node_function = func.name in self._node_function_names

                # For native agents, consider all functions with state param as node functions
                if agent_type == "native" and func.has_state_param:
                    func.is_node_function = True

                functions.append(func)

        return ParsedAgent(
            agent_type=agent_type,
            functions=functions,
            compile_lineno=compile_lineno,
            graph_variable=graph_variable,
            builder_variable=builder_variable,
            source_lines=self._source_lines,
            imports_end_lineno=imports_end_lineno,
            has_existing_arbiteros=has_existing_arbiteros,
        )

    def _is_state_graph_call(self, node: ast.Call) -> bool:
        """Check if a Call node is a StateGraph() instantiation."""
        if isinstance(node.func, ast.Name):
            return node.func.id == "StateGraph"
        if isinstance(node.func, ast.Attribute):
            return node.func.attr == "StateGraph"
        return False

    def _is_add_node_call(self, node: ast.Call) -> bool:
        """Check if a Call node is an add_node() call."""
        if isinstance(node.func, ast.Attribute):
            return node.func.attr == "add_node"
        return False

    def _is_compile_call(self, node: ast.Call) -> bool:
        """Check if a Call node is a compile() call."""
        if isinstance(node.func, ast.Attribute):
            return node.func.attr == "compile"
        return False

    def _extract_node_function_name(self, node: ast.Call) -> str | None:
        """Extract the function name from an add_node() call.

        Handles both:
        - builder.add_node("name", func)
        - builder.add_node(func)  # Uses func.__name__
        """
        if len(node.args) >= 1:
            # First arg could be string name or function reference
            first_arg = node.args[0]
            if isinstance(first_arg, ast.Name):
                return first_arg.id
            if isinstance(first_arg, ast.Constant) and isinstance(first_arg.value, str):
                # Name is a string, function should be second arg
                if len(node.args) >= 2:
                    second_arg = node.args[1]
                    if isinstance(second_arg, ast.Name):
                        return second_arg.id
        return None

    def _has_instruction_decorator(self, node: ast.FunctionDef) -> bool:
        """Check if function already has @os.instruction() decorator."""
        for decorator in node.decorator_list:
            if isinstance(decorator, ast.Call):
                if isinstance(decorator.func, ast.Attribute):
                    if decorator.func.attr == "instruction":
                        return True
        return False

    def _extract_function(
        self, node: ast.FunctionDef, source_code: str
    ) -> ParsedFunction:
        """Extract function information from an AST FunctionDef node."""
        # Get docstring
        docstring = ast.get_docstring(node)

        # Get source code for the function
        start_line = node.lineno - 1
        end_line = node.end_lineno or node.lineno
        func_lines = self._source_lines[start_line:end_line]
        source = "\n".join(func_lines)

        # Check if function has a state parameter
        has_state_param = False
        for arg in node.args.args:
            if arg.arg in ("state", "State"):
                has_state_param = True
                break

        return ParsedFunction(
            name=node.name,
            docstring=docstring,
            source_code=source,
            lineno=node.lineno,
            end_lineno=end_line,
            has_state_param=has_state_param,
        )
