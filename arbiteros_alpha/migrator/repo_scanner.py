"""Repository scanner for migration pipeline.

This module provides tools to scan a repository and extract structural information
about Python files, functions, imports, graph definitions, and state classes.
The output is structured JSON that can be passed to the LLM analyzer.
"""

import ast
import fnmatch
import logging
from pathlib import Path

from .schemas import (
    AgentFramework,
    FileInfo,
    FunctionInfo,
    GraphEdgeInfo,
    GraphNodeInfo,
    ImportInfo,
    RepoScanResult,
)

logger = logging.getLogger(__name__)


class RepoScanner:
    """Scans a repository and extracts structural information.

    This scanner walks through a repository directory and extracts:
    - Python file information
    - Function definitions with source code
    - Import statements
    - Graph node and edge definitions (for LangGraph)
    - State class definitions

    Example:
        >>> scanner = RepoScanner()
        >>> result = scanner.scan("/path/to/repo")
        >>> print(result.detected_framework)
        >>> for func in result.functions:
        ...     print(f"{func.name}: {func.file_path}")
    """

    def __init__(
        self,
        ignore_patterns: list[str] | None = None,
        include_patterns: list[str] | None = None,
    ) -> None:
        """Initialize the RepoScanner.

        Args:
            ignore_patterns: Glob patterns for files/directories to ignore.
                Defaults to common patterns like __pycache__, .git, etc.
            include_patterns: Glob patterns for files to include.
                Defaults to ["*.py"].
        """
        self.ignore_patterns = ignore_patterns or [
            "__pycache__",
            ".git",
            ".venv",
            "venv",
            "node_modules",
            "*.pyc",
            "*.pyo",
            ".pytest_cache",
            ".mypy_cache",
            "*.egg-info",
            "build",
            "dist",
        ]
        self.include_patterns = include_patterns or ["*.py"]

    def scan(self, repo_path: str | Path) -> RepoScanResult:
        """Scan a repository and extract structural information.

        Args:
            repo_path: Path to the repository root directory.

        Returns:
            RepoScanResult containing all extracted information.

        Raises:
            FileNotFoundError: If the repository path doesn't exist.
            NotADirectoryError: If the path is not a directory.
        """
        repo_path = Path(repo_path).resolve()

        if not repo_path.exists():
            raise FileNotFoundError(f"Repository path not found: {repo_path}")
        if not repo_path.is_dir():
            raise NotADirectoryError(f"Path is not a directory: {repo_path}")

        logger.info(f"Scanning repository: {repo_path}")

        python_files: list[FileInfo] = []
        all_functions: list[FunctionInfo] = []
        all_imports: dict[str, list[ImportInfo]] = {}
        graph_nodes: list[GraphNodeInfo] = []
        graph_edges: list[GraphEdgeInfo] = []
        state_classes: list[str] = []
        detected_framework = AgentFramework.UNKNOWN

        # Walk through the repository
        for file_path in self._walk_files(repo_path):
            rel_path = str(file_path.relative_to(repo_path))
            logger.debug(f"Processing file: {rel_path}")

            try:
                content = file_path.read_text(encoding="utf-8")
                lines = content.splitlines()

                # Add file info
                python_files.append(
                    FileInfo(
                        path=rel_path,
                        size_bytes=file_path.stat().st_size,
                        line_count=len(lines),
                    )
                )

                # Parse AST
                try:
                    tree = ast.parse(content)
                except SyntaxError as e:
                    logger.warning(f"Syntax error in {rel_path}: {e}")
                    continue

                # Extract information from AST
                file_imports = self._extract_imports(tree)
                all_imports[rel_path] = file_imports

                # Check for framework indicators in imports
                framework = self._detect_framework_from_imports(file_imports)
                if framework != AgentFramework.UNKNOWN:
                    detected_framework = framework

                # Extract functions
                functions = self._extract_functions(tree, rel_path, lines)
                all_functions.extend(functions)

                # Extract graph nodes and edges
                nodes, edges = self._extract_graph_structure(tree, rel_path)
                graph_nodes.extend(nodes)
                graph_edges.extend(edges)

                # Extract state classes
                states = self._extract_state_classes(tree)
                state_classes.extend(states)

            except Exception as e:
                logger.error(f"Error processing {rel_path}: {e}")
                continue

        # Remove duplicates from state classes
        state_classes = list(set(state_classes))

        logger.info(
            f"Scan complete: {len(python_files)} files, "
            f"{len(all_functions)} functions, {len(graph_nodes)} graph nodes"
        )

        return RepoScanResult(
            repo_path=str(repo_path),
            python_files=python_files,
            functions=all_functions,
            imports=all_imports,
            graph_nodes=graph_nodes,
            graph_edges=graph_edges,
            state_classes=state_classes,
            detected_framework=detected_framework,
        )

    def _walk_files(self, repo_path: Path) -> list[Path]:
        """Walk through repository and yield Python files.

        Args:
            repo_path: Root directory to walk.

        Yields:
            Path objects for each Python file found.
        """
        files = []
        for path in repo_path.rglob("*"):
            if path.is_file():
                # Check if should be ignored
                if self._should_ignore(path, repo_path):
                    continue
                # Check if should be included
                if self._should_include(path):
                    files.append(path)
        return files

    def _should_ignore(self, path: Path, repo_root: Path) -> bool:
        """Check if a path should be ignored.

        Args:
            path: Path to check.
            repo_root: Repository root for relative path calculation.

        Returns:
            True if the path matches any ignore pattern.
        """
        rel_path = path.relative_to(repo_root)

        # Check each part of the path
        for part in rel_path.parts:
            for pattern in self.ignore_patterns:
                if fnmatch.fnmatch(part, pattern):
                    return True

        # Check full path
        for pattern in self.ignore_patterns:
            if fnmatch.fnmatch(str(rel_path), pattern):
                return True
            if fnmatch.fnmatch(path.name, pattern):
                return True

        return False

    def _should_include(self, path: Path) -> bool:
        """Check if a path should be included.

        Args:
            path: Path to check.

        Returns:
            True if the path matches any include pattern.
        """
        for pattern in self.include_patterns:
            if fnmatch.fnmatch(path.name, pattern):
                return True
        return False

    def _extract_imports(self, tree: ast.AST) -> list[ImportInfo]:
        """Extract import statements from AST.

        Args:
            tree: Parsed AST of a Python file.

        Returns:
            List of ImportInfo objects.
        """
        imports = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(
                        ImportInfo(
                            module=alias.name,
                            names=[alias.asname or alias.name],
                            is_from_import=False,
                        )
                    )
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                names = [alias.name for alias in node.names]
                imports.append(
                    ImportInfo(
                        module=module,
                        names=names,
                        is_from_import=True,
                    )
                )

        return imports

    def _detect_framework_from_imports(
        self, imports: list[ImportInfo]
    ) -> AgentFramework:
        """Detect agent framework from import statements.

        Args:
            imports: List of imports from a file.

        Returns:
            Detected AgentFramework or UNKNOWN.
        """
        for imp in imports:
            module = imp.module.lower()
            if "langgraph" in module:
                return AgentFramework.LANGGRAPH
            if "crewai" in module:
                return AgentFramework.CREWAI
            if "autogen" in module:
                return AgentFramework.AUTOGEN

        return AgentFramework.UNKNOWN

    def _extract_functions(
        self, tree: ast.AST, file_path: str, source_lines: list[str]
    ) -> list[FunctionInfo]:
        """Extract function definitions from AST.

        Args:
            tree: Parsed AST of a Python file.
            file_path: Relative path to the file.
            source_lines: Source code split into lines.

        Returns:
            List of FunctionInfo objects.
        """
        functions = []

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                # Skip nested functions (only top-level and class methods)
                # We check if this function is a direct child of module or class
                is_top_level = False
                for parent in ast.walk(tree):
                    if hasattr(parent, "body") and node in parent.body:
                        if isinstance(parent, (ast.Module, ast.ClassDef)):
                            is_top_level = True
                            break

                if not is_top_level:
                    # Skip nested functions - they are handled via their parent factory
                    continue

                # Get source code
                start_line = node.lineno - 1
                end_line = node.end_lineno or node.lineno
                source_code = "\n".join(source_lines[start_line:end_line])

                # Get parameters
                parameters = [arg.arg for arg in node.args.args]

                # Get decorators
                decorators = []
                for decorator in node.decorator_list:
                    if isinstance(decorator, ast.Name):
                        decorators.append(decorator.id)
                    elif isinstance(decorator, ast.Call):
                        if isinstance(decorator.func, ast.Name):
                            decorators.append(decorator.func.id)
                        elif isinstance(decorator.func, ast.Attribute):
                            decorators.append(decorator.func.attr)
                    elif isinstance(decorator, ast.Attribute):
                        decorators.append(decorator.attr)

                # Check if this is a factory function
                is_factory = self._is_factory_function(node)

                functions.append(
                    FunctionInfo(
                        name=node.name,
                        file_path=file_path,
                        lineno=node.lineno,
                        end_lineno=end_line,
                        docstring=ast.get_docstring(node),
                        source_code=source_code,
                        is_async=isinstance(node, ast.AsyncFunctionDef),
                        is_factory=is_factory,
                        parameters=parameters,
                        decorators=decorators,
                    )
                )

        return functions

    def _is_factory_function(
        self, node: ast.FunctionDef | ast.AsyncFunctionDef
    ) -> bool:
        """Check if a function is a factory that returns another function.

        Args:
            node: Function AST node.

        Returns:
            True if the function contains a nested function and returns it.
        """
        nested_funcs = []
        return_names = []

        for child in ast.walk(node):
            # Find nested function definitions
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if child is not node:  # Not the function itself
                    nested_funcs.append(child.name)

            # Find return statements with names or functools.partial calls
            if isinstance(child, ast.Return):
                if isinstance(child.value, ast.Name):
                    # Direct return: return func_name
                    return_names.append(child.value.id)
                elif isinstance(child.value, ast.Call):
                    # Check for functools.partial(func_name, ...)
                    call = child.value
                    func = call.func
                    # Check for functools.partial or partial
                    is_partial = False
                    if isinstance(func, ast.Attribute):
                        if func.attr == "partial":
                            is_partial = True
                    elif isinstance(func, ast.Name):
                        if func.id == "partial":
                            is_partial = True

                    if is_partial and call.args:
                        # First argument to partial is the function
                        first_arg = call.args[0]
                        if isinstance(first_arg, ast.Name):
                            return_names.append(first_arg.id)

        # Check if any nested function is returned
        for name in return_names:
            if name in nested_funcs:
                return True

        return False

    def _extract_graph_structure(
        self, tree: ast.AST, file_path: str
    ) -> tuple[list[GraphNodeInfo], list[GraphEdgeInfo]]:
        """Extract graph node and edge definitions from AST.

        Args:
            tree: Parsed AST of a Python file.
            file_path: Relative path to the file.

        Returns:
            Tuple of (nodes, edges) lists.
        """
        nodes = []
        edges = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                # Check for add_node calls
                if isinstance(node.func, ast.Attribute):
                    if node.func.attr == "add_node":
                        node_info = self._parse_add_node(node, file_path)
                        if node_info:
                            nodes.append(node_info)

                    # Check for add_edge calls
                    elif node.func.attr == "add_edge":
                        edge_info = self._parse_add_edge(node)
                        if edge_info:
                            edges.append(edge_info)

                    # Check for add_conditional_edges calls
                    elif node.func.attr == "add_conditional_edges":
                        cond_edges = self._parse_conditional_edges(node)
                        edges.extend(cond_edges)

        return nodes, edges

    def _parse_add_node(self, node: ast.Call, file_path: str) -> GraphNodeInfo | None:
        """Parse an add_node call to extract node information.

        Args:
            node: AST Call node for add_node.
            file_path: File path containing the call.

        Returns:
            GraphNodeInfo or None if parsing fails.
        """
        if len(node.args) < 1:
            return None

        # First argument is the node name
        first_arg = node.args[0]
        node_name = None
        function_name = None

        if isinstance(first_arg, ast.Constant):
            node_name = str(first_arg.value)
        elif isinstance(first_arg, ast.Name):
            # Name is the function, node name might be its __name__
            function_name = first_arg.id
            node_name = first_arg.id

        # Second argument (if present) is the function
        if len(node.args) >= 2:
            second_arg = node.args[1]
            if isinstance(second_arg, ast.Name):
                function_name = second_arg.id
            elif isinstance(second_arg, ast.Call):
                # Could be a factory call like create_analyst(llm)
                if isinstance(second_arg.func, ast.Name):
                    function_name = second_arg.func.id

        if node_name and function_name:
            return GraphNodeInfo(
                node_name=node_name,
                function_name=function_name,
                file_path=file_path,
                lineno=node.lineno,
            )

        return None

    def _parse_add_edge(self, node: ast.Call) -> GraphEdgeInfo | None:
        """Parse an add_edge call to extract edge information.

        Args:
            node: AST Call node for add_edge.

        Returns:
            GraphEdgeInfo or None if parsing fails.
        """
        if len(node.args) < 2:
            return None

        source = self._get_node_name(node.args[0])
        target = self._get_node_name(node.args[1])

        if source and target:
            return GraphEdgeInfo(
                source=source,
                target=target,
                is_conditional=False,
            )

        return None

    def _parse_conditional_edges(self, node: ast.Call) -> list[GraphEdgeInfo]:
        """Parse an add_conditional_edges call to extract edge information.

        Args:
            node: AST Call node for add_conditional_edges.

        Returns:
            List of GraphEdgeInfo objects.
        """
        edges = []

        if len(node.args) < 2:
            return edges

        source = self._get_node_name(node.args[0])
        condition_func = None

        # Second arg is the condition function
        if isinstance(node.args[1], ast.Name):
            condition_func = node.args[1].id
        elif isinstance(node.args[1], ast.Attribute):
            condition_func = node.args[1].attr

        # Third arg can be a dict or list of targets
        if len(node.args) >= 3:
            targets_arg = node.args[2]

            if isinstance(targets_arg, ast.Dict):
                # Dict mapping condition results to target nodes
                for value in targets_arg.values:
                    target = self._get_node_name(value)
                    if source and target:
                        edges.append(
                            GraphEdgeInfo(
                                source=source,
                                target=target,
                                is_conditional=True,
                                condition_function=condition_func,
                            )
                        )
            elif isinstance(targets_arg, ast.List):
                # List of target nodes
                for elt in targets_arg.elts:
                    target = self._get_node_name(elt)
                    if source and target:
                        edges.append(
                            GraphEdgeInfo(
                                source=source,
                                target=target,
                                is_conditional=True,
                                condition_function=condition_func,
                            )
                        )

        return edges

    def _get_node_name(self, node: ast.AST) -> str | None:
        """Extract a string node name from an AST node.

        Args:
            node: AST node that might represent a node name.

        Returns:
            String name or None.
        """
        if isinstance(node, ast.Constant):
            return str(node.value)
        elif isinstance(node, ast.Name):
            # Could be START, END, or a variable
            return node.id
        return None

    def _extract_state_classes(self, tree: ast.AST) -> list[str]:
        """Extract state class definitions from AST.

        Args:
            tree: Parsed AST of a Python file.

        Returns:
            List of state class names.
        """
        state_classes = []

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                # Check if class inherits from TypedDict, BaseModel, or has "State" in name
                class_name = node.name

                # Check bases
                is_state_class = False
                for base in node.bases:
                    base_name = None
                    if isinstance(base, ast.Name):
                        base_name = base.id
                    elif isinstance(base, ast.Attribute):
                        base_name = base.attr

                    if base_name in ("TypedDict", "BaseModel", "State"):
                        is_state_class = True
                        break

                # Check class name
                if "State" in class_name or "state" in class_name.lower():
                    is_state_class = True

                if is_state_class:
                    state_classes.append(class_name)

        return state_classes


def scan_repository(
    repo_path: str | Path,
    ignore_patterns: list[str] | None = None,
    include_patterns: list[str] | None = None,
) -> RepoScanResult:
    """Convenience function to scan a repository.

    Args:
        repo_path: Path to the repository.
        ignore_patterns: Patterns to ignore.
        include_patterns: Patterns to include.

    Returns:
        RepoScanResult with all extracted information.
    """
    scanner = RepoScanner(
        ignore_patterns=ignore_patterns,
        include_patterns=include_patterns,
    )
    return scanner.scan(repo_path)
