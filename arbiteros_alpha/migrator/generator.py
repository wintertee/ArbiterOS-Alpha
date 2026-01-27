"""Code generator for migrating agents to ArbiterOS-governed agents.

This module handles the actual code migration, including:
- Adding imports
- Adding OS initialization
- Adding @os.instruction() decorators
- Adding register_compiled_graph() for LangGraph agents
- Creating backups before modification

Enhanced features:
- Multi-file generation using Jinja templates
- Generates governed_agents.py, policy_checkers.py, llm_schemas.py, policies.yaml
- Syntax validation of generated code
"""

import ast
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

from .classifier import NodeClassification
from .parser import AgentParser, ParsedAgent
from .schemas import (
    GeneratedFile,
    GovernanceWrapperSpec,
    InstructionTypeEnum,
    LLMSchemaDesignOutput,
    NodeClassificationBatch,
    NodeClassificationResult,
    PolicyDesignOutput,
    RepoAnalysisOutput,
    TransformationResult,
)


@dataclass
class MigrationResult:
    """Result of a code migration.

    Attributes:
        success: Whether the migration succeeded.
        modified_file: Path to the modified file.
        backup_file: Path to the backup file.
        changes: List of changes made.
        error: Error message if migration failed.
    """

    success: bool
    modified_file: str = ""
    backup_file: str = ""
    changes: list[str] = field(default_factory=list)
    error: str = ""


# Mapping from InstructionTypeEnum to import path
INSTRUCTION_IMPORT_MAP: dict[InstructionTypeEnum, tuple[str, str]] = {
    # CognitiveCore
    InstructionTypeEnum.GENERATE: ("CognitiveCore", "CognitiveCore.GENERATE"),
    InstructionTypeEnum.DECOMPOSE: ("CognitiveCore", "CognitiveCore.DECOMPOSE"),
    InstructionTypeEnum.REFLECT: ("CognitiveCore", "CognitiveCore.REFLECT"),
    # MemoryCore
    InstructionTypeEnum.LOAD: ("MemoryCore", "MemoryCore.LOAD"),
    InstructionTypeEnum.STORE: ("MemoryCore", "MemoryCore.STORE"),
    InstructionTypeEnum.COMPRESS: ("MemoryCore", "MemoryCore.COMPRESS"),
    InstructionTypeEnum.FILTER: ("MemoryCore", "MemoryCore.FILTER"),
    InstructionTypeEnum.STRUCTURE: ("MemoryCore", "MemoryCore.STRUCTURE"),
    InstructionTypeEnum.RENDER: ("MemoryCore", "MemoryCore.RENDER"),
    # ExecutionCore
    InstructionTypeEnum.TOOL_CALL: ("ExecutionCore", "ExecutionCore.TOOL_CALL"),
    InstructionTypeEnum.TOOL_BUILD: ("ExecutionCore", "ExecutionCore.TOOL_BUILD"),
    InstructionTypeEnum.DELEGATE: ("ExecutionCore", "ExecutionCore.DELEGATE"),
    InstructionTypeEnum.RESPOND: ("ExecutionCore", "ExecutionCore.RESPOND"),
    # NormativeCore
    InstructionTypeEnum.VERIFY: ("NormativeCore", "NormativeCore.VERIFY"),
    InstructionTypeEnum.CONSTRAIN: ("NormativeCore", "NormativeCore.CONSTRAIN"),
    InstructionTypeEnum.FALLBACK: ("NormativeCore", "NormativeCore.FALLBACK"),
    InstructionTypeEnum.INTERRUPT: ("NormativeCore", "NormativeCore.INTERRUPT"),
    # MetacognitiveCore
    InstructionTypeEnum.PREDICT_SUCCESS: (
        "MetacognitiveCore",
        "MetacognitiveCore.PREDICT_SUCCESS",
    ),
    InstructionTypeEnum.EVALUATE_PROGRESS: (
        "MetacognitiveCore",
        "MetacognitiveCore.EVALUATE_PROGRESS",
    ),
    InstructionTypeEnum.MONITOR_RESOURCES: (
        "MetacognitiveCore",
        "MetacognitiveCore.MONITOR_RESOURCES",
    ),
    # AdaptiveCore
    InstructionTypeEnum.UPDATE_KNOWLEDGE: (
        "AdaptiveCore",
        "AdaptiveCore.UPDATE_KNOWLEDGE",
    ),
    InstructionTypeEnum.REFINE_SKILL: ("AdaptiveCore", "AdaptiveCore.REFINE_SKILL"),
    InstructionTypeEnum.LEARN_PREFERENCE: (
        "AdaptiveCore",
        "AdaptiveCore.LEARN_PREFERENCE",
    ),
    InstructionTypeEnum.FORMULATE_EXPERIMENT: (
        "AdaptiveCore",
        "AdaptiveCore.FORMULATE_EXPERIMENT",
    ),
    # SocialCore
    InstructionTypeEnum.COMMUNICATE: ("SocialCore", "SocialCore.COMMUNICATE"),
    InstructionTypeEnum.NEGOTIATE: ("SocialCore", "SocialCore.NEGOTIATE"),
    InstructionTypeEnum.PROPOSE_VOTE: ("SocialCore", "SocialCore.PROPOSE_VOTE"),
    InstructionTypeEnum.FORM_COALITION: ("SocialCore", "SocialCore.FORM_COALITION"),
    # AffectiveCore
    InstructionTypeEnum.INFER_INTENT: ("AffectiveCore", "AffectiveCore.INFER_INTENT"),
    InstructionTypeEnum.MODEL_USER_STATE: (
        "AffectiveCore",
        "AffectiveCore.MODEL_USER_STATE",
    ),
    InstructionTypeEnum.ADAPT_RESPONSE: (
        "AffectiveCore",
        "AffectiveCore.ADAPT_RESPONSE",
    ),
    InstructionTypeEnum.MANAGE_TRUST: ("AffectiveCore", "AffectiveCore.MANAGE_TRUST"),
}


class CodeGenerator:
    """Generates transformed code with ArbiterOS governance.

    This class takes parsed agent information and classification results
    to produce transformed Python code with ArbiterOS decorators and setup.

    Enhanced with:
    - Multi-file generation using Jinja templates
    - Generates full governance modules
    - Syntax validation of generated code
    - Verification node generation for high-risk operations
    - Safety-first transformation with mandatory verification steps

    Example:
        >>> generator = CodeGenerator()
        >>> result = generator.transform(
        ...     file_path="agent.py",
        ...     parsed_agent=parsed_agent,
        ...     classifications=classifications,
        ... )
        >>> if result.success:
        ...     print(f"Backup: {result.backup_file}")
    """

    # Import lines to add
    ARBITEROS_IMPORTS = [
        "from arbiteros_alpha import ArbiterOSAlpha",
        "import arbiteros_alpha.instructions as Instr",
    ]

    # High-risk instruction types that require verification
    HIGH_RISK_INSTRUCTIONS = {
        InstructionTypeEnum.TOOL_CALL,
        InstructionTypeEnum.TOOL_BUILD,
        InstructionTypeEnum.RESPOND,
        InstructionTypeEnum.DELEGATE,
    }

    def __init__(self) -> None:
        """Initialize the CodeGenerator."""
        # Set up Jinja environment
        templates_dir = Path(__file__).parent / "templates"
        self._jinja_env = Environment(
            loader=FileSystemLoader(str(templates_dir)),
            autoescape=select_autoescape(default=False),
            trim_blocks=True,
            lstrip_blocks=True,
        )

    def transform(
        self,
        file_path: str | Path,
        parsed_agent: ParsedAgent,
        classifications: dict[str, NodeClassification],
        dry_run: bool = False,
    ) -> MigrationResult:
        """Migrate a file to add ArbiterOS governance.

        Args:
            file_path: Path to the source file.
            parsed_agent: Parsed agent information.
            classifications: Dict mapping function name to classification.
            dry_run: If True, don't modify files, just return what would change.

        Returns:
            MigrationResult with details of the migration.
        """
        file_path = Path(file_path)
        changes: list[str] = []

        try:
            # Read source
            source_lines = parsed_agent.source_lines.copy()

            # Track line offset as we insert lines
            offset = 0

            # 1. Add imports if not already present
            if not parsed_agent.has_existing_arbiteros:
                import_line = parsed_agent.imports_end_lineno
                for i, import_stmt in enumerate(self.ARBITEROS_IMPORTS):
                    source_lines.insert(import_line + offset + i, import_stmt)
                    changes.append(f"Added import: {import_stmt}")
                offset += len(self.ARBITEROS_IMPORTS)

                # Add blank line after imports
                source_lines.insert(import_line + offset, "")
                offset += 1

            # 2. Add OS initialization (only if not already present)
            if not parsed_agent.has_os_initialization:
                # Find the first function definition line
                first_func_line = None
                for func in parsed_agent.functions:
                    if first_func_line is None or func.lineno < first_func_line:
                        first_func_line = func.lineno

                if first_func_line is not None:
                    # Insert OS init before first function (with offset adjustment)
                    backend = parsed_agent.agent_type
                    os_init_lines = [
                        "",
                        f'os = ArbiterOSAlpha(backend="{backend}")',
                        "",
                    ]
                    insert_pos = first_func_line - 1 + offset
                    for i, line in enumerate(os_init_lines):
                        source_lines.insert(insert_pos + i, line)
                        if line.strip():
                            changes.append(
                                f"Added OS initialization at line {insert_pos + i + 1}"
                            )
                    offset += len(os_init_lines)

            # 3. Add decorators to functions
            # We need to process functions in reverse order (by line number)
            # so that insertions don't affect positions of functions we haven't processed yet
            funcs_with_class = [
                (func, classifications.get(func.name))
                for func in parsed_agent.functions
                if func.name in classifications
            ]
            funcs_with_class.sort(key=lambda x: x[0].lineno, reverse=True)

            # Track how many decorators we've added (for final offset calculation)
            decorators_added = 0
            for func, classification in funcs_with_class:
                if classification is None:
                    continue

                # Calculate insertion point (before function definition)
                # Use base offset (from imports + os init), not accumulated decorator offset
                # because we're processing in reverse order
                insert_line = func.lineno - 1 + offset

                # Build decorator
                instr_type = classification.instruction_type
                decorator = f"@os.instruction(Instr.{instr_type})"

                source_lines.insert(insert_line, decorator)
                changes.append(
                    f"Added @os.instruction(Instr.{instr_type}) to {func.name}"
                )
                decorators_added += 1

            # Update total offset with all decorators added
            offset += decorators_added

            # 4. Add register_compiled_graph() for LangGraph agents
            # Only add if not already present
            if (
                parsed_agent.agent_type == "langgraph"
                and parsed_agent.compile_lineno
                and not parsed_agent.has_register_compiled_graph
            ):
                compile_adjusted = parsed_agent.compile_lineno + offset - 1

                if parsed_agent.compile_in_return:
                    # Case: return workflow.compile()
                    # Need to: 1) change to assignment, 2) register, 3) return
                    graph_var = parsed_agent.graph_variable or "compiled_graph"

                    # Get the actual compile line
                    compile_line = source_lines[compile_adjusted].strip()

                    # Extract the compile call
                    import re

                    match = re.search(r"return\s+(.+\.compile\([^)]*\))", compile_line)
                    if match:
                        compile_call = match.group(1)

                        # Replace the return line with assignment
                        indent = len(source_lines[compile_adjusted]) - len(
                            source_lines[compile_adjusted].lstrip()
                        )
                        source_lines[compile_adjusted] = (
                            f"{' ' * indent}{graph_var} = {compile_call}"
                        )

                        # Add registration and return using dynamic get_arbiter_os()
                        source_lines.insert(compile_adjusted + 1, f"{' ' * indent}")
                        source_lines.insert(
                            compile_adjusted + 2,
                            f"{' ' * indent}# Register with ArbiterOS for governance tracking",
                        )
                        source_lines.insert(
                            compile_adjusted + 3,
                            f"{' ' * indent}# Use get_arbiter_os() dynamically to get the current instance",
                        )
                        source_lines.insert(
                            compile_adjusted + 4,
                            f"{' ' * indent}get_arbiter_os().register_compiled_graph({graph_var})",
                        )
                        source_lines.insert(compile_adjusted + 5, f"{' ' * indent}")
                        source_lines.insert(
                            compile_adjusted + 6, f"{' ' * indent}return {graph_var}"
                        )

                        changes.append(
                            "Modified return compile() to register with ArbiterOS"
                        )
                        offset += 6
                else:
                    # Case: graph = builder.compile()
                    # Just add register_compiled_graph after this line using dynamic get_arbiter_os()
                    graph_var = parsed_agent.graph_variable or "graph"
                    register_line = (
                        f"get_arbiter_os().register_compiled_graph({graph_var})"
                    )

                    # Insert after compile line
                    source_lines.insert(compile_adjusted + 1, register_line)
                    changes.append(
                        f"Added get_arbiter_os().register_compiled_graph({graph_var})"
                    )
                    offset += 1

            # Generate final source
            new_source = "\n".join(source_lines)

            if dry_run:
                return MigrationResult(
                    success=True,
                    modified_file=str(file_path),
                    backup_file="",
                    changes=changes,
                )

            # Create backup
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = file_path.with_name(
                f"{file_path.stem}_backup_{timestamp}{file_path.suffix}"
            )
            shutil.copy2(file_path, backup_path)

            # Write transformed file
            file_path.write_text(new_source, encoding="utf-8")

            return MigrationResult(
                success=True,
                modified_file=str(file_path),
                backup_file=str(backup_path),
                changes=changes,
            )

        except Exception as e:
            return MigrationResult(
                success=False,
                error=str(e),
                changes=changes,
            )

    def generate_transformed_source(
        self,
        parsed_agent: ParsedAgent,
        classifications: dict[str, NodeClassification],
    ) -> str:
        """Generate transformed source code without writing to file.

        Useful for previewing changes or testing.

        Args:
            parsed_agent: Parsed agent information.
            classifications: Dict mapping function name to classification.

        Returns:
            The transformed source code as a string.
        """
        source_lines = parsed_agent.source_lines.copy()
        offset = 0

        # Add imports
        if not parsed_agent.has_existing_arbiteros:
            import_line = parsed_agent.imports_end_lineno
            for i, import_stmt in enumerate(self.ARBITEROS_IMPORTS):
                source_lines.insert(import_line + offset + i, import_stmt)
            offset += len(self.ARBITEROS_IMPORTS)
            source_lines.insert(import_line + offset, "")
            offset += 1

        # Add OS initialization (only if not already present)
        if not parsed_agent.has_os_initialization:
            first_func_line = min(
                (f.lineno for f in parsed_agent.functions), default=None
            )
            if first_func_line is not None:
                backend = parsed_agent.agent_type
                os_init_lines = ["", f'os = ArbiterOSAlpha(backend="{backend}")', ""]
                insert_pos = first_func_line - 1 + offset
                for i, line in enumerate(os_init_lines):
                    source_lines.insert(insert_pos + i, line)
                offset += len(os_init_lines)

        # Add decorators (reverse order)
        funcs_with_class = [
            (func, classifications.get(func.name))
            for func in parsed_agent.functions
            if func.name in classifications
        ]
        funcs_with_class.sort(key=lambda x: x[0].lineno, reverse=True)

        decorators_added = 0
        for func, classification in funcs_with_class:
            if classification is None:
                continue
            insert_line = func.lineno - 1 + offset
            decorator = f"@os.instruction(Instr.{classification.instruction_type})"
            source_lines.insert(insert_line, decorator)
            decorators_added += 1

        offset += decorators_added

        # Add register_compiled_graph for LangGraph (only if not already present)
        if (
            parsed_agent.agent_type == "langgraph"
            and parsed_agent.compile_lineno
            and not parsed_agent.has_register_compiled_graph
        ):
            compile_adjusted = parsed_agent.compile_lineno + offset - 1

            if parsed_agent.compile_in_return:
                # Case: return workflow.compile()
                # Need to: 1) change to assignment, 2) register, 3) return
                graph_var = parsed_agent.graph_variable or "compiled_graph"

                # Get the actual compile line
                compile_line = source_lines[compile_adjusted].strip()

                # Extract the builder variable and method chain
                # Pattern: return <builder>.compile(...)
                import re

                match = re.search(r"return\s+(.+\.compile\([^)]*\))", compile_line)
                if match:
                    compile_call = match.group(1)

                    # Replace the return line with assignment
                    indent = len(source_lines[compile_adjusted]) - len(
                        source_lines[compile_adjusted].lstrip()
                    )
                    source_lines[compile_adjusted] = (
                        f"{' ' * indent}{graph_var} = {compile_call}"
                    )

                    # Add registration and return using dynamic get_arbiter_os()
                    source_lines.insert(compile_adjusted + 1, f"{' ' * indent}")
                    source_lines.insert(
                        compile_adjusted + 2,
                        f"{' ' * indent}# Register with ArbiterOS for governance tracking",
                    )
                    source_lines.insert(
                        compile_adjusted + 3,
                        f"{' ' * indent}# Use get_arbiter_os() dynamically to get the current instance",
                    )
                    source_lines.insert(
                        compile_adjusted + 4,
                        f"{' ' * indent}get_arbiter_os().register_compiled_graph({graph_var})",
                    )
                    source_lines.insert(compile_adjusted + 5, f"{' ' * indent}")
                    source_lines.insert(
                        compile_adjusted + 6, f"{' ' * indent}return {graph_var}"
                    )
                    offset += 6
            else:
                # Case: graph = builder.compile()
                # Just add register_compiled_graph after this line using dynamic get_arbiter_os()
                graph_var = parsed_agent.graph_variable or "graph"
                source_lines.insert(
                    compile_adjusted + 1,
                    f"get_arbiter_os().register_compiled_graph({graph_var})",
                )
                offset += 1

        return "\n".join(source_lines)

    # =========================================================================
    # Multi-File Generation (New)
    # =========================================================================

    def transform_repository(
        self,
        source_repo: str | Path,
        output_repo: str | Path,
        analysis: RepoAnalysisOutput,
        classifications: NodeClassificationBatch,
        policy_design: PolicyDesignOutput,
        schema_design: LLMSchemaDesignOutput,
        dry_run: bool = False,
    ) -> TransformationResult:
        """Transform an entire repository to use ArbiterOS governance.

        This method:
        1. Copies the source repository to the output location
        2. Modifies agent files in place with decorators
        3. Generates governance files (governed_agents.py, policy_checkers.py, etc.)

        Args:
            source_repo: Path to the source repository.
            output_repo: Path where the transformed repository will be created.
            analysis: Repository analysis output.
            classifications: Node classification results.
            policy_design: Policy design output.
            schema_design: Schema design output.
            dry_run: If True, don't modify files, just return what would change.

        Returns:
            TransformationResult with all changes made.
        """
        import fnmatch

        source_repo = Path(source_repo)
        output_repo = Path(output_repo)
        generated_files: list[GeneratedFile] = []
        modified_files: list[str] = []
        errors: list[str] = []
        warnings: list[str] = []

        # Patterns to ignore when copying
        ignore_patterns = [
            "__pycache__",
            "*.pyc",
            "*.pyo",
            ".git",
            ".venv",
            "venv",
            ".pytest_cache",
            ".mypy_cache",
            "*.egg-info",
            ".DS_Store",
            "uv.lock",
        ]

        def should_ignore(path: Path, repo_root: Path) -> bool:
            """Check if path should be ignored."""
            rel_path = path.relative_to(repo_root)
            for part in rel_path.parts:
                for pattern in ignore_patterns:
                    if fnmatch.fnmatch(part, pattern):
                        return True
            return False

        # Step 1: Copy repository structure (if not dry run)
        if not dry_run:
            if output_repo.exists():
                shutil.rmtree(output_repo)
            output_repo.mkdir(parents=True)

            # Copy files while ignoring certain patterns
            for src_path in source_repo.rglob("*"):
                if src_path.is_file() and not should_ignore(src_path, source_repo):
                    rel_path = src_path.relative_to(source_repo)
                    dst_path = output_repo / rel_path
                    dst_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src_path, dst_path)

        # Build a map of function name -> classification
        classification_map: dict[str, NodeClassificationResult] = {}
        for c in classifications.classifications:
            classification_map[c.function_name] = c

        # Build a map of file path -> list of functions to decorate
        file_functions: dict[str, list[NodeClassificationResult]] = {}
        for c in classifications.classifications:
            file_path = c.file_path
            if file_path not in file_functions:
                file_functions[file_path] = []
            file_functions[file_path].append(c)

        # Step 2: Determine where to place governance files
        # Look for an 'agents' directory or similar
        agents_dir = self._find_agents_directory(source_repo, analysis)

        # Step 3: Generate and place governed_agents.py
        try:
            governed_content = self._generate_governed_agents(analysis, classifications)
            if agents_dir:
                governed_path = agents_dir / "governed_agents.py"
            else:
                governed_path = Path("governed_agents.py")

            generated_files.append(
                GeneratedFile(
                    path=str(governed_path),
                    content=governed_content,
                    description="ArbiterOS governance wrappers and decorators",
                )
            )

            if not dry_run:
                full_path = output_repo / governed_path
                full_path.parent.mkdir(parents=True, exist_ok=True)
                full_path.write_text(governed_content, encoding="utf-8")
        except Exception as e:
            errors.append(f"Failed to generate governed_agents.py: {e}")

        # Build schema-function mapping for LLM call wiring
        schema_function_map: dict[str, str] = {}
        for schema in schema_design.schemas:
            for func_name in schema.function_names:
                schema_function_map[func_name] = schema.class_name

        # Step 4: Modify agent files to add decorators and wire schemas
        for file_path, funcs in file_functions.items():
            try:
                result = self._modify_agent_file(
                    source_file=source_repo / file_path,
                    output_file=output_repo / file_path if not dry_run else None,
                    functions=funcs,
                    governed_agents_path=governed_path,
                    file_path=file_path,
                    schema_function_map=schema_function_map,
                    dry_run=dry_run,
                )
                if result:
                    modified_files.append(file_path)
            except Exception as e:
                warnings.append(f"Failed to modify {file_path}: {e}")

        # Step 4b: Handle files with compile() calls (e.g., setup.py)
        # These files need register_compiled_graph() added even if they don't have node functions
        for py_file in output_repo.rglob("*.py"):
            if py_file.is_file() and py_file.name not in ["__init__.py", "__pycache__"]:
                rel_path = py_file.relative_to(output_repo)
                # Skip if already modified
                if str(rel_path) in modified_files:
                    continue

                try:
                    # Parse the file to check for compile() calls
                    parser = AgentParser()
                    parsed = parser.parse_file(py_file)

                    if (
                        parsed.agent_type == "langgraph"
                        and parsed.compile_lineno
                        and not parsed.has_register_compiled_graph
                        and parsed.compile_in_return
                    ):
                        # This file has a compile() in return statement but wasn't modified yet
                        content = py_file.read_text(encoding="utf-8")
                        source_lines = content.splitlines()

                        # Add "os = get_arbiter_os()" if not present and needed
                        has_os_import = (
                            "get_arbiter_os" in content or "arbiter_os" in content
                        )
                        offset = 0

                        if not has_os_import:
                            # Find end of imports
                            import_end = parsed.imports_end_lineno

                            # Calculate proper import path to governed_agents
                            file_parts = str(rel_path).split("/")
                            governed_parts = str(governed_path).split("/")

                            # Determine relative import
                            if (
                                len(file_parts) > 1
                                and file_parts[0] == governed_parts[0]
                            ):
                                # Same package
                                import_stmt = f"from {file_parts[0]}.agents.governed_agents import get_arbiter_os"
                            else:
                                import_stmt = (
                                    "from governed_agents import get_arbiter_os"
                                )

                            source_lines.insert(import_end, "")
                            source_lines.insert(
                                import_end + 1,
                                "# Import ArbiterOS governance - call get_arbiter_os() dynamically, not at module import",
                            )
                            source_lines.insert(import_end + 2, import_stmt)
                            # NOTE: Do NOT add "os = get_arbiter_os()" here - call it dynamically when registering
                            offset += 3

                        # Handle the compile in return
                        compile_adjusted = parsed.compile_lineno + offset - 1
                        graph_var = parsed.graph_variable or "compiled_graph"

                        compile_line = source_lines[compile_adjusted].strip()

                        import re

                        match = re.search(
                            r"return\s+(.+\.compile\([^)]*\))", compile_line
                        )
                        if match:
                            compile_call = match.group(1)

                            # Replace the return line with assignment
                            indent = len(source_lines[compile_adjusted]) - len(
                                source_lines[compile_adjusted].lstrip()
                            )
                            source_lines[compile_adjusted] = (
                                f"{' ' * indent}{graph_var} = {compile_call}"
                            )

                            # Add registration and return
                            # Use get_arbiter_os() dynamically to get the current instance
                            source_lines.insert(compile_adjusted + 1, f"{' ' * indent}")
                            source_lines.insert(
                                compile_adjusted + 2,
                                f"{' ' * indent}# Register with ArbiterOS for governance tracking",
                            )
                            source_lines.insert(
                                compile_adjusted + 3,
                                f"{' ' * indent}# Use get_arbiter_os() dynamically to get the current instance",
                            )
                            source_lines.insert(
                                compile_adjusted + 4,
                                f"{' ' * indent}get_arbiter_os().register_compiled_graph({graph_var})",
                            )
                            source_lines.insert(compile_adjusted + 5, f"{' ' * indent}")
                            source_lines.insert(
                                compile_adjusted + 6,
                                f"{' ' * indent}return {graph_var}",
                            )

                            # Write back
                            py_file.write_text(
                                "\n".join(source_lines), encoding="utf-8"
                            )
                            modified_files.append(str(rel_path))
                except Exception:
                    # Silently skip files that can't be parsed
                    pass

        # Step 5: Generate policies directory
        policies_dir = output_repo / "policies" if not dry_run else Path("policies")

        # Generate policy_checkers.py
        try:
            checkers_content = self._generate_policy_checkers(
                analysis, policy_design, classifications
            )
            generated_files.append(
                GeneratedFile(
                    path="policies/trading_checkers.py",
                    content=checkers_content,
                    description="Domain-specific policy checkers",
                )
            )
            if not dry_run:
                (policies_dir).mkdir(parents=True, exist_ok=True)
                (policies_dir / "trading_checkers.py").write_text(
                    checkers_content, encoding="utf-8"
                )
        except Exception as e:
            errors.append(f"Failed to generate policy_checkers.py: {e}")

        # Generate policy_routers.py (routers in separate file)
        try:
            routers_content = self._generate_policy_routers(
                analysis, policy_design, classifications
            )
            generated_files.append(
                GeneratedFile(
                    path="policies/trading_routers.py",
                    content=routers_content,
                    description="Domain-specific policy routers",
                )
            )
            if not dry_run:
                (policies_dir / "trading_routers.py").write_text(
                    routers_content, encoding="utf-8"
                )
        except Exception as e:
            errors.append(f"Failed to generate policy_routers.py: {e}")

        # Generate policies/__init__.py
        try:
            init_content = self._generate_policies_init(analysis, policy_design)
            generated_files.append(
                GeneratedFile(
                    path="policies/__init__.py",
                    content=init_content,
                    description="Policies package init",
                )
            )
            if not dry_run:
                (policies_dir / "__init__.py").write_text(
                    init_content, encoding="utf-8"
                )
        except Exception as e:
            errors.append(f"Failed to generate policies/__init__.py: {e}")

        # Generate policies.yaml at repo root
        try:
            yaml_content = self._generate_policies_yaml(analysis, policy_design)
            generated_files.append(
                GeneratedFile(
                    path="policies.yaml",
                    content=yaml_content,
                    description="Policy configuration file",
                )
            )
            if not dry_run:
                (output_repo / "policies.yaml").write_text(
                    yaml_content, encoding="utf-8"
                )
        except Exception as e:
            errors.append(f"Failed to generate policies.yaml: {e}")

        # Generate llm_schemas.py
        try:
            schemas_content = self._generate_llm_schemas(analysis, schema_design)
            generated_files.append(
                GeneratedFile(
                    path="llm_schemas.py",
                    content=schemas_content,
                    description="Pydantic schemas for LLM I/O",
                )
            )
            if not dry_run:
                (output_repo / "llm_schemas.py").write_text(
                    schemas_content, encoding="utf-8"
                )
        except Exception as e:
            errors.append(f"Failed to generate llm_schemas.py: {e}")

        # Generate verification_nodes.py for high-risk operations
        try:
            verification_content = self._generate_verification_nodes(
                analysis, classifications
            )
            generated_files.append(
                GeneratedFile(
                    path="verification_nodes.py",
                    content=verification_content,
                    description="Verification nodes for high-risk operations",
                )
            )
            if not dry_run:
                (output_repo / "verification_nodes.py").write_text(
                    verification_content, encoding="utf-8"
                )

            # Log high-risk nodes that were identified
            high_risk_nodes = self._identify_high_risk_nodes(classifications)
            if high_risk_nodes:
                warnings.append(
                    f"Generated verification nodes for {len(high_risk_nodes)} high-risk operations: "
                    f"{', '.join(n['function_name'] for n in high_risk_nodes)}. "
                    "These nodes require verification before execution."
                )
        except Exception as e:
            errors.append(f"Failed to generate verification_nodes.py: {e}")

        return TransformationResult(
            success=len(errors) == 0,
            generated_files=generated_files,
            modified_files=modified_files,
            backup_files=[],
            errors=errors,
            warnings=warnings,
            summary=f"Transformed repository to {output_repo} with {len(modified_files)} modified files and {len(generated_files)} generated files",
        )

    def _find_agents_directory(
        self, repo_path: Path, analysis: RepoAnalysisOutput
    ) -> Path | None:
        """Find the agents directory in a repository.

        Args:
            repo_path: Path to the repository.
            analysis: Repository analysis output.

        Returns:
            Path to agents directory or None if not found.
        """
        # Common patterns for agent directories
        patterns = ["agents", "tradingagents/agents", "src/agents", "lib/agents"]

        for pattern in patterns:
            agent_dir = repo_path / pattern
            if agent_dir.exists() and agent_dir.is_dir():
                return Path(pattern)

        # Try to find from entry point
        if analysis.entry_point_file:
            entry_dir = Path(analysis.entry_point_file).parent
            return entry_dir

        return None

    def _modify_agent_file(
        self,
        source_file: Path,
        output_file: Path | None,
        functions: list,
        governed_agents_path: Path,
        file_path: str,
        schema_function_map: dict[str, str] | None = None,
        dry_run: bool = False,
    ) -> bool:
        """Modify an agent file to add decorators and wire LLM schemas.

        Args:
            source_file: Source file path.
            output_file: Output file path (None for dry run).
            functions: List of functions to decorate.
            governed_agents_path: Path to governed_agents.py for imports.
            file_path: Relative file path for import calculation.
            schema_function_map: Dict mapping function names to schema class names.
            dry_run: If True, don't write changes.

        Returns:
            True if modifications were made.
        """
        if not source_file.exists():
            return False

        schema_function_map = schema_function_map or {}
        content = source_file.read_text(encoding="utf-8")
        lines = content.splitlines()

        # Calculate import path for governed_agents
        # Convert file paths to module paths
        file_dir = Path(file_path).parent
        governed_dir = governed_agents_path.parent

        # Build relative import
        if file_dir == governed_dir:
            import_from = ".governed_agents"
        else:
            # Calculate relative import
            # For simplicity, use the package structure
            # Get the package name from the directory structure
            parts = file_path.split("/")
            if len(parts) > 1:
                package_root = parts[0]
                import_from = f"{package_root}.agents.governed_agents"
            else:
                import_from = "governed_agents"

        # Track modifications
        modifications = []
        offset = 0

        # Find import section end (handle multi-line imports)
        import_end = 0
        in_multiline_import = False
        paren_depth = 0

        for i, line in enumerate(lines):
            stripped = line.strip()

            # Track parentheses for multi-line imports
            if in_multiline_import:
                paren_depth += line.count("(") - line.count(")")
                if paren_depth <= 0:
                    in_multiline_import = False
                    import_end = i + 1
                continue

            if stripped.startswith("import ") or stripped.startswith("from "):
                # Check if this is a multi-line import
                paren_depth = line.count("(") - line.count(")")
                if paren_depth > 0:
                    in_multiline_import = True
                else:
                    import_end = i + 1
            elif stripped and not stripped.startswith("#") and import_end > 0:
                break

        # Collect wrapper names needed
        wrapper_names = set()
        for func in functions:
            wrapper_names.add(func.wrapper_name)

        # Add import for governance wrappers
        if wrapper_names:
            import_line = (
                f"from {import_from} import {', '.join(sorted(wrapper_names))}"
            )
            lines.insert(import_end, import_line)
            modifications.append(f"Added import: {import_line}")
            offset += 1

        # Determine which schemas are needed for this file
        schemas_needed = set()
        for func in functions:
            if func.function_name in schema_function_map:
                schemas_needed.add(schema_function_map[func.function_name])
            # Also check nested function names (e.g., bull_node inside create_bull_researcher)
            nested_name = func.function_name.replace("create_", "").replace(
                "_researcher", "_node"
            )
            if nested_name in schema_function_map:
                schemas_needed.add(schema_function_map[nested_name])

        # Add schema imports if needed
        if schemas_needed:
            schema_import = (
                f"from llm_schemas import {', '.join(sorted(schemas_needed))}"
            )
            lines.insert(import_end + offset, schema_import)
            modifications.append(f"Added schema import: {schema_import}")
            offset += 1

        # Add decorators to functions
        # We need to find nested functions inside factory functions
        # Process in reverse order of line number to maintain correct positions
        decorator_insertions = []

        # First pass: identify all decorator insertions needed
        try:
            tree = ast.parse("\n".join(lines))
        except SyntaxError:
            # If we can't parse, skip decorator additions
            if modifications and not dry_run and output_file:
                output_file.write_text("\n".join(lines), encoding="utf-8")
            return len(modifications) > 0

        for func in functions:
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    # Check if this is a factory function
                    if node.name == func.function_name:
                        # Find nested function if this is a factory
                        nested_func = None
                        for child in node.body:
                            if isinstance(
                                child, (ast.FunctionDef, ast.AsyncFunctionDef)
                            ):
                                nested_func = child
                                break

                        if nested_func:
                            # Record the decorator to insert
                            # lineno is 1-based, we need 0-based for list index
                            decorator_line = nested_func.lineno - 1
                            indent = " " * (nested_func.col_offset)
                            decorator = f"{indent}@{func.wrapper_name}"
                            decorator_insertions.append(
                                (
                                    decorator_line,
                                    decorator,
                                    func.function_name,
                                    func.wrapper_name,
                                )
                            )
                        break

        # Sort by line number in descending order so we don't mess up line numbers
        decorator_insertions.sort(key=lambda x: x[0], reverse=True)

        # Second pass: insert decorators in reverse order
        for line_num, decorator, factory_name, wrapper_name in decorator_insertions:
            lines.insert(line_num, decorator)
            modifications.append(
                f"Added @{wrapper_name} to nested function in {factory_name}"
            )

        # Third pass: Wire schemas into LLM calls
        # Find and modify llm.invoke() calls to use with_structured_output()
        content_after_decorators = "\n".join(lines)
        modified_content = self._wire_schemas_into_llm_calls(
            content_after_decorators, functions, schema_function_map
        )

        if modified_content != content_after_decorators:
            lines = modified_content.splitlines()
            modifications.append(
                "Wired schemas into LLM calls using with_structured_output()"
            )

        if modifications and not dry_run and output_file:
            output_file.write_text("\n".join(lines), encoding="utf-8")

        return len(modifications) > 0

    def _wire_schemas_into_llm_calls(
        self,
        content: str,
        functions: list,
        schema_function_map: dict[str, str],
    ) -> str:
        """Wire Pydantic schemas into LLM invoke calls.

        This method intelligently adds structured output to LLM calls:
        1. Identifies which functions have schemas
        2. Finds LLM invocations (llm.invoke() or chain = prompt | llm)
        3. Adds .with_structured_output(Schema) to appropriate calls
        4. Wraps responses in .to_message() when added to message lists
        5. Skips tool-bound chains (those using .bind_tools())

        Args:
            content: File content to modify.
            functions: List of functions being modified.
            schema_function_map: Mapping of function names to schema class names.

        Returns:
            Modified content with schemas wired into appropriate LLM calls.
        """
        import re

        lines = content.splitlines()
        modified = False

        # Build a map of nested function names to schemas
        func_to_schema = {}
        for func in functions:
            # Check factory function name
            if func.function_name in schema_function_map:
                func_to_schema[func.function_name] = schema_function_map[
                    func.function_name
                ]
            # Also check nested function names (e.g., bull_node from create_bull_researcher)
            nested_name = func.function_name.replace("create_", "").replace(
                "_researcher", "_node"
            )
            nested_name2 = func.function_name.replace("create_", "") + "_node"
            if nested_name in schema_function_map:
                func_to_schema[func.function_name] = schema_function_map[nested_name]
            elif nested_name2 in schema_function_map:
                func_to_schema[func.function_name] = schema_function_map[nested_name2]

        if not func_to_schema:
            return content

        # Get the schema name (assuming one schema per file for now)
        schema_name = list(func_to_schema.values())[0] if func_to_schema else None
        if not schema_name:
            return content

        # Track which variables have structured output
        structured_vars = set()

        # Find and modify LLM calls
        for i, line in enumerate(lines):
            stripped = line.strip()

            # Pattern 1: chain = prompt | llm (without bind_tools)
            # We want to change to: chain = prompt | llm.with_structured_output(Schema)
            if (
                "chain = " in stripped
                and " | llm" in stripped
                and ".bind_tools" not in stripped
            ):
                if ".with_structured_output" not in stripped:
                    # Find the llm part
                    match = re.search(r"(\|\s*llm)(?!\w)", stripped)
                    if match:
                        # Replace "| llm" with "| llm.with_structured_output(Schema)"
                        new_line = stripped.replace(
                            match.group(1),
                            f"| llm.with_structured_output({schema_name})",
                        )
                        indent = len(line) - len(line.lstrip())
                        lines[i] = " " * indent + new_line
                        modified = True

            # Pattern 2: response = llm.invoke(prompt) or result = llm.invoke(...)
            # Change to: response = llm.with_structured_output(Schema).invoke(prompt)
            # But skip if it's part of a chain with bind_tools
            match = re.search(r"(\w+)\s*=\s*llm\.invoke\(", stripped)
            if match:
                var_name = match.group(1)
                # Check if bind_tools is nearby (look back a few lines)
                has_bind_tools = False
                for j in range(max(0, i - 5), i):
                    if ".bind_tools" in lines[j]:
                        has_bind_tools = True
                        break

                if not has_bind_tools and ".with_structured_output" not in stripped:
                    # Replace llm.invoke with llm.with_structured_output(Schema).invoke
                    new_line = re.sub(
                        r"llm\.invoke\(",
                        f"llm.with_structured_output({schema_name}).invoke(",
                        stripped,
                    )
                    indent = len(line) - len(line.lstrip())
                    lines[i] = " " * indent + new_line
                    structured_vars.add(var_name)
                    modified = True

            # Pattern 3: "messages": [result] or "messages": [response]
            # Change to: "messages": [result.to_message()]
            # Only for variables we know have structured output
            for var_name in structured_vars:
                pattern = f'"messages"\\s*:\\s*\\[{var_name}\\]'
                if re.search(pattern, stripped):
                    new_line = re.sub(
                        f"\\[{var_name}\\]", f"[{var_name}.to_message()]", stripped
                    )
                    indent = len(line) - len(line.lstrip())
                    lines[i] = " " * indent + new_line
                    modified = True

        if modified:
            return "\n".join(lines)
        return content

    def generate_governance_files(
        self,
        analysis: RepoAnalysisOutput,
        classifications: NodeClassificationBatch,
        policy_design: PolicyDesignOutput,
        schema_design: LLMSchemaDesignOutput,
        output_dir: str | Path,
        dry_run: bool = False,
    ) -> TransformationResult:
        """Generate all governance files for a repository transformation.

        Args:
            analysis: Repository analysis output.
            classifications: Node classification results.
            policy_design: Policy design output.
            schema_design: Schema design output.
            output_dir: Directory to write generated files.
            dry_run: If True, don't write files, just return what would be generated.

        Returns:
            TransformationResult with all generated files.
        """
        output_dir = Path(output_dir)
        generated_files: list[GeneratedFile] = []
        errors: list[str] = []
        warnings: list[str] = []

        # 1. Generate governed_agents.py
        try:
            governed_content = self._generate_governed_agents(analysis, classifications)
            generated_files.append(
                GeneratedFile(
                    path="governed_agents.py",
                    content=governed_content,
                    description="ArbiterOS governance wrappers and decorators",
                )
            )
        except Exception as e:
            errors.append(f"Failed to generate governed_agents.py: {e}")

        # 2. Generate policy_checkers.py
        try:
            checkers_content = self._generate_policy_checkers(
                analysis, policy_design, classifications
            )
            generated_files.append(
                GeneratedFile(
                    path="policy_checkers.py",
                    content=checkers_content,
                    description="Domain-specific policy checkers and routers",
                )
            )
        except Exception as e:
            errors.append(f"Failed to generate policy_checkers.py: {e}")

        # 3. Generate llm_schemas.py
        try:
            schemas_content = self._generate_llm_schemas(analysis, schema_design)
            generated_files.append(
                GeneratedFile(
                    path="llm_schemas.py",
                    content=schemas_content,
                    description="Pydantic schemas for LLM I/O",
                )
            )
        except Exception as e:
            errors.append(f"Failed to generate llm_schemas.py: {e}")

        # 4. Generate policies.yaml
        try:
            yaml_content = self._generate_policies_yaml(analysis, policy_design)
            generated_files.append(
                GeneratedFile(
                    path="policies.yaml",
                    content=yaml_content,
                    description="Policy configuration file",
                )
            )
        except Exception as e:
            errors.append(f"Failed to generate policies.yaml: {e}")

        # Validate generated Python files
        for gf in generated_files:
            if gf.path.endswith(".py"):
                is_valid, error = self._validate_python_syntax(gf.content)
                if not is_valid:
                    warnings.append(
                        f"Syntax warning in {gf.path}: {error}. "
                        "File generated but may need manual fixes."
                    )

        # Write files if not dry run
        if not dry_run and not errors:
            output_dir.mkdir(parents=True, exist_ok=True)
            for gf in generated_files:
                file_path = output_dir / gf.path
                file_path.write_text(gf.content, encoding="utf-8")

        return TransformationResult(
            success=len(errors) == 0,
            generated_files=generated_files,
            modified_files=[],
            backup_files=[],
            errors=errors,
            warnings=warnings,
            summary=self._build_summary(analysis, generated_files, errors),
        )

    def _generate_governed_agents(
        self,
        analysis: RepoAnalysisOutput,
        classifications: NodeClassificationBatch,
    ) -> str:
        """Generate governed_agents.py content."""
        # Build wrapper specifications
        wrappers: list[dict] = []
        instruction_imports: set[str] = set()

        # Group classifications by wrapper name
        wrapper_map: dict[str, GovernanceWrapperSpec] = {}

        for c in classifications.classifications:
            if c.wrapper_name not in wrapper_map:
                core_import, instr_import = INSTRUCTION_IMPORT_MAP.get(
                    c.instruction_type,
                    ("CognitiveCore", "CognitiveCore.GENERATE"),
                )
                instruction_imports.add(core_import)

                wrapper_map[c.wrapper_name] = GovernanceWrapperSpec(
                    wrapper_name=c.wrapper_name,
                    instruction_type=c.instruction_type,
                    core=c.core,
                    description=c.reasoning,
                    functions_to_wrap=[c.function_name],
                )
            else:
                wrapper_map[c.wrapper_name].functions_to_wrap.append(c.function_name)

        # Build wrapper data for template
        for wrapper_name, spec in wrapper_map.items():
            core_import, instr_import = INSTRUCTION_IMPORT_MAP.get(
                spec.instruction_type,
                ("CognitiveCore", "CognitiveCore.GENERATE"),
            )

            # Generate constant name from wrapper name
            constant_name = wrapper_name.upper().replace("GOVERN_", "") + "_INSTRUCTION"

            wrappers.append(
                {
                    "wrapper_name": wrapper_name,
                    "constant_name": constant_name,
                    "instruction_type": spec.instruction_type.value,
                    "instruction_import": instr_import,
                    "role_description": f"{wrapper_name.replace('govern_', '')} functions",
                    "function_description": spec.description
                    or "perform their designated task",
                }
            )

        template = self._jinja_env.get_template("governed_agents.py.jinja")
        return template.render(
            domain=analysis.domain,
            backend=analysis.framework.value,
            instruction_imports=sorted(instruction_imports),
            wrappers=wrappers,
        )

    def _generate_policy_checkers(
        self,
        analysis: RepoAnalysisOutput,
        policy_design: PolicyDesignOutput,
        classifications: NodeClassificationBatch,
    ) -> str:
        """Generate policy_checkers.py content."""
        # Collect instruction imports
        instruction_imports: set[str] = set()

        for checker in policy_design.checkers:
            for instr in checker.instructions_to_track:
                core_import, _ = INSTRUCTION_IMPORT_MAP.get(
                    instr, ("CognitiveCore", "CognitiveCore.GENERATE")
                )
                instruction_imports.add(core_import)

        # Add parameter types to checkers and convert instruction enums to full references
        checkers_with_types = []
        for checker in policy_design.checkers:
            checker_dict = checker.model_dump()
            checker_dict["parameter_types"] = self._infer_parameter_types(
                checker.parameters
            )
            # Convert instruction enum values to full qualified references
            checker_dict["instructions_to_track"] = [
                {
                    "value": instr.value,
                    "full_ref": INSTRUCTION_IMPORT_MAP.get(
                        instr, ("CognitiveCore", "CognitiveCore.GENERATE")
                    )[1],
                }
                for instr in checker.instructions_to_track
            ]
            checkers_with_types.append(checker_dict)

        # Add parameter types to routers
        routers_with_types = []
        for router in policy_design.routers:
            router_dict = router.model_dump()
            router_dict["parameter_types"] = self._infer_parameter_types(
                router.parameters
            )
            routers_with_types.append(router_dict)

        template = self._jinja_env.get_template("policy_checkers.py.jinja")
        return template.render(
            domain=analysis.domain,
            instruction_imports=sorted(instruction_imports),
            checkers=checkers_with_types,
            routers=routers_with_types,
        )

    def _generate_llm_schemas(
        self,
        analysis: RepoAnalysisOutput,
        schema_design: LLMSchemaDesignOutput,
    ) -> str:
        """Generate llm_schemas.py content."""
        template = self._jinja_env.get_template("llm_schemas.py.jinja")
        return template.render(
            domain=analysis.domain,
            schemas=[s.model_dump() for s in schema_design.schemas],
            additional_imports=schema_design.imports_needed,
        )

    def _generate_policies_yaml(
        self,
        analysis: RepoAnalysisOutput,
        policy_design: PolicyDesignOutput,
    ) -> str:
        """Generate policies.yaml content."""
        template = self._jinja_env.get_template("policies.yaml.jinja")
        return template.render(
            domain=analysis.domain,
            yaml_checkers=[c.model_dump() for c in policy_design.yaml_checkers],
            yaml_routers=[r.model_dump() for r in policy_design.yaml_routers],
        )

    def _generate_policy_routers(
        self,
        analysis: RepoAnalysisOutput,
        policy_design: PolicyDesignOutput,
        classifications: NodeClassificationBatch,
    ) -> str:
        """Generate trading_routers.py content with policy routers.

        Routers include:
        - Safe fallback mechanisms when routing decisions fail
        - Multiple trigger condition support
        - Detailed logging for audit trails
        - Type-safe threshold comparisons
        """
        # Collect instruction imports for routers
        instruction_imports: set[str] = set()

        for router in policy_design.routers:
            # Add default imports for routers
            instruction_imports.add("CognitiveCore")
            instruction_imports.add("NormativeCore")

        # Add parameter types to routers
        routers_with_types = []
        for router in policy_design.routers:
            router_dict = router.model_dump()
            router_dict["parameter_types"] = self._infer_parameter_types(
                router.parameters
            )
            routers_with_types.append(router_dict)

        # Generate routers content with enhanced safety features
        content = f'''"""{analysis.domain}-specific policy routers for ArbiterOS governance.

This module provides policy routers designed for the {analysis.domain}
multi-agent framework, enabling dynamic workflow routing with safety guarantees.

Safety Features:
- Fallback to safe defaults when routing conditions are ambiguous
- Detailed logging for audit trails
- Multiple trigger condition support
- Type-safe threshold comparisons

Generated by ArbiterOS Migration Tool.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

from arbiteros_alpha.history import History
from arbiteros_alpha.instructions import (
    {", ".join(sorted(instruction_imports))},
)
from arbiteros_alpha.policy import PolicyRouter

logger = logging.getLogger(__name__)


class RouterDecision:
    """Encapsulates a routing decision with audit information."""
    
    def __init__(
        self,
        target: Optional[str],
        reason: str,
        trigger_value: Any = None,
        threshold: float = None,
    ):
        self.target = target
        self.reason = reason
        self.trigger_value = trigger_value
        self.threshold = threshold
    
    def log(self, router_name: str) -> None:
        """Log the routing decision for audit trail."""
        if self.target:
            logger.warning(
                f"[{{router_name}}] ROUTING: {{self.reason}} "
                f"(value={{self.trigger_value}}, threshold={{self.threshold}}) -> {{self.target}}"
            )
        else:
            logger.debug(
                f"[{{router_name}}] Normal flow: {{self.reason}}"
            )


class SafeRouterMixin:
    """Mixin providing safe routing utilities."""
    
    def _safe_get_value(
        self,
        output_state: Dict[str, Any],
        key: str,
        default: Any = None
    ) -> Any:
        """Safely extract a value from output_state with fallback."""
        value = output_state.get(key, default)
        
        # Handle nested keys (e.g., "analysis.confidence")
        if value is None and "." in key:
            parts = key.split(".")
            current = output_state
            for part in parts:
                if isinstance(current, dict):
                    current = current.get(part)
                else:
                    return default
            return current if current is not None else default
        
        return value
    
    def _compare_threshold(
        self,
        value: Any,
        threshold: float,
        comparison: str = "<"
    ) -> bool:
        """Safely compare a value against threshold."""
        if value is None:
            return False
        
        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            logger.warning(f"Cannot compare non-numeric value: {{value}}")
            return False
        
        if comparison == "<":
            return numeric_value < threshold
        elif comparison == ">":
            return numeric_value > threshold
        elif comparison == "<=":
            return numeric_value <= threshold
        elif comparison == ">=":
            return numeric_value >= threshold
        elif comparison == "==":
            return numeric_value == threshold
        else:
            logger.error(f"Unknown comparison operator: {{comparison}}")
            return False

'''

        # Generate each router class
        for router in routers_with_types:
            params_code = ""
            for param_name, param_value in router.get("parameters", {}).items():
                param_type = router.get("parameter_types", {}).get(param_name, "Any")
                if isinstance(param_value, str):
                    params_code += f'    {param_name}: {param_type} = "{param_value}"\n'
                elif isinstance(param_value, list):
                    params_code += f"    {param_name}: list = field(default_factory=lambda: {param_value})\n"
                else:
                    params_code += f"    {param_name}: {param_type} = {param_value}\n"

            # Determine the key to check based on the trigger condition
            trigger_condition = router.get("trigger_condition", "")
            trigger_key = "confidence"  # default
            threshold = 0.7  # default
            comparison = "<"  # default

            if "confidence" in trigger_condition.lower():
                trigger_key = "confidence"
                threshold = 0.7
                comparison = "<"
            elif "risk" in trigger_condition.lower():
                trigger_key = "risk_level"
                threshold = 0.8
                comparison = ">"
            elif "quality" in trigger_condition.lower():
                trigger_key = "quality_score"
                threshold = 0.6
                comparison = "<"

            content += f'''
@dataclass
class {router["class_name"]}(PolicyRouter, SafeRouterMixin):
    """{router["description"]}

    This router enforces safety by dynamically redirecting workflow execution
    when trigger conditions are met. Includes fallback mechanisms for
    graceful degradation.

    Attributes:
        name: Human-readable name for this policy router.
        target: The node name to route to when triggered.
        fallback_target: Fallback node if primary target is unavailable.
        threshold: The threshold value for triggering routing.
        key: The key in output_state to check.
        enabled: Whether this router is active.
    """

    name: str
    target: str = "{router["target_node"]}"
    fallback_target: str = "human_review"  # Safe fallback
    threshold: float = {threshold}
    key: str = "{trigger_key}"
    enabled: bool = True
{params_code}

    def route_after(self, history: History) -> Optional[str]:
        """{router["route_logic_description"]}

        Trigger condition: {router["trigger_condition"]}

        This method includes:
        - Safe value extraction with fallbacks
        - Type-safe threshold comparison
        - Audit logging for all decisions
        - Fallback to safe target on errors

        Args:
            history: The execution history including the just-executed instruction.

        Returns:
            Target node name to route to, or None for normal flow.
        """
        if not self.enabled:
            logger.debug(f"[{{self.name}}] Router disabled, skipping")
            return None

        if not history.entries or not history.entries[-1]:
            logger.debug(f"[{{self.name}}] No history entries, normal flow")
            return None

        last_superstep = history.entries[-1]
        if not last_superstep:
            return None

        last_item = last_superstep[-1]
        output_state = last_item.output_state or {{}}

        # Safely extract the value to check
        value = self._safe_get_value(output_state, self.key)
        
        # If key not found, check for requires_human_review flag as safety fallback
        if value is None:
            if output_state.get("requires_human_review", False):
                decision = RouterDecision(
                    target=self.fallback_target,
                    reason="requires_human_review flag set, missing metric value",
                    trigger_value=None,
                    threshold=self.threshold
                )
                decision.log(self.name)
                return self.fallback_target
            
            logger.debug(f"[{{self.name}}] Key '{{self.key}}' not found in output_state, normal flow")
            return None
        
        # Type-safe comparison
        try:
            triggered = self._compare_threshold(value, self.threshold, "{comparison}")
        except Exception as e:
            logger.error(f"[{{self.name}}] Comparison error: {{e}}, routing to fallback")
            return self.fallback_target
        
        if triggered:
            decision = RouterDecision(
                target=self.target,
                reason="{router["trigger_condition"]}",
                trigger_value=value,
                threshold=self.threshold
            )
            decision.log(self.name)
            return self.target
        
        logger.debug(
            f"[{{self.name}}] Condition not met ({{value}} not {comparison} {{self.threshold}}), normal flow"
        )
        return None

'''

        # Add a combined safety router that checks multiple conditions
        content += '''

# =============================================================================
# Combined Safety Router
# =============================================================================

@dataclass
class CombinedSafetyRouter(PolicyRouter, SafeRouterMixin):
    """Combined router that checks multiple safety conditions.
    
    This router acts as a catch-all safety net, checking multiple conditions
    and routing to a safe node if ANY condition is met.
    
    Attributes:
        name: Human-readable name for this router.
        safe_target: Node to route to when any safety condition triggers.
        confidence_threshold: Minimum confidence required (below triggers).
        risk_threshold: Maximum risk allowed (above triggers).
        quality_threshold: Minimum quality required (below triggers).
    """
    
    name: str
    safe_target: str = "human_review"
    confidence_threshold: float = 0.5
    risk_threshold: float = 0.9
    quality_threshold: float = 0.4
    enabled: bool = True
    
    def route_after(self, history: History) -> Optional[str]:
        """Check multiple safety conditions and route if any trigger.
        
        Conditions checked (in order):
        1. confidence < confidence_threshold
        2. risk_level > risk_threshold
        3. quality_score < quality_threshold
        4. requires_human_review flag is True
        5. Any validation_errors present
        
        Args:
            history: Execution history.
            
        Returns:
            safe_target if any condition met, None otherwise.
        """
        if not self.enabled:
            return None
            
        if not history.entries or not history.entries[-1]:
            return None
            
        last_item = history.entries[-1][-1]
        output_state = last_item.output_state or {{}}
        
        # Check confidence
        confidence = self._safe_get_value(output_state, "confidence")
        if confidence is not None and self._compare_threshold(confidence, self.confidence_threshold, "<"):
            logger.warning(
                f"[{{self.name}}] Low confidence ({{confidence}} < {{self.confidence_threshold}}), "
                f"routing to {{self.safe_target}}"
            )
            return self.safe_target
        
        # Check risk level
        risk_level = self._safe_get_value(output_state, "risk_level")
        if risk_level is not None and self._compare_threshold(risk_level, self.risk_threshold, ">"):
            logger.warning(
                f"[{{self.name}}] High risk ({{risk_level}} > {{self.risk_threshold}}), "
                f"routing to {{self.safe_target}}"
            )
            return self.safe_target
        
        # Check quality score
        quality_score = self._safe_get_value(output_state, "quality_score")
        if quality_score is not None and self._compare_threshold(quality_score, self.quality_threshold, "<"):
            logger.warning(
                f"[{{self.name}}] Low quality ({{quality_score}} < {{self.quality_threshold}}), "
                f"routing to {{self.safe_target}}"
            )
            return self.safe_target
        
        # Check explicit flags
        if output_state.get("requires_human_review", False):
            logger.warning(f"[{{self.name}}] requires_human_review flag set, routing to {{self.safe_target}}")
            return self.safe_target
        
        # Check for validation errors
        validation_errors = output_state.get("validation_errors", [])
        if validation_errors:
            logger.warning(
                f"[{{self.name}}] Validation errors present: {{validation_errors}}, "
                f"routing to {{self.safe_target}}"
            )
            return self.safe_target
        
        return None
'''

        return content

    def _generate_policies_init(
        self,
        analysis: RepoAnalysisOutput,
        policy_design: PolicyDesignOutput,
    ) -> str:
        """Generate policies/__init__.py content."""
        checker_names = [c.class_name for c in policy_design.checkers]
        router_names = [r.class_name for r in policy_design.routers]

        content = f'''"""{analysis.domain}-specific policies for ArbiterOS governance.

This package provides custom policy checkers and routers designed for
the {analysis.domain} multi-agent framework.

Generated by ArbiterOS Migration Tool.
"""

from .trading_checkers import (
'''
        for name in checker_names:
            content += f"    {name},\n"
        content += """)
from .trading_routers import (
"""
        for name in router_names:
            content += f"    {name},\n"
        content += """)

__all__ = [
"""
        for name in checker_names + router_names:
            content += f'    "{name}",\n'
        content += "]\n"

        return content

    def _infer_parameter_types(self, parameters: dict) -> dict[str, str]:
        """Infer Python type annotations from parameter values."""
        type_map = {}
        for name, value in parameters.items():
            if isinstance(value, bool):
                type_map[name] = "bool"
            elif isinstance(value, int):
                type_map[name] = "int"
            elif isinstance(value, float):
                type_map[name] = "float"
            elif isinstance(value, str):
                type_map[name] = "str"
            elif isinstance(value, list):
                type_map[name] = "list"
            elif isinstance(value, set):
                type_map[name] = "Set[str]"
            elif isinstance(value, dict):
                type_map[name] = "dict"
            else:
                type_map[name] = "Any"
        return type_map

    def _validate_python_syntax(self, code: str) -> tuple[bool, str | None]:
        """Validate Python code syntax.

        Args:
            code: Python source code to validate.

        Returns:
            Tuple of (is_valid, error_message).
        """
        try:
            ast.parse(code)
            return True, None
        except SyntaxError as e:
            return False, f"Line {e.lineno}: {e.msg}"

    def _build_summary(
        self,
        analysis: RepoAnalysisOutput,
        generated_files: list[GeneratedFile],
        errors: list[str],
    ) -> str:
        """Build a summary of the transformation."""
        if errors:
            return f"Transformation completed with {len(errors)} errors."

        files_list = ", ".join(gf.path for gf in generated_files)
        return (
            f"Successfully generated {len(generated_files)} governance files "
            f"for {analysis.domain} domain: {files_list}"
        )

    def _identify_high_risk_nodes(
        self,
        classifications: NodeClassificationBatch,
    ) -> list[dict]:
        """Identify nodes that require verification before execution.

        High-risk nodes include:
        - TOOL_CALL: External system interactions (API calls, database writes)
        - TOOL_BUILD: Code generation
        - RESPOND: Final user-facing output
        - DELEGATE: Multi-agent delegation

        Args:
            classifications: Node classification results.

        Returns:
            List of high-risk node specifications with verification config.
        """
        high_risk_nodes = []

        for c in classifications.classifications:
            if c.instruction_type in self.HIGH_RISK_INSTRUCTIONS:
                # Determine risk thresholds based on instruction type
                if c.instruction_type == InstructionTypeEnum.TOOL_CALL:
                    # External system interaction - strictest verification
                    max_risk = 0.7
                    min_confidence = 0.7
                    require_approval = True
                    human_review = 0.85
                elif c.instruction_type == InstructionTypeEnum.RESPOND:
                    # User-facing output - moderate verification
                    max_risk = 0.8
                    min_confidence = 0.6
                    require_approval = False
                    human_review = 0.9
                else:
                    # Default high-risk thresholds
                    max_risk = 0.8
                    min_confidence = 0.6
                    require_approval = False
                    human_review = 0.9

                high_risk_nodes.append(
                    {
                        "function_name": c.function_name,
                        "instruction_type": c.instruction_type.value,
                        "core": c.core.value,
                        "max_risk_threshold": max_risk,
                        "min_confidence_threshold": min_confidence,
                        "require_explicit_approval": require_approval,
                        "human_review_threshold": human_review,
                        "auto_approve_low_risk": 0.3,
                        "rejected_target": "human_review",
                    }
                )

        return high_risk_nodes

    def _generate_verification_nodes(
        self,
        analysis: RepoAnalysisOutput,
        classifications: NodeClassificationBatch,
    ) -> str:
        """Generate verification nodes for high-risk operations.

        This creates a Python module containing:
        - VerificationConfig for each high-risk node
        - Verification node functions
        - Verification routers
        - Human review fallback node
        - Integration utilities

        Args:
            analysis: Repository analysis output.
            classifications: Node classification results.

        Returns:
            Generated Python code for verification nodes.
        """
        high_risk_nodes = self._identify_high_risk_nodes(classifications)

        if not high_risk_nodes:
            # No high-risk nodes, return minimal module
            return f'''"""Verification nodes for {analysis.domain} governance.

No high-risk nodes were identified that require verification.
This module is a placeholder for future verification requirements.

Generated by ArbiterOS Migration Tool.
"""

# No verification nodes generated - all nodes are considered low-risk
# To add verification, create high-risk node functions using:
#   from verification_nodes import create_verification_node, VerificationConfig
'''

        template = self._jinja_env.get_template("verification_nodes.py.jinja")
        return template.render(
            domain=analysis.domain,
            high_risk_nodes=high_risk_nodes,
        )
