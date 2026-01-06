"""Code generator for migrating agents to ArbiterOS-governed agents.

This module handles the actual code migration, including:
- Adding imports
- Adding OS initialization
- Adding @arbiter_os.instruction() decorators
- Adding register_compiled_graph() for LangGraph agents
- Creating backups before modification
"""

import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from .classifier import NodeClassification
from .parser import ParsedAgent


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


class CodeGenerator:
    """Generates transformed code with ArbiterOS governance.

    This class takes parsed agent information and classification results
    to produce transformed Python code with ArbiterOS decorators and setup.

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
    LANGFUSE_IMPORTS = [
        "from langfuse.langchain import CallbackHandler",
    ]

    def __init__(self) -> None:
        """Initialize the CodeGenerator."""
        pass

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

            # Add Langfuse imports if not already present
            if not parsed_agent.has_existing_langfuse:
                import_line = parsed_agent.imports_end_lineno
                # Use current offset (which includes ArbiterOS imports if added)
                for i, import_stmt in enumerate(self.LANGFUSE_IMPORTS):
                    source_lines.insert(import_line + offset + i, import_stmt)
                    changes.append(f"Added import: {import_stmt}")
                offset += len(self.LANGFUSE_IMPORTS)

            # 2. Add OS initialization
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
                    f'arbiter_os = ArbiterOSAlpha(backend="{backend}")',
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
                decorator = f"@arbiter_os.instruction(Instr.{instr_type})"

                source_lines.insert(insert_line, decorator)
                changes.append(
                    f"Added @arbiter_os.instruction(Instr.{instr_type}) to {func.name}"
                )
                decorators_added += 1

            # Update total offset with all decorators added
            offset += decorators_added

            # 4. Add Langfuse handler initialization if not already present
            if not parsed_agent.has_existing_langfuse:
                # Find a good place to insert (after OS initialization or before first function)
                if first_func_line is not None:
                    insert_pos = first_func_line - 1 + offset
                    # If we added OS init in this pass, insert after it (3 lines: blank, os init, blank)
                    # Otherwise, insert before first function
                    os_init_line_count = 3  # ["", "os = ...", ""]
                    if not parsed_agent.has_existing_arbiteros:
                        # We added OS init, so insert after it
                        insert_pos += os_init_line_count

                    langfuse_lines = [
                        "",
                        "langfuse_handler = CallbackHandler()",
                        "",
                    ]
                    for i, line in enumerate(langfuse_lines):
                        source_lines.insert(insert_pos + i, line)
                        if line.strip():
                            changes.append(
                                f"Added Langfuse handler initialization at line {insert_pos + i + 1}"
                            )
                    offset += len(langfuse_lines)

            # 5. Add register_compiled_graph() for LangGraph agents
            if parsed_agent.agent_type == "langgraph" and parsed_agent.compile_lineno:
                # Find the compile line with offset
                compile_adjusted = parsed_agent.compile_lineno + offset

                # Find the actual line in source
                graph_var = parsed_agent.graph_variable or "graph"
                register_line = f"arbiter_os.register_compiled_graph({graph_var})"

                # Insert after compile line
                source_lines.insert(compile_adjusted, register_line)
                changes.append(f"Added arbiter_os.register_compiled_graph({graph_var})")
                offset += 1

            # 6. Add langfuse_handler to graph invocation calls (simple single-line modification)
            if (
                not parsed_agent.has_existing_langfuse
                and parsed_agent.graph_invocation_lines
            ):
                # Process in reverse order to maintain line numbers
                for invoc_line in sorted(
                    parsed_agent.graph_invocation_lines, reverse=True
                ):
                    adjusted_line = invoc_line - 1 + offset
                    if adjusted_line < len(source_lines):
                        original_line = source_lines[adjusted_line]
                        # Only modify if it's a single-line call (simple heuristic)
                        if (
                            ".stream(" in original_line or ".invoke(" in original_line
                        ) and original_line.count("\n") == 0:
                            # Check if config already exists
                            if "config=" in original_line:
                                # Add langfuse_handler to existing config if callbacks not present
                                if (
                                    "langfuse_handler" not in original_line
                                    and "callbacks" not in original_line
                                ):
                                    # Simple insertion: add callbacks to config dict
                                    config_pos = original_line.find("config=")
                                    if config_pos != -1:
                                        brace_pos = original_line.find("{", config_pos)
                                        if brace_pos != -1:
                                            new_line = (
                                                original_line[: brace_pos + 1]
                                                + '"callbacks": [langfuse_handler], '
                                                + original_line[brace_pos + 1 :]
                                            )
                                            source_lines[adjusted_line] = new_line
                                            changes.append(
                                                f"Added langfuse_handler to config at line {invoc_line}"
                                            )
                            else:
                                # Add config parameter with langfuse_handler
                                paren_pos = original_line.rfind(")")
                                if paren_pos != -1:
                                    new_line = (
                                        original_line[:paren_pos]
                                        + ', config={"callbacks": [langfuse_handler]}'
                                        + original_line[paren_pos:]
                                    )
                                    source_lines[adjusted_line] = new_line
                                    changes.append(
                                        f"Added config with langfuse_handler at line {invoc_line}"
                                    )

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

        # Add Langfuse imports if not already present
        if not parsed_agent.has_existing_langfuse:
            import_line = parsed_agent.imports_end_lineno
            for i, import_stmt in enumerate(self.LANGFUSE_IMPORTS):
                source_lines.insert(import_line + offset + i, import_stmt)
            offset += len(self.LANGFUSE_IMPORTS)

        # Add OS initialization
        first_func_line = min((f.lineno for f in parsed_agent.functions), default=None)
        if first_func_line is not None:
            backend = parsed_agent.agent_type
            os_init_lines = [
                "",
                f'arbiter_os = ArbiterOSAlpha(backend="{backend}")',
                "",
            ]
            insert_pos = first_func_line - 1 + offset
            for i, line in enumerate(os_init_lines):
                source_lines.insert(insert_pos + i, line)
            offset += len(os_init_lines)

            # Add Langfuse handler initialization if not already present
            if not parsed_agent.has_existing_langfuse:
                # Insert after OS init (offset already includes os_init_lines)
                insert_pos = first_func_line - 1 + offset
                langfuse_lines = [
                    "",
                    "langfuse_handler = CallbackHandler()",
                    "",
                ]
                for i, line in enumerate(langfuse_lines):
                    source_lines.insert(insert_pos + i, line)
                offset += len(langfuse_lines)

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
            decorator = (
                f"@arbiter_os.instruction(Instr.{classification.instruction_type})"
            )
            source_lines.insert(insert_line, decorator)
            decorators_added += 1

        offset += decorators_added

        # Add register_compiled_graph for LangGraph
        if parsed_agent.agent_type == "langgraph" and parsed_agent.compile_lineno:
            compile_adjusted = parsed_agent.compile_lineno + offset
            graph_var = parsed_agent.graph_variable or "graph"
            source_lines.insert(
                compile_adjusted, f"arbiter_os.register_compiled_graph({graph_var})"
            )
            offset += 1

        # Add langfuse_handler to graph invocation calls (simple single-line modification)
        if (
            not parsed_agent.has_existing_langfuse
            and parsed_agent.graph_invocation_lines
        ):
            for invoc_line in sorted(parsed_agent.graph_invocation_lines, reverse=True):
                adjusted_line = invoc_line - 1 + offset
                if adjusted_line < len(source_lines):
                    original_line = source_lines[adjusted_line]
                    if (
                        ".stream(" in original_line or ".invoke(" in original_line
                    ) and original_line.count("\n") == 0:
                        if "config=" in original_line:
                            if (
                                "langfuse_handler" not in original_line
                                and "callbacks" not in original_line
                            ):
                                config_pos = original_line.find("config=")
                                if config_pos != -1:
                                    brace_pos = original_line.find("{", config_pos)
                                    if brace_pos != -1:
                                        source_lines[adjusted_line] = (
                                            original_line[: brace_pos + 1]
                                            + '"callbacks": [langfuse_handler], '
                                            + original_line[brace_pos + 1 :]
                                        )
                        else:
                            paren_pos = original_line.rfind(")")
                            if paren_pos != -1:
                                source_lines[adjusted_line] = (
                                    original_line[:paren_pos]
                                    + ', config={"callbacks": [langfuse_handler]}'
                                    + original_line[paren_pos:]
                                )

        return "\n".join(source_lines)
