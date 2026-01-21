"""Rich-based progress logger for migration pipeline.

This module provides beautiful, real-time progress logging for the
migration pipeline using the Rich library.
"""

from dataclasses import dataclass

from rich.console import Console
from rich.panel import Panel
from rich.table import Table


@dataclass
class ClassificationResult:
    """Result of classifying a single function.

    Attributes:
        function_name: Name of the function.
        instruction_type: The classified instruction type (e.g., "GENERATE").
        core: The core the instruction belongs to (e.g., "Cognitive").
        confidence: Confidence score from 0.0 to 1.0.
        reasoning: Explanation for the classification.
    """

    function_name: str
    instruction_type: str
    core: str
    confidence: float
    reasoning: str = ""


class MigrationLogger:
    """Rich-based logger for migration progress.

    Provides step-by-step visual feedback during the migration process
    with tables, progress indicators, and styled output.

    Example:
        >>> logger = MigrationLogger()
        >>> logger.start()
        >>> logger.step_parsing("agent.py")
        >>> logger.found_functions(["generate", "verify", "execute"])
        >>> logger.detected_agent_type("langgraph")
        >>> logger.step_classifying()
        >>> logger.show_classifications([...])
        >>> logger.step_backup("agent_backup.py")
        >>> logger.step_transforming()
        >>> logger.transformation_action("Added imports at line 1")
        >>> logger.complete("agent.py", "agent_backup.py")
    """

    def __init__(self, console: Console | None = None, verbose: bool = True) -> None:
        """Initialize the MigrationLogger.

        Args:
            console: Optional Rich Console instance. Creates new one if not provided.
            verbose: If True, show detailed output. If False, minimal output.
        """
        self.console = console or Console()
        self.verbose = verbose
        self._current_step = 0
        self._total_steps = 5

    def start(self) -> None:
        """Display the migration tool header."""
        self.console.print()
        self.console.print(
            Panel.fit(
                "[bold cyan]ArbiterOS Migration Tool[/bold cyan]",
                border_style="cyan",
            )
        )
        self.console.print()

    def step_parsing(self, filename: str) -> None:
        """Log the start of parsing step.

        Args:
            filename: Name of the file being parsed.
        """
        self._current_step = 1
        self._print_step("Parsing source file...")
        if self.verbose:
            self.console.print(f"      [dim]File: {filename}[/dim]")

    def found_functions(self, function_names: list[str]) -> None:
        """Log the functions found during parsing.

        Args:
            function_names: List of function names found.
        """
        if self.verbose:
            names = ", ".join(function_names) if function_names else "(none)"
            self.console.print(
                f"      [green]✓[/green] Found {len(function_names)} functions: {names}"
            )

    def detected_agent_type(
        self, agent_type: str, compile_line: int | None = None
    ) -> None:
        """Log the detected agent type.

        Args:
            agent_type: Either "langgraph" or "native".
            compile_line: Line number of compile() call for LangGraph agents.
        """
        type_display = (
            "LangGraph-based agent"
            if agent_type == "langgraph"
            else "native Python agent"
        )
        self.console.print(f"      [green]✓[/green] Detected: {type_display}")
        if compile_line and self.verbose:
            self.console.print(
                f"      [green]✓[/green] Compile location: line {compile_line}"
            )

    def step_classifying(self) -> None:
        """Log the start of classification step."""
        self._current_step = 2
        self.console.print()
        self._print_step("Classifying instruction types...")

    def show_classifications(self, results: list[ClassificationResult]) -> None:
        """Display classification results in a table.

        Args:
            results: List of classification results for each function.
        """
        if not results:
            self.console.print("      [dim](no functions to classify)[/dim]")
            return

        table = Table(show_header=True, header_style="bold", box=None, padding=(0, 2))
        table.add_column("Function", style="cyan")
        table.add_column("Instruction", style="yellow")
        table.add_column("Core", style="magenta")
        table.add_column("Confidence", justify="right")

        for result in results:
            confidence_pct = f"{result.confidence * 100:.0f}%"
            confidence_style = (
                "green"
                if result.confidence >= 0.8
                else "yellow"
                if result.confidence >= 0.6
                else "red"
            )
            table.add_row(
                result.function_name,
                result.instruction_type,
                result.core,
                f"[{confidence_style}]{confidence_pct}[/{confidence_style}]",
            )

        self.console.print()
        self.console.print(table)
        self.console.print()

    def step_confirmation(self) -> None:
        """Log the confirmation step."""
        self._current_step = 3
        self._print_step("User confirmation...")

    def prompt_confirmation(self) -> str:
        """Prompt user to confirm classifications.

        Returns:
            User's choice: 'y' for yes, 'n' for no, 'e' for edit.
        """
        self.console.print(
            "      Accept classifications? [Y]es / [N]o / [E]dit: ", end=""
        )
        try:
            response = input().strip().lower()
            return response if response in ("y", "n", "e", "yes", "no", "edit") else "y"
        except EOFError:
            return "y"

    def prompt_manual_classification(
        self, function_name: str, options: list[str]
    ) -> str:
        """Prompt user to manually select instruction type.

        Args:
            function_name: Name of the function to classify.
            options: List of available instruction types.

        Returns:
            Selected instruction type.
        """
        self.console.print(
            f"\n      [bold]Select instruction type for '{function_name}':[/bold]"
        )
        for i, option in enumerate(options, 1):
            self.console.print(f"        {i}. {option}")
        self.console.print("      Enter number: ", end="")
        try:
            choice = int(input().strip())
            if 1 <= choice <= len(options):
                return options[choice - 1]
        except (ValueError, EOFError):
            pass
        return options[0]  # Default to first option

    def step_backup(self, backup_path: str) -> None:
        """Log the backup step.

        Args:
            backup_path: Path where backup was saved.
        """
        self._current_step = 4
        self._print_step("Creating backup...")
        self.console.print(f"      [green]✓[/green] Saved: {backup_path}")

    def step_transforming(self) -> None:
        """Log the start of transformation step."""
        self._current_step = 5
        self.console.print()
        self._print_step("Transforming code...")

    def transformation_action(self, message: str) -> None:
        """Log a single transformation action.

        Args:
            message: Description of the action taken.
        """
        self.console.print(f"      [green]✓[/green] {message}")

    def complete(self, modified_file: str, backup_file: str) -> None:
        """Display completion message.

        Args:
            modified_file: Path to the modified file.
            backup_file: Path to the backup file.
        """
        self.console.print()
        self.console.print(
            Panel(
                f"[green]✓ Migration complete![/green]\n"
                f"[dim]Modified: {modified_file}[/dim]\n"
                f"[dim]Backup: {backup_file}[/dim]",
                border_style="green",
            )
        )
        self.console.print()

    def dry_run_complete(self, changes: list[str]) -> None:
        """Display dry-run completion message.

        Args:
            changes: List of changes that would be made.
        """
        self.console.print()
        self.console.print(
            Panel(
                "[yellow]Dry run complete - no files modified[/yellow]\n"
                "[dim]Changes that would be made:[/dim]",
                border_style="yellow",
            )
        )
        for change in changes:
            self.console.print(f"  • {change}")
        self.console.print()

    def error(self, message: str) -> None:
        """Display an error message.

        Args:
            message: The error message to display.
        """
        self.console.print(f"\n[red]✗ Error: {message}[/red]\n")

    def warning(self, message: str) -> None:
        """Display a warning message.

        Args:
            message: The warning message to display.
        """
        self.console.print(f"      [yellow]⚠ {message}[/yellow]")

    def info(self, message: str) -> None:
        """Display an info message.

        Args:
            message: The info message to display.
        """
        if self.verbose:
            self.console.print(f"      [dim]{message}[/dim]")

    def _print_step(self, message: str) -> None:
        """Print a step header.

        Args:
            message: The step message to display.
        """
        self.console.print(
            f"[bold][{self._current_step}/{self._total_steps}][/bold] {message}"
        )
