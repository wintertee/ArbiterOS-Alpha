"""CLI interface for ArbiterOS migration tool.

This module provides a Click-based command-line interface for migrating
existing agents into ArbiterOS-governed agents.

Enhanced with:
- Repo-level transformation support
- Multi-file generation
- Interactive and non-interactive modes

Usage:
    # Single file migration (legacy)
    uv run -m arbiteros_alpha.migrator migrate path/to/agent.py
    uv run -m arbiteros_alpha.migrator migrate path/to/agent.py --manual
    uv run -m arbiteros_alpha.migrator migrate path/to/agent.py --yes

    # Repo-level transformation (new)
    uv run -m arbiteros_alpha.migrator transform /path/to/repo
    uv run -m arbiteros_alpha.migrator transform /path/to/repo --output ./policies
    uv run -m arbiteros_alpha.migrator transform /path/to/repo --dry-run
"""

import sys
from pathlib import Path
from typing import Literal

import click
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from .analyzer import AnalyzerConfig, RepoAnalyzer
from .classifier import (
    ALL_INSTRUCTION_TYPES,
    ClassificationConfig,
    InstructionClassifier,
    NodeClassification,
)
from .generator import CodeGenerator
from .logger import ClassificationResult, MigrationLogger
from .parser import AgentParser
from .policy_designer import PolicyDesigner, PolicyDesignerConfig
from .repo_scanner import RepoScanner
from .schema_designer import SchemaDesigner, SchemaDesignerConfig


@click.group()
def cli():
    """ArbiterOS Migration Tool - Transform agents to governed agents."""
    pass


@cli.command()
@click.argument("file_path", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--type",
    "-t",
    "agent_type",
    type=click.Choice(["langgraph", "native", "auto"]),
    default="auto",
    help="Agent type: langgraph, native, or auto-detect (default: auto)",
)
@click.option(
    "--manual",
    "-m",
    is_flag=True,
    default=False,
    help="Use manual classification instead of LLM",
)
@click.option(
    "--yes",
    "-y",
    is_flag=True,
    default=False,
    help="Skip confirmation prompts (non-interactive mode)",
)
@click.option(
    "--dry-run",
    "-n",
    is_flag=True,
    default=False,
    help="Show what would be done without modifying files",
)
@click.option(
    "--api-key",
    envvar="OPENAI_API_KEY",
    help="OpenAI API key (or set OPENAI_API_KEY env var)",
)
@click.option(
    "--base-url",
    envvar="OPENAI_BASE_URL",
    help="OpenAI API base URL (or set OPENAI_BASE_URL env var)",
)
@click.option(
    "--model",
    default="gpt-4o",
    help="Model to use for classification (default: gpt-4o)",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    default=True,
    help="Verbose output (default: True)",
)
def migrate(
    file_path: Path,
    agent_type: Literal["langgraph", "native", "auto"],
    manual: bool,
    yes: bool,
    dry_run: bool,
    api_key: str | None,
    base_url: str | None,
    model: str,
    verbose: bool,
) -> None:
    """Migrate a single agent file to use ArbiterOS governance.

    FILE_PATH is the path to the Python agent file to migrate.

    Examples:

        # Auto-detect agent type and use LLM classification
        uv run -m arbiteros_alpha.migrator migrate my_agent.py

        # Manual classification (interactive prompts)
        uv run -m arbiteros_alpha.migrator migrate my_agent.py --manual

        # Non-interactive with LLM
        uv run -m arbiteros_alpha.migrator migrate my_agent.py --yes

        # Preview changes without modifying
        uv run -m arbiteros_alpha.migrator migrate my_agent.py --dry-run
    """
    console = Console()
    logger = MigrationLogger(console=console, verbose=verbose)

    try:
        # Start
        logger.start()

        # Step 1: Parse
        logger.step_parsing(str(file_path))
        parser = AgentParser()
        parsed = parser.parse_file(file_path)

        # Report findings
        func_names = [f.name for f in parsed.functions]
        logger.found_functions(func_names)

        # Override agent type if specified
        if agent_type != "auto":
            parsed.agent_type = agent_type  # type: ignore

        logger.detected_agent_type(parsed.agent_type, parsed.compile_lineno)

        # Check if already transformed
        if parsed.has_existing_arbiteros:
            logger.warning(
                "File already has ArbiterOS imports. Skipping import addition."
            )

        # Filter to node functions only
        node_functions = [f for f in parsed.functions if f.is_node_function]
        if not node_functions:
            # If file already has ArbiterOS, likely all node functions are decorated
            # Don't fall back to state-param functions in this case
            if parsed.has_existing_arbiteros or parsed.has_os_initialization:
                logger.info(
                    "No undecorated node functions found. "
                    "File appears to be already migrated."
                )
                sys.exit(0)

            logger.warning("No node functions found to migrate.")
            if not parsed.functions:
                logger.error("No functions found in file.")
                sys.exit(1)
            # Fall back to all functions with state param (only for fresh files)
            node_functions = [f for f in parsed.functions if f.has_state_param]
            if not node_functions:
                logger.info("Using all functions as potential nodes.")
                node_functions = parsed.functions

        # Step 2: Classify
        logger.step_classifying()

        classifications: dict[str, NodeClassification] = {}

        if manual:
            # Manual classification
            for func in node_functions:
                instruction_type = logger.prompt_manual_classification(
                    func.name,
                    ALL_INSTRUCTION_TYPES,
                )
                classifications[func.name] = NodeClassification(
                    instruction_type=instruction_type,
                    core=InstructionClassifier.get_instruction_core(instruction_type),
                    confidence=1.0,
                    reasoning="Manually selected",
                )
        else:
            # LLM classification
            if not api_key:
                logger.error(
                    "No API key provided. Set OPENAI_API_KEY environment variable "
                    "or use --api-key option. Use --manual for manual classification."
                )
                sys.exit(1)

            config = ClassificationConfig(
                api_key=api_key,
                base_url=base_url,
                model=model,
            )
            classifier = InstructionClassifier(config=config)

            for func in node_functions:
                logger.info(f"Classifying {func.name}...")
                try:
                    result = classifier.classify(func)
                    classifications[func.name] = result
                except Exception as e:
                    logger.error(f"Failed to classify {func.name}: {e}")
                    sys.exit(1)

        # Show classifications
        results = [
            ClassificationResult(
                function_name=name,
                instruction_type=c.instruction_type,
                core=c.core,
                confidence=c.confidence,
                reasoning=c.reasoning,
            )
            for name, c in classifications.items()
        ]
        logger.show_classifications(results)

        # Step 3: Confirm
        if not yes and not dry_run:
            logger.step_confirmation()
            response = logger.prompt_confirmation()
            if response in ("n", "no"):
                logger.info("Migration cancelled.")
                sys.exit(0)
            elif response in ("e", "edit"):
                # Allow editing each classification
                for func in node_functions:
                    current = classifications.get(func.name)
                    if current:
                        console.print(
                            f"\n      Current: {func.name} -> {current.instruction_type}"
                        )
                        new_type = logger.prompt_manual_classification(
                            f"{func.name} (press Enter to keep {current.instruction_type})",
                            ["(keep current)"] + ALL_INSTRUCTION_TYPES,
                        )
                        if new_type != "(keep current)":
                            classifications[func.name] = NodeClassification(
                                instruction_type=new_type,
                                core=InstructionClassifier.get_instruction_core(
                                    new_type
                                ),
                                confidence=1.0,
                                reasoning="Manually edited",
                            )

        # Step 4 & 5: Generate
        generator = CodeGenerator()

        if dry_run:
            result = generator.transform(
                file_path=file_path,
                parsed_agent=parsed,
                classifications=classifications,
                dry_run=True,
            )
            logger.dry_run_complete(result.changes)
        else:
            # Create backup
            logger.step_backup("(creating...)")

            result = generator.transform(
                file_path=file_path,
                parsed_agent=parsed,
                classifications=classifications,
                dry_run=False,
            )

            if result.success:
                logger.step_transforming()
                for change in result.changes:
                    logger.transformation_action(change)
                logger.complete(result.modified_file, result.backup_file)
            else:
                logger.error(f"Migration failed: {result.error}")
                sys.exit(1)

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        sys.exit(1)
    except SyntaxError as e:
        logger.error(f"Invalid Python syntax: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("\nMigration cancelled.")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise


@cli.command()
@click.argument("repo_path", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--output",
    "-o",
    "output_dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Output directory for transformed repo (default: repo_arbiteros/)",
)
@click.option(
    "--suffix",
    "-s",
    default="_arbiteros",
    help="Suffix for transformed repo name (default: _arbiteros)",
)
@click.option(
    "--dry-run",
    "-n",
    is_flag=True,
    default=False,
    help="Preview generated files without writing",
)
@click.option(
    "--interactive",
    "-i",
    is_flag=True,
    default=False,
    help="Interactive mode with confirmations at each step",
)
@click.option(
    "--skip-policies",
    is_flag=True,
    default=False,
    help="Skip policy checker/router generation",
)
@click.option(
    "--skip-schemas",
    is_flag=True,
    default=False,
    help="Skip LLM I/O schema generation",
)
@click.option(
    "--api-key",
    envvar="OPENAI_API_KEY",
    help="OpenAI API key (or set OPENAI_API_KEY env var)",
)
@click.option(
    "--base-url",
    envvar="OPENAI_BASE_URL",
    help="OpenAI API base URL (or set OPENAI_BASE_URL env var)",
)
@click.option(
    "--model",
    default="gpt-4o",
    help="Model to use for LLM calls (default: gpt-4o)",
)
def transform(
    repo_path: Path,
    output_dir: Path | None,
    suffix: str,
    dry_run: bool,
    interactive: bool,
    skip_policies: bool,
    skip_schemas: bool,
    api_key: str | None,
    base_url: str | None,
    model: str,
) -> None:
    """Transform an entire repository to use ArbiterOS governance.

    REPO_PATH is the path to the repository root directory.

    This command:
    1. Creates a copy of the repository with _arbiteros suffix
    2. Scans the repository for agent files and graph definitions
    3. Uses LLM to analyze the domain and agent roles
    4. Classifies node functions into ACF instruction types
    5. Designs domain-specific policies
    6. Modifies agent files in place with ArbiterOS decorators
    7. Generates governance files (governed_agents.py, policy_checkers.py, etc.)

    Examples:

        # Basic transformation (creates repo_arbiteros/)
        uv run -m arbiteros_alpha.migrator transform /path/to/my-agents

        # Custom suffix
        uv run -m arbiteros_alpha.migrator transform /path/to/my-agents --suffix _governed

        # Specify output directory
        uv run -m arbiteros_alpha.migrator transform /path/to/my-agents -o ./my-agents-governed

        # Preview without writing files
        uv run -m arbiteros_alpha.migrator transform /path/to/my-agents --dry-run

        # Interactive mode
        uv run -m arbiteros_alpha.migrator transform /path/to/my-agents --interactive
    """
    console = Console()

    # Check API key
    if not api_key:
        console.print(
            "[red]Error: No API key provided. "
            "Set OPENAI_API_KEY environment variable or use --api-key option.[/red]"
        )
        sys.exit(1)

    # Set default output directory (copy of repo with suffix)
    if output_dir is None:
        repo_name = repo_path.name
        # Remove any trailing suffixes like -main
        base_name = repo_name.rstrip("-main").rstrip("_main")
        output_dir = repo_path.parent / f"{base_name}{suffix}"

    # Header
    console.print()
    console.print(
        Panel.fit(
            "[bold cyan]ArbiterOS Repository Transformation Tool[/bold cyan]",
            border_style="cyan",
        )
    )
    console.print()
    console.print(f"[bold]Source Repository:[/bold] {repo_path}")
    console.print(f"[bold]Output Repository:[/bold] {output_dir}")
    console.print()

    try:
        # Step 1: Scan Repository
        console.print("[bold][1/5][/bold] Scanning repository...")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Scanning files...", total=None)
            scanner = RepoScanner()
            scan_result = scanner.scan(repo_path)
            progress.update(task, completed=True)

        console.print(
            f"      [green]✓[/green] Found {len(scan_result.python_files)} Python files"
        )
        console.print(
            f"      [green]✓[/green] Found {len(scan_result.functions)} functions"
        )
        console.print(
            f"      [green]✓[/green] Found {len(scan_result.graph_nodes)} graph nodes"
        )
        console.print(
            f"      [green]✓[/green] Detected framework: {scan_result.detected_framework.value}"
        )
        console.print()

        if interactive:
            if not click.confirm("Continue with analysis?", default=True):
                console.print("[yellow]Transformation cancelled.[/yellow]")
                sys.exit(0)

        # Step 2: Analyze Repository
        console.print("[bold][2/5][/bold] Analyzing repository with LLM...")

        analyzer_config = AnalyzerConfig(
            api_key=api_key,
            base_url=base_url,
            model=model,
        )

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Analyzing domain and agent roles...", total=None)
            analyzer = RepoAnalyzer(config=analyzer_config)
            analysis = analyzer.analyze(scan_result)
            progress.update(task, completed=True)

        console.print(f"      [green]✓[/green] Domain: {analysis.domain}")
        console.print(
            f"      [green]✓[/green] Agent roles: {len(analysis.agent_roles)}"
        )
        console.print(
            f"      [green]✓[/green] Workflow stages: {len(analysis.workflow_stages)}"
        )

        # Show agent roles table
        if analysis.agent_roles:
            console.print()
            table = Table(title="Agent Roles", box=None, padding=(0, 2))
            table.add_column("Role", style="cyan")
            table.add_column("Instruction", style="yellow")
            table.add_column("Description", style="dim")

            for role in analysis.agent_roles:
                table.add_row(
                    role.role_name,
                    role.suggested_instruction.value,
                    role.description[:50] + "..."
                    if len(role.description) > 50
                    else role.description,
                )

            console.print(table)

        console.print()

        if interactive:
            if not click.confirm("Continue with classification?", default=True):
                console.print("[yellow]Transformation cancelled.[/yellow]")
                sys.exit(0)

        # Step 3: Classify Functions
        console.print("[bold][3/5][/bold] Classifying node functions...")

        classifier_config = ClassificationConfig(
            api_key=api_key,
            base_url=base_url,
            model=model,
        )

        # Filter to relevant functions (node functions or functions in graph)
        functions_to_classify = [
            f
            for f in scan_result.functions
            if f.is_factory
            or any(n.function_name == f.name for n in scan_result.graph_nodes)
        ]

        # If no functions matched, use functions from agent roles
        if not functions_to_classify:
            role_functions = set()
            for role in analysis.agent_roles:
                role_functions.update(role.functions)
            functions_to_classify = [
                f for f in scan_result.functions if f.name in role_functions
            ]

        # If still no functions, use all functions
        if not functions_to_classify:
            functions_to_classify = scan_result.functions[:50]  # Limit to 50

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task(
                f"Classifying {len(functions_to_classify)} functions...", total=None
            )
            classifier = InstructionClassifier(config=classifier_config)
            classifications = classifier.classify_batch_with_context(
                functions_to_classify, analysis
            )
            progress.update(task, completed=True)

        console.print(
            f"      [green]✓[/green] Classified {len(classifications.classifications)} functions"
        )

        # Show classifications table
        if classifications.classifications:
            console.print()
            table = Table(title="Classifications", box=None, padding=(0, 2))
            table.add_column("Function", style="cyan")
            table.add_column("Instruction", style="yellow")
            table.add_column("Wrapper", style="magenta")
            table.add_column("Confidence", justify="right")

            for c in classifications.classifications[:20]:  # Show first 20
                conf_pct = f"{c.confidence * 100:.0f}%"
                conf_style = (
                    "green"
                    if c.confidence >= 0.8
                    else "yellow"
                    if c.confidence >= 0.6
                    else "red"
                )
                table.add_row(
                    c.function_name,
                    c.instruction_type.value,
                    c.wrapper_name,
                    f"[{conf_style}]{conf_pct}[/{conf_style}]",
                )

            if len(classifications.classifications) > 20:
                table.add_row(
                    f"... and {len(classifications.classifications) - 20} more",
                    "",
                    "",
                    "",
                )

            console.print(table)

        console.print()

        if interactive:
            if not click.confirm("Continue with policy design?", default=True):
                console.print("[yellow]Transformation cancelled.[/yellow]")
                sys.exit(0)

        # Step 4: Design Policies (optional)
        policy_design = None
        if not skip_policies:
            console.print("[bold][4/5][/bold] Designing policies...")

            policy_config = PolicyDesignerConfig(
                api_key=api_key,
                base_url=base_url,
                model=model,
            )

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task(
                    "Designing policy checkers and routers...", total=None
                )
                designer = PolicyDesigner(config=policy_config)
                policy_design = designer.design(analysis, classifications)
                progress.update(task, completed=True)

            console.print(
                f"      [green]✓[/green] Designed {len(policy_design.checkers)} checkers"
            )
            console.print(
                f"      [green]✓[/green] Designed {len(policy_design.routers)} routers"
            )
            console.print()
        else:
            console.print("[bold][4/5][/bold] Skipping policy design")
            console.print()
            # Create empty policy design
            from .schemas import PolicyDesignOutput

            policy_design = PolicyDesignOutput(
                checkers=[],
                routers=[],
                yaml_checkers=[],
                yaml_routers=[],
                design_rationale="Policy design skipped by user",
            )

        # Step 5: Design Schemas (optional)
        schema_design = None
        if not skip_schemas:
            console.print("[bold][5/5][/bold] Designing LLM I/O schemas...")

            schema_config = SchemaDesignerConfig(
                api_key=api_key,
                base_url=base_url,
                model=model,
            )

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task("Designing Pydantic schemas...", total=None)
                designer = SchemaDesigner(config=schema_config)
                # Pass policy_design to schema designer for policy-aware schema enrichment
                schema_design = designer.design(
                    analysis, functions_to_classify, policy_design
                )
                progress.update(task, completed=True)

            console.print(
                f"      [green]✓[/green] Designed {len(schema_design.schemas)} schemas"
            )
            console.print()
        else:
            console.print("[bold][5/5][/bold] Skipping schema design")
            console.print()
            # Create empty schema design
            from .schemas import LLMSchemaDesignOutput

            schema_design = LLMSchemaDesignOutput(
                schemas=[],
                imports_needed=[],
                design_rationale="Schema design skipped by user",
            )

        # Transform Repository
        console.print("[bold][6/6][/bold] Transforming repository...")

        generator = CodeGenerator()
        result = generator.transform_repository(
            source_repo=repo_path,
            output_repo=output_dir,
            analysis=analysis,
            classifications=classifications,
            policy_design=policy_design,
            schema_design=schema_design,
            dry_run=dry_run,
        )

        # Show results
        if result.success:
            console.print()
            if dry_run:
                console.print(
                    Panel(
                        "[yellow]Dry run complete - no files written[/yellow]\n"
                        "[dim]Files that would be generated/modified:[/dim]",
                        border_style="yellow",
                    )
                )
            else:
                console.print(
                    Panel(
                        "[green]✓ Transformation complete![/green]\n"
                        f"[dim]Output repository: {output_dir}[/dim]",
                        border_style="green",
                    )
                )

            # Show modified files
            if result.modified_files:
                console.print("\n[bold]Modified files with decorators:[/bold]")
                for mf in result.modified_files:
                    console.print(f"  📝 {mf}")

            # Show generated files
            if result.generated_files:
                console.print("\n[bold]Generated governance files:[/bold]")
                for gf in result.generated_files:
                    console.print(f"  📄 {gf.path}: {gf.description}")

            if result.warnings:
                console.print()
                console.print("[yellow]Warnings:[/yellow]")
                for warning in result.warnings:
                    console.print(f"  ⚠ {warning}")

            console.print()
        else:
            console.print()
            console.print(
                Panel(
                    "[red]✗ Transformation failed[/red]",
                    border_style="red",
                )
            )
            for error in result.errors:
                console.print(f"  • {error}")
            console.print()
            sys.exit(1)

    except KeyboardInterrupt:
        console.print("\n[yellow]Transformation cancelled.[/yellow]")
        sys.exit(130)
    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]")
        raise


# Legacy compatibility - allow calling without subcommand for single file
@cli.command(name="main", hidden=True)
@click.argument("file_path", type=click.Path(exists=True, path_type=Path))
@click.pass_context
def main_compat(ctx, file_path):
    """Legacy compatibility - redirects to migrate command."""
    ctx.invoke(migrate, file_path=file_path)


def main():
    """Entry point for the CLI."""
    cli()


if __name__ == "__main__":
    main()
