"""CLI interface for ArbiterOS migration tool.

This module provides a Click-based command-line interface for migrating
existing agents into ArbiterOS-governed agents.

Usage:
    uv run -m arbiteros_alpha.migrator path/to/agent.py
    uv run -m arbiteros_alpha.migrator path/to/agent.py --manual
    uv run -m arbiteros_alpha.migrator path/to/agent.py --yes
    uv run -m arbiteros_alpha.migrator path/to/agent.py --dry-run
"""

import sys
from pathlib import Path
from typing import Literal

import click
from rich.console import Console

from .classifier import (
    ALL_INSTRUCTION_TYPES,
    ClassificationConfig,
    InstructionClassifier,
    NodeClassification,
)
from .generator import CodeGenerator
from .logger import ClassificationResult, MigrationLogger
from .parser import AgentParser


@click.command()
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
def main(
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
    """Migrate an agent file to use ArbiterOS governance.

    FILE_PATH is the path to the Python agent file to migrate.

    Examples:

        # Auto-detect agent type and use LLM classification
        uv run -m arbiteros_alpha.migrator my_agent.py

        # Manual classification (interactive prompts)
        uv run -m arbiteros_alpha.migrator my_agent.py --manual

        # Non-interactive with LLM
        uv run -m arbiteros_alpha.migrator my_agent.py --yes

        # Preview changes without modifying
        uv run -m arbiteros_alpha.migrator my_agent.py --dry-run
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
            logger.warning("No node functions found to migrate.")
            if not parsed.functions:
                logger.error("No functions found in file.")
                sys.exit(1)
            # Fall back to all functions with state param
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


if __name__ == "__main__":
    main()
