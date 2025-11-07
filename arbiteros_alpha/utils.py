"""Utility functions for ArbiterOS-alpha.

This module contains helper functions for formatting and displaying
execution history and other utilities.
"""

import yaml
from rich.console import Console

from .core import History


def print_history(history: list[History]) -> None:
    """Print the execution history in a readable format.

    Args:
        history: List of History entries to display.

    Example:
        >>> from arbiteros_alpha import ArbiterOSAlpha
        >>> from arbiteros_alpha.utils import print_history
        >>> os = ArbiterOSAlpha()
        >>> # ... execute some instructions ...
        >>> print_history(os.history)
    """
    console = Console()
    console.print("\n[bold cyan]📋 Arbiter OS Execution History[/bold cyan]")
    console.print("=" * 80)

    for i, entry in enumerate(history, 1):
        # Format policy results
        check_results = entry.check_policy_results
        route_results = entry.route_policy_results

        # Header with instruction name
        console.print(f"\n[bold cyan][{i}] {entry.instruction.name}[/bold cyan]")
        console.print(f"[dim]  Timestamp: {entry.timestamp}[/dim]")

        # Format input state as YAML
        console.print("  [yellow]Input:[/yellow]")
        input_yaml = yaml.dump(
            entry.input_state, default_flow_style=False, sort_keys=False
        )
        for line in input_yaml.strip().split("\n"):
            console.print(f"    [dim]{line}[/dim]")

        # Format output state as YAML
        console.print("  [yellow]Output:[/yellow]")
        output_yaml = yaml.dump(
            entry.output_state, default_flow_style=False, sort_keys=False
        )
        for line in output_yaml.strip().split("\n"):
            console.print(f"    [dim]{line}[/dim]")

        # Show detailed policy check results
        console.print("  [yellow]Policy Checks:[/yellow]")
        if check_results:
            for policy_name, result in check_results.items():
                status = "[green]✓[/green]" if result else "[red]✗[/red]"
                console.print(f"    {status} {policy_name}")
        else:
            console.print("    [dim](none)[/dim]")

        # Show detailed policy route results
        console.print("  [yellow]Policy Routes:[/yellow]")
        if route_results:
            for policy_name, destination in route_results.items():
                if destination:
                    console.print(
                        f"    [magenta]→[/magenta] {policy_name} [bold magenta]⇒ {destination}[/bold magenta]"
                    )
                else:
                    console.print(f"    [dim]— {policy_name}[/dim]")
        else:
            console.print("    [dim](none)[/dim]")

    console.print("\n" + "=" * 80 + "\n")
