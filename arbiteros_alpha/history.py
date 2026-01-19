import datetime
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from .instructions import InstructionType

if TYPE_CHECKING:
    from .evaluation import EvaluationResult

logger = logging.getLogger(__name__)


@dataclass
class HistoryItem:
    """The minimal OS metadata for tracking instruction execution.

    Attributes:
        timestamp: When the instruction was executed.
        instruction: The instruction type that was executed.
        input_state: The state passed to the instruction.
        output_state: The state returned by the instruction.
        check_policy_results: Results of policy checkers (name -> passed/failed).
        route_policy_results: Results of policy routers (name -> target or None).
        evaluation_results: Results of node evaluators (name -> EvaluationResult).
    """

    timestamp: datetime.datetime
    instruction: InstructionType
    input_state: dict[str, Any]
    output_state: Any = field(default_factory=dict)
    check_policy_results: dict[str, bool] = field(default_factory=dict)
    route_policy_results: dict[str, str | None] = field(default_factory=dict)
    evaluation_results: dict[str, "EvaluationResult"] = field(default_factory=dict)


SuperStep = list[HistoryItem]


class History:
    def __init__(self) -> None:
        """Initialize an empty execution history."""
        self.entries: list[SuperStep] = []
        self.next_superstep: list[str] = []

    def enter_next_superstep(self, nodes: list[str]) -> None:
        if "__start__" in nodes or "__end__" in nodes:
            return
        logger.debug(f"Entering next superstep with nodes: {nodes}")
        self.next_superstep = nodes
        self.entries.append([])

    def add_entry(self, entry: HistoryItem) -> None:
        if not self.entries or len(self.entries[-1]) >= len(self.next_superstep):
            raise RuntimeError(
                "All nodes for the current superstep have already recorded entries.\n"
                "Hint: Did you forget to call \n"
                "    - register_compiled_graph() for langgraph backend or \n"
                "    - enter_next_superstep() for vanilla backend?"
            )
        self.entries[-1].append(entry)

    def pprint(self) -> None:
        import yaml
        from rich.console import Console

        console = Console()
        console.print("\n[bold cyan]📋 Arbiter OS Execution History[/bold cyan]")
        console.print("=" * 80)

        for superstep_idx, superstep in enumerate(self.entries, 1):
            console.print(
                f"\n[bold magenta]╔═══ SuperStep {superstep_idx} ═══╗[/bold magenta]"
            )

            for entry_idx, entry in enumerate(superstep, 1):
                # Format policy results
                check_results = entry.check_policy_results
                route_results = entry.route_policy_results

                # Header with instruction name
                console.print(
                    f"\n[bold cyan]  [{superstep_idx}.{entry_idx}] {entry.instruction.name}[/bold cyan]"
                )
                console.print(f"[dim]    Timestamp: {entry.timestamp}[/dim]")

                # Format input state as YAML
                console.print("    [yellow]Input:[/yellow]")
                input_yaml = yaml.dump(
                    entry.input_state, default_flow_style=False, sort_keys=False
                )
                for line in input_yaml.strip().split("\n"):
                    console.print(f"      [dim]{line}[/dim]")

                # Format output state as YAML
                console.print("    [yellow]Output:[/yellow]")
                output_yaml = yaml.dump(
                    entry.output_state, default_flow_style=False, sort_keys=False
                )
                for line in output_yaml.strip().split("\n"):
                    console.print(f"      [dim]{line}[/dim]")

                # Show detailed policy check results
                console.print("    [yellow]Policy Checks:[/yellow]")
                if check_results:
                    for policy_name, result in check_results.items():
                        status = "[green]✓[/green]" if result else "[red]✗[/red]"
                        console.print(f"      {status} {policy_name}")
                else:
                    console.print("      [dim](none)[/dim]")

                # Show detailed policy route results
                console.print("    [yellow]Policy Routes:[/yellow]")
                if route_results:
                    for policy_name, destination in route_results.items():
                        if destination:
                            console.print(
                                f"      [magenta]→[/magenta] {policy_name} [bold magenta]⇒ {destination}[/bold magenta]"
                            )
                        else:
                            console.print(f"      [dim]— {policy_name}[/dim]")
                else:
                    console.print("      [dim](none)[/dim]")

                # Show evaluation results
                console.print("    [yellow]Evaluations:[/yellow]")
                eval_results = entry.evaluation_results
                if eval_results:
                    for eval_name, eval_result in eval_results.items():
                        status = (
                            "[green]✓[/green]" if eval_result.passed else "[red]✗[/red]"
                        )
                        console.print(
                            f"      {status} {eval_name}: "
                            f"[cyan]score={eval_result.score:.2f}[/cyan] - {eval_result.feedback}"
                        )
                else:
                    console.print("      [dim](none)[/dim]")

            console.print(
                f"[bold magenta]╚{'═' * (len(f'SuperStep {superstep_idx}') + 9)}╝[/bold magenta]"
            )

        console.print("\n" + "=" * 80 + "\n")
