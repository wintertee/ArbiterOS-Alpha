"""Gradio-based UI dashboard for ArbiterOS checkpoint visualization.

This module provides a web-based dashboard for visualizing and managing
execution checkpoints, enabling time-travel functionality through an
intuitive user interface.
"""

import json
import logging
from typing import TYPE_CHECKING, Any

import gradio as gr
import yaml

if TYPE_CHECKING:
    from .core import ArbiterOSAlpha

logger = logging.getLogger(__name__)

# Custom CSS for a modern, clean look similar to Langfuse
CUSTOM_CSS = """
/* Main container styling */
.gradio-container {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
    max-width: 1400px !important;
    margin: 0 auto !important;
}

/* Header styling */
.header-title {
    font-size: 1.75rem !important;
    font-weight: 600 !important;
    color: #1a1a2e !important;
    margin-bottom: 0.5rem !important;
}

.header-subtitle {
    font-size: 0.95rem !important;
    color: #6b7280 !important;
    margin-bottom: 1.5rem !important;
}

/* Card styling */
.checkpoint-card {
    background: #ffffff !important;
    border: 1px solid #e5e7eb !important;
    border-radius: 12px !important;
    padding: 1.25rem !important;
    margin-bottom: 1rem !important;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05) !important;
}

/* Table styling */
.checkpoint-table {
    border-radius: 8px !important;
    overflow: hidden !important;
}

.checkpoint-table table {
    width: 100% !important;
    border-collapse: collapse !important;
}

.checkpoint-table th {
    background: #f8fafc !important;
    color: #374151 !important;
    font-weight: 500 !important;
    font-size: 0.875rem !important;
    padding: 0.875rem 1rem !important;
    text-align: left !important;
    border-bottom: 1px solid #e5e7eb !important;
}

.checkpoint-table td {
    padding: 0.875rem 1rem !important;
    border-bottom: 1px solid #f3f4f6 !important;
    font-size: 0.875rem !important;
    color: #4b5563 !important;
}

.checkpoint-table tr:hover td {
    background: #f9fafb !important;
}

/* State viewer styling */
.state-viewer {
    background: #1e1e2e !important;
    border-radius: 8px !important;
    padding: 1rem !important;
    font-family: 'JetBrains Mono', 'Fira Code', monospace !important;
    font-size: 0.85rem !important;
    overflow-x: auto !important;
}

/* Button styling */
.action-button {
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.625rem 1.25rem !important;
    font-weight: 500 !important;
    cursor: pointer !important;
    transition: all 0.2s ease !important;
}

.action-button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 12px rgba(99, 102, 241, 0.35) !important;
}

.secondary-button {
    background: #f3f4f6 !important;
    color: #374151 !important;
    border: 1px solid #e5e7eb !important;
}

.secondary-button:hover {
    background: #e5e7eb !important;
}

/* Badge styling */
.badge {
    display: inline-block !important;
    padding: 0.25rem 0.625rem !important;
    border-radius: 9999px !important;
    font-size: 0.75rem !important;
    font-weight: 500 !important;
}

.badge-fork {
    background: #fef3c7 !important;
    color: #92400e !important;
}

.badge-normal {
    background: #dbeafe !important;
    color: #1e40af !important;
}

/* Status indicator */
.status-dot {
    display: inline-block !important;
    width: 8px !important;
    height: 8px !important;
    border-radius: 50% !important;
    margin-right: 0.5rem !important;
}

.status-active {
    background: #10b981 !important;
}

.status-inactive {
    background: #9ca3af !important;
}

/* Timestamp styling */
.timestamp {
    color: #6b7280 !important;
    font-size: 0.8125rem !important;
}

/* Node name styling */
.node-name {
    font-weight: 500 !important;
    color: #1f2937 !important;
}

/* Checkpoint ID styling */
.checkpoint-id {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.8125rem !important;
    color: #6366f1 !important;
    background: #eef2ff !important;
    padding: 0.25rem 0.5rem !important;
    border-radius: 4px !important;
}

/* Section headers */
.section-header {
    font-size: 1rem !important;
    font-weight: 600 !important;
    color: #374151 !important;
    margin-bottom: 0.75rem !important;
    display: flex !important;
    align-items: center !important;
    gap: 0.5rem !important;
}

/* Comparison view */
.comparison-container {
    display: grid !important;
    grid-template-columns: 1fr 1fr !important;
    gap: 1rem !important;
}

/* Info panel */
.info-panel {
    background: #f0fdf4 !important;
    border: 1px solid #86efac !important;
    border-radius: 8px !important;
    padding: 1rem !important;
    margin-bottom: 1rem !important;
}

.info-panel-warning {
    background: #fef3c7 !important;
    border: 1px solid #fcd34d !important;
}
"""


def format_state_yaml(state: dict[str, Any] | None) -> str:
    """Format state dictionary as YAML for display.

    Args:
        state: The state dictionary to format.

    Returns:
        YAML formatted string.
    """
    if state is None:
        return "No state available"
    try:
        return yaml.dump(state, default_flow_style=False, sort_keys=False, indent=2)
    except Exception:
        return json.dumps(state, indent=2)


def format_state_json(state: dict[str, Any] | None) -> str:
    """Format state dictionary as JSON for display.

    Args:
        state: The state dictionary to format.

    Returns:
        JSON formatted string.
    """
    if state is None:
        return "{}"
    try:
        return json.dumps(state, indent=2)
    except Exception:
        return str(state)


class ArbiterOSDashboard:
    """Gradio-based dashboard for ArbiterOS checkpoint visualization.

    This class creates an interactive web dashboard for viewing and managing
    execution checkpoints, similar to Langfuse's audit log interface.

    Attributes:
        arbiter_os: The ArbiterOSAlpha instance to visualize.
        selected_checkpoint_id: Currently selected checkpoint ID.

    Example:
        >>> os = ArbiterOSAlpha(enable_checkpoints=True)
        >>> # ... run some instructions ...
        >>> dashboard = ArbiterOSDashboard(os)
        >>> dashboard.launch()
    """

    def __init__(self, arbiter_os: "ArbiterOSAlpha"):
        """Initialize the dashboard.

        Args:
            arbiter_os: The ArbiterOSAlpha instance to visualize.
        """
        self.arbiter_os = arbiter_os
        self.selected_checkpoint_id: str | None = None
        self._demo: gr.Blocks | None = None

    def get_checkpoint_table_data(self) -> list[list[str]]:
        """Get checkpoint data formatted for the Gradio dataframe.

        Returns:
            List of rows for the checkpoint table.
        """
        if self.arbiter_os.checkpoint_manager is None:
            return []

        table_data = self.arbiter_os.checkpoint_manager.to_table_data()
        rows = []
        for row in table_data:
            is_fork = "Fork" if row.get("is_fork") else "Execute"
            rows.append([
                row["time"],
                row["node"],
                row["checkpoint_id_short"],
                row["checkpoint_id"],  # Hidden full ID for selection
                row.get("next_nodes", "(end)"),
                is_fork,
            ])
        return rows

    def get_state_before(self, checkpoint_id: str) -> str:
        """Get the input state for a checkpoint.

        Args:
            checkpoint_id: The checkpoint ID to retrieve.

        Returns:
            YAML formatted input state.
        """
        if self.arbiter_os.checkpoint_manager is None:
            return "Checkpoints not enabled"

        entry = self.arbiter_os.checkpoint_manager.get_checkpoint_entry(checkpoint_id)
        if entry is None:
            return "Checkpoint not found"

        return format_state_yaml(entry.input_state)

    def get_state_after(self, checkpoint_id: str) -> str:
        """Get the output state for a checkpoint.

        Args:
            checkpoint_id: The checkpoint ID to retrieve.

        Returns:
            YAML formatted output state.
        """
        if self.arbiter_os.checkpoint_manager is None:
            return "Checkpoints not enabled"

        entry = self.arbiter_os.checkpoint_manager.get_checkpoint_entry(checkpoint_id)
        if entry is None:
            return "Checkpoint not found"

        return format_state_yaml(entry.output_state)

    def get_checkpoint_details(self, checkpoint_id: str) -> str:
        """Get detailed information about a checkpoint.

        Args:
            checkpoint_id: The checkpoint ID to retrieve.

        Returns:
            Formatted checkpoint details.
        """
        if self.arbiter_os.checkpoint_manager is None:
            return "Checkpoints not enabled"

        snapshot = self.arbiter_os.checkpoint_manager.get_checkpoint_by_id(checkpoint_id)
        if snapshot is None:
            return "Checkpoint not found"

        details = {
            "Checkpoint ID": snapshot.checkpoint_id,
            "Thread ID": snapshot.thread_id,
            "Node": snapshot.node_name,
            "Timestamp": snapshot.timestamp.isoformat(),
            "Next Nodes": list(snapshot.next_nodes) or ["(end)"],
            "Parent Checkpoint": snapshot.parent_checkpoint_id or "(none)",
            "Metadata": snapshot.metadata,
        }
        return format_state_yaml(details)

    def on_row_select(
        self, evt: gr.SelectData, table_data: list[list[str]]
    ) -> tuple[str, str, str, str]:
        """Handle row selection in the checkpoint table.

        Args:
            evt: Gradio selection event.
            table_data: Current table data.

        Returns:
            Tuple of (state_before, state_after, details, selected_id).
        """
        if evt.index is None or not table_data:
            return "Select a checkpoint", "Select a checkpoint", "Select a checkpoint", ""

        row_idx = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
        if row_idx >= len(table_data):
            return "Invalid selection", "Invalid selection", "Invalid selection", ""

        # Full checkpoint ID is in the 4th column (index 3)
        checkpoint_id = table_data[row_idx][3]
        self.selected_checkpoint_id = checkpoint_id

        state_before = self.get_state_before(checkpoint_id)
        state_after = self.get_state_after(checkpoint_id)
        details = self.get_checkpoint_details(checkpoint_id)

        return state_before, state_after, details, checkpoint_id

    def fork_checkpoint(
        self, checkpoint_id: str, new_state_json: str
    ) -> tuple[str, list[list[str]]]:
        """Fork from a checkpoint with modified state.

        Args:
            checkpoint_id: The checkpoint to fork from.
            new_state_json: JSON string of new state values.

        Returns:
            Tuple of (result message, updated table data).
        """
        if not checkpoint_id:
            return "Please select a checkpoint first", self.get_checkpoint_table_data()

        try:
            new_state = json.loads(new_state_json) if new_state_json.strip() else {}
        except json.JSONDecodeError as e:
            return f"Invalid JSON: {e}", self.get_checkpoint_table_data()

        try:
            thread_id, new_cp_id = self.arbiter_os.fork_from_checkpoint(
                checkpoint_id, new_state
            )
            return (
                f"Created fork: Thread {thread_id[:8]}..., Checkpoint {new_cp_id[:8]}...",
                self.get_checkpoint_table_data(),
            )
        except Exception as e:
            return f"Error: {e}", self.get_checkpoint_table_data()

    def update_checkpoint_state(
        self, checkpoint_id: str, new_state_json: str
    ) -> tuple[str, list[list[str]]]:
        """Update state at a checkpoint.

        Args:
            checkpoint_id: The checkpoint to update.
            new_state_json: JSON string of new state values.

        Returns:
            Tuple of (result message, updated table data).
        """
        if not checkpoint_id:
            return "Please select a checkpoint first", self.get_checkpoint_table_data()

        try:
            new_state = json.loads(new_state_json) if new_state_json.strip() else {}
        except json.JSONDecodeError as e:
            return f"Invalid JSON: {e}", self.get_checkpoint_table_data()

        try:
            new_cp_id = self.arbiter_os.update_state(checkpoint_id, new_state)
            return (
                f"Created updated checkpoint: {new_cp_id[:8]}...",
                self.get_checkpoint_table_data(),
            )
        except Exception as e:
            return f"Error: {e}", self.get_checkpoint_table_data()

    def refresh_table(self) -> list[list[str]]:
        """Refresh the checkpoint table data.

        Returns:
            Updated table data.
        """
        return self.get_checkpoint_table_data()

    def build(self) -> gr.Blocks:
        """Build the Gradio dashboard interface.

        Returns:
            The Gradio Blocks application.
        """
        with gr.Blocks(
            title="ArbiterOS Dashboard",
            css=CUSTOM_CSS,
            theme=gr.themes.Soft(
                primary_hue="indigo",
                secondary_hue="slate",
                neutral_hue="slate",
                font=gr.themes.GoogleFont("Inter"),
            ),
        ) as demo:
            # Header
            gr.Markdown(
                """
                <div style="padding: 1rem 0;">
                    <h1 class="header-title">ArbiterOS Execution History</h1>
                    <p class="header-subtitle">
                        Track execution checkpoints, view state changes, and perform time-travel operations.
                        Select a checkpoint to view details and perform actions.
                    </p>
                </div>
                """
            )

            # Main content
            with gr.Row():
                # Left panel - Checkpoint table
                with gr.Column(scale=3):
                    with gr.Group():
                        gr.Markdown("### Checkpoints")

                        # Refresh button
                        refresh_btn = gr.Button(
                            "Refresh",
                            variant="secondary",
                            size="sm",
                        )

                        # Checkpoint table
                        checkpoint_table = gr.Dataframe(
                            headers=[
                                "Time",
                                "Node",
                                "Checkpoint ID",
                                "Full ID",
                                "Next Nodes",
                                "Type",
                            ],
                            datatype=["str", "str", "str", "str", "str", "str"],
                            value=self.get_checkpoint_table_data(),
                            interactive=False,
                            col_count=(6, "fixed"),
                            wrap=True,
                        )

                # Right panel - Details and actions
                with gr.Column(scale=2):
                    # Selected checkpoint ID (hidden)
                    selected_id = gr.Textbox(
                        label="Selected Checkpoint",
                        interactive=False,
                        visible=True,
                    )

                    # State comparison
                    with gr.Tabs():
                        with gr.Tab("State Before"):
                            state_before = gr.Code(
                                label="Input State",
                                language="yaml",
                                value="Select a checkpoint to view state",
                                lines=12,
                            )

                        with gr.Tab("State After"):
                            state_after = gr.Code(
                                label="Output State",
                                language="yaml",
                                value="Select a checkpoint to view state",
                                lines=12,
                            )

                        with gr.Tab("Details"):
                            checkpoint_details = gr.Code(
                                label="Checkpoint Details",
                                language="yaml",
                                value="Select a checkpoint to view details",
                                lines=12,
                            )

            # Actions section
            gr.Markdown("### Time-Travel Actions")

            with gr.Row():
                with gr.Column():
                    gr.Markdown("**Update State**: Modify state values at the selected checkpoint")
                    new_state_input = gr.Textbox(
                        label="New State (JSON)",
                        placeholder='{"key": "new_value"}',
                        lines=3,
                    )

                with gr.Column():
                    with gr.Row():
                        update_btn = gr.Button(
                            "Update State",
                            variant="primary",
                        )
                        fork_btn = gr.Button(
                            "Fork Execution",
                            variant="secondary",
                        )

                    action_result = gr.Textbox(
                        label="Result",
                        interactive=False,
                    )

            # Branches view
            gr.Markdown("### Execution Branches")

            with gr.Row():
                branches_display = gr.JSON(
                    label="Active Branches",
                    value=self._get_branches_info(),
                )
                refresh_branches_btn = gr.Button("Refresh Branches", size="sm")

            # Event handlers
            checkpoint_table.select(
                fn=self.on_row_select,
                inputs=[checkpoint_table],
                outputs=[state_before, state_after, checkpoint_details, selected_id],
            )

            refresh_btn.click(
                fn=self.refresh_table,
                outputs=[checkpoint_table],
            )

            update_btn.click(
                fn=self.update_checkpoint_state,
                inputs=[selected_id, new_state_input],
                outputs=[action_result, checkpoint_table],
            )

            fork_btn.click(
                fn=self.fork_checkpoint,
                inputs=[selected_id, new_state_input],
                outputs=[action_result, checkpoint_table],
            )

            refresh_branches_btn.click(
                fn=self._get_branches_info,
                outputs=[branches_display],
            )

            self._demo = demo
            return demo

    def _get_branches_info(self) -> dict[str, Any]:
        """Get information about execution branches.

        Returns:
            Dictionary with branch information.
        """
        if self.arbiter_os.checkpoint_manager is None:
            return {"status": "Checkpoints not enabled"}

        branches = self.arbiter_os.checkpoint_manager.get_branches()
        info = {}
        for thread_id, checkpoint_ids in branches.items():
            info[f"Thread {thread_id[:8]}..."] = {
                "checkpoints": len(checkpoint_ids),
                "latest": checkpoint_ids[-1][:12] + "..." if checkpoint_ids else None,
            }
        return info

    def launch(self, **kwargs: Any) -> None:
        """Launch the Gradio dashboard.

        Args:
            **kwargs: Additional arguments to pass to gr.Blocks.launch().
        """
        if self._demo is None:
            self.build()

        default_kwargs = {
            "share": False,
            "server_name": "127.0.0.1",
            "server_port": 7860,
        }
        default_kwargs.update(kwargs)

        if self._demo is not None:
            self._demo.launch(**default_kwargs)


def create_dashboard(arbiter_os: "ArbiterOSAlpha") -> ArbiterOSDashboard:
    """Create and return an ArbiterOS dashboard instance.

    Args:
        arbiter_os: The ArbiterOSAlpha instance to visualize.

    Returns:
        The configured dashboard instance.

    Example:
        >>> os = ArbiterOSAlpha(enable_checkpoints=True)
        >>> dashboard = create_dashboard(os)
        >>> dashboard.launch()
    """
    return ArbiterOSDashboard(arbiter_os)


def launch_dashboard(
    arbiter_os: "ArbiterOSAlpha",
    share: bool = False,
    server_name: str = "127.0.0.1",
    server_port: int = 7860,
) -> None:
    """Convenience function to create and launch a dashboard.

    Args:
        arbiter_os: The ArbiterOSAlpha instance to visualize.
        share: Whether to create a public Gradio share link.
        server_name: Server hostname to bind to.
        server_port: Port to run the server on.

    Example:
        >>> os = ArbiterOSAlpha(enable_checkpoints=True)
        >>> launch_dashboard(os)  # Opens browser to http://127.0.0.1:7860
    """
    dashboard = create_dashboard(arbiter_os)
    dashboard.launch(share=share, server_name=server_name, server_port=server_port)

