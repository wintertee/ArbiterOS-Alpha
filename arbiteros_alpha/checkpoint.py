"""Checkpoint management for ArbiterOS time-travel functionality.

This module provides checkpoint management capabilities that wrap LangGraph's
InMemorySaver to enable time-travel features like state history retrieval,
state updates at checkpoints, and execution forking.
"""

import copy
import datetime
import logging
import uuid
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class StateSnapshot:
    """Represents a snapshot of execution state at a specific checkpoint.

    Attributes:
        checkpoint_id: Unique identifier for this checkpoint.
        thread_id: The thread this checkpoint belongs to.
        timestamp: When the checkpoint was created.
        node_name: The name of the node that was executed.
        values: The state values at this checkpoint.
        next_nodes: List of nodes scheduled to execute next.
        parent_checkpoint_id: ID of the parent checkpoint (for forks).
        metadata: Additional metadata about the checkpoint.
    """

    checkpoint_id: str
    thread_id: str
    timestamp: datetime.datetime
    node_name: str
    values: dict[str, Any]
    next_nodes: tuple[str, ...] = field(default_factory=tuple)
    parent_checkpoint_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CheckpointEntry:
    """Internal storage for a checkpoint with its full context.

    Attributes:
        snapshot: The state snapshot.
        input_state: State before the node executed.
        output_state: State after the node executed.
    """

    snapshot: StateSnapshot
    input_state: dict[str, Any]
    output_state: dict[str, Any]


class CheckpointManager:
    """Manages checkpoints for ArbiterOS execution history.

    This class provides a simplified checkpoint management layer that can work
    independently or alongside LangGraph's native checkpointing. It stores
    execution snapshots and enables time-travel operations.

    Attributes:
        thread_id: The current thread identifier.
        checkpoints: Dictionary mapping checkpoint IDs to their entries.

    Example:
        >>> manager = CheckpointManager()
        >>> checkpoint_id = manager.create_checkpoint(
        ...     node_name="generate",
        ...     input_state={"query": "hello"},
        ...     output_state={"query": "hello", "response": "world"}
        ... )
        >>> history = manager.get_state_history()
        >>> forked_id = manager.fork_from_checkpoint(checkpoint_id, {"query": "hi"})
    """

    def __init__(self, thread_id: str | None = None):
        """Initialize the CheckpointManager.

        Args:
            thread_id: Optional thread identifier. If not provided, a new
                UUID will be generated.
        """
        self.thread_id = thread_id or str(uuid.uuid4())
        self.checkpoints: dict[str, CheckpointEntry] = {}
        self._checkpoint_order: list[str] = []  # Maintains insertion order
        self._branches: dict[str, list[str]] = {
            self.thread_id: []
        }  # thread_id -> checkpoint_ids
        logger.debug(f"CheckpointManager initialized with thread_id: {self.thread_id}")

    def create_checkpoint(
        self,
        node_name: str,
        input_state: dict[str, Any],
        output_state: dict[str, Any],
        next_nodes: tuple[str, ...] = (),
        metadata: dict[str, Any] | None = None,
        parent_checkpoint_id: str | None = None,
    ) -> str:
        """Create a new checkpoint for the current execution state.

        Args:
            node_name: Name of the node that was executed.
            input_state: The state before node execution.
            output_state: The state after node execution.
            next_nodes: Tuple of nodes scheduled to execute next.
            metadata: Optional additional metadata.
            parent_checkpoint_id: Optional parent checkpoint for forks.

        Returns:
            The unique checkpoint ID for the created checkpoint.
        """
        checkpoint_id = str(uuid.uuid4())
        timestamp = datetime.datetime.now()

        # Determine the parent checkpoint if not explicitly provided
        if parent_checkpoint_id is None and self._checkpoint_order:
            parent_checkpoint_id = self._checkpoint_order[-1]

        snapshot = StateSnapshot(
            checkpoint_id=checkpoint_id,
            thread_id=self.thread_id,
            timestamp=timestamp,
            node_name=node_name,
            values=copy.deepcopy(output_state),
            next_nodes=next_nodes,
            parent_checkpoint_id=parent_checkpoint_id,
            metadata=metadata or {},
        )

        entry = CheckpointEntry(
            snapshot=snapshot,
            input_state=copy.deepcopy(input_state),
            output_state=copy.deepcopy(output_state),
        )

        self.checkpoints[checkpoint_id] = entry
        self._checkpoint_order.append(checkpoint_id)
        self._branches[self.thread_id].append(checkpoint_id)

        logger.debug(
            f"Created checkpoint {checkpoint_id[:8]}... for node '{node_name}'"
        )
        return checkpoint_id

    def get_state_history(self, thread_id: str | None = None) -> list[StateSnapshot]:
        """Retrieve all checkpoints for a thread in chronological order.

        Args:
            thread_id: Optional thread ID. Defaults to current thread.

        Returns:
            List of StateSnapshot objects in chronological order.
        """
        target_thread = thread_id or self.thread_id
        checkpoint_ids = self._branches.get(target_thread, [])

        snapshots = []
        for cp_id in checkpoint_ids:
            if cp_id in self.checkpoints:
                snapshots.append(self.checkpoints[cp_id].snapshot)

        # Sort by timestamp to ensure chronological order
        snapshots.sort(key=lambda s: s.timestamp)
        return snapshots

    def get_checkpoint_by_id(self, checkpoint_id: str) -> StateSnapshot | None:
        """Retrieve a specific checkpoint by its ID.

        Args:
            checkpoint_id: The unique checkpoint identifier.

        Returns:
            The StateSnapshot if found, None otherwise.
        """
        entry = self.checkpoints.get(checkpoint_id)
        return entry.snapshot if entry else None

    def get_checkpoint_entry(self, checkpoint_id: str) -> CheckpointEntry | None:
        """Retrieve the full checkpoint entry including input/output states.

        Args:
            checkpoint_id: The unique checkpoint identifier.

        Returns:
            The CheckpointEntry if found, None otherwise.
        """
        return self.checkpoints.get(checkpoint_id)

    def update_state(
        self,
        checkpoint_id: str,
        values: dict[str, Any],
    ) -> str:
        """Update state at a checkpoint, creating a new fork.

        This creates a new checkpoint with modified values, effectively
        forking the execution history from the specified checkpoint.

        Args:
            checkpoint_id: The checkpoint ID to fork from.
            values: New state values to apply.

        Returns:
            The new checkpoint ID for the forked state.

        Raises:
            ValueError: If the checkpoint_id doesn't exist.
        """
        original_entry = self.checkpoints.get(checkpoint_id)
        if original_entry is None:
            raise ValueError(f"Checkpoint {checkpoint_id} not found")

        # Merge the original output state with new values
        merged_state = copy.deepcopy(original_entry.output_state)
        merged_state.update(values)

        # Create a new checkpoint as a fork
        new_checkpoint_id = self.create_checkpoint(
            node_name=f"fork_from_{original_entry.snapshot.node_name}",
            input_state=original_entry.output_state,
            output_state=merged_state,
            next_nodes=original_entry.snapshot.next_nodes,
            metadata={"forked_from": checkpoint_id, "fork_type": "state_update"},
            parent_checkpoint_id=checkpoint_id,
        )

        logger.info(
            f"Created fork {new_checkpoint_id[:8]}... from checkpoint {checkpoint_id[:8]}..."
        )
        return new_checkpoint_id

    def fork_from_checkpoint(
        self,
        checkpoint_id: str,
        new_state: dict[str, Any] | None = None,
        new_thread_id: str | None = None,
    ) -> tuple[str, str]:
        """Create a new execution branch from an existing checkpoint.

        This is useful for parallel testing - you can fork from a checkpoint
        and run alternative execution paths.

        Args:
            checkpoint_id: The checkpoint ID to fork from.
            new_state: Optional new state values to apply at the fork point.
            new_thread_id: Optional new thread ID for the fork. If not provided,
                a new UUID will be generated.

        Returns:
            Tuple of (new_thread_id, new_checkpoint_id).

        Raises:
            ValueError: If the checkpoint_id doesn't exist.
        """
        original_entry = self.checkpoints.get(checkpoint_id)
        if original_entry is None:
            raise ValueError(f"Checkpoint {checkpoint_id} not found")

        # Create a new thread for the fork
        fork_thread_id = new_thread_id or str(uuid.uuid4())
        self._branches[fork_thread_id] = []

        # Determine the forked state
        forked_state = copy.deepcopy(original_entry.output_state)
        if new_state:
            forked_state.update(new_state)

        # Create the initial checkpoint for the new branch
        new_checkpoint_id = str(uuid.uuid4())
        timestamp = datetime.datetime.now()

        snapshot = StateSnapshot(
            checkpoint_id=new_checkpoint_id,
            thread_id=fork_thread_id,
            timestamp=timestamp,
            node_name="fork_point",
            values=forked_state,
            next_nodes=original_entry.snapshot.next_nodes,
            parent_checkpoint_id=checkpoint_id,
            metadata={
                "forked_from": checkpoint_id,
                "original_thread": self.thread_id,
                "fork_type": "branch",
            },
        )

        entry = CheckpointEntry(
            snapshot=snapshot,
            input_state=copy.deepcopy(original_entry.output_state),
            output_state=forked_state,
        )

        self.checkpoints[new_checkpoint_id] = entry
        self._branches[fork_thread_id].append(new_checkpoint_id)

        logger.info(
            f"Created new branch {fork_thread_id[:8]}... from checkpoint {checkpoint_id[:8]}..."
        )
        return fork_thread_id, new_checkpoint_id

    def get_branches(self) -> dict[str, list[str]]:
        """Get all execution branches and their checkpoint IDs.

        Returns:
            Dictionary mapping thread IDs to their checkpoint ID lists.
        """
        return copy.deepcopy(self._branches)

    def get_latest_checkpoint(
        self, thread_id: str | None = None
    ) -> StateSnapshot | None:
        """Get the most recent checkpoint for a thread.

        Args:
            thread_id: Optional thread ID. Defaults to current thread.

        Returns:
            The most recent StateSnapshot, or None if no checkpoints exist.
        """
        history = self.get_state_history(thread_id)
        return history[-1] if history else None

    def get_state_at_checkpoint(self, checkpoint_id: str) -> dict[str, Any] | None:
        """Get the state values at a specific checkpoint.

        Args:
            checkpoint_id: The checkpoint ID to retrieve state for.

        Returns:
            The state dictionary, or None if checkpoint doesn't exist.
        """
        entry = self.checkpoints.get(checkpoint_id)
        return copy.deepcopy(entry.output_state) if entry else None

    def clear(self) -> None:
        """Clear all checkpoints and reset the manager."""
        self.checkpoints.clear()
        self._checkpoint_order.clear()
        self._branches = {self.thread_id: []}
        logger.debug("CheckpointManager cleared")

    def to_table_data(self) -> list[dict[str, Any]]:
        """Convert checkpoints to a table-friendly format for UI display.

        Returns:
            List of dictionaries suitable for displaying in a table.
        """
        rows = []
        for cp_id in self._checkpoint_order:
            entry = self.checkpoints.get(cp_id)
            if entry:
                snapshot = entry.snapshot
                rows.append(
                    {
                        "time": snapshot.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                        "node": snapshot.node_name,
                        "checkpoint_id": snapshot.checkpoint_id,
                        "checkpoint_id_short": snapshot.checkpoint_id[:12] + "...",
                        "thread_id": snapshot.thread_id[:8] + "...",
                        "next_nodes": ", ".join(snapshot.next_nodes) or "(end)",
                        "is_fork": snapshot.metadata.get("fork_type") is not None,
                    }
                )
        return rows
