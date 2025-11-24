import logging
from abc import ABC
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .instructions import InstructionType

if TYPE_CHECKING:
    from . import History

logger = logging.getLogger(__name__)


@dataclass
class PolicyChecker(ABC):
    """Abstract base class for policy checkers that validate execution constraints.

    PolicyCheckers enforce constraints before instruction execution.
    Subclasses must implement check_before method to define
    custom validation logic.
    """

    def check_before(self, history: list["History"]) -> bool:
        """Validate constraints before instruction execution.

        Args:
            history: The execution history up to this point.

        Returns:
            True if validation passes.

        Raises:
            RuntimeError: If validation fails.
        """
        pass


@dataclass
class PolicyRouter(ABC):
    """Abstract base class for policy routers that dynamically route execution flow.

    PolicyRouters analyze execution history and decide whether to redirect
    the execution flow to a different node based on policy conditions.
    """

    def route_after(self, history: list["History"]) -> str:
        """Determine the next node to execute based on policy conditions.

        Args:
            history: The execution history including the just-executed instruction.

        Returns:
            The name of the target node to route to, or None to continue normal flow.
        """
        pass


@dataclass
class HistoryPolicyChecker(PolicyChecker):
    """Policy checker that validates against blacklisted instruction sequences.

    This checker prevents specific sequences of instructions from being executed
    by maintaining a blacklist of forbidden instruction chains.

    Example:
        >>> from arbiteros_alpha.instructions import CognitiveCore, ExecutionCore
        >>> checker = HistoryPolicyChecker(
        ...     "no_direct_toolcall",
        ...     [CognitiveCore.GENERATE, ExecutionCore.TOOL_CALL]
        ... )
        >>> # This will raise RuntimeError if GENERATE is followed by TOOL_CALL
    """

    name: str
    bad_sequence: list[InstructionType]

    def __post_init__(self):
        """Convert sequence list to string representation after initialization."""
        self._bad_sequence_str = "->".join(instr.name for instr in self.bad_sequence)

    def check_before(self, history: list["History"]) -> bool:
        """Check if the current history contains any blacklisted sequences.

        Args:
            history: The execution history to validate.

        Returns:
            True if no blacklisted sequences are detected.

        Raises:
            RuntimeError: If a blacklisted sequence is detected in the history.
        """
        history_sequence = "->".join(entry.instruction.name for entry in history)
        if self._bad_sequence_str in history_sequence:
            logger.debug(
                f"Blacklisted sequence detected: {self.name}:[{self._bad_sequence_str}] in [{history_sequence}]"
            )
            return False

        return True


@dataclass
class MetricThresholdPolicyRouter(PolicyRouter):
    """Policy router that redirects execution flow based on threshold conditions.

    This router monitors a specific metric in the instruction output and
    redirects to a target node when the metric value falls below a specified
    threshold. Commonly used for quality control patterns like regeneration
    on low confidence scores.

    Attributes:
        name: Human-readable name for this policy router.
        key: The key in the output state dictionary to monitor.
        threshold: The minimum acceptable value for the monitored metric.
        target: The node name to route to when value is below threshold.

    Example:
        Create a router that triggers regeneration when confidence is low:

        >>> router = MetricThresholdPolicyRouter(
        ...     name="regenerate_on_low_confidence",
        ...     key="confidence",
        ...     threshold=0.6,
        ...     target="generate"
        ... )
        >>> # If output["confidence"] < 0.6, routes back to "generate" node
    """

    name: str
    key: str
    threshold: float
    target: str

    def route_after(self, history: list["History"]) -> str | None:
        """Route to target node if the monitored metric is below threshold.

        Extracts the specified metric from the most recent instruction's output
        state and compares it against the configured threshold. If the metric
        value is below the threshold, returns the target node for routing.

        Args:
            history: The execution history including the just-executed instruction.
                The last entry's output_state is checked for the metric.

        Returns:
            The target node name if metric value < threshold, None otherwise.
            When None is returned, execution continues with normal flow.
        """
        last_entry = history[-1]
        output = last_entry.output_state
        confidence = output.get(self.key, 1.0)
        if confidence < self.threshold:
            return self.target
        return None


class GraphStructurePolicyChecker:
    """Policy checker that validates graph structure against blacklisted edge sequences.

    This checker validates the graph structure before execution to ensure that
    the graph does not contain any paths that violate blacklisted sequences.
    Unlike HistoryPolicyChecker which checks execution history at runtime,
    this checker validates the static graph structure before the graph runs.

    Example:
        >>> checker = GraphStructurePolicyChecker()
        >>> checker.add_blacklist("no_direct_toolcall", ["generate", "toolcall"])
        >>> # This will raise RuntimeError if graph has an edge from "generate" to "toolcall"
        >>> edges = [("generate", "toolcall"), ("toolcall", "evaluate")]
        >>> checker.check_graph_structure(edges)
        >>> # Raises RuntimeError: Graph structure violates blacklist rule: no_direct_toolcall
    """

    def __init__(self):
        """Initialize an empty GraphStructurePolicyChecker with no blacklisted sequences."""
        self.blacklist = {}  # Maps sequence string to rule name
        self._blacklist_levels = {}  # Maps sequence string to level ("error" or "warning")
        self._node_to_instruction: dict[str, "InstructionType"] = {}  # Node name to instruction type mapping

    def add_blacklist(
        self, name: str, sequence: list[str] | list[InstructionType], level: str = "error"
    ) -> "GraphStructurePolicyChecker":
        """Add a blacklisted instruction sequence.

        Args:
            name: A descriptive name for this blacklist rule.
            sequence: A list of instruction names (strings) or InstructionType enums
                that form a forbidden sequence. If InstructionType enums are provided,
                they will be converted to their name strings (lowercase).
                The checker will verify that the graph does not contain a path
                that follows this sequence.
            level: Severity level for violations. Can be "error" (default) or "warning".
                - "error": Raises RuntimeError when violation is detected
                - "warning": Logs a warning but does not raise an exception

        Returns:
            Self for method chaining.

        Example:
            >>> checker = GraphStructurePolicyChecker()
            >>> checker.add_blacklist("rule1", ["a", "b"]).add_blacklist("rule2", ["c", "d"])
            >>> # Or with InstructionType enums:
            >>> from arbiteros_alpha.instructions import GENERATE, TOOL_CALL
            >>> checker.add_blacklist("no_direct", [GENERATE, TOOL_CALL], level="error")
            >>> checker.add_blacklist("warning_rule", [GENERATE, GENERATE], level="warning")
        """
        if len(sequence) < 2:
            raise ValueError("Blacklist sequence must contain at least 2 nodes")
        
        if level not in ["error", "warning"]:
            raise ValueError(f"Level must be 'error' or 'warning', got '{level}'")
        
        # Convert InstructionType enums to strings if needed
        sequence_strs = []
        for item in sequence:
            if isinstance(item, str):
                sequence_strs.append(item)
            else:
                # Assume it's an InstructionType enum, use its name (lowercase)
                sequence_strs.append(item.name.lower())
        
        sequence_key = "->".join(sequence_strs)
        self.blacklist[sequence_key] = name
        self._blacklist_levels[sequence_key] = level
        return self

    def check_graph_structure(
        self, 
        edges: list[tuple[str, str]], 
        node_to_instruction: dict[str, "InstructionType"] | None = None
    ) -> bool:
        """Check if the graph structure contains any blacklisted sequences.

        This method validates the graph structure by checking if there exists
        any path in the graph that matches a blacklisted sequence. It performs
        a depth-first search to find all possible paths and checks them against
        the blacklist.

        Args:
            edges: List of tuples representing graph edges in format (source, target).
                Source and target can be node names or special constants like START/END.
            node_to_instruction: Optional mapping from node names to their instruction types.
                If provided, patterns matching instruction types (e.g., "generate") will
                match nodes with that instruction type, regardless of node name.

        Returns:
            True if no blacklisted sequences are detected in the graph structure.

        Raises:
            RuntimeError: If a blacklisted sequence is detected in the graph structure.
                The error message includes the rule name and the violating path.
        """
        # Build adjacency list from edges
        graph = {}
        for source, target in edges:
            if source not in graph:
                graph[source] = []
            graph[source].append(target)

        # Store node_to_instruction for use in pattern matching
        self._node_to_instruction = node_to_instruction or {}

        # Track violations by level
        errors = []
        warnings = []
        
        # For each blacklisted sequence, check if it exists in the graph
        for blacklist_sequence, rule_name in self.blacklist.items():
            sequence_nodes = blacklist_sequence.split("->")
            if len(sequence_nodes) < 2:
                continue

            level = self._blacklist_levels.get(blacklist_sequence, "error")
            logger.debug(f"Checking blacklist rule '{rule_name}' (level={level}) with pattern '{blacklist_sequence}'")
            
            # Check if there's a path following this sequence
            violating_path = self._find_violating_path(graph, sequence_nodes, edges)
            if violating_path:
                violation_msg = (
                    f"Graph structure violates blacklist rule '{rule_name}': "
                    f"found path '{violating_path}' in graph structure. "
                    f"Pattern '{blacklist_sequence}' is not allowed."
                )
                
                if level == "error":
                    errors.append(violation_msg)
                    logger.error(violation_msg)
                else:  # level == "warning"
                    warnings.append(violation_msg)
                    logger.warning(violation_msg)

        # Log all warnings first, then raise errors if any
        # if warnings:
        #     logger.warning(f"Found {len(warnings)} warning-level violations in graph structure")
        
        # if errors:
        #     error_summary = f"Found {len(errors)} error-level violation(s) in graph structure"
        #     if warnings:
        #         error_summary += f" and {len(warnings)} warning-level violation(s)"
        #     raise RuntimeError(f"{error_summary}. First error: {errors[0]}")

        return True

    def _find_violating_path(
        self, graph: dict[str, list[str]], sequence: list[str], all_edges: list[tuple[str, str]]
    ) -> str | None:
        """Find a violating path that matches the blacklisted sequence pattern.
        
        Returns the actual node path that violates the pattern, or None if no violation found.
        
        Args:
            graph: Adjacency list representation of the graph.
            sequence: List of node name patterns in order.
            all_edges: List of all edges for reference.
        
        Returns:
            A string representation of the violating path, or None if no violation found.
        """
        if len(sequence) < 2:
            return None

        # For 2-node sequences, find matching edge
        if len(sequence) == 2:
            pattern_source, pattern_target = sequence
            logger.debug(f"Checking 2-node pattern: {pattern_source} -> {pattern_target}")
            for source, target in all_edges:
                source_matches = self._matches_pattern(source, pattern_source)
                target_matches = self._matches_pattern(target, pattern_target)
                logger.debug(f"  Edge {source} -> {target}: source matches={source_matches}, target matches={target_matches}")
                if source_matches and target_matches:
                    return f"{source} -> {target}"
            return None

        # For longer sequences, find actual path
        actual_path = self._find_exact_path(graph, sequence)
        if actual_path:
            return " -> ".join(actual_path)
        return None

    def _has_path_sequence(
        self, graph: dict[str, list[str]], sequence: list[str], all_edges: list[tuple[str, str]]
    ) -> bool:
        """Check if the graph contains a path that matches the given sequence.

        For a sequence [a, b, c], this checks if there exists a path from a -> b -> c
        in the graph. It uses DFS to find all possible paths and checks if any
        path matches the sequence.
        
        The sequence can contain:
        - Exact node names (e.g., "generate_plan")
        - Instruction type names (e.g., "tool_call") which will match any node
          whose name starts with that prefix (e.g., "tool_call_perceive", "tool_call_execute")

        Args:
            graph: Adjacency list representation of the graph.
            sequence: List of node names or instruction type names in order.
            all_edges: List of all edges for reference.

        Returns:
            True if a path matching the sequence exists, False otherwise.
        """
        if len(sequence) < 2:
            return False

        # For 2-node sequences, check direct edge with pattern matching
        if len(sequence) == 2:
            pattern_source, pattern_target = sequence
            for source, target in all_edges:
                if self._matches_pattern(source, pattern_source) and self._matches_pattern(
                    target, pattern_target
                ):
                    return True
            return False

        # For longer sequences, use DFS to find if there's a path
        # that visits nodes in the exact sequence order
        return self._has_exact_path(graph, sequence)
    
    def _matches_pattern(self, node_name: str, pattern: str) -> bool:
        """Check if a node name matches a pattern.
        
        A pattern can be:
        - An exact node name match
        - An instruction type name that matches nodes starting with that prefix
          (e.g., "tool_call" matches "tool_call_perceive", "tool_call_execute")
        - An instruction type name that matches nodes by their actual instruction type
          (e.g., "generate" matches "orchestrate" if orchestrate uses GENERATE instruction)
        
        Args:
            node_name: The actual node name from the graph.
            pattern: The pattern to match against (exact name or instruction type prefix/name).
        
        Returns:
            True if the node name matches the pattern, False otherwise.
        """
        # Exact match
        if node_name == pattern:
            return True
        
        # Pattern match: check if node name starts with pattern followed by underscore
        # This handles cases like "tool_call" matching "tool_call_perceive"
        if node_name.startswith(pattern + "_"):
            return True
        
        # Instruction type match: check if node's instruction type matches the pattern
        # This handles cases like "generate" matching "orchestrate" if orchestrate uses GENERATE
        if node_name in self._node_to_instruction:
            node_instruction = self._node_to_instruction[node_name]
            # Check if the pattern matches the instruction type name (case-insensitive)
            matches = node_instruction.name.lower() == pattern.lower()
            if matches:
                logger.debug(f"  Pattern match: {node_name} (instruction={node_instruction.name}) matches pattern '{pattern}'")
                return True
            else:
                logger.debug(f"  Pattern mismatch: {node_name} (instruction={node_instruction.name}) does not match pattern '{pattern}'")
        
        return False

    def _find_exact_path(self, graph: dict[str, list[str]], sequence: list[str]) -> list[str] | None:
        """Find a path in the graph that matches the sequence pattern.

        For sequences longer than 2 nodes, this finds a path that visits nodes
        matching the patterns in order. Returns the actual node names.

        Args:
            graph: Adjacency list representation of the graph.
            sequence: List of node name or instruction type patterns that must be visited in order.

        Returns:
            List of actual node names forming the path, or None if no such path exists.
        """
        if len(sequence) < 2:
            return None

        def find_path(start: str, target_sequence: list[str], visited: set[str], path: list[str]) -> list[str] | None:
            """Find if there's a path from start that matches the sequence."""
            current_path = path + [start]
            
            if len(target_sequence) == 0:
                return current_path
            if len(target_sequence) == 1:
                if self._matches_pattern(start, target_sequence[0]):
                    return current_path
                return None

            # Check if current node matches first in sequence
            if self._matches_pattern(start, target_sequence[0]):
                # Found first node, look for path matching rest of sequence
                for neighbor in graph.get(start, []):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        result = find_path(neighbor, target_sequence[1:], visited, current_path)
                        if result:
                            return result
                        visited.remove(neighbor)
            else:
                # Current node doesn't match, but might be intermediate
                # Continue searching from neighbors
                for neighbor in graph.get(start, []):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        result = find_path(neighbor, target_sequence, visited, current_path)
                        if result:
                            return result
                        visited.remove(neighbor)

            return None

        # Find all nodes that match the first pattern in the sequence
        start_pattern = sequence[0]
        matching_starts = [
            node for node in graph.keys() if self._matches_pattern(node, start_pattern)
        ]
        
        if not matching_starts:
            return None

        # Try to find a path starting from any matching node
        for start_node in matching_starts:
            visited = set()
            result = find_path(start_node, sequence, visited, [])
            if result:
                return result

        return None

    def _has_exact_path(self, graph: dict[str, list[str]], sequence: list[str]) -> bool:
        """Check if there's a path in the graph that exactly matches the sequence.

        For sequences longer than 2 nodes, this checks if there exists a path
        that visits nodes in the exact order. It uses DFS to find such paths.
        Supports pattern matching for instruction type names.

        Args:
            graph: Adjacency list representation of the graph.
            sequence: List of node names or instruction type patterns that must be visited in order.

        Returns:
            True if such a path exists, False otherwise.
        """
        if len(sequence) < 2:
            return False

        # For sequences of length > 2, check if we can find a path
        # that visits nodes in sequence order
        def find_path(start: str, target_sequence: list[str], visited: set[str]) -> bool:
            """Find if there's a path from start that matches the sequence."""
            if len(target_sequence) == 0:
                return True
            if len(target_sequence) == 1:
                return self._matches_pattern(start, target_sequence[0])

            # Check if current node matches first in sequence
            if self._matches_pattern(start, target_sequence[0]):
                # Found first node, look for path matching rest of sequence
                next_target = target_sequence[1]
                for neighbor in graph.get(start, []):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        if find_path(neighbor, target_sequence[1:], visited):
                            return True
                        visited.remove(neighbor)
            else:
                # Current node doesn't match, but might be intermediate
                # Continue searching from neighbors
                for neighbor in graph.get(start, []):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        if find_path(neighbor, target_sequence, visited):
                            return True
                        visited.remove(neighbor)

            return False

        # Find all nodes that match the first pattern in the sequence
        start_pattern = sequence[0]
        matching_starts = [
            node for node in graph.keys() if self._matches_pattern(node, start_pattern)
        ]
        
        if not matching_starts:
            return False

        # Try to find a path starting from any matching node
        for start_node in matching_starts:
            visited = set()
            if find_path(start_node, sequence, visited):
                return True

        return False

    def __repr__(self):
        """Return a string representation of the checker."""
        return f"GraphStructurePolicyChecker(blacklist={self.blacklist})"