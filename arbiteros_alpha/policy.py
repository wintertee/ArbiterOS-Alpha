import logging
from abc import ABC
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import History

logger = logging.getLogger(__name__)


class PolicyChecker(ABC):
    """Abstract base class for policy checkers that validate execution constraints.

    PolicyCheckers enforce constraints before or after instruction execution.
    Subclasses must implement check_before and check_after methods to define
    custom validation logic.
    """

    def check_before(self, history: "History") -> bool:
        """Validate constraints before instruction execution.

        Args:
            history: The execution history up to this point.

        Returns:
            True if validation passes.

        Raises:
            RuntimeError: If validation fails.
        """
        pass

    def check_after(self, history: "History") -> bool:
        """Validate constraints after instruction execution.

        Args:
            history: The execution history including the just-executed instruction.

        Returns:
            True if validation passes.

        Raises:
            RuntimeError: If validation fails.
        """
        pass


class PolicyRouter(ABC):
    """Abstract base class for policy routers that dynamically route execution flow.

    PolicyRouters analyze execution history and decide whether to redirect
    the execution flow to a different node based on policy conditions.
    """

    def route(self, history: "History") -> str:
        """Determine the next node to execute based on policy conditions.

        Args:
            history: The execution history including the just-executed instruction.

        Returns:
            The name of the target node to route to, or None to continue normal flow.
        """
        pass


class HistoryPolicyChecker(PolicyChecker):
    """Policy checker that validates against blacklisted instruction sequences.

    This checker prevents specific sequences of instructions from being executed
    by maintaining a blacklist of forbidden instruction chains.

    Example:
        >>> checker = HistoryPolicyChecker()
        >>> checker.add_blacklist("no_direct_toolcall", ["generate", "toolcall"])
        >>> # This will raise RuntimeError if "generate" is followed by "toolcall"
    """

    def __init__(self):
        """Initialize an empty HistoryPolicyChecker with no blacklisted sequences."""
        self.blacklist = {}

    def add_blacklist(self, name: str, sequence: list[str]) -> "HistoryPolicyChecker":
        """Add a blacklisted instruction sequence.

        Args:
            name: A descriptive name for this blacklist rule.
            sequence: A list of instruction names that form a forbidden sequence.

        Returns:
            Self for method chaining.

        Example:
            >>> checker = HistoryPolicyChecker()
            >>> checker.add_blacklist("rule1", ["a", "b"]).add_blacklist("rule2", ["c", "d"])
        """
        self.blacklist["->".join(sequence)] = name
        return self

    def check_before(self, history: list["History"]) -> bool:
        """Check if the current history contains any blacklisted sequences.

        Args:
            history: The execution history to validate.

        Returns:
            True if no blacklisted sequences are detected.

        Raises:
            RuntimeError: If a blacklisted sequence is detected in the history.
        """
        history_sequence = "->".join(entry["instruction"] for entry in history)
        for blacklist_item in self.blacklist:
            if blacklist_item in history_sequence:
                logger.error(
                    f"Blacklisted sequence detected: {blacklist_item} in {history_sequence}"
                )
        return True

    def __repr__(self):
        """Return a string representation of the checker."""
        return f"HistoryPolicyChecker(blacklist={self.blacklist})"


class ConfidencePolicyRouter(PolicyRouter):
    """Policy router that redirects flow based on confidence scores.

    This router monitors a confidence metric in the instruction output and
    redirects to a target node when the confidence falls below a threshold.

    Example:
        >>> router = ConfidencePolicyRouter("confidence", threshold=0.5, target="generate")
        >>> # If output["confidence"] < 0.5, route back to "generate" node
    """

    def __init__(self, key: str, threshold: float, target: str):
        """Initialize a confidence-based policy router.

        Args:
            key: The key in the output dict that contains the confidence score.
            threshold: The minimum acceptable confidence value (0.0 to 1.0).
            target: The node name to route to when confidence is below threshold.
        """
        self.key = key
        self.threshold = threshold
        self.target = target

    def route(self, history: list["History"]) -> str | None:
        """Route to target node if confidence is below threshold.

        Args:
            history: The execution history including the just-executed instruction.

        Returns:
            The target node name if confidence < threshold, None otherwise.
        """
        last_entry = history[-1]
        output = last_entry["output"]
        confidence = output.get(self.key, 1.0)
        if confidence < self.threshold:
            return self.target
        return None

    def __repr__(self):
        """Return a string representation of the router."""
        return f"ConfidencePolicyRouter(key={self.key}, threshold={self.threshold}, target={self.target})"
