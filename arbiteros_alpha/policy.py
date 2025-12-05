import logging
from abc import ABC
from dataclasses import dataclass

from .history import History
from .instructions import InstructionType

logger = logging.getLogger(__name__)


@dataclass
class PolicyChecker(ABC):
    """Abstract base class for policy checkers that validate execution constraints.

    PolicyCheckers enforce constraints before instruction execution.
    Subclasses must implement check_before method to define
    custom validation logic.
    """

    def check_before(self, history: History) -> bool:
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

    def route_after(self, history: History) -> str:
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

    def check_before(self, history: History) -> bool:
        """Check if the current history contains any blacklisted sequences.

        Args:
            history: The execution history to validate.

        Returns:
            True if no blacklisted sequences are detected.

        Raises:
            RuntimeError: If a blacklisted sequence is detected in the history.
        """

        n_work = len(history.entries)

        n_pat = len(self.bad_sequence)

        logger.debug(
            f"HistoryPolicyChecker '{self.name}': checking {n_work} completed supersteps against pattern of length {n_pat}"
        )
        logger.debug(f"Bad sequence pattern: {[i.name for i in self.bad_sequence]}")

        # 如果 pattern 比 workflow 还长，直接不可能
        if n_pat > n_work:
            return True

        # 滑动窗口：遍历 workflow 中所有可能的起始位置
        # 我们只需要检查 workflow[i] 到 workflow[i + n_pat] 这段区间
        for i in range(n_work - n_pat + 1):
            match = True
            # 检查 pattern 的每一个节点是否在对应的 workflow 层级中
            for j in range(n_pat):
                # i 是 workflow 的起始偏移量，j 是 pattern 的索引
                current_stage_workers = history.entries[i + j]
                current_stage_workers = [
                    item.instruction for item in current_stage_workers
                ]
                target_worker = self.bad_sequence[j]

                logger.debug(
                    f"  Window[{i}][{j}]: checking if {target_worker.name} in {[w.name for w in current_stage_workers]}"
                )

                if target_worker not in current_stage_workers:
                    match = False
                    break

            # 如果在这个窗口 i 中，pattern 所有节点都匹配上了，则成功
            if match:
                logger.debug(
                    f"Blacklisted sequence detected: {self.name}:[{self.bad_sequence}]"
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

    def route_after(self, history: History) -> str | None:
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
        try:
            last_entry = history.entries[-1][-1]
        except IndexError:
            return None
        output = last_entry.output_state
        confidence = output.get(self.key, 1.0)
        if confidence < self.threshold:
            return self.target
        return None
