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

    name: str

    def check_before(self, history: History) -> bool:
        """Validate constraints before instruction execution.

        Args:
            history: The execution history up to this point.

        Returns:
            True if validation passes.

        Raises:
            RuntimeError: If validation fails.
        """
        raise NotImplementedError


@dataclass
class PolicyRouter(ABC):
    """Abstract base class for policy routers that dynamically route execution flow.

    PolicyRouters analyze execution history and decide whether to redirect
    the execution flow to a different node based on policy conditions.
    """

    name: str

    def route_after(self, history: History) -> str | None:
        """Determine the next node to execute based on policy conditions.

        Args:
            history: The execution history including the just-executed instruction.

        Returns:
            The name of the target node to route to, or None to continue normal flow.
        """
        raise NotImplementedError


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

        Only checks windows that include the most recent superstep to avoid
        redundant checks of already-validated history.

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

        if n_pat > n_work:
            return True

        # Only check windows that include the most recent superstep (n_work - 1)
        # This avoids redundant checks of previously validated history
        start_idx = max(0, n_work - n_pat)

        for i in range(start_idx, n_work - n_pat + 1):
            match = True
            for j in range(n_pat):
                # i for workflow offset, j for pattern index
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
        if not history.entries or not history.entries[-1]:
            return None
        last_entry = history.entries[-1][-1]
        output = last_entry.output_state
        confidence = output.get(self.key, 1.0)
        if confidence < self.threshold:
            return self.target
        return None


# =============================================================================
# Human-in-the-Loop Policies
# =============================================================================


class HumanInterruptRequest(Exception):
    """Exception raised when human intervention is required.

    This exception signals that the workflow should pause and wait for
    human input before proceeding. It contains all necessary context
    for the human reviewer.

    Attributes:
        policy_name: Name of the policy that triggered the interrupt.
        reason: Human-readable reason for the interrupt.
        context: Additional context for the human reviewer.
        required_actions: List of actions the human must take.
        timeout_seconds: How long to wait for human response (None = indefinite).
    """

    def __init__(
        self,
        policy_name: str,
        reason: str,
        context: dict = None,
        required_actions: list[str] = None,
        timeout_seconds: int | None = None,
    ):
        self.policy_name = policy_name
        self.reason = reason
        self.context = context or {}
        self.required_actions = required_actions or ["approve", "reject", "modify"]
        self.timeout_seconds = timeout_seconds
        super().__init__(f"[{policy_name}] Human intervention required: {reason}")


@dataclass
class HumanInterruptPolicyChecker(PolicyChecker):
    """Policy checker that requires human approval for critical decisions.

    This checker pauses execution and requests human intervention when
    specified conditions are met. It's designed for high-stakes operations
    where automated decisions are insufficient.

    Attributes:
        name: Human-readable name for this policy checker.
        trigger_on_high_risk: Interrupt when risk_level > risk_threshold.
        trigger_on_low_confidence: Interrupt when confidence < confidence_threshold.
        trigger_on_first_execution: Interrupt on first execution of a type.
        risk_threshold: Risk level that triggers interrupt (default: 0.85).
        confidence_threshold: Confidence below which to interrupt (default: 0.4).
        instructions_requiring_approval: Instruction types that always need approval.

    Example:
        >>> checker = HumanInterruptPolicyChecker(
        ...     name="require_human_for_trades",
        ...     trigger_on_high_risk=True,
        ...     risk_threshold=0.8,
        ...     instructions_requiring_approval=[ExecutionCore.TOOL_CALL]
        ... )
        >>> arbiter_os.add_policy_checker(checker)
    """

    name: str
    trigger_on_high_risk: bool = True
    trigger_on_low_confidence: bool = True
    trigger_on_first_execution: bool = False
    risk_threshold: float = 0.85
    confidence_threshold: float = 0.4
    instructions_requiring_approval: list[InstructionType] = None

    def __post_init__(self):
        if self.instructions_requiring_approval is None:
            self.instructions_requiring_approval = []

    def check_before(self, history: History) -> bool:
        """Check if human intervention is required before proceeding.

        Evaluates multiple conditions to determine if the workflow should
        pause for human review:
        1. High risk level in recent outputs
        2. Low confidence in recent outputs
        3. Instruction type requires explicit approval
        4. First-time execution of certain instruction types

        Args:
            history: The execution history up to this point.

        Returns:
            True if no human intervention needed.

        Raises:
            HumanInterruptRequest: When human intervention is required.
        """
        if not history.entries:
            return True

        last_superstep = history.entries[-1]
        if not last_superstep:
            return True

        last_item = last_superstep[-1]
        output_state = last_item.output_state or {}
        instruction = last_item.instruction

        # Check instruction type requiring approval
        if instruction in self.instructions_requiring_approval:
            raise HumanInterruptRequest(
                policy_name=self.name,
                reason=f"Instruction {instruction.name} requires human approval",
                context={
                    "instruction": instruction.name,
                    "input_state": last_item.input_state,
                    "output_state": output_state,
                },
                required_actions=["approve", "reject"],
            )

        # Check high risk
        if self.trigger_on_high_risk:
            risk_level = output_state.get("risk_level")
            if risk_level is not None and risk_level > self.risk_threshold:
                raise HumanInterruptRequest(
                    policy_name=self.name,
                    reason=f"High risk detected: {risk_level:.2f} > {self.risk_threshold}",
                    context={
                        "risk_level": risk_level,
                        "threshold": self.risk_threshold,
                        "output_state": output_state,
                    },
                    required_actions=["approve", "reject", "adjust_parameters"],
                )

        # Check low confidence
        if self.trigger_on_low_confidence:
            confidence = output_state.get("confidence")
            if confidence is not None and confidence < self.confidence_threshold:
                raise HumanInterruptRequest(
                    policy_name=self.name,
                    reason=f"Low confidence: {confidence:.2f} < {self.confidence_threshold}",
                    context={
                        "confidence": confidence,
                        "threshold": self.confidence_threshold,
                        "output_state": output_state,
                    },
                    required_actions=["approve", "reject", "request_retry"],
                )

        # Check first execution
        if self.trigger_on_first_execution:
            instruction_count = 0
            for superstep in history.entries:
                for item in superstep:
                    if item.instruction == instruction:
                        instruction_count += 1

            if instruction_count <= 1:
                raise HumanInterruptRequest(
                    policy_name=self.name,
                    reason=f"First execution of {instruction.name} requires review",
                    context={
                        "instruction": instruction.name,
                        "execution_count": instruction_count,
                    },
                    required_actions=["approve", "reject"],
                )

        return True


@dataclass
class VerificationRequirementChecker(PolicyChecker):
    """Policy checker that requires verification before high-risk operations.

    This checker ensures that a VERIFY instruction has been executed
    before allowing high-risk operations (like TOOL_CALL) to proceed.

    Attributes:
        name: Human-readable name for this checker.
        target_instructions: Instructions that require verification.
        verification_instruction: The instruction type that provides verification.
        min_verifications: Minimum number of verifications required.
        strict_mode: If True, raises error instead of returning False.

    Example:
        >>> from arbiteros_alpha.instructions import ExecutionCore, NormativeCore
        >>> checker = VerificationRequirementChecker(
        ...     name="verify_before_tool_call",
        ...     target_instructions=[ExecutionCore.TOOL_CALL],
        ...     verification_instruction=NormativeCore.VERIFY,
        ...     min_verifications=1
        ... )
        >>> arbiter_os.add_policy_checker(checker)
    """

    name: str
    target_instructions: list[InstructionType] = None
    verification_instruction: InstructionType = None
    min_verifications: int = 1
    strict_mode: bool = True

    def __post_init__(self):
        if self.target_instructions is None:
            self.target_instructions = []

    def check_before(self, history: History) -> bool:
        """Check that verification has been performed before target instructions.

        Counts VERIFY instructions and ensures the required minimum have
        been executed before allowing target instructions to proceed.

        Args:
            history: The execution history.

        Returns:
            True if sufficient verification has been performed.

        Raises:
            RuntimeError: In strict_mode when verification is insufficient.
        """
        if not history.entries:
            return True

        # Count verifications
        verification_count = 0
        for superstep in history.entries:
            for item in superstep:
                if (
                    self.verification_instruction is not None
                    and item.instruction == self.verification_instruction
                ):
                    verification_count += 1

        # Check if current instruction requires verification
        last_superstep = history.entries[-1]
        if last_superstep:
            last_instruction = last_superstep[-1].instruction
            if last_instruction in self.target_instructions:
                if verification_count < self.min_verifications:
                    error_msg = (
                        f"Instruction {last_instruction.name} requires "
                        f"{self.min_verifications} verification(s), "
                        f"found {verification_count}"
                    )
                    logger.warning(f"[{self.name}] {error_msg}")

                    if self.strict_mode:
                        raise RuntimeError(f"[{self.name}] {error_msg}")
                    return False

        return True


@dataclass
class MustUseToolsChecker(PolicyChecker):
    """Ensure at least one tool has been called before responding.

    This checker prevents agents from generating responses without first
    using any tools to gather information. Common use case: ensuring
    an agent doesn't hallucinate answers without consulting tools.

    Example:
        ```python
        from arbiteros_alpha.policy import MustUseToolsChecker
        from arbiteros_alpha.instructions import ExecutionCore

        arbiter_os.add_policy_checker(
            MustUseToolsChecker(
                name="must_use_tools",
                respond_instruction=ExecutionCore.RESPOND
            )
        )
        ```
    """

    name: str
    respond_instruction: InstructionType

    def check_before(self, history: History) -> bool:
        """Check if we're about to RESPOND without any prior TOOL_CALL.

        This is called before each instruction executes. The history's last item
        is the instruction that is ABOUT TO BE EXECUTED (with empty output_state).
        All previous items are completed instructions with full output_state.

        Args:
            history: The execution history to validate.

        Returns:
            False if policy is violated (attempting RESPOND without tools).
            True otherwise.
        """
        # Get all items from history
        all_items = [item for superstep in history.entries for item in superstep]

        if not all_items:
            return True

        # The last item is the instruction about to be executed
        current_item = all_items[-1]

        # If we're about to execute RESPOND, check if tools were used
        if current_item.instruction == self.respond_instruction:
            # Import here to avoid circular dependency
            from .instructions import ExecutionCore

            # Check previous history (excluding current item) for TOOL_CALL
            has_tool_calls = any(
                item.instruction == ExecutionCore.TOOL_CALL for item in all_items[:-1]
            )

            if not has_tool_calls:
                logger.error(
                    f"[{self.name}] Policy violation: Attempting to execute "
                    f"{self.respond_instruction.name} without any prior TOOL_CALL."
                )
                return False

        return True
