import datetime
import functools
import logging
from typing import Any, Callable, TypedDict

from langgraph.types import Command

from .policy import PolicyChecker, PolicyRouter

logger = logging.getLogger(__name__)


class History(TypedDict):
    """The minimal OS metadata"""

    timestamp: str
    instruction: str
    input: dict[str, Any]
    output: dict[str, Any]


class ArbiterOSAlpha:
    """Main ArbiterOS coordinator for policy-driven LangGraph execution.

    ArbiterOSAlpha provides a lightweight governance layer on top of LangGraph,
    enabling policy-based validation and dynamic routing without modifying
    the underlying graph structure.

    Attributes:
        history: List of execution history entries with timestamps and I/O.
        policy_checkers: List of PolicyChecker instances for validation.
        policy_routers: List of PolicyRouter instances for dynamic routing.

    Example:
        >>> os = ArbiterOSAlpha()
        >>> os.add_policy_checker(HistoryPolicyChecker().add_blacklist("rule", ["a", "b"]))
        >>> os.add_policy_router(ConfidencePolicyRouter("confidence", 0.5, "retry"))
        >>> @os.instruction("generate")
        ... def generate(state): return {"result": "output"}
    """

    def __init__(self):
        """Initialize ArbiterOSAlpha with empty history and no policies."""
        self.history: list[History] = []
        self.policy_checkers: list[PolicyChecker] = []
        self.policy_routers: list[PolicyRouter] = []

    def add_policy_checker(self, checker: PolicyChecker) -> None:
        """Register a policy checker for validation.

        Args:
            checker: A PolicyChecker instance to validate execution constraints.
        """
        logger.debug(f"Adding policy checker: {checker}")
        self.policy_checkers.append(checker)

    def add_policy_router(self, router: PolicyRouter) -> None:
        """Register a policy router for dynamic flow control.

        Args:
            router: A PolicyRouter instance to dynamically route execution.
        """
        logger.debug(f"Adding policy router: {router}")
        self.policy_routers.append(router)

    def check_before(self):
        """Execute all policy checkers before instruction execution.

        Returns:
            True if all checkers pass.

        Raises:
            RuntimeError: If any checker fails validation.
        """
        logger.debug(f"Running {len(self.policy_checkers)} policy checkers (before)")
        for checker in self.policy_checkers:
            checker.check_before(self.history)
        return True

    def check_after(self):
        """Execute all policy checkers after instruction execution.

        Returns:
            True if all checkers pass.

        Raises:
            RuntimeError: If any checker fails validation.
        """
        logger.debug(f"Running {len(self.policy_checkers)} policy checkers (after)")
        for checker in self.policy_checkers:
            checker.check_after(self.history)
        return True

    def route(self) -> str | None:
        """Determine if execution should be routed to a different node.

        Consults all registered policy routers in order. Returns the first
        non-None routing decision.

        Returns:
            The target node name if routing is triggered, None otherwise.
        """
        logger.debug(f"Checking {len(self.policy_routers)} policy routers")
        for router in self.policy_routers:
            decision = router.route(self.history)
            if decision:
                logger.warning(f"Router {router} decided to route to: {decision}")
                return decision
        logger.debug("No routing triggered, continuing normal flow")
        return None

    def instruction(self, name: str) -> Callable[[Callable], Callable]:
        """Decorator to wrap LangGraph node functions with policy governance.

        This decorator adds policy validation, execution history tracking,
        and dynamic routing to LangGraph node functions. It's the core
        integration point between ArbiterOS and LangGraph.

        Args:
            name: A unique identifier for this instruction/node.

        Returns:
            A decorator function that wraps the target node function.

        Example:
            >>> @os.instruction("generate")
            ... def generate(state: State) -> State:
            ...     return {"field": "value"}
            >>> # Function now includes policy checks and history tracking
        """

        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs) -> Any:
                logger.debug(f"Executing instruction: {name}")

                self.check_before()

                result = func(*args, **kwargs)
                logger.debug(f"Instruction {name} returned: {result}")

                self.history.append(
                    {
                        "timestamp": datetime.datetime.now().isoformat(),
                        "instruction": name,
                        "input": args[0],
                        "output": result,
                    }
                )
                self.check_after()

                destination = self.route()

                if destination:
                    logger.debug(f"Routing from {name} to {destination}")
                    return Command(update=result, goto=destination)

                return result

            return wrapper

        return decorator
