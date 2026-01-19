"""Core components for ArbiterOS-alpha.

This module contains the main classes and functionality for policy-driven
governance of LangGraph execution.
"""

import datetime
import functools
import inspect
import logging
import weakref
from typing import Any, Callable, Literal

from langgraph.pregel import Pregel, _loop
from langgraph.types import Command

from .evaluation import EvaluationResult, NodeEvaluator
from .history import History, HistoryItem
from .instructions import InstructionType
from .policy import PolicyChecker, PolicyRouter

logger = logging.getLogger(__name__)

# Global registry to map Pregel (CompiledStateGraph) instances to their corresponding ArbiterOSAlpha instances
_pregel_to_arbiter_map: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()

# Global registry to map PregelLoop instances to their parent Pregel instances
_loop_to_pregel_map: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


class ArbiterOSAlpha:
    """Main ArbiterOS coordinator for policy-driven LangGraph execution.

    ArbiterOSAlpha provides a lightweight governance layer on top of LangGraph,
    enabling policy-based validation and dynamic routing without modifying
    the underlying graph structure.

    Attributes:
        backend: The execution backend in use.
        history: List of execution history entries with timestamps and I/O.
        policy_checkers: List of PolicyChecker instances for validation.
        policy_routers: List of PolicyRouter instances for dynamic routing.

    Example:
        ```python
        os = ArbiterOSAlpha(backend="langgraph")
        os.add_policy_checker(
            HistoryPolicyChecker("require_verification", ["generate", "execute"])
        )
        os.add_policy_router(ConfidencePolicyRouter("confidence", 0.5, "retry"))

        @os.instruction("generate")
        def generate(state):
            return {"result": "output"}
        ```
    """

    def __init__(self, backend: Literal["langgraph", "vanilla"] = "langgraph"):
        """Initialize the ArbiterOSAlpha instance.

        Args:
            backend: The execution backend to use.
                - "langgraph": Use an agent based on the LangGraph framework.
                - "vanilla": Use the framework-less ('from scratch') agent implementation.
        """
        self.backend = backend
        self.history: History = History()
        self.policy_checkers: list[PolicyChecker] = []
        self.policy_routers: list[PolicyRouter] = []
        self.evaluators: list[NodeEvaluator] = []

        if self.backend == "langgraph":
            self._patch_pregel_loop()

    def add_policy_checker(self, checker: PolicyChecker) -> None:
        """Register a policy checker for validation.

        Args:
            checker: A PolicyChecker instance to validate execution constraints.
        """
        logger.debug(f"Adding policy checker: {checker}")
        self.policy_checkers.append(checker)

    def add_policy_router(self, router: PolicyRouter) -> None:
        """Register a policy router for dynamic flow control.

        Policy routers are only supported when using the "langgraph" backend.

        Args:
            router: A PolicyRouter instance to dynamically route execution.

        Raises:
            RuntimeError: If the backend is not "langgraph".
        """
        if self.backend != "langgraph":
            raise RuntimeError(
                "Policy routers are only supported with the 'langgraph' backend."
            )
        logger.debug(f"Adding policy router: {router}")
        self.policy_routers.append(router)

    def add_evaluator(self, evaluator: NodeEvaluator) -> None:
        """Register a node evaluator for quality assessment.

        Evaluators assess node execution quality after completion. Unlike
        policy checkers, they do not block execution but provide feedback
        and scores for monitoring, RL training, or self-improvement.

        Args:
            evaluator: A NodeEvaluator instance to assess execution quality.
        """
        logger.debug(f"Adding evaluator: {evaluator}")
        self.evaluators.append(evaluator)

    def _check_before(self) -> tuple[dict[str, bool], bool]:
        """Execute all policy checkers before instruction execution.

        Returns:
            A dictionary mapping checker names to their validation results.
            A final boolean indicating if all checkers passed.
        """
        results = {}
        logger.debug(f"Running {len(self.policy_checkers)} policy checkers (before)")
        for checker in self.policy_checkers:
            result = checker.check_before(self.history)

            if result is False:
                results[checker.name] = result
                logger.error(f"Policy checker {checker} failed validation.")

        return results, all(results.values())

    def _route_after(self) -> tuple[dict[str, str | None], str | None]:
        """Determine if execution should be routed to a different node.

        Consults all registered policy routers in order. Returns the first
        non-None routing decision.

        Returns:
            A dictionary mapping checker names to their route destination.
            A final str indicating the final route destination.
        """
        results = {}
        destination = None
        used_router = None
        logger.debug(f"Checking {len(self.policy_routers)} policy routers")
        for router in self.policy_routers:
            decision = router.route_after(self.history)

            if decision:
                results[router.name] = decision
                used_router = router
                destination = decision

        decision_count = sum(1 for v in results.values() if v is not None)
        if decision_count > 1:
            logger.error(
                "Multiple routers decided to route. Fallback to first decision."
            )

        if destination is not None:
            logger.warning(f"Router {used_router} decision made to: {destination}")
        return results, destination

    def _evaluate_node(self) -> dict[str, EvaluationResult]:
        """Execute all evaluators on the most recent node.

        Evaluators assess the quality of the node execution that just completed.
        The node's HistoryItem (including output_state) has already been added
        to history and can be accessed via `history.entries[-1][-1]`.

        Only evaluators whose target_instructions match the current node's
        instruction type will be executed. If target_instructions is None,
        the evaluator runs on all nodes.

        Returns:
            A dictionary mapping evaluator names to their evaluation results.

        Note:
            Evaluator failures are logged but do not raise exceptions or
            interrupt execution. This ensures evaluation does not break
            the workflow.
        """
        results = {}
        current_item = self.history.entries[-1][-1]
        current_instruction = current_item.instruction

        logger.debug(f"Running evaluators for instruction: {current_instruction.name}")

        for evaluator in self.evaluators:
            # Check if this evaluator should run for this instruction type
            if evaluator.target_instructions is not None:
                if current_instruction not in evaluator.target_instructions:
                    logger.debug(
                        f"Skipping evaluator {evaluator.name} "
                        f"(not targeting {current_instruction.name})"
                    )
                    continue

            try:
                result = evaluator.evaluate(self.history)
                results[evaluator.name] = result
                logger.info(
                    f"Evaluator {evaluator.name}: score={result.score:.2f}, "
                    f"passed={result.passed}, feedback={result.feedback}"
                )
            except Exception as e:
                logger.error(
                    f"Evaluator {evaluator.name} failed with error: {e}",
                    exc_info=True,
                )
                # Evaluation failures should not interrupt execution
        return results

    def instruction(
        self, instruction_type: InstructionType
    ) -> Callable[[Callable], Callable]:
        """Decorator to wrap LangGraph node functions with policy governance.

        This decorator adds policy validation, execution history tracking,
        and dynamic routing to LangGraph node functions. It's the core
        integration point between ArbiterOS and LangGraph.

        Args:
            instruction_type: An instruction type from one of the Core enums
                (CognitiveCore, MemoryCore, ExecutionCore, NormativeCore,
                MetacognitiveCore, AdaptiveCore, SocialCore, or AffectiveCore).

        Returns:
            A decorator function that wraps the target node function.

        Example:
            ```python
            from arbiteros_alpha.instructions import CognitiveCore

            @os.instruction(CognitiveCore.GENERATE)
            def generate(state: State) -> State:
                return {"field": "value"}
            # Function now includes policy checks and history tracking
            ```
        """
        # Validate that instruction_type is a valid InstructionType enum
        if not isinstance(instruction_type, InstructionType.__args__):
            raise TypeError(
                f"instruction_type must be an instance of one of the Core enums, got {type(instruction_type)}"
            )

        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs) -> Any:
                logger.debug(
                    f"Executing instruction: {instruction_type.__class__.__name__}.{instruction_type.name}"
                )

                # Capture input state from arguments
                input_state = None
                if self.backend == "vanilla":
                    # For vanilla backend, capture all named arguments
                    sig = inspect.signature(func)
                    bound_args = sig.bind(*args, **kwargs)
                    bound_args.apply_defaults()
                    input_state = bound_args.arguments
                else:
                    # For langgraph backend, usually the first argument is the state
                    input_state = args[0] if args else None

                history_item = HistoryItem(
                    timestamp=datetime.datetime.now(),
                    instruction=instruction_type,
                    input_state=input_state,
                )

                if self.backend == "vanilla":
                    self.history.enter_next_superstep([instruction_type.name])

                self.history.add_entry(history_item)

                history_item.check_policy_results, all_passed = self._check_before()

                result = func(*args, **kwargs)
                logger.debug(f"Instruction {instruction_type.name} returned: {result}")
                history_item.output_state = result

                # Evaluate node execution quality
                if self.evaluators:
                    history_item.evaluation_results = self._evaluate_node()

                history_item.route_policy_results, destination = self._route_after()

                if destination:
                    return Command(update=result, goto=destination)

                return result

            return wrapper

        return decorator

    def _patch_pregel_loop(self) -> None:
        """Patch the Pregel loop to track planned nodes in history.

        This method patches LangGraph's internal PregelLoop.__init__ and tick methods
        to enable superstep tracking across multiple ArbiterOSAlpha instances.

        Mechanism:
            The patch uses a two-level mapping chain to associate PregelLoop instances
            with their corresponding ArbiterOSAlpha instances:

            1. PregelLoop -> Pregel mapping (_loop_to_pregel_map):
               - Patched PregelLoop.__init__ inspects the call stack using inspect.stack()
               - Finds the parent Pregel (CompiledStateGraph) instance that created the loop
               - Stores this relationship in a WeakKeyDictionary for automatic cleanup

            2. Pregel -> ArbiterOSAlpha mapping (_pregel_to_arbiter_map):
               - Established via compile_graph() or register_compiled_graph()
               - Maps each compiled graph to its governing OS instance
               - Also uses WeakKeyDictionary to prevent memory leaks

            3. History tracking via patched tick():
               - Each tick, looks up: PregelLoop -> Pregel -> ArbiterOSAlpha
               - Extracts planned nodes from loop_self.tasks
               - Records them in the correct OS instance's history

        Thread Safety:
            Global patching happens only once per process (checked via _arbiteros_patched).
            Multiple ArbiterOSAlpha instances share the same patched methods but maintain
            separate histories through the mapping chain.

        Notes:
            - Uses WeakKeyDictionary to avoid preventing garbage collection
            - PregelLoop instances are created fresh on each invoke()/stream() call
            - Stack inspection adds minimal overhead (~1-2 frame traversals)
        """
        # Check if already patched globally to avoid duplicate patching
        if hasattr(_loop.PregelLoop.__init__, "_arbiteros_patched"):
            logger.debug("PregelLoop already patched globally")
            return

        # Patch __init__ to establish PregelLoop -> Pregel mapping
        original_init = _loop.PregelLoop.__init__

        def patched_init(loop_self: _loop.PregelLoop, *args, **kwargs):
            # Call the original __init__
            original_init(loop_self, *args, **kwargs)

            # Find the parent Pregel instance from the call stack
            for frame_info in inspect.stack()[1:]:
                frame_locals = frame_info.frame.f_locals
                if "self" in frame_locals:
                    obj = frame_locals["self"]
                    if isinstance(obj, Pregel):
                        _loop_to_pregel_map[loop_self] = obj
                        logger.debug(f"Mapped PregelLoop {loop_self} to Pregel {obj}")
                        break

        # Patch tick to use the mapping for history tracking
        original_tick = _loop.PregelLoop.tick

        def patched_tick(loop_self: _loop.PregelLoop):
            # Call the original method to perform the planning
            result = original_tick(loop_self)

            # === INJECTED CODE ===
            # Look up the chain: PregelLoop -> Pregel -> ArbiterOSAlpha
            pregel_instance = _loop_to_pregel_map.get(loop_self)
            if pregel_instance is not None:
                os_instance = _pregel_to_arbiter_map.get(pregel_instance)
                if os_instance is not None:
                    # This runs after planning is done for the step
                    planned_nodes = [t.name for t in loop_self.tasks.values()]
                    if planned_nodes:
                        os_instance.history.enter_next_superstep(planned_nodes)
                        logger.info(
                            f"Nodes in next superstep for OS {os_instance}: {planned_nodes}"
                        )
            # =====================

            return result

        # Mark as patched to prevent duplicate patching
        patched_init._arbiteros_patched = True
        patched_tick._arbiteros_patched = True
        _loop.PregelLoop.__init__ = patched_init
        _loop.PregelLoop.tick = patched_tick
        logger.debug("PregelLoop.__init__ and tick successfully patched globally")

    def register_compiled_graph(self, compiled_graph: Pregel) -> None:
        """Register a compiled LangGraph (Pregel) to be tracked by this ArbiterOSAlpha.

        This method should be called after compiling a LangGraph to associate
        the resulting Pregel instance (CompiledStateGraph) with this OS instance
        for history tracking.

        Args:
            compiled_graph: The Pregel instance returned from StateGraph.compile().
        """
        _pregel_to_arbiter_map[compiled_graph] = self
        logger.debug(
            f"Registered Pregel {compiled_graph} (type={type(compiled_graph).__name__}) to ArbiterOSAlpha {self}"
        )
        logger.debug(f"Current map size: {len(_pregel_to_arbiter_map)}")
