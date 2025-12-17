"""Custom policy definitions for the plan-and-execute agent.

This module allows developers to define custom policies programmatically for
the plan-and-execute agent. Policies defined here cannot override kernel
policies - conflicts will be detected and reported as errors.

Policies include two types:
1. Alert-only checkers: Only log warnings/alerts, do not block execution
2. Routing policies: Can trigger flow redirection but don't modify node code
"""

from __future__ import annotations

import logging
from typing import Dict, List, Union

from arbiteros_alpha.policy import (
    GraphStructurePolicyChecker,
    HistoryPolicyChecker,
    MetricThresholdPolicyRouter,
    PolicyChecker,
    PolicyRouter,
)

import arbiteros_alpha.instructions as Instr

logger = logging.getLogger(__name__)


PolicyType = Union[PolicyChecker, PolicyRouter, GraphStructurePolicyChecker]


class AlertOnlyPolicyChecker(PolicyChecker):
    """Base class for alert-only policy checkers that only log warnings.
    
    These checkers monitor execution and alert on issues but do not block
    execution or modify node behavior. They are purely observational.
    """

    def __init__(self, name: str, alert_level: str = "warning"):
        """Initialize alert-only checker.
        
        Args:
            name: Name of the policy checker.
            alert_level: Logging level for alerts ("warning" or "info").
        """
        self.name = name
        self.alert_level = alert_level

    def _alert(self, message: str) -> None:
        """Log an alert message without blocking execution.
        
        Args:
            message: Alert message to log.
        """
        if self.alert_level == "warning":
            logger.warning(f"[POLICY ALERT] {self.name}: {message}")
        else:
            logger.info(f"[POLICY ALERT] {self.name}: {message}")


class PlanQualityAlertChecker(AlertOnlyPolicyChecker):
    """Alert-only checker that monitors plan quality and alerts on low quality.
    
    This checker only logs warnings when plan quality is low, but does not
    block execution. The original workflow continues unchanged.
    """

    def __init__(
        self, 
        name: str = "alert_plan_quality", 
        threshold: float = 0.4,
        alert_level: str = "warning",
        target_nodes: list[str] | None = None
    ):
        """Initialize plan quality alert checker.
        
        Args:
            name: Name of the policy checker.
            threshold: Threshold below which to alert (0.0-1.0).
            alert_level: Logging level for alerts.
            target_nodes: List of node names where this checker should apply.
                Defaults to ["planner", "replan"] since only these nodes output plans.
        """
        super().__init__(name, alert_level)
        self.threshold = threshold
        # Note: target_nodes should use function names, not graph node names
        self.target_nodes = target_nodes if target_nodes is not None else ["plan_step", "replan_step"]

    def check_before(self, history: list) -> bool:
        """Check plan quality and alert if below threshold.
        
        This checker calculates plan quality independently from state,
        without requiring node functions to compute it.
        
        Args:
            history: Execution history up to this point.
            
        Returns:
            Always returns True (never blocks execution).
        """
        if not history:
            return True
        
        # Get the last entry's output state
        last_entry = history[-1]
        output_state = last_entry.output_state
        
        # Calculate plan quality from state (independent of node functions)
        plan = output_state.get("plan", [])
        if not plan:
            plan_quality = 0.0
        else:
            step_count = len(plan)
            # Simple quality heuristic: optimal range is 3-7 steps
            if step_count == 0:
                plan_quality = 0.0
            elif step_count > 10:
                plan_quality = 0.3
            elif step_count < 2:
                plan_quality = 0.5
            else:
                plan_quality = min(1.0, 0.7 + 0.1 * (7 - abs(step_count - 5)))
            
            # Check step clarity (simple heuristic)
            avg_step_length = sum(len(step) for step in plan) / max(len(plan), 1)
            clarity_bonus = min(0.2, avg_step_length / 100.0)
            plan_quality = min(1.0, plan_quality + clarity_bonus)
        
        if plan_quality < self.threshold:
            self._alert(
                f"Plan quality {plan_quality:.2f} is below threshold {self.threshold}. "
                "Consider replanning for better results."
            )
        
        return True


class StepCountAlertChecker(AlertOnlyPolicyChecker):
    """Alert-only checker that monitors step count and alerts on high counts.
    
    This checker only logs warnings when step count is high, but does not
    block execution. The original workflow continues unchanged.
    """

    def __init__(
        self, 
        name: str = "alert_step_count", 
        max_steps: int = 15,
        alert_level: str = "warning",
        target_nodes: list[str] | None = None
    ):
        """Initialize step count alert checker.
        
        Args:
            name: Name of the policy checker.
            max_steps: Threshold above which to alert.
            alert_level: Logging level for alerts.
            target_nodes: List of node names where this checker should apply.
                Defaults to ["agent"] since only this node executes steps.
        """
        super().__init__(name, alert_level)
        self.max_steps = max_steps
        # Note: target_nodes should use function names, not graph node names
        self.target_nodes = target_nodes if target_nodes is not None else ["execute_step"]

    def check_before(self, history: list) -> bool:
        """Check step count and alert if above threshold.
        
        This checker calculates step count independently from state,
        without requiring node functions to compute it.
        
        Args:
            history: Execution history up to this point.
            
        Returns:
            Always returns True (never blocks execution).
        """
        if not history:
            return True
        
        # Get the last entry's output state
        last_entry = history[-1]
        output_state = last_entry.output_state
        
        # Calculate step count from past_steps (independent of node functions)
        past_steps = output_state.get("past_steps", [])
        step_count = len(past_steps)
        max_allowed = output_state.get("max_steps", self.max_steps)
        
        if step_count >= max_allowed:
            self._alert(
                f"Step count {step_count} has reached or exceeded threshold {max_allowed}. "
                "Consider replanning to prevent potential infinite loops."
            )
        
        return True


class ErrorCountAlertChecker(AlertOnlyPolicyChecker):
    """Alert-only checker that monitors error count and alerts on high counts.
    
    This checker only logs warnings when error count is high, but does not
    block execution. The original workflow continues unchanged.
    """

    def __init__(
        self, 
        name: str = "alert_error_count", 
        max_errors: int = 3,
        alert_level: str = "warning",
        target_nodes: list[str] | None = None
    ):
        """Initialize error count alert checker.
        
        Args:
            name: Name of the policy checker.
            max_errors: Threshold above which to alert.
            alert_level: Logging level for alerts.
            target_nodes: List of node names where this checker should apply.
                Defaults to ["agent"] since only this node can encounter errors.
        """
        super().__init__(name, alert_level)
        self.max_errors = max_errors
        # Note: target_nodes should use function names, not graph node names
        self.target_nodes = target_nodes if target_nodes is not None else ["execute_step"]

    def check_before(self, history: list) -> bool:
        """Check error count and alert if above threshold.
        
        This checker calculates error count independently from state,
        without requiring node functions to compute it.
        
        Args:
            history: Execution history up to this point.
            
        Returns:
            Always returns True (never blocks execution).
        """
        if not history:
            return True
        
        # Get the last entry's output state
        last_entry = history[-1]
        output_state = last_entry.output_state
        
        # Calculate error count from past_steps (independent of node functions)
        # Simple heuristic: check for error indicators in past step results
        past_steps = output_state.get("past_steps", [])
        error_indicators = [
            "error",
            "failed",
            "unable to",
            "cannot",
            "connection issue",
            "not available",
        ]
        error_count = sum(
            1 for _, result in past_steps
            if any(indicator.lower() in str(result).lower() for indicator in error_indicators)
        )
        
        if error_count >= self.max_errors:
            self._alert(
                f"Error count {error_count} has reached or exceeded threshold {self.max_errors}. "
                "Consider replanning to recover from errors."
            )
        
        return True


class PlanQualityPolicyRouter(PolicyRouter):
    """Policy router that routes based on calculated plan quality.
    
    This router calculates plan quality independently and can trigger
    routing without requiring node functions to compute it.
    """

    def __init__(
        self,
        name: str = "replan_on_low_plan_quality",
        threshold: float = 0.5,
        target: str = "planner",
        target_nodes: list[str] | None = None
    ):
        """Initialize plan quality router.
        
        Args:
            name: Name of the router.
            threshold: Minimum plan quality required (0.0-1.0).
            target: Target node to route to when quality is below threshold.
            target_nodes: List of node names where this router should apply.
                Defaults to ["planner", "replan"] since only these nodes output plans.
        """
        self.name = name
        self.threshold = threshold
        self.target = target
        # Note: target_nodes should use function names, not graph node names
        # Function names: plan_step, execute_step, replan_step
        # Graph node names: planner, agent, replan
        self.target_nodes = target_nodes if target_nodes is not None else ["plan_step", "replan_step"]

    def route_after(self, history: list) -> str | None:
        """Route to target node if plan quality is below threshold.
        
        Calculates plan quality independently from state.
        Prevents infinite loops by checking if we just came from the target node.
        
        Args:
            history: Execution history including the just-executed instruction.
            
        Returns:
            Target node name if quality < threshold, None otherwise.
        """
        if not history:
            return None
        
        last_entry = history[-1]
        current_node_name = last_entry.node_name
        output_state = last_entry.output_state
        
        # Debug: Print current node and output state keys
        print(f"[DEBUG PlanQualityPolicyRouter] Current node: {current_node_name}, "
              f"Output state keys: {list(output_state.keys())}, "
              f"Target nodes: {self.target_nodes}, History length: {len(history)}")
        
        # Early return if output state doesn't contain "plan" key
        # This happens when replan_step returns "response" instead of "plan"
        if "plan" not in output_state:
            print(f"[DEBUG PlanQualityPolicyRouter] No 'plan' key in output_state, skipping")
            return None
        
        # Prevent infinite loops: don't route if we just came from the target
        # Check if the previous node was the target (to avoid planner -> replan -> planner loops)
        # Only check if we have at least 2 entries in history
        if len(history) >= 2:
            prev_entry = history[-2]
            prev_node = prev_entry.node_name
            if prev_node == self.target:
                # Just came from target, don't route back immediately
                print(f"[DEBUG PlanQualityPolicyRouter] Just came from target {self.target}, skipping to prevent loop")
                return None
        
        # Calculate plan quality independently
        # If output_state doesn't have "plan" key, it means replan_step returned "response" instead
        # In this case, we should not check plan quality (workflow is ending)
        if "plan" not in output_state:
            print(f"[DEBUG PlanQualityPolicyRouter] No 'plan' key in output_state (likely returned 'response'), skipping quality check")
            return None
        
        plan = output_state.get("plan", [])
        print(f"[DEBUG PlanQualityPolicyRouter] Plan: {plan}, Node: {current_node_name}")
        if not plan:
            plan_quality = 0.0
        else:
            step_count = len(plan)
            if step_count == 0:
                plan_quality = 0.0
            elif step_count > 10:
                plan_quality = 0.3
            elif step_count < 2:
                plan_quality = 0.5
            else:
                plan_quality = min(1.0, 0.7 + 0.1 * (7 - abs(step_count - 5)))
            
            avg_step_length = sum(len(step) for step in plan) / max(len(plan), 1)
            clarity_bonus = min(0.2, avg_step_length / 100.0)
            plan_quality = min(1.0, plan_quality + clarity_bonus)
        
        if plan_quality < self.threshold:
            logger.info(
                f"[POLICY ROUTER] {self.name}: Plan quality {plan_quality:.2f} below "
                f"threshold {self.threshold}, routing to {self.target}"
            )
            return self.target
        
        return None

    def get_metrics(self, history: list) -> dict[str, float] | None:
        """Calculate and return plan quality metrics to be merged into output state.
        
        This method is called by the framework to collect metrics that will be
        automatically merged into the output state, allowing state tracking
        without modifying node functions.
        
        Args:
            history: Execution history including the just-executed instruction.
            
        Returns:
            Dictionary with plan_quality_score (if plan found) or response_quality_score
            (if response found), or None if neither found.
        """
        if not history:
            return None
        
        last_entry = history[-1]
        output_state = last_entry.output_state
        metrics = {}
        
        # Calculate plan quality if plan exists
        if "plan" in output_state:
            plan = output_state.get("plan", [])
            if not plan:
                plan_quality = 0.0
            else:
                step_count = len(plan)
                if step_count == 0:
                    plan_quality = 0.0
                elif step_count > 10:
                    plan_quality = 0.3
                elif step_count < 2:
                    plan_quality = 0.5
                else:
                    plan_quality = min(1.0, 0.7 + 0.1 * (7 - abs(step_count - 5)))
                
                avg_step_length = sum(len(step) for step in plan) / max(len(plan), 1)
                clarity_bonus = min(0.2, avg_step_length / 100.0)
                plan_quality = min(1.0, plan_quality + clarity_bonus)
            
            metrics["plan_quality_score"] = plan_quality
        
        # Calculate response quality if response exists
        if "response" in output_state:
            response = output_state.get("response", "")
            if response:
                # Simple quality heuristic: longer responses with key information are better
                response_length = len(response)
                has_answer_indicators = any(
                    word in response.lower()
                    for word in ["is", "are", "was", "were", "the", "answer", "result"]
                )
                response_quality = min(1.0, 0.5 + 0.3 * (response_length > 50) + 0.2 * has_answer_indicators)
                metrics["response_quality_score"] = response_quality
            else:
                metrics["response_quality_score"] = 0.0
        
        return metrics if metrics else None


class StepCountPolicyRouter(PolicyRouter):
    """Policy router that routes based on calculated step count.
    
    This router calculates step count independently and can trigger
    routing without requiring node functions to compute it.
    """

    def __init__(
        self,
        name: str = "replan_on_high_step_count",
        threshold: float = 10.0,
        target: str = "planner",
        target_nodes: list[str] | None = None
    ):
        """Initialize step count router.
        
        Args:
            name: Name of the router.
            threshold: Maximum step count before routing.
            target: Target node to route to when step count exceeds threshold.
            target_nodes: List of node names where this router should apply.
                Defaults to ["execute_step"] since only this node executes steps.
                Note: Use function names, not graph node names.
        """
        self.name = name
        self.threshold = threshold
        self.target = target
        # Note: target_nodes should use function names, not graph node names
        self.target_nodes = target_nodes if target_nodes is not None else ["execute_step"]

    def route_after(self, history: list) -> str | None:
        """Route to target node if step count exceeds threshold.
        
        Calculates step count independently from state.
        Prevents infinite loops by checking if we just came from the target node.
        
        Args:
            history: Execution history including the just-executed instruction.
            
        Returns:
            Target node name if step_count >= threshold, None otherwise.
        """
        if not history:
            return None
        
        # Prevent infinite loops: don't route if we just came from the target
        # Only check if we have at least 2 entries in history
        if len(history) >= 2:
            prev_entry = history[-2]
            prev_node = prev_entry.node_name
            if prev_node == self.target:
                # Just came from target, don't route back immediately
                return None
        
        last_entry = history[-1]
        output_state = last_entry.output_state
        
        # Calculate step count independently
        past_steps = output_state.get("past_steps", [])
        step_count = len(past_steps)
        
        if step_count >= self.threshold:
            logger.info(
                f"[POLICY ROUTER] {self.name}: Step count {step_count} exceeds "
                f"threshold {self.threshold}, routing to {self.target}"
            )
            return self.target
        
        return None

    def get_metrics(self, history: list) -> dict[str, int] | None:
        """Calculate and return step count metrics to be merged into output state.
        
        Args:
            history: Execution history including the just-executed instruction.
            
        Returns:
            Dictionary with step_count, or None if not applicable.
        """
        if not history:
            return None
        
        last_entry = history[-1]
        output_state = last_entry.output_state
        
        # Calculate step count from past_steps
        past_steps = output_state.get("past_steps", [])
        step_count = len(past_steps)
        
        return {"step_count": step_count}


class ExecutionSuccessPolicyRouter(PolicyRouter):
    """Policy router that routes based on calculated execution success rate.
    
    This router calculates execution success rate independently and can trigger
    routing without requiring node functions to compute it.
    """

    def __init__(
        self,
        name: str = "replan_on_low_execution_success",
        threshold: float = 0.5,
        target: str = "planner",
        target_nodes: list[str] | None = None
    ):
        """Initialize execution success router.
        
        Args:
            name: Name of the router.
            threshold: Minimum execution success rate required (0.0-1.0).
            target: Target node to route to when success rate is below threshold.
            target_nodes: List of node names where this router should apply.
                Defaults to ["execute_step"] since only this node executes steps.
                Note: Use function names, not graph node names.
        """
        self.name = name
        self.threshold = threshold
        self.target = target
        # Note: target_nodes should use function names, not graph node names
        self.target_nodes = target_nodes if target_nodes is not None else ["execute_step"]

    def route_after(self, history: list) -> str | None:
        """Route to target node if execution success rate is below threshold.
        
        Calculates execution success rate independently from state.
        Prevents infinite loops by checking if we just came from the target node.
        
        Args:
            history: Execution history including the just-executed instruction.
            
        Returns:
            Target node name if success_rate < threshold, None otherwise.
        """
        if not history:
            return None
        
        # Prevent infinite loops: don't route if we just came from the target
        # Only check if we have at least 2 entries in history
        if len(history) >= 2:
            prev_entry = history[-2]
            prev_node = prev_entry.node_name
            if prev_node == self.target:
                # Just came from target, don't route back immediately
                return None
        
        last_entry = history[-1]
        output_state = last_entry.output_state
        
        # Calculate execution success rate independently
        past_steps = output_state.get("past_steps", [])
        if not past_steps:
            return None
        
        error_indicators = [
            "error",
            "failed",
            "unable to",
            "cannot",
            "connection issue",
            "not available",
        ]
        error_count = sum(
            1 for _, result in past_steps
            if any(indicator.lower() in str(result).lower() for indicator in error_indicators)
        )
        
        total_steps = len(past_steps)
        success_steps = total_steps - error_count
        execution_success_score = success_steps / max(total_steps, 1)
        
        if execution_success_score < self.threshold:
            logger.info(
                f"[POLICY ROUTER] {self.name}: Execution success rate {execution_success_score:.2f} "
                f"below threshold {self.threshold}, routing to {self.target}"
            )
            return self.target
        
        return None

    def get_metrics(self, history: list) -> dict[str, float] | None:
        """Calculate and return execution success metrics to be merged into output state.
        
        Args:
            history: Execution history including the just-executed instruction.
            
        Returns:
            Dictionary with execution_success_score and error_count, or None if not applicable.
        """
        if not history:
            return None
        
        last_entry = history[-1]
        output_state = last_entry.output_state
        
        # Calculate execution success rate independently
        past_steps = output_state.get("past_steps", [])
        if not past_steps:
            return None
        
        error_indicators = [
            "error",
            "failed",
            "unable to",
            "cannot",
            "connection issue",
            "not available",
        ]
        error_count = sum(
            1 for _, result in past_steps
            if any(indicator.lower() in str(result).lower() for indicator in error_indicators)
        )
        
        total_steps = len(past_steps)
        success_steps = total_steps - error_count
        execution_success_score = success_steps / max(total_steps, 1)
        
        return {
            "execution_success_score": execution_success_score,
            "error_count": error_count,
        }


def get_policies() -> Dict[str, List[PolicyType]]:
    """Return custom policy instances for the plan-and-execute agent.

    Policies are divided into two categories:
    1. Alert-only checkers: Monitor and alert on issues without blocking execution
    2. Routing policies: Can redirect flow but don't modify node code (DISABLED by default)
    
    NOTE: Routing policies are disabled by default because they can interfere with
    the original workflow logic. The workflow should match the notebook behavior,
    with policies only monitoring and alerting, not changing the execution flow.
    To enable routing policies, uncomment them in the policy_routers list below.

    Returns:
        Dictionary with keys:
        - policy_checkers: List of PolicyChecker instances (alert-only)
        - policy_routers: List of PolicyRouter instances (currently empty to preserve workflow)
        - graph_structure_checkers: List of GraphStructurePolicyChecker instances
    """
    return {
        "policy_checkers": [
            # Alert-only checkers: Only log warnings, never block execution
            PlanQualityAlertChecker(
                name="alert_plan_quality",
                threshold=0.4,
                alert_level="warning",
            ),
            StepCountAlertChecker(
                name="alert_step_count",
                max_steps=15,
                alert_level="warning",
            ),
            ErrorCountAlertChecker(
                name="alert_error_count",
                max_errors=3,
                alert_level="warning",
            ),
        ],
        "policy_routers": [
            # Routing policies: DISABLED by default to maintain original workflow
            # These routers can redirect flow but may interfere with normal execution.
            # Uncomment to enable routing-based governance (use with caution).
            # 
            PlanQualityPolicyRouter(
                name="replan_on_low_plan_quality",
                threshold=0.5,
                target="planner",
                target_nodes=["plan_step", "replan_step"],  # Use function names, not graph node names
            ),
            ExecutionSuccessPolicyRouter(
                name="replan_on_low_execution_success",
                threshold=0.5,
                target="planner",
                target_nodes=["execute_step"],  # Use function names, not graph node names
            ),
            StepCountPolicyRouter(
                name="replan_on_high_step_count",
                threshold=10.0,
                target="replan",
                target_nodes=["execute_step"],  # Use function names, not graph node names
            ),
        ],
        "graph_structure_checkers": [
            # Graph structure checkers: Validate workflow structure (alert-only)
            GraphStructurePolicyChecker().add_blacklist(
                name="no_execution_without_plan",
                sequence=["agent", "agent"],  # Cannot execute twice without replanning
                level="warning",  # Warning level: only alerts, doesn't block
            ),
            GraphStructurePolicyChecker().add_blacklist(
                name="no_infinite_replan_loop",
                sequence=["replan", "replan"],
                level="warning",  # Warning level: only alerts, doesn't block
            ),
        ],
    }
