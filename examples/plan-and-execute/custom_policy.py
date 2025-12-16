"""Custom policy definitions for the plan-and-execute agent.

This module allows developers to define custom policies programmatically for
the plan-and-execute agent. Policies defined here cannot override kernel
policies - conflicts will be detected and reported as errors.

Policies include:
- Plan quality validation
- Step count limits
- Error handling and recovery
- Execution success monitoring
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


class PlanQualityPolicyChecker(PolicyChecker):
    """Policy checker that validates plan quality before execution.
    
    Prevents execution if plan quality is too low, forcing replanning.
    """

    def __init__(self, name: str = "check_plan_quality", threshold: float = 0.4):
        """Initialize plan quality checker.
        
        Args:
            name: Name of the policy checker.
            threshold: Minimum plan quality score required (0.0-1.0).
        """
        self.name = name
        self.threshold = threshold

    def check_before(self, history: list) -> bool:
        """Check if plan quality meets threshold before execution.
        
        Args:
            history: Execution history up to this point.
            
        Returns:
            True if plan quality is acceptable, False otherwise.
            
        Raises:
            RuntimeError: If plan quality is below threshold.
        """
        if not history:
            return True
        
        # Get the last entry's output state
        last_entry = history[-1]
        output_state = last_entry.output_state
        
        plan_quality = output_state.get("plan_quality_score", 1.0)
        
        if plan_quality < self.threshold:
            logger_msg = (
                f"Plan quality {plan_quality:.2f} below threshold {self.threshold}. "
                "Replanning required."
            )
            raise RuntimeError(logger_msg)
        
        return True


class StepCountPolicyChecker(PolicyChecker):
    """Policy checker that limits the number of execution steps.
    
    Prevents infinite loops by limiting total steps before forcing replanning.
    """

    def __init__(self, name: str = "check_step_count", max_steps: int = 15):
        """Initialize step count checker.
        
        Args:
            name: Name of the policy checker.
            max_steps: Maximum allowed steps before replanning.
        """
        self.name = name
        self.max_steps = max_steps

    def check_before(self, history: list) -> bool:
        """Check if step count is within limits.
        
        Args:
            history: Execution history up to this point.
            
        Returns:
            True if step count is acceptable, False otherwise.
            
        Raises:
            RuntimeError: If step count exceeds maximum.
        """
        if not history:
            return True
        
        # Get the last entry's output state
        last_entry = history[-1]
        output_state = last_entry.output_state
        
        step_count = output_state.get("step_count", 0)
        max_allowed = output_state.get("max_steps", self.max_steps)
        
        if step_count >= max_allowed:
            logger_msg = (
                f"Step count {step_count} exceeds maximum {max_allowed}. "
                "Replanning required to prevent infinite loop."
            )
            raise RuntimeError(logger_msg)
        
        return True


class ErrorCountPolicyChecker(PolicyChecker):
    """Policy checker that limits consecutive errors.
    
    Prevents continued execution after too many errors, forcing replanning.
    """

    def __init__(self, name: str = "check_error_count", max_errors: int = 3):
        """Initialize error count checker.
        
        Args:
            name: Name of the policy checker.
            max_errors: Maximum allowed consecutive errors.
        """
        self.name = name
        self.max_errors = max_errors

    def check_before(self, history: list) -> bool:
        """Check if error count is within limits.
        
        Args:
            history: Execution history up to this point.
            
        Returns:
            True if error count is acceptable, False otherwise.
            
        Raises:
            RuntimeError: If error count exceeds maximum.
        """
        if not history:
            return True
        
        # Get the last entry's output state
        last_entry = history[-1]
        output_state = last_entry.output_state
        
        error_count = output_state.get("error_count", 0)
        
        if error_count >= self.max_errors:
            logger_msg = (
                f"Error count {error_count} exceeds maximum {self.max_errors}. "
                "Replanning required to recover from errors."
            )
            raise RuntimeError(logger_msg)
        
        return True


def get_policies() -> Dict[str, List[PolicyType]]:
    """Return custom policy instances for the plan-and-execute agent.

    Returns:
        Dictionary with keys:
        - policy_checkers: List of PolicyChecker instances
        - policy_routers: List of PolicyRouter instances
        - graph_structure_checkers: List of GraphStructurePolicyChecker instances
    """
    return {
        "policy_checkers": [
            PlanQualityPolicyChecker(
                name="check_plan_quality_before_execution",
                threshold=0.4,
            ),
            StepCountPolicyChecker(
                name="check_step_count_limit",
                max_steps=15,
            ),
            ErrorCountPolicyChecker(
                name="check_error_count_limit",
                max_errors=3,
            ),
        ],
        "policy_routers": [
            # Route back to planner if plan quality is too low
            MetricThresholdPolicyRouter(
                name="replan_on_low_plan_quality",
                key="plan_quality_score",
                threshold=0.5,
                target="planner",
            ),
            # Route back to planner if execution success rate is too low
            MetricThresholdPolicyRouter(
                name="replan_on_low_execution_success",
                key="execution_success_score",
                threshold=0.5,
                target="planner",
            ),
            # Route back to planner if too many steps executed
            # This is handled by StepCountPolicyChecker, but router provides alternative
            MetricThresholdPolicyRouter(
                name="replan_on_high_step_count",
                key="step_count",
                threshold=10.0,  # Note: threshold is compared as float
                target="planner",
            ),
        ],
        "graph_structure_checkers": [
            # Prevent direct execution without planning
            GraphStructurePolicyChecker().add_blacklist(
                name="no_execution_without_plan",
                sequence=["agent", "agent"],  # Cannot execute twice without replanning
                level="warning",
            ),
            # Prevent infinite replanning loops
            GraphStructurePolicyChecker().add_blacklist(
                name="no_infinite_replan_loop",
                sequence=["replan", "replan"],
                level="warning",
            ),
        ],
    }
