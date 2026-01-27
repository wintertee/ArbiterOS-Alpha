"""ArbiterOS-alpha: Policy-driven governance layer for LangGraph.

This package provides lightweight governance for LangGraph workflows through
policy-based validation and dynamic routing, with time-travel checkpoint support.

Safety Features:
- PolicyCheckers that BLOCK unsafe operations (not just log warnings)
- Verification requirement enforcement for high-risk actions
- Human-in-the-loop interrupt capability for critical decisions
- Safe fallback routing when conditions are ambiguous
"""

from .core import ArbiterOSAlpha
from .evaluation import EvaluationResult, NodeEvaluator, ThresholdEvaluator
from .history import History, HistoryItem
from .policy import (
    HistoryPolicyChecker,
    HumanInterruptPolicyChecker,
    HumanInterruptRequest,
    MetricThresholdPolicyRouter,
    PolicyChecker,
    PolicyRouter,
    VerificationRequirementChecker,
)

__all__ = [
    # Core
    "ArbiterOSAlpha",
    # Evaluation
    "EvaluationResult",
    "NodeEvaluator",
    "ThresholdEvaluator",
    # History
    "History",
    "HistoryItem",
    # Policy - Base classes
    "PolicyChecker",
    "PolicyRouter",
    # Policy - Built-in checkers
    "HistoryPolicyChecker",
    "VerificationRequirementChecker",
    "HumanInterruptPolicyChecker",
    # Policy - Built-in routers
    "MetricThresholdPolicyRouter",
    # Policy - Exceptions
    "HumanInterruptRequest",
]
