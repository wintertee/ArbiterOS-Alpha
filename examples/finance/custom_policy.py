"""Custom policy definitions for the finance example.

This module mirrors the GUI agent's custom policy entry point so that the
finance example exposes the same extension surface area. The finance demo
currently relies on YAML-defined MetricThresholdPolicyRouters, so the lists
returned here are empty. Developers can add HistoryPolicyChecker or
GraphStructurePolicyChecker instances programmatically when needed.
"""

from __future__ import annotations

from typing import Dict, List, Union

from arbiteros_alpha.policy import (
    GraphStructurePolicyChecker,
    PolicyChecker,
    PolicyRouter,
)


PolicyType = Union[PolicyChecker, PolicyRouter, GraphStructurePolicyChecker]


def get_policies() -> Dict[str, List[PolicyType]]:
    """Return custom policy instances for the finance agent."""
    return {
        "policy_checkers": [],
        "policy_routers": [],
        "graph_structure_checkers": [],
    }


