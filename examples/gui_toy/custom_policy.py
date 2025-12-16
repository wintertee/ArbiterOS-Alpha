"""Custom policy definitions for developers.

This file allows developers to define custom policies programmatically.
Policies defined here cannot override kernel policies - conflicts will be detected
and reported as errors.

Example:
    from arbiteros_alpha.policy import HistoryPolicyChecker, MetricThresholdPolicyRouter
    import arbiteros_alpha.instructions as Instr

    def get_policies():
        return {
            "policy_checkers": [
                HistoryPolicyChecker(
                    name="custom_checker",
                    bad_sequence=[Instr.GENERATE, Instr.TOOL_CALL]
                )
            ],
            "policy_routers": [
                MetricThresholdPolicyRouter(
                    name="custom_router",
                    key="confidence",
                    threshold=0.8,
                    target="retry"
                )
            ],
            "graph_structure_checkers": []
        }
"""

# Developers can define policies here
# This is a template - uncomment and modify as needed

# from arbiteros_alpha.policy import (
#     HistoryPolicyChecker,
#     MetricThresholdPolicyRouter,
#     GraphStructurePolicyChecker,
# )
# import arbiteros_alpha.instructions as Instr


def get_policies():
    """Return custom policy instances.

    Returns:
        Dictionary with keys:
        - policy_checkers: List of PolicyChecker instances
        - policy_routers: List of PolicyRouter instances
        - graph_structure_checkers: List of GraphStructurePolicyChecker instances
    """
    # Example: Define custom policies here
    # return {
    #     "policy_checkers": [
    #         HistoryPolicyChecker(
    #             name="custom_checker",
    #             bad_sequence=[Instr.GENERATE, Instr.TOOL_CALL]
    #         )
    #     ],
    #     "policy_routers": [
    #         MetricThresholdPolicyRouter(
    #             name="custom_router",
    #             key="confidence",
    #             threshold=0.8,
    #             target="retry"
    #         )
    #     ],
    #     "graph_structure_checkers": []
    # }

    # Empty by default - developers can add their policies
    return {
        "policy_checkers": [],
        "policy_routers": [],
        "graph_structure_checkers": [],
    }

