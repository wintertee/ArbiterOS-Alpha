"""Policy loader for ArbiterOS-alpha.

This module provides functionality to load policies from YAML files, separating
policy class definitions from rule instances. Supports both kernel policies
(read-only) and custom policies (developer-defined).
"""

import logging
import pathlib
from typing import Any

import yaml

from .instructions import InstructionType
from .policy import (
    GraphStructurePolicyChecker,
    HistoryPolicyChecker,
    MetricThresholdPolicyRouter,
    PolicyChecker,
    PolicyRouter,
)

logger = logging.getLogger(__name__)


class PolicyLoader:
    """Loads and instantiates policies from YAML configuration files.

    This class handles loading policies from both kernel policy files (read-only)
    and custom policy files (developer-defined). It separates policy class
    definitions from rule instances, allowing developers to define rules in YAML
    while policy classes remain in code.

    Example YAML structure:
        policy_checkers:
          - type: HistoryPolicyChecker
            name: no_skip_to_execute
            bad_sequence: [GENERATE, TOOL_CALL]
        
        policy_routers:
          - type: MetricThresholdPolicyRouter
            name: revisit_reason_when_low_confidence
            key: confidence
            threshold: 0.7
            target: reason
        
        graph_structure_checkers:
          - type: GraphStructurePolicyChecker
            blacklists:
              - name: no_direct_execute_without_reason
                sequence: [TOOL_CALL, TOOL_CALL]
                level: error
              - name: no_skip_to_execute
                sequence: [GENERATE, TOOL_CALL]
                level: error
    """

    # Registry mapping policy type names to their classes
    POLICY_CLASS_REGISTRY = {
        "HistoryPolicyChecker": HistoryPolicyChecker,
        "MetricThresholdPolicyRouter": MetricThresholdPolicyRouter,
        "GraphStructurePolicyChecker": GraphStructurePolicyChecker,
    }

    @classmethod
    def load_from_file(cls, file_path: str | pathlib.Path) -> dict[str, Any]:
        """Load policy configuration from a YAML file.

        Args:
            file_path: Path to the YAML policy file.

        Returns:
            Dictionary containing loaded policies with keys:
            - policy_checkers: List of PolicyChecker instances
            - policy_routers: List of PolicyRouter instances
            - graph_structure_checkers: List of GraphStructurePolicyChecker instances

        Raises:
            FileNotFoundError: If the policy file does not exist.
            yaml.YAMLError: If the YAML file is malformed.
            ValueError: If policy configuration is invalid.
        """
        file_path = pathlib.Path(file_path)
        if not file_path.exists():
            logger.warning(f"Policy file not found: {file_path}")
            return {
                "policy_checkers": [],
                "policy_routers": [],
                "graph_structure_checkers": [],
            }

        with open(file_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        if config is None:
            logger.warning(f"Policy file is empty: {file_path}")
            return {
                "policy_checkers": [],
                "policy_routers": [],
                "graph_structure_checkers": [],
            }

        return cls._parse_config(config)

    @classmethod
    def _parse_config(cls, config: dict[str, Any] | None) -> dict[str, Any]:
        """Parse YAML configuration into policy instances.

        Args:
            config: Dictionary containing policy configuration from YAML.
                Can be None if the YAML file is empty or only contains comments.

        Returns:
            Dictionary with policy instances organized by type.

        Raises:
            ValueError: If policy configuration is invalid.
        """
        result = {
            "policy_checkers": [],
            "policy_routers": [],
            "graph_structure_checkers": [],
        }

        if config is None:
            return result

        # Load policy checkers
        if "policy_checkers" in config and config["policy_checkers"] is not None:
            for checker_config in config["policy_checkers"]:
                checker = cls._instantiate_checker(checker_config)
                if checker:
                    result["policy_checkers"].append(checker)

        # Load policy routers
        if "policy_routers" in config and config["policy_routers"] is not None:
            for router_config in config["policy_routers"]:
                router = cls._instantiate_router(router_config)
                if router:
                    result["policy_routers"].append(router)

        # Load graph structure checkers
        if (
            "graph_structure_checkers" in config
            and config["graph_structure_checkers"] is not None
        ):
            for checker_config in config["graph_structure_checkers"]:
                checker = cls._instantiate_graph_structure_checker(checker_config)
                if checker:
                    result["graph_structure_checkers"].append(checker)

        return result

    @classmethod
    def _instantiate_checker(cls, config: dict[str, Any]) -> PolicyChecker | None:
        """Instantiate a PolicyChecker from configuration.

        Args:
            config: Dictionary containing checker configuration.

        Returns:
            PolicyChecker instance or None if configuration is invalid.

        Raises:
            ValueError: If policy type is unknown or configuration is invalid.
        """
        policy_type = config.get("type")
        if not policy_type:
            logger.error("Policy checker missing 'type' field")
            return None

        if policy_type not in cls.POLICY_CLASS_REGISTRY:
            logger.error(f"Unknown policy checker type: {policy_type}")
            return None

        policy_class = cls.POLICY_CLASS_REGISTRY[policy_type]

        if policy_type == "HistoryPolicyChecker":
            name = config.get("name")
            bad_sequence = config.get("bad_sequence", [])
            if not name:
                logger.error("HistoryPolicyChecker missing 'name' field")
                return None
            if not bad_sequence:
                logger.error("HistoryPolicyChecker missing 'bad_sequence' field")
                return None

            # Convert instruction names to InstructionType enums
            instruction_sequence = cls._parse_instruction_sequence(bad_sequence)
            return policy_class(name=name, bad_sequence=instruction_sequence)

        logger.warning(f"Unsupported policy checker type: {policy_type}")
        return None

    @classmethod
    def _instantiate_router(cls, config: dict[str, Any]) -> PolicyRouter | None:
        """Instantiate a PolicyRouter from configuration.

        Args:
            config: Dictionary containing router configuration.

        Returns:
            PolicyRouter instance or None if configuration is invalid.

        Raises:
            ValueError: If policy type is unknown or configuration is invalid.
        """
        policy_type = config.get("type")
        if not policy_type:
            logger.error("Policy router missing 'type' field")
            return None

        if policy_type not in cls.POLICY_CLASS_REGISTRY:
            logger.error(f"Unknown policy router type: {policy_type}")
            return None

        policy_class = cls.POLICY_CLASS_REGISTRY[policy_type]

        if policy_type == "MetricThresholdPolicyRouter":
            name = config.get("name")
            key = config.get("key")
            threshold = config.get("threshold")
            target = config.get("target")

            if not all([name, key, threshold is not None, target]):
                logger.error(
                    "MetricThresholdPolicyRouter missing required fields: "
                    "name, key, threshold, target"
                )
                return None

            return policy_class(
                name=name, key=key, threshold=float(threshold), target=target
            )

        logger.warning(f"Unsupported policy router type: {policy_type}")
        return None

    @classmethod
    def _instantiate_graph_structure_checker(
        cls, config: dict[str, Any]
    ) -> GraphStructurePolicyChecker | None:
        """Instantiate a GraphStructurePolicyChecker from configuration.

        Args:
            config: Dictionary containing graph structure checker configuration.

        Returns:
            GraphStructurePolicyChecker instance or None if configuration is invalid.

        Raises:
            ValueError: If policy type is unknown or configuration is invalid.
        """
        policy_type = config.get("type")
        if not policy_type:
            logger.error("Graph structure checker missing 'type' field")
            return None

        if policy_type != "GraphStructurePolicyChecker":
            logger.error(f"Unknown graph structure checker type: {policy_type}")
            return None

        checker = GraphStructurePolicyChecker()

        # Load blacklists if present
        if "blacklists" in config:
            for blacklist_config in config["blacklists"]:
                name = blacklist_config.get("name")
                sequence = blacklist_config.get("sequence", [])
                level = blacklist_config.get("level", "error")

                if not name:
                    logger.error("Blacklist missing 'name' field")
                    continue
                if not sequence:
                    logger.error(f"Blacklist '{name}' missing 'sequence' field")
                    continue

                # Convert instruction names to InstructionType enums or keep as strings
                # GraphStructurePolicyChecker accepts both
                parsed_sequence = cls._parse_instruction_sequence_or_strings(sequence)
                checker.add_blacklist(name=name, sequence=parsed_sequence, level=level)

        return checker

    @classmethod
    def _parse_instruction_sequence(
        cls, sequence: list[str]
    ) -> list[InstructionType]:
        """Parse a list of instruction names into InstructionType enums.

        Args:
            sequence: List of instruction names (e.g., ["GENERATE", "TOOL_CALL"]).

        Returns:
            List of InstructionType enums.

        Raises:
            ValueError: If an instruction name is invalid.
        """
        result = []
        for instr_name in sequence:
            try:
                instruction = _get_instruction_by_name(instr_name)
                result.append(instruction)
            except ValueError as e:
                logger.error(f"Invalid instruction name '{instr_name}': {e}")
                raise
        return result

    @classmethod
    def _parse_instruction_sequence_or_strings(
        cls, sequence: list[str]
    ) -> list[str | InstructionType]:
        """Parse a list of instruction names, keeping as strings or converting to enums.

        GraphStructurePolicyChecker can accept both strings and InstructionType enums,
        so we try to convert but keep as strings if conversion fails (for pattern matching).

        Args:
            sequence: List of instruction names (e.g., ["GENERATE", "TOOL_CALL"]).

        Returns:
            List of InstructionType enums or strings.
        """
        result = []
        for instr_name in sequence:
            try:
                instruction = _get_instruction_by_name(instr_name)
                result.append(instruction)
            except ValueError:
                # Keep as string for pattern matching in GraphStructurePolicyChecker
                result.append(instr_name)
        return result

    @classmethod
    def load_kernel_and_custom_policies(
        cls,
        kernel_policy_path: str | pathlib.Path | None = None,
        custom_policy_path: str | pathlib.Path | None = None,
    ) -> dict[str, Any]:
        """Load policies from both kernel and custom policy files.

        Kernel policies are loaded first, then custom policies. Custom policies
        can extend kernel policies but cannot override them (both are applied).

        Args:
            kernel_policy_path: Path to kernel policy file. If None, uses default
                location: arbiteros_alpha/kernel_policy_list.yaml
            custom_policy_path: Path to custom policy file. If None, uses default
                location: examples/custom_policy_list.yaml

        Returns:
            Dictionary containing all loaded policies with keys:
            - policy_checkers: List of PolicyChecker instances
            - policy_routers: List of PolicyRouter instances
            - graph_structure_checkers: List of GraphStructurePolicyChecker instances
        """
        # Default paths
        if kernel_policy_path is None:
            kernel_policy_path = pathlib.Path(__file__).parent / "kernel_policy_list.yaml"
        if custom_policy_path is None:
            # Try to find custom_policy_list.yaml in examples directory
            examples_dir = pathlib.Path(__file__).parent.parent / "examples"
            custom_policy_path = examples_dir / "custom_policy_list.yaml"

        result = {
            "policy_checkers": [],
            "policy_routers": [],
            "graph_structure_checkers": [],
        }

        # Load kernel policies first (read-only, cannot be modified)
        logger.info(f"Loading kernel policies from: {kernel_policy_path}")
        kernel_policies = cls.load_from_file(kernel_policy_path)
        result["policy_checkers"].extend(kernel_policies["policy_checkers"])
        result["policy_routers"].extend(kernel_policies["policy_routers"])
        result["graph_structure_checkers"].extend(
            kernel_policies["graph_structure_checkers"]
        )

        # Load custom policies (developer-defined, can extend kernel policies)
        logger.info(f"Loading custom policies from: {custom_policy_path}")
        custom_policies = cls.load_from_file(custom_policy_path)
        result["policy_checkers"].extend(custom_policies["policy_checkers"])
        result["policy_routers"].extend(custom_policies["policy_routers"])
        result["graph_structure_checkers"].extend(
            custom_policies["graph_structure_checkers"]
        )

        logger.info(
            f"Loaded {len(result['policy_checkers'])} checkers, "
            f"{len(result['policy_routers'])} routers, "
            f"{len(result['graph_structure_checkers'])} graph structure checkers"
        )

        return result


def _get_instruction_by_name(name: str) -> InstructionType:
    """Get an InstructionType enum by its name.

    Args:
        name: Instruction name (e.g., "GENERATE", "TOOL_CALL").

    Returns:
        InstructionType enum instance.

    Raises:
        ValueError: If instruction name is not found.
    """
    from . import instructions

    # Check all instruction cores
    for core_name in [
        "CognitiveCore",
        "MemoryCore",
        "ExecutionCore",
        "NormativeCore",
        "MetacognitiveCore",
        "AdaptiveCore",
        "SocialCore",
        "AffectiveCore",
    ]:
        core = getattr(instructions, core_name, None)
        if core:
            for instruction in core:
                if instruction.name == name:
                    return instruction

    raise ValueError(f"Unknown instruction name: {name}")

