"""Core components for ArbiterOS-alpha.

This module contains the main classes and functionality for policy-driven
governance of LangGraph execution.
"""

import datetime
import functools
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, TypedDict, Union

from langgraph.graph import StateGraph
from langgraph.types import Command

from .instructions import InstructionType
from .policy import GraphStructurePolicyChecker, PolicyChecker, PolicyRouter
from .policy_loader import PolicyLoader
from .schemas import (
    get_input_schema,
    get_output_schema,
    register_input_schema,
    register_output_schema,
    validate_input,
    validate_output,
)

if TYPE_CHECKING:
    from langchain_core.runnables.graph import Graph
    from langgraph.graph.state import CompiledStateGraph

logger = logging.getLogger(__name__)


@dataclass
class History:
    """The minimal OS metadata for tracking instruction execution.

    Attributes:
        timestamp: When the instruction was executed.
        instruction: The instruction type that was executed.
        node_name: The name of the node/function that was executed.
        input_state: The state passed to the instruction.
        output_state: The state returned by the instruction.
        check_policy_results: Results of policy checkers (name -> passed/failed).
        route_policy_results: Results of policy routers (name -> target or None).
    """

    timestamp: datetime.datetime
    instruction: InstructionType
    node_name: str
    input_state: dict[str, Any]
    output_state: dict[str, Any] = field(default_factory=dict)
    check_policy_results: dict[str, bool] = field(default_factory=dict)
    route_policy_results: dict[str, str | None] = field(default_factory=dict)


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
        >>> os.add_policy_checker(HistoryPolicyChecker("require_verification",["generate", "execute"]))
        >>> os.add_policy_router(ConfidencePolicyRouter("confidence", 0.5, "retry"))
        >>> @os.instruction("generate")
        ... def generate(state): return {"result": "output"}
    """

    def __init__(self, validate_schemas: bool = False, strict_schema_validation: bool = False):
        """Initialize ArbiterOSAlpha with empty history and no policies.

        Args:
            validate_schemas: If True, enable input/output schema validation
                for all instructions. Defaults to False.
            strict_schema_validation: If True, require all schema fields to be
                present. If False, only validate that provided fields match
                schema types. Only applies if validate_schemas is True.
                Defaults to False.
        """
        self.history: list[History] = []
        self.policy_checkers: list[PolicyChecker] = []
        self.policy_routers: list[PolicyRouter] = []
        self.graph_structure_checkers: list[GraphStructurePolicyChecker] = []
        # Map from function to instruction type for graph structure validation
        self._function_to_instruction: dict[Callable, InstructionType] = {}
        # Schema validation settings
        self.validate_schemas = validate_schemas
        self.strict_schema_validation = strict_schema_validation

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

    def add_graph_structure_checker(
        self, checker: GraphStructurePolicyChecker
    ) -> None:
        """Register a graph structure checker for pre-execution validation.

        Graph structure checkers validate the graph structure before execution
        to ensure it doesn't violate any blacklisted sequences. This is useful
        for catching structural violations before the graph runs.

        Args:
            checker: A GraphStructurePolicyChecker instance to validate graph structure.
        """
        logger.debug(f"Adding graph structure checker: {checker}")
        self.graph_structure_checkers.append(checker)

    def load_policies(
        self,
        kernel_policy_path: str | None = None,
        custom_policy_path: str | None = None,
    ) -> None:
        """Load policies from YAML files and register them.

        This method loads policies from both kernel policy files (read-only,
        defined by the kernel) and custom policy files (developer-defined).
        Kernel policies are loaded first, then custom policies. Both sets
        of policies are applied.

        Args:
            kernel_policy_path: Optional path to kernel policy file.
                If None, uses default: arbiteros_alpha/kernel_policy_list.yaml
            custom_policy_path: Optional path to custom policy file.
                If None, uses default: examples/custom_policy_list.yaml

        Example:
            >>> os = ArbiterOSAlpha()
            >>> # Load from default locations
            >>> os.load_policies()
            >>> # Or specify custom paths
            >>> os.load_policies(
            ...     kernel_policy_path="path/to/kernel_policies.yaml",
            ...     custom_policy_path="path/to/custom_policies.yaml"
            ... )
        """
        policies = PolicyLoader.load_kernel_and_custom_policies(
            kernel_policy_path=kernel_policy_path,
            custom_policy_path=custom_policy_path,
        )

        # Register all loaded policies
        for checker in policies["policy_checkers"]:
            self.add_policy_checker(checker)

        for router in policies["policy_routers"]:
            self.add_policy_router(router)

        for checker in policies["graph_structure_checkers"]:
            self.add_graph_structure_checker(checker)

        logger.info(
            f"Loaded and registered {len(policies['policy_checkers'])} checkers, "
            f"{len(policies['policy_routers'])} routers, "
            f"{len(policies['graph_structure_checkers'])} graph structure checkers"
        )

    def get_instruction_schema(
        self, instruction_type: InstructionType
    ) -> tuple[type, type]:
        """Get input and output schemas for an instruction type.

        Args:
            instruction_type: The instruction type to get schemas for.

        Returns:
            A tuple of (input_schema_class, output_schema_class).

        Example:
            >>> from arbiteros_alpha.instructions import CognitiveCore
            >>> os = ArbiterOSAlpha()
            >>> input_schema, output_schema = os.get_instruction_schema(CognitiveCore.GENERATE)
            >>> print(input_schema.__name__)  # GenerateInputSchema
        """
        return get_input_schema(instruction_type), get_output_schema(instruction_type)

    def register_instruction_schema(
        self,
        instruction_type: InstructionType,
        input_schema: type | None = None,
        output_schema: type | None = None,
    ) -> None:
        """Register custom schemas for an instruction type.

        This allows users to define custom input/output schemas for specific
        instruction types, overriding the default schemas.

        Args:
            instruction_type: The instruction type to register schemas for.
            input_schema: Optional custom input schema TypedDict class.
            output_schema: Optional custom output schema TypedDict class.

        Example:
            >>> from typing import TypedDict
            >>> class CustomInput(TypedDict):
            ...     custom_field: str
            >>> os.register_instruction_schema(
            ...     CognitiveCore.GENERATE,
            ...     input_schema=CustomInput
            ... )
        """
        if input_schema:
            register_input_schema(instruction_type, input_schema)
            logger.debug(f"Registered custom input schema for {instruction_type.name}")
        if output_schema:
            register_output_schema(instruction_type, output_schema)
            logger.debug(f"Registered custom output schema for {instruction_type.name}")

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
    
    def _extract_node_to_instruction(
        self, graph: Union[StateGraph, "Graph", "CompiledStateGraph", Any]
    ) -> dict[str, InstructionType]:
        """Extract node name to instruction type mapping from a graph object.
        
        Args:
            graph: A StateGraph, CompiledStateGraph, or Graph object.
        
        Returns:
            A dictionary mapping node names to their instruction types.
        """
        node_to_instruction: dict[str, InstructionType] = {}
        
        # If it's a compiled graph, get the Graph object first
        if hasattr(graph, "get_graph"):
            graph = graph.get_graph()
        
        # If it's a StateGraph builder, access nodes directly
        if hasattr(graph, "nodes"):
            # StateGraph builder has nodes as a dict
            # Values are StateNodeSpec objects, not functions directly
            for node_name, node_spec in graph.nodes.items():
                # Extract the actual function from StateNodeSpec
                actual_func = None
                
                # StateNodeSpec has a 'runnable' attribute which is a RunnableCallable
                if hasattr(node_spec, "runnable"):
                    runnable = node_spec.runnable
                    # RunnableCallable has a 'func' attribute with the actual function
                    if hasattr(runnable, "func"):
                        actual_func = runnable.func
                
                # If we couldn't extract from runnable, try direct access (for other graph types)
                if actual_func is None:
                    # Try to use node_spec directly if it's already a function
                    if callable(node_spec):
                        actual_func = node_spec
                
                # Look up instruction type for this function
                if actual_func is not None:
                    # We store both original function and wrapper, so we can find it directly
                    if actual_func in self._function_to_instruction:
                        node_to_instruction[node_name] = self._function_to_instruction[actual_func]
                    else:
                        # Try to find original function if wrapped
                        original_func = actual_func
                        while hasattr(original_func, "__wrapped__"):
                            original_func = original_func.__wrapped__
                            if original_func in self._function_to_instruction:
                                node_to_instruction[node_name] = self._function_to_instruction[original_func]
                                break
        
        return node_to_instruction

    @staticmethod
    def extract_edges_from_graph(
        graph: Union[StateGraph, "Graph", "CompiledStateGraph", Any]
    ) -> list[tuple[str, str]]:
        """Extract edges from a LangGraph StateGraph, compiled graph, or Graph object.

        This helper method automatically extracts edge information from various
        LangGraph graph representations, eliminating the need to manually maintain
        edge lists. It also extracts conditional edges by analyzing routing functions.

        Args:
            graph: Can be one of:
                - StateGraph builder (before compilation)
                - CompiledStateGraph (after compilation)
                - Graph object (from get_graph() method)

        Returns:
            List of tuples representing graph edges in format (source, target).

        Example:
            >>> builder = StateGraph(MyState)
            >>> builder.add_edge("a", "b")
            >>> edges = ArbiterOSAlpha.extract_edges_from_graph(builder)
            >>> # Or from compiled graph:
            >>> compiled = builder.compile()
            >>> edges = ArbiterOSAlpha.extract_edges_from_graph(compiled)
        """
        edges_list: list[tuple[str, str]] = []
        original_graph = graph

        # If it's a compiled graph, get the Graph object first
        if hasattr(graph, "get_graph"):
            graph = graph.get_graph()

        # If it's a StateGraph builder, access edges directly
        if hasattr(graph, "edges") and not hasattr(graph, "source"):
            # This is a StateGraph builder with edges attribute
            for edge in graph.edges:
                if hasattr(edge, "source") and hasattr(edge, "target"):
                    edges_list.append((edge.source, edge.target))
                elif isinstance(edge, tuple) and len(edge) >= 2:
                    edges_list.append((edge[0], edge[1]))
            
            # Also extract conditional edges from branches
            if hasattr(original_graph, "branches") and hasattr(original_graph, "nodes"):
                edges_list.extend(ArbiterOSAlpha._extract_conditional_edges(original_graph))
        elif hasattr(graph, "edges"):
            # This is a Graph object from get_graph()
            for edge in graph.edges:
                if hasattr(edge, "source") and hasattr(edge, "target"):
                    edges_list.append((edge.source, edge.target))
                elif isinstance(edge, tuple) and len(edge) >= 2:
                    edges_list.append((edge[0], edge[1]))
        else:
            raise ValueError(
                f"Cannot extract edges from graph object of type {type(graph)}. "
                "Expected StateGraph, CompiledStateGraph, or Graph object."
            )

        logger.debug(f"Extracted {len(edges_list)} edges from graph: {edges_list}")
        return edges_list

    @staticmethod
    def _extract_conditional_edges(graph: StateGraph) -> list[tuple[str, str]]:
        """Extract conditional edges from a StateGraph by analyzing routing functions.
        
        This method attempts to extract all possible targets from conditional edges
        by analyzing the routing function's source code for string literal returns.
        
        Args:
            graph: A StateGraph builder with branches.
        
        Returns:
            List of tuples representing conditional edges (source, target).
        """
        import ast
        import inspect
        
        conditional_edges: list[tuple[str, str]] = []
        
        if not hasattr(graph, "branches") or not hasattr(graph, "nodes"):
            return conditional_edges
        
        # Get all node names for validation
        node_names = set(graph.nodes.keys())
        
        for source_node, branch_dict in graph.branches.items():
            for branch_key, branch_spec in branch_dict.items():
                if hasattr(branch_spec, "path") and hasattr(branch_spec.path, "func"):
                    route_func = branch_spec.path.func
                    try:
                        # Get function source code
                        source_code = inspect.getsource(route_func)
                        # Parse AST to find string literal returns
                        tree = ast.parse(source_code)
                        for node in ast.walk(tree):
                            # Look for return statements with string literals
                            if isinstance(node, ast.Return) and node.value:
                                if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                                    target = node.value.value
                                    if target in node_names:
                                        conditional_edges.append((source_node, target))
                                elif isinstance(node.value, ast.Str):  # Python < 3.8
                                    target = node.value.s
                                    if target in node_names:
                                        conditional_edges.append((source_node, target))
                                # Handle ternary expressions: "a" if condition else "b"
                                elif isinstance(node.value, ast.IfExp):
                                    for expr in [node.value.body, node.value.orelse]:
                                        if isinstance(expr, ast.Constant) and isinstance(expr.value, str):
                                            target = expr.value
                                            if target in node_names:
                                                conditional_edges.append((source_node, target))
                                        elif isinstance(expr, ast.Str):  # Python < 3.8
                                            target = expr.s
                                            if target in node_names:
                                                conditional_edges.append((source_node, target))
                    except (OSError, SyntaxError, AttributeError) as e:
                        logger.debug(f"Could not extract conditional edges from {route_func}: {e}")
                        continue
        
        logger.debug(f"Extracted {len(conditional_edges)} conditional edges: {conditional_edges}")
        return conditional_edges

    def visualize_workflow(
        self,
        graph_or_edges: Union[StateGraph, "Graph", "CompiledStateGraph", list[tuple[str, str]], Any],
        output_file: str | None = None,
        format: str = "mermaid",
    ) -> str:
        """Generate a visual representation of the workflow graph.

        This method creates a visual diagram of the workflow structure, showing
        nodes and edges. Supports multiple output formats.

        Args:
            graph_or_edges: Can be one of:
                - StateGraph builder (edges will be extracted automatically)
                - CompiledStateGraph (after compilation)
                - Graph object from get_graph() (edges will be extracted automatically)
                - List of tuples representing graph edges in format (source, target)
            output_file: Optional path to save the visualization. If None, returns the
                visualization as a string.
            format: Output format. Currently supports:
                - "mermaid": Mermaid diagram syntax (default)

        Returns:
            The visualization as a string. If output_file is provided, also saves to file.

        Example:
            >>> os = ArbiterOSAlpha()
            >>> builder = StateGraph(MyState)
            >>> builder.add_edge("a", "b")
            >>> # Generate mermaid diagram
            >>> mermaid_code = os.visualize_workflow(builder)
            >>> print(mermaid_code)
            >>> # Save to file
            >>> os.visualize_workflow(builder, output_file="workflow.md")
        """
        # Extract edges and node information
        if isinstance(graph_or_edges, list):
            edges = graph_or_edges
            node_to_instruction: dict[str, InstructionType] = {}
        else:
            edges = self.extract_edges_from_graph(graph_or_edges)
            node_to_instruction = self._extract_node_to_instruction(graph_or_edges)

        if format == "mermaid":
            diagram = self._generate_mermaid_diagram(edges, node_to_instruction)
        else:
            raise ValueError(f"Unsupported format: {format}. Supported formats: 'mermaid'")

        if output_file:
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(diagram)
            logger.info(f"Workflow visualization saved to {output_file}")

        return diagram

    def _generate_mermaid_diagram(
        self,
        edges: list[tuple[str, str]],
        node_to_instruction: dict[str, InstructionType],
    ) -> str:
        """Generate a Mermaid flowchart diagram from edges and node information.

        Args:
            edges: List of tuples representing graph edges.
            node_to_instruction: Mapping from node names to instruction types.

        Returns:
            Mermaid diagram code as a string.
        """
        lines = ["graph TD"]
        
        # Collect all unique nodes
        all_nodes = set()
        for source, target in edges:
            all_nodes.add(source)
            all_nodes.add(target)

        # Map special nodes to valid Mermaid IDs
        node_id_map = {}
        for node in sorted(all_nodes):
            if node in ["__start__", "__end__"]:
                # Special nodes - use valid Mermaid identifiers
                if node == "__start__":
                    node_id_map[node] = "Start"
                else:
                    node_id_map[node] = "End"
            else:
                # Regular nodes - use node name as-is (Mermaid handles most valid Python identifiers)
                node_id_map[node] = node

        # Add node definitions with instruction type labels
        for node in sorted(all_nodes):
            node_id = node_id_map[node]
            if node in ["__start__", "__end__"]:
                # Special nodes
                if node == "__start__":
                    lines.append(f'    {node_id}[Start]:::startNode')
                else:
                    lines.append(f'    {node_id}[End]:::endNode')
            elif node in node_to_instruction:
                # Node with instruction type
                instruction = node_to_instruction[node]
                instruction_name = instruction.name.replace("_", " ").title()
                lines.append(f'    {node_id}["{node}<br/>({instruction_name})"]')
            else:
                # Regular node
                lines.append(f'    {node_id}["{node}"]')

        # Add edges
        for source, target in edges:
            source_id = node_id_map[source]
            target_id = node_id_map[target]
            lines.append(f"    {source_id} --> {target_id}")

        # Add styling (use endNode instead of end to avoid Mermaid reserved keyword)
        lines.append("    classDef startNode fill:#90EE90,stroke:#333,stroke-width:2px")
        lines.append("    classDef endNode fill:#FFB6C1,stroke:#333,stroke-width:2px")

        return "\n".join(lines)

    def validate_graph_structure(
        self,
        graph_or_edges: Union[StateGraph, "Graph", "CompiledStateGraph", list[tuple[str, str]], Any],
        visualize: bool = False,
        visualization_file: str | None = None,
    ) -> bool:
        """Validate graph structure against all registered graph structure checkers.

        This method should be called after building the graph but before execution
        to ensure the graph structure complies with all policy rules. It checks
        all registered GraphStructurePolicyChecker instances.

        Args:
            graph_or_edges: Can be one of:
                - StateGraph builder (edges will be extracted automatically)
                - CompiledStateGraph (edges will be extracted automatically)
                - Graph object from get_graph() (edges will be extracted automatically)
                - List of tuples representing graph edges in format (source, target)
                  Example: [("START", "generate"), ("generate", "toolcall")]
            visualize: If True, generate and optionally save a workflow visualization.
            visualization_file: Optional path to save the visualization. If None and
                visualize=True, the diagram will be printed to console.

        Returns:
            True if all graph structure checkers pass.

        Raises:
            RuntimeError: If any graph structure checker detects a violation.
                The error message includes details about the violating rule and path.
            ValueError: If graph_or_edges is not a recognized graph type or edge list.

        Example:
            >>> os = ArbiterOSAlpha()
            >>> checker = GraphStructurePolicyChecker().add_blacklist("rule", ["a", "b"])
            >>> os.add_graph_structure_checker(checker)
            >>> # Option 1: Pass graph builder or compiled graph directly
            >>> builder = StateGraph(MyState)
            >>> builder.add_edge("a", "b")
            >>> compiled = builder.compile()
            >>> os.validate_graph_structure(compiled)  # Edges extracted automatically
            >>> # Option 2: Pass edges list manually
            >>> edges = [("START", "a"), ("a", "b"), ("b", "END")]
            >>> os.validate_graph_structure(edges)
            >>> # Option 3: Validate and visualize
            >>> os.validate_graph_structure(builder, visualize=True, visualization_file="workflow.md")
        """
        # Extract edges if a graph object was passed
        if isinstance(graph_or_edges, list):
            edges = graph_or_edges
            node_to_instruction: dict[str, InstructionType] = {}
        else:
            edges = self.extract_edges_from_graph(graph_or_edges)
            node_to_instruction = self._extract_node_to_instruction(graph_or_edges)

        logger.debug(
            f"Validating graph structure with {len(self.graph_structure_checkers)} checkers"
        )
        logger.debug(f"Node to instruction mapping: {node_to_instruction}")
        logger.debug(f"Edges: {edges}")
        
        # Generate visualization if requested
        if visualize:
            diagram = self.visualize_workflow(graph_or_edges, output_file=visualization_file)
            if not visualization_file:
                logger.info("Workflow visualization:\n" + diagram)
        
        for checker in self.graph_structure_checkers:
            checker.check_graph_structure(edges, node_to_instruction=node_to_instruction)
        logger.debug("Graph structure validation finished.")
        return True


    def instruction(
        self, instruction_type: InstructionType
    ) -> Callable[[Callable], Callable]:
        """Decorator to wrap LangGraph node functions with policy governance.

        This decorator adds policy validation, execution history tracking,
        schema validation, and dynamic routing to LangGraph node functions.
        It's the core integration point between ArbiterOS and LangGraph.

        Args:
            instruction_type: An instruction type from one of the Core enums
                (CognitiveCore, MemoryCore, ExecutionCore, NormativeCore,
                MetacognitiveCore, AdaptiveCore, SocialCore, or AffectiveCore).

        Returns:
            A decorator function that wraps the target node function.

        Example:
            >>> from arbiteros_alpha.instructions import CognitiveCore
            >>> os = ArbiterOSAlpha(validate_schemas=True)
            >>> @os.instruction(CognitiveCore.GENERATE)
            ... def generate(state: State) -> State:
            ...     return {"content": "generated text", "reasoning": "..."}
            >>> # Function now includes policy checks, schema validation, and history tracking
        """
        # Validate that instruction_type is a valid InstructionType enum
        if not isinstance(instruction_type, InstructionType.__args__):
            raise TypeError(
                f"instruction_type must be an instance of one of the Core enums, got {type(instruction_type)}"
            )

        def decorator(func: Callable) -> Callable:
            # Store function to instruction type mapping for graph structure validation
            # Store both original function and wrapper so we can find it from either
            self._function_to_instruction[func] = instruction_type
            
            # Get schemas for this instruction type if validation is enabled
            input_schema = None
            output_schema = None
            if self.validate_schemas:
                input_schema = get_input_schema(instruction_type)
                output_schema = get_output_schema(instruction_type)
                logger.debug(
                    f"Schema validation enabled for {instruction_type.name}: "
                    f"input={input_schema.__name__}, output={output_schema.__name__}"
                )
            
            @functools.wraps(func)
            def wrapper(*args, **kwargs) -> Any:
                logger.debug(
                    f"Executing instruction: {instruction_type.__class__.__name__}.{instruction_type.name}"
                )

                # Extract input state (first positional argument or from kwargs)
                input_state = args[0] if args else kwargs.get("state", {})
                
                # Validate input schema if enabled
                if self.validate_schemas and input_schema:
                    if isinstance(input_state, dict):
                        is_valid, errors = validate_input(
                            instruction_type, input_state, self.strict_schema_validation
                        )
                        if not is_valid:
                            error_msg = f"Input schema validation failed for {instruction_type.name}: {', '.join(errors)}"
                            logger.warning(error_msg)
                            if self.strict_schema_validation:
                                raise ValueError(error_msg)
                    else:
                        logger.debug(
                            f"Input state is not a dict, skipping schema validation: {type(input_state)}"
                        )

                # Get function name for display
                function_name = func.__name__
                
                self.history.append(
                    History(
                        timestamp=datetime.datetime.now(),
                        instruction=instruction_type,
                        node_name=function_name,
                        input_state=input_state if isinstance(input_state, dict) else {"raw": input_state},
                    )
                )

                self.history[-1].check_policy_results, all_passed = self._check_before()

                result = func(*args, **kwargs)
                logger.debug(f"Instruction {instruction_type.name} returned: {result}")
                
                # Validate output schema if enabled
                if self.validate_schemas and output_schema:
                    # Handle Command objects (from LangGraph routing)
                    output_to_validate = result
                    if isinstance(result, Command) and hasattr(result, "update"):
                        output_to_validate = result.update if result.update else {}
                    
                    if isinstance(output_to_validate, dict):
                        is_valid, errors = validate_output(
                            instruction_type, output_to_validate, self.strict_schema_validation
                        )
                        if not is_valid:
                            error_msg = f"Output schema validation failed for {instruction_type.name}: {', '.join(errors)}"
                            logger.warning(error_msg)
                            if self.strict_schema_validation:
                                raise ValueError(error_msg)
                    else:
                        logger.debug(
                            f"Output is not a dict, skipping schema validation: {type(output_to_validate)}"
                        )
                
                # Store output in history
                if isinstance(result, Command) and hasattr(result, "update"):
                    self.history[-1].output_state = result.update if result.update else {}
                else:
                    self.history[-1].output_state = result if isinstance(result, dict) else {"raw": result}

                self.history[-1].route_policy_results, destination = self._route_after()

                if destination:
                    return Command(update=result, goto=destination)

                return result

            # Also store wrapper so we can find it from graph nodes
            self._function_to_instruction[wrapper] = instruction_type
            
            return wrapper

        return decorator
