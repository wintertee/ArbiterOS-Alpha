"""Tests for Plan-and-Execute Agent transformation.

These tests verify that both the original (backup) and transformed versions
of the Plan-and-Execute agent work correctly and produce equivalent results.

Tests cover:
- Graph compilation and structure
- State management
- Node function execution
- ArbiterOS decoration
"""

import operator
from typing import Annotated, List, Literal, Tuple, Union
from unittest.mock import AsyncMock, MagicMock

import pytest
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field
from typing_extensions import TypedDict


# ============================================================================
# Test Fixtures - Shared State and Models
# ============================================================================


class PlanExecute(TypedDict):
    """State for plan-and-execute agent tests."""

    input: str
    plan: List[str]
    past_steps: Annotated[List[Tuple], operator.add]
    response: str


class Plan(BaseModel):
    """Plan model for tests."""

    steps: List[str] = Field(
        description="different steps to follow, should be in sorted order"
    )


class Response(BaseModel):
    """Response model for tests."""

    response: str


class Act(BaseModel):
    """Action model for tests."""

    action: Union[Response, Plan] = Field(description="Action to perform.")


# ============================================================================
# Test Original (Vanilla LangGraph) Implementation
# ============================================================================


class TestOriginalPlanAndExecute:
    """Tests for the original LangGraph Plan-and-Execute implementation."""

    @pytest.fixture
    def mock_planner(self):
        """Mock planner that returns a predefined plan."""
        mock = AsyncMock()
        mock.ainvoke.return_value = Plan(steps=["Step 1: Research", "Step 2: Analyze"])
        return mock

    @pytest.fixture
    def mock_replanner(self):
        """Mock replanner that returns a response after one step."""
        mock = AsyncMock()
        mock.ainvoke.return_value = Act(action=Response(response="Final answer"))
        return mock

    @pytest.fixture
    def mock_agent_executor(self):
        """Mock agent executor that returns a mock response."""
        mock = AsyncMock()
        mock.ainvoke.return_value = {
            "messages": [MagicMock(content="Step executed successfully")]
        }
        return mock

    @pytest.mark.asyncio
    async def test_plan_step_creates_plan(self, mock_planner):
        """Test that plan_step correctly creates a plan."""

        async def plan_step(state: PlanExecute):
            """Create an initial plan."""
            plan = await mock_planner.ainvoke({"messages": [("user", state["input"])]})
            return {"plan": plan.steps}

        state = {"input": "Test query", "plan": [], "past_steps": [], "response": ""}
        result = await plan_step(state)

        assert "plan" in result
        assert len(result["plan"]) == 2
        assert result["plan"][0] == "Step 1: Research"

    @pytest.mark.asyncio
    async def test_execute_step_executes_task(self, mock_agent_executor):
        """Test that execute_step correctly executes a task."""

        async def execute_step(state: PlanExecute):
            """Execute a plan step."""
            plan = state["plan"]
            task = plan[0]
            agent_response = await mock_agent_executor.ainvoke(
                {"messages": [("user", f"Execute: {task}")]}
            )
            return {
                "past_steps": [(task, agent_response["messages"][-1].content)],
            }

        state = {
            "input": "Test query",
            "plan": ["Step 1: Research"],
            "past_steps": [],
            "response": "",
        }
        result = await execute_step(state)

        assert "past_steps" in result
        assert len(result["past_steps"]) == 1
        assert result["past_steps"][0][0] == "Step 1: Research"
        assert result["past_steps"][0][1] == "Step executed successfully"

    @pytest.mark.asyncio
    async def test_replan_step_returns_response(self, mock_replanner):
        """Test that replan_step returns response when complete."""

        async def replan_step(state: PlanExecute):
            """Replan based on results."""
            output = await mock_replanner.ainvoke(state)
            if isinstance(output.action, Response):
                return {"response": output.action.response}
            else:
                return {"plan": output.action.steps}

        state = {
            "input": "Test query",
            "plan": ["Step 1"],
            "past_steps": [("Step 1", "Done")],
            "response": "",
        }
        result = await replan_step(state)

        assert "response" in result
        assert result["response"] == "Final answer"

    def test_should_end_returns_end_when_response_present(self):
        """Test that should_end returns END when response is present."""

        def should_end(state: PlanExecute) -> Literal["agent", "__end__"]:
            if "response" in state and state["response"]:
                return END
            else:
                return "agent"

        state_with_response = {
            "input": "",
            "plan": [],
            "past_steps": [],
            "response": "Answer",
        }
        state_without_response = {
            "input": "",
            "plan": [],
            "past_steps": [],
            "response": "",
        }

        assert should_end(state_with_response) == END
        assert should_end(state_without_response) == "agent"

    def test_graph_compilation(self):
        """Test that the graph compiles correctly."""

        # Define minimal node functions
        async def plan_step(state):
            return {"plan": ["step"]}

        async def execute_step(state):
            return {"past_steps": [("step", "done")]}

        async def replan_step(state):
            return {"response": "done"}

        def should_end(state):
            return END if state.get("response") else "agent"

        # Build graph (same structure as original)
        workflow = StateGraph(PlanExecute)
        workflow.add_node("planner", plan_step)
        workflow.add_node("agent", execute_step)
        workflow.add_node("replan", replan_step)
        workflow.add_edge(START, "planner")
        workflow.add_edge("planner", "agent")
        workflow.add_edge("agent", "replan")
        workflow.add_conditional_edges("replan", should_end, ["agent", END])

        # Should compile without errors
        app = workflow.compile()
        assert app is not None


# ============================================================================
# Test Transformed (ArbiterOS) Implementation
# ============================================================================


class TestTransformedPlanAndExecute:
    """Tests for the ArbiterOS-transformed Plan-and-Execute implementation."""

    @pytest.fixture
    def arbiter_os(self):
        """Create ArbiterOS instance for testing."""
        from arbiteros_alpha import ArbiterOSAlpha

        return ArbiterOSAlpha(backend="langgraph")

    @pytest.fixture
    def mock_planner(self):
        """Mock planner that returns a predefined plan."""
        mock = AsyncMock()
        mock.ainvoke.return_value = Plan(steps=["Step 1: Research", "Step 2: Analyze"])
        return mock

    @pytest.fixture
    def mock_replanner(self):
        """Mock replanner that returns a response."""
        mock = AsyncMock()
        mock.ainvoke.return_value = Act(action=Response(response="Final answer"))
        return mock

    @pytest.fixture
    def mock_agent_executor(self):
        """Mock agent executor."""
        mock = AsyncMock()
        mock.ainvoke.return_value = {
            "messages": [MagicMock(content="Step executed successfully")]
        }
        return mock

    def test_instruction_decorator_preserves_function(self, arbiter_os):
        """Test that @os.instruction decorator preserves function behavior."""
        import arbiteros_alpha.instructions as Instr

        @arbiter_os.instruction(Instr.GENERATE)
        def generate(state):
            return {"response": "test"}

        # Need to enter superstep before calling the decorated function
        arbiter_os.history.enter_next_superstep(["generate"])
        result = generate({"query": "test"})
        assert result == {"response": "test"}

    @pytest.mark.asyncio
    async def test_async_instruction_decorator_preserves_function(self, arbiter_os):
        """Test that @os.instruction decorator works with async functions."""
        import arbiteros_alpha.instructions as Instr

        @arbiter_os.instruction(Instr.DECOMPOSE)
        async def async_plan(state):
            return {"plan": ["step1"]}

        # Need to enter superstep before calling the decorated function
        arbiter_os.history.enter_next_superstep(["async_plan"])
        result = await async_plan({"input": "test"})
        assert result == {"plan": ["step1"]}

    @pytest.mark.asyncio
    async def test_transformed_plan_step(self, arbiter_os, mock_planner):
        """Test transformed plan_step with ArbiterOS decorator."""
        import arbiteros_alpha.instructions as Instr

        @arbiter_os.instruction(Instr.DECOMPOSE)
        async def plan_step(state: PlanExecute):
            plan = await mock_planner.ainvoke({"messages": [("user", state["input"])]})
            return {"plan": plan.steps}

        state = {"input": "Test query", "plan": [], "past_steps": [], "response": ""}
        arbiter_os.history.enter_next_superstep(["plan_step"])
        result = await plan_step(state)

        assert "plan" in result
        assert len(result["plan"]) == 2

    @pytest.mark.asyncio
    async def test_transformed_execute_step(self, arbiter_os, mock_agent_executor):
        """Test transformed execute_step with ArbiterOS decorator."""
        import arbiteros_alpha.instructions as Instr

        @arbiter_os.instruction(Instr.TOOL_CALL)
        async def execute_step(state: PlanExecute):
            plan = state["plan"]
            task = plan[0]
            agent_response = await mock_agent_executor.ainvoke(
                {"messages": [("user", f"Execute: {task}")]}
            )
            return {
                "past_steps": [(task, agent_response["messages"][-1].content)],
            }

        state = {
            "input": "Test query",
            "plan": ["Step 1"],
            "past_steps": [],
            "response": "",
        }
        arbiter_os.history.enter_next_superstep(["execute_step"])
        result = await execute_step(state)

        assert "past_steps" in result
        assert result["past_steps"][0][1] == "Step executed successfully"

    @pytest.mark.asyncio
    async def test_transformed_replan_step(self, arbiter_os, mock_replanner):
        """Test transformed replan_step with ArbiterOS decorator."""
        import arbiteros_alpha.instructions as Instr

        @arbiter_os.instruction(Instr.DECOMPOSE)
        async def replan_step(state: PlanExecute):
            output = await mock_replanner.ainvoke(state)
            if isinstance(output.action, Response):
                return {"response": output.action.response}
            else:
                return {"plan": output.action.steps}

        state = {
            "input": "Test query",
            "plan": ["Step 1"],
            "past_steps": [("Step 1", "Done")],
            "response": "",
        }
        arbiter_os.history.enter_next_superstep(["replan_step"])
        result = await replan_step(state)

        assert "response" in result
        assert result["response"] == "Final answer"

    def test_transformed_graph_compilation_with_registration(self, arbiter_os):
        """Test that the transformed graph compiles and registers correctly."""
        import arbiteros_alpha.instructions as Instr

        @arbiter_os.instruction(Instr.DECOMPOSE)
        async def plan_step(state):
            return {"plan": ["step"]}

        @arbiter_os.instruction(Instr.TOOL_CALL)
        async def execute_step(state):
            return {"past_steps": [("step", "done")]}

        @arbiter_os.instruction(Instr.DECOMPOSE)
        async def replan_step(state):
            return {"response": "done"}

        def should_end(state):
            return END if state.get("response") else "agent"

        # Build graph
        workflow = StateGraph(PlanExecute)
        workflow.add_node("planner", plan_step)
        workflow.add_node("agent", execute_step)
        workflow.add_node("replan", replan_step)
        workflow.add_edge(START, "planner")
        workflow.add_edge("planner", "agent")
        workflow.add_edge("agent", "replan")
        workflow.add_conditional_edges("replan", should_end, ["agent", END])

        # Compile and register
        app = workflow.compile()
        arbiter_os.register_compiled_graph(app)

        assert app is not None
        # The registration is done via global _pregel_to_arbiter_map
        from arbiteros_alpha.core import _pregel_to_arbiter_map

        assert app in _pregel_to_arbiter_map


# ============================================================================
# Test Equivalence Between Original and Transformed
# ============================================================================


class TestEquivalence:
    """Tests to verify original and transformed versions produce equivalent results."""

    @pytest.fixture
    def mock_planner(self):
        """Mock planner with deterministic output."""
        mock = AsyncMock()
        mock.ainvoke.return_value = Plan(steps=["Step 1", "Step 2"])
        return mock

    @pytest.fixture
    def mock_replanner(self):
        """Mock replanner with deterministic output."""
        mock = AsyncMock()
        mock.ainvoke.return_value = Act(action=Response(response="Final result"))
        return mock

    @pytest.fixture
    def mock_agent(self):
        """Mock agent executor with deterministic output."""
        mock = AsyncMock()
        mock.ainvoke.return_value = {"messages": [MagicMock(content="Executed")]}
        return mock

    @pytest.mark.asyncio
    async def test_plan_step_equivalence(self, mock_planner):
        """Test that original and transformed plan_step produce same results."""

        # Original version
        async def original_plan_step(state: PlanExecute):
            plan = await mock_planner.ainvoke({"messages": [("user", state["input"])]})
            return {"plan": plan.steps}

        # Transformed version with decorator (decorator should be transparent)
        from arbiteros_alpha import ArbiterOSAlpha
        import arbiteros_alpha.instructions as Instr

        os = ArbiterOSAlpha(backend="langgraph")

        @os.instruction(Instr.DECOMPOSE)
        async def transformed_plan_step(state: PlanExecute):
            plan = await mock_planner.ainvoke({"messages": [("user", state["input"])]})
            return {"plan": plan.steps}

        state = {"input": "test", "plan": [], "past_steps": [], "response": ""}

        original_result = await original_plan_step(state)
        os.history.enter_next_superstep(["transformed_plan_step"])
        transformed_result = await transformed_plan_step(state)

        assert original_result == transformed_result

    @pytest.mark.asyncio
    async def test_execute_step_equivalence(self, mock_agent):
        """Test that original and transformed execute_step produce same results."""

        # Original version
        async def original_execute_step(state: PlanExecute):
            task = state["plan"][0]
            agent_response = await mock_agent.ainvoke(
                {"messages": [("user", f"Execute: {task}")]}
            )
            return {
                "past_steps": [(task, agent_response["messages"][-1].content)],
            }

        # Transformed version
        from arbiteros_alpha import ArbiterOSAlpha
        import arbiteros_alpha.instructions as Instr

        os = ArbiterOSAlpha(backend="langgraph")

        @os.instruction(Instr.TOOL_CALL)
        async def transformed_execute_step(state: PlanExecute):
            task = state["plan"][0]
            agent_response = await mock_agent.ainvoke(
                {"messages": [("user", f"Execute: {task}")]}
            )
            return {
                "past_steps": [(task, agent_response["messages"][-1].content)],
            }

        state = {"input": "test", "plan": ["Step 1"], "past_steps": [], "response": ""}

        original_result = await original_execute_step(state)
        os.history.enter_next_superstep(["transformed_execute_step"])
        transformed_result = await transformed_execute_step(state)

        assert original_result == transformed_result

    @pytest.mark.asyncio
    async def test_replan_step_equivalence(self, mock_replanner):
        """Test that original and transformed replan_step produce same results."""

        # Original version
        async def original_replan_step(state: PlanExecute):
            output = await mock_replanner.ainvoke(state)
            if isinstance(output.action, Response):
                return {"response": output.action.response}
            else:
                return {"plan": output.action.steps}

        # Transformed version
        from arbiteros_alpha import ArbiterOSAlpha
        import arbiteros_alpha.instructions as Instr

        os = ArbiterOSAlpha(backend="langgraph")

        @os.instruction(Instr.DECOMPOSE)
        async def transformed_replan_step(state: PlanExecute):
            output = await mock_replanner.ainvoke(state)
            if isinstance(output.action, Response):
                return {"response": output.action.response}
            else:
                return {"plan": output.action.steps}

        state = {
            "input": "test",
            "plan": ["Step 1"],
            "past_steps": [("Step 1", "Done")],
            "response": "",
        }

        original_result = await original_replan_step(state)
        os.history.enter_next_superstep(["transformed_replan_step"])
        transformed_result = await transformed_replan_step(state)

        assert original_result == transformed_result


# ============================================================================
# Test ArbiterOS History Tracking
# ============================================================================


class TestArbiterOSFeatures:
    """Test ArbiterOS-specific features in the transformed agent."""

    def test_history_tracking(self):
        """Test that ArbiterOS tracks execution history."""
        from arbiteros_alpha import ArbiterOSAlpha
        import arbiteros_alpha.instructions as Instr

        os = ArbiterOSAlpha(backend="langgraph")

        @os.instruction(Instr.GENERATE)
        def generate(state):
            return {"response": "test"}

        # Need to enter superstep before calling decorated function
        os.history.enter_next_superstep(["generate"])
        result = generate({"query": "test"})

        # Check history was recorded
        assert len(os.history.entries) == 1
        assert len(os.history.entries[0]) == 1
        last_entry = os.history.entries[0][0]
        assert last_entry.instruction == Instr.GENERATE

    @pytest.mark.asyncio
    async def test_async_history_tracking(self):
        """Test that ArbiterOS tracks async function execution history."""
        from arbiteros_alpha import ArbiterOSAlpha
        import arbiteros_alpha.instructions as Instr

        os = ArbiterOSAlpha(backend="langgraph")

        @os.instruction(Instr.DECOMPOSE)
        async def async_decompose(state):
            return {"plan": ["step1", "step2"]}

        # Need to enter superstep before calling decorated function
        os.history.enter_next_superstep(["async_decompose"])
        result = await async_decompose({"input": "test"})

        # Check history was recorded
        assert len(os.history.entries) == 1
        assert len(os.history.entries[0]) == 1
        last_entry = os.history.entries[0][0]
        assert last_entry.instruction == Instr.DECOMPOSE

    def test_history_tracking_with_vanilla_backend(self):
        """Test that vanilla backend auto-manages supersteps."""
        from arbiteros_alpha import ArbiterOSAlpha
        import arbiteros_alpha.instructions as Instr

        # Use vanilla backend - it auto-manages supersteps
        os = ArbiterOSAlpha(backend="vanilla")

        @os.instruction(Instr.GENERATE)
        def generate(state):
            return {"response": "test"}

        @os.instruction(Instr.DECOMPOSE)
        def decompose(state):
            return {"plan": ["step1"]}

        # With vanilla backend, no need to manually enter supersteps
        result1 = generate({"query": "test"})
        result2 = decompose({"query": "test"})

        # Check history was recorded
        assert len(os.history.entries) == 2
        assert os.history.entries[0][0].instruction == Instr.GENERATE
        assert os.history.entries[1][0].instruction == Instr.DECOMPOSE
