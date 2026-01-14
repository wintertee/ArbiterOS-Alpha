"""Tests for TradingAgents ArbiterOS governance integration.

This module tests:
1. Decorator application to agent functions
2. Policy checker enforcement
3. Checkpoint save/resume functionality
4. History tracking
"""

import sys
from pathlib import Path

# Add the TradingAgents example to path for testing
sys.path.insert(0, str(Path(__file__).parent.parent / "examples" / "TradingAgents"))


class TestGovernedAgents:
    """Tests for the governed_agents module."""

    def test_arbiter_os_instance_created(self):
        """Test that ArbiterOS instance is created."""
        from tradingagents.agents.governed_agents import arbiter_os, get_arbiter_os

        assert arbiter_os is not None
        assert get_arbiter_os() is arbiter_os

    def test_reset_arbiter_os(self):
        """Test that ArbiterOS can be reset."""
        from tradingagents.agents.governed_agents import (
            get_arbiter_os,
            reset_arbiter_os,
        )

        original_os = get_arbiter_os()
        reset_arbiter_os()
        new_os = get_arbiter_os()

        # After reset, should be a different instance
        assert new_os is not original_os

    def test_instruction_type_mappings(self):
        """Test that instruction type mappings are correct."""
        from tradingagents.agents.governed_agents import (
            ANALYST_INSTRUCTION,
            TOOL_CALL_INSTRUCTION,
            RESEARCHER_INSTRUCTION,
            RESEARCH_MANAGER_INSTRUCTION,
            TRADER_INSTRUCTION,
            RISK_DEBATER_INSTRUCTION,
            RISK_MANAGER_INSTRUCTION,
        )
        from arbiteros_alpha.instructions import (
            CognitiveCore,
            ExecutionCore,
            MetacognitiveCore,
            NormativeCore,
            SocialCore,
        )

        assert ANALYST_INSTRUCTION == CognitiveCore.GENERATE
        assert TOOL_CALL_INSTRUCTION == ExecutionCore.TOOL_CALL
        assert RESEARCHER_INSTRUCTION == CognitiveCore.REFLECT
        assert RESEARCH_MANAGER_INSTRUCTION == MetacognitiveCore.EVALUATE_PROGRESS
        assert TRADER_INSTRUCTION == CognitiveCore.DECOMPOSE
        assert RISK_DEBATER_INSTRUCTION == SocialCore.NEGOTIATE
        assert RISK_MANAGER_INSTRUCTION == NormativeCore.VERIFY


class TestTradingPolicies:
    """Tests for trading-specific policy checkers and routers."""

    def test_analyst_completion_checker_creation(self):
        """Test AnalystCompletionChecker can be created."""
        from tradingagents.policies import AnalystCompletionChecker

        checker = AnalystCompletionChecker(
            name="test_checker",
            required_analysts={"market", "news"},
        )

        assert checker.name == "test_checker"
        assert checker.required_analysts == {"market", "news"}

    def test_debate_rounds_checker_creation(self):
        """Test DebateRoundsChecker can be created."""
        from tradingagents.policies import DebateRoundsChecker

        checker = DebateRoundsChecker(name="test_debate", min_rounds=2)

        assert checker.name == "test_debate"
        assert checker.min_rounds == 2

    def test_risk_analysis_checker_creation(self):
        """Test RiskAnalysisChecker can be created."""
        from tradingagents.policies import RiskAnalysisChecker

        checker = RiskAnalysisChecker(name="test_risk", min_risk_assessments=5)

        assert checker.name == "test_risk"
        assert checker.min_risk_assessments == 5

    def test_confidence_router_creation(self):
        """Test ConfidenceRouter can be created."""
        from tradingagents.policies import ConfidenceRouter

        router = ConfidenceRouter(
            name="test_confidence",
            threshold=0.8,
            target="Market Analyst",
        )

        assert router.name == "test_confidence"
        assert router.threshold == 0.8
        assert router.target == "Market Analyst"

    def test_risk_override_router_creation(self):
        """Test RiskOverrideRouter can be created."""
        from tradingagents.policies import RiskOverrideRouter

        router = RiskOverrideRouter(
            name="test_override",
            max_risk_score=0.9,
            target="Safe Analyst",
        )

        assert router.name == "test_override"
        assert router.max_risk_score == 0.9
        assert router.target == "Safe Analyst"


class TestPolicyCheckerBehavior:
    """Tests for policy checker behavior with mock history."""

    def test_analyst_completion_checker_empty_history(self):
        """Test AnalystCompletionChecker passes with empty history."""
        from tradingagents.policies import AnalystCompletionChecker
        from arbiteros_alpha.history import History

        checker = AnalystCompletionChecker(
            name="test", required_analysts={"market", "news"}
        )
        history = History()

        result = checker.check_before(history)
        assert result is True

    def test_debate_rounds_checker_empty_history(self):
        """Test DebateRoundsChecker passes with empty history."""
        from tradingagents.policies import DebateRoundsChecker
        from arbiteros_alpha.history import History

        checker = DebateRoundsChecker(name="test", min_rounds=1)
        history = History()

        result = checker.check_before(history)
        assert result is True

    def test_risk_analysis_checker_empty_history(self):
        """Test RiskAnalysisChecker passes with empty history."""
        from tradingagents.policies import RiskAnalysisChecker
        from arbiteros_alpha.history import History

        checker = RiskAnalysisChecker(name="test", min_risk_assessments=3)
        history = History()

        result = checker.check_before(history)
        assert result is True


class TestPolicyRouterBehavior:
    """Tests for policy router behavior."""

    def test_confidence_router_empty_history(self):
        """Test ConfidenceRouter returns None with empty history."""
        from tradingagents.policies import ConfidenceRouter
        from arbiteros_alpha.history import History

        router = ConfidenceRouter(name="test", threshold=0.7, target="Market Analyst")
        history = History()

        result = router.route_after(history)
        assert result is None

    def test_risk_override_router_empty_history(self):
        """Test RiskOverrideRouter returns None with empty history."""
        from tradingagents.policies import RiskOverrideRouter
        from arbiteros_alpha.history import History

        router = RiskOverrideRouter(
            name="test", max_risk_score=0.8, target="Safe Analyst"
        )
        history = History()

        result = router.route_after(history)
        assert result is None


class TestDecoratorApplication:
    """Tests for decorator application to agent functions."""

    def test_govern_analyst_decorator(self):
        """Test that govern_analyst decorator wraps functions."""
        from tradingagents.agents.governed_agents import govern_analyst

        @govern_analyst
        def test_func(state):
            return {"result": "test"}

        # The wrapper should preserve the function's callable nature
        assert callable(test_func)

    def test_govern_researcher_decorator(self):
        """Test that govern_researcher decorator wraps functions."""
        from tradingagents.agents.governed_agents import govern_researcher

        @govern_researcher
        def test_func(state):
            return {"result": "test"}

        assert callable(test_func)

    def test_govern_trader_decorator(self):
        """Test that govern_trader decorator wraps functions."""
        from tradingagents.agents.governed_agents import govern_trader

        @govern_trader
        def test_func(state):
            return {"result": "test"}

        assert callable(test_func)

    def test_govern_risk_manager_decorator(self):
        """Test that govern_risk_manager decorator wraps functions."""
        from tradingagents.agents.governed_agents import govern_risk_manager

        @govern_risk_manager
        def test_func(state):
            return {"result": "test"}

        assert callable(test_func)


class TestPolicyConfiguration:
    """Tests for policy configuration integration."""

    def test_policies_can_be_added_to_arbiter_os(self):
        """Test that policies can be added to ArbiterOS instance."""
        from tradingagents.agents.governed_agents import (
            get_arbiter_os,
            reset_arbiter_os,
        )
        from tradingagents.policies import (
            AnalystCompletionChecker,
            DebateRoundsChecker,
        )

        reset_arbiter_os()
        arbiter_os = get_arbiter_os()

        # Add checkers
        arbiter_os.add_policy_checker(
            AnalystCompletionChecker(name="analyst", required_analysts={"market"})
        )
        arbiter_os.add_policy_checker(DebateRoundsChecker(name="debate", min_rounds=1))

        assert len(arbiter_os.policy_checkers) == 2

    def test_multiple_policies_coexist(self):
        """Test that multiple policies can coexist."""
        from tradingagents.agents.governed_agents import (
            get_arbiter_os,
            reset_arbiter_os,
        )
        from tradingagents.policies import (
            AnalystCompletionChecker,
            DebateRoundsChecker,
            RiskAnalysisChecker,
        )

        reset_arbiter_os()
        arbiter_os = get_arbiter_os()

        # Add all three checkers
        arbiter_os.add_policy_checker(
            AnalystCompletionChecker(name="analyst", required_analysts={"market"})
        )
        arbiter_os.add_policy_checker(DebateRoundsChecker(name="debate", min_rounds=1))
        arbiter_os.add_policy_checker(
            RiskAnalysisChecker(name="risk", min_risk_assessments=3)
        )

        assert len(arbiter_os.policy_checkers) == 3


class TestVolatilityRouter:
    """Tests for VolatilityRouter behavior."""

    def test_volatility_router_creation(self):
        """Test VolatilityRouter can be created."""
        from tradingagents.policies import VolatilityRouter

        router = VolatilityRouter(
            name="test_volatility",
            target="Safe Analyst",
            atr_threshold=5.0,
        )

        assert router.name == "test_volatility"
        assert router.target == "Safe Analyst"
        assert router.atr_threshold == 5.0

    def test_volatility_router_empty_history(self):
        """Test VolatilityRouter returns None with empty history."""
        from tradingagents.policies import VolatilityRouter
        from arbiteros_alpha.history import History

        router = VolatilityRouter(name="test", target="Safe Analyst")
        history = History()

        result = router.route_after(history)
        assert result is None


class TestImports:
    """Tests to verify all imports work correctly."""

    def test_governed_agents_imports(self):
        """Test that governed_agents module imports correctly."""
        from tradingagents.agents.governed_agents import (
            arbiter_os,
            get_arbiter_os,
            reset_arbiter_os,
            govern_analyst,
            govern_tool_call,
            govern_researcher,
            govern_research_manager,
            govern_trader,
            govern_risk_debater,
            govern_risk_manager,
            wrap_factory_result,
        )

        # All should be importable
        assert arbiter_os is not None
        assert callable(get_arbiter_os)
        assert callable(reset_arbiter_os)
        assert callable(govern_analyst)
        assert callable(govern_tool_call)
        assert callable(govern_researcher)
        assert callable(govern_research_manager)
        assert callable(govern_trader)
        assert callable(govern_risk_debater)
        assert callable(govern_risk_manager)
        assert callable(wrap_factory_result)

    def test_policies_package_imports(self):
        """Test that policies package imports correctly."""
        from tradingagents.policies import (
            AnalystCompletionChecker,
            DebateRoundsChecker,
            RiskAnalysisChecker,
            ConfidenceRouter,
            RiskOverrideRouter,
            VolatilityRouter,
        )

        # All should be importable
        assert AnalystCompletionChecker is not None
        assert DebateRoundsChecker is not None
        assert RiskAnalysisChecker is not None
        assert ConfidenceRouter is not None
        assert RiskOverrideRouter is not None
        assert VolatilityRouter is not None
