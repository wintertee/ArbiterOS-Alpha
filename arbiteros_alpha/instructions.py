from enum import Enum, auto
from typing import Union


class CognitiveCore(Enum):
    """Governs probabilistic reasoning. Its outputs are always treated as unverified until subjected to explicit checks."""

    GENERATE = auto()
    """Invokes the LLM for text generation, reasoning, or formulating a query. This is the most general-purpose cognitive instruction, producing content including text, hypotheses, and queries."""
    DECOMPOSE = auto()
    """Breaks a complex task into a sequence of smaller, manageable sub-tasks or creates a formal plan of execution. Transforms complex problems into structured, actionable components."""
    REFLECT = auto()
    """Performs self-critique on generated output to identify flaws, biases, and areas for improvement. Often produces a structured critique to guide subsequent GENERATE steps and self-diagnosis of prior outputs."""


GENERATE = CognitiveCore.GENERATE
DECOMPOSE = CognitiveCore.DECOMPOSE
REFLECT = CognitiveCore.REFLECT


class MemoryCore(Enum):
    """Manages the LLM's limited context window and connections to persistent memory."""

    LOAD = auto()
    """Retrieves information from an external knowledge base (e.g., a vector store or document) to ground the agent."""
    STORE = auto()
    """Writes or updates information in long-term memory, enabling agent learning and persistence."""
    COMPRESS = auto()
    """Reduces the token count of context using methods like summarization or keyword extraction to manage the limited context window."""
    FILTER = auto()
    """Selectively prunes the context to keep only the most relevant information for the current task."""
    STRUCTURE = auto()
    """Transforms unstructured text into a structured format (e.g., JSON) according to a predefined schema."""
    RENDER = auto()
    """Transforms a structured data object (e.g., JSON) into coherent natural language for presentation to a user."""


LOAD = MemoryCore.LOAD
STORE = MemoryCore.STORE
COMPRESS = MemoryCore.COMPRESS
FILTER = MemoryCore.FILTER
STRUCTURE = MemoryCore.STRUCTURE
RENDER = MemoryCore.RENDER


class ExecutionCore(Enum):
    """Interfaces with deterministic external systems. These are high-stakes actions requiring strict controls."""

    TOOL_CALL = auto()
    """Executes a predefined, external, deterministic function (e.g., API calls, database queries, code interpreters). Provides structured interaction with vetted external services."""
    TOOL_BUILD = auto()
    """Writes new code to create novel tools on-the-fly. Enables dynamic capability extension through programmatic generation of custom functions and utilities."""
    DELEGATE = auto()
    """Passes sub-tasks to specialized agents in multi-agent systems. Facilitates hierarchical task decomposition and leverages domain-specific expertise across agent networks."""
    RESPOND = auto()
    """Yields final, user-facing output and signals task completion. Serves as the terminal instruction in any execution workflow, ensuring proper task closure."""


TOOL_CALL = ExecutionCore.TOOL_CALL
TOOL_BUILD = ExecutionCore.TOOL_BUILD
DELEGATE = ExecutionCore.DELEGATE
RESPOND = ExecutionCore.RESPOND


class NormativeCore(Enum):
    """Enforces human-defined rules, checks, and fallback strategies. This domain anchors ARBITEROS's claim to systematic reliability."""

    VERIFY = auto()
    """Performs objective correctness checks against verifiable sources of truth (e.g., schemas, unit tests, databases)."""
    CONSTRAIN = auto()
    """Applies normative compliance rules ('constitution') to outputs, checking for safety, style, or ethical violations."""
    FALLBACK = auto()
    """Executes predefined recovery strategies when preceding instructions fail (e.g., failed TOOL CALL)."""
    INTERRUPT = auto()
    """Pauses execution to request human input, preserving agent state for oversight."""


VERIFY = NormativeCore.VERIFY
CONSTRAIN = NormativeCore.CONSTRAIN
FALLBACK = NormativeCore.FALLBACK
INTERRUPT = NormativeCore.INTERRUPT


class MetacognitiveCore(Enum):
    """Enables heuristic self-assessment and resource tracking, supporting adaptive routing in the Arbiter Loop."""

    PREDICT_SUCCESS = auto()
    """Estimates the probability of successfully completing the current task or plan, providing anticipatory assessment of feasibility."""
    EVALUATE_PROGRESS = auto()
    """Performs strategic assessment of the agent's current reasoning path to answer heuristic, goal-oriented questions about viability and productivity."""
    MONITOR_RESOURCES = auto()
    """Tracks key performance indicators including token usage, computational cost, and latency against predefined budgets and thresholds."""


PREDICT_SUCCESS = MetacognitiveCore.PREDICT_SUCCESS
EVALUATE_PROGRESS = MetacognitiveCore.EVALUATE_PROGRESS
MONITOR_RESOURCES = MetacognitiveCore.MONITOR_RESOURCES


class AdaptiveCore(Enum):
    """Governing autonomous learning and self-improvement within the ArbiterOS paradigm"""

    UPDATE_KNOWLEDGE = auto()
    """Integrates new information into knowledge base via autonomous curriculum generation, web data retrieval, and distillation processes"""
    REFINE_SKILL = auto()
    """Improves existing capabilities through self-generated code testing, fine-tuning on new data, or techniques like Self-Taught Optimizer (STOP)"""
    LEARN_PREFERENCE = auto()
    """Internalizes feedback from human interaction or environmental rewards via Direct Preference Optimization (DPO) or Reinforcement Learning from Human Feedback (RLHF)"""
    FORMULATE_EXPERIMENT = auto()
    """Designs and proposes experiments for active learning loops to discover environmental properties or self-capabilities"""


UPDATE_KNOWLEDGE = AdaptiveCore.UPDATE_KNOWLEDGE
REFINE_SKILL = AdaptiveCore.REFINE_SKILL
LEARN_PREFERENCE = AdaptiveCore.LEARN_PREFERENCE
FORMULATE_EXPERIMENT = AdaptiveCore.FORMULATE_EXPERIMENT


class SocialCore(Enum):
    """Enabling governable inter-agent collaboration in multi-agent systems"""

    COMMUNICATE = auto()
    """Sends a structured message to another agent, following a defined protocol for inter-agent coordination"""
    NEGOTIATE = auto()
    """Engages in a multi-turn dialogue with another agent to reach a mutually acceptable agreement on a resource or plan"""
    PROPOSE_VOTE = auto()
    """Submits a formal proposal to a group of agents and initiates a consensus-forming protocol"""
    FORM_COALITION = auto()
    """Dynamically forms a temporary group or 'crew' of agents to tackle a specific sub-task, defining roles and shared objectives, as seen in frameworks like CrewAI"""


COMMUNICATE = SocialCore.COMMUNICATE
NEGOTIATE = SocialCore.NEGOTIATE
PROPOSE_VOTE = SocialCore.PROPOSE_VOTE
FORM_COALITION = SocialCore.FORM_COALITION


class AffectiveCore(Enum):
    """Enabling governed socio-emotional reasoning for human-agent teaming"""

    INFER_INTENT = auto()
    """Analyzes user communication to infer underlying goals, preferences, or values that may not be explicitly stated"""
    MODEL_USER_STATE = auto()
    """Constructs or updates a model of the user's current cognitive or emotional state (e.g., confused, frustrated) based on interaction history"""
    ADAPT_RESPONSE = auto()
    """Modifies a planned response to align with the user's inferred state or established preferences (e.g., adjusting tone, verbosity, or level of detail)"""
    MANAGE_TRUST = auto()
    """Evaluates the history of interactions to assess the level of trust the user has in the agent and proposes actions to build or repair that trust"""


INFER_INTENT = AffectiveCore.INFER_INTENT
MODEL_USER_STATE = AffectiveCore.MODEL_USER_STATE
ADAPT_RESPONSE = AffectiveCore.ADAPT_RESPONSE
MANAGE_TRUST = AffectiveCore.MANAGE_TRUST


# Union type for all instruction types
InstructionType = Union[
    CognitiveCore,
    MemoryCore,
    ExecutionCore,
    NormativeCore,
    MetacognitiveCore,
    AdaptiveCore,
    SocialCore,
    AffectiveCore,
]
