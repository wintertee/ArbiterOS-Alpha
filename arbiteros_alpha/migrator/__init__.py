"""ArbiterOS Migrator: Migration tool for converting agents to ArbiterOS-governed agents.

This module provides tools to migrate existing LangGraph or native Python agents
into ArbiterOS-governed agents with automatic instruction type classification.

Enhanced with:
- Repo-level transformation
- LLM-powered analysis and classification
- Policy and schema design
- Multi-file code generation
"""

from .analyzer import AnalyzerConfig, RepoAnalyzer, analyze_repository
from .classifier import (
    ClassificationConfig,
    InstructionClassifier,
    NodeClassification,
    classify_functions_with_context,
)
from .generator import CodeGenerator, MigrationResult
from .logger import MigrationLogger
from .parser import AgentParser, ParsedAgent, ParsedFunction
from .policy_designer import PolicyDesigner, PolicyDesignerConfig, design_policies
from .repo_scanner import RepoScanner, scan_repository
from .schema_designer import SchemaDesigner, SchemaDesignerConfig, design_schemas
from .schemas import (
    AgentFramework,
    CoreType,
    FunctionInfo,
    GeneratedFile,
    InstructionTypeEnum,
    LLMSchemaDesignOutput,
    NodeClassificationBatch,
    NodeClassificationResult,
    PolicyDesignOutput,
    RepoAnalysisOutput,
    RepoScanResult,
    TransformationResult,
    TransformConfig,
)

__all__ = [
    # Parser
    "AgentParser",
    "ParsedAgent",
    "ParsedFunction",
    # Scanner
    "RepoScanner",
    "scan_repository",
    # Analyzer
    "RepoAnalyzer",
    "AnalyzerConfig",
    "analyze_repository",
    # Classifier
    "InstructionClassifier",
    "ClassificationConfig",
    "NodeClassification",
    "classify_functions_with_context",
    # Policy Designer
    "PolicyDesigner",
    "PolicyDesignerConfig",
    "design_policies",
    # Schema Designer
    "SchemaDesigner",
    "SchemaDesignerConfig",
    "design_schemas",
    # Generator
    "CodeGenerator",
    "MigrationResult",
    # Logger
    "MigrationLogger",
    # Schemas
    "AgentFramework",
    "CoreType",
    "InstructionTypeEnum",
    "FunctionInfo",
    "RepoScanResult",
    "RepoAnalysisOutput",
    "NodeClassificationResult",
    "NodeClassificationBatch",
    "PolicyDesignOutput",
    "LLMSchemaDesignOutput",
    "GeneratedFile",
    "TransformationResult",
    "TransformConfig",
]
