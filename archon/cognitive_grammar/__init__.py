"""
Adaptive Agentic Cognitive Grammar Module

This module implements cognitive grammar parsing and hypergraph-structured 
tensor fields for agentic workflows, providing distributed tensor network
orchestration with adaptive attention allocation.
"""

from .cognitive_parser import CognitiveGrammarParser
from .hypergraph import HypergraphEncoder, HypergraphNode, HypergraphLink
from .tensor_kernels import NodeTensorKernel, LinkTensorKernel, AttentionTensorKernel
from .attention_engine import ECANAttentionEngine
from .tensor_network import DistributedTensorNetwork
from .grammar_updater import AdaptiveGrammarUpdater
from .knowledge_adapters import VectorKnowledgeAdapter

__all__ = [
    'CognitiveGrammarParser',
    'HypergraphEncoder', 
    'HypergraphNode',
    'HypergraphLink',
    'NodeTensorKernel',
    'LinkTensorKernel', 
    'AttentionTensorKernel',
    'ECANAttentionEngine',
    'DistributedTensorNetwork',
    'AdaptiveGrammarUpdater',
    'VectorKnowledgeAdapter'
]