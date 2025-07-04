"""
Cognitive Grammar Integration for Archon

Integrates cognitive grammar with the existing Archon workflow to provide
enhanced understanding and processing of agentic task requests.
"""

try:
    import numpy as np
except ImportError:
    # Use mock numpy if not available
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'tests'))
    import mock_numpy as np

from typing import Dict, List, Any, Optional
import asyncio
import json

from .cognitive_grammar import (
    CognitiveGrammarParser,
    HypergraphEncoder,
    ECANAttentionEngine,
    DistributedTensorNetwork,
    AdaptiveGrammarUpdater,
    LocalKnowledgeAdapter
)


class CognitiveEnhancedState:
    """Enhanced agent state with cognitive grammar analysis"""
    
    def __init__(self, original_state: Dict[str, Any]):
        self.original_state = original_state
        self.cognitive_elements = []
        self.hypergraph_nodes = {}
        self.hypergraph_links = {}
        self.attention_allocation = {}
        self.complexity_score = 0.0
        self.semantic_depth = 0
        self.cognitive_insights = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            **self.original_state,
            'cognitive_analysis': {
                'elements_count': len(self.cognitive_elements),
                'complexity_score': self.complexity_score,
                'semantic_depth': self.semantic_depth,
                'attention_allocation': self.attention_allocation,
                'cognitive_insights': self.cognitive_insights
            }
        }


class CognitiveWorkflowEnhancer:
    """
    Enhances Archon workflows with cognitive grammar processing
    """
    
    def __init__(self):
        self.parser = CognitiveGrammarParser()
        self.encoder = HypergraphEncoder()
        self.attention_engine = ECANAttentionEngine(total_attention_capacity=50.0)
        self.tensor_network = DistributedTensorNetwork(self.attention_engine)
        self.grammar_updater = AdaptiveGrammarUpdater(self.parser)
        self.knowledge_adapter = LocalKnowledgeAdapter()
        
        # Add sample knowledge for better parsing
        self.knowledge_adapter.add_sample_knowledge()
        
        # Performance tracking
        self.processing_history = []
        self.enhancement_stats = {
            'total_processed': 0,
            'complexity_improvements': 0,
            'attention_optimizations': 0
        }
    
    def enhance_user_message(self, user_message: str, context: Optional[Dict[str, Any]] = None) -> CognitiveEnhancedState:
        """
        Enhance user message with cognitive grammar analysis
        
        Args:
            user_message: Original user message
            context: Additional context information
            
        Returns:
            Enhanced state with cognitive analysis
        """
        # Create base state
        base_state = {
            'latest_user_message': user_message,
            'context': context or {}
        }
        
        enhanced_state = CognitiveEnhancedState(base_state)
        
        try:
            # 1. Parse cognitive elements
            elements = self.parser.parse(user_message)
            enhanced_state.cognitive_elements = elements
            
            if elements:
                # 2. Create hypergraph representation
                scheme_graph = self.parser.extract_scheme_patterns(elements)
                nodes, links = self.encoder.encode_scheme_pattern(scheme_graph)
                
                enhanced_state.hypergraph_nodes = nodes
                enhanced_state.hypergraph_links = links
                enhanced_state.complexity_score = scheme_graph['meta']['complexity_score']
                enhanced_state.semantic_depth = max(e.semantic_depth for e in elements)
                
                # 3. Attention allocation
                attention_allocation = self._allocate_attention_for_elements(elements)
                enhanced_state.attention_allocation = attention_allocation
                
                # 4. Generate cognitive insights
                insights = self._generate_cognitive_insights(elements, scheme_graph, attention_allocation)
                enhanced_state.cognitive_insights = insights
                
                # 5. Record for learning
                self._record_processing_trace(enhanced_state)
            
            self.enhancement_stats['total_processed'] += 1
            
        except Exception as e:
            print(f"Error in cognitive enhancement: {e}")
            # Return state with minimal enhancement
            enhanced_state.cognitive_insights = {'error': str(e)}
        
        return enhanced_state
    
    def _allocate_attention_for_elements(self, elements: List[Any]) -> Dict[str, float]:
        """Allocate attention to cognitive elements"""
        allocation = {}
        
        # Register elements with attention engine
        for element in elements:
            element_id = f"{element.element_type.value}_{element.name}"
            
            # Calculate salience based on semantic depth and confidence
            confidence = element.attributes.get('confidence', 0.5)
            salience = (element.semantic_depth / 10.0) * confidence
            urgency = confidence  # Simple urgency based on confidence
            complexity = element.semantic_depth / 5.0
            
            self.attention_engine.register_attention_unit(
                element_id, salience, urgency, complexity
            )
        
        # Allocate attention
        if elements:
            allocation = self.attention_engine.allocate_attention()
        
        return allocation
    
    def _generate_cognitive_insights(self, elements: List[Any], scheme_graph: Dict[str, Any], 
                                   attention_allocation: Dict[str, float]) -> Dict[str, Any]:
        """Generate insights from cognitive analysis"""
        insights = {}
        
        if elements:
            # Element type distribution
            element_types = [e.element_type.value for e in elements]
            type_counts = {}
            for et in element_types:
                type_counts[et] = type_counts.get(et, 0) + 1
            
            insights['element_distribution'] = type_counts
            
            # Complexity analysis
            avg_depth = sum(e.semantic_depth for e in elements) / len(elements)
            insights['average_semantic_depth'] = avg_depth
            insights['complexity_category'] = self._categorize_complexity(scheme_graph['meta']['complexity_score'])
            
            # Attention insights
            if attention_allocation:
                max_attention_element = max(attention_allocation.items(), key=lambda x: x[1])
                insights['primary_focus'] = max_attention_element[0]
                insights['attention_diversity'] = len(attention_allocation)
            
            # Recommendations
            insights['workflow_recommendations'] = self._generate_workflow_recommendations(elements, attention_allocation)
        
        return insights
    
    def _categorize_complexity(self, complexity_score: float) -> str:
        """Categorize complexity score"""
        if complexity_score < 2.0:
            return "simple"
        elif complexity_score < 5.0:
            return "moderate"
        elif complexity_score < 8.0:
            return "complex"
        else:
            return "very_complex"
    
    def _generate_workflow_recommendations(self, elements: List[Any], 
                                         attention_allocation: Dict[str, float]) -> List[str]:
        """Generate workflow recommendations based on cognitive analysis"""
        recommendations = []
        
        if not elements:
            return ["Consider providing more specific requirements"]
        
        # Check for agent elements
        agent_elements = [e for e in elements if e.element_type.value == 'agent']
        if len(agent_elements) > 1:
            recommendations.append("Multiple agents detected - consider multi-agent coordination")
        
        # Check complexity
        avg_depth = sum(e.semantic_depth for e in elements) / len(elements)
        if avg_depth > 5:
            recommendations.append("High complexity detected - recommend iterative development")
        
        # Check attention distribution
        if attention_allocation:
            attention_values = list(attention_allocation.values())
            if max(attention_values) > 0.7:
                recommendations.append("Focused attention pattern - consider specialized agent design")
            elif max(attention_values) < 0.3:
                recommendations.append("Distributed attention - consider modular architecture")
        
        # Check for specific patterns
        element_names = [e.name.lower() for e in elements]
        if any(word in ' '.join(element_names) for word in ['complex', 'advanced', 'sophisticated']):
            recommendations.append("Advanced requirements detected - recommend thorough planning phase")
        
        return recommendations or ["Standard workflow approach recommended"]
    
    def _record_processing_trace(self, enhanced_state: CognitiveEnhancedState):
        """Record processing trace for learning"""
        trace = {
            'timestamp': asyncio.get_event_loop().time() if asyncio.get_event_loop().is_running() else 0,
            'elements_count': len(enhanced_state.cognitive_elements),
            'complexity_score': enhanced_state.complexity_score,
            'semantic_depth': enhanced_state.semantic_depth,
            'attention_count': len(enhanced_state.attention_allocation)
        }
        
        self.processing_history.append(trace)
        
        # Keep only recent history
        if len(self.processing_history) > 100:
            self.processing_history = self.processing_history[-100:]
    
    def enhance_scope_definition(self, scope_text: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance scope definition with cognitive insights"""
        enhanced_state = self.enhance_user_message(scope_text, context)
        
        enhancement = {
            'original_scope': scope_text,
            'cognitive_enhancement': enhanced_state.cognitive_insights,
            'complexity_analysis': {
                'score': enhanced_state.complexity_score,
                'category': enhanced_state.cognitive_insights.get('complexity_category', 'unknown'),
                'semantic_depth': enhanced_state.semantic_depth
            },
            'recommended_approach': enhanced_state.cognitive_insights.get('workflow_recommendations', [])
        }
        
        return enhancement
    
    def enhance_agent_creation(self, agent_request: str, existing_scope: str) -> Dict[str, Any]:
        """Enhance agent creation with cognitive understanding"""
        # Combine request with scope for context
        combined_text = f"{agent_request}\n\nContext: {existing_scope}"
        enhanced_state = self.enhance_user_message(combined_text)
        
        # Generate agent-specific insights
        agent_insights = {
            'cognitive_complexity': enhanced_state.complexity_score,
            'required_capabilities': [],
            'attention_priorities': enhanced_state.attention_allocation,
            'architectural_suggestions': []
        }
        
        # Analyze for specific capabilities
        if enhanced_state.cognitive_elements:
            for element in enhanced_state.cognitive_elements:
                if element.element_type.value == 'agent':
                    agent_insights['required_capabilities'].append(element.name)
                elif element.element_type.value == 'task':
                    agent_insights['required_capabilities'].append(f"task_{element.name}")
        
        # Architectural suggestions based on complexity
        if enhanced_state.complexity_score > 5:
            agent_insights['architectural_suggestions'].append("Consider modular agent design")
        if enhanced_state.semantic_depth > 6:
            agent_insights['architectural_suggestions'].append("Implement hierarchical reasoning")
        if len(enhanced_state.attention_allocation) > 3:
            agent_insights['architectural_suggestions'].append("Use attention-based resource allocation")
        
        return agent_insights
    
    def get_enhancement_statistics(self) -> Dict[str, Any]:
        """Get statistics about cognitive enhancements"""
        stats = self.enhancement_stats.copy()
        
        if self.processing_history:
            recent_history = self.processing_history[-20:]  # Last 20 processes
            stats['recent_metrics'] = {
                'average_complexity': sum(h['complexity_score'] for h in recent_history) / len(recent_history),
                'average_elements': sum(h['elements_count'] for h in recent_history) / len(recent_history),
                'processing_trend': 'improving' if len(recent_history) > 1 and 
                                  recent_history[-1]['complexity_score'] > recent_history[0]['complexity_score'] 
                                  else 'stable'
            }
        
        # Add component statistics
        stats['attention_statistics'] = self.attention_engine.get_attention_statistics()
        stats['grammar_statistics'] = self.grammar_updater.get_learning_statistics()
        
        return stats
    
    def export_cognitive_state(self) -> Dict[str, Any]:
        """Export complete cognitive state for persistence"""
        return {
            'enhancement_stats': self.enhancement_stats,
            'processing_history': self.processing_history[-50:],  # Last 50 entries
            'attention_config': self.attention_engine.export_configuration(),
            'grammar_learning': self.grammar_updater.export_learning_state(),
            'tensor_network_state': self.tensor_network.export_network_state() if hasattr(self.tensor_network, 'export_network_state') else {}
        }


# Integration functions for existing Archon workflow
async def enhance_define_scope_with_reasoner(state: Dict[str, Any], enhancer: CognitiveWorkflowEnhancer) -> Dict[str, Any]:
    """
    Enhanced version of define_scope_with_reasoner that includes cognitive analysis
    """
    user_message = state.get('latest_user_message', '')
    
    # Get cognitive enhancement
    enhanced_state = enhancer.enhance_user_message(user_message)
    
    # Add cognitive insights to the scope definition
    scope_enhancement = enhancer.enhance_scope_definition(user_message, state)
    
    # Update state with cognitive analysis
    enhanced_archon_state = enhanced_state.to_dict()
    enhanced_archon_state['cognitive_scope_enhancement'] = scope_enhancement
    
    return enhanced_archon_state


async def enhance_coder_agent(state: Dict[str, Any], enhancer: CognitiveWorkflowEnhancer) -> Dict[str, Any]:
    """
    Enhanced version of coder_agent that uses cognitive understanding
    """
    user_message = state.get('latest_user_message', '')
    scope = state.get('scope', '')
    
    # Get cognitive enhancement for agent creation
    agent_insights = enhancer.enhance_agent_creation(user_message, scope)
    
    # Add insights to state
    enhanced_state = state.copy()
    enhanced_state['cognitive_agent_insights'] = agent_insights
    
    return enhanced_state


def create_cognitive_enhanced_workflow() -> CognitiveWorkflowEnhancer:
    """
    Create a cognitive workflow enhancer for integration with Archon
    """
    return CognitiveWorkflowEnhancer()