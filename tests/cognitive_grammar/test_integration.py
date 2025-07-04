"""
Test Cognitive Grammar Integration

Integration tests for cognitive grammar with existing Archon workflow.
"""

import pytest
import numpy as np
import asyncio
from typing import Dict, Any

# Import cognitive grammar components
from archon.cognitive_grammar import (
    CognitiveGrammarParser,
    HypergraphEncoder, 
    NodeTensorKernel,
    LinkTensorKernel,
    AttentionTensorKernel,
    ECANAttentionEngine,
    DistributedTensorNetwork,
    AdaptiveGrammarUpdater,
    VectorKnowledgeAdapter,
    LocalKnowledgeAdapter
)


class TestCognitiveGrammarIntegration:
    """Integration tests for the complete cognitive grammar system"""
    
    def setup_method(self):
        """Setup test components"""
        self.parser = CognitiveGrammarParser()
        self.encoder = HypergraphEncoder()
        self.attention_engine = ECANAttentionEngine()
        self.tensor_network = DistributedTensorNetwork(self.attention_engine)
        self.grammar_updater = AdaptiveGrammarUpdater(self.parser)
        self.knowledge_adapter = LocalKnowledgeAdapter()
        
        # Add sample knowledge
        self.knowledge_adapter.add_sample_knowledge()
    
    def test_end_to_end_cognitive_flow(self):
        """Test complete cognitive flow from parsing to tensor network execution"""
        
        # 1. Parse agentic task request
        task_request = """Create an intelligent coding agent that can analyze requirements, 
                         search knowledge bases, and generate Python code using best practices"""
        
        elements = self.parser.parse(task_request)
        assert len(elements) > 0
        
        # 2. Convert to hypergraph representation
        scheme_graph = self.parser.extract_scheme_patterns(elements)
        nodes, links = self.encoder.encode_scheme_pattern(scheme_graph)
        
        assert len(nodes) > 0
        assert len(links) >= 0  # May be 0 for simple cases
        
        # 3. Build tensor network
        element_mapping = self.tensor_network.build_from_hypergraph(nodes, links)
        assert len(element_mapping) > 0
        
        # 4. Verify network structure
        network_stats = self.tensor_network.get_network_statistics()
        assert network_stats['total_kernels'] > 0
        assert network_stats['network_state'] == 'initializing'
    
    async def test_async_workflow_activation(self):
        """Test asynchronous workflow activation"""
        
        # Setup simple network
        task_request = "Create a reasoning agent for data analysis"
        elements = self.parser.parse(task_request)
        scheme_graph = self.parser.extract_scheme_patterns(elements)
        nodes, links = self.encoder.encode_scheme_pattern(scheme_graph)
        
        element_mapping = self.tensor_network.build_from_hypergraph(nodes, links)
        
        # Initialize network
        await self.tensor_network.initialize_network()
        
        # Create initial activation
        initial_activation = {}
        if element_mapping:
            first_kernel_id = list(element_mapping.values())[0]
            initial_activation[first_kernel_id] = np.random.normal(0, 0.1, (4, 4))
        
        # Activate workflow (short run for testing)
        final_states = await self.tensor_network.activate_workflow(
            initial_activation, max_iterations=10
        )
        
        assert len(final_states) > 0
        assert self.tensor_network.state.value in ['converging', 'stable', 'active']
    
    def test_attention_allocation_integration(self):
        """Test attention allocation with cognitive elements"""
        
        # Register cognitive elements with attention engine
        self.attention_engine.register_attention_unit("coding_agent", 0.8, 0.7, 2.0)
        self.attention_engine.register_attention_unit("reasoning_agent", 0.9, 0.8, 2.5)
        self.attention_engine.register_attention_unit("data_processor", 0.6, 0.5, 1.5)
        
        # Allocate attention
        allocation = self.attention_engine.allocate_attention()
        
        assert len(allocation) == 3
        assert all(amount >= 0 for amount in allocation.values())
        
        # Test attention updates
        self.attention_engine.update_unit_salience("coding_agent", 0.2, 0.3)
        new_allocation = self.attention_engine.allocate_attention()
        
        # Coding agent should receive more attention
        assert new_allocation["coding_agent"] >= allocation["coding_agent"]
    
    def test_knowledge_integration(self):
        """Test knowledge adapter integration"""
        
        # Test knowledge retrieval
        async def run_knowledge_test():
            fragments = await self.knowledge_adapter.retrieve_knowledge("AI agent", limit=3)
            assert len(fragments) > 0
            
            # Test tensor conversion
            tensor_input = self.knowledge_adapter.convert_to_tensor_input(fragments)
            assert tensor_input.shape[0] > 0
            assert tensor_input.shape[1] == self.knowledge_adapter.embedding_dim
            
            return fragments
        
        # Run async test
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            fragments = loop.run_until_complete(run_knowledge_test())
            assert len(fragments) > 0
        finally:
            loop.close()
    
    def test_grammar_learning_cycle(self):
        """Test learning and grammar updating cycle"""
        
        # Create initial elements
        elements = self.parser.parse("Create intelligent agent for complex tasks")
        initial_count = len(elements)
        
        # Simulate learning traces
        for i in range(15):  # Enough to trigger update
            self.grammar_updater.record_learning_trace(
                element_id=f"test_element_{i % 3}",
                performance_score=0.7 + 0.1 * np.random.random(),
                attention_received=0.5 + 0.2 * np.random.random(),
                activation_pattern=np.random.normal(0, 0.1, 10),
                context_factors={'iteration': i}
            )
        
        # Check that updates were generated
        learning_stats = self.grammar_updater.get_learning_statistics()
        assert learning_stats['total_traces'] >= 15
        assert learning_stats['total_updates'] > 0
    
    def test_hypergraph_statistics(self):
        """Test hypergraph statistical analysis"""
        
        # Create complex hypergraph
        complex_request = """Design a multi-agent system with reasoning agents, 
                           coding agents, knowledge retrievers, and workflow orchestrators 
                           that collaborate on software development tasks"""
        
        elements = self.parser.parse(complex_request)
        scheme_graph = self.parser.extract_scheme_patterns(elements)
        nodes, links = self.encoder.encode_scheme_pattern(scheme_graph)
        
        # Get statistics
        stats = self.encoder.compute_graph_statistics()
        
        assert stats['nodes'] > 0
        assert stats['density'] >= 0.0
        assert stats['avg_degree'] >= 0.0
        assert stats['complexity'] >= 0.0
    
    def test_tensor_kernel_interaction(self):
        """Test interaction between different tensor kernels"""
        
        # Create kernels
        node_kernel = NodeTensorKernel(8, 4, "agent")
        link_kernel = LinkTensorKernel(2, 3, 4)
        attention_kernel = AttentionTensorKernel(2, 2, 2)
        
        # Test node forward pass
        node_output = node_kernel.forward(
            np.ones(8) * 0.5,
            np.ones(4) * 0.3
        )
        assert node_output.shape == (8, 4)
        
        # Test link forward pass
        link_output = link_kernel.forward(
            np.ones(2) * 0.6,
            np.ones(3) * 0.4
        )
        assert link_output.shape == (2, 3, 4)
        
        # Test attention forward pass
        attention_output = attention_kernel.forward(
            np.ones(2) * 0.7,
            np.ones(2) * 0.8,
            np.ones(2) * 0.9
        )
        assert attention_output.shape == (2, 2, 2)
        assert np.allclose(np.sum(attention_output), 1.0)
    
    def test_performance_tracking(self):
        """Test performance tracking across components"""
        
        # Test attention engine performance
        attention_stats = self.attention_engine.get_attention_statistics()
        assert 'total_units' in attention_stats
        assert 'attention_utilization' in attention_stats
        
        # Test grammar updater performance
        learning_stats = self.grammar_updater.get_learning_statistics()
        assert 'total_traces' in learning_stats
        assert 'update_strategy' in learning_stats
        
        # Test knowledge adapter performance
        knowledge_stats = self.knowledge_adapter.get_knowledge_statistics()
        assert 'embedding_dimension' in knowledge_stats
    
    def test_error_handling(self):
        """Test error handling in cognitive grammar components"""
        
        # Test empty input handling
        empty_elements = self.parser.parse("")
        assert empty_elements == []
        
        empty_scheme = self.parser.extract_scheme_patterns([])
        assert empty_scheme['meta']['total_elements'] == 0
        
        # Test invalid tensor operations
        kernel = NodeTensorKernel(4, 4, "test")
        
        # Test with mismatched input sizes
        output = kernel.forward(
            np.ones(2),  # Wrong size
            np.ones(6)   # Wrong size
        )
        assert output.shape == (4, 4)  # Should handle gracefully
    
    def test_configuration_export_import(self):
        """Test configuration export and import"""
        
        # Test attention engine export/import
        attention_config = self.attention_engine.export_configuration()
        assert 'attention_capacity' in attention_config
        assert 'units' in attention_config
        
        new_attention_engine = ECANAttentionEngine()
        new_attention_engine.import_configuration(attention_config)
        
        # Should have same configuration
        assert new_attention_engine.attention_resource.total_capacity == self.attention_engine.attention_resource.total_capacity
        
        # Test grammar updater export/import
        learning_state = self.grammar_updater.export_learning_state()
        assert 'pattern_weights' in learning_state
        assert 'learning_parameters' in learning_state
        
        new_updater = AdaptiveGrammarUpdater(self.parser)
        new_updater.import_learning_state(learning_state)
        
        # Should have same pattern weights
        assert new_updater.pattern_weights == self.grammar_updater.pattern_weights


class TestCognitiveGrammarWithArchon:
    """Test integration with existing Archon components"""
    
    def test_agent_state_compatibility(self):
        """Test compatibility with Archon AgentState structure"""
        
        # Simulate Archon agent state
        mock_agent_state = {
            'latest_user_message': 'Create a Python coding agent with error handling',
            'scope': 'AI agent development',
            'advisor_output': 'Recommend using Pydantic AI framework'
        }
        
        # Parse user message with cognitive grammar
        parser = CognitiveGrammarParser()
        elements = parser.parse(mock_agent_state['latest_user_message'])
        
        # Should extract relevant cognitive elements
        assert len(elements) > 0
        
        # Convert to scheme representation for tensor network
        scheme_graph = parser.extract_scheme_patterns(elements)
        assert 'nodes' in scheme_graph
        assert scheme_graph['meta']['total_elements'] > 0
    
    def test_workflow_enhancement(self):
        """Test how cognitive grammar can enhance existing workflows"""
        
        # Simulate workflow enhancement
        original_prompt = "Create an AI agent"
        
        # Parse with cognitive grammar
        parser = CognitiveGrammarParser()
        elements = parser.parse(original_prompt)
        
        if elements:
            # Enhanced understanding
            enhanced_context = {
                'cognitive_elements': len(elements),
                'complexity_score': parser.extract_scheme_patterns(elements)['meta']['complexity_score'],
                'semantic_depth': max(e.semantic_depth for e in elements),
                'element_types': list(set(e.element_type.value for e in elements))
            }
            
            assert enhanced_context['cognitive_elements'] > 0
            assert enhanced_context['complexity_score'] >= 0
            assert enhanced_context['semantic_depth'] > 0


if __name__ == "__main__":
    # Run integration tests
    print("Running Cognitive Grammar Integration Tests...")
    
    test = TestCognitiveGrammarIntegration()
    test.setup_method()
    
    print("\n1. Testing end-to-end cognitive flow...")
    test.test_end_to_end_cognitive_flow()
    print("✓ End-to-end cognitive flow test passed")
    
    print("\n2. Testing attention allocation integration...")
    test.test_attention_allocation_integration()
    print("✓ Attention allocation integration test passed")
    
    print("\n3. Testing knowledge integration...")
    test.test_knowledge_integration()
    print("✓ Knowledge integration test passed")
    
    print("\n4. Testing grammar learning cycle...")
    test.test_grammar_learning_cycle()
    print("✓ Grammar learning cycle test passed")
    
    print("\n5. Testing hypergraph statistics...")
    test.test_hypergraph_statistics()
    print("✓ Hypergraph statistics test passed")
    
    print("\n6. Testing tensor kernel interaction...")
    test.test_tensor_kernel_interaction()
    print("✓ Tensor kernel interaction test passed")
    
    print("\n7. Testing performance tracking...")
    test.test_performance_tracking()
    print("✓ Performance tracking test passed")
    
    print("\n8. Testing error handling...")
    test.test_error_handling()
    print("✓ Error handling test passed")
    
    print("\n9. Testing configuration export/import...")
    test.test_configuration_export_import()
    print("✓ Configuration export/import test passed")
    
    print("\nTesting Archon Integration...")
    archon_test = TestCognitiveGrammarWithArchon()
    
    print("\n10. Testing agent state compatibility...")
    archon_test.test_agent_state_compatibility()
    print("✓ Agent state compatibility test passed")
    
    print("\n11. Testing workflow enhancement...")
    archon_test.test_workflow_enhancement()
    print("✓ Workflow enhancement test passed")
    
    print("\n" + "="*50)
    print("🧠 All Cognitive Grammar Integration Tests Passed! 🧠")
    print("="*50)