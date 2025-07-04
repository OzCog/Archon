"""
Simple test runner for cognitive grammar without external dependencies
"""

import sys
import os
import numpy as np

# Add the project root to path
sys.path.insert(0, '/home/runner/work/Archon/Archon')

try:
    from archon.cognitive_grammar.cognitive_parser import (
        CognitiveGrammarParser, GrammarElement, GrammarElementType
    )
    from archon.cognitive_grammar.tensor_kernels import (
        NodeTensorKernel, LinkTensorKernel, AttentionTensorKernel
    )
    
    def test_cognitive_parser():
        """Test cognitive grammar parser"""
        print("Testing Cognitive Grammar Parser...")
        
        parser = CognitiveGrammarParser()
        
        # Test 1: Basic initialization
        assert len(parser.grammar_patterns) > 0
        print("✓ Parser initialization")
        
        # Test 2: Simple parsing
        text = "Create a coding agent that writes Python code"
        elements = parser.parse(text)
        assert len(elements) > 0
        print("✓ Simple parsing")
        
        # Test 3: Scheme pattern extraction
        scheme_graph = parser.extract_scheme_patterns(elements)
        assert 'nodes' in scheme_graph
        assert 'links' in scheme_graph
        print("✓ Scheme pattern extraction")
        
        print("Cognitive Parser tests passed!\n")
    
    def test_tensor_kernels():
        """Test tensor kernels"""
        print("Testing Tensor Kernels...")
        
        # Test Node Kernel
        node_kernel = NodeTensorKernel(8, 4, "agent")
        assert node_kernel.tensor_shape == (8, 4)
        
        input_features = np.ones(8) * 0.5
        context_state = np.ones(4) * 0.3
        output = node_kernel.forward(input_features, context_state)
        assert output.shape == (8, 4)
        print("✓ Node kernel")
        
        # Test Link Kernel
        link_kernel = LinkTensorKernel(2, 3, 4)
        assert link_kernel.tensor_shape == (2, 3, 4)
        
        source_activations = np.array([0.5, 0.7])
        relation_mask = np.array([1.0, 0.5, 0.8])
        output = link_kernel.forward(source_activations, relation_mask)
        assert output.shape == (2, 3, 4)
        print("✓ Link kernel")
        
        # Test Attention Kernel
        attention_kernel = AttentionTensorKernel(2, 3, 2)
        assert attention_kernel.tensor_shape == (2, 3, 2)
        
        agent_states = np.array([0.8, 0.6])
        task_demands = np.array([0.9, 0.7, 0.5])
        resource_availability = np.array([0.8, 0.9])
        output = attention_kernel.forward(agent_states, task_demands, resource_availability)
        assert output.shape == (2, 3, 2)
        print("✓ Attention kernel")
        
        print("Tensor Kernel tests passed!\n")
    
    def test_integration():
        """Test basic integration"""
        print("Testing Basic Integration...")
        
        # Create parser and parse text
        parser = CognitiveGrammarParser()
        text = "Create an intelligent agent for data analysis tasks"
        elements = parser.parse(text)
        
        # Convert to scheme graph
        scheme_graph = parser.extract_scheme_patterns(elements)
        
        # Verify structure
        assert scheme_graph['meta']['total_elements'] == len(elements)
        assert len(scheme_graph['nodes']) > 0
        
        print("✓ Parser to scheme conversion")
        
        # Test tensor kernel creation from cognitive elements
        if scheme_graph['nodes']:
            node_data = scheme_graph['nodes'][0]
            semantic_depth = node_data.get('semantic_depth', 1)
            
            # Create tensor kernel based on semantic depth
            n_features = max(8, semantic_depth * 2)
            n_contexts = max(4, semantic_depth)
            
            kernel = NodeTensorKernel(n_features, n_contexts, node_data.get('type', 'agent'))
            assert kernel.tensor_shape == (n_features, n_contexts)
            print("✓ Cognitive element to tensor kernel")
        
        print("Basic Integration tests passed!\n")
    
    # Run all tests
    if __name__ == "__main__":
        print("="*60)
        print("🧠 COGNITIVE GRAMMAR TEST SUITE 🧠")
        print("="*60)
        print()
        
        try:
            test_cognitive_parser()
            test_tensor_kernels()
            test_integration()
            
            print("="*60)
            print("🎉 ALL TESTS PASSED! 🎉")
            print("="*60)
            print("\nCognitive Grammar implementation is working correctly!")
            print("Ready for integration with Archon workflow.")
            
        except Exception as e:
            print(f"❌ Test failed: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure all cognitive grammar modules are available")
    sys.exit(1)