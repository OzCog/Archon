"""
Test Tensor Kernels

Tests for cognitive grammar tensor kernels (nodes, links, attention).
"""

import pytest
import numpy as np
from archon.cognitive_grammar.tensor_kernels import (
    NodeTensorKernel, LinkTensorKernel, AttentionTensorKernel,
    TensorNetworkValidator
)


class TestNodeTensorKernel:
    
    def test_node_kernel_initialization(self):
        """Test node kernel initializes correctly"""
        kernel = NodeTensorKernel(16, 8, "agent")
        
        assert kernel.tensor_shape == (16, 8)
        assert kernel.n_features == 16
        assert kernel.n_contexts == 8
        assert kernel.node_type == "agent"
        assert kernel.tensor_data.shape == (16, 8)
    
    def test_node_kernel_forward_pass(self):
        """Test node kernel forward pass"""
        kernel = NodeTensorKernel(8, 4, "task")
        
        input_features = np.ones(8) * 0.5
        context_state = np.ones(4) * 0.3
        
        output = kernel.forward(input_features, context_state)
        
        assert output.shape == (8, 4)
        assert np.all(np.abs(output) <= 1.0)  # tanh bounded output
    
    def test_node_kernel_attention_update(self):
        """Test attention weight updates"""
        kernel = NodeTensorKernel(4, 4, "agent")
        initial_attention = kernel.attention_weight
        
        kernel.update_attention(0.5)  # Positive signal
        assert kernel.attention_weight > initial_attention
        
        kernel.update_attention(-0.3)  # Negative signal  
        assert kernel.attention_weight < (initial_attention + 0.5 * 0.01)
    
    def test_node_kernel_activation_statistics(self):
        """Test activation statistics tracking"""
        kernel = NodeTensorKernel(4, 4, "task")
        
        # Perform several forward passes
        for i in range(10):
            input_features = np.random.normal(0, 0.1, 4)
            context_state = np.random.normal(0, 0.1, 4)
            kernel.forward(input_features, context_state)
        
        stats = kernel.get_activation_statistics()
        assert 'mean' in stats
        assert 'std' in stats
        assert 'min' in stats
        assert 'max' in stats
        assert len(kernel.activation_history) == 10


class TestLinkTensorKernel:
    
    def test_link_kernel_initialization(self):
        """Test link kernel initializes correctly"""
        kernel = LinkTensorKernel(3, 6, 4)
        
        assert kernel.tensor_shape == (3, 6, 4)
        assert kernel.n_links == 3
        assert kernel.n_relation_types == 6
        assert kernel.n_weights == 4
    
    def test_link_kernel_forward_pass(self):
        """Test link kernel forward pass"""
        kernel = LinkTensorKernel(2, 3, 4)
        
        source_activations = np.array([0.5, 0.7])
        relation_mask = np.array([1.0, 0.5, 0.8])
        
        output = kernel.forward(source_activations, relation_mask)
        
        assert output.shape == (2, 3, 4)
        assert np.all(output >= 0.0) and np.all(output <= 1.0)  # sigmoid bounded
    
    def test_link_weight_updates(self):
        """Test link weight updates"""
        kernel = LinkTensorKernel(2, 2, 3)
        
        initial_weights = kernel.tensor_data[0, 0, :].copy()
        weight_update = np.array([0.1, -0.1, 0.05])
        
        kernel.update_link_weights(0, 0, weight_update)
        
        updated_weights = kernel.tensor_data[0, 0, :]
        assert not np.array_equal(initial_weights, updated_weights)
    
    def test_propagation_statistics(self):
        """Test propagation statistics"""
        kernel = LinkTensorKernel(2, 2, 3)
        
        # Perform several forward passes
        for i in range(5):
            source_activations = np.random.uniform(0, 1, 2)
            relation_mask = np.random.uniform(0, 1, 2)
            kernel.forward(source_activations, relation_mask)
        
        stats = kernel.get_propagation_statistics()
        assert 'mean' in stats
        assert 'efficiency' in stats
        assert len(kernel.propagation_history) == 5


class TestAttentionTensorKernel:
    
    def test_attention_kernel_initialization(self):
        """Test attention kernel initializes correctly"""
        kernel = AttentionTensorKernel(3, 4, 5)
        
        assert kernel.tensor_shape == (3, 4, 5)
        assert kernel.n_agents == 3
        assert kernel.n_tasks == 4
        assert kernel.n_resources == 5
        
        # Should be normalized probability distribution
        assert np.allclose(np.sum(kernel.tensor_data, axis=(1, 2)), np.ones(3))
    
    def test_attention_forward_pass(self):
        """Test attention kernel forward pass"""
        kernel = AttentionTensorKernel(2, 3, 2)
        
        agent_states = np.array([0.8, 0.6])
        task_demands = np.array([0.9, 0.7, 0.5])
        resource_availability = np.array([0.8, 0.9])
        
        output = kernel.forward(agent_states, task_demands, resource_availability)
        
        assert output.shape == (2, 3, 2)
        assert np.allclose(np.sum(output), 1.0)  # Should sum to 1
        assert np.all(output >= 0.0)  # Should be non-negative
    
    def test_attention_temperature_control(self):
        """Test temperature control for attention sharpness"""
        kernel = AttentionTensorKernel(2, 2, 2)
        
        agent_states = np.array([1.0, 0.5])
        task_demands = np.array([1.0, 0.5])
        resource_availability = np.array([1.0, 1.0])
        
        # Low temperature (sharp attention)
        kernel.set_temperature(0.1)
        output_sharp = kernel.forward(agent_states, task_demands, resource_availability)
        
        # High temperature (diffuse attention)
        kernel.set_temperature(10.0)
        output_diffuse = kernel.forward(agent_states, task_demands, resource_availability)
        
        # Sharp attention should have higher max value
        assert np.max(output_sharp) > np.max(output_diffuse)
    
    def test_attention_weight_updates(self):
        """Test attention weight updates"""
        kernel = AttentionTensorKernel(2, 2, 2)
        
        initial_weight = kernel.tensor_data[0, 0, 0]
        kernel.update_attention_weights(0, 0, 0, 0.1)
        updated_weight = kernel.tensor_data[0, 0, 0]
        
        assert updated_weight != initial_weight
        
        # Should maintain normalization
        assert np.allclose(np.sum(kernel.tensor_data[0, :, :]), 1.0, rtol=1e-3)
    
    def test_attention_focus_metrics(self):
        """Test attention focus metrics"""
        kernel = AttentionTensorKernel(2, 2, 2)
        
        focus_info = kernel.get_attention_focus()
        
        assert 'max_attention_idx' in focus_info
        assert 'max_attention_value' in focus_info
        assert 'concentration_ratio' in focus_info
        assert 'temperature' in focus_info
    
    def test_agent_workload(self):
        """Test agent workload calculation"""
        kernel = AttentionTensorKernel(3, 2, 2)
        
        workload = kernel.get_agent_workload(0)
        
        assert 'total_attention' in workload
        assert 'task_distribution' in workload
        assert 'resource_distribution' in workload
        assert len(workload['task_distribution']) == 2
        assert len(workload['resource_distribution']) == 2


class TestTensorNetworkValidator:
    
    def test_node_kernel_validation(self):
        """Test node kernel validation"""
        valid_kernel = NodeTensorKernel(8, 4, "agent")
        checks = TensorNetworkValidator.validate_node_kernel(valid_kernel)
        
        assert checks['valid_shape']
        assert checks['positive_dimensions']
        assert checks['finite_values']
        assert checks['reasonable_size']
        assert checks['valid_attention']
    
    def test_link_kernel_validation(self):
        """Test link kernel validation"""
        valid_kernel = LinkTensorKernel(2, 3, 4)
        checks = TensorNetworkValidator.validate_link_kernel(valid_kernel)
        
        assert checks['valid_shape']
        assert checks['positive_dimensions']
        assert checks['finite_values']
        assert checks['reasonable_size']
        assert checks['bounded_weights']
    
    def test_attention_kernel_validation(self):
        """Test attention kernel validation"""
        valid_kernel = AttentionTensorKernel(2, 3, 4)
        checks = TensorNetworkValidator.validate_attention_kernel(valid_kernel)
        
        assert checks['valid_shape']
        assert checks['positive_dimensions']
        assert checks['finite_values']
        assert checks['probability_distribution']
        assert checks['positive_values']
        assert checks['valid_temperature']
    
    def test_tensor_compatibility(self):
        """Test tensor compatibility validation"""
        kernels = [
            NodeTensorKernel(4, 4, "agent"),
            LinkTensorKernel(2, 3, 4),
            AttentionTensorKernel(2, 2, 2)
        ]
        
        checks = TensorNetworkValidator.validate_tensor_compatibility(kernels)
        
        assert 'consistent_dtypes' in checks
        assert 'reasonable_total_size' in checks


if __name__ == "__main__":
    # Run basic tests without pytest
    print("Testing Node Tensor Kernel...")
    
    test_node = TestNodeTensorKernel()
    test_node.test_node_kernel_initialization()
    print("✓ Node kernel initialization")
    
    test_node.test_node_kernel_forward_pass()
    print("✓ Node kernel forward pass")
    
    test_node.test_node_kernel_attention_update()
    print("✓ Node kernel attention update")
    
    print("\nTesting Link Tensor Kernel...")
    
    test_link = TestLinkTensorKernel()
    test_link.test_link_kernel_initialization()
    print("✓ Link kernel initialization")
    
    test_link.test_link_kernel_forward_pass()
    print("✓ Link kernel forward pass")
    
    print("\nTesting Attention Tensor Kernel...")
    
    test_attention = TestAttentionTensorKernel()
    test_attention.test_attention_kernel_initialization()
    print("✓ Attention kernel initialization")
    
    test_attention.test_attention_forward_pass()
    print("✓ Attention kernel forward pass")
    
    test_attention.test_attention_temperature_control()
    print("✓ Attention temperature control")
    
    print("\nTesting Tensor Network Validator...")
    
    test_validator = TestTensorNetworkValidator()
    test_validator.test_node_kernel_validation()
    print("✓ Node kernel validation")
    
    test_validator.test_tensor_compatibility()
    print("✓ Tensor compatibility validation")
    
    print("\nAll tensor kernel tests passed!")