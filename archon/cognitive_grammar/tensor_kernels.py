"""
Tensor Kernels for Cognitive Grammar

Implements ggml-style tensor kernels for nodes, links, and attention
with explicit tensor shapes and cognitive operations.
"""

try:
    import numpy as np
except ImportError:
    # Use mock numpy for testing
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'tests'))
    import mock_numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import json


class TensorKernel(ABC):
    """Abstract base class for tensor kernels"""
    
    def __init__(self, tensor_shape: Tuple[int, ...], dtype: type = np.float32):
        self.tensor_shape = tensor_shape
        self.dtype = dtype
        self.tensor_data = np.zeros(tensor_shape, dtype=dtype)
        self.metadata = {}
    
    @abstractmethod
    def forward(self, *args, **kwargs) -> np.ndarray:
        """Forward pass computation"""
        pass
    
    @abstractmethod
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Backward pass for gradient computation"""
        pass
    
    def get_tensor_info(self) -> Dict[str, Any]:
        """Get tensor information"""
        return {
            'shape': self.tensor_shape,
            'dtype': str(self.dtype),
            'size': self.tensor_data.size,
            'memory_mb': self.tensor_data.nbytes / (1024 * 1024)
        }


@dataclass
class NodeTensorKernel(TensorKernel):
    """
    Tensor kernel for hypergraph nodes
    Shape: [n_features, n_contexts] representing cognitive node state
    """
    
    def __init__(self, n_features: int, n_contexts: int, node_type: str = "generic"):
        super().__init__((n_features, n_contexts))
        self.n_features = n_features
        self.n_contexts = n_contexts
        self.node_type = node_type
        self.activation_history = []
        self.attention_weight = 1.0
        
        # Initialize with small random values
        self.tensor_data = np.random.normal(0, 0.1, (n_features, n_contexts)).astype(self.dtype)
        
        # Add type-specific initialization
        self._initialize_by_type()
    
    def _initialize_by_type(self):
        """Initialize tensor based on node type"""
        type_initializers = {
            'agent': self._init_agent_kernel,
            'task': self._init_task_kernel,
            'resource': self._init_resource_kernel,
            'workflow': self._init_workflow_kernel,
            'knowledge': self._init_knowledge_kernel,
            'constraint': self._init_constraint_kernel
        }
        
        initializer = type_initializers.get(self.node_type, self._init_generic_kernel)
        initializer()
    
    def _init_agent_kernel(self):
        """Initialize agent-specific kernel parameters"""
        # Agent nodes have higher activation potential
        self.tensor_data *= 1.5
        self.attention_weight = 1.2
        
    def _init_task_kernel(self):
        """Initialize task-specific kernel parameters"""
        # Task nodes have structured feature patterns
        self.tensor_data[:self.n_features//2, :] *= 1.3
        self.attention_weight = 1.0
        
    def _init_resource_kernel(self):
        """Initialize resource-specific kernel parameters"""
        # Resource nodes have lower base activation
        self.tensor_data *= 0.8
        self.attention_weight = 0.8
        
    def _init_workflow_kernel(self):
        """Initialize workflow-specific kernel parameters"""  
        # Workflow nodes have high connectivity features
        self.tensor_data *= 1.4
        self.attention_weight = 1.3
        
    def _init_knowledge_kernel(self):
        """Initialize knowledge-specific kernel parameters"""
        # Knowledge nodes have rich contextual features
        self.tensor_data[:, :self.n_contexts//2] *= 1.6
        self.attention_weight = 1.1
        
    def _init_constraint_kernel(self):
        """Initialize constraint-specific kernel parameters"""
        # Constraint nodes have focused, sparse features
        sparse_mask = np.random.binomial(1, 0.3, self.tensor_data.shape)
        self.tensor_data *= sparse_mask
        self.attention_weight = 0.9
    
    def _init_generic_kernel(self):
        """Initialize generic kernel parameters"""
        pass  # Use default initialization
    
    def forward(self, input_features: np.ndarray, context_state: np.ndarray) -> np.ndarray:
        """
        Forward pass: compute node activation
        
        Args:
            input_features: Input feature vector [n_features]
            context_state: Context state vector [n_contexts]
            
        Returns:
            Output activation [n_features, n_contexts]
        """
        # Ensure input dimensions match
        if input_features.shape[0] != self.n_features:
            input_features = self._resize_vector(input_features, self.n_features)
        if context_state.shape[0] != self.n_contexts:
            context_state = self._resize_vector(context_state, self.n_contexts)
        
        # Outer product to create interaction tensor
        interaction = np.outer(input_features, context_state)
        
        # Apply kernel transformation
        output = self.tensor_data * interaction
        
        # Apply attention weighting
        output *= self.attention_weight
        
        # Apply activation function (tanh for bounded output)
        output = np.tanh(output)
        
        # Store activation for history
        current_activation = np.mean(output)
        self.activation_history.append(current_activation)
        
        # Keep only recent history
        if len(self.activation_history) > 100:
            self.activation_history = self.activation_history[-100:]
        
        return output
    
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Backward pass for gradient computation"""
        # Simple gradient computation for tanh activation
        # grad_tanh = 1 - tanh^2(x)
        tanh_grad = 1 - np.tanh(self.tensor_data) ** 2
        grad_input = grad_output * tanh_grad * self.attention_weight
        
        return grad_input
    
    def update_attention(self, attention_signal: float, learning_rate: float = 0.01):
        """Update attention weight based on external signal"""
        self.attention_weight += learning_rate * attention_signal
        self.attention_weight = np.clip(self.attention_weight, 0.1, 2.0)
    
    def _resize_vector(self, vector: np.ndarray, target_size: int) -> np.ndarray:
        """Resize vector to target size"""
        if vector.size == target_size:
            return vector
        elif vector.size < target_size:
            # Pad with zeros
            padded = np.zeros(target_size)
            padded[:vector.size] = vector.flatten()
            return padded
        else:
            # Truncate
            return vector.flatten()[:target_size]
    
    def get_activation_statistics(self) -> Dict[str, float]:
        """Get statistics about node activation"""
        if not self.activation_history:
            return {'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0}
        
        history = np.array(self.activation_history)
        return {
            'mean': np.mean(history),
            'std': np.std(history),
            'min': np.min(history),
            'max': np.max(history),
            'trend': np.polyfit(range(len(history)), history, 1)[0] if len(history) > 1 else 0.0
        }


@dataclass
class LinkTensorKernel(TensorKernel):
    """
    Tensor kernel for hypergraph links
    Shape: [n_links, n_relation_types, n_weights] representing link dynamics
    """
    
    def __init__(self, n_links: int, n_relation_types: int, n_weights: int):
        super().__init__((n_links, n_relation_types, n_weights))
        self.n_links = n_links
        self.n_relation_types = n_relation_types
        self.n_weights = n_weights
        self.propagation_history = []
        
        # Initialize link tensor with identity-like structure
        self.tensor_data = np.zeros((n_links, n_relation_types, n_weights), dtype=self.dtype)
        
        # Initialize each link-relation combination
        for i in range(n_links):
            for j in range(n_relation_types):
                # Diagonal emphasis for direct relationships
                self.tensor_data[i, j, :] = np.eye(n_weights)[j % n_weights] if j < n_weights else np.ones(n_weights) * 0.1
    
    def forward(self, source_activations: np.ndarray, relation_mask: np.ndarray) -> np.ndarray:
        """
        Forward pass: propagate activations through links
        
        Args:
            source_activations: Source node activations [n_links]
            relation_mask: Active relation types [n_relation_types]
            
        Returns:
            Propagated activations [n_links, n_relation_types, n_weights]
        """
        # Ensure input dimensions
        if source_activations.shape[0] != self.n_links:
            source_activations = self._resize_activations(source_activations, self.n_links)
        if relation_mask.shape[0] != self.n_relation_types:
            relation_mask = self._resize_activations(relation_mask, self.n_relation_types)
        
        # Expand dimensions for broadcasting
        source_expanded = source_activations[:, np.newaxis, np.newaxis]  # [n_links, 1, 1]
        relation_expanded = relation_mask[np.newaxis, :, np.newaxis]     # [1, n_relation_types, 1]
        
        # Apply link transformations
        output = self.tensor_data * source_expanded * relation_expanded
        
        # Apply sigmoid activation for bounded propagation
        output = 1.0 / (1.0 + np.exp(-output))
        
        # Store propagation statistics
        propagation_strength = np.mean(output)
        self.propagation_history.append(propagation_strength)
        
        # Keep recent history
        if len(self.propagation_history) > 100:
            self.propagation_history = self.propagation_history[-100:]
        
        return output
    
    def backward(self, grad_output: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Backward pass for link gradients"""
        # Sigmoid gradient: sigmoid(x) * (1 - sigmoid(x))
        sigmoid_output = 1.0 / (1.0 + np.exp(-self.tensor_data))
        sigmoid_grad = sigmoid_output * (1 - sigmoid_output)
        
        grad_tensor = grad_output * sigmoid_grad
        
        # Compute gradients w.r.t. inputs
        grad_source = np.sum(grad_tensor, axis=(1, 2))
        grad_relation = np.sum(grad_tensor, axis=(0, 2))
        
        return grad_source, grad_relation
    
    def update_link_weights(self, link_idx: int, relation_idx: int, weight_update: np.ndarray):
        """Update specific link weights"""
        if 0 <= link_idx < self.n_links and 0 <= relation_idx < self.n_relation_types:
            self.tensor_data[link_idx, relation_idx, :] += weight_update[:self.n_weights]
            
            # Normalize to prevent explosion
            self.tensor_data[link_idx, relation_idx, :] = np.clip(
                self.tensor_data[link_idx, relation_idx, :], -2.0, 2.0
            )
    
    def _resize_activations(self, activations: np.ndarray, target_size: int) -> np.ndarray:
        """Resize activation vector to target size"""
        if activations.size == target_size:
            return activations.flatten()
        elif activations.size < target_size:
            padded = np.zeros(target_size)
            padded[:activations.size] = activations.flatten()
            return padded
        else:
            return activations.flatten()[:target_size]
    
    def get_propagation_statistics(self) -> Dict[str, float]:
        """Get link propagation statistics"""
        if not self.propagation_history:
            return {'mean': 0.0, 'std': 0.0, 'efficiency': 0.0}
        
        history = np.array(self.propagation_history)
        return {
            'mean': np.mean(history),
            'std': np.std(history),
            'efficiency': np.mean(history) / (np.std(history) + 1e-8),
            'trend': np.polyfit(range(len(history)), history, 1)[0] if len(history) > 1 else 0.0
        }


@dataclass
class AttentionTensorKernel(TensorKernel):
    """
    Tensor kernel for attention allocation
    Shape: [n_agents, n_tasks, n_resources] representing attention distribution
    """
    
    def __init__(self, n_agents: int, n_tasks: int, n_resources: int):
        super().__init__((n_agents, n_tasks, n_resources))
        self.n_agents = n_agents
        self.n_tasks = n_tasks
        self.n_resources = n_resources
        self.attention_history = []
        self.temperature = 1.0  # For softmax temperature scaling
        
        # Initialize with uniform attention distribution
        self.tensor_data = np.ones((n_agents, n_tasks, n_resources), dtype=self.dtype)
        self.tensor_data /= np.sum(self.tensor_data, axis=(1, 2), keepdims=True)
    
    def forward(self, agent_states: np.ndarray, task_demands: np.ndarray, resource_availability: np.ndarray) -> np.ndarray:
        """
        Forward pass: compute attention allocation
        
        Args:
            agent_states: Agent activation states [n_agents]
            task_demands: Task urgency/importance [n_tasks] 
            resource_availability: Resource availability [n_resources]
            
        Returns:
            Attention allocation tensor [n_agents, n_tasks, n_resources]
        """
        # Ensure input dimensions
        if agent_states.shape[0] != self.n_agents:
            agent_states = self._resize_to_dim(agent_states, self.n_agents)
        if task_demands.shape[0] != self.n_tasks:
            task_demands = self._resize_to_dim(task_demands, self.n_tasks)
        if resource_availability.shape[0] != self.n_resources:
            resource_availability = self._resize_to_dim(resource_availability, self.n_resources)
        
        # Create 3D interaction tensor
        interaction = np.zeros((self.n_agents, self.n_tasks, self.n_resources))
        
        for i in range(self.n_agents):
            for j in range(self.n_tasks):
                for k in range(self.n_resources):
                    # Compute attention based on agent-task-resource compatibility
                    interaction[i, j, k] = agent_states[i] * task_demands[j] * resource_availability[k]
        
        # Apply current attention weights
        weighted_interaction = self.tensor_data * interaction
        
        # Apply temperature-scaled softmax for attention distribution
        scaled_logits = weighted_interaction / self.temperature
        
        # Softmax across all dimensions
        exp_logits = np.exp(scaled_logits - np.max(scaled_logits))  # Numerical stability
        attention_output = exp_logits / np.sum(exp_logits)
        
        # Store attention statistics
        attention_entropy = self._compute_entropy(attention_output)
        self.attention_history.append(attention_entropy)
        
        # Keep recent history
        if len(self.attention_history) > 100:
            self.attention_history = self.attention_history[-100:]
        
        return attention_output
    
    def backward(self, grad_output: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Backward pass for attention gradients"""
        # Softmax gradient computation
        softmax_output = self.forward(
            np.ones(self.n_agents), 
            np.ones(self.n_tasks), 
            np.ones(self.n_resources)
        )
        
        # Simplified gradient computation
        grad_agents = np.sum(grad_output, axis=(1, 2))
        grad_tasks = np.sum(grad_output, axis=(0, 2))
        grad_resources = np.sum(grad_output, axis=(0, 1))
        
        return grad_agents, grad_tasks, grad_resources
    
    def update_attention_weights(self, agent_idx: int, task_idx: int, resource_idx: int, 
                               weight_delta: float, learning_rate: float = 0.01):
        """Update specific attention weight"""
        if (0 <= agent_idx < self.n_agents and 
            0 <= task_idx < self.n_tasks and 
            0 <= resource_idx < self.n_resources):
            
            self.tensor_data[agent_idx, task_idx, resource_idx] += learning_rate * weight_delta
            
            # Renormalize to maintain probability distribution
            self._renormalize_attention()
    
    def _renormalize_attention(self):
        """Renormalize attention tensor to maintain valid probability distribution"""
        # Ensure positive values
        self.tensor_data = np.maximum(self.tensor_data, 1e-8)
        
        # Normalize so each agent's attention sums to 1
        for i in range(self.n_agents):
            total = np.sum(self.tensor_data[i, :, :])
            if total > 0:
                self.tensor_data[i, :, :] /= total
    
    def _resize_to_dim(self, vector: np.ndarray, target_dim: int) -> np.ndarray:
        """Resize vector to target dimension"""
        flat = vector.flatten()
        if flat.size == target_dim:
            return flat
        elif flat.size < target_dim:
            result = np.zeros(target_dim)
            result[:flat.size] = flat
            return result
        else:
            return flat[:target_dim]
    
    def _compute_entropy(self, distribution: np.ndarray) -> float:
        """Compute entropy of attention distribution"""
        flat_dist = distribution.flatten()
        flat_dist = flat_dist[flat_dist > 1e-8]  # Avoid log(0)
        return -np.sum(flat_dist * np.log(flat_dist))
    
    def set_temperature(self, temperature: float):
        """Set softmax temperature for attention sharpness"""
        self.temperature = max(0.1, temperature)  # Minimum temperature to avoid division by zero
    
    def get_attention_focus(self) -> Dict[str, Any]:
        """Get information about attention focus patterns"""
        # Find most attended agent-task-resource combination
        max_idx = np.unravel_index(np.argmax(self.tensor_data), self.tensor_data.shape)
        
        # Compute concentration metrics
        max_attention = np.max(self.tensor_data)
        mean_attention = np.mean(self.tensor_data)
        attention_concentration = max_attention / mean_attention
        
        return {
            'max_attention_idx': max_idx,
            'max_attention_value': max_attention,
            'concentration_ratio': attention_concentration,
            'entropy_history': self.attention_history[-10:] if self.attention_history else [],
            'temperature': self.temperature
        }
    
    def get_agent_workload(self, agent_idx: int) -> Dict[str, float]:
        """Get workload distribution for a specific agent"""
        if 0 <= agent_idx < self.n_agents:
            agent_attention = self.tensor_data[agent_idx, :, :]
            return {
                'total_attention': np.sum(agent_attention),
                'task_distribution': np.sum(agent_attention, axis=1).tolist(),
                'resource_distribution': np.sum(agent_attention, axis=0).tolist(),
                'peak_attention': np.max(agent_attention)
            }
        return {}


class TensorNetworkValidator:
    """Validates tensor kernel configurations and operations"""
    
    @staticmethod
    def validate_node_kernel(kernel: NodeTensorKernel) -> Dict[str, bool]:
        """Validate node tensor kernel configuration"""
        checks = {
            'valid_shape': len(kernel.tensor_shape) == 2,
            'positive_dimensions': all(d > 0 for d in kernel.tensor_shape),
            'finite_values': np.all(np.isfinite(kernel.tensor_data)),
            'reasonable_size': kernel.tensor_data.size < 1e6,  # < 1M elements
            'valid_attention': 0.1 <= kernel.attention_weight <= 2.0
        }
        return checks
    
    @staticmethod
    def validate_link_kernel(kernel: LinkTensorKernel) -> Dict[str, bool]:
        """Validate link tensor kernel configuration"""
        checks = {
            'valid_shape': len(kernel.tensor_shape) == 3,
            'positive_dimensions': all(d > 0 for d in kernel.tensor_shape),
            'finite_values': np.all(np.isfinite(kernel.tensor_data)),
            'reasonable_size': kernel.tensor_data.size < 1e6,
            'bounded_weights': np.all(np.abs(kernel.tensor_data) <= 5.0)
        }
        return checks
    
    @staticmethod
    def validate_attention_kernel(kernel: AttentionTensorKernel) -> Dict[str, bool]:
        """Validate attention tensor kernel configuration"""
        checks = {
            'valid_shape': len(kernel.tensor_shape) == 3,
            'positive_dimensions': all(d > 0 for d in kernel.tensor_shape),
            'finite_values': np.all(np.isfinite(kernel.tensor_data)),
            'probability_distribution': np.allclose(np.sum(kernel.tensor_data), kernel.n_agents, rtol=1e-3),
            'positive_values': np.all(kernel.tensor_data >= 0),
            'valid_temperature': kernel.temperature > 0
        }
        return checks
    
    @staticmethod
    def validate_tensor_compatibility(kernels: List[TensorKernel]) -> Dict[str, bool]:
        """Validate compatibility between multiple tensor kernels"""
        if not kernels:
            return {'empty_list': False}
        
        # Check dimension compatibility
        node_kernels = [k for k in kernels if isinstance(k, NodeTensorKernel)]
        link_kernels = [k for k in kernels if isinstance(k, LinkTensorKernel)]
        attention_kernels = [k for k in kernels if isinstance(k, AttentionTensorKernel)]
        
        checks = {
            'consistent_dtypes': len(set(k.dtype for k in kernels)) <= 1,
            'reasonable_total_size': sum(k.tensor_data.size for k in kernels) < 1e7,
            'node_feature_consistency': len(set(k.n_features for k in node_kernels)) <= 1 if node_kernels else True,
            'attention_agent_consistency': True  # Additional checks can be added
        }
        
        return checks