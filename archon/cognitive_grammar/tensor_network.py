"""
Distributed Tensor Network Orchestrator

Orchestrates cognitive workflows across distributed tensor kernels with
message-passing, consensus mechanisms, and attention-driven activation.
"""

try:
    import numpy as np
except ImportError:
    # Use mock numpy for testing
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'tests'))
    import mock_numpy as np
from typing import Dict, List, Any, Optional, Tuple, Set, Callable
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import time
import json
import uuid
from concurrent.futures import ThreadPoolExecutor

from .tensor_kernels import NodeTensorKernel, LinkTensorKernel, AttentionTensorKernel, TensorKernel
from .attention_engine import ECANAttentionEngine, AttentionUnit
from .hypergraph import HypergraphNode, HypergraphLink, HypergraphEncoder


class NetworkState(Enum):
    """States of the distributed tensor network"""
    INITIALIZING = "initializing"
    ACTIVE = "active"
    CONVERGING = "converging"
    STABLE = "stable"
    ERROR = "error"


class MessageType(Enum):
    """Types of messages in the network"""
    ACTIVATION = "activation"
    GRADIENT = "gradient"
    ATTENTION = "attention"
    CONSENSUS = "consensus"
    CONTROL = "control"


@dataclass
class NetworkMessage:
    """Message passed between tensor kernels"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    message_type: MessageType = MessageType.ACTIVATION
    source_id: str = ""
    target_id: str = ""
    payload: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    priority: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary for serialization"""
        return {
            'id': self.id,
            'type': self.message_type.value,
            'source': self.source_id,
            'target': self.target_id,
            'payload': self.payload,
            'timestamp': self.timestamp,
            'priority': self.priority
        }


@dataclass
class KernelNode:
    """Wrapper for tensor kernels in the distributed network"""
    id: str
    kernel: TensorKernel
    kernel_type: str  # 'node', 'link', 'attention'
    current_state: np.ndarray
    message_queue: List[NetworkMessage] = field(default_factory=list)
    connections: Set[str] = field(default_factory=set)
    last_update: float = field(default_factory=time.time)
    activation_count: int = 0
    performance_metrics: Dict[str, float] = field(default_factory=dict)


class DistributedTensorNetwork:
    """
    Orchestrates distributed tensor network for cognitive workflows.
    Manages activation spreading, attention allocation, and consensus building.
    """
    
    def __init__(self, attention_engine: Optional[ECANAttentionEngine] = None,
                 max_workers: int = 4):
        self.attention_engine = attention_engine or ECANAttentionEngine()
        self.kernel_nodes: Dict[str, KernelNode] = {}
        self.network_topology: Dict[str, Set[str]] = {}
        self.message_bus: List[NetworkMessage] = []
        self.state = NetworkState.INITIALIZING
        
        # Network parameters
        self.convergence_threshold = 1e-4
        self.max_iterations = 1000
        self.activation_decay = 0.95
        self.message_timeout = 30.0  # seconds
        
        # Execution control
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.is_running = False
        self.iteration_count = 0
        
        # Performance tracking
        self.performance_history: List[Dict[str, Any]] = []
        self.convergence_history: List[float] = []
        
        # Consensus mechanisms
        self.consensus_threshold = 0.8
        self.voting_nodes: Set[str] = set()
        
    def register_kernel(self, kernel_id: str, kernel: TensorKernel, 
                       kernel_type: str) -> KernelNode:
        """Register a tensor kernel in the distributed network"""
        kernel_node = KernelNode(
            id=kernel_id,
            kernel=kernel,
            kernel_type=kernel_type,
            current_state=np.zeros(kernel.tensor_shape)
        )
        
        self.kernel_nodes[kernel_id] = kernel_node
        self.network_topology[kernel_id] = set()
        
        # Register with attention engine
        complexity = np.prod(kernel.tensor_shape) / 1000.0  # Rough complexity measure
        self.attention_engine.register_attention_unit(
            kernel_id, 
            initial_salience=0.5,
            complexity=complexity
        )
        
        return kernel_node
    
    def connect_kernels(self, source_id: str, target_id: str, bidirectional: bool = True):
        """Create connection between tensor kernels"""
        if source_id in self.network_topology and target_id in self.kernel_nodes:
            self.network_topology[source_id].add(target_id)
            self.kernel_nodes[source_id].connections.add(target_id)
            
            if bidirectional and target_id in self.network_topology:
                self.network_topology[target_id].add(source_id)
                self.kernel_nodes[target_id].connections.add(source_id)
    
    def build_from_hypergraph(self, nodes: Dict[str, HypergraphNode], 
                            links: Dict[str, HypergraphLink]) -> Dict[str, str]:
        """
        Build tensor network from hypergraph representation
        
        Args:
            nodes: Hypergraph nodes
            links: Hypergraph links
            
        Returns:
            Mapping from hypergraph element IDs to kernel IDs
        """
        element_to_kernel = {}
        
        # Create node kernels
        for node_id, node in nodes.items():
            n_features, n_contexts = node.get_tensor_shape()
            node_kernel = NodeTensorKernel(n_features, n_contexts, node.node_type.value)
            
            kernel_id = f"node_{node_id}"
            self.register_kernel(kernel_id, node_kernel, "node")
            element_to_kernel[node_id] = kernel_id
            
            # Initialize kernel with node features
            if hasattr(node, 'features') and hasattr(node, 'contexts'):
                node_kernel.tensor_data[:len(node.features), :len(node.contexts)] = np.outer(
                    node.features[:n_features], node.contexts[:n_contexts]
                )
        
        # Create link kernels
        for link_id, link in links.items():
            n_links = 1
            n_relation_types = 6  # Number of LinkType enum values
            n_weights = max(len(link.source_nodes) + len(link.target_nodes), 4)
            
            link_kernel = LinkTensorKernel(n_links, n_relation_types, n_weights)
            
            kernel_id = f"link_{link_id}"
            self.register_kernel(kernel_id, link_kernel, "link")
            element_to_kernel[link_id] = kernel_id
        
        # Create attention kernel
        n_agents = len([k for k in self.kernel_nodes.values() if k.kernel_type == "node"])
        n_tasks = len([k for k in self.kernel_nodes.values() if k.kernel_type == "link"])
        n_resources = max(n_agents + n_tasks, 8)
        
        attention_kernel = AttentionTensorKernel(n_agents, n_tasks, n_resources)
        self.register_kernel("attention_global", attention_kernel, "attention")
        
        # Connect kernels based on hypergraph links
        for link_id, link in links.items():
            link_kernel_id = element_to_kernel[link_id]
            
            # Connect link to its source and target nodes
            for source_node_id in link.source_nodes:
                if source_node_id in element_to_kernel:
                    source_kernel_id = element_to_kernel[source_node_id]
                    self.connect_kernels(source_kernel_id, link_kernel_id)
            
            for target_node_id in link.target_nodes:
                if target_node_id in element_to_kernel:
                    target_kernel_id = element_to_kernel[target_node_id]
                    self.connect_kernels(link_kernel_id, target_kernel_id)
        
        return element_to_kernel
    
    async def initialize_network(self):
        """Initialize the distributed tensor network"""
        self.state = NetworkState.INITIALIZING
        
        # Initialize all kernel states
        for kernel_node in self.kernel_nodes.values():
            if kernel_node.kernel_type == "node":
                # Initialize with random activation
                kernel_node.current_state = np.random.normal(0, 0.1, kernel_node.kernel.tensor_shape)
            elif kernel_node.kernel_type == "link":
                # Initialize with identity-like structure
                kernel_node.current_state = np.ones(kernel_node.kernel.tensor_shape) * 0.1
            elif kernel_node.kernel_type == "attention":
                # Initialize with uniform attention
                kernel_node.current_state = np.ones(kernel_node.kernel.tensor_shape)
                kernel_node.current_state /= np.sum(kernel_node.current_state)
        
        # Clear message bus
        self.message_bus.clear()
        
        # Reset iteration count
        self.iteration_count = 0
        
        self.state = NetworkState.ACTIVE
    
    async def activate_workflow(self, initial_activation: Dict[str, np.ndarray],
                              max_iterations: Optional[int] = None) -> Dict[str, np.ndarray]:
        """
        Activate cognitive workflow across the tensor network
        
        Args:
            initial_activation: Initial activation for specific kernels
            max_iterations: Maximum iterations (uses default if None)
            
        Returns:
            Final activation states for all kernels
        """
        if self.state == NetworkState.INITIALIZING:
            await self.initialize_network()
        
        self.is_running = True
        max_iter = max_iterations or self.max_iterations
        
        # Apply initial activations
        for kernel_id, activation in initial_activation.items():
            if kernel_id in self.kernel_nodes:
                self.kernel_nodes[kernel_id].current_state = activation
                
                # Send activation messages to connected kernels
                await self._broadcast_activation(kernel_id, activation)
        
        # Main iteration loop
        for iteration in range(max_iter):
            self.iteration_count = iteration
            
            if not self.is_running:
                break
            
            # Process messages
            await self._process_message_bus()
            
            # Update attention allocation
            attention_allocation = self.attention_engine.allocate_attention()
            
            # Update kernel states
            convergence_delta = await self._update_kernel_states(attention_allocation)
            
            # Check convergence
            self.convergence_history.append(convergence_delta)
            if convergence_delta < self.convergence_threshold:
                self.state = NetworkState.CONVERGING
                if len(self.convergence_history) >= 10:
                    recent_deltas = self.convergence_history[-10:]
                    if all(delta < self.convergence_threshold for delta in recent_deltas):
                        self.state = NetworkState.STABLE
                        break
            
            # Track performance
            await self._track_iteration_performance(iteration, convergence_delta)
            
            # Small delay to prevent overwhelming
            await asyncio.sleep(0.001)
        
        self.is_running = False
        
        # Return final states
        return {kernel_id: node.current_state 
                for kernel_id, node in self.kernel_nodes.items()}
    
    async def _broadcast_activation(self, source_id: str, activation: np.ndarray):
        """Broadcast activation to connected kernels"""
        if source_id not in self.kernel_nodes:
            return
        
        connections = self.kernel_nodes[source_id].connections
        
        for target_id in connections:
            message = NetworkMessage(
                message_type=MessageType.ACTIVATION,
                source_id=source_id,
                target_id=target_id,
                payload={
                    'activation': activation.tolist(),
                    'source_type': self.kernel_nodes[source_id].kernel_type
                }
            )
            
            # Add to message bus and target queue
            self.message_bus.append(message)
            if target_id in self.kernel_nodes:
                self.kernel_nodes[target_id].message_queue.append(message)
    
    async def _process_message_bus(self):
        """Process all messages in the message bus"""
        current_time = time.time()
        
        # Remove expired messages
        self.message_bus = [msg for msg in self.message_bus 
                          if current_time - msg.timestamp < self.message_timeout]
        
        # Process messages for each kernel
        tasks = []
        for kernel_id, kernel_node in self.kernel_nodes.items():
            if kernel_node.message_queue:
                task = self._process_kernel_messages(kernel_id)
                tasks.append(task)
        
        if tasks:
            await asyncio.gather(*tasks)
    
    async def _process_kernel_messages(self, kernel_id: str):
        """Process messages for a specific kernel"""
        kernel_node = self.kernel_nodes[kernel_id]
        processed_messages = []
        
        for message in kernel_node.message_queue:
            try:
                if message.message_type == MessageType.ACTIVATION:
                    await self._handle_activation_message(kernel_id, message)
                elif message.message_type == MessageType.GRADIENT:
                    await self._handle_gradient_message(kernel_id, message)
                elif message.message_type == MessageType.ATTENTION:
                    await self._handle_attention_message(kernel_id, message)
                elif message.message_type == MessageType.CONSENSUS:
                    await self._handle_consensus_message(kernel_id, message)
                
                processed_messages.append(message)
                
            except Exception as e:
                print(f"Error processing message {message.id} for kernel {kernel_id}: {e}")
        
        # Remove processed messages
        kernel_node.message_queue = [msg for msg in kernel_node.message_queue 
                                   if msg not in processed_messages]
    
    async def _handle_activation_message(self, kernel_id: str, message: NetworkMessage):
        """Handle activation message"""
        kernel_node = self.kernel_nodes[kernel_id]
        
        # Extract activation from message
        activation_data = message.payload.get('activation', [])
        if not activation_data:
            return
        
        activation = np.array(activation_data)
        
        # Update salience based on received activation
        activation_strength = np.mean(np.abs(activation))
        self.attention_engine.update_unit_salience(kernel_id, activation_strength * 0.1)
        
        # Process activation based on kernel type
        if kernel_node.kernel_type == "node":
            await self._process_node_activation(kernel_id, activation)
        elif kernel_node.kernel_type == "link":
            await self._process_link_activation(kernel_id, activation)
        elif kernel_node.kernel_type == "attention":
            await self._process_attention_activation(kernel_id, activation)
    
    async def _process_node_activation(self, kernel_id: str, activation: np.ndarray):
        """Process activation for node kernels"""
        kernel_node = self.kernel_nodes[kernel_id]
        node_kernel = kernel_node.kernel
        
        if isinstance(node_kernel, NodeTensorKernel):
            # Prepare input features and context
            current_shape = kernel_node.current_state.shape
            input_features = np.mean(activation.reshape(-1)[:current_shape[0]]) * np.ones(current_shape[0])
            context_state = np.ones(current_shape[1]) * 0.5
            
            # Forward pass
            new_state = node_kernel.forward(input_features, context_state)
            
            # Update current state with temporal integration
            alpha = 0.7  # Integration rate
            kernel_node.current_state = alpha * new_state + (1 - alpha) * kernel_node.current_state
            
            kernel_node.activation_count += 1
            kernel_node.last_update = time.time()
    
    async def _process_link_activation(self, kernel_id: str, activation: np.ndarray):
        """Process activation for link kernels"""
        kernel_node = self.kernel_nodes[kernel_id]
        link_kernel = kernel_node.kernel
        
        if isinstance(link_kernel, LinkTensorKernel):
            # Prepare source activations and relation mask
            n_links = link_kernel.n_links
            n_relations = link_kernel.n_relation_types
            
            source_activations = np.ones(n_links) * np.mean(activation)
            relation_mask = np.ones(n_relations)
            
            # Forward pass
            new_state = link_kernel.forward(source_activations, relation_mask)
            
            # Update current state
            alpha = 0.6
            kernel_node.current_state = alpha * new_state + (1 - alpha) * kernel_node.current_state
            
            kernel_node.activation_count += 1
            kernel_node.last_update = time.time()
    
    async def _process_attention_activation(self, kernel_id: str, activation: np.ndarray):
        """Process activation for attention kernels"""
        kernel_node = self.kernel_nodes[kernel_id]
        attention_kernel = kernel_node.kernel
        
        if isinstance(attention_kernel, AttentionTensorKernel):
            # Prepare agent, task, and resource states
            n_agents = attention_kernel.n_agents
            n_tasks = attention_kernel.n_tasks
            n_resources = attention_kernel.n_resources
            
            activation_mean = np.mean(activation)
            agent_states = np.ones(n_agents) * activation_mean
            task_demands = np.ones(n_tasks) * activation_mean
            resource_availability = np.ones(n_resources) * 0.8
            
            # Forward pass
            new_state = attention_kernel.forward(agent_states, task_demands, resource_availability)
            
            # Update current state
            alpha = 0.8
            kernel_node.current_state = alpha * new_state + (1 - alpha) * kernel_node.current_state
            
            kernel_node.activation_count += 1
            kernel_node.last_update = time.time()
    
    async def _handle_gradient_message(self, kernel_id: str, message: NetworkMessage):
        """Handle gradient message for learning"""
        # Placeholder for gradient-based learning
        pass
    
    async def _handle_attention_message(self, kernel_id: str, message: NetworkMessage):
        """Handle attention reallocation message"""
        attention_delta = message.payload.get('attention_delta', 0.0)
        self.attention_engine.update_unit_salience(kernel_id, attention_delta)
    
    async def _handle_consensus_message(self, kernel_id: str, message: NetworkMessage):
        """Handle consensus building message"""
        # Add to voting nodes
        self.voting_nodes.add(kernel_id)
        
        # Check if we have enough votes for consensus
        total_kernels = len(self.kernel_nodes)
        if len(self.voting_nodes) / total_kernels >= self.consensus_threshold:
            # Consensus reached - potentially trigger network-wide action
            await self._handle_consensus_reached()
    
    async def _handle_consensus_reached(self):
        """Handle when network consensus is reached"""
        # Clear voting nodes
        self.voting_nodes.clear()
        
        # Could trigger convergence check or state synchronization
        print(f"Network consensus reached at iteration {self.iteration_count}")
    
    async def _update_kernel_states(self, attention_allocation: Dict[str, float]) -> float:
        """Update all kernel states and return convergence delta"""
        total_delta = 0.0
        
        for kernel_id, kernel_node in self.kernel_nodes.items():
            old_state = kernel_node.current_state.copy()
            
            # Apply attention weighting
            attention_weight = attention_allocation.get(kernel_id, 1.0)
            kernel_node.current_state *= attention_weight
            
            # Apply activation decay
            kernel_node.current_state *= self.activation_decay
            
            # Calculate state change
            delta = np.linalg.norm(kernel_node.current_state - old_state)
            total_delta += delta
            
            # Update performance metrics
            kernel_node.performance_metrics['state_change'] = delta
            kernel_node.performance_metrics['attention_weight'] = attention_weight
            kernel_node.performance_metrics['activation_magnitude'] = np.linalg.norm(kernel_node.current_state)
        
        return total_delta / max(len(self.kernel_nodes), 1)
    
    async def _track_iteration_performance(self, iteration: int, convergence_delta: float):
        """Track performance metrics for current iteration"""
        performance = {
            'iteration': iteration,
            'convergence_delta': convergence_delta,
            'timestamp': time.time(),
            'active_kernels': len([n for n in self.kernel_nodes.values() if n.activation_count > 0]),
            'message_count': len(self.message_bus),
            'network_state': self.state.value,
            'attention_statistics': self.attention_engine.get_attention_statistics()
        }
        
        self.performance_history.append(performance)
        
        # Keep only recent history
        if len(self.performance_history) > 1000:
            self.performance_history = self.performance_history[-1000:]
    
    def get_network_statistics(self) -> Dict[str, Any]:
        """Get comprehensive network statistics"""
        kernel_stats = {}
        for kernel_id, kernel_node in self.kernel_nodes.items():
            kernel_stats[kernel_id] = {
                'type': kernel_node.kernel_type,
                'activation_count': kernel_node.activation_count,
                'connections': len(kernel_node.connections),
                'message_queue_size': len(kernel_node.message_queue),
                'last_update': kernel_node.last_update,
                'performance_metrics': kernel_node.performance_metrics
            }
        
        return {
            'network_state': self.state.value,
            'total_kernels': len(self.kernel_nodes),
            'total_connections': sum(len(connections) for connections in self.network_topology.values()),
            'iteration_count': self.iteration_count,
            'convergence_history': self.convergence_history[-10:],
            'message_bus_size': len(self.message_bus),
            'kernel_statistics': kernel_stats,
            'attention_statistics': self.attention_engine.get_attention_statistics()
        }
    
    def stop_network(self):
        """Stop the network execution"""
        self.is_running = False
        self.state = NetworkState.STABLE
    
    def export_network_state(self) -> Dict[str, Any]:
        """Export complete network state for persistence"""
        return {
            'kernel_states': {
                kernel_id: {
                    'current_state': node.current_state.tolist(),
                    'kernel_type': node.kernel_type,
                    'activation_count': node.activation_count,
                    'performance_metrics': node.performance_metrics
                }
                for kernel_id, node in self.kernel_nodes.items()
            },
            'network_topology': {
                kernel_id: list(connections) 
                for kernel_id, connections in self.network_topology.items()
            },
            'attention_configuration': self.attention_engine.export_configuration(),
            'network_parameters': {
                'convergence_threshold': self.convergence_threshold,
                'max_iterations': self.max_iterations,
                'activation_decay': self.activation_decay,
                'message_timeout': self.message_timeout,
                'consensus_threshold': self.consensus_threshold
            },
            'performance_history': self.performance_history[-100:],  # Last 100 entries
            'convergence_history': self.convergence_history[-100:]
        }