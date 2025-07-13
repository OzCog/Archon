"""
Hypergraph Data Structures

Implements hypergraph nodes and links for cognitive grammar representation
with tensor-compatible structures.
"""

try:
    import numpy as np
except ImportError:
    # Use mock numpy for testing
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'tests'))
    import mock_numpy as np
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
import uuid


class NodeType(Enum):
    """Types of hypergraph nodes"""
    AGENT_NODE = "agent"
    TASK_NODE = "task"
    RESOURCE_NODE = "resource" 
    WORKFLOW_NODE = "workflow"
    KNOWLEDGE_NODE = "knowledge"
    CONSTRAINT_NODE = "constraint"


class LinkType(Enum):
    """Types of hypergraph links"""
    PERFORMS = "performs"
    USES = "uses"
    ORCHESTRATES = "orchestrates"
    RELATES_TO = "relates_to"
    DEPENDS_ON = "depends_on"
    PRODUCES = "produces"


@dataclass
class HypergraphNode:
    """
    A node in the cognitive hypergraph with tensor-compatible structure
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    node_type: NodeType = NodeType.AGENT_NODE
    name: str = ""
    semantic_depth: int = 1
    features: np.ndarray = field(default_factory=lambda: np.zeros(16))  # Feature vector
    contexts: np.ndarray = field(default_factory=lambda: np.zeros(8))   # Context vector
    activation: float = 0.0
    attention_weight: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize tensor dimensions based on semantic depth and type"""
        if isinstance(self.features, list):
            self.features = np.array(self.features)
        if isinstance(self.contexts, list):
            self.contexts = np.array(self.contexts)
            
        # Adjust tensor dimensions based on semantic depth
        feature_dim = max(16, self.semantic_depth * 4)
        context_dim = max(8, self.semantic_depth * 2)
        
        if self.features.size != feature_dim:
            self.features = np.zeros(feature_dim)
        if self.contexts.size != context_dim:
            self.contexts = np.zeros(context_dim)
    
    def get_tensor_shape(self) -> Tuple[int, int]:
        """Get tensor shape as [n_features, n_contexts]"""
        return (len(self.features), len(self.contexts))
    
    def update_activation(self, input_signal: float, decay_rate: float = 0.1):
        """Update node activation with temporal decay"""
        self.activation = self.activation * (1 - decay_rate) + input_signal
        self.activation = max(0.0, min(1.0, self.activation))  # Clamp to [0,1]
    
    def compute_feature_similarity(self, other: 'HypergraphNode') -> float:
        """Compute cosine similarity between feature vectors"""
        if self.features.size != other.features.size:
            return 0.0
        
        norm_self = np.linalg.norm(self.features)
        norm_other = np.linalg.norm(other.features)
        
        if norm_self == 0 or norm_other == 0:
            return 0.0
            
        return np.dot(self.features, other.features) / (norm_self * norm_other)


@dataclass  
class HypergraphLink:
    """
    A link in the cognitive hypergraph connecting multiple nodes
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    link_type: LinkType = LinkType.RELATES_TO
    source_nodes: Set[str] = field(default_factory=set)
    target_nodes: Set[str] = field(default_factory=set)
    weight: float = 1.0
    relation_tensor: np.ndarray = field(default_factory=lambda: np.ones((4, 4)))
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize link tensor dimensions"""
        if isinstance(self.relation_tensor, list):
            self.relation_tensor = np.array(self.relation_tensor)
        
        # Ensure relation tensor has proper dimensions
        n_sources = len(self.source_nodes) or 1
        n_targets = len(self.target_nodes) or 1
        expected_shape = (max(n_sources, 4), max(n_targets, 4))
        
        if self.relation_tensor.shape != expected_shape:
            self.relation_tensor = np.ones(expected_shape) * self.weight
    
    def get_tensor_shape(self) -> Tuple[int, int, int]:
        """Get tensor shape as [n_links, n_relation_types, n_weights]"""
        return (1, len(LinkType), self.relation_tensor.size)
    
    def add_source_node(self, node_id: str):
        """Add a source node to the link"""
        self.source_nodes.add(node_id)
        self._update_tensor_dimensions()
    
    def add_target_node(self, node_id: str):
        """Add a target node to the link"""
        self.target_nodes.add(node_id)
        self._update_tensor_dimensions()
    
    def _update_tensor_dimensions(self):
        """Update relation tensor dimensions when nodes are added"""
        n_sources = len(self.source_nodes)
        n_targets = len(self.target_nodes)
        
        if n_sources > 0 and n_targets > 0:
            new_shape = (max(n_sources, 4), max(n_targets, 4))
            if self.relation_tensor.shape != new_shape:
                old_tensor = self.relation_tensor
                self.relation_tensor = np.ones(new_shape) * self.weight
                
                # Copy old values if possible
                min_rows = min(old_tensor.shape[0], new_shape[0])
                min_cols = min(old_tensor.shape[1], new_shape[1])
                self.relation_tensor[:min_rows, :min_cols] = old_tensor[:min_rows, :min_cols]
    
    def compute_link_strength(self, source_activations: Dict[str, float], 
                            target_activations: Dict[str, float]) -> float:
        """Compute link strength based on connected node activations"""
        if not self.source_nodes or not self.target_nodes:
            return 0.0
        
        source_sum = sum(source_activations.get(node_id, 0.0) for node_id in self.source_nodes)
        target_sum = sum(target_activations.get(node_id, 0.0) for node_id in self.target_nodes)
        
        avg_source = source_sum / len(self.source_nodes)
        avg_target = target_sum / len(self.target_nodes)
        
        return self.weight * avg_source * avg_target


class HypergraphEncoder:
    """
    Encodes cognitive grammar elements into hypergraph structures
    """
    
    def __init__(self):
        self.nodes: Dict[str, HypergraphNode] = {}
        self.links: Dict[str, HypergraphLink] = {}
        self.node_index: Dict[str, str] = {}  # name -> id mapping
        
    def encode_scheme_pattern(self, scheme_graph: Dict[str, Any]) -> Tuple[Dict[str, HypergraphNode], Dict[str, HypergraphLink]]:
        """
        Encode Scheme-style hypergraph pattern into node/link structures
        
        Args:
            scheme_graph: Scheme hypergraph representation from parser
            
        Returns:
            Tuple of (nodes_dict, links_dict)
        """
        # Clear existing structures
        self.nodes.clear()
        self.links.clear()
        self.node_index.clear()
        
        # Create nodes
        for node_data in scheme_graph.get('nodes', []):
            node = self._create_node_from_data(node_data)
            self.nodes[node.id] = node
            self.node_index[node.name] = node.id
        
        # Create links
        for link_data in scheme_graph.get('links', []):
            link = self._create_link_from_data(link_data)
            if link:
                self.links[link.id] = link
        
        # Initialize feature vectors based on graph structure
        self._initialize_node_features()
        
        return self.nodes.copy(), self.links.copy()
    
    def _create_node_from_data(self, node_data: Dict[str, Any]) -> HypergraphNode:
        """Create hypergraph node from parsed data"""
        node_type_map = {
            'agent': NodeType.AGENT_NODE,
            'task': NodeType.TASK_NODE,
            'resource': NodeType.RESOURCE_NODE,
            'workflow': NodeType.WORKFLOW_NODE,
            'knowledge': NodeType.KNOWLEDGE_NODE,
            'constraint': NodeType.CONSTRAINT_NODE
        }
        
        node_type = node_type_map.get(node_data.get('type', 'agent'), NodeType.AGENT_NODE)
        semantic_depth = node_data.get('semantic_depth', 1)
        
        node = HypergraphNode(
            id=node_data.get('id', str(uuid.uuid4())),
            node_type=node_type,
            name=node_data.get('name', ''),
            semantic_depth=semantic_depth,
            metadata=node_data.get('attributes', {})
        )
        
        return node
    
    def _create_link_from_data(self, link_data: Dict[str, Any]) -> Optional[HypergraphLink]:
        """Create hypergraph link from parsed data"""
        link_type_map = {
            'performs': LinkType.PERFORMS,
            'uses': LinkType.USES,
            'orchestrates': LinkType.ORCHESTRATES,
            'relates_to': LinkType.RELATES_TO,
            'depends_on': LinkType.DEPENDS_ON,
            'produces': LinkType.PRODUCES
        }
        
        source_id = link_data.get('source')
        target_name = link_data.get('target_name')
        
        if not source_id or not target_name:
            return None
        
        # Find target node ID
        target_id = self.node_index.get(target_name)
        if not target_id:
            return None
        
        link_type = link_type_map.get(
            link_data.get('relationship_type', 'relates_to'), 
            LinkType.RELATES_TO
        )
        
        link = HypergraphLink(
            id=link_data.get('id', str(uuid.uuid4())),
            link_type=link_type,
            weight=link_data.get('weight', 1.0),
            metadata=link_data
        )
        
        link.add_source_node(source_id)
        link.add_target_node(target_id)
        
        return link
    
    def _initialize_node_features(self):
        """Initialize node feature vectors based on graph topology"""
        for node in self.nodes.values():
            # Initialize features based on node type and connections
            type_features = self._get_type_features(node.node_type)
            connection_features = self._get_connection_features(node.id)
            
            # Combine type and connection features
            feature_dim = node.features.size
            combined_features = np.zeros(feature_dim)
            
            # Fill with type features
            type_len = min(len(type_features), feature_dim // 2)
            combined_features[:type_len] = type_features[:type_len]
            
            # Fill with connection features  
            conn_len = min(len(connection_features), feature_dim - type_len)
            combined_features[type_len:type_len + conn_len] = connection_features[:conn_len]
            
            node.features = combined_features
            
            # Initialize context features based on semantic depth
            context_features = np.random.normal(0, 0.1, node.contexts.size)
            context_features *= node.semantic_depth / 10.0  # Scale by semantic depth
            node.contexts = context_features
    
    def _get_type_features(self, node_type: NodeType) -> np.ndarray:
        """Get feature vector encoding for node type"""
        type_vectors = {
            NodeType.AGENT_NODE: [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8, 0.9],
            NodeType.TASK_NODE: [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.7, 0.6],
            NodeType.RESOURCE_NODE: [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.5, 0.4],
            NodeType.WORKFLOW_NODE: [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.9, 0.8],
            NodeType.KNOWLEDGE_NODE: [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.6, 0.7],
            NodeType.CONSTRAINT_NODE: [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.4, 0.5]
        }
        return np.array(type_vectors.get(node_type, [0.0] * 8))
    
    def _get_connection_features(self, node_id: str) -> np.ndarray:
        """Get feature vector based on node connections"""
        in_degree = 0
        out_degree = 0
        
        for link in self.links.values():
            if node_id in link.target_nodes:
                in_degree += 1
            if node_id in link.source_nodes:
                out_degree += 1
        
        # Normalize degrees
        max_degree = max(1, len(self.nodes))
        norm_in = in_degree / max_degree
        norm_out = out_degree / max_degree
        
        # Create connection feature vector
        connection_features = [
            norm_in,      # Normalized in-degree
            norm_out,     # Normalized out-degree
            norm_in + norm_out,  # Total connectivity
            abs(norm_in - norm_out),  # Degree imbalance
            min(norm_in, norm_out),   # Min degree
            max(norm_in, norm_out),   # Max degree
            norm_in * norm_out,       # Degree product
            (norm_in + norm_out) / 2  # Average degree
        ]
        
        return np.array(connection_features)
    
    def get_adjacency_matrix(self) -> np.ndarray:
        """Get adjacency matrix representation of the hypergraph"""
        if not self.nodes:
            return np.array([])
        
        node_ids = list(self.nodes.keys())
        n_nodes = len(node_ids)
        adjacency = np.zeros((n_nodes, n_nodes))
        
        id_to_index = {node_id: i for i, node_id in enumerate(node_ids)}
        
        for link in self.links.values():
            for source_id in link.source_nodes:
                for target_id in link.target_nodes:
                    if source_id in id_to_index and target_id in id_to_index:
                        i = id_to_index[source_id]
                        j = id_to_index[target_id]
                        adjacency[i, j] = link.weight
        
        return adjacency
    
    def compute_graph_statistics(self) -> Dict[str, float]:
        """Compute statistical properties of the hypergraph"""
        n_nodes = len(self.nodes)
        n_links = len(self.links)
        
        if n_nodes == 0:
            return {'nodes': 0, 'links': 0, 'density': 0, 'avg_degree': 0}
        
        # Compute density
        max_possible_links = n_nodes * (n_nodes - 1)
        density = n_links / max_possible_links if max_possible_links > 0 else 0
        
        # Compute average degree
        total_degree = sum(len(link.source_nodes) + len(link.target_nodes) 
                          for link in self.links.values())
        avg_degree = total_degree / n_nodes if n_nodes > 0 else 0
        
        return {
            'nodes': n_nodes,
            'links': n_links, 
            'density': density,
            'avg_degree': avg_degree,
            'complexity': n_nodes * avg_degree
        }