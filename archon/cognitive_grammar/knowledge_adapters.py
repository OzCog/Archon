"""
Knowledge Adapters for Cognitive Grammar

Adapters for integrating existing knowledge bases (Supabase, vector embeddings)
with the cognitive grammar tensor network.
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
import json
import asyncio
from abc import ABC, abstractmethod

# Type hints for external dependencies
try:
    from supabase import Client as SupabaseClient
    from openai import AsyncOpenAI
except ImportError:
    SupabaseClient = Any
    AsyncOpenAI = Any


@dataclass
class KnowledgeFragment:
    """Represents a fragment of knowledge from external sources"""
    id: str
    content: str
    vector_embedding: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)
    source: str = "unknown"
    confidence: float = 1.0
    timestamp: float = 0.0
    
    def to_tensor_compatible(self, target_dim: int = 512) -> np.ndarray:
        """Convert to tensor-compatible representation"""
        # Resize embedding to target dimension
        if self.vector_embedding.size == target_dim:
            return self.vector_embedding
        elif self.vector_embedding.size < target_dim:
            # Pad with zeros
            padded = np.zeros(target_dim)
            padded[:self.vector_embedding.size] = self.vector_embedding
            return padded
        else:
            # Truncate or downsample
            return self.vector_embedding[:target_dim]


class KnowledgeAdapter(ABC):
    """Abstract base class for knowledge adapters"""
    
    @abstractmethod
    async def retrieve_knowledge(self, query: str, limit: int = 10) -> List[KnowledgeFragment]:
        """Retrieve knowledge fragments relevant to query"""
        pass
    
    @abstractmethod
    async def store_knowledge(self, fragment: KnowledgeFragment) -> bool:
        """Store a knowledge fragment"""
        pass
    
    @abstractmethod
    def convert_to_tensor_input(self, fragments: List[KnowledgeFragment]) -> np.ndarray:
        """Convert knowledge fragments to tensor input"""
        pass


class VectorKnowledgeAdapter(KnowledgeAdapter):
    """
    Adapter for vector database knowledge integration (Supabase + OpenAI embeddings)
    """
    
    def __init__(self, supabase_client: Optional[SupabaseClient] = None,
                 embedding_client: Optional[AsyncOpenAI] = None,
                 embedding_model: str = "text-embedding-3-small",
                 table_name: str = "site_pages"):
        self.supabase = supabase_client
        self.embedding_client = embedding_client
        self.embedding_model = embedding_model
        self.table_name = table_name
        
        # Tensor configuration
        self.embedding_dim = 1536  # OpenAI embedding dimension
        self.max_fragments = 50    # Maximum fragments to process at once
        
        # Cache for performance
        self.embedding_cache: Dict[str, np.ndarray] = {}
        self.fragment_cache: Dict[str, List[KnowledgeFragment]] = {}
        
    async def retrieve_knowledge(self, query: str, limit: int = 10) -> List[KnowledgeFragment]:
        """Retrieve knowledge fragments using vector similarity search"""
        if not self.supabase or not self.embedding_client:
            return []
        
        # Check cache first
        cache_key = f"{query}_{limit}"
        if cache_key in self.fragment_cache:
            return self.fragment_cache[cache_key]
        
        try:
            # Get query embedding
            query_embedding = await self._get_embedding(query)
            
            # Perform vector similarity search
            result = self.supabase.rpc(
                'match_site_pages',
                {
                    'query_embedding': query_embedding.tolist(),
                    'match_count': limit,
                    'filter': {}
                }
            ).execute()
            
            fragments = []
            if result.data:
                for item in result.data:
                    # Extract embedding from database
                    embedding_data = item.get('embedding', [])
                    if embedding_data:
                        embedding = np.array(embedding_data)
                    else:
                        # Generate embedding if missing
                        embedding = await self._get_embedding(item.get('content', ''))
                    
                    fragment = KnowledgeFragment(
                        id=str(item.get('id', '')),
                        content=item.get('content', ''),
                        vector_embedding=embedding,
                        metadata={
                            'title': item.get('title', ''),
                            'url': item.get('url', ''),
                            'chunk_number': item.get('chunk_number', 0),
                            'similarity_score': item.get('similarity', 0.0)
                        },
                        source='supabase_vector_db',
                        confidence=item.get('similarity', 0.0),
                        timestamp=item.get('created_at', 0)
                    )
                    fragments.append(fragment)
            
            # Cache results
            self.fragment_cache[cache_key] = fragments
            
            return fragments
            
        except Exception as e:
            print(f"Error retrieving knowledge: {e}")
            return []
    
    async def store_knowledge(self, fragment: KnowledgeFragment) -> bool:
        """Store knowledge fragment in vector database"""
        if not self.supabase:
            return False
        
        try:
            # Ensure embedding exists
            if fragment.vector_embedding.size == 0:
                fragment.vector_embedding = await self._get_embedding(fragment.content)
            
            # Prepare data for insertion
            data = {
                'content': fragment.content,
                'embedding': fragment.vector_embedding.tolist(),
                'metadata': fragment.metadata,
                'url': fragment.metadata.get('url', ''),
                'title': fragment.metadata.get('title', ''),
                'chunk_number': fragment.metadata.get('chunk_number', 0),
                'summary': fragment.content[:200] + '...' if len(fragment.content) > 200 else fragment.content
            }
            
            # Insert into database
            result = self.supabase.table(self.table_name).insert(data).execute()
            
            return len(result.data) > 0
            
        except Exception as e:
            print(f"Error storing knowledge: {e}")
            return False
    
    def convert_to_tensor_input(self, fragments: List[KnowledgeFragment]) -> np.ndarray:
        """Convert knowledge fragments to tensor input for cognitive grammar"""
        if not fragments:
            return np.zeros((1, self.embedding_dim))
        
        # Limit number of fragments
        limited_fragments = fragments[:self.max_fragments]
        
        # Stack embeddings
        embeddings = []
        for fragment in limited_fragments:
            tensor_embedding = fragment.to_tensor_compatible(self.embedding_dim)
            
            # Weight by confidence
            weighted_embedding = tensor_embedding * fragment.confidence
            embeddings.append(weighted_embedding)
        
        # Convert to tensor
        tensor_input = np.stack(embeddings)
        
        # Apply pooling if too many fragments
        if tensor_input.shape[0] > self.max_fragments:
            # Use max pooling to preserve important features
            pooled = np.max(tensor_input.reshape(-1, self.max_fragments, self.embedding_dim), axis=1)
            return pooled
        
        return tensor_input
    
    async def _get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for text with caching"""
        if not self.embedding_client:
            return np.random.normal(0, 0.1, self.embedding_dim)
        
        # Check cache
        if text in self.embedding_cache:
            return self.embedding_cache[text]
        
        try:
            response = await self.embedding_client.embeddings.create(
                model=self.embedding_model,
                input=text
            )
            embedding = np.array(response.data[0].embedding)
            
            # Cache result
            self.embedding_cache[text] = embedding
            
            return embedding
            
        except Exception as e:
            print(f"Error getting embedding: {e}")
            # Return random embedding as fallback
            return np.random.normal(0, 0.1, self.embedding_dim)
    
    def create_knowledge_tensor_kernel(self, fragments: List[KnowledgeFragment]) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Create tensor kernel representation of knowledge fragments
        
        Returns:
            Tuple of (tensor_data, metadata)
        """
        if not fragments:
            return np.zeros((1, self.embedding_dim)), {}
        
        # Convert fragments to tensor
        tensor_data = self.convert_to_tensor_input(fragments)
        
        # Create metadata
        metadata = {
            'fragment_count': len(fragments),
            'sources': list(set(f.source for f in fragments)),
            'avg_confidence': np.mean([f.confidence for f in fragments]),
            'content_summary': [f.content[:100] for f in fragments[:5]],  # Sample content
            'tensor_shape': tensor_data.shape
        }
        
        return tensor_data, metadata
    
    def adapt_for_cognitive_node(self, query: str, node_features: int, node_contexts: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Adapt knowledge for cognitive node tensor kernel
        
        Args:
            query: Query to retrieve relevant knowledge
            node_features: Number of node features
            node_contexts: Number of node contexts
            
        Returns:
            Tuple of (feature_vector, context_vector)
        """
        # This would be called asynchronously in practice
        # For now, return dummy data
        feature_vector = np.random.normal(0, 0.1, node_features)
        context_vector = np.random.normal(0, 0.1, node_contexts)
        
        return feature_vector, context_vector
    
    async def adapt_for_cognitive_node_async(self, query: str, node_features: int, node_contexts: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Async version of adapt_for_cognitive_node
        """
        # Retrieve relevant knowledge
        fragments = await self.retrieve_knowledge(query, limit=10)
        
        if not fragments:
            # Return default vectors if no knowledge found
            feature_vector = np.ones(node_features) * 0.1
            context_vector = np.ones(node_contexts) * 0.1
            return feature_vector, context_vector
        
        # Convert knowledge to tensor format
        knowledge_tensor, metadata = self.create_knowledge_tensor_kernel(fragments)
        
        # Adapt to node dimensions
        # Feature vector: aggregate knowledge content
        if knowledge_tensor.size >= node_features:
            feature_vector = np.mean(knowledge_tensor, axis=0)[:node_features]
        else:
            feature_vector = np.zeros(node_features)
            feature_vector[:knowledge_tensor.shape[1]] = np.mean(knowledge_tensor, axis=0)
        
        # Context vector: metadata and confidence information
        context_vector = np.zeros(node_contexts)
        if len(fragments) > 0:
            # Fill context with aggregated information
            context_vector[0] = len(fragments) / 10.0  # Normalized fragment count
            context_vector[1] = metadata['avg_confidence']
            context_vector[2] = len(metadata['sources']) / 5.0  # Normalized source diversity
            
            # Fill remaining context with pooled embeddings
            if knowledge_tensor.size > 0 and node_contexts > 3:
                pooled_context = np.mean(knowledge_tensor, axis=(0, 1)) if knowledge_tensor.ndim > 1 else np.mean(knowledge_tensor)
                remaining_slots = min(node_contexts - 3, 10)
                if isinstance(pooled_context, np.ndarray):
                    context_vector[3:3+remaining_slots] = pooled_context[:remaining_slots]
                else:
                    context_vector[3] = pooled_context
        
        return feature_vector, context_vector
    
    def get_knowledge_statistics(self) -> Dict[str, Any]:
        """Get statistics about knowledge adapter"""
        return {
            'embedding_cache_size': len(self.embedding_cache),
            'fragment_cache_size': len(self.fragment_cache),
            'embedding_dimension': self.embedding_dim,
            'max_fragments': self.max_fragments,
            'embedding_model': self.embedding_model,
            'table_name': self.table_name,
            'has_supabase': self.supabase is not None,
            'has_embedding_client': self.embedding_client is not None
        }
    
    def clear_cache(self):
        """Clear adapter caches"""
        self.embedding_cache.clear()
        self.fragment_cache.clear()


class LocalKnowledgeAdapter(KnowledgeAdapter):
    """
    Simple local knowledge adapter for testing and development
    """
    
    def __init__(self):
        self.knowledge_store: Dict[str, KnowledgeFragment] = {}
        self.embedding_dim = 512
        
    async def retrieve_knowledge(self, query: str, limit: int = 10) -> List[KnowledgeFragment]:
        """Retrieve knowledge using simple text matching"""
        query_lower = query.lower()
        matches = []
        
        for fragment in self.knowledge_store.values():
            if query_lower in fragment.content.lower():
                # Simple similarity score based on word overlap
                query_words = set(query_lower.split())
                content_words = set(fragment.content.lower().split())
                similarity = len(query_words.intersection(content_words)) / len(query_words.union(content_words))
                
                fragment.confidence = similarity
                matches.append(fragment)
        
        # Sort by confidence and return top matches
        matches.sort(key=lambda x: x.confidence, reverse=True)
        return matches[:limit]
    
    async def store_knowledge(self, fragment: KnowledgeFragment) -> bool:
        """Store knowledge fragment locally"""
        # Generate simple embedding if missing
        if fragment.vector_embedding.size == 0:
            fragment.vector_embedding = self._generate_simple_embedding(fragment.content)
        
        self.knowledge_store[fragment.id] = fragment
        return True
    
    def convert_to_tensor_input(self, fragments: List[KnowledgeFragment]) -> np.ndarray:
        """Convert fragments to tensor input"""
        if not fragments:
            return np.zeros((1, self.embedding_dim))
        
        embeddings = [fragment.to_tensor_compatible(self.embedding_dim) for fragment in fragments]
        return np.stack(embeddings)
    
    def _generate_simple_embedding(self, text: str) -> np.ndarray:
        """Generate simple embedding based on text characteristics"""
        # Very simple embedding: character frequencies + length + word count
        embedding = np.zeros(self.embedding_dim)
        
        # Character frequency features (first 256 dimensions)
        for i, char in enumerate(text[:256]):
            if i < 256:
                embedding[i] = ord(char) / 255.0
        
        # Text statistics (next features)
        if self.embedding_dim > 256:
            embedding[256] = len(text) / 1000.0  # Normalized length
            embedding[257] = len(text.split()) / 100.0  # Normalized word count
            embedding[258] = text.count('.') / 10.0  # Normalized sentence count
            
        # Random noise for remaining dimensions
        if self.embedding_dim > 259:
            embedding[259:] = np.random.normal(0, 0.1, self.embedding_dim - 259)
        
        return embedding
    
    def add_sample_knowledge(self):
        """Add some sample knowledge for testing"""
        samples = [
            ("ai_agent", "AI agents are autonomous software entities that can perceive their environment and take actions to achieve goals."),
            ("pydantic", "Pydantic is a Python library that provides data validation and settings management using Python type annotations."),
            ("langraph", "LangGraph is a library for building stateful, multi-actor applications with large language models."),
            ("attention", "Attention mechanisms in AI allow models to focus on relevant parts of input when making decisions."),
            ("tensor", "Tensors are multi-dimensional arrays that represent data in machine learning and neural networks.")
        ]
        
        for i, (topic, content) in enumerate(samples):
            fragment = KnowledgeFragment(
                id=f"sample_{i}",
                content=content,
                vector_embedding=self._generate_simple_embedding(content),
                metadata={'topic': topic, 'type': 'sample'},
                source='local_sample',
                confidence=1.0
            )
            self.knowledge_store[fragment.id] = fragment


class KnowledgeIntegrationManager:
    """
    Manages multiple knowledge adapters and integrates them with cognitive grammar
    """
    
    def __init__(self):
        self.adapters: Dict[str, KnowledgeAdapter] = {}
        self.integration_weights: Dict[str, float] = {}
        
    def register_adapter(self, name: str, adapter: KnowledgeAdapter, weight: float = 1.0):
        """Register a knowledge adapter"""
        self.adapters[name] = adapter
        self.integration_weights[name] = weight
    
    async def retrieve_integrated_knowledge(self, query: str, limit: int = 10) -> List[KnowledgeFragment]:
        """Retrieve knowledge from all adapters and integrate results"""
        all_fragments = []
        
        # Retrieve from all adapters
        for name, adapter in self.adapters.items():
            try:
                fragments = await adapter.retrieve_knowledge(query, limit)
                
                # Apply adapter weight
                weight = self.integration_weights[name]
                for fragment in fragments:
                    fragment.confidence *= weight
                    fragment.metadata['adapter'] = name
                
                all_fragments.extend(fragments)
                
            except Exception as e:
                print(f"Error retrieving from adapter {name}: {e}")
        
        # Remove duplicates and sort by confidence
        unique_fragments = self._deduplicate_fragments(all_fragments)
        unique_fragments.sort(key=lambda x: x.confidence, reverse=True)
        
        return unique_fragments[:limit]
    
    def _deduplicate_fragments(self, fragments: List[KnowledgeFragment]) -> List[KnowledgeFragment]:
        """Remove duplicate fragments based on content similarity"""
        if not fragments:
            return []
        
        unique = []
        for fragment in fragments:
            is_duplicate = False
            
            for existing in unique:
                # Simple duplicate check based on content overlap
                content_similarity = self._calculate_content_similarity(fragment.content, existing.content)
                if content_similarity > 0.8:  # High similarity threshold
                    is_duplicate = True
                    # Keep the one with higher confidence
                    if fragment.confidence > existing.confidence:
                        unique.remove(existing)
                        unique.append(fragment)
                    break
            
            if not is_duplicate:
                unique.append(fragment)
        
        return unique
    
    def _calculate_content_similarity(self, content1: str, content2: str) -> float:
        """Calculate similarity between two text contents"""
        words1 = set(content1.lower().split())
        words2 = set(content2.lower().split())
        
        if not words1 and not words2:
            return 1.0
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)
    
    async def create_integrated_tensor_input(self, query: str, target_shape: Tuple[int, ...]) -> np.ndarray:
        """Create tensor input by integrating knowledge from all adapters"""
        fragments = await self.retrieve_integrated_knowledge(query, limit=20)
        
        if not fragments:
            return np.zeros(target_shape)
        
        # Combine tensors from all adapters
        combined_tensors = []
        
        for name, adapter in self.adapters.items():
            adapter_fragments = [f for f in fragments if f.metadata.get('adapter') == name]
            if adapter_fragments:
                tensor_input = adapter.convert_to_tensor_input(adapter_fragments)
                combined_tensors.append(tensor_input)
        
        if not combined_tensors:
            return np.zeros(target_shape)
        
        # Concatenate or pool tensors
        if len(combined_tensors) == 1:
            combined = combined_tensors[0]
        else:
            # Stack and then pool
            try:
                stacked = np.concatenate(combined_tensors, axis=0)
                combined = np.mean(stacked, axis=0, keepdims=True)
            except ValueError:
                # Fallback to first tensor if shapes don't match
                combined = combined_tensors[0]
        
        # Reshape to target shape
        if combined.shape == target_shape:
            return combined
        else:
            # Reshape or pad/truncate as needed
            flat_combined = combined.flatten()
            flat_target_size = np.prod(target_shape)
            
            if flat_combined.size >= flat_target_size:
                reshaped = flat_combined[:flat_target_size].reshape(target_shape)
            else:
                padded = np.zeros(flat_target_size)
                padded[:flat_combined.size] = flat_combined
                reshaped = padded.reshape(target_shape)
            
            return reshaped
    
    def get_integration_statistics(self) -> Dict[str, Any]:
        """Get statistics about knowledge integration"""
        adapter_stats = {}
        for name, adapter in self.adapters.items():
            if hasattr(adapter, 'get_knowledge_statistics'):
                adapter_stats[name] = adapter.get_knowledge_statistics()
            else:
                adapter_stats[name] = {'type': type(adapter).__name__}
        
        return {
            'adapter_count': len(self.adapters),
            'adapter_weights': self.integration_weights,
            'adapter_statistics': adapter_stats
        }