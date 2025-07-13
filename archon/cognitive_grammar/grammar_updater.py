"""
Adaptive Grammar Updater

Updates and refines cognitive grammar based on learning traces and
performance feedback from the tensor network.
"""

try:
    import numpy as np
except ImportError:
    # Use mock numpy for testing
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'tests'))
    import mock_numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
import time

from .cognitive_parser import CognitiveGrammarParser, GrammarElement, GrammarElementType
from .hypergraph import HypergraphEncoder


class UpdateStrategy(Enum):
    """Strategies for grammar updating"""
    INCREMENTAL = "incremental"  # Small, continuous updates
    BATCH = "batch"             # Periodic batch updates
    ADAPTIVE = "adaptive"       # Strategy adapts based on performance
    REINFORCEMENT = "reinforcement"  # Reinforcement learning style


@dataclass
class GrammarUpdate:
    """Represents a single grammar update"""
    id: str
    timestamp: float
    update_type: str  # "pattern", "weight", "structure"
    target_element: str
    old_value: Any
    new_value: Any
    confidence: float
    performance_impact: float = 0.0


@dataclass
class LearningTrace:
    """Trace of learning events for grammar adaptation"""
    timestamp: float
    element_id: str
    performance_score: float
    attention_received: float
    activation_pattern: np.ndarray
    context_factors: Dict[str, Any]


class AdaptiveGrammarUpdater:
    """
    Updates cognitive grammar patterns based on tensor network performance
    and attention allocation patterns.
    """
    
    def __init__(self, parser: CognitiveGrammarParser, 
                 update_strategy: UpdateStrategy = UpdateStrategy.ADAPTIVE):
        self.parser = parser
        self.update_strategy = update_strategy
        self.learning_traces: List[LearningTrace] = []
        self.grammar_updates: List[GrammarUpdate] = []
        
        # Learning parameters
        self.learning_rate = 0.01
        self.momentum = 0.9
        self.decay_rate = 0.95
        
        # Pattern tracking
        self.pattern_performance: Dict[str, List[float]] = {}
        self.pattern_usage_count: Dict[str, int] = {}
        self.pattern_weights: Dict[str, float] = {}
        
        # Adaptation parameters
        self.min_confidence_threshold = 0.3
        self.performance_window = 100
        self.update_frequency = 10  # Updates every N learning traces
        
        # Initialize pattern weights
        self._initialize_pattern_weights()
    
    def _initialize_pattern_weights(self):
        """Initialize weights for all grammar patterns"""
        for pattern_type, patterns in self.parser.grammar_patterns.items():
            for i, pattern in enumerate(patterns):
                pattern_key = f"{pattern_type}_{i}"
                self.pattern_weights[pattern_key] = 1.0
                self.pattern_performance[pattern_key] = []
                self.pattern_usage_count[pattern_key] = 0
    
    def record_learning_trace(self, element_id: str, performance_score: float,
                            attention_received: float, activation_pattern: np.ndarray,
                            context_factors: Optional[Dict[str, Any]] = None):
        """Record a learning trace for later grammar adaptation"""
        trace = LearningTrace(
            timestamp=time.time(),
            element_id=element_id,
            performance_score=performance_score,
            attention_received=attention_received,
            activation_pattern=activation_pattern.copy(),
            context_factors=context_factors or {}
        )
        
        self.learning_traces.append(trace)
        
        # Trigger update if enough traces accumulated
        if len(self.learning_traces) % self.update_frequency == 0:
            self.update_grammar()
    
    def update_grammar(self) -> List[GrammarUpdate]:
        """
        Update grammar based on accumulated learning traces
        
        Returns:
            List of grammar updates applied
        """
        if not self.learning_traces:
            return []
        
        updates = []
        
        if self.update_strategy == UpdateStrategy.INCREMENTAL:
            updates = self._incremental_update()
        elif self.update_strategy == UpdateStrategy.BATCH:
            updates = self._batch_update()
        elif self.update_strategy == UpdateStrategy.REINFORCEMENT:
            updates = self._reinforcement_update()
        else:  # ADAPTIVE
            updates = self._adaptive_update()
        
        # Apply updates
        for update in updates:
            self._apply_grammar_update(update)
        
        self.grammar_updates.extend(updates)
        
        # Keep only recent traces
        if len(self.learning_traces) > self.performance_window * 2:
            self.learning_traces = self.learning_traces[-self.performance_window:]
        
        return updates
    
    def _incremental_update(self) -> List[GrammarUpdate]:
        """Perform incremental grammar updates"""
        updates = []
        recent_traces = self.learning_traces[-self.update_frequency:]
        
        # Analyze recent performance
        performance_by_element = {}
        for trace in recent_traces:
            if trace.element_id not in performance_by_element:
                performance_by_element[trace.element_id] = []
            performance_by_element[trace.element_id].append(trace.performance_score)
        
        # Update patterns based on performance
        for element_id, scores in performance_by_element.items():
            avg_performance = np.mean(scores)
            
            # Find associated patterns
            associated_patterns = self._find_patterns_for_element(element_id)
            
            for pattern_key in associated_patterns:
                old_weight = self.pattern_weights[pattern_key]
                
                # Incremental weight update
                weight_update = self.learning_rate * (avg_performance - 0.5)
                new_weight = old_weight + weight_update
                new_weight = max(0.1, min(2.0, new_weight))  # Clamp
                
                if abs(new_weight - old_weight) > 0.01:  # Significant change
                    update = GrammarUpdate(
                        id=f"inc_{len(self.grammar_updates)}",
                        timestamp=time.time(),
                        update_type="weight",
                        target_element=pattern_key,
                        old_value=old_weight,
                        new_value=new_weight,
                        confidence=min(len(scores) / 10.0, 1.0),
                        performance_impact=avg_performance
                    )
                    updates.append(update)
        
        return updates
    
    def _batch_update(self) -> List[GrammarUpdate]:
        """Perform batch grammar updates"""
        updates = []
        
        if len(self.learning_traces) < self.performance_window:
            return updates
        
        # Analyze performance trends
        performance_trends = self._analyze_performance_trends()
        
        # Update patterns based on trends
        for pattern_key, trend_data in performance_trends.items():
            slope = trend_data['slope']
            confidence = trend_data['confidence']
            avg_performance = trend_data['avg_performance']
            
            if confidence > self.min_confidence_threshold:
                old_weight = self.pattern_weights[pattern_key]
                
                # Batch weight adjustment based on trend
                if slope > 0.01:  # Improving performance
                    weight_multiplier = 1.1
                elif slope < -0.01:  # Declining performance
                    weight_multiplier = 0.9
                else:
                    weight_multiplier = 1.0
                
                new_weight = old_weight * weight_multiplier
                new_weight = max(0.1, min(2.0, new_weight))
                
                if abs(new_weight - old_weight) > 0.05:
                    update = GrammarUpdate(
                        id=f"batch_{len(self.grammar_updates)}",
                        timestamp=time.time(),
                        update_type="weight",
                        target_element=pattern_key,
                        old_value=old_weight,
                        new_value=new_weight,
                        confidence=confidence,
                        performance_impact=avg_performance
                    )
                    updates.append(update)
        
        return updates
    
    def _reinforcement_update(self) -> List[GrammarUpdate]:
        """Perform reinforcement learning style updates"""
        updates = []
        
        # Group traces by element and calculate rewards
        element_rewards = {}
        for trace in self.learning_traces[-self.performance_window:]:
            if trace.element_id not in element_rewards:
                element_rewards[trace.element_id] = []
            
            # Reward is combination of performance and attention
            reward = trace.performance_score * trace.attention_received
            element_rewards[trace.element_id].append(reward)
        
        # Apply reinforcement updates
        for element_id, rewards in element_rewards.items():
            if len(rewards) < 5:  # Need minimum samples
                continue
            
            avg_reward = np.mean(rewards)
            reward_variance = np.var(rewards)
            
            # Find patterns and apply reinforcement
            patterns = self._find_patterns_for_element(element_id)
            
            for pattern_key in patterns:
                old_weight = self.pattern_weights[pattern_key]
                
                # Reinforcement update rule
                if avg_reward > 0.6:  # High reward
                    weight_update = self.learning_rate * (1 + avg_reward)
                elif avg_reward < 0.4:  # Low reward
                    weight_update = -self.learning_rate * (1 - avg_reward)
                else:
                    weight_update = 0
                
                # Add exploration bonus for high variance
                if reward_variance > 0.1:
                    weight_update += self.learning_rate * 0.1
                
                new_weight = old_weight + weight_update
                new_weight = max(0.1, min(2.0, new_weight))
                
                if abs(weight_update) > 0.01:
                    update = GrammarUpdate(
                        id=f"rl_{len(self.grammar_updates)}",
                        timestamp=time.time(),
                        update_type="weight",
                        target_element=pattern_key,
                        old_value=old_weight,
                        new_value=new_weight,
                        confidence=min(len(rewards) / 20.0, 1.0),
                        performance_impact=avg_reward
                    )
                    updates.append(update)
        
        return updates
    
    def _adaptive_update(self) -> List[GrammarUpdate]:
        """Adaptive update strategy that chooses best approach"""
        # Analyze current learning state
        recent_traces = self.learning_traces[-20:] if len(self.learning_traces) >= 20 else self.learning_traces
        
        if not recent_traces:
            return []
        
        # Calculate performance variance
        performances = [trace.performance_score for trace in recent_traces]
        performance_variance = np.var(performances)
        
        # Decide strategy based on context
        if performance_variance > 0.2:
            # High variance - use reinforcement learning
            return self._reinforcement_update()
        elif len(self.learning_traces) >= self.performance_window:
            # Enough data for batch analysis
            return self._batch_update()
        else:
            # Default to incremental
            return self._incremental_update()
    
    def _analyze_performance_trends(self) -> Dict[str, Dict[str, float]]:
        """Analyze performance trends for each pattern"""
        trends = {}
        
        # Group traces by pattern
        pattern_traces = {}
        for trace in self.learning_traces[-self.performance_window:]:
            patterns = self._find_patterns_for_element(trace.element_id)
            for pattern_key in patterns:
                if pattern_key not in pattern_traces:
                    pattern_traces[pattern_key] = []
                pattern_traces[pattern_key].append(trace)
        
        # Calculate trends
        for pattern_key, traces in pattern_traces.items():
            if len(traces) < 10:  # Need minimum samples
                continue
            
            performances = [trace.performance_score for trace in traces]
            timestamps = [trace.timestamp for trace in traces]
            
            # Calculate linear trend
            if len(performances) > 1:
                # Normalize timestamps
                t_min = min(timestamps)
                t_norm = [(t - t_min) / 3600.0 for t in timestamps]  # Hours
                
                # Linear regression
                slope, intercept = np.polyfit(t_norm, performances, 1)
                
                # Calculate confidence (R-squared)
                predicted = [slope * t + intercept for t in t_norm]
                ss_res = sum((p - a) ** 2 for p, a in zip(predicted, performances))
                ss_tot = sum((p - np.mean(performances)) ** 2 for p in performances)
                r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                
                trends[pattern_key] = {
                    'slope': slope,
                    'confidence': r_squared,
                    'avg_performance': np.mean(performances),
                    'sample_count': len(performances)
                }
        
        return trends
    
    def _find_patterns_for_element(self, element_id: str) -> List[str]:
        """Find grammar patterns associated with an element"""
        # This is a simplified mapping - in practice, would need
        # to track which patterns were used to extract each element
        patterns = []
        
        # Extract element type from ID (assuming format like "agent_name")
        parts = element_id.split('_')
        if len(parts) >= 2:
            element_type = parts[0]
            
            # Find patterns for this element type
            pattern_type_key = f"{element_type}_patterns"
            if pattern_type_key in self.parser.grammar_patterns:
                pattern_count = len(self.parser.grammar_patterns[pattern_type_key])
                for i in range(pattern_count):
                    patterns.append(f"{pattern_type_key}_{i}")
        
        return patterns
    
    def _apply_grammar_update(self, update: GrammarUpdate):
        """Apply a grammar update to the parser"""
        if update.update_type == "weight":
            self.pattern_weights[update.target_element] = update.new_value
        elif update.update_type == "pattern":
            # Would modify actual patterns in parser
            pass
        elif update.update_type == "structure":
            # Would modify parser structure
            pass
    
    def create_updated_parser(self) -> CognitiveGrammarParser:
        """Create a new parser with updated weights"""
        # Create new parser instance
        updated_parser = CognitiveGrammarParser()
        
        # Apply learned weights (simplified - would need more sophisticated integration)
        # For now, we just track the weights separately
        updated_parser.pattern_weights = self.pattern_weights.copy()
        
        return updated_parser
    
    def get_learning_statistics(self) -> Dict[str, Any]:
        """Get comprehensive learning statistics"""
        if not self.learning_traces:
            return {'no_data': True}
        
        recent_traces = self.learning_traces[-50:]
        
        # Performance statistics
        performances = [trace.performance_score for trace in recent_traces]
        attention_scores = [trace.attention_received for trace in recent_traces]
        
        # Pattern statistics
        pattern_stats = {}
        for pattern_key, weight in self.pattern_weights.items():
            usage_count = self.pattern_usage_count.get(pattern_key, 0)
            performance_history = self.pattern_performance.get(pattern_key, [])
            
            pattern_stats[pattern_key] = {
                'weight': weight,
                'usage_count': usage_count,
                'avg_performance': np.mean(performance_history) if performance_history else 0.0,
                'performance_trend': np.polyfit(range(len(performance_history)), performance_history, 1)[0] if len(performance_history) > 1 else 0.0
            }
        
        return {
            'total_traces': len(self.learning_traces),
            'total_updates': len(self.grammar_updates),
            'avg_performance': np.mean(performances),
            'performance_trend': np.polyfit(range(len(performances)), performances, 1)[0] if len(performances) > 1 else 0.0,
            'avg_attention': np.mean(attention_scores),
            'update_strategy': self.update_strategy.value,
            'pattern_statistics': pattern_stats,
            'recent_update_count': len([u for u in self.grammar_updates if time.time() - u.timestamp < 3600])  # Last hour
        }
    
    def export_learning_state(self) -> Dict[str, Any]:
        """Export learning state for persistence"""
        return {
            'pattern_weights': self.pattern_weights,
            'pattern_performance': {k: v[-100:] for k, v in self.pattern_performance.items()},  # Last 100
            'pattern_usage_count': self.pattern_usage_count,
            'learning_parameters': {
                'learning_rate': self.learning_rate,
                'momentum': self.momentum,
                'decay_rate': self.decay_rate,
                'min_confidence_threshold': self.min_confidence_threshold,
                'performance_window': self.performance_window,
                'update_frequency': self.update_frequency
            },
            'update_strategy': self.update_strategy.value,
            'grammar_updates': [
                {
                    'id': update.id,
                    'timestamp': update.timestamp,
                    'update_type': update.update_type,
                    'target_element': update.target_element,
                    'confidence': update.confidence,
                    'performance_impact': update.performance_impact
                }
                for update in self.grammar_updates[-100:]  # Last 100 updates
            ]
        }
    
    def import_learning_state(self, state: Dict[str, Any]):
        """Import learning state from persistence"""
        self.pattern_weights = state.get('pattern_weights', {})
        self.pattern_performance = state.get('pattern_performance', {})
        self.pattern_usage_count = state.get('pattern_usage_count', {})
        
        # Import parameters
        params = state.get('learning_parameters', {})
        self.learning_rate = params.get('learning_rate', 0.01)
        self.momentum = params.get('momentum', 0.9)
        self.decay_rate = params.get('decay_rate', 0.95)
        self.min_confidence_threshold = params.get('min_confidence_threshold', 0.3)
        self.performance_window = params.get('performance_window', 100)
        self.update_frequency = params.get('update_frequency', 10)
        
        # Import strategy
        strategy_str = state.get('update_strategy', 'adaptive')
        self.update_strategy = UpdateStrategy(strategy_str)
    
    def reset_learning(self):
        """Reset all learning state"""
        self.learning_traces.clear()
        self.grammar_updates.clear()
        self.pattern_performance.clear()
        self.pattern_usage_count.clear()
        self._initialize_pattern_weights()