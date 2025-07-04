"""
ECAN-Style Attention Allocation Engine

Implements Economic Attention Network (ECAN) style attention allocation 
with dynamic resource distribution and cognitive salience tracking.
"""

try:
    import numpy as np
except ImportError:
    # Use mock numpy for testing
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'tests'))
    import mock_numpy as np
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import time
import json


class AttentionMode(Enum):
    """Attention allocation modes"""
    FOCUSED = "focused"      # Concentrate on few high-priority items
    DISTRIBUTED = "distributed"  # Spread attention across many items
    ADAPTIVE = "adaptive"    # Dynamically adjust based on context
    EMERGENCY = "emergency"  # Crisis mode with rapid reallocation


@dataclass
class AttentionUnit:
    """Basic unit of attention with economic properties"""
    id: str
    salience: float = 0.0          # How important/noticeable
    urgency: float = 0.0           # How time-critical
    complexity: float = 1.0        # How much attention required
    allocated_attention: float = 0.0   # Currently allocated attention
    historical_performance: List[float] = field(default_factory=list)
    last_update: float = field(default_factory=time.time)
    
    def update_salience(self, new_salience: float, decay_rate: float = 0.1):
        """Update salience with temporal decay"""
        time_decay = np.exp(-decay_rate * (time.time() - self.last_update))
        self.salience = self.salience * time_decay + new_salience * (1 - time_decay)
        self.last_update = time.time()
    
    def get_attention_value(self) -> float:
        """Compute attention value based on salience, urgency, and complexity"""
        return (self.salience * self.urgency) / max(self.complexity, 0.1)


@dataclass
class AttentionResource:
    """Represents available attention resources"""
    total_capacity: float = 100.0
    available: float = 100.0
    allocated: float = 0.0
    efficiency: float = 1.0
    last_refresh: float = field(default_factory=time.time)
    
    def allocate(self, amount: float) -> bool:
        """Allocate attention resources"""
        if self.available >= amount:
            self.available -= amount
            self.allocated += amount
            return True
        return False
    
    def release(self, amount: float):
        """Release attention resources"""
        released = min(amount, self.allocated)
        self.available += released
        self.allocated -= released
    
    def refresh(self, refresh_rate: float = 0.1):
        """Gradually refresh attention resources"""
        time_elapsed = time.time() - self.last_refresh
        refresh_amount = self.total_capacity * refresh_rate * time_elapsed
        
        self.available = min(self.total_capacity, self.available + refresh_amount)
        self.allocated = max(0, self.total_capacity - self.available)
        self.last_refresh = time.time()


class ECANAttentionEngine:
    """
    Economic Attention Network (ECAN) style attention allocation engine.
    Manages attention as a limited resource with economic principles.
    """
    
    def __init__(self, total_attention_capacity: float = 100.0, 
                 min_attention_threshold: float = 0.1):
        self.attention_resource = AttentionResource(total_attention_capacity)
        self.attention_units: Dict[str, AttentionUnit] = {}
        self.allocation_history: List[Dict[str, float]] = []
        self.min_threshold = min_attention_threshold
        self.attention_mode = AttentionMode.ADAPTIVE
        
        # ECAN parameters
        self.salience_decay = 0.05      # How quickly salience decays
        self.urgency_amplifier = 2.0    # Multiplier for urgent items
        self.complexity_penalty = 0.5   # Penalty for complex items
        self.focus_temperature = 1.0    # Controls focus vs distribution
        
        # Market-like dynamics
        self.attention_price = 1.0      # Cost of attention allocation
        self.demand_elasticity = 0.8    # How demand responds to price
        
        # Performance tracking
        self.performance_history: List[Dict[str, Any]] = []
        
    def register_attention_unit(self, unit_id: str, initial_salience: float = 0.5,
                              initial_urgency: float = 0.5, complexity: float = 1.0) -> AttentionUnit:
        """Register a new unit that can receive attention"""
        unit = AttentionUnit(
            id=unit_id,
            salience=initial_salience,
            urgency=initial_urgency,
            complexity=complexity
        )
        self.attention_units[unit_id] = unit
        return unit
    
    def update_unit_salience(self, unit_id: str, salience_delta: float, 
                           urgency_delta: float = 0.0):
        """Update salience and urgency for a specific unit"""
        if unit_id in self.attention_units:
            unit = self.attention_units[unit_id]
            unit.update_salience(unit.salience + salience_delta)
            unit.urgency = max(0.0, min(1.0, unit.urgency + urgency_delta))
    
    def allocate_attention(self, explicit_demands: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        """
        Main attention allocation function using ECAN principles
        
        Args:
            explicit_demands: Optional explicit attention demands for specific units
            
        Returns:
            Dictionary mapping unit_id to allocated attention amount
        """
        # Refresh attention resources
        self.attention_resource.refresh()
        
        # Collect all units requiring attention
        active_units = {uid: unit for uid, unit in self.attention_units.items() 
                       if unit.salience > self.min_threshold or 
                       (explicit_demands and uid in explicit_demands)}
        
        if not active_units:
            return {}
        
        # Calculate attention demands
        demands = self._calculate_attention_demands(active_units, explicit_demands)
        
        # Apply market dynamics
        adjusted_demands = self._apply_market_dynamics(demands)
        
        # Perform allocation based on current mode
        allocation = self._perform_allocation(adjusted_demands)
        
        # Update allocation history
        self.allocation_history.append(allocation.copy())
        if len(self.allocation_history) > 1000:
            self.allocation_history = self.allocation_history[-1000:]
        
        # Update unit allocations
        for unit_id, amount in allocation.items():
            if unit_id in self.attention_units:
                self.attention_units[unit_id].allocated_attention = amount
        
        # Track performance
        self._track_performance(allocation)
        
        return allocation
    
    def _calculate_attention_demands(self, active_units: Dict[str, AttentionUnit],
                                   explicit_demands: Optional[Dict[str, float]]) -> Dict[str, float]:
        """Calculate raw attention demands for each unit"""
        demands = {}
        
        for unit_id, unit in active_units.items():
            # Base demand from attention value
            base_demand = unit.get_attention_value()
            
            # Apply urgency amplification
            if unit.urgency > 0.7:
                base_demand *= self.urgency_amplifier
            
            # Apply complexity penalty
            complexity_factor = 1.0 + (unit.complexity - 1.0) * self.complexity_penalty
            base_demand *= complexity_factor
            
            # Incorporate explicit demands
            if explicit_demands and unit_id in explicit_demands:
                explicit_factor = explicit_demands[unit_id]
                base_demand = base_demand * 0.7 + explicit_factor * 0.3
            
            demands[unit_id] = max(0.0, base_demand)
        
        return demands
    
    def _apply_market_dynamics(self, demands: Dict[str, float]) -> Dict[str, float]:
        """Apply economic market dynamics to attention demands"""
        total_demand = sum(demands.values())
        available_supply = self.attention_resource.available
        
        if total_demand <= available_supply:
            # Abundant attention - no adjustment needed
            return demands
        
        # Scarce attention - apply price mechanism
        supply_demand_ratio = available_supply / total_demand
        self.attention_price = 1.0 / supply_demand_ratio
        
        # Apply demand elasticity
        adjusted_demands = {}
        for unit_id, demand in demands.items():
            price_effect = (self.attention_price ** (-self.demand_elasticity))
            adjusted_demand = demand * price_effect
            adjusted_demands[unit_id] = adjusted_demand
        
        return adjusted_demands
    
    def _perform_allocation(self, demands: Dict[str, float]) -> Dict[str, float]:
        """Perform final attention allocation based on mode and constraints"""
        allocation = {}
        
        if self.attention_mode == AttentionMode.FOCUSED:
            allocation = self._focused_allocation(demands)
        elif self.attention_mode == AttentionMode.DISTRIBUTED:
            allocation = self._distributed_allocation(demands)
        elif self.attention_mode == AttentionMode.EMERGENCY:
            allocation = self._emergency_allocation(demands)
        else:  # ADAPTIVE
            allocation = self._adaptive_allocation(demands)
        
        return allocation
    
    def _focused_allocation(self, demands: Dict[str, float]) -> Dict[str, float]:
        """Allocate attention in focused mode - concentrate on top items"""
        # Sort by demand and allocate to top items only
        sorted_demands = sorted(demands.items(), key=lambda x: x[1], reverse=True)
        allocation = {}
        remaining_attention = self.attention_resource.available
        
        # Focus on top 20% of items or minimum 3 items
        focus_count = max(3, len(sorted_demands) // 5)
        
        for i, (unit_id, demand) in enumerate(sorted_demands[:focus_count]):
            if remaining_attention <= 0:
                break
                
            # Allocate proportionally among focused items
            allocation_amount = min(demand * 2.0, remaining_attention / max(1, focus_count - i))
            allocation[unit_id] = allocation_amount
            remaining_attention -= allocation_amount
        
        return allocation
    
    def _distributed_allocation(self, demands: Dict[str, float]) -> Dict[str, float]:
        """Allocate attention in distributed mode - spread across all items"""
        total_demand = sum(demands.values())
        if total_demand == 0:
            return {}
        
        allocation = {}
        available = self.attention_resource.available
        
        # Proportional allocation with some minimum guarantees
        for unit_id, demand in demands.items():
            proportion = demand / total_demand
            base_allocation = available * proportion
            
            # Ensure minimum allocation for active units
            min_allocation = min(self.min_threshold, available / len(demands))
            final_allocation = max(base_allocation, min_allocation)
            
            allocation[unit_id] = final_allocation
        
        # Normalize if over-allocated
        total_allocated = sum(allocation.values())
        if total_allocated > available:
            normalization_factor = available / total_allocated
            allocation = {uid: amount * normalization_factor 
                         for uid, amount in allocation.items()}
        
        return allocation
    
    def _emergency_allocation(self, demands: Dict[str, float]) -> Dict[str, float]:
        """Emergency allocation mode - prioritize urgent items only"""
        urgent_units = {uid: demand for uid, demand in demands.items()
                       if uid in self.attention_units and 
                       self.attention_units[uid].urgency > 0.8}
        
        if not urgent_units:
            # Fall back to focused allocation if no urgent items
            return self._focused_allocation(demands)
        
        # Allocate all available attention to urgent items
        total_urgent_demand = sum(urgent_units.values())
        allocation = {}
        
        for unit_id, demand in urgent_units.items():
            proportion = demand / total_urgent_demand
            allocation[unit_id] = self.attention_resource.available * proportion
        
        return allocation
    
    def _adaptive_allocation(self, demands: Dict[str, float]) -> Dict[str, float]:
        """Adaptive allocation that switches between modes based on context"""
        # Analyze current situation
        total_demand = sum(demands.values())
        available = self.attention_resource.available
        demand_pressure = total_demand / max(available, 1.0)
        
        # Count urgent items
        urgent_count = sum(1 for uid in demands.keys() 
                          if uid in self.attention_units and 
                          self.attention_units[uid].urgency > 0.7)
        urgent_ratio = urgent_count / max(len(demands), 1)
        
        # Decide allocation strategy
        if urgent_ratio > 0.5:
            # Many urgent items - emergency mode
            return self._emergency_allocation(demands)
        elif demand_pressure > 2.0:
            # High demand pressure - focused mode
            return self._focused_allocation(demands)
        else:
            # Normal operation - distributed mode
            return self._distributed_allocation(demands)
    
    def _track_performance(self, allocation: Dict[str, float]):
        """Track allocation performance metrics"""
        performance = {
            'timestamp': time.time(),
            'total_allocated': sum(allocation.values()),
            'allocation_efficiency': self._calculate_allocation_efficiency(allocation),
            'attention_utilization': sum(allocation.values()) / self.attention_resource.total_capacity,
            'allocation_diversity': self._calculate_allocation_diversity(allocation),
            'mode': self.attention_mode.value
        }
        
        self.performance_history.append(performance)
        if len(self.performance_history) > 1000:
            self.performance_history = self.performance_history[-1000:]
    
    def _calculate_allocation_efficiency(self, allocation: Dict[str, float]) -> float:
        """Calculate how efficiently attention was allocated"""
        if not allocation:
            return 0.0
        
        total_value = 0.0
        total_cost = 0.0
        
        for unit_id, amount in allocation.items():
            if unit_id in self.attention_units:
                unit = self.attention_units[unit_id]
                value = unit.get_attention_value() * amount
                cost = amount * self.attention_price
                
                total_value += value
                total_cost += cost
        
        return total_value / max(total_cost, 0.1)
    
    def _calculate_allocation_diversity(self, allocation: Dict[str, float]) -> float:
        """Calculate diversity of attention allocation (entropy-based)"""
        if not allocation:
            return 0.0
        
        total = sum(allocation.values())
        if total == 0:
            return 0.0
        
        # Calculate normalized distribution
        distribution = [amount / total for amount in allocation.values()]
        
        # Calculate entropy
        entropy = -sum(p * np.log(p) for p in distribution if p > 0)
        
        # Normalize by maximum possible entropy
        max_entropy = np.log(len(allocation))
        return entropy / max_entropy if max_entropy > 0 else 0.0
    
    def set_attention_mode(self, mode: AttentionMode):
        """Set the attention allocation mode"""
        self.attention_mode = mode
    
    def get_attention_statistics(self) -> Dict[str, Any]:
        """Get comprehensive attention allocation statistics"""
        current_allocations = {uid: unit.allocated_attention 
                             for uid, unit in self.attention_units.items()}
        
        recent_performance = self.performance_history[-10:] if self.performance_history else []
        
        return {
            'current_allocations': current_allocations,
            'total_units': len(self.attention_units),
            'active_units': len([u for u in self.attention_units.values() 
                               if u.allocated_attention > 0]),
            'attention_utilization': sum(current_allocations.values()) / self.attention_resource.total_capacity,
            'average_efficiency': np.mean([p['allocation_efficiency'] for p in recent_performance]) if recent_performance else 0.0,
            'average_diversity': np.mean([p['allocation_diversity'] for p in recent_performance]) if recent_performance else 0.0,
            'current_mode': self.attention_mode.value,
            'attention_price': self.attention_price,
            'available_attention': self.attention_resource.available
        }
    
    def simulate_attention_trace(self, time_steps: int = 100, 
                               random_events: bool = True) -> List[Dict[str, Any]]:
        """
        Simulate attention allocation over time for analysis
        
        Args:
            time_steps: Number of simulation steps
            random_events: Whether to inject random salience/urgency changes
            
        Returns:
            List of allocation traces over time
        """
        trace = []
        
        for step in range(time_steps):
            # Inject random events if enabled
            if random_events and np.random.random() < 0.1:
                # Random salience spike
                unit_id = np.random.choice(list(self.attention_units.keys()))
                self.update_unit_salience(unit_id, np.random.normal(0.3, 0.1))
            
            if random_events and np.random.random() < 0.05:
                # Random urgency spike
                unit_id = np.random.choice(list(self.attention_units.keys()))
                self.update_unit_salience(unit_id, 0.0, np.random.normal(0.4, 0.1))
            
            # Perform allocation
            allocation = self.allocate_attention()
            
            # Record trace
            trace_entry = {
                'step': step,
                'allocation': allocation.copy(),
                'statistics': self.get_attention_statistics(),
                'mode': self.attention_mode.value
            }
            trace.append(trace_entry)
            
            # Apply temporal decay
            for unit in self.attention_units.values():
                unit.update_salience(unit.salience * (1 - self.salience_decay))
                unit.urgency *= 0.95  # Urgency decays faster
        
        return trace
    
    def export_configuration(self) -> Dict[str, Any]:
        """Export current configuration for persistence"""
        return {
            'attention_capacity': self.attention_resource.total_capacity,
            'min_threshold': self.min_threshold,
            'salience_decay': self.salience_decay,
            'urgency_amplifier': self.urgency_amplifier,
            'complexity_penalty': self.complexity_penalty,
            'focus_temperature': self.focus_temperature,
            'attention_price': self.attention_price,
            'demand_elasticity': self.demand_elasticity,
            'current_mode': self.attention_mode.value,
            'units': {
                uid: {
                    'salience': unit.salience,
                    'urgency': unit.urgency,
                    'complexity': unit.complexity,
                    'allocated_attention': unit.allocated_attention
                }
                for uid, unit in self.attention_units.items()
            }
        }
    
    def import_configuration(self, config: Dict[str, Any]):
        """Import configuration from persistence"""
        # Update engine parameters
        self.attention_resource.total_capacity = config.get('attention_capacity', 100.0)
        self.min_threshold = config.get('min_threshold', 0.1)
        self.salience_decay = config.get('salience_decay', 0.05)
        self.urgency_amplifier = config.get('urgency_amplifier', 2.0)
        self.complexity_penalty = config.get('complexity_penalty', 0.5)
        self.focus_temperature = config.get('focus_temperature', 1.0)
        self.attention_price = config.get('attention_price', 1.0)
        self.demand_elasticity = config.get('demand_elasticity', 0.8)
        
        # Set mode
        mode_str = config.get('current_mode', 'adaptive')
        self.attention_mode = AttentionMode(mode_str)
        
        # Restore units
        if 'units' in config:
            self.attention_units.clear()
            for uid, unit_data in config['units'].items():
                unit = AttentionUnit(
                    id=uid,
                    salience=unit_data.get('salience', 0.0),
                    urgency=unit_data.get('urgency', 0.0),
                    complexity=unit_data.get('complexity', 1.0),
                    allocated_attention=unit_data.get('allocated_attention', 0.0)
                )
                self.attention_units[uid] = unit