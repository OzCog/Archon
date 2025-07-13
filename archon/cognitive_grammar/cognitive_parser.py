"""
Cognitive Grammar Parser

Parses agentic task requests and decomposes them into structured cognitive 
grammar patterns for hypergraph representation.
"""

import re
import json
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class GrammarElementType(Enum):
    """Types of cognitive grammar elements"""
    AGENT = "agent"
    TASK = "task" 
    RESOURCE = "resource"
    WORKFLOW = "workflow"
    KNOWLEDGE = "knowledge"
    CONSTRAINT = "constraint"


@dataclass
class GrammarElement:
    """A parsed cognitive grammar element"""
    element_type: GrammarElementType
    name: str
    attributes: Dict[str, Any]
    relationships: List[str]
    semantic_depth: int = 1
    

class CognitiveGrammarParser:
    """
    Parses agentic task requests into cognitive grammar patterns.
    Extracts semantic elements and their relationships for hypergraph construction.
    """
    
    def __init__(self):
        self.grammar_patterns = self._initialize_patterns()
        self.context_stack = []
        
    def _initialize_patterns(self) -> Dict[str, List[str]]:
        """Initialize regex patterns for cognitive element detection"""
        return {
            'agent_patterns': [
                r'(?:create|build|implement)\s+(?:an?\s+)?(\w+)\s+agent',
                r'agent\s+(?:that|which)\s+(\w+)',
                r'(\w+)\s+specialist\s+agent'
            ],
            'task_patterns': [
                r'(?:perform|execute|do)\s+(\w+)',
                r'task\s+(?:of|for)\s+(\w+)',
                r'need\s+to\s+(\w+)'
            ],
            'resource_patterns': [
                r'(?:using|with|via)\s+(\w+)',
                r'access\s+(?:to\s+)?(\w+)',
                r'integrate\s+(?:with\s+)?(\w+)'
            ],
            'workflow_patterns': [
                r'workflow\s+(?:for|that)\s+(\w+)',
                r'process\s+(?:of|for)\s+(\w+)',
                r'pipeline\s+(?:for|that)\s+(\w+)'
            ],
            'knowledge_patterns': [
                r'knowledge\s+(?:about|of)\s+(\w+)',
                r'information\s+(?:about|on)\s+(\w+)',
                r'data\s+(?:about|from)\s+(\w+)'
            ],
            'constraint_patterns': [
                r'must\s+(\w+)',
                r'should\s+(\w+)',
                r'required\s+to\s+(\w+)'
            ]
        }
    
    def parse(self, text: str) -> List[GrammarElement]:
        """
        Parse input text into cognitive grammar elements
        
        Args:
            text: Input agentic task description
            
        Returns:
            List of parsed grammar elements
        """
        elements = []
        
        # Clean and normalize text
        normalized_text = self._normalize_text(text)
        
        # Extract elements by type
        for element_type in GrammarElementType:
            pattern_key = f"{element_type.value}_patterns"
            if pattern_key in self.grammar_patterns:
                extracted_elements = self._extract_elements(
                    normalized_text, 
                    element_type, 
                    self.grammar_patterns[pattern_key]
                )
                elements.extend(extracted_elements)
        
        # Identify relationships between elements
        elements = self._identify_relationships(elements, normalized_text)
        
        # Calculate semantic depth
        elements = self._calculate_semantic_depth(elements)
        
        return elements
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for parsing"""
        # Convert to lowercase
        text = text.lower()
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters except necessary ones
        text = re.sub(r'[^\w\s\-_]', ' ', text)
        return text.strip()
    
    def _extract_elements(self, text: str, element_type: GrammarElementType, patterns: List[str]) -> List[GrammarElement]:
        """Extract elements of a specific type using patterns"""
        elements = []
        
        for pattern in patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                if match.groups():
                    element_name = match.group(1)
                    # Extract context around the match
                    start = max(0, match.start() - 50)
                    end = min(len(text), match.end() + 50)
                    context = text[start:end]
                    
                    element = GrammarElement(
                        element_type=element_type,
                        name=element_name,
                        attributes={
                            'context': context,
                            'position': match.start(),
                            'confidence': self._calculate_confidence(match, context)
                        },
                        relationships=[]
                    )
                    elements.append(element)
        
        return self._deduplicate_elements(elements)
    
    def _calculate_confidence(self, match: re.Match, context: str) -> float:
        """Calculate confidence score for element extraction"""
        # Simple confidence based on context richness
        context_words = len(context.split())
        base_confidence = 0.7
        context_bonus = min(0.3, context_words * 0.01)
        return base_confidence + context_bonus
    
    def _deduplicate_elements(self, elements: List[GrammarElement]) -> List[GrammarElement]:
        """Remove duplicate elements based on name and type"""
        seen = set()
        unique_elements = []
        
        for element in elements:
            key = (element.element_type, element.name)
            if key not in seen:
                seen.add(key)
                unique_elements.append(element)
        
        return unique_elements
    
    def _identify_relationships(self, elements: List[GrammarElement], text: str) -> List[GrammarElement]:
        """Identify relationships between elements"""
        for i, element in enumerate(elements):
            for j, other_element in enumerate(elements):
                if i != j:
                    relationship = self._find_relationship(element, other_element, text)
                    if relationship:
                        element.relationships.append(f"{relationship}:{other_element.name}")
        
        return elements
    
    def _find_relationship(self, element1: GrammarElement, element2: GrammarElement, text: str) -> Optional[str]:
        """Find relationship between two elements"""
        # Simple proximity-based relationship detection
        pos1 = element1.attributes.get('position', 0)
        pos2 = element2.attributes.get('position', 0)
        
        distance = abs(pos1 - pos2)
        
        if distance < 100:  # Elements are close in text
            if element1.element_type == GrammarElementType.AGENT and element2.element_type == GrammarElementType.TASK:
                return "performs"
            elif element1.element_type == GrammarElementType.TASK and element2.element_type == GrammarElementType.RESOURCE:
                return "uses"
            elif element1.element_type == GrammarElementType.WORKFLOW and element2.element_type == GrammarElementType.AGENT:
                return "orchestrates"
            else:
                return "relates_to"
        
        return None
    
    def _calculate_semantic_depth(self, elements: List[GrammarElement]) -> List[GrammarElement]:
        """Calculate semantic depth based on relationships and complexity"""
        for element in elements:
            # Base depth
            depth = 1
            
            # Add depth based on relationships
            depth += len(element.relationships) * 0.5
            
            # Add depth based on context complexity
            context = element.attributes.get('context', '')
            context_complexity = len(context.split()) / 10
            depth += context_complexity
            
            # Add depth based on element type
            type_depths = {
                GrammarElementType.AGENT: 3,
                GrammarElementType.WORKFLOW: 4,
                GrammarElementType.TASK: 2,
                GrammarElementType.RESOURCE: 1,
                GrammarElementType.KNOWLEDGE: 2,
                GrammarElementType.CONSTRAINT: 1
            }
            depth += type_depths.get(element.element_type, 1)
            
            element.semantic_depth = int(min(depth, 10))  # Cap at 10
        
        return elements
    
    def extract_scheme_patterns(self, elements: List[GrammarElement]) -> Dict[str, Any]:
        """
        Convert parsed elements to Scheme-like hypergraph patterns
        
        Args:
            elements: Parsed grammar elements
            
        Returns:
            Scheme-style hypergraph representation
        """
        scheme_graph = {
            'nodes': [],
            'links': [],
            'meta': {
                'total_elements': len(elements),
                'complexity_score': self._calculate_complexity_score(elements)
            }
        }
        
        # Convert elements to nodes
        for element in elements:
            node = {
                'id': f"{element.element_type.value}_{element.name}",
                'type': element.element_type.value,
                'name': element.name,
                'semantic_depth': element.semantic_depth,
                'attributes': element.attributes
            }
            scheme_graph['nodes'].append(node)
        
        # Convert relationships to links
        link_id = 0
        for element in elements:
            for relationship in element.relationships:
                rel_type, target = relationship.split(':', 1)
                link = {
                    'id': f"link_{link_id}",
                    'source': f"{element.element_type.value}_{element.name}",
                    'target_name': target,
                    'relationship_type': rel_type,
                    'weight': 1.0
                }
                scheme_graph['links'].append(link)
                link_id += 1
        
        return scheme_graph
    
    def _calculate_complexity_score(self, elements: List[GrammarElement]) -> float:
        """Calculate overall complexity score of the parsed grammar"""
        if not elements:
            return 0.0
        
        total_depth = sum(e.semantic_depth for e in elements)
        total_relationships = sum(len(e.relationships) for e in elements)
        
        complexity = (total_depth / len(elements)) + (total_relationships / len(elements))
        return min(complexity, 10.0)  # Cap at 10.0