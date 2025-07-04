"""
Test Cognitive Grammar Parser

Tests for parsing agentic task requests into cognitive grammar patterns.
"""

import pytest
import numpy as np
from archon.cognitive_grammar.cognitive_parser import (
    CognitiveGrammarParser, GrammarElement, GrammarElementType
)


class TestCognitiveGrammarParser:
    
    def setup_method(self):
        """Setup test parser"""
        self.parser = CognitiveGrammarParser()
    
    def test_parser_initialization(self):
        """Test parser initializes correctly"""
        assert len(self.parser.grammar_patterns) > 0
        assert 'agent_patterns' in self.parser.grammar_patterns
        assert 'task_patterns' in self.parser.grammar_patterns
    
    def test_parse_simple_agent_request(self):
        """Test parsing simple agent creation request"""
        text = "Create a coding agent that writes Python code"
        elements = self.parser.parse(text)
        
        assert len(elements) > 0
        agent_elements = [e for e in elements if e.element_type == GrammarElementType.AGENT]
        assert len(agent_elements) > 0
        
        # Should find "coding" agent
        coding_agent = next((e for e in agent_elements if "coding" in e.name), None)
        assert coding_agent is not None
        assert coding_agent.semantic_depth > 0
    
    def test_parse_complex_workflow(self):
        """Test parsing complex workflow with multiple elements"""
        text = """Create an intelligent workflow that uses a reasoning agent 
                 to analyze tasks, a coding agent to implement solutions, 
                 and integrate with external knowledge databases"""
        
        elements = self.parser.parse(text)
        
        # Should find multiple element types
        element_types = {e.element_type for e in elements}
        assert GrammarElementType.AGENT in element_types
        assert GrammarElementType.WORKFLOW in element_types or GrammarElementType.TASK in element_types
        
        # Should have relationships
        has_relationships = any(len(e.relationships) > 0 for e in elements)
        assert has_relationships
    
    def test_relationship_extraction(self):
        """Test extraction of relationships between elements"""
        text = "The coding agent uses the knowledge database to generate code"
        elements = self.parser.parse(text)
        
        # Find agent and resource elements
        agents = [e for e in elements if e.element_type == GrammarElementType.AGENT]
        resources = [e for e in elements if e.element_type in [GrammarElementType.RESOURCE, GrammarElementType.KNOWLEDGE]]
        
        if agents and resources:
            # Should have some relationships
            agent_relationships = sum(len(e.relationships) for e in agents)
            assert agent_relationships > 0
    
    def test_semantic_depth_calculation(self):
        """Test semantic depth calculation"""
        simple_text = "Create agent"
        complex_text = "Create an advanced multi-modal reasoning agent that integrates with external knowledge bases and uses sophisticated attention mechanisms"
        
        simple_elements = self.parser.parse(simple_text)
        complex_elements = self.parser.parse(complex_text)
        
        if simple_elements and complex_elements:
            simple_depth = max(e.semantic_depth for e in simple_elements)
            complex_depth = max(e.semantic_depth for e in complex_elements)
            assert complex_depth >= simple_depth
    
    def test_scheme_pattern_extraction(self):
        """Test conversion to Scheme-like patterns"""
        text = "Create a reasoning agent that performs analysis tasks using data resources"
        elements = self.parser.parse(text)
        scheme_graph = self.parser.extract_scheme_patterns(elements)
        
        assert 'nodes' in scheme_graph
        assert 'links' in scheme_graph
        assert 'meta' in scheme_graph
        
        assert len(scheme_graph['nodes']) > 0
        assert scheme_graph['meta']['total_elements'] == len(elements)
        
        # Nodes should have required fields
        for node in scheme_graph['nodes']:
            assert 'id' in node
            assert 'type' in node
            assert 'semantic_depth' in node
    
    def test_empty_input(self):
        """Test handling of empty input"""
        elements = self.parser.parse("")
        assert elements == []
        
        scheme_graph = self.parser.extract_scheme_patterns([])
        assert scheme_graph['meta']['total_elements'] == 0
    
    def test_normalization(self):
        """Test text normalization"""
        text = "CREATE AN   AGENT!!!"
        normalized = self.parser._normalize_text(text)
        assert normalized == "create an agent"
    
    def test_confidence_calculation(self):
        """Test confidence calculation for extractions"""
        text = "Create a sophisticated AI agent for complex reasoning tasks"
        elements = self.parser.parse(text)
        
        for element in elements:
            assert 'confidence' in element.attributes
            confidence = element.attributes['confidence']
            assert 0.0 <= confidence <= 1.0
    
    def test_complexity_score(self):
        """Test complexity score calculation"""
        simple_elements = self.parser.parse("Create agent")
        complex_elements = self.parser.parse("Create sophisticated multi-agent system with distributed reasoning and external integrations")
        
        simple_scheme = self.parser.extract_scheme_patterns(simple_elements)
        complex_scheme = self.parser.extract_scheme_patterns(complex_elements)
        
        simple_complexity = simple_scheme['meta']['complexity_score']
        complex_complexity = complex_scheme['meta']['complexity_score']
        
        assert complex_complexity >= simple_complexity


if __name__ == "__main__":
    # Run basic tests without pytest
    test = TestCognitiveGrammarParser()
    test.setup_method()
    
    print("Testing parser initialization...")
    test.test_parser_initialization()
    print("✓ Parser initialization test passed")
    
    print("Testing simple agent request parsing...")
    test.test_parse_simple_agent_request()
    print("✓ Simple agent request test passed")
    
    print("Testing complex workflow parsing...")
    test.test_parse_complex_workflow()
    print("✓ Complex workflow test passed")
    
    print("Testing scheme pattern extraction...")
    test.test_scheme_pattern_extraction()
    print("✓ Scheme pattern extraction test passed")
    
    print("Testing empty input handling...")
    test.test_empty_input()
    print("✓ Empty input test passed")
    
    print("\nAll cognitive grammar parser tests passed!")