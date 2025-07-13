import sys
sys.path.append("../")
from transformation.transformation import CodeTransformation
from transformation.ast_utils import _parse, _flatten_tree, _flatten_from_node, _apply_edits
from tree_sitter import Tree, Node
from typing import List, Dict, Set, Optional
from dataclasses import dataclass

@dataclass
class Reaction:
    """A reaction characterizes a code snippet with input and output contexts."""
    input_types: List[str]  # Types of variables used in the snippet
    output_type: str        # Return type of the statement
    node: Node             # The tree-sitter node
    compatible_vars: Dict[str, str]  # Variable mappings

class SosieTransformationBase(CodeTransformation):
    """Base class for all sosie transformations using tree-sitter."""
    
    def __init__(self, input: str, programming_language: str):
        super().__init__(input, programming_language)
        self.reactions = []
        self.variable_contexts = {}
        self._extract_program_analysis()
    
    def _extract_program_analysis(self):
        """Extract reactions and variable contexts from the program."""
        tree = _parse(self.programming_language, self.input)
        nodes = _flatten_tree(tree)
        
        # Extract reactions and variable contexts
        for node in nodes:
            if self._is_statement_node(node):
                reaction = self._create_reaction(node)
                if reaction:
                    self.reactions.append(reaction)
            
            if self._is_variable_declaration(node):
                self._update_variable_context(node)
    
    def _is_statement_node(self, node: Node) -> bool:
        """Check if node represents a statement that can be transformed."""
        statement_types = {
            "python": ["assignment", "expression_statement", "return_statement"],
            "java": ["local_variable_declaration", "expression_statement", "return_statement"],
            "c": ["declaration", "expression_statement", "return_statement"]
        }
        return node.type in statement_types.get(self.programming_language.lower(), [])
    
    def _is_variable_declaration(self, node: Node) -> bool:
        """Check if node declares a variable."""
        var_types = {
            "python": ["assignment"],
            "java": ["local_variable_declaration", "parameter"],
            "c": ["declaration", "parameter_declaration"]
        }
        return node.type in var_types.get(self.programming_language.lower(), [])
    
    def _create_reaction(self, node: Node) -> Optional[Reaction]:
        """Create a reaction from a statement node."""
        input_types = self._extract_input_types(node)
        output_type = self._extract_output_type(node)
        compatible_vars = self._extract_variable_names(node)
        
        return Reaction(
            input_types=input_types,
            output_type=output_type,
            node=node,
            compatible_vars=compatible_vars
        )
    
    def _extract_input_types(self, node: Node) -> List[str]:
        """Extract input variable types from a node."""
        types = []
        child_nodes = _flatten_from_node(node)
        
        for child in child_nodes:
            if child.type == "identifier":
                # Simplified type extraction
                var_name = child.text.decode('utf-8')
                types.append(self._get_variable_type(var_name))
        
        return list(set(types))  # Remove duplicates
    
    def _extract_output_type(self, node: Node) -> str:
        """Extract output type from a node."""
        if "return" in node.text.decode('utf-8'):
            return "return_type"
        elif "=" in node.text.decode('utf-8'):
            return "assignment_type"
        else:
            return "void"
    
    def _extract_variable_names(self, node: Node) -> Dict[str, str]:
        """Extract variable names and their types from a node."""
        variables = {}
        child_nodes = _flatten_from_node(node)
        
        for child in child_nodes:
            if child.type == "identifier":
                var_name = child.text.decode('utf-8')
                var_type = self._get_variable_type(var_name)
                variables[var_name] = var_type
        
        return variables
    
    def _get_variable_type(self, var_name: str) -> str:
        """Get the type of a variable (simplified)."""
        # This is a simplified type system
        if var_name.endswith('_count') or var_name.endswith('_num'):
            return "int"
        elif var_name.endswith('_name') or var_name.endswith('_str'):
            return "string"
        elif var_name.endswith('_flag') or var_name.endswith('_bool'):
            return "boolean"
        else:
            return "unknown"
    
    def _update_variable_context(self, node: Node):
        """Update variable context based on declarations."""
        line_num = node.start_point[0]  # Row number
        if line_num not in self.variable_contexts:
            self.variable_contexts[line_num] = set()
        
        variables = self._extract_variable_names(node)
        self.variable_contexts[line_num].update(variables.keys())
    
    def _is_covered_by_tests(self, node: Node) -> bool:
        """Always return True since we don't care about test coverage."""
        return True
    
    def find_transplantation_points(self) -> List[Node]:
        """Find valid transplantation points (all non-critical statements)."""
        tree = _parse(self.programming_language, self.input)
        nodes = _flatten_tree(tree)
        
        transplantation_points = []
        for node in nodes:
            if (self._is_statement_node(node) and 
                not self._is_critical_statement(node)):
                transplantation_points.append(node)
        
        return transplantation_points
    
    def _is_critical_statement(self, node: Node) -> bool:
        """Check if statement is critical (shouldn't be deleted/replaced)."""
        critical_types = {
            "python": ["return_statement", "break_statement", "continue_statement"],
            "java": ["return_statement", "break_statement", "continue_statement"],
            "c": ["return_statement", "break_statement", "continue_statement"]
        }
        
        node_text = node.text.decode('utf-8').lower()
        critical_keywords = ["return", "break", "continue", "throw", "exit"]
        
        return (node.type in critical_types.get(self.programming_language.lower(), []) or
                any(keyword in node_text for keyword in critical_keywords))