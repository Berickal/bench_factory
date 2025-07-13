from tree_sitter import Node, Tree
from transformation.ast_utils import ASTQuery, _flatten_from_node, _apply_edits, _parse
from typing import List, Optional
from enum import Enum
import random


class ArithmeticOperatorType(Enum):
    """Enumeration of arithmetic operators with their string representations."""
    ADDITION = "+"
    SUBTRACTION = "-"
    MULTIPLICATION = "*"
    DIVISION = "/"
    MODULUS = "%"
    XOR = "^"
    BITWISE_AND = "&"
    BITWISE_OR = "|"
    ADDITION_ASSIGNMENT = "+="
    SUBTRACTION_ASSIGNMENT = "-="
    MULTIPLICATION_ASSIGNMENT = "*="
    DIVISION_ASSIGNMENT = "/="
    MODULUS_ASSIGNMENT = "%="
    BITWISE_XOR_ASSIGNMENT = "^="
    BITWISE_AND_ASSIGNMENT = "&="
    BITWISE_OR_ASSIGNMENT = "|="


class LogicalOperatorType(Enum):
    """Enumeration of logical operators with their string representations."""
    AND_SYMBOLIC = "&&"
    AND_WORD = "and"
    OR_SYMBOLIC = "||" 
    OR_WORD = "or"
    NOT_SYMBOLIC = "!"
    NOT_WORD = "not"


class ArithmeticOperator(ASTQuery):
    """
    A query that matches arithmetic operator nodes in the AST.
    This class extends the ASTQuery to specifically identify arithmetic
    operators such as addition, subtraction, multiplication, division, and modulus.
    """

    def query(self, node : Optional[Node]) -> bool:
        """Checks if the node is an arithmetic operator."""
        # Check for specific operator node types in tree-sitter
        if node is None:
            node = self.root
        arithmetic_node_types = [
            "+", "-", "*", "/", "%", "^", "&", "|",
            "binary_operator", "augmented_assignment",
            "update_expression"
        ]
        
        if node.type in arithmetic_node_types:
            return True
            
        # Also check the actual text content for operators
        if node.text:
            text = node.text.decode('utf-8').strip()
            arithmetic_symbols = [op.value for op in ArithmeticOperatorType]
            if text in arithmetic_symbols:
                return True
                
        return False
    
    def find(self) -> List[Node]:
        """Finds all arithmetic operator nodes in the AST."""
        nodes = _flatten_from_node(self.root)
        result = []
        for n in nodes:
            if self.query(n):
                result.append(n)
        return result
    

class LogicalOperator(ASTQuery):
    """
    A query that matches logical operator nodes in the AST.
    This class extends the ASTQuery to specifically identify logical
    operators such as AND, OR, and NOT.
    """

    def query(self, node : Optional[Node]) -> bool:
        """Checks if the node is a logical operator (AND/OR)."""
        # Check for specific logical operator node types
        if node is None:
            node = self.root
        logical_node_types = [
            "&&", "||", "and", "or",
            "boolean_operator", "binary_operator"
        ]
        
        if node.type in logical_node_types:
            return True
            
        # Check the actual text content
        if node.text:
            text = node.text.decode('utf-8').strip()
            logical_symbols = ["&&", "||", "and", "or"]
            if text in logical_symbols:
                return True
                
        return False
    
    def query_not(self, node : Optional[Node]) -> bool:
        """Checks if the node is a NOT operator."""
        not_node_types = ["!", "not", "unary_operator"]
        
        if node is None:
            node = self.root
        if node.type in not_node_types:
            return True
            
        if node.text:
            text = node.text.decode('utf-8').strip()
            if text in ["!", "not"]:
                return True
                
        return False
    
    def find(self) -> List[Node]:
        """Finds all logical operator nodes in the AST (AND/OR)."""
        nodes = _flatten_from_node(self.root)
        result = []
        for n in nodes:
            if self.query(n):
                result.append(n)
        return result
    
    def find_not(self) -> List[Node]:
        """Finds all NOT operator nodes in the AST."""
        nodes = _flatten_from_node(self.root)
        result = []
        for n in nodes:
            if self.query_not(n):
                result.append(n)
        return result


class ComparisonOperator(ASTQuery):
    """
    A query that matches comparison operator nodes in the AST.
    """
    
    def query(self, node : Optional[Node]) -> bool:
        """Checks if the node is a comparison operator."""
        comparison_node_types = [
            "==", "!=", "<", ">", "<=", ">=",
            "comparison_operator", "binary_operator"
        ]

        if node is None:
            node = self.root 
        if node.type in comparison_node_types:
            return True
            
        # Check the actual text content
        if node.text:
            text = node.text.decode('utf-8').strip()
            comparison_symbols = ["==", "!=", "<", ">", "<=", ">="]
            if text in comparison_symbols:
                return True
                
        return False
    
    def find(self) -> List[Node]:
        """Finds all comparison operator nodes in the AST."""
        nodes = _flatten_from_node(self.root)
        result = []
        for n in nodes:
            if self.query(n):
                result.append(n)
        return result


def replace_arithmetic_operators(tree: Tree, source_code: str, once: bool = False) -> str:
    """
    Replace arithmetic operators in the AST with randomly chosen operators.
    
    Args:
        tree (Tree): The tree-sitter Tree object representing the AST.
        source_code (str): The original source code.
        once (bool): If True, replace only the first occurrence of each operator.
        
    Returns:
        str: Modified source code with arithmetic operators replaced.
    """
    query = ArithmeticOperator(tree.root_node)
    edits = []
    operators_replaced = set()
    arithmetic_replacements = {
        "+": ["-", "*", "/", "%", "+"],
        "-": ["+", "*", "/", "%", "-"],
        "*": ["+", "-", "/", "%", "*"],
        "/": ["+", "-", "*", "%", "/"],
        "%": ["+", "-", "*", "/", "%"],
        "^": ["&", "|", "^"],
        "&": ["^", "|", "&"],
        "|": ["^", "&", "|"],
        "+=": ["-=", "*=", "/=", "%=", "+="],
        "-=": ["+=", "*=", "/=", "%=", "-="],
        "*=": ["+=", "-=", "/=", "%=", "*="],
        "/=": ["+=", "-=", "*=", "%=", "/="],
        "%=": ["+=", "-=", "*=", "/=", "%="],
        "^=": ["&=", "|=", "^="],
        "&=": ["^=", "|=", "&="],
        "|=": ["^=", "&=", "|="]
    }
    
    for node in query.find():
        if not node.text:
            continue
            
        old_operator = node.text.decode('utf-8').strip()

        if once and old_operator in operators_replaced:
            continue
            
        if old_operator in arithmetic_replacements:
            new_operator = random.choice(arithmetic_replacements[old_operator])
            
            edits.append({
                'start_byte': node.start_byte,
                'end_byte': node.end_byte,
                'new_text': new_operator
            })
            
            operators_replaced.add(old_operator)
            
            if once:
                break
    
    return _apply_edits(source_code, edits)


def replace_logical_operators(tree: Tree, source_code: str, once: bool = False) -> str:
    """
    Replace logical operators in the AST with the opposite logical operators.
    
    Args:
        tree (Tree): The tree-sitter Tree object representing the AST.
        source_code (str): The original source code.
        once (bool): If True, replace only the first occurrence of each operator.
        
    Returns:
        str: Modified source code with logical operators replaced.
    """
    query = LogicalOperator(tree.root_node)
    edits = []
    operators_replaced = set()
    
    logical_replacements = {
        "&&": "||",
        "||": "&&",
        "and": "or",
        "or": "and"
    }

    for node in query.find():
        if not node.text:
            continue
            
        old_operator = node.text.decode('utf-8').strip()
        if once and old_operator in operators_replaced:
            continue
            
        if old_operator in logical_replacements:
            new_operator = logical_replacements[old_operator]
            
            edits.append({
                'start_byte': node.start_byte,
                'end_byte': node.end_byte,
                'new_text': new_operator
            })
            
            operators_replaced.add(old_operator)
            
            if once:
                break

    for node in query.find_not(tree.root_node):
        if not node.text:
            continue
            
        old_operator = node.text.decode('utf-8').strip()
        
        if once and old_operator in operators_replaced:
            continue
            
        if old_operator in ["!", "not"]:
            # Remove the NOT operator
            edits.append({
                'start_byte': node.start_byte,
                'end_byte': node.end_byte,
                'new_text': ""  # Remove the operator
            })
            
            operators_replaced.add(old_operator)
            
            if once:
                break
    
    return _apply_edits(source_code, edits)


def replace_comparison_operators(tree: Tree, source_code: str, once: bool = False) -> str:
    """
    Replace comparison operators in the AST with different comparison operators.
    
    Args:
        tree (Tree): The tree-sitter Tree object representing the AST.
        source_code (str): The original source code.
        once (bool): If True, replace only the first occurrence of each operator.
        
    Returns:
        str: Modified source code with comparison operators replaced.
    """
    query = ComparisonOperator(tree.root_node)
    edits = []
    operators_replaced = set()
    
    comparison_replacements = {
        "==": ["!=", "<", ">", "<=", ">="],
        "!=": ["==", "<", ">", "<=", ">="],
        "<": [">", "<=", ">=", "==", "!="],
        ">": ["<", "<=", ">=", "==", "!="],
        "<=": [">=", "<", ">", "==", "!="],
        ">=": ["<=", "<", ">", "==", "!="]
    }
    
    for node in query.find():
        if not node.text:
            continue
            
        old_operator = node.text.decode('utf-8').strip()
        
        if once and old_operator in operators_replaced:
            continue
            
        if old_operator in comparison_replacements:
            new_operator = random.choice(comparison_replacements[old_operator])
            
            edits.append({
                'start_byte': node.start_byte,
                'end_byte': node.end_byte,
                'new_text': new_operator
            })
            
            operators_replaced.add(old_operator)
            
            if once:
                break
    
    return _apply_edits(source_code, edits)
