from tree_sitter import Tree, Node
from tree_sitter_language_pack import get_parser, SupportedLanguage
import ast
import parso
from dataclasses import dataclass
from typing import List, Dict

from enum import Enum
from abc import abstractmethod, ABC

def _parse(language: SupportedLanguage, program_str: str, encoding="utf-8") -> Tree:
    """Returns a tree-sitter representation of a program string.

    Args:
        language (str): A programming language supported by
            tree_sitter_language_pack.
        program_str (str): A string representation of a program.
        encoding (str, optional): The encoding of program_str. Defaults to
            "utf-8".

    Returns:
        Tree: A tree-sitter representation of the given program string.
    """
    parser = get_parser(language)
    return parser.parse(bytes(program_str, encoding=encoding))

def _flatten_tree(tree: Tree) -> List[Node]:
    """Returns a flattened list of all node types in the tree.

    Args:
        tree (Tree): A tree-sitter representation of a program.
    Returns:
        List[Node]: A flattened list of all node types in the
            tree.
    """
    def _flatten(node: Node, nodes: List[Node]):
        nodes.append(node)
        for child in node.children:
            _flatten(child, nodes)

    nodes = []
    _flatten(tree.root_node, nodes)
    return nodes

def _flatten_from_node(node: Node) -> List[Node]:
    """Returns a flattened list of all nodes starting from a given node.

    Args:
        node (Node): A tree-sitter Node to start flattening from.
    Returns:
        List[Node]: A flattened list of all nodes in the subtree.
    """
    def _flatten(node: Node, nodes: List[Node]):
        nodes.append(node)
        for child in node.children:
            _flatten(child, nodes)

    nodes = []
    _flatten(node, nodes)
    return nodes

def _apply_edits(source_code: str, edits: List[Dict]) -> str:
    """Apply a list of edits to source code.
    
    Args:
        source_code (str): The original source code
        edits (List[Dict]): List of edits with start_byte, end_byte, new_text
        
    Returns:
        str: Modified source code
    """
    sorted_edits = sorted(edits, key=lambda x: x['start_byte'], reverse=True)
    
    result = source_code
    for edit in sorted_edits:
        start = edit['start_byte']
        end = edit['end_byte']
        new_text = edit['new_text']
        result = result[:start] + new_text + result[end:]
    
    return result

@dataclass
class ASTQuery(ABC):
    root: Node

    @abstractmethod
    def query(self, node: Node) -> bool:
        """Returns True if the node matches the query."""
        pass

    @abstractmethod
    def find(self, node: Node) -> List[Node]:
        """Returns a list of nodes that match the query."""
        pass


