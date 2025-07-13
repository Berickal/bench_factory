from tree_sitter import Node, Tree
import sys
sys.path.append("../")
from transformation.transformation import CodeTransformation

from dataclasses import dataclass
from typing import List, override, Optional
from transformation.ast_utils import ASTQuery, _flatten_from_node, _apply_edits, _parse

class NumberQuery(ASTQuery):
    """A query that matches number nodes in the AST."""

    def query(self, node : Optional[Node] = None) -> bool:
        """Checks if the node is a number."""
        if node is None:
            node = self.root
        if "number" in node.type or "integer" in node.type:
            return True

    def find(self) -> List[Node]:
        """Finds all number nodes in the AST."""
        nodes = _flatten_from_node(self.root)
        result = []
        for n in nodes:
            if self.query(n):
                result.append(n)
        return result


@dataclass
class NumberTransformer(CodeTransformation):
    """A transformation that replaces numbers in the code with a placeholder."""
    
    def __init__(self, input: str, programming_language):
        super().__init__(input, programming_language)
        tree = _parse(language=self.programming_language, program_str=self.input)
        self.query = NumberQuery(root=tree.root_node)

    @override
    def check(self, **kwargs) -> bool:
        """Checks if the transformation can be applied."""
        return bool(self.query.find())

    @override
    def apply(self, **kwargs) -> str:
        """Applies the transformation to the input code."""
        number_nodes = self.query.find()
        edits = []  
        num = int(number_nodes[0].text.decode('utf-8'))
        num += 1
        edits.append({
            'start_byte': number_nodes[0].start_byte,
            'end_byte': number_nodes[0].end_byte,
            'new_text': str(num)
        })
        
        return _apply_edits(self.input, edits)
    