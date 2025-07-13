from tree_sitter import Node, Tree

from typing import List
import string
import random
from transformation.ast_utils import _apply_edits
from transformation.transformation import CodeTransformation

def remove_colon(tree: Tree, source_code: str, once : bool = True) -> str:
    """Removes one random colon from the source code."""
    edits = []
    return _apply_edits(source_code, []).replace(":", "", 1)
    

def remove_parentheses(tree: Tree, source_code: str, once : bool = True) -> str:
    """Removes one parenthese from the source code."""
    return _apply_edits(source_code, []).replace("(", "", 1)
    

def remove_brackets(tree: Tree, source_code: str, once : bool = True) -> str:
    """Removes brackets from the source code."""
    return _apply_edits(source_code, []).replace("[", "", 1)
    

class SyntaxErrorTransformation(CodeTransformation):
    def __init__(self, input: str, programming_language):
        super().__init__(input, programming_language)
        self.tree = None

    def check(self, **kwargs) -> bool:
        """Checks if the transformation can be applied."""
        # This transformation can always be applied
        return True
    
    def apply(self, **kwargs) -> str:
        """Applies the transformation to the input code."""
        # Randomly choose a syntax error to introduce
        number_iteration = random.randint(1, 3)
        for _ in range(number_iteration):
            # Choose a random syntax error function
            errors = [remove_colon, remove_parentheses, remove_brackets]
            error_function = random.choice(errors)
            # Apply the syntax error function
            self.input = error_function(self.tree, self.input)
        return self.input