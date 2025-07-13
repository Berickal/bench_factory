from transformation.operator_mutation.operator_utils import ComparisonOperator, replace_comparison_operators
import sys
from tree_sitter_language_pack import SupportedLanguage
from dataclasses import dataclass
from typing import override
sys.path.append("../")
from transformation.transformation import CodeTransformation
from transformation.ast_utils import _parse

@dataclass
class LogicalMutation(CodeTransformation):
    """A transformation that mutates logical operators in the code."""
    
    def __init__(self, input: str, programming_language: SupportedLanguage):
        super().__init__(input, programming_language)

    @override
    def check(self, **kwargs) -> bool:
        """Checks if the transformation can be applied to the input code."""
        tree = _parse(language=self.programming_language, program_str=self.input)
        return len(ComparisonOperator(tree.root_node).find()) > 0

    @override
    def apply(self, **kwargs) -> str:
        """Applies the logical mutation transformation to the input code."""
        tree = _parse(language=self.programming_language, program_str=self.input)
        new_code = replace_comparison_operators(tree, self.input)
        return new_code