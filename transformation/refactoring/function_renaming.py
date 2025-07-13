from transformation.refactoring.refactoring_utils import FunctionQuery, function_refactoring
import sys
from tree_sitter_language_pack import SupportedLanguage
from dataclasses import dataclass
from typing import override
sys.path.append("../")
from transformation.transformation import CodeTransformation
from transformation.ast_utils import _parse


@dataclass
class FunctionRenaming(CodeTransformation):
    function_query: FunctionQuery

    def __init__(self, input: str, programming_language: SupportedLanguage):
        super().__init__(input, programming_language)
        self.tree = _parse(language=self.programming_language, program_str=self.input)
        self.function_query = FunctionQuery(self.tree.root_node)

    @override
    def check(self, **kwargs) -> bool:
        """Checks if the transformation can be applied to the input code."""
        function_nodes = self.function_query.find(self.tree.root_node)
        return len(function_nodes) > 0
    
    @override
    def apply(self, **kwargs) -> str:
        """Applies the function renaming transformation to the input code."""
        new_code = function_refactoring(self.tree, self.input)
        return new_code