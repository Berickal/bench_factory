from transformation.refactoring.refactoring_utils import ClassQuery, class_refactoring
import sys
from tree_sitter_language_pack import SupportedLanguage
from dataclasses import dataclass
from typing import override
sys.path.append("../")
from transformation.transformation import CodeTransformation
from transformation.ast_utils import _parse

@dataclass
class ClassRenaming(CodeTransformation):
    class_query: ClassQuery

    def __init__(self, input: str, programming_language: SupportedLanguage):
        super().__init__(input, programming_language)
        self.tree = _parse(language=self.programming_language, program_str=self.input)
        self.class_query = ClassQuery(self.tree.root_node)

    @override
    def check(self, **kwargs) -> bool:
        """
            Checks if the transformation can be applied to the input code.
        """
        class_nodes = self.class_query.find(self.tree.root_node)
        return len(class_nodes) > 0

    @override
    def apply(self, **kwargs) -> str:
        """
            Applies the class renaming transformation to the input code.
        """
        new_code = class_refactoring(self.tree, self.input, once=kwargs.get('once', False))
        return new_code