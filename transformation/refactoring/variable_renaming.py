from transformation.refactoring.refactoring_utils import VariableQuery, variable_refactoring
import sys
from tree_sitter_language_pack import SupportedLanguage
from dataclasses import dataclass
from typing import override
sys.path.append("../")
from transformation.transformation import CodeTransformation
from transformation.ast_utils import _parse


@dataclass
class VariableRenaming(CodeTransformation):
    variable_query: VariableQuery

    def __init__(self, input: str, programming_language: SupportedLanguage):
        super().__init__(input, programming_language)
        self.tree = _parse(language=self.programming_language, program_str=self.input)
        self.variable_query = VariableQuery(self.tree.root_node)

    @override
    def check(self, **kwargs) -> bool:
        """
            Checks if the transformation can be applied to the input code.
        """
        variable_nodes = self.variable_query.find(self.tree.root_node)
        return len(variable_nodes) > 0

    @override
    def apply(self, **kwargs) -> str:
        """
            Applies the variable renaming transformation to the input code.
        """
        new_code = variable_refactoring(self.tree, self.input, once=kwargs.get('once', False))
        return new_code
    

@dataclass
class VariableRenamingSynonym(CodeTransformation):
    variable_query: VariableQuery

    def __init__(self, input: str, programming_language: SupportedLanguage):
        super().__init__(input, programming_language)
        self.tree = _parse(language=self.programming_language, program_str=self.input)
        self.variable_query = VariableQuery(self.tree.root_node)

    @override
    def check(self, **kwargs) -> bool:
        """
            Checks if the transformation can be applied to the input code.
        """
        variable_nodes = self.variable_query.find(self.tree.root_node)
        return len(variable_nodes) > 0

    @override
    def apply(self, **kwargs) -> str:
        """
            Applies the variable renaming transformation with synonyms to the input code.
        """
        new_code = variable_refactoring(self.tree, self.input, once=kwargs.get('once', False), synonym=True)
        return new_code
    
@dataclass
class VariableRenamingAntonym(CodeTransformation):
    variable_query: VariableQuery

    def __init__(self, input: str, programming_language: SupportedLanguage):
        super().__init__(input, programming_language)
        self.tree = _parse(language=self.programming_language, program_str=self.input)
        self.variable_query = VariableQuery(self.tree.root_node)

    @override
    def check(self, **kwargs) -> bool:
        """
            Checks if the transformation can be applied to the input code.
        """
        variable_nodes = self.variable_query.find(self.tree.root_node)
        return len(variable_nodes) > 0

    @override
    def apply(self, **kwargs) -> str:
        """
            Applies the variable renaming transformation with antonyms to the input code.
        """
        new_code = variable_refactoring(self.tree, self.input, once=kwargs.get('once', False), antonym=True)
        return new_code