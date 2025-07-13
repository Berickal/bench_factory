from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import override
from tree_sitter_language_pack import SupportedLanguage

@dataclass
class Transformation(ABC):
    """
        Abstract representation of a transformation.
    """
    input : str

    def __init__(self, input : str):
        """
            Constructor of the transformation

            Args:
                name (str) : the name of the transformation
                input (str) : the input to transform
        """
        self.input = input

    @abstractmethod
    def check(self, input : str, **kwargs) -> bool:
        """
            Abstract method allowing to check whether a transformation is
            applicable given an input.

            Args:
                input (str) : the input to check : can be code / natural text ...
            Return :
                bool : the applicability of the transformation
        """
        pass

    @abstractmethod
    def apply(self, input : str, **kwargs) -> str:
        """
            Abstract method to apply the transformation on the given input

            Args:
                input (str) : The input to transform
            Return
                str : The transformed input
        """
        pass


@dataclass
class CodeTransformation(Transformation):
    """
        Abstract representation of a code transformation.
    """
    programming_language : SupportedLanguage

    def __init__(self, input : str, programming_language : SupportedLanguage):
        """
            Constructor of the code transformation

            Args:
                name (str) : the name of the transformation
                input (str) : the input to transform
                programming_language (SupportedLanguage) : the programming language of the input
        """
        super().__init__(input)
        self.programming_language = programming_language

    @override
    @abstractmethod
    def check(self, **kwargs) -> bool:
        """
            Abstract method allowing to check whether a code transformation is
            applicable given an input.

            Args:
                input (str) : the input to check
            Return :
                bool : the applicability of the transformation
        """
        pass

    @override
    @abstractmethod
    def apply(self, **kwargs) -> str:
        """
            Abstract method to apply the code transformation on the given input

            Args:
                input (str) : The input to transform
            Return
                str : The transformed input
        """
        pass