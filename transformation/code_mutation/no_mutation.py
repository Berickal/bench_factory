import sys
sys.path.append("../")
from transformation.transformation import CodeTransformation
from typing import override

class NoMutation(CodeTransformation):
    """
    A no-operation transformation that does not change the code.
    This is used as a placeholder when no mutation is needed.
    """

    @override
    def apply(self, **kwargs) -> str:
        """
        Returns the original code without any modifications.
        
        Args:
            code (str): The original code to be transformed.
        
        Returns:
            str: The original code unchanged.
        """
        return self.input
    
    @override
    def check(self, **kwargs) -> bool:
        """
            Checks if the transformation is applicable.
            This transformation always returns True, indicating that it can be applied
            regardless of the input.

            Args:
                input (str): The input code to check.
                
            Returns:
                bool: Always returns True, indicating the transformation can be applied.
        """
        return True