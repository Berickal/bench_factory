import sys
sys.path.append("../")
from transformation.transformation import Transformation

#Refactoring transformations
from transformation.refactoring.variable_renaming import VariableRenaming, VariableRenamingSynonym, VariableRenamingAntonym
from transformation.refactoring.function_renaming import FunctionRenaming
from transformation.refactoring.class_renaming import ClassRenaming

RefactoringTransformations = [
    VariableRenaming,
    FunctionRenaming,
    ClassRenaming,
    VariableRenamingSynonym,
    VariableRenamingAntonym
]


#Logical Mutation
from transformation.operator_mutation.arithmetic_mutation import ArithmeticMutation
from transformation.operator_mutation.logical_mutation import LogicalMutation
from transformation.operator_mutation.comparison_mutation import ComparisonMutation
from transformation.code_mutation.number_replacer import NumberTransformer

LogicalMutationTransformations = [
    ArithmeticMutation,
    LogicalMutation,
    ComparisonMutation,
    NumberTransformer
]

from transformation.code_mutation.no_mutation import NoMutation

# Code Sosification
from transformation.code_sosification.random_transformation import DeleteASTNode, AddASTNode, ReplaceASTNode

CodeSosificationTransformations = [
    DeleteASTNode,
    AddASTNode,
    ReplaceASTNode,
]

from transformation.obfuscation.opaque_predicate import OpaquePredicateTransformation
from transformation.code_mutation.syntax_error import SyntaxErrorTransformation

ObfuscationTransformations = [
    OpaquePredicateTransformation,
    SyntaxErrorTransformation,
]

# Combining all transformations into a single list
ALL_TRANSFORMATIONS = RefactoringTransformations + LogicalMutationTransformations + CodeSosificationTransformations + ObfuscationTransformations + [NoMutation]
