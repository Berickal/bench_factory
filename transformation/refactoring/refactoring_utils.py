from tree_sitter import Node, Tree
import sys
sys.path.append("../")

from typing import List
import string
import random
from transformation.ast_utils import ASTQuery, _flatten_from_node, _apply_edits

from nlaugmenter.transformations.synonym_substitution.transformation import SynonymSubstitution
from nlaugmenter.transformations.antonyms_substitute.transformation import AntonymsSubstitute

class VariableQuery(ASTQuery):
    """A query that matches variable nodes in the AST."""
    EXCLUDE_PARENT_TYPES = [
        "function_definition", "class_definition", 
        "method_declaration", "class_declaration",
        "package_declaration", "dotted_name", "as", "aliased_import",
        "import"
    ]

    EXCLUDE_VALUE_TYPES = [
        "string", "number", "boolean", "null", "undefined", "len", "bool", "enumerate", "range", "int", "float", "complex",
        "array", "object", "tuple", "set", "map", 'list', "str", "self", "this", "super", "lambda", "function", "method"
    ]

    def query(self, node: Node) -> bool:
        """Checks if the node is a variable declaration or usage."""
        if node.type == "identifier":
            if node.parent:
                parent_type = node.parent.type
                if parent_type in self.EXCLUDE_PARENT_TYPES:
                    return False
            return True
        return False

    def find(self, node: Node) -> List[Node]:
        """Finds all variable nodes in the AST."""
        nodes = _flatten_from_node(node)
        result = []
        for n in nodes:
            if self.query(n):
                result.append(n)
        return result
    
    def get_variable_names(self) -> List[str]:
        """Returns a list of unique variable names from the variable nodes."""
        variable_nodes = self.find(self.root)
        names = []
        for node in variable_nodes:
            if node.text:
                name = node.text.decode('utf-8')
                # Exclude certain types of values
                if name not in self.EXCLUDE_VALUE_TYPES and name not in names:
                    names.append(name)
        return names


class FunctionQuery(ASTQuery):
    """A query that matches function nodes in the AST."""

    def query(self, node: Node) -> bool:
        """Checks if the node is a function definition."""
        if node.type in ["function_definition", "method_declaration"]:
            return True
        return False

    def find(self, node: Node) -> List[Node]:
        """Finds all function nodes in the AST."""
        nodes = _flatten_from_node(node)
        result = []
        for n in nodes:
            if self.query(n):
                result.append(n)
        return result
    
    def get_function_names(self) -> List[str]:
        """Returns a list of function names from the function nodes."""
        function_nodes = self.find(self.root)
        names = []
        for node in function_nodes:
            name_node = node.child_by_field_name('name')
            if name_node and name_node.text:
                name = name_node.text.decode('utf-8')
                if name not in names:
                    names.append(name)
        return names
    
class ClassQuery(ASTQuery):
    """A query that matches class nodes in the AST."""

    def query(self, node: Node) -> bool:
        """Checks if the node is a class definition."""
        if node.type in ["class_definition", "class_declaration"]:
            return True
        return False

    def find(self, node: Node) -> List[Node]:
        """Finds all class nodes in the AST."""
        nodes = _flatten_from_node(node)
        result = []
        for n in nodes:
            if self.query(n):
                result.append(n)
        return result
    
    def get_class_names(self) -> List[str]:
        """Returns a list of class names from the class nodes."""
        class_nodes = self.find(self.root)
        names = []
        for node in class_nodes:
            name_node = node.child_by_field_name('name')
            if name_node and name_node.text:
                name = name_node.text.decode('utf-8')
                if name not in names:
                    names.append(name)
        return names


def _get_variable_new_name(number: int, **kwargs) -> List[str]:
    """Generates a list of new variable names based on a given number."""
    new_names = []

    if kwargs.get('random', False):
        for _ in range(number):
            new_name = _get_random_identifier()
            new_names.append(new_name)
        return new_names
    
    elif kwargs.get('synonym', False):
        variables = kwargs.get('variables', [])
        for var in variables:
            new_name = synonym_generator(var)
            new_names.append(new_name)
        return new_names
    
    elif kwargs.get('antonym', False):
        variables = kwargs.get('variables', [])
        for var in variables:
            new_name = antonym_generator(var)
            new_names.append(new_name)
        return new_names
    
    # Default case: generate simple variable names like var_0, var_1, ...
    for i in range(number):
        new_name = f"var_{i}"
        new_names.append(new_name)
    return new_names

def _get_random_identifier(length: int = 8) -> str:
    """Generates a random identifier of a given length."""
    first_char = random.choice(string.ascii_letters)
    rest_chars = ''.join(random.choices(string.ascii_letters + string.digits, k=length-1))
    return first_char + rest_chars


def variable_refactoring(tree: Tree, source_code: str, once: bool = False, **kwargs) -> str:
    """Refactors variable names in the AST by renaming them to new names.

    Args:
        tree (Tree): A tree-sitter representation of a program.
        source_code (str): The original source code.
        once (bool, optional): If True, only refactor variable names once. Defaults to False.

    Returns:
        str: The modified source code with refactored variable names.
    """
    variable_query = VariableQuery(root=tree.root_node)
    variables = variable_query.get_variable_names()
    if not variables:
        return source_code

    #kwargs = kwargs.get("kwargs", {})
    kwargs['variables'] = variables
    new_names = _get_variable_new_name(len(variables), **kwargs)
    name_mapping = dict(zip(variables, new_names))
    
    edits = []
    for node in _flatten_from_node(tree.root_node):
        if node.type == "identifier" and node.text:
            old_name = node.text.decode('utf-8')
            if old_name in name_mapping:
                if variable_query.query(node):
                    edits.append({
                        'start_byte': node.start_byte,
                        'end_byte': node.end_byte,
                        'new_text': name_mapping[old_name]
                    })
                    
                    if once:
                        break
    
    return _apply_edits(source_code, edits)


def function_refactoring(tree: Tree, source_code: str, **kwargs) -> str:
    """Refactors function names in the AST by renaming them to new names.

    Args:
        tree (Tree): A tree-sitter representation of a program.
        source_code (str): The original source code.

    Returns:
        str: The modified source code with refactored function names.
    """
    function_query = FunctionQuery(root=tree.root_node)
    functions = function_query.get_function_names()
    kwargs = kwargs.get("kwargs", {})
    
    if not functions:
        return source_code

    new_names = _get_variable_new_name(len(functions), **kwargs)
    name_mapping = dict(zip(functions, new_names))
    
    edits = []
    
    # Find all function definitions and calls
    for node in _flatten_from_node(tree.root_node):
        if node.type == "identifier" and node.text:
            old_name = node.text.decode('utf-8')
            if old_name in name_mapping:
                # Check if this is a function name in a definition
                if (node.parent and 
                    node.parent.type in ["function_definition", "method_declaration"] and
                    node.parent.child_by_field_name('name') == node):
                    edits.append({
                        'start_byte': node.start_byte,
                        'end_byte': node.end_byte,
                        'new_text': name_mapping[old_name]
                    })
                # Check if this is a function call
                elif (node.parent and node.parent.type in ["call", "method_invocation"]):
                    edits.append({
                        'start_byte': node.start_byte,
                        'end_byte': node.end_byte,
                        'new_text': name_mapping[old_name]
                    })
    
    return _apply_edits(source_code, edits)

def class_refactoring(tree: Tree, source_code: str) -> str:
    """Refactors class names in the AST by renaming them to new names.

    Args:
        tree (Tree): A tree-sitter representation of a program.
        source_code (str): The original source code.

    Returns:
        str: The modified source code with refactored class names.
    """
    class_query = ClassQuery(root=tree.root_node)
    classes = class_query.get_class_names()
    
    if not classes:
        return source_code

    new_names = [_get_random_identifier() for _ in range(len(classes))]
    name_mapping = dict(zip(classes, new_names))
    
    edits = []
    
    for node in _flatten_from_node(tree.root_node):
        if node.type == "identifier" and node.text:
            old_name = node.text.decode('utf-8')
            if old_name in name_mapping:
                if (node.parent and 
                    node.parent.type in ["class_definition", "class_declaration"] and
                    node.parent.child_by_field_name('name') == node):
                    edits.append({
                        'start_byte': node.start_byte,
                        'end_byte': node.end_byte,
                        'new_text': name_mapping[old_name]
                    })
    
    return _apply_edits(source_code, edits)

def synonym_generator(identifier : str) ->  str:
    """Generates a synonym for a given identifier using SynonymSubstitution.

    Args:
        identifier (str): The identifier to generate a synonym for.

    Returns:
        str: A synonym for the identifier.
    """
    if "_" in identifier:
        identifier = identifier.replace("_", " ")
        identifier = SynonymSubstitution().generate(identifier)[0].replace(" ", "_")
    elif any(char.isupper() for char in identifier):
        for char in string.ascii_uppercase:
            if char in identifier:
                identifier = identifier.replace(char, " " + char)
        identifier = SynonymSubstitution().generate(identifier)[0]
        identifier = identifier.split()
        identifier = [identifier[0]] + [word.capitalize() for word in identifier[1:]]
        identifier = "".join(identifier)

    return identifier

def antonym_generator(identifier : str) ->  str:
    """Generates an antonym for a given identifier using AntonymsSubstitute.

    Args:
        identifier (str): The identifier to generate an antonym for.

    Returns:
        str: An antonym for the identifier.
    """
    if "_" in identifier:
        identifier = identifier.replace("_", " ")
        identifier = AntonymsSubstitute().generate(identifier)[0].replace(" ", "_")
    elif any(char.isupper() for char in identifier):
        for char in string.ascii_uppercase:
            if char in identifier:
                identifier = identifier.replace(char, " " + char)
        identifier = AntonymsSubstitute ().generate(identifier)[0]
        identifier = identifier.split()
        identifier = [identifier[0]] + [word.capitalize() for word in identifier[1:]]
        identifier = "".join(identifier)

    return identifier