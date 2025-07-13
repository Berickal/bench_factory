import sys
sys.path.append("../")
from transformation.transformation import CodeTransformation
from transformation.ast_utils import _parse, _flatten_tree, _flatten_from_node, _apply_edits
from tree_sitter import Tree, Node
from typing import override, List, Dict, Set, Tuple, Optional
import random
import math

class OpaquePredicateTransformation(CodeTransformation):
    """
    Transformation that inserts opaque predicates into code.
    
    Opaque predicates are expressions that always evaluate to the same value
    but are difficult for static analysis to determine. They can be used to:
    1. Create functionally equivalent code variants (sosies)
    2. Obfuscate code to make reverse engineering harder
    3. Add computational diversity while preserving semantics
    """

    def __init__(self, input: str, programming_language: str):
        super().__init__(input, programming_language)
        self.available_variables = {}
        self._extract_variable_info()

    def _extract_variable_info(self):
        """Extract available variables and their types from the program."""
        tree = _parse(self.programming_language, self.input)
        nodes = _flatten_tree(tree)
        
        for node in nodes:
            if self._is_variable_declaration(node):
                var_info = self._extract_variable_from_declaration(node)
                if var_info:
                    line_num = node.start_point[0]
                    if line_num not in self.available_variables:
                        self.available_variables[line_num] = []
                    self.available_variables[line_num].append(var_info)

    def _is_variable_declaration(self, node: Node) -> bool:
        """Check if node declares a variable."""
        var_types = {
            "python": ["assignment"],
            "java": ["local_variable_declaration", "parameter"],
            "c": ["declaration", "parameter_declaration"]
        }
        return node.type in var_types.get(self.programming_language.lower(), [])

    def _extract_variable_from_declaration(self, node: Node) -> Optional[Dict[str, str]]:
        """Extract variable name and type from declaration."""
        text = node.text.decode('utf-8')
        
        if self.programming_language.lower() == "python":
            # Simple pattern matching for Python assignments
            if "=" in text:
                parts = text.split("=")
                var_name = parts[0].strip()
                # Try to infer type from value
                if parts[1].strip().isdigit():
                    return {"name": var_name, "type": "int"}
                elif "\"" in parts[1] or "'" in parts[1]:
                    return {"name": var_name, "type": "str"}
                else:
                    return {"name": var_name, "type": "unknown"}
        
        elif self.programming_language.lower() == "java":
            # Java variable declarations: "int x = 5;"
            if "int " in text:
                var_name = self._extract_java_var_name(text, "int")
                return {"name": var_name, "type": "int"}
            elif "double " in text or "float " in text:
                var_name = self._extract_java_var_name(text, "double")
                return {"name": var_name, "type": "double"}
            elif "String " in text:
                var_name = self._extract_java_var_name(text, "String")
                return {"name": var_name, "type": "String"}
        
        elif self.programming_language.lower() == "c":
            # C variable declarations: "int x = 5;"
            if "int " in text:
                var_name = self._extract_c_var_name(text, "int")
                return {"name": var_name, "type": "int"}
            elif "double " in text or "float " in text:
                var_name = self._extract_c_var_name(text, "double")
                return {"name": var_name, "type": "double"}
        
        return None

    def _extract_java_var_name(self, text: str, type_name: str) -> str:
        """Extract variable name from Java declaration."""
        parts = text.replace(type_name, "").strip().split("=")[0].strip()
        return parts.split()[0] if parts.split() else "unknown"

    def _extract_c_var_name(self, text: str, type_name: str) -> str:
        """Extract variable name from C declaration."""
        parts = text.replace(type_name, "").strip().split("=")[0].strip()
        return parts.split()[0] if parts.split() else "unknown"

    @override
    def check(self, **kwargs) -> bool:
        """Check if opaque predicates can be inserted."""
        tree = _parse(self.programming_language, self.input)
        nodes = _flatten_tree(tree)
        
        # Look for insertion points (statements, if conditions, loops)
        insertion_points = []
        for node in nodes:
            if self._is_valid_insertion_point(node):
                insertion_points.append(node)
        
        return len(insertion_points) > 0

    def _is_valid_insertion_point(self, node: Node) -> bool:
        """Check if node is a valid point to insert opaque predicates."""
        # Can insert before statements, in if conditions, or in loops
        valid_types = {
            "python": ["assignment", "expression_statement", "if_statement", "for_statement", "while_statement"],
            "java": ["local_variable_declaration", "expression_statement", "if_statement", "for_statement", "while_statement"],
            "c": ["declaration", "expression_statement", "if_statement", "for_statement", "while_statement"]
        }
        
        return node.type in valid_types.get(self.programming_language.lower(), [])

    @override
    def apply(self, **kwargs) -> str:
        """Apply opaque predicate transformation."""
        mode = kwargs.get('mode', 'conditional')  # 'conditional', 'assignment', 'loop'
        predicate_type = kwargs.get('predicate_type', 'random')  # 'random', 'mathematical', 'variable_based'
        
        tree = _parse(self.programming_language, self.input)
        nodes = _flatten_tree(tree)
        
        # Find insertion points
        insertion_points = [node for node in nodes if self._is_valid_insertion_point(node)]
        
        if not insertion_points:
            return self.input
        
        # Pick random insertion point
        target_node = random.choice(insertion_points)
        
        # Generate opaque predicate based on mode
        opaque_code = self._generate_opaque_predicate(mode, predicate_type, target_node)
        
        if not opaque_code:
            return self.input
        
        # Apply the transformation
        edits = []
        
        if mode == 'conditional':
            # Wrap existing code in opaque conditional
            edits = self._create_conditional_wrapper(target_node, opaque_code)
        elif mode == 'assignment':
            # Insert opaque assignment before the statement
            edits = self._create_assignment_insertion(target_node, opaque_code)
        elif mode == 'loop':
            # Insert opaque loop
            edits = self._create_loop_insertion(target_node, opaque_code)
        
        return _apply_edits(self.input, edits)

    def _generate_opaque_predicate(self, mode: str, predicate_type: str, context_node: Node) -> str:
        """Generate an opaque predicate based on the specified type."""
        
        if predicate_type == 'mathematical':
            return self._generate_mathematical_opaque()
        elif predicate_type == 'variable_based':
            return self._generate_variable_based_opaque(context_node)
        else:  # random
            return self._generate_random_opaque()

    def _generate_mathematical_opaque(self) -> str:
        """Generate mathematically-based opaque predicates."""
        opcodes = [
            # Always true predicates
            "7 * 7 * 13 * 13 % 4 == 1",  # (7*7*13*13) % 4 = 1
            "(2 * 2 * 2) == 8",
            "3 * 3 + 4 * 4 == 5 * 5",  # Pythagorean theorem
            "100 % 10 == 0",
            
            # Always false predicates  
            "5 * 5 + 6 * 6 == 7 * 7 + 1",  # 61 == 50
            "2 + 2 == 5",
            "1 == 0",
            "10 % 3 == 0",
        ]
        
        return random.choice(opcodes)

    def _generate_variable_based_opaque(self, context_node: Node) -> str:
        """Generate opaque predicates based on available variables."""
        # Get variables available at this point
        line_num = context_node.start_point[0]
        available_vars = []
        
        # Collect variables from current and previous lines
        for line in range(max(0, line_num - 10), line_num + 1):
            if line in self.available_variables:
                available_vars.extend(self.available_variables[line])
        
        if not available_vars:
            return self._generate_mathematical_opaque()
        
        # Filter for integer variables
        int_vars = [var for var in available_vars if var['type'] == 'int']
        
        if len(int_vars) >= 2:
            var1 = random.choice(int_vars)
            var2 = random.choice(int_vars)
            
            # Generate variable-based opaque predicates
            opcodes = [
                f"{var1['name']} * {var1['name']} >= 0",  # Always true
                f"{var1['name']} == {var1['name']}",      # Always true
                f"({var1['name']} + {var2['name']}) - {var2['name']} == {var1['name']}",  # Always true
                f"{var1['name']} * 0 == 0",               # Always true
                f"{var1['name']} + 1 > {var1['name']}",   # Always true (assuming no overflow)
                f"{var1['name']} < {var1['name']} - 1",   # Always false
                f"{var1['name']} * 0 == 1",               # Always false
            ]
            
            return random.choice(opcodes)
        
        elif int_vars:
            var = int_vars[0]
            return f"{var['name']} == {var['name']}"  # Always true
        
        return self._generate_mathematical_opaque()

    def _generate_random_opaque(self) -> str:
        """Generate random opaque predicates."""
        # Mix of different types
        all_opcodes = [
            # Mathematical
            "42 % 7 == 0",
            "16 == 4 * 4",
            "9 + 1 == 10",
            
            # String-based (for languages that support it)
            '"hello".length() == 5' if self.programming_language.lower() == 'java' else 'len("hello") == 5',
            
            # Always false
            "1 > 2",
            "0 == 1",
            "5 < 3",
        ]
        
        return random.choice(all_opcodes)

    def _create_conditional_wrapper(self, target_node: Node, opaque_predicate: str) -> List[Dict]:
        """Wrap the target statement in an opaque conditional."""
        original_code = target_node.text.decode('utf-8')
        
        if self.programming_language.lower() == "python":
            wrapper = f"""if {opaque_predicate}:
    {self._indent_code(original_code, "    ")}
else:
    pass  # This branch never executes"""
            
        elif self.programming_language.lower() == "java":
            wrapper = f"""if ({opaque_predicate}) {{
    {self._indent_code(original_code, "    ")}
}} else {{
    // This branch never executes
}}"""
            
        elif self.programming_language.lower() == "c":
            wrapper = f"""if ({opaque_predicate}) {{
    {self._indent_code(original_code, "    ")}
}} else {{
    /* This branch never executes */
}}"""
        
        else:
            return []
        
        return [{
            "start_byte": target_node.start_byte,
            "end_byte": target_node.end_byte,
            "new_text": wrapper
        }]

    def _create_assignment_insertion(self, target_node: Node, opaque_predicate: str) -> List[Dict]:
        """Insert an opaque assignment before the target statement."""
        
        if self.programming_language.lower() == "python":
            opaque_assignment = f"_opaque_var = {opaque_predicate}\n"
            
        elif self.programming_language.lower() == "java":
            opaque_assignment = f"boolean _opaqueVar = {opaque_predicate};\n"
            
        elif self.programming_language.lower() == "c":
            opaque_assignment = f"int _opaque_var = {opaque_predicate};\n"
        
        else:
            return []
        
        return [{
            "start_byte": target_node.start_byte,
            "end_byte": target_node.start_byte,
            "new_text": opaque_assignment
        }]

    def _create_loop_insertion(self, target_node: Node, opaque_predicate: str) -> List[Dict]:
        """Insert an opaque loop that never executes."""
        
        # Create a loop that never runs (false condition) or runs exactly once (true condition)
        false_predicate = f"!({opaque_predicate})"  # Negate a true predicate to make it false
        
        if self.programming_language.lower() == "python":
            opaque_loop = f"""# Opaque loop - never executes
while {false_predicate}:
    pass
"""
            
        elif self.programming_language.lower() == "java":
            opaque_loop = f"""// Opaque loop - never executes
while ({false_predicate}) {{
    // Dead code
}}
"""
            
        elif self.programming_language.lower() == "c":
            opaque_loop = f"""/* Opaque loop - never executes */
while ({false_predicate}) {{
    /* Dead code */
}}
"""
        
        else:
            return []
        
        return [{
            "start_byte": target_node.start_byte,
            "end_byte": target_node.start_byte,
            "new_text": opaque_loop
        }]

    def _indent_code(self, code: str, indent: str) -> str:
        """Add indentation to code."""
        lines = code.split('\n')
        indented_lines = [indent + line if line.strip() else line for line in lines]
        return '\n'.join(indented_lines)

    def generate_multiple_variants(self, count: int = 5) -> List[str]:
        """Generate multiple variants using different opaque predicate strategies."""
        variants = []
        
        modes = ['conditional', 'assignment', 'loop']
        predicate_types = ['mathematical', 'variable_based', 'random']
        
        for i in range(count):
            mode = random.choice(modes)
            pred_type = random.choice(predicate_types)
            
            if self.check(self.input):
                variant = self.apply(self.input, mode=mode, predicate_type=pred_type)
                if variant != self.input and variant not in variants:
                    variants.append(variant)
        
        return variants


# SPECIALIZED OPAQUE PREDICATE TRANSFORMATIONS

class ConditionalOpaqueTransformation(OpaquePredicateTransformation):
    """Specialized transformation that only adds opaque conditionals."""
    
    @override
    def apply(self, **kwargs) -> str:
        kwargs['mode'] = 'conditional'
        return super().apply(**kwargs)


class AssignmentOpaqueTransformation(OpaquePredicateTransformation):
    """Specialized transformation that only adds opaque assignments."""
    
    @override
    def apply(self, **kwargs) -> str:
        kwargs['mode'] = 'assignment'
        return super().apply(**kwargs)


class LoopOpaqueTransformation(OpaquePredicateTransformation):
    """Specialized transformation that only adds opaque loops."""
    
    @override
    def apply(self, **kwargs) -> str:
        kwargs['mode'] = 'loop'
        return super().apply(**kwargs)
