import sys
sys.path.append("../")
from transformation.ast_utils import _apply_edits
from tree_sitter import Tree, Node
from typing import override, Dict
import random
from transformation.code_sosification.sosification import SosieTransformationBase
import random

class AddSteroidTransformation(SosieTransformationBase):
    """Add statements with type compatibility and variable renaming."""
    
    @override
    def check(self, **kwargs) -> bool:
        return len(self.reactions) > 0 and len(self.find_transplantation_points()) > 0
    
    @override
    def apply(self, **kwargs) -> str:
        transplantation_points = self.find_transplantation_points()
        if not transplantation_points:
            return self.input
        
        target_node = random.choice(transplantation_points)
        target_types = set(self._extract_input_types(target_node))
        target_vars = self._extract_variable_names(target_node)
        
        compatible_reactions = []
        for reaction in self.reactions:
            reaction_types = set(reaction.input_types)
            if reaction_types.issubset(target_types) or not reaction_types:
                compatible_reactions.append(reaction)
        
        if not compatible_reactions:
            return self.input
        
        chosen_reaction = random.choice(compatible_reactions)
        
        enhanced_code = self._apply_variable_renaming(
            chosen_reaction.node.text.decode('utf-8'),
            chosen_reaction.compatible_vars,
            target_vars
        )
        
        edits = [{
            "start_byte": target_node.end_byte,
            "end_byte": target_node.end_byte,
            "new_text": f"\n{enhanced_code}"
        }]
        
        return _apply_edits(self.input, edits)
    
    def _apply_variable_renaming(self, code: str, source_vars: Dict[str, str], 
                               target_vars: Dict[str, str]) -> str:
        """Apply variable renaming to make transplant compatible."""
        result = code
        
        for source_var, source_type in source_vars.items():
            compatible_targets = [var for var, var_type in target_vars.items() 
                                if var_type == source_type]
            
            if compatible_targets:
                target_var = random.choice(compatible_targets)
                result = result.replace(source_var, target_var)
        
        return result


class ReplaceSteroidTransformation(AddSteroidTransformation):
    """Replace statements with type compatibility and variable renaming."""
    
    @override
    def apply(self, **kwargs) -> str:
        transplantation_points = self.find_transplantation_points()
        if not transplantation_points:
            return self.input
        
        target_node = random.choice(transplantation_points)
        target_types = set(self._extract_input_types(target_node))
        target_vars = self._extract_variable_names(target_node)
        
        compatible_reactions = []
        for reaction in self.reactions:
            if reaction.node == target_node:
                continue
            reaction_types = set(reaction.input_types)
            if reaction_types.issubset(target_types) or not reaction_types:
                compatible_reactions.append(reaction)
        
        if not compatible_reactions:
            return self.input
        
        chosen_reaction = random.choice(compatible_reactions)
        
        enhanced_code = self._apply_variable_renaming(
            chosen_reaction.node.text.decode('utf-8'),
            chosen_reaction.compatible_vars,
            target_vars
        )
        
        edits = [{
            "start_byte": target_node.start_byte,
            "end_byte": target_node.end_byte,
            "new_text": enhanced_code
        }]
        
        return _apply_edits(self.input, edits)