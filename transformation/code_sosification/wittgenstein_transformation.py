import sys
sys.path.append("../")
from transformation.ast_utils import _apply_edits
from tree_sitter import Tree, Node
from typing import override
import random
from transformation.code_sosification.sosification import SosieTransformationBase
import random


class AddWittgensteinTransformation(SosieTransformationBase):
    """Add statements based on variable name similarity."""
    
    @override
    def check(self, **kwargs) -> bool:
        return len(self.find_transplantation_points()) > 0
    
    @override
    def apply(self, **kwargs) -> str:
        transplantation_points = self.find_transplantation_points()
        if not transplantation_points:
            return self.input
        
        target_node = random.choice(transplantation_points)
        target_vars = set(self._extract_variable_names(target_node).keys())
        
        compatible_reactions = []
        for reaction in self.reactions:
            reaction_vars = set(reaction.compatible_vars.keys())
            if target_vars & reaction_vars:  # Intersection
                compatible_reactions.append(reaction)
        
        if not compatible_reactions:
            return self.input
        
        chosen_reaction = random.choice(compatible_reactions)
        added_code = chosen_reaction.node.text.decode('utf-8')
        
        edits = [{
            "start_byte": target_node.end_byte,
            "end_byte": target_node.end_byte,
            "new_text": f"\n{added_code}"
        }]
        
        return _apply_edits(self.input, edits)


class ReplaceWittgensteinTransformation(AddWittgensteinTransformation):
    """Replace statements based on variable name similarity."""
    
    @override
    def apply(self, **kwargs) -> str:
        transplantation_points = self.find_transplantation_points()
        if not transplantation_points:
            return self.input
        
        target_node = random.choice(transplantation_points)
        target_vars = set(self._extract_variable_names(target_node).keys())
        
        # Find reactions with similar variable names (excluding self)
        compatible_reactions = []
        for reaction in self.reactions:
            if reaction.node == target_node:
                continue
            reaction_vars = set(reaction.compatible_vars.keys())
            if target_vars & reaction_vars:
                compatible_reactions.append(reaction)
        
        if not compatible_reactions:
            return self.input
        
        chosen_reaction = random.choice(compatible_reactions)
        replacement_code = chosen_reaction.node.text.decode('utf-8')
        
        edits = [{
            "start_byte": target_node.start_byte,
            "end_byte": target_node.end_byte,
            "new_text": replacement_code
        }]
        
        return _apply_edits(self.input, edits)