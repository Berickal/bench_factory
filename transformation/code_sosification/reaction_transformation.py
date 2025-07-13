import sys
sys.path.append("../")
from transformation.ast_utils import _apply_edits
from tree_sitter import Tree, Node
from typing import override
import random
from transformation.code_sosification.sosification import SosieTransformationBase
import random

class AddReactionTransformation(SosieTransformationBase):
    """Add statements based on type compatibility."""
    
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
        
        compatible_reactions = []
        for reaction in self.reactions:
            reaction_types = set(reaction.input_types)
            if reaction_types.issubset(target_types) or not reaction_types:
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


class ReplaceReactionTransformation(AddReactionTransformation):
    """Replace statements based on type compatibility."""
    
    @override
    def apply(self, **kwargs) -> str:
        transplantation_points = self.find_transplantation_points()
        if not transplantation_points:
            return self.input
        
        target_node = random.choice(transplantation_points)
        target_types = set(self._extract_input_types(target_node))
        
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
        replacement_code = chosen_reaction.node.text.decode('utf-8')
        
        edits = [{
            "start_byte": target_node.start_byte,
            "end_byte": target_node.end_byte,
            "new_text": replacement_code
        }]
        
        return _apply_edits(self.input, edits)