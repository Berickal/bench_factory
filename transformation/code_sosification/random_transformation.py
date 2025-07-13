import sys
sys.path.append("../")
from transformation.ast_utils import _apply_edits
from tree_sitter import Tree, Node
from typing import override
import random
from transformation.code_sosification.sosification import SosieTransformationBase
import random

class DeleteASTNode(SosieTransformationBase):
    """Remove unnecessary statements."""
    
    @override
    def check(self, **kwargs) -> bool:
        transplantation_points = self.find_transplantation_points()
        return len(transplantation_points) > 0
    
    @override
    def apply(self, **kwargs) -> str:
        transplantation_points = self.find_transplantation_points()
        if not transplantation_points:
            return self.input
        
        target_node = random.choice(transplantation_points)
        
        edits = [{
            "start_byte": target_node.start_byte,
            "end_byte": target_node.end_byte,
            "new_text": "" 
        }]
        
        return _apply_edits(self.input, edits)


class AddASTNode(SosieTransformationBase):
    """Add random statements after transplantation points."""
    
    @override
    def check(self, **kwargs) -> bool:
        return len(self.reactions) > 0 and len(self.find_transplantation_points()) > 0
    
    @override
    def apply(self, **kwargs) -> str:
        transplantation_points = self.find_transplantation_points()
        if not transplantation_points or not self.reactions:
            return self.input
        
        target_node = random.choice(transplantation_points)
        random_reaction = random.choice(self.reactions)
        
        added_code = random_reaction.node.text.decode('utf-8')
        
        edits = [{
            "start_byte": target_node.end_byte,
            "end_byte": target_node.end_byte,
            "new_text": f"\n{added_code}"
        }]
        
        return _apply_edits(self.input, edits)


class ReplaceASTNode(SosieTransformationBase):
    """Replace statements with random statements."""
    
    @override
    def check(self, **kwargs) -> bool:
        return len(self.reactions) > 0 and len(self.find_transplantation_points()) > 0
    
    @override
    def apply(self, **kwargs) -> str:
        transplantation_points = self.find_transplantation_points()
        if not transplantation_points or not self.reactions:
            return self.input
        
        target_node = random.choice(transplantation_points)
        
        available_reactions = [r for r in self.reactions if r.node != target_node]
        if not available_reactions:
            return self.input
        
        random_reaction = random.choice(available_reactions)
        replacement_code = random_reaction.node.text.decode('utf-8')
        
        edits = [{
            "start_byte": target_node.start_byte,
            "end_byte": target_node.end_byte,
            "new_text": replacement_code
        }]
        
        return _apply_edits(self.input, edits)