from dataclasses import dataclass, asdict
import sys
sys.path.append("../")
from models.task_gen import LLMGen
from evaluation.metrics.metric import Metric
from typing import Optional
import json

@dataclass
class Performance:
    metric : Metric
    score : float

@dataclass
class MCTNode:
    ids : int
    instance : LLMGen
    performance : Performance
    transformation : str
    perplexity : float
    parent_node : Optional[int]
    process : bool
    num_instances : Optional[int] = 10
    perf_diff : Optional[float] = 0
    perplexity_diff : Optional[float] = 0

    def get_score_deviation(self, parent: 'MCTNode') -> float:
        """
        Calculate the deviation of the performance score from the parent node.
        
        Args:
            parent: Parent MCTNode to compare with
            
        Returns:
            float: Deviation of the score from the parent node's score
        """
        if parent is None or parent.performance is None:
            return 0.0
        return self.performance.score - parent.performance.score
    
    def get_perplexity_deviation(self, parent: 'MCTNode') -> float:
        """
        Calculate the deviation of the perplexity from the parent node.
        
        Args:
            parent: Parent MCTNode to compare with
            
        Returns:
            float: Deviation of the perplexity from the parent node's perplexity
        """
        if parent is None or parent.perplexity is None:
            return 0.0
        return self.perplexity - parent.perplexity
    

@dataclass
class MCTTree:
    nodes : list[MCTNode]

    def add_node(self, node: MCTNode):
        """
        Add a node to the tree with the constraint that each branch 
        must not contain the same transformation more than once.
        
        Args:
            node: MCTNode to be added
            
        Returns:
            MCTTree: Updated tree with the new node added
        """
        node.ids = len(self.nodes)
        if node.parent_node is None:
            self.nodes.append(node)
            return self
        
        parent_exists = any(n.ids == node.parent_node for n in self.nodes)
        if not parent_exists:
            raise ValueError(f"Parent node with id {node.parent_node} does not exist")
        
        transformations_in_path = self._get_transformations_in_path(node.parent_node)
        if node.transformation in transformations_in_path:
            return self
        
        self.nodes.append(node)

        return self
    
    def update_node(self, node_id: int, new_node: MCTNode):
        """
        Update an existing node in the tree.
        
        Args:
            node_id: ID of the node to update
            new_node: New MCTNode with updated values
        Returns:
            MCTTree: Updated tree with the new node
        """
        for i, node in enumerate(self.nodes):
            if node.ids == node_id:
                self.nodes[i] = new_node
                return self
        return self
    
    def _get_transformations_in_path(self, node_id: int) -> set[str]:
        """
        Get all transformations in the path from root to the given node.
        
        Args:
            node_id: ID of the node to trace back to root
            
        Returns:
            set[str]: Set of transformations in the path
        """
        transformations = set()
        current_id = node_id
        
        while current_id is not None:
            current_node = next((n for n in self.nodes if n.ids == current_id), None)
            if current_node is None:
                break
            
            transformations.add(current_node.transformation)
            current_id = current_node.parent_node
        
        return transformations
    
    def get_node_by_id(self, node_id: int) -> Optional[MCTNode]:
        """
        Get a node by its ID.
        
        Args:
            node_id: ID of the node to retrieve
            
        Returns:
            Optional[MCTNode]: The node if found, None otherwise
        """
        return next((n for n in self.nodes if n.ids == node_id), None)
    
    
    def get_children(self, parent_id: int) -> list[MCTNode]:
        """
        Get all children of a given parent node.
        
        Args:
            parent_id: ID of the parent node
            
        Returns:
            list[MCTNode]: List of child nodes
        """
        return [n for n in self.nodes if n.parent_node == parent_id]
    

    def walk(self, node: MCTNode) -> any:
        """
        Walk through the tree starting from a given node and yield each node.
        Args:
            node: MCTNode to start walking from
        Yields:
            MCTNode: Each node in the tree starting from the given node
        """

        yield node
        
        children = self.get_children(node.ids)
        for child in children:
            yield from self.walk(child)

    def get_flattened(self) -> list[MCTNode]:
        """
        Get a flattened list of all nodes in the tree.
        
        Returns:
            list[MCTNode]: Flattened list of all nodes
        """
        return list(self.walk(self.nodes[0])) if self.nodes else []
    
    def get_all_unprocessed(self) -> list[MCTNode]:
        """
        Get all unprocessed nodes in the tree.
        
        Returns:
            list[MCTNode]: List of unprocessed nodes
        """
        return [n for n in self.nodes if not n.process]
    
    def get_number_of_nodes(self) -> int:
        """
        Get the total number of nodes in the tree.
        
        Returns:
            int: Total number of nodes
        """
        return len(self.nodes)
    
    def get_terminal_nodes(self) -> list[MCTNode]:
        """
        Get all terminal (leaf) nodes in the tree.
        Terminal nodes are nodes that have no children.
        
        Returns:
            list[MCTNode]: List of all terminal nodes
        """
        terminal_nodes = []
        for node in self.nodes:
            # A node is terminal if it has no children
            if not self.get_children(node.ids):
                terminal_nodes.append(node)
        return terminal_nodes
    
    def get_path_from_node(self, node_id: int) -> list[MCTNode]:
        """
        Get the path from the given node to the root node.
        
        Args:
            node_id: ID of the starting node
            
        Returns:
            list[MCTNode]: List of nodes in the path from the given node to root
                          (including both the starting node and root node)
        """
        path = []
        current_id = node_id
        
        while current_id is not None:
            current_node = self.get_node_by_id(current_id)
            if current_node is None:
                break
            
            path.append(current_node)
            current_id = current_node.parent_node
        
        return path
    
    def _get_all_transformation_set(self) -> list[set[str]]:
        terminal_nodes = self.get_terminal_nodes()
        trans_set = []
        for node in terminal_nodes:
            trans_set.append(self._get_transformations_in_path(node.ids))
        
        return trans_set
    
    def remove_node(self, node_id: int) -> bool:
        """
        Remove a node and all its descendants from the tree.
        
        Args:
            node_id: ID of the node to remove
            
        Returns:
            bool: True if node was removed successfully, False if node not found
        """
        # Find the node to remove
        node_to_remove = self.get_node_by_id(node_id)
        if node_to_remove is None:
            return False
        
        # Get all nodes to remove (the node and all its descendants)
        nodes_to_remove = list(self.walk(node_to_remove))
        node_ids_to_remove = {node.ids for node in nodes_to_remove}
        
        # Remove all these nodes from the tree
        self.nodes = [node for node in self.nodes if node.ids not in node_ids_to_remove]
        
        return True
    
    def _validate_tree(self) -> bool:
        trans_list = self._get_all_transformation_set()
        for trans in trans_list:
            if trans_list.count(trans) > 1:
                return False
        return True
    
    def _check_duplicate_path(self, node_ids : int) -> bool:
        trans_set = self._get_all_transformation_set()
        trans = self._get_transformations_in_path(node_ids)
        return trans_set.count(trans) > 1

    def clean_tree(self):
        while not self._validate_tree():
            terminal_node = self.get_terminal_nodes()
            for node in terminal_node:
                if self._check_duplicate_path(node.ids):
                    self.remove_node(node.ids)

    def compute_performance_diff(self) -> None:
        """
        Compute the performance difference for each node in the tree.
        The performance difference is calculated as the difference between
        the node's score and its parent's score.
        """
        for node in self.nodes:
            if node.parent_node is not None:
                parent_node = self.get_node_by_id(node.parent_node)
                if parent_node:
                    node.perf_diff = node.performance.score - parent_node.performance.score
                    node.perplexity_diff = node.perplexity - parent_node.perplexity
                else:
                    node.perf_diff = 0.0
                    node.perplexity_diff = 0.0
            else:
                node.perf_diff = 0.0
                node.perplexity_diff = 0.0

    def save_tree(self, filename: Optional[str]) -> None:
        """
        Save the tree to a JSONL file.
        Each line contains one node's data in JSON format.
        
        Args:
            filename: Path to the output JSONL file
        """
        if not filename:
            filename = "default.jsonl"
        with open(filename, 'w') as f:
            for node in self.nodes:
                f.write(json.dumps(asdict(node)) + '\n')

    def load_tree(self, filename: str) -> None:
        """
        Load the tree from a JSONL file.
        Each line should contain one node's data in JSON format.
        
        Args:
            filename: Path to the input JSONL file
        """
        self.nodes = []
        with open(filename, 'r') as f:
            for line in f:
                if line.strip():  # Skip empty lines
                    node_dict = json.loads(line.strip())
                    
                    # Reconstruct LLMGen object from dictionary
                    instance_dict = node_dict['instance']
                    instance = LLMGen(
                        input=instance_dict.get('input'),
                        ref_output=instance_dict.get('ref_output'),
                        metadata=instance_dict.get('metadata'),
                        models=instance_dict.get('models'),
                        llm_response=instance_dict.get('llm_response', [])
                    )
                    
                    # Reconstruct Performance object from dictionary
                    performance_dict = node_dict['performance']
                    performance = Performance(
                        metric=performance_dict.get('metric'),  # Keep as dict or None
                        score=performance_dict.get('score', 0.0)
                    )
                    num_instances = node_dict.get('num_instances', 1)
                    
                    # Reconstruct MCTNode
                    node = MCTNode(
                        ids=node_dict['ids'],
                        instance=instance,
                        performance=performance,
                        transformation=node_dict['transformation'],
                        perplexity=node_dict.get('perplexity', 0.0),
                        parent_node=node_dict.get('parent_node'),
                        process=node_dict.get('process', False),
                        num_instances=num_instances
                    )
                    
                    self.nodes.append(node)

    
 

def load_tree(filename: str) -> "MCTTree":
    """
    Load a tree from a JSONL file (standalone function).
    
    Args:
        filename: Path to the input JSONL file
        
    Returns:
        Tree: Loaded tree
    """
    tree = MCTTree([])
    tree.load_tree(filename)
    return tree