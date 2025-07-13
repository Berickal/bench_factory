import sys
sys.path.append("../")
from evaluation.trees.tree import MCTTree, MCTNode, Performance
from typing import Optional, Dict, List, Tuple
from collections import defaultdict

def merge_trees_level_by_level(trees: List[MCTTree]) -> MCTTree:
    """
    Merge multiple MCTTree instances level by level to avoid instance count inflation.
    
    The algorithm:
    1. Start with root nodes - merge their statistics
    2. For each level, identify unique transformation paths
    3. Add missing nodes and merge statistics for existing ones
    4. Maintain correct parent-child relationships
    
    Args:
        trees: List of MCTTree instances to merge
        
    Returns:
        MCTTree: Merged tree with correct instance counts
    """
    if not trees:
        return MCTTree(nodes=[])
    
    if len(trees) == 1:
        # Ensure instance counts are correct for single tree
        _fix_single_tree_instance_counts(trees[0])
        return trees[0]
    
    # Clean all trees first
    for tree in trees:
        tree.clean_tree()
        tree.compute_performance_diff()
        _fix_single_tree_instance_counts(tree)
    
    # Create result tree
    merged_tree = MCTTree(nodes=[])
    
    # Step 1: Merge root nodes
    root_nodes = [tree.nodes[0] for tree in trees if tree.nodes]
    if not root_nodes:
        return merged_tree
    
    merged_root = _merge_nodes_with_same_transformation(root_nodes, parent_id=None)
    merged_root.ids = 0
    merged_tree.nodes.append(merged_root)
    
    # Step 2: Process each level
    current_level = 0
    while True:
        nodes_at_level = _get_nodes_at_level_from_all_trees(trees, current_level + 1)
        if not nodes_at_level:
            break  # No more levels to process
        
        # Group nodes by their parent transformation path and their own transformation
        grouped_nodes = _group_nodes_by_parent_path_and_transformation(
            nodes_at_level, trees, merged_tree
        )
        
        # Process each group
        for (parent_id, transformation), node_list in grouped_nodes.items():
            # Check if this transformation already exists for this parent
            existing_child = _find_child_with_transformation(merged_tree, parent_id, transformation)
            
            if existing_child:
                # Merge statistics into existing node
                _merge_statistics_into_existing_node(existing_child, node_list)
            else:
                # Create new merged node
                merged_node = _merge_nodes_with_same_transformation(node_list, parent_id)
                merged_node.ids = len(merged_tree.nodes)
                merged_tree.nodes.append(merged_node)
        
        current_level += 1
    
    merged_tree.clean_tree()
    return merged_tree


def _fix_single_tree_instance_counts(tree: MCTTree):
    """Fix instance counts for a single tree based on llm_response length."""
    for node in tree.nodes:
        if hasattr(node.instance, 'llm_response') and node.instance.llm_response:
            node.num_instances = len(node.instance.llm_response)
        elif isinstance(node.instance, dict) and 'llm_response' in node.instance:
            node.num_instances = len(node.instance['llm_response'])
        else:
            node.num_instances = 0


def _get_nodes_at_level_from_all_trees(trees: List[MCTTree], level: int) -> List[Tuple[MCTNode, int]]:
    """
    Get all nodes at a specific level from all trees.
    
    Returns:
        List of tuples (node, tree_index)
    """
    nodes_at_level = []
    
    for tree_idx, tree in enumerate(trees):
        for node in tree.nodes:
            node_level = _calculate_node_level(tree, node.ids)
            if node_level == level:
                nodes_at_level.append((node, tree_idx))
    
    return nodes_at_level


def _calculate_node_level(tree: MCTTree, node_id: int) -> int:
    """Calculate the level/depth of a node in the tree (root = 0)."""
    level = 0
    current_id = node_id
    
    while current_id is not None:
        node = tree.get_node_by_id(current_id)
        if node is None or node.parent_node is None:
            break
        current_id = node.parent_node
        level += 1
    
    return level


def _group_nodes_by_parent_path_and_transformation(
    nodes_at_level: List[Tuple[MCTNode, int]], 
    trees: List[MCTTree], 
    merged_tree: MCTTree
) -> Dict[Tuple[Optional[int], str], List[MCTNode]]:
    """
    Group nodes by their parent's path in the merged tree and their transformation.
    
    Returns:
        Dict mapping (parent_id_in_merged_tree, transformation) to list of nodes
    """
    grouped = defaultdict(list)
    
    for node, tree_idx in nodes_at_level:
        # Find the corresponding parent in the merged tree
        parent_path = _get_transformation_path_to_parent(trees[tree_idx], node.ids)
        merged_parent_id = _find_node_by_transformation_path(merged_tree, parent_path)
        
        # Group by (parent_id, transformation)
        key = (merged_parent_id, node.transformation)
        grouped[key].append(node)
    
    return grouped


def _get_transformation_path_to_parent(tree: MCTTree, node_id: int) -> List[str]:
    """Get the transformation path from root to the parent of the given node."""
    path = tree.get_path_from_node(node_id)
    path.reverse()  # Root first
    
    # Remove the current node, keep only path to parent
    if len(path) > 1:
        return [node.transformation for node in path[:-1]]
    else:
        return []  # Node is root or has no parent


def _find_node_by_transformation_path(tree: MCTTree, transformation_path: List[str]) -> Optional[int]:
    """Find a node in the tree by following a transformation path from root."""
    if not transformation_path:
        return None
    
    current_node_id = 0  # Start at root
    
    for transformation in transformation_path[1:]:  # Skip root transformation
        children = tree.get_children(current_node_id)
        found = False
        
        for child in children:
            if child.transformation == transformation:
                current_node_id = child.ids
                found = True
                break
        
        if not found:
            return None  # Path doesn't exist in merged tree
    
    return current_node_id


def _find_child_with_transformation(tree: MCTTree, parent_id: int, transformation: str) -> Optional[MCTNode]:
    """Find a child node with specific transformation under given parent."""
    children = tree.get_children(parent_id)
    for child in children:
        if child.transformation == transformation:
            return child
    return None


def _merge_nodes_with_same_transformation(nodes: List[MCTNode], parent_id: Optional[int]) -> MCTNode:
    """
    Merge multiple nodes with the same transformation into a single node.
    
    Args:
        nodes: List of nodes with the same transformation to merge
        parent_id: Parent node ID in the merged tree
        
    Returns:
        MCTNode: New merged node
    """
    if not nodes:
        raise ValueError("Cannot merge empty list of nodes")
    
    template_node = nodes[0]
    
    # Calculate total instances
    total_instances = sum(node.num_instances for node in nodes)
    
    # Calculate weighted averages
    weighted_score = 0.0
    weighted_perplexity = 0.0
    weighted_perf_diff = 0.0
    
    if total_instances > 0:
        for node in nodes:
            weight = node.num_instances / total_instances
            weighted_score += _get_performance_score(node) * weight
            weighted_perplexity += (node.perplexity or 0.0) * weight
            weighted_perf_diff += (node.perf_diff or 0.0) * weight

    
    # Create merged performance
    merged_performance = Performance(
        metric=_get_performance_metric(template_node),
        score=weighted_score
    )
    
    # Merge llm_responses
    merged_llm_responses = []
    for node in nodes:
        if hasattr(node.instance, 'llm_response') and node.instance.llm_response:
            merged_llm_responses.extend(node.instance.llm_response)
        elif isinstance(node.instance, dict) and 'llm_response' in node.instance:
            merged_llm_responses.extend(node.instance['llm_response'])
    
    # Create merged instance
    if hasattr(template_node.instance, 'llm_response'):
        merged_instance = template_node.instance
        merged_instance.llm_response = merged_llm_responses
    else:
        # Handle dict format
        merged_instance = dict(template_node.instance) if isinstance(template_node.instance, dict) else template_node.instance
        if isinstance(merged_instance, dict):
            merged_instance['llm_response'] = merged_llm_responses
    
    return MCTNode(
        ids=0,  # Will be set when added to tree
        instance=merged_instance,
        performance=merged_performance,
        transformation=template_node.transformation,
        perplexity=weighted_perplexity,
        parent_node=parent_id,
        process=template_node.process,
        num_instances=total_instances,
        perf_diff=weighted_perf_diff
    )


def _merge_statistics_into_existing_node(target_node: MCTNode, source_nodes: List[MCTNode]):
    """Merge statistics from source nodes into existing target node."""
    all_nodes = [target_node] + source_nodes
    
    # Calculate new total instances
    total_instances = sum(node.num_instances for node in all_nodes)
    
    # Calculate weighted averages
    weighted_score = 0.0
    weighted_perplexity = 0.0
    
    if total_instances > 0:
        for node in all_nodes:
            weight = node.num_instances / total_instances
            weighted_score += _get_performance_score(node) * weight
            weighted_perplexity += (node.perplexity or 0.0) * weight
    
    # Update target node
    if isinstance(target_node.performance, dict):
        target_node.performance['score'] = weighted_score
    else:
        target_node.performance.score = weighted_score
    
    target_node.perplexity = weighted_perplexity
    target_node.num_instances = total_instances
    
    # Merge llm_responses
    for source_node in source_nodes:
        if hasattr(target_node.instance, 'llm_response') and hasattr(source_node.instance, 'llm_response'):
            if source_node.instance.llm_response:
                target_node.instance.llm_response.extend(source_node.instance.llm_response)
        elif isinstance(target_node.instance, dict) and isinstance(source_node.instance, dict):
            if 'llm_response' in source_node.instance and source_node.instance['llm_response']:
                target_node.instance.setdefault('llm_response', []).extend(source_node.instance['llm_response'])


def _get_performance_score(node: MCTNode) -> float:
    """Extract performance score from node, handling both dict and object formats."""
    if node.performance is None:
        return 0.0
    if isinstance(node.performance, dict):
        return node.performance.get('score', 0.0)
    else:
        return node.performance.score if hasattr(node.performance, 'score') else 0.0


def _get_performance_metric(node: MCTNode):
    """Extract performance metric from node, handling both dict and object formats."""
    if node.performance is None:
        return None
    if isinstance(node.performance, dict):
        return node.performance.get('metric', None)
    else:
        return node.performance.metric if hasattr(node.performance, 'metric') else None


# Updated main merge function
def merge_trees(trees: List[MCTTree]) -> MCTTree:
    """
    Updated merge_trees function using level-by-level approach.
    
    Args:
        trees: List of MCTTree instances to merge
        
    Returns:
        MCTTree: Merged tree with correct instance counts
    """
    return merge_trees_level_by_level(trees)


# Validation function
def validate_merged_tree(merged_tree: MCTTree, original_trees: List[MCTTree]) -> Dict[str, any]:
    """
    Validate that the merged tree has correct statistics.
    
    Returns:
        Dict with validation results
    """
    # Calculate expected total instances from original trees
    expected_root_instances = sum(
        tree.nodes[0].num_instances for tree in original_trees if tree.nodes
    )
    
    actual_root_instances = merged_tree.nodes[0].num_instances if merged_tree.nodes else 0
    
    # Count unique transformation paths
    unique_paths = set()
    for tree in original_trees:
        for terminal_node in tree.get_terminal_nodes():
            path = tree.get_path_from_node(terminal_node.ids)
            path_signature = tuple(node.transformation for node in reversed(path))
            unique_paths.add(path_signature)
    
    merged_paths = set()
    for terminal_node in merged_tree.get_terminal_nodes():
        path = merged_tree.get_path_from_node(terminal_node.ids)
        path_signature = tuple(node.transformation for node in reversed(path))
        merged_paths.add(path_signature)
    
    return {
        'expected_root_instances': expected_root_instances,
        'actual_root_instances': actual_root_instances,
        'instances_match': expected_root_instances == actual_root_instances,
        'expected_unique_paths': len(unique_paths),
        'actual_unique_paths': len(merged_paths),
        'paths_match': unique_paths == merged_paths,
        'missing_paths': unique_paths - merged_paths,
        'extra_paths': merged_paths - unique_paths
    }