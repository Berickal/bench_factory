import sys
sys.path.append("../")
from evaluation.trees.tree import MCTTree, MCTNode, Performance
from typing import Optional
from pyvis.network import Network

def merge_trees(trees: list[MCTTree]) -> MCTTree:
    """
    Merge multiple MCTTree instances by combining their unique paths and merging instances.
    For nodes with the same transformation path, merge their instances and update performance.
    
    Args:
        trees: List of MCTTree instances to merge
        
    Returns:
        MCTTree: Merged tree containing all unique paths with merged instances
    """
    if not trees:
        return MCTTree(nodes=[])
    
    if len(trees) == 1:
        return trees[0]
    
    merged_tree = MCTTree(nodes=[])
    
    # Collect all unique paths from all trees
    all_paths = {}  # path_key -> list of nodes with this path from different trees
    
    for tree in trees:
        tree.clean_tree()  # Ensure each tree is clean before processing
        tree.compute_performance_diff()
        terminal_nodes = tree.get_terminal_nodes()
        for terminal_node in terminal_nodes:
            path = tree.get_path_from_node(terminal_node.ids)
            path.reverse()  # From root to terminal
            path_key = _create_path_key_from_nodes(path)
            
            if path_key not in all_paths:
                all_paths[path_key] = []
            all_paths[path_key].append(path)
    
    # Add each unique path to the merged tree
    for path_key, path_variants in all_paths.items():
        # Use the first variant as the base path
        base_path = path_variants[0]
        
        # Add the complete path to the merged tree
        _add_complete_path_to_tree(merged_tree, base_path)
        
        # If there are multiple variants of this path, merge their instances
        if len(path_variants) > 1:
            # Find all nodes in merged tree for this path and merge instances
            _merge_path_instances(merged_tree, path_variants)
    
    merged_tree.clean_tree()
    return merged_tree


def _merge_path_instances(merged_tree: MCTTree, path_variants: list[list[MCTNode]]):
    """
    Merge instances for all nodes in a path when multiple variants exist.
    
    Args:
        merged_tree: The merged tree containing the base path
        path_variants: List of path variants from different trees
    """
    base_path = path_variants[0]
    
    # For each level in the path, merge instances from all variants
    for level in range(len(base_path)):
        # Find the corresponding node in merged tree
        if level == 0:
            # Root node
            merged_node = merged_tree.nodes[0] if merged_tree.nodes else None
        else:
            # Find node by walking down the path
            merged_node = _find_node_in_merged_path(merged_tree, base_path[:level+1])
        
        if merged_node:
            # Merge instances from all variants at this level
            for variant_idx in range(1, len(path_variants)):
                if level < len(path_variants[variant_idx]):
                    variant_node = path_variants[variant_idx][level]
                    _merge_node_instances(merged_node, variant_node)


def _find_node_in_merged_path(merged_tree: MCTTree, path_prefix: list[MCTNode]) -> Optional[MCTNode]:
    """
    Find a node in the merged tree by following a path prefix.
    
    Args:
        merged_tree: Tree to search in
        path_prefix: Path from root to the desired node
        
    Returns:
        Optional[MCTNode]: Node if found, None otherwise
    """
    current_node = merged_tree.nodes[0] if merged_tree.nodes else None
    
    for i in range(1, len(path_prefix)):
        if current_node is None:
            return None
        
        target_transformation = path_prefix[i].transformation
        children = merged_tree.get_children(current_node.ids)
        
        current_node = None
        for child in children:
            if child.transformation == target_transformation:
                current_node = child
                break
    
    return current_node
    
    merged_tree.clean_tree()
    return merged_tree


def _add_complete_path_to_tree(tree: MCTTree, path: list[MCTNode]):
    """
    Add a complete path to the tree, ensuring all parent-child relationships are correct.
    
    Args:
        tree: Tree to add the path to
        path: Complete path from root to terminal (in order)
    """
    parent_id = None
    
    for i, node in enumerate(path):
        # For root node (parent_id is None), check if root already exists
        if parent_id is None and tree.nodes:
            # Check if root with same transformation already exists
            root_node = tree.nodes[0]  # Root is always the first node
            if root_node.transformation == node.transformation:
                parent_id = root_node.ids
                continue
        
        # Check if a node with this transformation already exists at this level
        existing_node = _find_node_at_level_with_transformation(tree, node.transformation, parent_id)
        
        if existing_node:
            # Use existing node as parent for next level
            parent_id = existing_node.ids
        else:
            # Create new node
            new_node = _copy_node(node)
            new_node.parent_node = parent_id
            tree.add_node(new_node)
            parent_id = new_node.ids


def _find_node_at_level_with_transformation(tree: MCTTree, transformation: str, parent_id: Optional[int]) -> Optional[MCTNode]:
    """
    Find a node with the given transformation that has the specified parent.
    
    Args:
        tree: Tree to search in
        transformation: Transformation to look for
        parent_id: Expected parent ID
        
    Returns:
        Optional[MCTNode]: Node if found, None otherwise
    """
    for node in tree.nodes:
        if node.transformation == transformation and node.parent_node == parent_id:
            return node
    return None


def _find_terminal_node_by_path(tree: MCTTree, path: list[MCTNode]) -> Optional[MCTNode]:
    """
    Find the terminal node in the tree that corresponds to the given path.
    
    Args:
        tree: Tree to search in
        path: Path from root to terminal
        
    Returns:
        Optional[MCTNode]: Terminal node if found, None otherwise
    """
    # Create the path key and find matching terminal node
    path_key = _create_path_key_from_nodes(path)
    
    for terminal_node in tree.get_terminal_nodes():
        terminal_path = tree.get_path_from_node(terminal_node.ids)
        terminal_path.reverse()
        terminal_path_key = _create_path_key_from_nodes(terminal_path)
        
        if path_key == terminal_path_key:
            return terminal_node
    
    return None


def _merge_node_instances(target_node: MCTNode, source_node: MCTNode):
    """
    Merge instances and update performance metrics for two nodes with the same transformation path.
    
    Args:
        target_node: Node in target tree to merge into
        source_node: Node in source tree to merge from
    """
    # Merge LLM responses
    if target_node.instance.llm_response is None:
        target_node.instance.llm_response = []
    if source_node.instance.llm_response is None:
        source_node.instance.llm_response = []
    
    target_node.instance.llm_response.extend(source_node.instance.llm_response)
    
    # Update number of instances
    target_num_instances = target_node.num_instances or 0
    source_num_instances = source_node.num_instances or 0
    new_num_instances = target_num_instances + source_num_instances
    
    # Calculate weighted average of performance scores
    if target_node.performance and source_node.performance:
        target_score = target_node.performance.score
        source_score = source_node.performance.score
        
        # Weighted average based on number of instances
        new_score = (target_score * target_num_instances + source_score * source_num_instances) / new_num_instances
        target_node.performance.score = new_score
    
    # Calculate weighted average of perplexity
    target_perplexity = target_node.perplexity
    source_perplexity = source_node.perplexity
    new_perplexity = (target_perplexity * target_num_instances + source_perplexity * source_num_instances) / new_num_instances
    
    # Calculate weighted average of performance variation
    target_variation = target_node.perf_diff
    source_variation = source_node.perf_diff
    new_perf_diff = (target_variation * target_num_instances + source_variation * source_num_instances) / new_num_instances

    # Update node attributes
    target_node.num_instances = new_num_instances
    target_node.perplexity = new_perplexity
    target_node.perf_diff = new_perf_diff


def _create_path_key_from_nodes(path_nodes: list[MCTNode]) -> str:
    """
    Create a unique key representing the transformation path from a list of nodes.
    
    Args:
        path_nodes: List of nodes representing a path
        
    Returns:
        str: Unique path key
    """
    transformations = [node.transformation for node in path_nodes]
    return "->".join(transformations)


def _copy_node(node: MCTNode) -> MCTNode:
    """
    Create a deep copy of a node.
    
    Args:
        node: Node to copy
        
    Returns:
        MCTNode: Copied node
    """
    from copy import deepcopy
    return deepcopy(node)


def tree_visualisation(tree : MCTTree, filename: str = "tree.html", title: str = "MCT Tree Visualization"):
    """
    Visualize the MCTTree using pyvis and save it to an HTML file.
    
    Args:
        tree: MCTTree instance to visualize
        filename: Name of the output HTML file
        title: Title for the visualization
    """
    tree.clean_tree()
    tree.compute_performance_diff()

    ids, labels, edges, colors, titles = [], [], [], [], []
    for n in tree.nodes:
        ids.append(n.ids)
        labels.append(n.transformation)
        if n.parent_node is not None:
            edges.append((n.parent_node, n.ids))
        if n.perf_diff >= 0:
            colors.append("#39744a")
        elif n.perf_diff >= -0.1:
            colors.append("#18a999")
        elif n.perf_diff == 0.0 and n.num_instances == 0:
            colors.append("#777777")
        else:
            colors.append("#df2935")

        titles.append(f"Instances: {n.num_instances}<br>Score: {n.performance.score if n.performance else 'N/A'}<br>Perplexity: {n.perplexity}<br>Performance Diff: {n.perf_diff}")

    net = Network(height="100%", width="100vh")
    net.add_nodes(ids, label=labels, title=titles, color=colors, shape="box")
    net.add_edges(edges)
    net.save_graph(filename)