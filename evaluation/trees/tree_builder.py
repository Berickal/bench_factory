import sys
sys.path.append("../")

from transformation.invocation import ALL_TRANSFORMATIONS
from nlaugmenter.invocation import ALL_NL_TRANSFORMATIONS
from transformation.transformation import Transformation
from evaluation.trees.tree import MCTTree, MCTNode, Performance
from models.task_gen import LLMGen
from transformation.code_mutation.no_mutation import NoMutation
from typing import Optional
from data_loader.tasks import Instance

def build_node(instance : Optional[Instance], code: Optional[str], trans: Transformation, node_id: int, parent_id: Optional[int] = None) -> MCTNode:
    """
    Build a single MCTNode with the given parameters.
    
    Args:
        code (str): The code content
        trans (Transformation): The transformation applied
        node_id (int): Unique identifier for the node
        parent_id (Optional[int]): ID of the parent node
        
    Returns:
        MCTNode: A new node with the specified properties
    """
    if not instance and not code:
        raise("Error : No value present")
    
    return MCTNode(
        ids=node_id,
        instance=LLMGen(
            input=instance.input if instance else code,
            ref_output=instance.ref_output if instance else None,
            metadata=instance.metadata if instance else None,
            models=None,
            llm_response=[]
        ),
        performance=Performance(
            metric=None,
            score=0.0
        ),
        transformation=trans.__class__.__name__,
        perplexity=0.0,
        parent_node=parent_id,
        process=False
    )

def build_root_node(instance : Optional[Instance], code: str) -> MCTNode:
    if not instance and not code:
        raise("Error : No value present")
    
    return MCTNode(
        ids=0,
        instance=LLMGen(
            input=instance.input if instance else code,
            ref_output=instance.ref_output if instance else None,
            metadata=instance.metadata if instance else None,
            models=None,
            llm_response=[]
        ),
        performance=Performance(
            metric=None,
            score=0.0
        ),
        transformation=NoMutation.__name__,
        perplexity=0.0,
        parent_node=None,
        process=False
    )

def compute_sub_nodes(node : MCTNode, last_id : Optional[int], is_code : bool = True, **kwargs) -> list[MCTNode]:
    """
    Compute sub-nodes for a given MCTNode by applying all transformations.
    This function iterates through all available transformations and applies them
    to the code in the node, generating new MCTNodes for each transformation.
    Args:
        node (MCTNode): The node for which to compute sub-nodes
    Returns:
        list[MCTNode]: A list of MCTNodes generated from transformations
    """
    transformations = ALL_TRANSFORMATIONS if is_code else ALL_NL_TRANSFORMATIONS
    new_nodes = []
    last_id = last_id if last_id is not None else node.ids
    for trans_class in transformations:
        try:
            if is_code:
                trans = trans_class(input=node.instance.input, programming_language=kwargs.get('programming_language', 'python'))
                if trans.check():
                    new_nodes.append(
                        MCTNode(
                            ids= last_id,
                            instance=LLMGen(
                                input=trans.apply(**kwargs),
                                ref_output=node.instance.ref_output,
                                metadata=node.instance.metadata,
                                models=None,
                                llm_response=[]
                            ),
                            performance=Performance(
                                metric=None,
                                score=0.0
                            ),
                            transformation=trans.__class__.__name__,
                            perplexity=0.0,
                            parent_node=node.ids,
                            process=False
                        )
                    )
            else :
                trans = trans_class()
                alt_input = trans.generate(node.instance.input)
                new_nodes.append(
                    MCTNode(
                        ids= last_id,
                        instance=LLMGen(
                            input=alt_input[0] if type(alt_input) == list else alt_input,
                            ref_output=node.instance.ref_output,
                            metadata=node.instance.metadata,
                            models=None,
                            llm_response=[]
                        ),
                        performance=Performance(
                            metric=None,
                            score=0.0
                        ),
                        transformation=trans.__class__.__name__,
                        perplexity=0.0,
                        parent_node=node.ids,
                        process=False
                    )
                )
            last_id += 1
        except Exception as e:
                print("Exception during transformation application:", e)
                continue
    return new_nodes



def build_tree(instance : Optional[Instance] | None, code: Optional[str] | None, max_depth: int = 3, **kwargs) -> MCTTree:
    """
    Build a tree structure recursively with depth limit. The root node is created 
    with the transformation 'NoMutation'.
    
    Args:
        code (str): The code to be transformed.
        max_depth (int): Maximum depth of the tree (default: 3)

    Returns:
        Tree: A tree structure containing nodes for each transformation.
    """
    tree = MCTTree([])
    root_node = build_root_node(instance, code)
    tree.add_node(root_node)


    while len(tree.get_all_unprocessed()) > 0:
        for node in tree.get_all_unprocessed():
            _node = node
            new_nodes = compute_sub_nodes(_node, tree.get_number_of_nodes(), **kwargs)
            _node.process = True
            for n in new_nodes:
                tree = tree.add_node(n)
            tree = tree.update_node(node.ids, _node)

            if((len(tree.nodes)%100) == 0):
                print(f"Current number of nodes: {len(tree.nodes)}")

    return tree


def build_nl_tree(instance: Optional[Instance] | None, code: Optional[str] | None, max_depth: int = 3) -> MCTTree:
    """
    Build a tree structure for natural language transformations recursively with depth limit.
    The root node is created with the transformation 'NoMutation'.
    
    Args:
        code (str): The natural language text to be transformed.
        max_depth (int): Maximum depth of the tree (default: 3)

    Returns:
        Tree: A tree structure containing nodes for each transformation.
    """
    tree = MCTTree([])
    root_node = build_root_node(instance, code)
    tree.add_node(root_node)

    while len(tree.get_all_unprocessed()) > 0:
        for node in tree.get_all_unprocessed():
            _node = node
            new_nodes = compute_sub_nodes(_node, tree.get_number_of_nodes(), is_code=False)
            _node.process = True
            for n in new_nodes:
                tree = tree.add_node(n)
            tree = tree.update_node(node.ids, _node)

            if((len(tree.nodes)%100) == 0):
                print(f"Current number of nodes: {len(tree.nodes)}")

    return tree