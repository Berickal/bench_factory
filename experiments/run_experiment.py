#!/usr/bin/env python3

import sys
import os
import argparse
sys.path.append("../")
from evaluation.trees.tree_builder import build_tree, build_nl_tree
from evaluation.trees.optimized import build_tree_logic_optimized
from evaluation.trees.tree_utils import merge_trees
from evaluation.trees.tree import MCTTree, MCTNode, load_tree
from data_loader.tasks import Instance, Benchmark
import concurrent.futures

from models.llm_client import get_client_for_model, LlmClient
from models.models import MODEL_NAMES_TO_MODELS
from data_loader.benchmarks.benchmark_loader import BENCHMARK_NAME_TO_BENCHMARKS
from models.task_gen import LLMGen
from evaluation.metrics.pass_test import evaluate_quixbugs_instance, evaluate_human_eval_instance, evaluate_mbbp_instance, evaluate_condefect_instance, TestReport
from tqdm import tqdm
from typing import Tuple, List


DEFAULT_NUM_SAMPLE = 1
RUN_PARALLEL = 5

def main():
    parser = argparse.ArgumentParser(description="Evaluate QuixBugs benchmark with LLMs.")
    parser.add_argument("--model", type=str, choices=MODEL_NAMES_TO_MODELS.keys(), required=True, help="LLM model to use for evaluation.")
    parser.add_argument("--benchmark", type=str, choices=BENCHMARK_NAME_TO_BENCHMARKS.keys(), default="QuixBugs", help="Benchmark to evaluate.")
    parser.add_argument("--system_prompt", type=str, default="./system_prompt/PROGRAM_REPAIR.txt", help="Path to the system prompt file.")
    parser.add_argument("--num_samples", type=int, default=DEFAULT_NUM_SAMPLE, help="Number of samples to generate for each instance.")
    parser.add_argument("--run_parallel", action='store_true', help="Run the evaluation in parallel.")
    parser.add_argument("--output_dir", type=str, default="./results/", help="Output directory to save the evaluation results.")
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    benchmark = BENCHMARK_NAME_TO_BENCHMARKS[args.benchmark].load_benchmark()

    model = get_client_for_model(MODEL_NAMES_TO_MODELS[args.model])
    system_prompt = read_system_prompt(args.system_prompt)
    
    main_tree = evaluate_benchmark(
        benchmark=benchmark,
        model=model,
        system_prompt=system_prompt,
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        run_parallel=args.run_parallel
    )
    
    # Save final merged tree
    final_output = os.path.join(args.output_dir, "merged_results.json")
    main_tree.save_tree(filename=final_output)
    print(f"Evaluation completed. Final results saved to {final_output}")


def clean_code(response: str) -> str:
    """
    Clean the code response by removing leading and trailing whitespace.
    
    Args:
        response (str): The code response to clean.
        
    Returns:
        str: The cleaned code response.
    """
    if "```python" in response:
        response = response.split("```python")[1]
    if "```" in response:
        response = response.split("```")[0]
    return response.strip()


def process_node(node: MCTNode, model: LlmClient, system_prompt: str, num_samples: int, benchmark_name : str = "Quixbugs") -> MCTNode:
    """
    Process a single node: generate responses and evaluate them.
    
    Args:
        node: The node to process
        model: The LLM client
        system_prompt: System prompt for generation
        num_samples: Number of samples to generate
        
    Returns:
        MCTNode: The processed node with responses and evaluation
    """
    try:
        if benchmark_name.lower() == "TestEval".lower():
            instruction = build_test_generation_instruction(
                file_path="./system_prompt/TEST_GENERATION.txt",
                node=node
            )
            node.instance.input = instruction
        #print(node.instance.input)

        # Generate responses if not already present
        if len(node.instance.llm_response) == 0:
            responses = []
            for _ in range(num_samples):
                response = model.send_prompt(
                    system_prompt=system_prompt,
                    user_prompt=node.instance.input
                )
                if type(response) == list:
                    response = response[0]
                responses.append(clean_code(response))
            node.instance.llm_response = responses
        else:
            print(f"Node {node.ids} already has responses, skipping generation.")
        
        # Evaluate responses
        passed = 0
        for response in node.instance.llm_response:
            if benchmark_name.lower() == "Quixbugs".lower():
                test_report = evaluate_quixbugs_instance(
                    response=response,
                    tests=node.instance.metadata['tests'],
                    timeout=10,
                    programming_language=node.instance.metadata['programming_language']
                )
            elif benchmark_name.lower() == "HumanEval".lower():
                test_report = evaluate_human_eval_instance(
                    response=response,
                    tests=node.instance.metadata['tests'],
                    timeout=10,
                    ref_output=node.instance.ref_output
                )
            elif benchmark_name.lower() == "MBPP".lower():
                test_report = evaluate_mbbp_instance(
                    response=response,
                    tests=node.instance.metadata['tests'],
                    timeout=10,
                    programming_language=node.instance.metadata['programming_language']
                )
            elif benchmark_name.lower() == "Condefects".lower():
                test_report = evaluate_condefect_instance(
                    response=response,
                    test_in=node.instance.metadata['test_in'],
                    test_out=node.instance.metadata['test_out'],
                    timeout=10,
                    ref_output=node.instance.ref_output
                )
            elif benchmark_name.lower() == "TestEval".lower():
                test_report = TestReport(
                    passed=False,
                    functional_error= False,
                    runtime_error = False,
                    message = ""
                )
            else:
                raise ValueError(f"Unknown benchmark name: {benchmark_name}")
            if test_report.passed:
                passed += 1
        
        node.performance.score = passed / len(node.instance.llm_response)
        node.performance.metric = "pass_test"
        node.num_instances = len(node.instance.llm_response)
        node.instance.models = model._model
        
        return node
        
    except Exception as e:
        print(f"Error processing node {node.ids}: {e}")
        # Return node with default values
        node.performance.score = 0.0
        node.performance.metric = "pass_test"
        return node


def process_instance_tree(instance: Instance, model: LlmClient, system_prompt: str, 
                         output_dir: str, num_samples: int, run_parallel: bool, nl_input : bool = False, benchmark_name : str = "Quixbugs") -> MCTTree:
    """
    Process a single instance: build tree, generate responses, and evaluate.
    
    Args:
        instance: The instance to process
        model: The LLM client
        system_prompt: System prompt for generation
        output_dir: Directory to save results
        num_samples: Number of samples to generate
        run_parallel: Whether to use parallel processing for nodes
        
    Returns:
        MCTTree: The processed tree
    """
    instance_name = str(instance.metadata['name']).replace("/", "_")
    filename = os.path.join(output_dir, f"{instance_name}.json")
    
    if nl_input:
        instance_tree = build_nl_tree(instance=instance, code=None)
    else:
        instance_tree = build_tree_logic_optimized(instance=instance, code=None)
    instance_tree.clean_tree()
    
    if os.path.exists(filename):
        try:
            existing_tree = load_tree(filename)
            instance_tree = merge_trees([instance_tree, existing_tree])
        except Exception as e:
            print(f"Warning: Could not load existing tree for {instance_name}: {e}")
    
    print(f"Processing instance: {instance_name}")
    
    nodes_to_process = [node for node in instance_tree.nodes 
                       if len(node.instance.llm_response) == 0]
    
    # Disable parallel processing for HuggingFace models to avoid tokenizer warnings
    from models.models import Platform
    is_huggingface = model._model.platform == Platform.HUGGINGFACE
    use_parallel = run_parallel and not is_huggingface and len(nodes_to_process) > 1
    
    if use_parallel:
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(RUN_PARALLEL, len(nodes_to_process))) as executor:
            future_to_node = {
                executor.submit(process_node, node, model, system_prompt, num_samples, benchmark_name): node.ids 
                for node in nodes_to_process
            }
            
            for future in tqdm(concurrent.futures.as_completed(future_to_node), 
                             total=len(future_to_node), 
                             desc=f"Processing {instance_name}"):
                try:
                    processed_node = future.result()
                    instance_tree.update_node(processed_node.ids, processed_node)
                except Exception as e:
                    node_id = future_to_node[future]
                    print(f"Error processing node {node_id}: {e}")
    else:
        # Process sequentially
        for node in tqdm(nodes_to_process, desc=f"Processing {instance_name}"):
            processed_node = process_node(node, model, system_prompt, num_samples, benchmark_name)
            instance_tree.update_node(processed_node.ids, processed_node)
    
    try:
        instance_tree.save_tree(filename=filename)
    except Exception as e:
        print(f"Error saving tree for {instance_name}: {e}")
    
    return instance_tree

def evaluate_benchmark(benchmark: Benchmark, model: LlmClient, system_prompt: str, 
                      output_dir: str, num_samples: int, run_parallel: bool) -> MCTTree:
    """
    Evaluate the entire benchmark.
    
    Args:
        benchmark: The benchmark to evaluate
        model: The LLM client
        system_prompt: System prompt for generation
        output_dir: Directory to save results
        num_samples: Number of samples to generate
        run_parallel: Whether to use parallel processing
        
    Returns:
        MCTTree: The merged tree from all instances
    """
    processed_trees = []
    
    for instance in benchmark.data:
        try:
            instance_tree = process_instance_tree(
                instance=instance,
                model=model,
                system_prompt=system_prompt,
                output_dir=output_dir,
                num_samples=num_samples,
                run_parallel=run_parallel,
                nl_input=benchmark.type in ["CODE_SYNTHESIS"],
                benchmark_name=benchmark.name
            )
            processed_trees.append(instance_tree)
        except Exception as e:
            print(f"Error processing instance {instance.metadata.get('name', 'unknown')}: {e}")
            continue
    
    # Merge all trees
    if processed_trees:
        main_tree = merge_trees(processed_trees)
        main_tree.clean_tree()
        return main_tree
    else:
        print("Warning: No trees were successfully processed")
        return MCTTree(nodes=[])


def read_system_prompt(file_path: str) -> str:
    """
    Read the system prompt from a file.
    
    Args:
        file_path (str): The path to the system prompt file.
        
    Returns:
        str: The system prompt.
    """
    try:
        with open(file_path, 'r') as file:
            return " ".join(file.read().splitlines())
    except Exception as e:
        print(f"Error reading system prompt from {file_path}: {e}")
        return "You are a helpful assistant that fixes bugs in Python code."


def build_test_generation_instruction(file_path : str, node : MCTNode) -> str:
    """
    Build the test generation instruction based on the node instance.

    Args:
        file_path (str): The path to the instruction file.
        node (MCTNode): The node containing the instance.

    Returns:
        str: The test generation instruction.
    """
    try:
        with open(file_path, 'r') as file:
            instruction = file.read().strip()
        func_name = node.instance.input.rsplit("def ", 1)[-1].split("(", 1)[0].strip()
        instruction = instruction.replace("{func_name}", func_name)
        instruction = instruction.replace("{program}", node.instance.input.strip())
        return instruction
    except Exception as e:
        print(f"Error reading test generation instruction from {file_path}: {e}")
        return "Generate tests for the following code."


if __name__ == "__main__":
    main()