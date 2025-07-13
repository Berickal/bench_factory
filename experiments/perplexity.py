#!/usr/bin/env python3

import os
import sys
import argparse
import json
import logging
from pathlib import Path
from typing import List, Optional
import torch
from tqdm import tqdm

sys.path.append("../")
from evaluation.trees.tree import MCTTree, MCTNode, load_tree

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    print("Warning: transformers not available")


class SimplePerplexityCalculator:
    """Simple perplexity calculator with batch processing"""
    
    def __init__(self, model_path: str, use_gpu: bool = True):
        self.model_path = model_path
        self.use_gpu = use_gpu and torch.cuda.is_available()
        
        if not HAS_TRANSFORMERS:
            raise ImportError("transformers and torch are required")
        
        self._load_model()
    
    def _load_model(self):
        """Load model and tokenizer"""
        print(f"Loading model: {self.model_path}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model
        device_map = "auto" if self.use_gpu else None
        torch_dtype = torch.float16 if self.use_gpu else torch.float32
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch_dtype,
            device_map=device_map
        )
        
        self.model.eval()
        print(f"Model loaded. Device: {next(self.model.parameters()).device}")
    
    def calculate_perplexity(self, text: str) -> float:
        """Calculate perplexity for a single text"""
        if not text or not text.strip():
            return float('inf')
        
        try:
            # Tokenize
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512
            )
            
            if self.use_gpu:
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            # Calculate perplexity
            with torch.no_grad():
                outputs = self.model(**inputs, labels=inputs["input_ids"])
                loss = outputs.loss
                perplexity = torch.exp(loss).item()
            
            return perplexity
            
        except Exception as e:
            print(f"Error calculating perplexity: {e}")
            return float('inf')
    
    def calculate_batch_perplexity(self, texts: List[str], batch_size: int = 8) -> List[float]:
        """Calculate perplexity for a batch of texts"""
        perplexities = []
        
        # Process in batches
        for i in tqdm(range(0, len(texts), batch_size), desc="Calculating perplexities"):
            batch_texts = texts[i:i + batch_size]
            
            for text in batch_texts:
                perplexity = self.calculate_perplexity(text)
                perplexities.append(perplexity)
        
        return perplexities


def get_text_from_node(node: MCTNode, text_field: str = "llm_response") -> str:
    """Extract text from a node for perplexity calculation"""
    try:
        if text_field == "llm_response":
            if node.instance.llm_response and len(node.instance.llm_response) > 0:
                return str(node.instance.input) + str(node.instance.llm_response[0])
        elif text_field == "input":
            return str(node.instance.input) if node.instance.input else ""
        elif text_field == "ref_output":
            return str(node.instance.ref_output) if node.instance.ref_output else ""
        
        return ""
    except Exception as e:
        print(f"Error extracting text from node {node.ids}: {e}")
        return ""


def update_tree_perplexity(tree: MCTTree, calculator: SimplePerplexityCalculator,
                          text_field: str = "llm_response", batch_size: int = 8) -> MCTTree:
    """Update perplexity values for all nodes in a tree"""
    print(f"Updating perplexity for {len(tree.nodes)} nodes")
    
    # Extract texts from all nodes
    texts = []
    for node in tree.nodes:
        text = get_text_from_node(node, text_field)
        texts.append(text)
    
    # Calculate perplexities in batches
    perplexities = calculator.calculate_batch_perplexity(texts, batch_size)
    
    # Update nodes with new perplexity values
    for node, perplexity in zip(tree.nodes, perplexities):
        node.perplexity = perplexity
    
    return tree


def process_json_files(folder_path: str, model_path: str, 
                      text_field: str = "llm_response",
                      batch_size: int = 8,
                      output_folder: Optional[str] = None,
                      create_backup: bool = True):
    """Process all JSON files in a folder"""
    
    folder = Path(folder_path)
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder_path}")
    
    # Find all JSON files
    json_files = list(folder.glob("*.json"))
    if not json_files:
        print(f"No JSON files found in {folder_path}")
        return
    
    print(f"Found {len(json_files)} JSON files")
    
    # Initialize calculator
    calculator = SimplePerplexityCalculator(model_path)
    
    # Setup output folder
    if output_folder:
        output_path = Path(output_folder)
        output_path.mkdir(exist_ok=True)
    else:
        output_path = folder
    
    # Process each file
    processed = 0
    errors = 0
    total_nodes = 0
    
    for json_file in tqdm(json_files, desc="Processing files"):
        try:
            print(f"\nProcessing: {json_file.name}")
            
            # Create backup if requested
            if create_backup and not output_folder:
                backup_file = json_file.with_suffix(".json.backup")
                backup_file.write_bytes(json_file.read_bytes())
            
            # Load tree
            tree = load_tree(str(json_file))
            
            if not tree.nodes:
                print(f"Empty tree in {json_file.name}, skipping")
                continue
            
            # Update perplexity
            updated_tree = update_tree_perplexity(tree, calculator, text_field, batch_size)
            
            # Save updated tree
            if output_folder:
                output_file = output_path / json_file.name
            else:
                output_file = json_file
            
            updated_tree.save_tree(str(output_file))
            
            processed += 1
            total_nodes += len(updated_tree.nodes)
            print(f"Updated {len(updated_tree.nodes)} nodes")
            
        except Exception as e:
            print(f"Error processing {json_file.name}: {e}")
            errors += 1
            continue
    
    # Print summary
    print(f"\n{'='*50}")
    print("SUMMARY")
    print(f"{'='*50}")
    print(f"Files processed: {processed}")
    print(f"Files with errors: {errors}")
    print(f"Total nodes updated: {total_nodes}")
    if processed > 0:
        print(f"Average nodes per file: {total_nodes / processed:.1f}")


def main():
    parser = argparse.ArgumentParser(description="Simple perplexity calculator for JSON tree files")
    parser.add_argument("--folder", type=str, required=True,
                       help="Folder containing JSON files")
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to HuggingFace model")
    parser.add_argument("--text_field", type=str, default="llm_response",
                       choices=["llm_response", "input", "ref_output"],
                       help="Field to use for perplexity calculation")
    parser.add_argument("--batch_size", type=int, default=8,
                       help="Batch size for processing")
    parser.add_argument("--output_folder", type=str,
                       help="Output folder (updates in place if not specified)")
    parser.add_argument("--no_backup", action="store_true",
                       help="Don't create backup files")
    parser.add_argument("--no_gpu", action="store_true",
                       help="Use CPU only")
    
    args = parser.parse_args()
    
    # Validate inputs
    if not Path(args.folder).exists():
        print(f"Error: Folder does not exist: {args.folder}")
        return 1
    
    print(f"Starting perplexity calculation...")
    print(f"Folder: {args.folder}")
    print(f"Model: {args.model_path}")
    print(f"Text field: {args.text_field}")
    print(f"Batch size: {args.batch_size}")
    print(f"GPU: {not args.no_gpu}")
    
    try:
        process_json_files(
            folder_path=args.folder,
            model_path=args.model_path,
            text_field=args.text_field,
            batch_size=args.batch_size,
            output_folder=args.output_folder,
            create_backup=not args.no_backup
        )
        
        print("\nPerplexity calculation completed successfully!")
        return 0
        
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())