import sys
sys.path.append("../")

from transformation.invocation import ALL_TRANSFORMATIONS
from transformation.transformation import Transformation
from evaluation.trees.tree import MCTTree, MCTNode, Performance
from models.task_gen import LLMGen
from transformation.code_mutation.no_mutation import NoMutation
from typing import Optional, Set, Dict, List, Tuple, FrozenSet
from data_loader.tasks import Instance
from collections import defaultdict, deque
import hashlib
import heapq
from dataclasses import dataclass
from functools import lru_cache
import concurrent.futures

@dataclass(frozen=True)
class TransformationPath:
    """Immutable representation of a transformation sequence."""
    transformations: Tuple[str, ...]
    code_hash: str
    
    def __hash__(self):
        return hash((self.transformations, self.code_hash))

@dataclass
class NodeCandidate:
    """Represents a potential node before creation."""
    input_code: str
    transformation: str
    parent_path: TransformationPath
    priority: float = 0.0
    
    def __lt__(self, other):
        return self.priority > other.priority  # Higher priority first

class SmartTreeBuilder:
    """
    Logic-optimized tree builder focusing on core algorithmic improvements:
    
    1. **Code Similarity Detection**: Avoid generating similar code variants
    2. **Beam Search**: Focus on most promising paths instead of exhaustive search
    3. **Dynamic Path Pruning**: Stop unproductive exploration early
    4. **Transformation Diversity**: Ensure diverse exploration without redundancy
    5. **Resource-Aware Building**: Intelligent stopping based on diminishing returns
    """
    
    def __init__(self, similarity_threshold: float = 0.95, max_similar_variants: int = 3):
        self.similarity_threshold = similarity_threshold
        self.max_similar_variants = max_similar_variants
        
        # Caches for optimization
        self.similarity_cache: Dict[Tuple[str, str], float] = {}
        self.path_registry: Dict[FrozenSet[str], List[TransformationPath]] = defaultdict(list)
        self.transformation_applicability: Dict[str, Dict[str, bool]] = defaultdict(dict)
        
        # Statistics
        self.stats = {
            'nodes_created': 0,
            'nodes_pruned_similarity': 0,
            'nodes_pruned_redundancy': 0,
            'transformations_skipped': 0
        }
    
    @lru_cache(maxsize=10000)
    def _get_code_hash(self, code: str) -> str:
        """Get normalized hash of code for similarity detection."""
        # Normalize code by removing whitespace variations and comments
        lines = [line.strip() for line in code.split('\n') if line.strip() and not line.strip().startswith('#')]
        normalized = '\n'.join(lines)
        return hashlib.md5(normalized.encode()).hexdigest()[:16]
    
    def _calculate_code_similarity(self, code1: str, code2: str) -> float:
        """Calculate similarity between two code snippets using multiple metrics."""
        if code1 == code2:
            return 1.0
            
        cache_key = (code1[:100], code2[:100])  # Use prefix for cache key
        if cache_key in self.similarity_cache:
            return self.similarity_cache[cache_key]
        
        # Normalize both codes
        def normalize_code(code):
            lines = [line.strip() for line in code.split('\n') if line.strip() and not line.strip().startswith('#')]
            return '\n'.join(lines)
        
        norm1 = normalize_code(code1)
        norm2 = normalize_code(code2)
        
        if not norm1 or not norm2:
            similarity = 0.0
        elif norm1 == norm2:
            similarity = 1.0
        else:
            # Use line-based similarity for code
            lines1 = set(norm1.split('\n'))
            lines2 = set(norm2.split('\n'))
            
            if not lines1 and not lines2:
                similarity = 1.0
            elif not lines1 or not lines2:
                similarity = 0.0
            else:
                intersection = len(lines1 & lines2)
                union = len(lines1 | lines2)
                similarity = intersection / union if union > 0 else 0.0
        
        self.similarity_cache[cache_key] = similarity
        return similarity
    
    def _is_code_too_similar(self, new_code: str, existing_codes: List[str]) -> bool:
        """Check if new code is too similar to existing codes."""
        for existing_code in existing_codes:
            similarity = self._calculate_code_similarity(new_code, existing_code)
            if similarity > self.similarity_threshold:
                self.stats['nodes_pruned_similarity'] += 1
                return True
        return False
    
    def _calculate_transformation_priority(self, trans_class, input_code: str, 
                                         transformation_history: Set[str], depth: int) -> float:
        """Calculate priority score for a transformation based on context."""
        trans_name = trans_class.__name__
        
        # CRITICAL: Skip transformations already used in this path
        # This prevents loops and ensures each transformation is used at most once per path
        if trans_name in transformation_history:
            return 0.0  # Skip already used transformations in this path
        
        priority = 1.0
        
        # Exploration bonus: prefer transformations not heavily used at this depth
        exploration_bonus = 1.0
        similar_count = sum(1 for hist_trans in transformation_history 
                          if hist_trans.startswith(trans_name.split('_')[0]))
        if similar_count == 0:
            exploration_bonus = 1.2  # Slight bonus for unexplored types
        
        # Code length factor: some transformations work better on certain code sizes
        code_length = len(input_code)
        if code_length < 100:  # Short code
            priority *= 1.1
        elif code_length > 1000:  # Long code
            priority *= 0.9  # Slightly prefer for complex code
        
        # Depth factor: prefer simpler transformations at greater depths
        depth_factor = max(0.5, 1.0 - (depth * 0.1))
        priority *= depth_factor
        
        return priority * exploration_bonus
    
    def _should_expand_path(self, path: TransformationPath, depth: int) -> bool:
        """Determine if a path should be expanded further using smart criteria."""
        
        # Basic depth limit
        if depth >= 5:
            return False
        
        # Check for too many similar paths (redundancy prevention)
        path_signature = frozenset(path.transformations)
        similar_paths = self.path_registry.get(path_signature, [])
        if len(similar_paths) >= self.max_similar_variants:
            self.stats['nodes_pruned_redundancy'] += 1
            return False
        
        # Pattern detection: avoid unproductive sequences
        if len(path.transformations) >= 3:
            # Check for immediate reversals (A -> B -> A pattern)
            last_three = path.transformations[-3:]
            if last_three[0] == last_three[2]:  # A-B-A pattern
                return False
            
            # Check for longer cycles (A -> B -> C -> A pattern)
            if len(path.transformations) >= 4:
                for i in range(len(path.transformations) - 3):
                    if path.transformations[i] in path.transformations[i+2:]:
                        return False  # Found a cycle
        
        # CRITICAL: Ensure no duplicate transformations in the entire path
        if len(set(path.transformations)) != len(path.transformations):
            return False  # Duplicate transformations found
        
        # Length-based stopping: very long transformation chains are often unproductive
        if len(path.transformations) > 8:
            return False
        
        return True
    
    def _get_applicable_transformations(self, input_code: str, prog_lang: str) -> List[Tuple[float, type, Transformation]]:
        """Get applicable transformations with their applicability check cached."""
        applicable = []
        
        for trans_class in ALL_TRANSFORMATIONS:
            trans_name = trans_class.__name__
            
            # Check cache first
            code_hash = self._get_code_hash(input_code)
            cache_key = f"{code_hash}_{trans_name}_{prog_lang}"
            
            if cache_key in self.transformation_applicability:
                if not self.transformation_applicability[cache_key]:
                    continue
                # If cached as applicable, still need to create instance
            
            try:
                trans = trans_class(input=input_code, programming_language=prog_lang)
                is_applicable = trans.check()
                
                # Cache the result
                self.transformation_applicability[cache_key] = is_applicable
                
                if is_applicable:
                    applicable.append((1.0, trans_class, trans))  # Base priority of 1.0
                else:
                    self.stats['transformations_skipped'] += 1
                    
            except Exception as e:
                # Cache as not applicable
                print("Failed transformation " + trans_name + ": Exception occurred" + e)
                self.transformation_applicability[cache_key] = False
                self.stats['transformations_skipped'] += 1
                continue
        
        return applicable
    
    def _generate_smart_candidates(self, node: MCTNode, current_path: TransformationPath,
                                  existing_codes: List[str], depth: int, **kwargs) -> List[NodeCandidate]:
        """Generate candidate nodes using intelligent selection."""
        candidates = []
        prog_lang = kwargs.get('programming_language', 'python')
        transformation_history = set(current_path.transformations)
        
        # Debug: Print current path for loop detection
        if len(current_path.transformations) > 3:
            print(f"Debug: Path at depth {depth}: {' -> '.join(current_path.transformations[-4:])}")
        
        # Get applicable transformations
        applicable_transformations = self._get_applicable_transformations(node.instance.input, prog_lang)
        
        # Calculate priorities and sort
        prioritized_transformations = []
        for base_priority, trans_class, trans in applicable_transformations:
            priority = self._calculate_transformation_priority(
                trans_class, node.instance.input, transformation_history, depth
            )
            if priority > 0:  # Only include if not filtered out
                prioritized_transformations.append((priority, trans_class, trans))
        
        # Sort by priority (highest first) - only compare the priority value
        prioritized_transformations.sort(key=lambda x: x[0], reverse=True)
        
        # Generate candidates from transformations
        codes_in_batch = []
        for priority, trans_class, trans in prioritized_transformations:
            try:
                new_code = trans.apply(**kwargs)
                
                # Skip if too similar to existing code
                if self._is_code_too_similar(new_code, existing_codes + codes_in_batch):
                    continue
                
                codes_in_batch.append(new_code)
                
                new_path = TransformationPath(
                    transformations=current_path.transformations + (trans_class.__name__,),
                    code_hash=self._get_code_hash(new_code)
                )
                
                candidates.append(NodeCandidate(
                    input_code=new_code,
                    transformation=trans_class.__name__,
                    parent_path=new_path,
                    priority=priority
                ))
                
            except Exception:
                continue
        
        return candidates
    
    def build_tree_smart(self, instance: Optional[Instance], code: Optional[str],
                        max_nodes: int = 1000, max_depth: int = 4,
                        beam_width: int = 10, **kwargs) -> MCTTree:
        """
        Build tree using smart logic with beam search and intelligent pruning.
        
        Args:
            instance: Task instance
            code: Code string
            max_nodes: Maximum number of nodes to create
            max_depth: Maximum tree depth
            beam_width: Number of best candidates to keep at each level
            **kwargs: Additional transformation arguments
        
        Returns:
            MCTTree: Optimized tree with intelligent path selection
        """
        tree = MCTTree([])
        
        # Create root
        root_input = instance.input if instance else code
        root_node = MCTNode(
            ids=0,
            instance=LLMGen(
                input=root_input,
                ref_output=instance.ref_output if instance else None,
                metadata=instance.metadata if instance else None,
                models=None,
                llm_response=[]
            ),
            performance=Performance(metric=None, score=0.0),
            transformation=NoMutation.__name__,
            perplexity=0.0,
            parent_node=None,
            process=False
        )
        tree.add_node(root_node)
        
        # Initialize with root path
        root_path = TransformationPath(
            transformations=(NoMutation.__name__,),
            code_hash=self._get_code_hash(root_input)
        )
        
        # Track nodes by depth for beam search
        nodes_by_depth: Dict[int, List[Tuple[float, MCTNode, TransformationPath]]] = defaultdict(list)
        nodes_by_depth[0] = [(1.0, root_node, root_path)]
        existing_codes = [root_input]
        
        depth = 0
        while depth < max_depth and tree.get_number_of_nodes() < max_nodes:
            current_level = nodes_by_depth.get(depth, [])
            if not current_level:
                break
            
            next_level_candidates = []
            
            # Process each node at current depth
            for priority, node, path in current_level:
                if not self._should_expand_path(path, depth):
                    continue
                
                # Generate smart candidates
                candidates = self._generate_smart_candidates(
                    node, path, existing_codes, depth, **kwargs
                )
                
                # Create nodes for valid candidates
                for candidate in candidates:
                    if tree.get_number_of_nodes() >= max_nodes:
                        break
                    
                    new_node = MCTNode(
                        ids=tree.get_number_of_nodes(),
                        instance=LLMGen(
                            input=candidate.input_code,
                            ref_output=node.instance.ref_output,
                            metadata=node.instance.metadata,
                            models=None,
                            llm_response=[]
                        ),
                        performance=Performance(metric=None, score=0.0),
                        transformation=candidate.transformation,
                        perplexity=0.0,
                        parent_node=node.ids,
                        process=False
                    )
                    
                    tree.add_node(new_node)
                    existing_codes.append(candidate.input_code)
                    
                    next_level_candidates.append((
                        candidate.priority, new_node, candidate.parent_path
                    ))
                    
                    # Register path
                    path_sig = frozenset(candidate.parent_path.transformations)
                    self.path_registry[path_sig].append(candidate.parent_path)
                    
                    self.stats['nodes_created'] += 1
            
            # Beam search: keep only top candidates for next level
            next_level_candidates.sort(key=lambda x: x[0], reverse=True)  # Sort by priority
            nodes_by_depth[depth + 1] = next_level_candidates[:beam_width]
            
            depth += 1
            
            if tree.get_number_of_nodes() % 100 == 0:
                print(f"Depth {depth}: {tree.get_number_of_nodes()} nodes, "
                      f"beam width: {len(nodes_by_depth.get(depth, []))}")
        
        # Mark all nodes as processed
        for node in tree.nodes:
            node.process = True
        
        #print(f"Smart tree building completed:")
        #print(f"  Nodes created: {self.stats['nodes_created']}")
        #print(f"  Pruned (similarity): {self.stats['nodes_pruned_similarity']}")
        #print(f"  Pruned (redundancy): {self.stats['nodes_pruned_redundancy']}")
        #print(f"  Transformations skipped: {self.stats['transformations_skipped']}")
        #print(f"  Final tree size: {tree.get_number_of_nodes()}")
        #print(f"  Terminal nodes: {len(tree.get_terminal_nodes())}")
        #print(f"  Similarity cache size: {len(self.similarity_cache)}")
        
        # Debug: Check for any loops in the final tree
        self._debug_check_loops(tree)
        
        return tree
    
    def _debug_check_loops(self, tree: MCTTree):
        """Debug method to detect and report any loops in the tree."""
        loop_count = 0
        for node in tree.get_terminal_nodes():
            path = tree.get_path_from_node(node.ids)
            transformations = [n.transformation for n in path]
            
            # Check for duplicates in this path
            if len(set(transformations)) != len(transformations):
                loop_count += 1
                
                # Find the duplicate
                seen = set()
                for i, trans in enumerate(transformations):
                    if trans in seen:
                        break
                    seen.add(trans)
        

class AdaptiveTreeBuilder:
    """
    Adaptive tree builder that learns from transformation success patterns.
    """
    
    def __init__(self):
        self.transformation_success_rates: Dict[str, float] = defaultdict(lambda: 1.0)
        self.transformation_usage_count: Dict[str, int] = defaultdict(int)
        self.code_pattern_success: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(lambda: 1.0))
    
    def _update_transformation_metrics(self, transformation: str, code_pattern: str, success: bool):
        """Update metrics for a transformation based on its success."""
        # Update global success rate
        current_rate = self.transformation_success_rates[transformation]
        usage_count = self.transformation_usage_count[transformation]
        
        # Weighted average with more weight on recent results
        weight = min(0.1, 1.0 / (usage_count + 1))
        self.transformation_success_rates[transformation] = (
            (1 - weight) * current_rate + weight * (1.0 if success else 0.0)
        )
        
        # Update pattern-specific success
        pattern_rate = self.code_pattern_success[code_pattern][transformation]
        self.code_pattern_success[code_pattern][transformation] = (
            (1 - weight) * pattern_rate + weight * (1.0 if success else 0.0)
        )
        
        self.transformation_usage_count[transformation] += 1
    
    def _get_code_pattern(self, code: str) -> str:
        """Extract a simple pattern from code for learning."""
        lines = [line.strip() for line in code.split('\n') if line.strip()]
        
        # Simple pattern based on code structure
        patterns = []
        for line in lines[:5]:  # Look at first 5 lines
            if 'def ' in line:
                patterns.append('function')
            elif 'class ' in line:
                patterns.append('class')
            elif 'for ' in line or 'while ' in line:
                patterns.append('loop')
            elif 'if ' in line:
                patterns.append('conditional')
            elif '=' in line and not line.startswith(' '):
                patterns.append('assignment')
        
        return '_'.join(patterns[:3]) if patterns else 'simple'
    
    def _get_adaptive_priority(self, trans_class, input_code: str) -> float:
        """Get priority based on learned success patterns."""
        trans_name = trans_class.__name__
        code_pattern = self._get_code_pattern(input_code)
        
        # Base priority from global success rate
        global_success = self.transformation_success_rates[trans_name]
        
        # Pattern-specific success rate
        pattern_success = self.code_pattern_success[code_pattern][trans_name]
        
        # Combine with slight preference for pattern-specific learning
        adaptive_priority = 0.7 * global_success + 0.3 * pattern_success
        
        # Exploration bonus for rarely used transformations
        usage_count = self.transformation_usage_count[trans_name]
        if usage_count < 5:
            adaptive_priority *= 1.1  # Small exploration bonus
        
        return adaptive_priority
    
    def build_tree_adaptive(self, instance: Optional[Instance], code: Optional[str],
                           **kwargs) -> MCTTree:
        """Build tree with adaptive learning from previous builds."""
        smart_builder = SmartTreeBuilder()
        
        # Override priority calculation with adaptive version
        original_calc = smart_builder._calculate_transformation_priority
        
        def adaptive_calc(trans_class, input_code, transformation_history, depth):
            base_priority = original_calc(trans_class, input_code, transformation_history, depth)
            if base_priority <= 0:  # Respect filtering decisions
                return 0.0
            
            adaptive_factor = self._get_adaptive_priority(trans_class, input_code)
            return base_priority * adaptive_factor
        
        smart_builder._calculate_transformation_priority = adaptive_calc
        
        # Build tree
        tree = smart_builder.build_tree_smart(instance, code, **kwargs)
        
        # Learn from results
        root_input = instance.input if instance else code
        root_pattern = self._get_code_pattern(root_input)
        
        for node in tree.nodes:
            if node.transformation != NoMutation.__name__:
                # Simple success metric: node was created and not pruned
                success = True  # In this context, creation means success
                self._update_transformation_metrics(node.transformation, root_pattern, success)
        
        return tree

# Main interface functions
def build_tree_logic_optimized(instance: Optional[Instance], code: Optional[str],
                              strategy: str = 'smart', **kwargs) -> MCTTree:
    """
    Build tree with logic optimizations focused on algorithmic improvements.
    
    Args:
        instance: Task instance
        code: Code string  
        strategy: 'smart' for intelligent pruning, 'adaptive' for learning-based
        **kwargs: Additional arguments (max_nodes, max_depth, beam_width, etc.)
    
    Returns:
        MCTTree: Logic-optimized tree
    """
    if strategy == 'adaptive':
        builder = AdaptiveTreeBuilder()
        return builder.build_tree_adaptive(instance, code, **kwargs)
    else:  # smart strategy
        builder = SmartTreeBuilder()
        return builder.build_tree_smart(instance, code, **kwargs)

# Backward compatibility
def build_tree_smart(instance: Optional[Instance], code: Optional[str], **kwargs) -> MCTTree:
    """Backward compatibility wrapper for smart tree building."""
    return build_tree_logic_optimized(instance, code, strategy='smart', **kwargs)