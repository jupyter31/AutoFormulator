"""
MCTS Search Trace Logger

This module provides comprehensive logging of MCTS search traces including:
- Prompts sent to LLMs
- Raw responses from LLMs
- Rewards/values assigned
- Node outcomes (pruned, valid, syntax error, etc.)
- Full search paths (both winning and discarded)
"""

import json
import os
from typing import Dict, List, Any, Optional
from datetime import datetime
import threading


class MCTSTracer:
    """
    Singleton tracer for MCTS search process.
    Captures all LLM interactions, node evaluations, and search decisions.
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(MCTSTracer, cls).__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self.traces: Dict[str, Dict[str, Any]] = {}
        self.current_problem_id: Optional[str] = None
        self.output_file: Optional[str] = None
        self._initialized = True
    
    def start_problem(self, problem_id: str, problem_description: str, output_dir: str, ground_truth: Optional[Any] = None):
        """Initialize tracing for a new problem."""
        self.current_problem_id = problem_id
        self.output_file = os.path.join(output_dir, "mcts_traces.jsonl")
        
        self.traces[problem_id] = {
            "problem_id": problem_id,
            "problem_description": problem_description,
            "ground_truth": ground_truth,
            "timestamp": datetime.now().isoformat(),
            "nodes": {},
            "search_paths": [],
            "final_result": None,
            "metadata": {}
        }
    
    def log_node_creation(self, node_id: str, depth: int, step: str, parent_id: Optional[str] = None):
        """Log the creation of a new node."""
        if self.current_problem_id is None:
            return
        
        trace = self.traces[self.current_problem_id]
        trace["nodes"][node_id] = {
            "node_id": node_id,
            "depth": depth,
            "step": step,
            "parent_id": parent_id,
            "llm_calls": [],
            "formulation": {},
            "code_execution": None,
            "outcome": "created",
            "reward": None,
            "children": [],
            "metadata": {}
        }
        
        if parent_id and parent_id in trace["nodes"]:
            trace["nodes"][parent_id]["children"].append(node_id)
    
    def log_llm_call(self, node_id: str, prompt: str, response: str, call_type: str = "generation"):
        """
        Log an LLM interaction for a node.
        
        Args:
            node_id: Identifier for the node
            prompt: The prompt sent to the LLM
            response: The raw response from the LLM
            call_type: Type of call (generation, ranking, evaluation, etc.)
        """
        if self.current_problem_id is None or node_id not in self.traces[self.current_problem_id]["nodes"]:
            return
        
        node_trace = self.traces[self.current_problem_id]["nodes"][node_id]
        node_trace["llm_calls"].append({
            "call_type": call_type,
            "prompt": prompt,
            "response": response,
            "timestamp": datetime.now().isoformat()
        })
    
    def log_formulation(self, node_id: str, step: str, formulation_str: str, formulation_eval: Dict):
        """Log the formulation produced by a node."""
        if self.current_problem_id is None or node_id not in self.traces[self.current_problem_id]["nodes"]:
            return
        
        node_trace = self.traces[self.current_problem_id]["nodes"][node_id]
        node_trace["formulation"][step] = {
            "formulation_str": formulation_str,
            "formulation_eval": formulation_eval
        }
    
    def log_code_execution(self, node_id: str, code: str, output: str, is_correct: bool, error_msg: Optional[str] = None):
        """Log code execution results for a node."""
        if self.current_problem_id is None or node_id not in self.traces[self.current_problem_id]["nodes"]:
            return
        
        node_trace = self.traces[self.current_problem_id]["nodes"][node_id]
        node_trace["code_execution"] = {
            "code": code,
            "output": output,
            "is_correct": is_correct,
            "error_msg": error_msg
        }
    
    def log_node_outcome(self, node_id: str, outcome: str, reason: Optional[str] = None):
        """
        Log the final outcome for a node.
        
        Outcomes: 'valid', 'pruned', 'empty', 'syntax_error', 'incorrect', 
                  'clustered', 'low_scored', 'selected', 'terminal'
        """
        if self.current_problem_id is None or node_id not in self.traces[self.current_problem_id]["nodes"]:
            return
        
        node_trace = self.traces[self.current_problem_id]["nodes"][node_id]
        node_trace["outcome"] = outcome
        if reason:
            node_trace["metadata"]["outcome_reason"] = reason
    
    def log_node_reward(self, node_id: str, reward: float, reward_type: str = "local"):
        """Log the reward/value assigned to a node."""
        if self.current_problem_id is None or node_id not in self.traces[self.current_problem_id]["nodes"]:
            return
        
        node_trace = self.traces[self.current_problem_id]["nodes"][node_id]
        if node_trace["reward"] is None:
            node_trace["reward"] = {}
        node_trace["reward"][reward_type] = reward
    
    def log_search_path(self, path_nodes: List[str], path_type: str = "rollout", final_reward: Optional[float] = None):
        """
        Log a complete search path through the tree.
        
        Args:
            path_nodes: List of node IDs in the path
            path_type: 'rollout', 'dfs', 'winning', 'discarded'
            final_reward: Final reward for this path
        """
        if self.current_problem_id is None:
            return
        
        trace = self.traces[self.current_problem_id]
        trace["search_paths"].append({
            "path_nodes": path_nodes,
            "path_type": path_type,
            "final_reward": final_reward,
            "timestamp": datetime.now().isoformat()
        })
    
    def log_expansion(self, parent_id: str, children_ids: List[str], selected_ids: List[str], 
                      pruned_ids: List[str], rewards: Optional[Dict[str, float]] = None):
        """Log node expansion details including which children were selected/pruned."""
        if self.current_problem_id is None or parent_id not in self.traces[self.current_problem_id]["nodes"]:
            return
        
        node_trace = self.traces[self.current_problem_id]["nodes"][parent_id]
        node_trace["metadata"]["expansion"] = {
            "all_children": children_ids,
            "selected_children": selected_ids,
            "pruned_children": pruned_ids,
            "rewards": rewards or {}
        }
    
    def set_final_result(self, result: str, best_objective: Optional[float] = None, 
                        winning_path: Optional[List[str]] = None):
        """Set the final result for the current problem."""
        if self.current_problem_id is None:
            return
        
        trace = self.traces[self.current_problem_id]
        trace["final_result"] = {
            "status": result,
            "best_objective": best_objective,
            "winning_path": winning_path,
            "timestamp": datetime.now().isoformat()
        }
    
    def add_metadata(self, key: str, value: Any):
        """Add metadata to the current problem trace."""
        if self.current_problem_id is None:
            return
        
        trace = self.traces[self.current_problem_id]
        trace["metadata"][key] = value
    
    def save_trace(self, append: bool = True):
        """Save the current problem trace to file."""
        if self.current_problem_id is None or self.output_file is None:
            return
        
        trace = self.traces[self.current_problem_id]
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(self.output_file), exist_ok=True)
        
        mode = 'a' if append else 'w'
        with open(self.output_file, mode) as f:
            f.write(json.dumps(trace, default=str) + '\n')
        
        print(f"[TRACER] Saved trace for problem {self.current_problem_id} to {self.output_file}")
    
    def get_trace(self, problem_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Get the trace for a specific problem or the current problem."""
        pid = problem_id or self.current_problem_id
        return self.traces.get(pid)
    
    def clear_current(self):
        """Clear the current problem from memory (after saving)."""
        if self.current_problem_id and self.current_problem_id in self.traces:
            del self.traces[self.current_problem_id]
        self.current_problem_id = None
        self.output_file = None
    
    @staticmethod
    def get_node_id(node) -> str:
        """Generate a unique identifier for a node."""
        # Use the node's directory path as a unique identifier
        if hasattr(node, 'this_dir'):
            return node.this_dir
        elif hasattr(node, 'idx') and hasattr(node, 'this_step'):
            return f"{node.this_step}_{node.idx}"
        else:
            return str(id(node))


# Global tracer instance
_tracer = MCTSTracer()


def get_tracer() -> MCTSTracer:
    """Get the global tracer instance."""
    return _tracer
