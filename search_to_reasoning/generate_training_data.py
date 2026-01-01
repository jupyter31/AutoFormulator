"""
Generate SFT and DPO training data from MCTS traces
Fixed version that works with the actual trace structure
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Tuple, Any

class MCTSTrainingDataGenerator:
    """Generate SFT and DPO training data from MCTS traces"""
    
    def __init__(self):
        self.sft_data = []
        self.dpo_pairs = []
        
    def parse_trace_directory(self, traces_dir: str) -> Tuple[List[Dict], List[Dict]]:
        """Parse directory containing problem trace files"""
        all_sft = []
        all_dpo = []
        
        traces_path = Path(traces_dir)
        
        # Iterate through all problem directories
        problem_dirs = sorted([d for d in traces_path.iterdir() if d.is_dir()])
        print(f"Found {len(problem_dirs)} problem directories")
        
        for problem_dir in problem_dirs:
            trace_file = problem_dir / 'mcts_traces.jsonl'
            if not trace_file.exists():
                continue
            
            try:
                with open(trace_file, 'r', encoding='utf-8') as f:
                    line = f.readline()
                    if not line.strip():
                        continue
                    
                    trace_data = json.loads(line)
                    sft, dpo = self.parse_single_trace(trace_data)
                    all_sft.extend(sft)
                    all_dpo.extend(dpo)
                    
                    if len(all_sft) % 50 == 0 and len(all_sft) > 0:
                        print(f"Processed {problem_dir.name}: {len(all_sft)} SFT samples, {len(all_dpo)} DPO pairs so far")
                        
            except Exception as e:
                print(f"Error processing {problem_dir.name}: {e}")
        
        return all_sft, all_dpo
    
    def parse_single_trace(self, trace_data: Dict) -> Tuple[List[Dict], List[Dict]]:
        """Parse a single problem trace to extract training data"""
        
        problem_id = trace_data.get('problem_id')
        problem_desc = trace_data.get('problem_description', '')
        nodes = trace_data.get('nodes', {})
        search_paths = trace_data.get('search_paths', [])
        
        sft_data = []
        dpo_pairs = []
        
        # Find winning paths (paths that reached optimal solution)
        winning_paths = []
        for path_info in search_paths:
            path = path_info.get('path_nodes', [])  # Use 'path_nodes' key
            if not path:
                continue
                
            # Check if this path found a solution
            final_node_id = path[-1]
            final_node = nodes.get(final_node_id, {})
            
            # A winning path has a terminal node with correct execution
            if (final_node.get('outcome') == 'terminal' and 
                final_node.get('code_execution', {}) and
                final_node.get('code_execution', {}).get('is_correct')):
                winning_paths.append(path)
        
        # Process each winning path
        for path in winning_paths:
            # Generate SFT data from the winning path
            for i, node_id in enumerate(path[:-1]):  # Skip terminal node
                node = nodes.get(node_id, {})
                next_node_id = path[i + 1]
                next_node = nodes.get(next_node_id, {})
                
                # Get LLM calls from this node
                llm_calls = node.get('llm_calls', [])
                if not llm_calls:
                    continue
                
                # Process each LLM call
                for llm_call in llm_calls:
                    prompt = llm_call.get('prompt', '')
                    response = llm_call.get('response', '')
                    
                    # Extract the chosen formulation
                    chosen = self._extract_formulation_from_response(
                        response, next_node.get('formulation', {})
                    )
                    
                    if chosen:
                        reward_val = 0
                        reward = next_node.get('reward')
                        if isinstance(reward, dict):
                            reward_val = reward.get('local', 0)
                        
                        sft_data.append({
                            'problem_id': problem_id,
                            'problem_description': problem_desc,
                            'step': next_node.get('step', 'unknown'),
                            'prompt': prompt,
                            'completion': chosen,
                            'reward': reward_val
                        })
                
                # Generate DPO pairs from siblings
                children = node.get('children', [])
                if len(children) > 1 and next_node_id in children:
                    dpo = self._generate_dpo_pairs(node, next_node_id, nodes, problem_id, problem_desc)
                    dpo_pairs.extend(dpo)
        
        # Also extract DPO pairs from failed paths for negative examples
        for node_id, node in nodes.items():
            if node.get('outcome') in ['low_scored', 'clustered']:
                parent_id = node.get('parent_id')
                if not parent_id:
                    continue
                
                parent = nodes.get(parent_id, {})
                siblings = parent.get('children', [])
                
                # Find a successful sibling
                for sibling_id in siblings:
                    if sibling_id == node_id:
                        continue
                    sibling = nodes.get(sibling_id, {})
                    if sibling.get('outcome') in ['selected', 'terminal']:
                        # Create DPO pair
                        llm_calls = parent.get('llm_calls', [])
                        if llm_calls:
                            prompt = llm_calls[0].get('prompt', '')
                            
                            dpo_pairs.append({
                                'problem_id': problem_id,
                                'problem_description': problem_desc,
                                'step': node.get('step', 'unknown'),
                                'prompt': prompt,
                                'chosen': self._get_node_formulation(sibling),
                                'rejected': self._get_node_formulation(node),
                                'reason': f"Rejected: {node.get('outcome')} | Chosen: {sibling.get('outcome')}"
                            })
                        break
        
        return sft_data, dpo_pairs
    
    def _extract_formulation_from_response(self, response: str, formulation: Dict) -> str:
        """Extract formulation from LLM response - handle multiple candidates and preserve reasoning"""
        
        # The response contains multiple candidates separated by ---RESPONSE SEPARATOR---
        candidates = response.split('---RESPONSE SEPARATOR---')
        
        # Try to match with the formulation that was actually used
        if formulation:
            formulation_str = self._get_formulation_string(formulation)
            
            # Find the candidate that best matches
            for candidate in candidates:
                candidate = candidate.strip()
                
                # Check if there's reasoning text before the code block
                code_start_idx = candidate.find('```')
                reasoning_text = ""
                
                if code_start_idx > 0:
                    # Extract reasoning text before code block
                    reasoning_text = candidate[:code_start_idx].strip()
                
                # Extract code from candidate
                code_pattern = r'```python(.*?)```'
                matches = re.findall(code_pattern, candidate, re.DOTALL)
                if matches:
                    code = matches[0].strip()
                    
                    # Simple heuristic: if they share common variable names/patterns
                    if formulation_str and any(
                        key_word in formulation_str for key_word in ['parameters', 'decision_variables', 'objective', 'constraints']
                    ):
                        # Combine reasoning text (if any) with code
                        if reasoning_text:
                            return f"{reasoning_text}\n\n{code}"
                        return code
        
        # Fallback: return first valid candidate with reasoning
        for candidate in candidates:
            candidate = candidate.strip()
            
            code_start_idx = candidate.find('```')
            reasoning_text = ""
            
            if code_start_idx > 0:
                reasoning_text = candidate[:code_start_idx].strip()
            
            code_pattern = r'```python(.*?)```'
            matches = re.findall(code_pattern, candidate, re.DOTALL)
            if matches:
                code = matches[0].strip()
                # Combine reasoning text (if any) with code
                if reasoning_text:
                    return f"{reasoning_text}\n\n{code}"
                return code
        
        # Last resort: return formulation string if available
        if formulation:
            return self._get_formulation_string(formulation)
        
        return response.strip()[:500]  # Truncate if nothing works
    
    def _get_node_formulation(self, node: Dict) -> str:
        """Extract formulation string from node"""
        formulation = node.get('formulation', {})
        if formulation:
            result = self._get_formulation_string(formulation)
            if result:
                return result
        return f"# Empty formulation for node: {node.get('node_id', 'unknown')}"
    
    def _get_formulation_string(self, formulation: Dict) -> str:
        """Convert formulation dict to string"""
        parts = []
        for key in ['parameters', 'decision_variables', 'objective', 'equality_constraints', 'inequality_constraints']:
            if key in formulation and formulation[key]:
                form_data = formulation[key]
                if isinstance(form_data, dict) and 'formulation_str' in form_data:
                    parts.append(form_data['formulation_str'])
        return '\n\n'.join(parts) if parts else ''
    
    def _generate_dpo_pairs(self, parent_node: Dict, chosen_id: str, 
                           all_nodes: Dict, problem_id: str, problem_desc: str) -> List[Dict]:
        """Generate DPO pairs by comparing chosen child with siblings"""
        
        dpo_pairs = []
        children = parent_node.get('children', [])
        chosen_node = all_nodes.get(chosen_id, {})
        
        for sibling_id in children:
            if sibling_id == chosen_id:
                continue
            
            sibling = all_nodes.get(sibling_id, {})
            
            # Check if sibling is worse
            should_reject = False
            reason = ""
            
            if sibling.get('outcome') in ['low_scored', 'clustered']:
                should_reject = True
                reason = f"Outcome: {sibling.get('outcome')}"
            else:
                # Compare rewards
                chosen_reward = 0
                sibling_reward = 0
                
                chosen_rew = chosen_node.get('reward')
                if isinstance(chosen_rew, dict):
                    chosen_reward = chosen_rew.get('local', 0)
                
                sibling_rew = sibling.get('reward')
                if isinstance(sibling_rew, dict):
                    sibling_reward = sibling_rew.get('local', 0)
                
                if sibling_reward < chosen_reward:
                    should_reject = True
                    reason = f"Lower reward: {sibling_reward:.2f} vs {chosen_reward:.2f}"
            
            if should_reject:
                llm_calls = parent_node.get('llm_calls', [])
                if llm_calls:
                    prompt = llm_calls[0].get('prompt', '')
                    
                    dpo_pairs.append({
                        'problem_id': problem_id,
                        'problem_description': problem_desc,
                        'step': chosen_node.get('step', 'unknown'),
                        'prompt': prompt,
                        'chosen': self._get_node_formulation(chosen_node),
                        'rejected': self._get_node_formulation(sibling),
                        'reason': reason
                    })
        
        return dpo_pairs

def main():
    """Example usage"""
    
    generator = MCTSTrainingDataGenerator()
    
    # Find trace directory
    trace_dir = None
    for possible_path in ["batch_traces_results",
                          "quick_test_output",
                          "test_batch"]:
        if Path(possible_path).exists() and Path(possible_path).is_dir():
            trace_dir = possible_path
            break
    
    if not trace_dir:
        print("No trace directory found!")
        return
    
    print(f"Processing traces from: {trace_dir}")
    print("=" * 80)
    
    # Generate training data
    sft_data, dpo_pairs = generator.parse_trace_directory(trace_dir)
    
    print("\n" + "=" * 80)
    print(f"✓ Generated {len(sft_data)} SFT samples")
    print(f"✓ Generated {len(dpo_pairs)} DPO pairs (before filtering)")
    
    # Save to files
    output_dir = Path("search_to_reasoning/training_data")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save SFT data
    sft_file = output_dir / "sft_data.jsonl"
    with open(sft_file, 'w', encoding='utf-8') as f:
        for item in sft_data:
            f.write(json.dumps(item) + '\n')
    print(f"\nSaved SFT data to: {sft_file}")
    
    # Filter out identical pairs (chosen == rejected)
    filtered_dpo_pairs = [item for item in dpo_pairs if item['chosen'] != item['rejected']]
    print(f"\nFiltered DPO pairs: {len(dpo_pairs)} -> {len(filtered_dpo_pairs)} (removed {len(dpo_pairs) - len(filtered_dpo_pairs)} identical pairs)")
    
    # Save DPO data
    dpo_file = output_dir / "dpo_pairs.jsonl"
    with open(dpo_file, 'w', encoding='utf-8') as f:
        for item in filtered_dpo_pairs:
            f.write(json.dumps(item) + '\n')
    print(f"Saved DPO data to: {dpo_file}")
    
    # Print samples
    if sft_data:
        print("\n" + "="*80)
        print("Sample SFT Entry:")
        print("-"*80)
        sample = sft_data[0]
        print(f"Problem: {sample['problem_id']}")
        print(f"Step: {sample['step']}")
        print(f"Prompt: {sample['prompt'][:200]}...")
        print(f"Completion: {sample['completion'][:200]}...")
        print(f"Reward: {sample.get('reward', 'N/A')}")
    
    if dpo_pairs:
        print("\n" + "="*80)
        print("Sample DPO Pair:")
        print("-"*80)
        # Use filtered_dpo_pairs if available, otherwise original
        sample_pairs = filtered_dpo_pairs if 'filtered_dpo_pairs' in locals() and filtered_dpo_pairs else dpo_pairs
        if sample_pairs:
            sample = sample_pairs[0]
            print(f"Problem: {sample['problem_id']}")
            print(f"Step: {sample['step']}")
            print(f"Prompt: {sample['prompt'][:200]}...")
            print(f"Chosen: {sample['chosen'][:200]}...")
            print(f"Rejected: {sample['rejected'][:200]}...")
            print(f"Reason: {sample['reason']}")
        else:
            print("No valid DPO pairs after filtering")

if __name__ == "__main__":
    main()
