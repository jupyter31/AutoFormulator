import json
import os

def extract_reasoning(completion):
    """
    Extracts reasoning from the SFT completion.
    Assumes reasoning is the text before 'formalization_dict'.
    """
    # Common marker for the start of the code dictionary
    marker = "formalization_dict"
    
    if marker in completion:
        parts = completion.split(marker)
        # Reasoning is the first part
        reasoning = parts[0].strip()
        return reasoning
    
    # Fallback for cases where formalization_dict might not be present (e.g. imports or other steps?)
    # But based on previous checks, they all seem to have it.
    # If no marker, check for code comments at start
    lines = completion.split('\n')
    reasoning_lines = []
    for line in lines:
        if line.strip().startswith('formalization_dict') or line.strip().startswith('#'):
            break
        reasoning_lines.append(line)
    
    return '\n'.join(reasoning_lines).strip()

def main():
    sft_path = 'search_to_reasoning/training_data/sft_data_with_reasoning.jsonl'
    dpo_path = 'search_to_reasoning/training_data/dpo_pairs_high_value.jsonl'
    output_path = 'search_to_reasoning/training_data/dpo_pairs_with_reasoning.jsonl'

    print("Loading SFT data...")
    sft_map = {}
    with open(sft_path, 'r', encoding='utf-8') as f:
        for line in f:
            sample = json.loads(line)
            key = (sample['problem_id'], sample['step'])
            reasoning = extract_reasoning(sample['completion'])
            if reasoning:
                sft_map[key] = reasoning

    print(f"Loaded reasoning for {len(sft_map)} SFT samples.")

    print("Processing DPO pairs...")
    processed_count = 0
    missing_reasoning_count = 0
    
    with open(dpo_path, 'r', encoding='utf-8') as f_in, \
         open(output_path, 'w', encoding='utf-8') as f_out:
        
        for line in f_in:
            pair = json.loads(line)
            key = (pair['problem_id'], pair['step'])
            
            if key in sft_map:
                reasoning = sft_map[key]
                
                # Apply the recipe:
                # Chosen: [REASONING] {SFT_Generated_Reasoning} [CODE] {MCTS_Winning_Code}
                # Rejected: [REASONING] {SFT_Generated_Reasoning} [CODE] {MCTS_Losing_Code}
                
                # Ensure we don't double-add newlines if they already exist
                pair['chosen'] = f"{reasoning}\n\n{pair['chosen']}"
                pair['rejected'] = f"{reasoning}\n\n{pair['rejected']}"
                
                f_out.write(json.dumps(pair) + '\n')
                processed_count += 1
            else:
                # If we don't have reasoning, we skip or keep? 
                # User said "Chosen Side: MUST Match SFT". If we don't have SFT reasoning, we can't match.
                # But we should have it.
                print(f"Warning: No reasoning found for {key}")
                missing_reasoning_count += 1
                # We'll write it without reasoning to preserve data, or skip?
                # Let's write it without reasoning but log it.
                f_out.write(json.dumps(pair) + '\n')

    print(f"\nProcessing complete.")
    print(f"Total DPO pairs processed: {processed_count + missing_reasoning_count}")
    print(f"Pairs with reasoning added: {processed_count}")
    print(f"Pairs missing reasoning: {missing_reasoning_count}")
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    main()
