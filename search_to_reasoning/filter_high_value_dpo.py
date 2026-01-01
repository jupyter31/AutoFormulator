"""
Filter DPO pairs to keep only high-value learning signals.

High-Value Pairs (KEEP):
- Execution errors: Chosen compiles/solves, Rejected throws errors
- Low-scored outcomes: Rejected has objectively worse MCTS score
- Reward differences: Clear numerical preference signal

Low-Value Pairs (REMOVE):
- "clustered": Cosmetic differences (variable names, formatting)
- Pure stylistic preferences with no semantic difference
"""

import json
from pathlib import Path
from collections import Counter

def is_high_value_pair(pair):
    """
    Determine if a DPO pair provides meaningful learning signal.
    
    High-value criteria:
    1. Rejected has execution error or low MCTS score
    2. Rejected has significantly lower reward
    3. Rejected is marked as "terminal" failure
    
    Low-value criteria:
    1. "clustered" - likely cosmetic differences
    2. Both are valid but differ only in style
    """
    reason = pair['reason']
    
    # KEEP: Pairs with clear quality difference
    high_value_indicators = [
        'low_scored',           # Rejected has worse MCTS score
        'terminal',             # Rejected is terminal failure
        'Lower reward:',        # Explicit reward difference
    ]
    
    # REMOVE: Pairs with only cosmetic differences
    low_value_indicators = [
        'clustered',            # Just different valid alternatives
    ]
    
    # Check if reason contains high-value indicators
    for indicator in high_value_indicators:
        if indicator in reason:
            return True
    
    # Check if reason contains low-value indicators
    for indicator in low_value_indicators:
        if indicator in reason:
            return False
    
    # Default: keep if unclear
    return True

def main():
    # Load DPO pairs
    input_file = Path("search_to_reasoning/training_data/dpo_pairs.jsonl")
    output_file = Path("search_to_reasoning/training_data/dpo_pairs_high_value.jsonl")
    
    print("Loading DPO pairs...")
    pairs = [json.loads(line) for line in open(input_file, encoding='utf-8')]
    
    print(f"Original DPO pairs: {len(pairs)}")
    
    # Analyze rejection reasons
    reasons = Counter([p['reason'] for p in pairs])
    print(f"\n{'='*80}")
    print("Rejection Reason Distribution:")
    print('='*80)
    for reason, count in reasons.most_common():
        print(f"{reason:50s} {count:6d} ({count/len(pairs)*100:5.1f}%)")
    
    # Filter for high-value pairs
    high_value_pairs = [p for p in pairs if is_high_value_pair(p)]
    removed_pairs = len(pairs) - len(high_value_pairs)
    
    print(f"\n{'='*80}")
    print("Filtering Results:")
    print('='*80)
    print(f"Original pairs:       {len(pairs):6d}")
    print(f"High-value pairs:     {len(high_value_pairs):6d} ({len(high_value_pairs)/len(pairs)*100:5.1f}%)")
    print(f"Removed (low-value):  {removed_pairs:6d} ({removed_pairs/len(pairs)*100:5.1f}%)")
    
    # Analyze what we kept vs removed
    kept_reasons = Counter([p['reason'] for p in high_value_pairs])
    removed_reason_pairs = [p for p in pairs if not is_high_value_pair(p)]
    removed_reasons = Counter([p['reason'] for p in removed_reason_pairs])
    
    print(f"\n{'='*80}")
    print("KEPT - High-Value Reason Distribution:")
    print('='*80)
    for reason, count in kept_reasons.most_common():
        print(f"{reason:50s} {count:6d} ({count/len(high_value_pairs)*100:5.1f}%)")
    
    print(f"\n{'='*80}")
    print("REMOVED - Low-Value Reason Distribution:")
    print('='*80)
    for reason, count in removed_reasons.most_common():
        print(f"{reason:50s} {count:6d} ({count/len(removed_reason_pairs)*100:5.1f}%)")
    
    # Save high-value pairs
    print(f"\n{'='*80}")
    print(f"Saving high-value pairs to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        for pair in high_value_pairs:
            f.write(json.dumps(pair) + '\n')
    
    file_size_mb = output_file.stat().st_size / (1024 * 1024)
    print(f"✓ Saved {len(high_value_pairs)} high-value DPO pairs ({file_size_mb:.2f} MB)")
    
    # Print example of what we removed
    print(f"\n{'='*80}")
    print("Example of REMOVED low-value pair (cosmetic difference):")
    print('='*80)
    if removed_reason_pairs:
        example = removed_reason_pairs[0]
        print(f"Problem: {example['problem_id']}, Step: {example['step']}")
        print(f"Reason: {example['reason']}")
        print(f"\nChosen:  {example['chosen'][:200]}...")
        print(f"\nRejected: {example['rejected'][:200]}...")
    
    # Print example of what we kept
    print(f"\n{'='*80}")
    print("Example of KEPT high-value pair (quality difference):")
    print('='*80)
    if high_value_pairs:
        # Find a low_scored example
        low_scored_examples = [p for p in high_value_pairs if 'low_scored' in p['reason']]
        if low_scored_examples:
            example = low_scored_examples[0]
            print(f"Problem: {example['problem_id']}, Step: {example['step']}")
            print(f"Reason: {example['reason']}")
            print(f"\nChosen:  {example['chosen'][:200]}...")
            print(f"\nRejected: {example['rejected'][:200]}...")

if __name__ == "__main__":
    main()
