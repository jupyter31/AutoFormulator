import json
import os

input_file = 'search_to_reasoning/training_data/sft_data_with_reasoning.jsonl'
output_file = 'search_to_reasoning/training_data/sft_data_filtered.jsonl'

def filter_data():
    print(f"Reading from {input_file}...")
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found.")
        return

    count_total = 0
    count_kept = 0
    rewards_kept = []

    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:
        
        for line in f_in:
            count_total += 1
            try:
                data = json.loads(line)
                reward = data.get('reward', 0.0)
                
                if reward >= 0.8:
                    f_out.write(line)
                    count_kept += 1
                    rewards_kept.append(reward)
            except json.JSONDecodeError:
                print(f"Skipping invalid JSON line: {line[:50]}...")
                continue

    print(f"Total lines processed: {count_total}")
    print(f"Lines kept (reward >= 0.8): {count_kept}")
    if rewards_kept:
        print(f"Min reward kept: {min(rewards_kept)}")
        print(f"Max reward kept: {max(rewards_kept)}")
        print(f"Unique rewards kept: {sorted(list(set(rewards_kept)))}")
    print(f"Filtered data saved to {output_file}")

if __name__ == "__main__":
    filter_data()
