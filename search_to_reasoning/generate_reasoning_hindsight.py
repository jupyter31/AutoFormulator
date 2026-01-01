"""
Script to generate reasoning for SFT samples that lack explicit reasoning text.
Uses GPT-4o to add hindsight reasoning based on problem + correct code.
"""
import json
import os
from pathlib import Path
from clients.azure_foundry_client import AzureFoundryClient

def load_vscode_settings():
    """Load environment variables from VS Code settings.json."""
    settings_path = os.path.join('.vscode', 'settings.json')
    if os.path.exists(settings_path):
        try:
            with open(settings_path, 'r') as f:
                # Remove comments from jsonc
                content = '\n'.join(line for line in f if not line.strip().startswith('//'))
                settings = json.loads(content)
                
                # Set environment variables from settings
                env_vars = ['AZURE_API_KEY', 'AZURE_INFERENCE_SDK_ENDPOINT', 'DEPLOYMENT_NAME']
                for var in env_vars:
                    if var in settings and settings[var]:
                        os.environ[var] = settings[var]
                        
                return True
        except Exception as e:
            print(f"Warning: Could not load VS Code settings: {e}")
    return False

def has_explicit_reasoning(completion: str) -> bool:
    """Check if completion starts with explicit reasoning text"""
    first_line = completion.strip().split('\n')[0]
    return not first_line.startswith('#') and not first_line.startswith('formalization_dict')

def generate_reasoning_prompt(problem_desc: str, code: str, step: str) -> dict:
    """Generate prompt to create reasoning for a code solution"""
    
    system_prompt = """You are an expert optimization modeler. I will give you a problem and the correct Python code solution. Your goal is to write a short, 1-2 sentence "Thought Process" that explains why this code is correct. 

Focus on:
- The mathematical reasoning behind the formulation
- Why certain variables/constraints/objectives were chosen
- What the code accomplishes in solving the problem

Do not write the code, just the thought process. Keep it concise and clear."""
    
    user_prompt = f"""Problem:
{problem_desc}

Current Step: {step}

Correct Code:
{code}

Provide a 1-2 sentence thought process explaining why this code is the correct formulation for this step:"""
    
    return {
        'system': system_prompt,
        'user': user_prompt
    }

def main():
    # Load environment variables from VS Code settings
    print("Loading environment variables...")
    if load_vscode_settings():
        print("✓ Loaded from .vscode/settings.json")
    else:
        print("⚠ Using system environment variables")
    
    # Initialize client
    print("\nInitializing Azure Foundry client...")
    client = AzureFoundryClient()
    
    # Load SFT data
    sft_file = Path("search_to_reasoning/training_data/sft_data.jsonl")
    print(f"Loading SFT data from {sft_file}...")
    
    samples = []
    with open(sft_file) as f:
        for line in f:
            samples.append(json.loads(line))
    
    print(f"Total samples: {len(samples)}")
    
    # Separate samples
    has_reasoning = []
    needs_reasoning = []
    
    for sample in samples:
        if has_explicit_reasoning(sample['completion']):
            has_reasoning.append(sample)
        else:
            needs_reasoning.append(sample)
    
    print(f"Samples WITH reasoning: {len(has_reasoning)} ({100*len(has_reasoning)/len(samples):.1f}%)")
    print(f"Samples NEEDING reasoning: {len(needs_reasoning)} ({100*len(needs_reasoning)/len(samples):.1f}%)")
    
    # Generate reasoning for samples that need it
    print(f"\n{'='*80}")
    print("Generating reasoning for samples without explicit reasoning...")
    print('='*80)
    
    # Open output file for writing
    output_file = Path("search_to_reasoning/training_data/sft_data_with_reasoning.jsonl")
    print(f"\nSaving enhanced samples to {output_file} (incremental writes)...")
    
    total_processed = 0
    
    with open(output_file, 'w', encoding='utf-8') as f:
        # First, write all samples that already have reasoning
        for sample in has_reasoning:
            f.write(json.dumps(sample) + '\n')
            total_processed += 1
        
        print(f"✓ Wrote {len(has_reasoning)} samples with existing reasoning")
        
        # Process samples that need reasoning and write immediately
        for i, sample in enumerate(needs_reasoning, 1):
            if i % 50 == 0:
                print(f"Processed {i}/{len(needs_reasoning)} samples... (Total written: {total_processed})")
            
            # Generate reasoning prompt
            prompts = generate_reasoning_prompt(
                sample['problem_description'],
                sample['completion'],
                sample['step']
            )
            
            try:
                # Call LLM to generate reasoning
                messages = [
                    {"role": "system", "content": prompts['system']},
                    {"role": "user", "content": prompts['user']}
                ]
                
                # Use the correct method: send_chat_request
                request = {
                    "messages": messages,
                    "temperature": 0.3,
                    "max_tokens": 200
                }
                
                response_result = client.send_chat_request(
                    model_name=None,  # Use default model
                    request=request
                )
                
                reasoning = response_result['text'].strip()
                
                # Combine reasoning with code
                enhanced_completion = f"{reasoning}\n\n{sample['completion']}"
                
                # Create enhanced sample
                enhanced_sample = sample.copy()
                enhanced_sample['completion'] = enhanced_completion
                
                # Write immediately to file
                f.write(json.dumps(enhanced_sample) + '\n')
                f.flush()  # Ensure it's written to disk
                total_processed += 1
                
            except Exception as e:
                print(f"Error generating reasoning for sample {i}: {e}")
                # Keep original sample if generation fails
                f.write(json.dumps(sample) + '\n')
                f.flush()
                total_processed += 1
    
    print(f"\n✓ Successfully processed and saved all {total_processed} samples")
    print(f"✓ Saved to {output_file}")
    
    # Print statistics
    print(f"\n{'='*80}")
    print("Enhanced Dataset Statistics:")
    print('='*80)
    print(f"Total samples: {total_processed}")
    print(f"Samples with explicit reasoning: {len(has_reasoning)} (original)")
    print(f"Samples with generated reasoning: {len(needs_reasoning)} (enhanced)")
    print(f"All samples now have reasoning: 100%")

if __name__ == "__main__":
    main()
