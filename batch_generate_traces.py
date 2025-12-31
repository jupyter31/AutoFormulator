"""
Batch Trace Generation Script

Generate MCTS traces for multiple problems from the dataset.

Usage:
    python batch_generate_traces.py --start 0 --end 5
    python batch_generate_traces.py --indices 0 1 2 5 10
    python batch_generate_traces.py --all  # Run on entire dataset
"""

import argparse
import pandas as pd
import os
import sys
import json
from datetime import datetime

# Load VS Code settings for environment variables
def load_vscode_settings():
    settings_path = os.path.join('.vscode', 'settings.json')
    if os.path.exists(settings_path):
        with open(settings_path, 'r') as f:
            settings = json.load(f)
            for key, value in settings.items():
                if key.isupper() and isinstance(value, str):
                    os.environ[key] = value
        print("Loaded environment variables from VS Code settings")
    else:
        print("Warning: .vscode/settings.json not found")

load_vscode_settings()

from MCTS_used import MCTS


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Batch generate MCTS traces")
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--start', type=int, help='Start index (inclusive)')
    group.add_argument('--indices', type=int, nargs='+', help='Specific indices to run')
    group.add_argument('--all', action='store_true', help='Run on all problems')
    
    parser.add_argument('--end', type=int, help='End index (exclusive), used with --start')
    parser.add_argument('--output-dir', type=str, default='batch_traces_results',
                        help='Output directory for traces')
    parser.add_argument('--data-file', type=str, default='data/NL4OPT.json',
                        help='Path to data file')
    parser.add_argument('--engine', type=str, default='gpt-4o',
                        help='Engine to use (uses DEPLOYMENT_NAME from settings)')
    parser.add_argument('--n-reward', type=int, default=3,
                        help='Number of reward evaluations')
    parser.add_argument('--n-used-gpt', type=int, default=4,
                        help='Number of GPT generations')
    parser.add_argument('--n-top', type=int, default=2,
                        help='Number of top children to keep')
    
    return parser.parse_args()


def load_data(data_file):
    """Load the dataset."""
    if not os.path.exists(data_file):
        print(f"Error: Data file not found: {data_file}")
        sys.exit(1)
    
    df = pd.read_json(data_file, lines=True)
    print(f"Loaded {len(df)} problems from {data_file}")
    return df


def run_single_problem(idx, df, output_base_dir, mcts_params):
    """Run MCTS on a single problem and generate trace."""
    try:
        # Get problem data
        problem_str = df.iloc[idx].en_question
        ground_truth = df.iloc[idx].en_answer
        problem_idx = df.iloc[idx].name if hasattr(df.iloc[idx], 'name') else idx
        
        print(f"\n{'=' * 80}")
        print(f"Problem {idx} (ID: {problem_idx})")
        print(f"{'=' * 80}")
        print(f"Question: {problem_str[:150]}...")
        print(f"Ground Truth: {ground_truth}")
        
        # Create output directory
        output_dir = os.path.join(output_base_dir, f"problem_{problem_idx}")
        os.makedirs(output_dir, exist_ok=True)
        
        # Run MCTS
        print(f"\nStarting MCTS search...")
        start_time = datetime.now()
        
        mcts = MCTS(**mcts_params)
        mcts.dfs_from_scratch(
            problem_str,
            output_dir,
            ground_truth=ground_truth
        )
        
        elapsed = (datetime.now() - start_time).total_seconds()
        
        # Verify trace was created
        trace_file = os.path.join(output_dir, "mcts_traces.jsonl")
        if os.path.exists(trace_file):
            file_size = os.path.getsize(trace_file)
            print(f"✓ Trace generated: {trace_file} ({file_size:,} bytes)")
            print(f"✓ Time: {elapsed:.1f}s")
            return True, elapsed, None
        else:
            print(f"✗ WARNING: Trace file not found")
            return False, elapsed, "Trace file not created"
            
    except Exception as e:
        print(f"✗ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return False, 0, str(e)


def main():
    """Main entry point."""
    args = parse_args()
    
    # Load data
    df = load_data(args.data_file)
    
    # Determine which indices to run
    if args.all:
        indices = list(range(len(df)))
    elif args.indices:
        indices = args.indices
    else:  # --start
        if args.end is None:
            print("Error: --end required when using --start")
            sys.exit(1)
        indices = list(range(args.start, args.end))
    
    # Validate indices
    invalid_indices = [i for i in indices if i >= len(df)]
    if invalid_indices:
        print(f"Warning: Invalid indices (out of range): {invalid_indices}")
        indices = [i for i in indices if i < len(df)]
    
    if not indices:
        print("Error: No valid indices to process")
        sys.exit(1)
    
    # Setup MCTS parameters
    mcts_params = {
        'engine_used': args.engine,
        'n_reward': args.n_reward,
        'n_used_gpt': args.n_used_gpt,
        'n_top': args.n_top
    }
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Print configuration
    print("\n" + "=" * 80)
    print("BATCH TRACE GENERATION")
    print("=" * 80)
    print(f"Problems to process: {len(indices)}")
    print(f"Indices: {indices}")
    print(f"Output directory: {args.output_dir}")
    print(f"Engine: {args.engine}")
    print(f"Parameters: n_reward={args.n_reward}, n_used_gpt={args.n_used_gpt}, n_top={args.n_top}")
    print("=" * 80)
    
    # Run on each problem
    results = []
    total_time = 0
    
    for i, idx in enumerate(indices, 1):
        print(f"\n[{i}/{len(indices)}] Processing problem {idx}...")
        
        success, elapsed, error = run_single_problem(
            idx, df, args.output_dir, mcts_params
        )
        
        results.append({
            'index': idx,
            'success': success,
            'time': elapsed,
            'error': error
        })
        total_time += elapsed
    
    # Print summary
    print("\n" + "=" * 80)
    print("BATCH PROCESSING COMPLETE")
    print("=" * 80)
    
    successful = sum(1 for r in results if r['success'])
    failed = len(results) - successful
    
    print(f"\nResults:")
    print(f"  Total: {len(results)}")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    print(f"  Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    if successful > 0:
        print(f"  Average time per problem: {total_time/successful:.1f}s")
    
    # List failed problems
    if failed > 0:
        print(f"\nFailed problems:")
        for r in results:
            if not r['success']:
                print(f"  Problem {r['index']}: {r['error']}")
    
    # List generated traces
    print(f"\nGenerated traces:")
    for r in results:
        if r['success']:
            problem_idx = df.iloc[r['index']].name if hasattr(df.iloc[r['index']], 'name') else r['index']
            trace_file = os.path.join(args.output_dir, f"problem_{problem_idx}", "mcts_traces.jsonl")
            if os.path.exists(trace_file):
                file_size = os.path.getsize(trace_file)
                print(f"  Problem {r['index']}: {trace_file} ({file_size:,} bytes)")
    
    print(f"\nAll traces saved to: {args.output_dir}/")
    print("=" * 80 + "\n")
    
    # Return success if all problems succeeded
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
