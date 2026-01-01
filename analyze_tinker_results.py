"""
Analyze the quality of model results for solved NL4OPT problems.
Usage: python analyze_tinker_results.py [--results-dir RESULTS_DIR]
"""

import os
import json
import pandas as pd
import argparse
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description="Analyze NL4OPT results")
    parser.add_argument("--results-dir", type=str, default="NL4OPT_results",
                        help="Results directory to analyze (default: NL4OPT_results)")
    return parser.parse_args()

args = parse_args()

# Load ground truth
df = pd.read_json("data/NL4OPT.json", lines=True)

results_dir = Path(args.results_dir)
problems = sorted([d for d in results_dir.iterdir() if d.is_dir() and d.name.startswith("problem_")])

print(f"Found {len(problems)} solved problems")
print("=" * 80)

all_results = []

for problem_dir in problems:
    problem_idx = int(problem_dir.name.split("_")[1])
    results_file = problem_dir / "all_results.jsonl"
    
    if not results_file.exists():
        continue
    
    # Read all solution attempts
    solutions = []
    with open(results_file, 'r') as f:
        for line in f:
            solutions.append(json.loads(line))
    
    if not solutions:
        continue
    
    # Get ground truth
    ground_truth = df.iloc[problem_idx].en_answer
    
    # Analyze solutions
    optimal_count = 0
    valid_solutions = []
    
    for sol in solutions:
        obj_val = sol.get('best_objective')
        if obj_val and obj_val != '-':
            try:
                obj_float = float(obj_val)
                valid_solutions.append(obj_float)
                
                # Check if optimal (within 5% tolerance)
                try:
                    gt_float = float(ground_truth)
                    if gt_float != 0:
                        relative_error = abs((obj_float - gt_float) / gt_float)
                        if relative_error <= 0.05:
                            optimal_count += 1
                    else:
                        if abs(obj_float) <= 0.05:
                            optimal_count += 1
                except:
                    pass
            except:
                pass
    
    if valid_solutions:
        result = {
            'problem_idx': problem_idx,
            'ground_truth': ground_truth,
            'total_attempts': len(solutions),
            'valid_solutions': len(valid_solutions),
            'optimal_count': optimal_count,
            'success_rate': optimal_count / len(solutions) * 100,
            'best_objective': min(valid_solutions) if valid_solutions else None,
            'avg_objective': sum(valid_solutions) / len(valid_solutions) if valid_solutions else None
        }
        all_results.append(result)
        
        print(f"Problem {problem_idx}: {optimal_count}/{len(solutions)} optimal ({result['success_rate']:.1f}%)")
        print(f"  Ground Truth: {ground_truth}, Best Found: {result['best_objective']}")

print("=" * 80)
print("\nSUMMARY:")
print(f"Total problems analyzed: {len(all_results)}")

if all_results:
    avg_success_rate = sum(r['success_rate'] for r in all_results) / len(all_results)
    problems_with_optimal = sum(1 for r in all_results if r['optimal_count'] > 0)
    
    print(f"Problems with at least one optimal solution: {problems_with_optimal}/{len(all_results)} ({problems_with_optimal/len(all_results)*100:.1f}%)")
    print(f"Average success rate per problem: {avg_success_rate:.1f}%")
    
    total_attempts = sum(r['total_attempts'] for r in all_results)
    total_optimal = sum(r['optimal_count'] for r in all_results)
    print(f"Overall optimal solutions: {total_optimal}/{total_attempts} ({total_optimal/total_attempts*100:.1f}%)")
    
    print("\n" + "=" * 80)
    print("METRICS BREAKDOWN:")
    print("=" * 80)
    print(f"1. Success Rate (problems with ≥1 optimal): {problems_with_optimal/len(all_results)*100:.1f}%")
    print(f"2. Accuracy (avg % optimal per problem):    {avg_success_rate:.1f}%")
    print(f"3. Overall Correct (all attempts):          {total_optimal/total_attempts*100:.1f}%")
