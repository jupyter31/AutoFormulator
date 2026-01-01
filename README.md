# Autoformulation of Mathematical Optimization Models Using LLMs

This repository contains the code of the paper

[Autoformulation of Mathematical Optimization Models Using LLMs](https://arxiv.org/pdf/2411.01679) published at ICML 2025

<img src="figure-1.png" width="1000"/>

## Overview

The repository implements the **Autoformulation of mathematical to** formulate step by step. It includes scripts for two primary experimental tasks:

- **MCTS**: Model formulating step by step mathematical optimization models. Including prunning, ranking self-evaluation, etc.
- **DFS**: A way to construct a dense tree for ablation studies.



### API keys

For OpenAI models, place your key in `utils.py`.

## Running Experiments

### Batch Trace Generation Pipeline

Generate MCTS traces for multiple problems using the `batch_generate_traces.py` script:

#### Basic Usage

```bash
# Run on a range of problems (e.g., problems 0-123)
python batch_generate_traces.py --start 0 --end 50 --output-dir "results_dir" --engine "gpt-4o"

# Run on specific problem indices
python batch_generate_traces.py --indices 0 5 10 15 --output-dir "results_dir" --engine "gpt-4o"

# Run on entire dataset
python batch_generate_traces.py --all --output-dir "results_dir" --engine "gpt-4o"
```

#### Model Configuration Options

**GPT-4o (OpenAI):**
```bash
python batch_generate_traces.py --start 0 --end 50 \
    --output-dir "gpt4o_results" \
    --engine "gpt-4o" \
    --n-used-gpt 4  # 4-shot approach
```

**Tinker (Fine-tuned models):**
```bash
python batch_generate_traces.py --start 0 --end 50 \
    --output-dir "tinker_results" \
    --engine "tinker://682b7b52-9080-58b6-9be7-c91c44b0bdcd:train:0/sampler_weights/final" \
    --n-used-gpt 1  # 1-shot approach
```

**Ollama (Local models):**
```bash
python batch_generate_traces.py --start 0 --end 50 \
    --output-dir "ollama_results" \
    --engine "ollama:deepseek-r1:8b" \
    --n-used-gpt 1
```

#### MCTS Parameters

- `--n-used-gpt`: Number of LLM generations per step (default: 4)
  - `1` for 1-shot (~31 LLM calls per problem, faster)
  - `4` for 4-shot (~124 LLM calls per problem, higher quality)
- `--n-reward`: Number of reward evaluations (default: 3)
- `--n-top`: Number of top candidates to keep (default: 2)

#### Example Experiments

**1-shot GPT-4o (Cost-efficient):**
```bash
python batch_generate_traces.py --start 0 --end 123 \
    --output-dir "gpt4o_single_call_results" \
    --engine "gpt-4o" \
    --n-used-gpt 1
```

**4-shot Tinker v3 (Best fine-tuned model):**
```bash
python batch_generate_traces.py --start 0 --end 123 \
    --output-dir "tinker_4shot_results_v3" \
    --engine "tinker://682b7b52-9080-58b6-9be7-c91c44b0bdcd:train:0/sampler_weights/final" \
    --n-used-gpt 4
```

### Evaluation Metrics Calculation

After generating traces, evaluate the results using `analyze_tinker_results.py`:

```bash
python analyze_tinker_results.py --results-dir "your_results_directory"
```

#### Metrics Explained

The script calculates four key metrics:

1. **Success Rate**: Percentage of problems with at least one optimal solution
   - Formula: `(Problems with ≥1 optimal) / (Total problems attempted)`
   
2. **Completion Rate**: Percentage of problems that finished MCTS exploration
   - Formula: `(Problems completed) / (Total problems attempted)`
   
3. **Quality on Completed**: Average accuracy for problems that completed
   - Formula: `Average(optimal solutions / total attempts)` for completed problems only
   
4. **Overall Correct**: Same as Success Rate (problems solved / total attempted)

#### Example Evaluation

```bash
# Evaluate GPT-4o results
python analyze_tinker_results.py --results-dir "gpt4o_single_call_results"

# Evaluate Tinker results
python analyze_tinker_results.py --results-dir "tinker_4shot_results_v3"

# Evaluate baseline
python analyze_tinker_results.py --results-dir "NL4OPT_results"
```

#### Sample Output

```
Found 123 problem directories
================================================================================
Problem 0: 1/1 optimal (100.0%)
  Ground Truth: 1160.0, Best Found: 1160.0
...
================================================================================

SUMMARY:
Total problems attempted: 123
Problems completed: 87 (70.7%)
Problems failed/incomplete: 36 (29.3%)

Problems with at least one optimal solution: 76/123 (61.8%)
Average success rate (completed only): 68.3%

================================================================================
METRICS BREAKDOWN (treating incomplete as failures):
================================================================================
1. Success Rate (problems solved):          61.8%
2. Completion Rate (problems finished):     70.7%
3. Quality on Completed (avg accuracy):     68.3%
4. Overall Correct (same as #1):            61.8%
```

### Experimental Results Summary

| Model | Configuration | Success Rate | Completion Rate | Quality on Completed |
|-------|--------------|--------------|-----------------|---------------------|
| GPT-4o | 1-shot | 67.3% | 80.8% | 83.3% |
| GPT-4o | 4-shot | 92.7% | N/A | 90.1% |
| Tinker v3 | 1-shot | 14.6% | 17.9% | 81.8% |
| Tinker v3 | 4-shot | 61.8% | 70.7% | 68.3% |

### NL4OPT

Example of running the whole dataset NL4OPT.

```bash
sh run_all_NLP
```

The code is still not fully clean yet. However, all the components are here.

## Citation

```bibtex
@inproceedings{astorgaautoformulation,
  title={Autoformulation of Mathematical Optimization Models Using LLMs},
  author={Astorga, Nicol{\'a}s and Liu, Tennison and Xiao, Yuanzhang and van der Schaar, Mihaela},
  booktitle={Forty-second International Conference on Machine Learning}
}
```

