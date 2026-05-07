# Statistical Analysis

Reproducible statistical pipelines for the Olfactory Perception Benchmark
paper. Each script is self-contained and writes its outputs (CSV summaries
and LaTeX tables) into a sibling `stats_outputs/` directory that is
recreated on each run.

## Requirements

- Python 3.9+
- `numpy`, `pandas`, `scipy`, `matplotlib` (matplotlib only for figure-producing scripts)

## Inputs

All scripts read from the repository's existing data directories at the
repository root:

- `Results/benchmark_results/*_OP_Benchmark.csv` (per-model 1,010-question outputs)
- `Results/translations_results_RATA/translations_results_<model>/*.csv` (per-language RATA, for §5.5)
- `Results/multilingual_mmlu/*.csv` (Global-MMLU subsamples, for §5.5)

## Scripts

Module | Purpose
---|---
`stats_core.py` | Shared helpers: token extraction, paired tests (McNemar, Wilcoxon, paired-t), bootstrap CIs, Holm correction, Cohen's d, Steiger's Z, hierarchical bootstrap.
`stats_smiles_vs_name.py` | Per-model SMILES-vs-Name paired test (McNemar / Wilcoxon) with Holm correction.
`stats_smiles_vs_name_per_task.py` | Per-task × per-model SMILES-vs-Name breakdown.
`stats_per_task_ci.py` | Table-3-style accuracy table with bootstrap 95% CIs and significance vs.\ leader set. Supports `--prompt name` (default) or `--prompt smiles`.
`stats_reasoning_budget.py` | Within-family paired tests for reasoning budget (e.g., DeepSeek 8K vs 16K vs 32K).
`stats_mmlu_vs_rata.py` | DeepSeek-32K Global-MMLU vs RATA F1 correlation (single-model).
`stats_mmlu_vs_rata_multimodel.py` | Same correlation across all 7 multilingual models, plus residual analysis.
`stats_correlation_smiles_vs_name.py` | Steiger's Z test on dependent correlations between continuous-rating predictions and human ratings.
`stats_model_vs_model.py` | Top-K pairwise model-vs-model significance matrices, per task.
`run_all_stats.py` | Runs all of the above in sequence.

## Excluding models

Most scripts accept a `--exclude MODEL [MODEL ...]` flag. Identifiers match
the CSV stems in `Results/benchmark_results/` (e.g. `LlaSMol_7B`,
`ChemLLM_20B`, `OLMo_2_32B`). Default is to include every CSV present.

## Outputs

Each script writes to `stats_outputs/` in this directory:

- CSV files with per-model / per-task / per-language summaries
- LaTeX `.tex` snippets ready to `\input{}` from a paper

Outputs are git-ignored so the repository stays clean; re-run any script to
regenerate them.

## Usage

```bash
cd Analysis/statistical_analysis

# Default: load every CSV in Results/benchmark_results/
python stats_smiles_vs_name.py

# With exclusions
python stats_smiles_vs_name.py --exclude OLMo_2_32B

# All scripts at once
python run_all_stats.py
```
