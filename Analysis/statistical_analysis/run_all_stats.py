"""
One-shot runner: regenerates every statistical analysis used in the paper's
significance section. Outputs land in Analysis/stats_outputs/.

Adds, on top of the existing Analysis.ipynb (which only does descriptive
bootstrap CIs on per-task accuracy):

  1. SMILES vs Name paired tests, per model (McNemar binary, Wilcoxon F1)
     — Holm-corrected across the 21 models.
  2. Top-K model-vs-model significance matrices, per task (McNemar / Wilcoxon)
     — Holm-corrected within each task.
  3. Reasoning-budget paired tests within each family (GPT-5, Gemini, Grok,
     DeepSeek, Claude Opus 4.6) — Holm-corrected within family.
  4. Per-task accuracy with 95% bootstrap CIs and significance vs. the top
     model — Holm-corrected over 20 pairs per task. Drop-in replacement for
     paper Table 3.
"""
from __future__ import annotations

import importlib
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

MODULES = [
    "stats_smiles_vs_name",
    "stats_model_vs_model",
    "stats_reasoning_budget",
    "stats_per_task_ci",
]


def main() -> None:
    for name in MODULES:
        print("\n" + "=" * 78)
        print(f"=== {name}")
        print("=" * 78)
        mod = importlib.import_module(name)
        mod.main()
    print("\nAll outputs in:", HERE / "stats_outputs")


if __name__ == "__main__":
    main()
