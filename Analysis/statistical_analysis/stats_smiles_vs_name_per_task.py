"""
Per-(task, model) SMILES vs.\ compound-name paired test.

Section 5.2 of the paper makes graded per-task claims:
    * "OC exhibits the smallest gaps"
    * "OIn, OPl, and SIT show the largest gaps"
    * "multi-label tasks (RATA, ORA) show modest gaps"

This script attaches p-values to those claims by running, for every
(task, model) pair:
    * binary tasks (OC, OPD, OIn, OPl, OS, SIT)  → McNemar's exact test
    * multilabel  (RATA, ORA)                    → Wilcoxon signed-rank on F1

p-values are Holm-corrected within each task across the evaluated models.
For the small-N task SIT (n=30) we additionally report a sign-flip
permutation p-value (asymptotic McNemar can be unreliable below n≈30).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from stats_core import (
    CATEGORY_DISPLAY,
    CATEGORY_ORDER,
    MULTIANSWER_CATS,
    auto_paired_test,
    exclude_models,
    holm_bonferroni,
    load_all_scores,
    permutation_test_paired,
    stars,
)

OUT_DIR = Path(__file__).resolve().parent / "stats_outputs"
OUT_DIR.mkdir(exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--exclude", nargs="*", default=[],
        help="Model identifiers to drop from the analysis (matches CSV stems "
             "in Results/benchmark_results/). Default: include all.",
    )
    args = parser.parse_args()

    scores, meta = load_all_scores()
    scores = exclude_models(scores, args.exclude)
    cats = meta["question_category"].astype(str).str.lower().values
    print(f"[loaded] {len(scores)} models: {sorted(scores.keys())}")

    rows = []
    for cat_lower, cat_disp in CATEGORY_DISPLAY.items():
        mask = cats == cat_lower
        n_q = int(mask.sum())
        is_multilabel = cat_lower in MULTIANSWER_CATS

        # Per-model raw test
        per_task_records = []
        raw_p = []
        for model, vecs in scores.items():
            a = vecs["name"][mask]
            b = vecs["smiles"][mask]
            r = auto_paired_test(a, b)
            # Permutation for small n (e.g. SIT)
            p_perm = None
            if n_q <= 30:
                _, p_perm = permutation_test_paired(a, b)
            per_task_records.append({
                "task": cat_disp,
                "n": n_q,
                "model": model,
                "mean_name": float(a.mean()),
                "mean_smiles": float(b.mean()),
                "diff_pp": 100 * (a.mean() - b.mean()),
                "ci_lo_pp": 100 * r["ci_lo"],
                "ci_hi_pp": 100 * r["ci_hi"],
                "test": r["test"],
                "p_raw": r["p_value"],
                "p_perm": p_perm,
                "n_discordant": r.get("n_discordant"),
                "cohens_d": r["cohens_d"],
                "is_multilabel": is_multilabel,
            })
            raw_p.append(r["p_value"])
        adj = holm_bonferroni(raw_p)
        for rec, padj in zip(per_task_records, adj):
            rec["p_holm"] = padj
        rows.extend(per_task_records)

    df = pd.DataFrame(rows)
    out_csv = OUT_DIR / "smiles_vs_name_per_task.csv"
    df.to_csv(out_csv, index=False)
    print(f"[written] {out_csv}")

    # ---- Per-task summary: how many models show a sig gap, mean gap, range ----
    print("\nPer-task SMILES-vs-Name summary (compound-name minus SMILES):")
    summary_rows = []
    for cat in CATEGORY_ORDER:
        sub = df[df["task"] == cat]
        if sub.empty:
            continue
        n_sig = int((sub["p_holm"] < 0.05).sum())
        n_total = len(sub)
        diffs = sub["diff_pp"].values
        summary_rows.append({
            "task": cat,
            "n_questions": int(sub["n"].iloc[0]),
            "n_models_sig_holm": n_sig,
            "n_models_total": n_total,
            "mean_gap_pp": float(np.mean(diffs)),
            "median_gap_pp": float(np.median(diffs)),
            "min_gap_pp": float(np.min(diffs)),
            "max_gap_pp": float(np.max(diffs)),
        })
    summary = pd.DataFrame(summary_rows)
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    summary.to_csv(OUT_DIR / "smiles_vs_name_per_task_summary.csv", index=False)
    print(f"[written] {OUT_DIR / 'smiles_vs_name_per_task_summary.csv'}")

    # ---- LaTeX heatmap-style table: tasks × top-N models ----
    tex_path = OUT_DIR / "smiles_vs_name_per_task_table.tex"
    # pick the strongest 10 models by question-weighted Name accuracy across tasks
    top_models = (
        df.groupby("model")["mean_name"].mean()
        .sort_values(ascending=False)
        .head(12).index.tolist()
    )
    # also keep an "all models" ranking (used by the transposed variant) so the
    # narrower layout can show every model without page overflow.
    all_models_ranked = (
        df.groupby("model")["mean_name"].mean()
        .sort_values(ascending=False)
        .index.tolist()
    )
    with tex_path.open("w") as fh:
        fh.write("% Auto-generated by stats_smiles_vs_name_per_task.py\n")
        fh.write("\\begin{table}[t]\n\\centering\\tiny\n")
        fh.write("\\caption{Per-task SMILES vs.\\ compound-name paired test, top "
                 f"{len(top_models)} models. Each cell is "
                 "$\\Delta$ (Name $-$ SMILES) in pp; binary tasks use McNemar's "
                 "exact test on per-question correct/incorrect, multilabel tasks "
                 "(RATA, ORA) use the Wilcoxon signed-rank test on per-question "
                 "F1. Holm--Bonferroni $p$-values are corrected within each task. "
                 "\\textsuperscript{***}$p<0.001$, \\textsuperscript{**}$p<0.01$, "
                 "\\textsuperscript{*}$p<0.05$, ns: not significant.}\n")
        fh.write("\\label{tab:smiles_vs_name_per_task}\n")
        ncol = len(top_models)
        fh.write("\\begin{tabular}{l" + "c" * ncol + "}\n\\toprule\n")
        fh.write("Task & " + " & ".join(m.replace('_', ' ') for m in top_models) + " \\\\\n")
        fh.write("\\midrule\n")
        for cat in CATEGORY_ORDER:
            sub = df[df["task"] == cat]
            if sub.empty:
                continue
            cells = []
            for m in top_models:
                row = sub[sub["model"] == m]
                if row.empty:
                    cells.append("—")
                    continue
                r = row.iloc[0]
                marker = stars(r["p_holm"]) or "ns"
                marker_disp = "" if marker == "ns" else f"\\textsuperscript{{{marker}}}"
                cells.append(f"{r['diff_pp']:+.1f}{marker_disp}")
            fh.write(f"{cat} & " + " & ".join(cells) + " \\\\\n")
        fh.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    print(f"[written] {tex_path}")

    # ---- LaTeX heatmap-style table TRANSPOSED: top-12 models × 8 tasks ----
    # Rows = top-12 models (same set as the wide layout), columns = 8 tasks.
    # 9 columns wide, fits comfortably in single-column NeurIPS layout.
    tex_t_path = OUT_DIR / "smiles_vs_name_per_task_table_transposed.tex"
    with tex_t_path.open("w") as fh:
        fh.write("% Auto-generated by stats_smiles_vs_name_per_task.py (transposed layout)\n")
        fh.write("\\begin{table}[t]\n\\centering\\scriptsize\n")
        fh.write("\\setlength{\\tabcolsep}{4pt}\n")
        fh.write("\\caption{Per-task SMILES vs.\\ compound-name paired test, top "
                 f"{len(top_models)} models. Each cell is "
                 "$\\Delta$ (Name $-$ SMILES) in pp; binary tasks use McNemar's "
                 "exact test on per-question correct/incorrect, multilabel tasks "
                 "(RATA, ORA) use the Wilcoxon signed-rank test on per-question "
                 "F1. Holm--Bonferroni $p$-values are corrected within each task. "
                 "\\textsuperscript{***}$p<0.001$, \\textsuperscript{**}$p<0.01$, "
                 "\\textsuperscript{*}$p<0.05$, ns: not significant.}\n")
        fh.write("\\label{tab:smiles_vs_name_per_task}\n")
        fh.write("\\begin{tabular}{l" + "c" * len(CATEGORY_ORDER) + "}\n\\toprule\n")
        fh.write("Model & " + " & ".join(CATEGORY_ORDER) + " \\\\\n")
        fh.write("\\midrule\n")
        for m in top_models:
            cells = []
            for cat in CATEGORY_ORDER:
                row = df[(df["task"] == cat) & (df["model"] == m)]
                if row.empty:
                    cells.append("—"); continue
                r = row.iloc[0]
                marker = stars(r["p_holm"]) or "ns"
                marker_disp = "" if marker == "ns" else f"\\textsuperscript{{{marker}}}"
                cells.append(f"{r['diff_pp']:+.1f}{marker_disp}")
            fh.write(f"{m.replace('_',' ')} & " + " & ".join(cells) + " \\\\\n")
        fh.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    print(f"[written] {tex_t_path}")

    # ---- Console: per-task significance counts (the §5.2 claims test) ----
    print("\nSection 5.2 claim test:")
    for _, r in summary.iterrows():
        print(f"  {r['task']:>4s} (n={int(r['n_questions'])}): "
              f"{int(r['n_models_sig_holm'])}/{int(r['n_models_total'])} models sig (Holm); "
              f"mean Δ = {r['mean_gap_pp']:+.1f} pp, "
              f"range [{r['min_gap_pp']:+.1f}, {r['max_gap_pp']:+.1f}]")


if __name__ == "__main__":
    main()
