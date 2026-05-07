"""
Per-language MMLU vs. RATA residual analysis across ALL 7 multilingual models.

Extends stats_mmlu_vs_rata.py from a single-model (DeepSeek 32K) snapshot to
the full set used in paper Figure 4c:

    DeepSeek-Reasoner (32K), Grok 4.1 Fast, GPT-5 (high), GPT-5 Pro,
    GPT-5.2 Pro, Claude Opus 4.5 (high), Gemini 2.5 Pro (32K).

For each model we compute:
    1. per-language MMLU accuracy (200-q stratified Global-MMLU)
    2. per-language RATA F1 (multilabel)
    3. Pearson r(MMLU, RATA) across the 21 languages, with bootstrap 95% CI
    4. OLS y = a + b*x; per-language residual

Then the cross-model summary:
    * average per-language MMLU and RATA F1 across the 7 models
    * Pearson r on the average (more stable, less noisy)
    * residual stability: rank correlation of per-language residuals between
      pairs of models — if residuals agree across providers, the
      olfactory-vocabulary explanation is robust.

Outputs:
    Results/mmlu_vs_rata/per_language_mmlu_vs_rata_multimodel.csv
    Analysis/stats_outputs/mmlu_vs_rata_multimodel_scatter.pdf/.png
    Analysis/stats_outputs/mmlu_vs_rata_multimodel_summary.tex
"""
from __future__ import annotations

import re
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
RATA_ROOT = REPO / "Results" / "translations_results_RATA"
MMLU_DIR = REPO / "Results" / "multilingual_mmlu"
BENCH_DIR = REPO / "Results" / "benchmark_results"
OUT_DIR = REPO / "Analysis" / "stats_outputs"
OUT_RESULTS_DIR = REPO / "Results" / "mmlu_vs_rata"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# Per-model bookkeeping: (display name, RATA dir name, RATA file prefix,
# MMLU file tag, OP-benchmark CSV stem for English-RATA fallback)
MODELS = [
    ("DeepSeek 32K",        "translations_results_deepseek_reasoner_32K",
        "deepseek_reasoner",        "DeepSeek_32K",          "Deepseek_32K"),
    ("Grok 4.1 Fast",       "translations_results_grok",
        "grok_4_1_fast_reasoning",  "Grok_4_1_fast",         "Grok_4_1_fast"),
    ("GPT-5 high",          "translations_results_GPT5_high",
        "gpt5_high",                "GPT_5_high",            "GPT_5_high"),
    ("GPT-5 Pro",           "translations_results_GPT5_pro_high",
        "gpt5_pro_high",            "GPT_5_pro",             "GPT_5_pro"),
    ("GPT-5.2 Pro",         "translations_results_GPT5.2_pro",
        "gpt5.2_pro_high",          "GPT_5.2_pro",           "GPT_5.2_pro"),
    ("Claude Opus 4.5",     "translations_results_claude_opus_4.5",
        "claude_opus_4.5_high",     "Claude_opus_4.5_high",  "Claude_opus_4.5"),
    ("Gemini 2.5 Pro 32K",  "translations_results_gemini2_5_pro",
        "gemini2_5_pro",            "Gemini_2.5_pro_32768",  "Gemini_2.5_pro_32768"),
]

LANGS = ["English","French","Spanish","Russian","German","Italian","Portuguese",
         "Japanese","Turkish","Arabic","Korean","Polish","Greek","Mandarin Chinese",
         "Swahili","Persian","Ukranian","Hindi","Swedish","Egyptian","Bengali"]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _split_set(cell) -> set[str]:
    if pd.isna(cell) or cell is None:
        return set()
    parts = re.split(r"[;\n\r\t]+|,(?!\d)", str(cell))
    return {p.strip().lower() for p in parts if p.strip()}


def f1_multilabel(pred: set, truth: set) -> float:
    if not pred and not truth:
        return 1.0
    tp = len(pred & truth)
    if tp == 0:
        return 0.0
    fp = len(pred - truth); fn = len(truth - pred)
    return 2.0 * tp / (2.0 * tp + fp + fn)


def per_language_rata_f1(model_dir: str, file_prefix: str, lang: str,
                          english_op_stem: str) -> tuple[float, int]:
    """Mean per-question multilabel F1 for a (model, language) pair."""
    if lang == "English":
        # English isn't in the translations dir — pull RATA rows from the OP CSV.
        op = pd.read_csv(BENCH_DIR / f"{english_op_stem}_OP_Benchmark.csv")
        sub = op[op["question_category"].astype(str).str.lower() == "rata"]
        if sub.empty or "answer_to_prompt_2" not in sub.columns:
            return float("nan"), 0
        f1s = []
        for _, r in sub.iterrows():
            pred = _split_set(r["answer_to_prompt_2"])
            truth = _split_set(r["answer"])
            f1s.append(f1_multilabel(pred, truth))
        return float(np.mean(f1s)), len(f1s)

    # Non-English: glob the per-language file inside the model's RATA dir
    candidates = list((RATA_ROOT / model_dir).glob(f"{file_prefix}_rata_100_{lang}*.csv"))
    if not candidates:
        return float("nan"), 0
    df = pd.read_csv(candidates[0])
    if "answer_to_prompt_2" not in df.columns or "answer" not in df.columns:
        return float("nan"), 0
    f1s = []
    for _, r in df.iterrows():
        pred = _split_set(r["answer_to_prompt_2"])
        truth = _split_set(r["answer"])
        f1s.append(f1_multilabel(pred, truth))
    return float(np.mean(f1s)) if f1s else float("nan"), len(f1s)


def per_language_mmlu_acc(mmlu_tag: str, lang: str) -> tuple[float, int]:
    f = MMLU_DIR / f"{lang.replace(' ', '_')}_{mmlu_tag}_MMLU.csv"
    if not f.exists():
        return float("nan"), 0
    df = pd.read_csv(f)
    if "is_correct" not in df.columns:
        return float("nan"), 0
    n = int(df["model_answer"].notna().sum())
    nc = int(df["is_correct"].sum())
    return (nc / max(1, n)), n


def pearson_with_boot_ci(x: np.ndarray, y: np.ndarray,
                          n_boot: int = 5000) -> tuple[float, float, float, float, int]:
    """Returns (r, fz_lo, fz_hi, p, n) using Fisher-z + scipy if available."""
    mask = ~np.isnan(x) & ~np.isnan(y)
    x, y = x[mask], y[mask]
    n = len(x)
    if n < 4:
        return (float("nan"),) * 4 + (n,)
    xm, ym = x - x.mean(), y - y.mean()
    r = float((xm * ym).sum() / np.sqrt((xm**2).sum() * (ym**2).sum()))
    z = np.arctanh(np.clip(r, -0.999999, 0.999999))
    se = 1.0 / np.sqrt(max(n - 3, 1))
    lo = float(np.tanh(z - 1.959963984540054 * se))
    hi = float(np.tanh(z + 1.959963984540054 * se))
    try:
        from scipy import stats as scistats
        _, p = scistats.pearsonr(x, y); p = float(p)
    except Exception:
        p = float(2 * (1 - 0.5 * (1 + np.tanh(abs(z) / max(se, 1e-12)))))
    return (r, lo, hi, p, n)


def ols_residuals(x: np.ndarray, y: np.ndarray) -> tuple[float, float, np.ndarray]:
    mask = ~np.isnan(x) & ~np.isnan(y)
    if mask.sum() < 2:
        return (float("nan"), float("nan"), np.full_like(y, np.nan))
    xv, yv = x[mask], y[mask]
    A = np.vstack([np.ones(len(xv)), xv]).T
    a, b = np.linalg.lstsq(A, yv, rcond=None)[0]
    yhat = a + b * x
    return float(a), float(b), y - yhat


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> None:
    rows = []
    for disp, ratadir, prefix, mmlu_tag, en_op in MODELS:
        for lang in LANGS:
            mmlu_acc, n_mmlu = per_language_mmlu_acc(mmlu_tag, lang)
            rata_f1, n_rata = per_language_rata_f1(ratadir, prefix, lang, en_op)
            rows.append({
                "model": disp, "language": lang,
                "mmlu_acc": mmlu_acc, "n_mmlu": n_mmlu,
                "rata_f1": rata_f1, "n_rata": n_rata,
            })

    df = pd.DataFrame(rows)
    out_csv = OUT_RESULTS_DIR / "per_language_mmlu_vs_rata_multimodel.csv"
    df.to_csv(out_csv, index=False)
    print(f"[written] {out_csv}")

    # ---- per-model regressions ----
    print("\nPer-model regressions:")
    print(f"{'model':<22s}  {'r':>6s}  {'95% CI':>16s}  {'p':>8s}  {'n':>3s}")
    print("-" * 64)
    per_model_summary = []
    per_model_resid = {}        # model -> {language: residual_pp}
    per_model_lines = {}        # model -> dict for plotting (a, b, x, y)
    for disp, *_ in MODELS:
        sub = df[df["model"] == disp].sort_values("language")
        x = sub["mmlu_acc"].values; y = sub["rata_f1"].values
        r, lo, hi, p, n = pearson_with_boot_ci(x, y)
        a, b, resid = ols_residuals(x, y)
        per_model_summary.append({
            "model": disp, "r": r, "ci_lo": lo, "ci_hi": hi, "p": p, "n": n,
            "ols_a": a, "ols_b": b,
        })
        per_model_resid[disp] = dict(zip(sub["language"], 100 * resid))
        per_model_lines[disp] = {
            "x": x, "y": y, "a": a, "b": b, "language": sub["language"].tolist(),
        }
        print(f"{disp:<22s}  {r:+.3f}  [{lo:+.2f}, {hi:+.2f}]  {p:.3g}  {n:3d}")

    pms = pd.DataFrame(per_model_summary)
    pms.to_csv(OUT_RESULTS_DIR / "mmlu_vs_rata_per_model.csv", index=False)
    print(f"\n[written] {OUT_RESULTS_DIR / 'mmlu_vs_rata_per_model.csv'}")

    # ---- cross-model average regression ----
    avg = (df.groupby("language", as_index=False)
              .agg({"mmlu_acc": "mean", "rata_f1": "mean"})
              .sort_values("language"))
    print("\nCross-model average per-language MMLU vs RATA F1:")
    print(avg.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    x_avg = avg["mmlu_acc"].values; y_avg = avg["rata_f1"].values
    r, lo, hi, p, n = pearson_with_boot_ci(x_avg, y_avg)
    a, b, resid_avg = ols_residuals(x_avg, y_avg)
    print(f"\nCross-model average:  r = {r:+.3f},  95% CI [{lo:+.2f}, {hi:+.2f}],  p = {p:.3g}")
    print(f"OLS:  rata_f1 = {a:+.4f} + {b:+.4f} * mmlu_acc")

    avg_with_resid = avg.assign(predicted_f1=a + b * x_avg, residual=y_avg - (a + b * x_avg))
    avg_csv = OUT_RESULTS_DIR / "mmlu_vs_rata_cross_model_average.csv"
    avg_with_resid.to_csv(avg_csv, index=False)
    print(f"[written] {avg_csv}")

    print("\nTop residuals (over-perform on RATA vs MMLU baseline, cross-model average):")
    top = avg_with_resid.sort_values("residual", ascending=False).head(7)
    print(top[["language", "mmlu_acc", "rata_f1", "predicted_f1", "residual"]]
          .to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\nBottom residuals:")
    bot = avg_with_resid.sort_values("residual").head(7)
    print(bot[["language", "mmlu_acc", "rata_f1", "predicted_f1", "residual"]]
          .to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    # ---- residual robustness: do all 7 models agree on which languages over- /
    # under-perform? Compute pairwise Spearman of residuals across models. ----
    resid_df = pd.DataFrame(per_model_resid).reindex(LANGS)
    resid_df.to_csv(OUT_RESULTS_DIR / "mmlu_vs_rata_residuals_by_model.csv")
    print(f"\n[written] {OUT_RESULTS_DIR / 'mmlu_vs_rata_residuals_by_model.csv'}")
    try:
        from scipy.stats import spearmanr
        models_list = list(per_model_resid)
        print("\nPairwise Spearman of per-language residuals across models "
              "(robustness check):")
        for i, m1 in enumerate(models_list):
            for m2 in models_list[i + 1:]:
                v1 = resid_df[m1].values; v2 = resid_df[m2].values
                mask = ~np.isnan(v1) & ~np.isnan(v2)
                if mask.sum() < 4:
                    continue
                rho, _ = spearmanr(v1[mask], v2[mask])
                print(f"  {m1:<22s} vs {m2:<22s}  rho = {rho:+.3f}")
    except Exception as e:
        print(f"  [skipped Spearman: {e}]")

    # ---- Plot: cross-model average scatter ----
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7.5, 6.5))
        # scatter cross-model average
        ax.scatter(100 * x_avg, 100 * y_avg, s=80, c="#2F4858",
                   edgecolor="white", linewidth=1.2, zorder=3,
                   label="Cross-model average (7 models)")
        for i, lang in enumerate(avg["language"]):
            if not (np.isnan(x_avg[i]) or np.isnan(y_avg[i])):
                ax.annotate(lang, (100 * x_avg[i], 100 * y_avg[i]),
                            fontsize=8, xytext=(5, 4), textcoords="offset points")
        # OLS line of cross-model average
        xv = np.linspace(x_avg[~np.isnan(x_avg)].min(),
                          x_avg[~np.isnan(x_avg)].max(), 50)
        ax.plot(100 * xv, 100 * (a + b * xv), color="#D68C78", lw=2.4,
                label=f"OLS (avg): r = {r:+.2f}, p = {p:.3g}")
        # Light per-model scatter overlay
        cmap = plt.cm.tab10
        for i, (m, payload) in enumerate(per_model_lines.items()):
            ax.scatter(100 * payload["x"], 100 * payload["y"],
                       s=10, alpha=0.25, color=cmap(i % 10),
                       label=m if i < 7 else None, zorder=2)
        ax.set_xlabel("Per-language Global-MMLU accuracy (%)", fontsize=12)
        ax.set_ylabel("Per-language RATA F1 (%)", fontsize=12)
        ax.set_title("Per-language olfactory performance vs.\\ general LLM capability\n"
                     "21 RATA languages × 7 multilingual models", fontsize=11)
        ax.grid(True, linestyle=":", alpha=0.4)
        ax.legend(loc="lower right", fontsize=8, frameon=False)
        out_pdf = OUT_DIR / "mmlu_vs_rata_multimodel_scatter.pdf"
        out_png = OUT_DIR / "mmlu_vs_rata_multimodel_scatter.png"
        fig.tight_layout(); fig.savefig(out_pdf); fig.savefig(out_png, dpi=150)
        print(f"\n[written] {out_pdf}")
        print(f"[written] {out_png}")
    except Exception as e:
        print(f"[plot skipped] {e}")

    # ---- LaTeX summary ----
    out_tex = OUT_DIR / "mmlu_vs_rata_multimodel_summary.tex"
    with out_tex.open("w") as fh:
        fh.write("% Auto-generated by stats_mmlu_vs_rata_multimodel.py\n")
        fh.write("\\begin{table}[t]\\centering\\small\n")
        fh.write(
            "\\caption{Per-language Global-MMLU vs.\\ RATA F1 across all seven "
            "multilingual models from Figure~4c. Pearson $r$ is reported per model "
            "and on the cross-model average. The OLS regression $\\text{RATA F1} = "
            f"{a:+.3f} + {b:+.3f}\\,\\text{{MMLU acc}}$ on the cross-model average "
            f"yields $r = {r:+.3f}$ ($p = {p:.3g}$, 95\\% CI $[{lo:+.2f}, {hi:+.2f}]$). "
            "Residuals identify languages whose olfactory performance over- or "
            "under-shoots their general-capability baseline. Compound-name "
            "prompts; default temperature.}\n")
        fh.write("\\label{tab:mmlu_vs_rata_multimodel}\n")
        fh.write("\\begin{tabular}{lcccccc}\n\\toprule\n")
        fh.write("Model & Pearson $r$ & 95\\% CI & $p$ & $n$ & OLS $a$ & OLS $b$ \\\\\n")
        fh.write("\\midrule\n")
        for row in per_model_summary:
            fh.write(f"{row['model']} & {row['r']:+.3f} & "
                     f"[{row['ci_lo']:+.2f}, {row['ci_hi']:+.2f}] & "
                     f"{row['p']:.3g} & {row['n']:d} & "
                     f"{row['ols_a']:+.3f} & {row['ols_b']:+.3f} \\\\\n")
        fh.write("\\midrule\n")
        fh.write(f"\\textbf{{Cross-model average}} & {r:+.3f} & "
                 f"[{lo:+.2f}, {hi:+.2f}] & {p:.3g} & {n:d} & "
                 f"{a:+.3f} & {b:+.3f} \\\\\n")
        fh.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    print(f"[written] {out_tex}")


if __name__ == "__main__":
    main()
