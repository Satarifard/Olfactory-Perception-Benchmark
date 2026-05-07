"""
Per-language MMLU vs. RATA residual analysis (§5.5 control).

For each of the 21 RATA languages:
    x = per-language Global-MMLU accuracy (DeepSeek 32K, default temp, 200 q stratified)
    y = per-language RATA F1 (DeepSeek 32K, name-prompt, on the existing
                              translations_results_RATA outputs)

We compute:
    1. Pearson r between MMLU and RATA across the 21 languages, with
       Fisher-z 95% CI and bootstrap 95% CI.
    2. OLS fit y ~ a + b*x, with residuals per language.
    3. Top/bottom residuals, identifying which languages "punch above" or
       "below" their general-capability baseline on olfactory tasks.

Outputs:
    Results/mmlu_vs_rata/per_language_table.csv
    Analysis/stats_outputs/mmlu_vs_rata_scatter.pdf
    Analysis/stats_outputs/mmlu_vs_rata_summary.tex
"""
from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
RATA_DIR = REPO / "Results" / "translations_results_RATA" / "translations_results_deepseek_reasoner_32K"
MMLU_DIR = REPO / "Results" / "multilingual_mmlu"
OUT_DIR = REPO / "Analysis" / "stats_outputs"
OUT_RESULTS_DIR = REPO / "Results" / "mmlu_vs_rata"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Map RATA language → CSV-filename token in translations_results_RATA
LANG_TO_FILETAG = {
    "English": "rata_100",                  # English uses base file
    "French": "rata_100_French",
    "Spanish": "rata_100_Spanish",
    "Russian": "rata_100_Russian",
    "German": "rata_100_German",
    "Italian": "rata_100_Italian",
    "Portuguese": "rata_100_Portuguese",
    "Japanese": "rata_100_Japanese",
    "Turkish": "rata_100_Turkish",
    "Arabic": "rata_100_Arabic",
    "Korean": "rata_100_Korean",
    "Polish": "rata_100_Polish",
    "Greek": "rata_100_Greek",
    "Mandarin Chinese": "rata_100_Mandarin Chinese",
    "Swahili": "rata_100_Swahili",
    "Persian": "rata_100_Persian",
    "Ukranian": "rata_100_Ukranian",
    "Hindi": "rata_100_Hindi",
    "Swedish": "rata_100_Swedish",
    "Egyptian": "rata_100_Egyptian",
    "Bengali": "rata_100_Bengali",
}


def f1_multilabel(pred: list[str], truth: list[str]) -> float:
    P, T = set(pred), set(truth)
    if not P and not T:
        return 1.0
    tp = len(P & T)
    if tp == 0:
        return 0.0
    return 2.0 * tp / (2.0 * tp + len(P - T) + len(T - P))


def parse_label_set(s: object) -> list[str]:
    if pd.isna(s) or s is None:
        return []
    parts = re.split(r"[;\n\r\t]+|,(?!\d)", str(s))
    return [p.strip().lower() for p in parts if p.strip()]


def per_language_rata_f1(lang: str) -> tuple[float, int]:
    """Return (mean per-question multilabel F1, n_questions) for one RATA language."""
    tag = LANG_TO_FILETAG[lang]
    # Find the CSV in the deepseek_reasoner_32K output dir matching this tag.
    candidates = list(RATA_DIR.glob(f"*{tag}_gpt5*.csv")) + list(RATA_DIR.glob(f"*{tag}*.csv"))
    if not candidates:
        return (float("nan"), 0)
    df = pd.read_csv(candidates[0])
    if "answer_to_prompt_2" not in df.columns or "answer" not in df.columns:
        return (float("nan"), 0)
    f1s = []
    for _, row in df.iterrows():
        pred = parse_label_set(row["answer_to_prompt_2"])
        truth = parse_label_set(row["answer"])
        f1s.append(f1_multilabel(pred, truth))
    return (float(np.mean(f1s)) if f1s else float("nan"), len(f1s))


def per_language_mmlu_acc(lang: str) -> tuple[float, int]:
    f = MMLU_DIR / f"{lang.replace(' ', '_')}_DeepSeek_32K_MMLU.csv"
    if not f.exists():
        return (float("nan"), 0)
    df = pd.read_csv(f)
    if "is_correct" not in df.columns:
        return (float("nan"), 0)
    n = int(df["model_answer"].notna().sum())
    nc = int(df["is_correct"].sum())
    return (nc / max(1, n), n)


def pearson_with_ci(x: np.ndarray, y: np.ndarray, n_boot: int = 5000, alpha: float = 0.05):
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    mask = ~np.isnan(x) & ~np.isnan(y)
    x = x[mask]; y = y[mask]
    n = len(x)
    if n < 4:
        return (float("nan"), float("nan"), float("nan"), float("nan"), n)
    xm = x - x.mean(); ym = y - y.mean()
    r = float((xm * ym).sum() / np.sqrt((xm**2).sum() * (ym**2).sum()))
    # Fisher-z CI
    z = np.arctanh(np.clip(r, -0.999999, 0.999999))
    se = 1.0 / np.sqrt(max(n - 3, 1))
    fz_lo = float(np.tanh(z - 1.959963984540054 * se))
    fz_hi = float(np.tanh(z + 1.959963984540054 * se))
    # Bootstrap CI
    rng = np.random.default_rng(0)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        xb, yb = x[idx], y[idx]
        xm_, ym_ = xb - xb.mean(), yb - yb.mean()
        denom = np.sqrt((xm_**2).sum() * (ym_**2).sum())
        boots[i] = (xm_ * ym_).sum() / denom if denom > 0 else 0
    boot_lo = float(np.percentile(boots, 100 * alpha / 2))
    boot_hi = float(np.percentile(boots, 100 * (1 - alpha / 2)))
    # Two-sided p via Fisher-z normal approximation
    p = float(2 * (1 - 0.5 * (1 + np.tanh(abs(z) / se))))
    # safer: use scipy if available
    try:
        from scipy import stats as scistats
        _, p = scistats.pearsonr(x, y)
        p = float(p)
    except Exception:
        pass
    return (r, fz_lo, fz_hi, boot_lo, boot_hi, p, n)


def main():
    rows = []
    for lang in LANG_TO_FILETAG:
        rata_f1, n_rata = per_language_rata_f1(lang)
        mmlu_acc, n_mmlu = per_language_mmlu_acc(lang)
        rows.append({
            "language": lang,
            "mmlu_acc": mmlu_acc,
            "n_mmlu": n_mmlu,
            "rata_f1": rata_f1,
            "n_rata": n_rata,
        })
    df = pd.DataFrame(rows)
    print("Per-language MMLU vs RATA (DeepSeek-Reasoner 32K, default temp):")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # Pearson + bootstrap CI
    x = df["mmlu_acc"].values
    y = df["rata_f1"].values
    r, fz_lo, fz_hi, boot_lo, boot_hi, p, n = pearson_with_ci(x, y)
    print()
    print(f"Pearson r (n={n}): r = {r:+.3f}")
    print(f"  Fisher-z 95% CI : [{fz_lo:+.3f}, {fz_hi:+.3f}]")
    print(f"  Bootstrap 95% CI: [{boot_lo:+.3f}, {boot_hi:+.3f}]")
    print(f"  two-sided p     : {p:.4g}")

    # OLS fit y = a + b*x  (in proportion units, then we'll express in pp)
    mask = ~np.isnan(x) & ~np.isnan(y)
    xv, yv = x[mask], y[mask]
    A = np.vstack([np.ones(len(xv)), xv]).T
    a, b = np.linalg.lstsq(A, yv, rcond=None)[0]
    yhat = a + b * x
    df["pred_rata_f1"] = yhat
    df["residual"] = df["rata_f1"] - df["pred_rata_f1"]
    print()
    print(f"OLS: rata_f1 = {a:+.4f} + {b:+.4f} * mmlu_acc")
    print()
    print("Top residuals (better olfactory than MMLU predicts):")
    print(df.sort_values("residual", ascending=False)[["language", "mmlu_acc", "rata_f1", "pred_rata_f1", "residual"]].head(5).to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print()
    print("Bottom residuals (worse olfactory than MMLU predicts):")
    print(df.sort_values("residual")[["language", "mmlu_acc", "rata_f1", "pred_rata_f1", "residual"]].head(5).to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # Save table
    out_csv = OUT_RESULTS_DIR / "per_language_mmlu_vs_rata.csv"
    df.to_csv(out_csv, index=False)
    print(f"\n[written] {out_csv}")

    # ---- Scatter plot ----
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.scatter(100 * x[mask], 100 * y[mask], s=60, c="#2F4858",
                   edgecolor="white", linewidth=1.2, zorder=3)
        for i, lang in enumerate(df["language"]):
            if not np.isnan(x[i]):
                ax.annotate(lang, (100 * x[i], 100 * y[i]), fontsize=9,
                            xytext=(5, 5), textcoords="offset points")
        # OLS line
        xx = np.linspace(min(xv), max(xv), 50)
        ax.plot(100 * xx, 100 * (a + b * xx), color="#D68C78", linewidth=2,
                label=f"OLS: r = {r:+.2f}, p = {p:.3g}")
        ax.set_xlabel("Per-language Global-MMLU accuracy (%)", fontsize=12)
        ax.set_ylabel("Per-language RATA F1 (%)", fontsize=12)
        ax.set_title("Per-language olfactory performance vs. general LLM capability\n"
                     "DeepSeek-Reasoner 32K (default temp), 21 RATA languages",
                     fontsize=11)
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.legend(loc="best", frameon=False)
        out_pdf = OUT_DIR / "mmlu_vs_rata_scatter.pdf"
        out_png = OUT_DIR / "mmlu_vs_rata_scatter.png"
        fig.tight_layout()
        fig.savefig(out_pdf)
        fig.savefig(out_png, dpi=150)
        print(f"[written] {out_pdf}")
        print(f"[written] {out_png}")
    except Exception as e:
        print(f"[plot skipped] {e}")

    # ---- LaTeX summary ----
    out_tex = OUT_DIR / "mmlu_vs_rata_summary.tex"
    with out_tex.open("w") as fh:
        fh.write("% Auto-generated by stats_mmlu_vs_rata.py\n")
        fh.write("\\begin{table}[t]\\centering\\small\n")
        fh.write("\\caption{Per-language Global-MMLU accuracy and RATA F1 for "
                 "DeepSeek-Reasoner~32K, with OLS residuals. RATA F1 is per-question "
                 "multilabel F1 averaged across 100 RATA questions; MMLU accuracy is "
                 "on a 200-question stratified subsample of Global-MMLU. "
                 f"Pearson $r = {r:+.3f}$, 95\\% bootstrap CI $[{boot_lo:+.3f}, {boot_hi:+.3f}]$, "
                 f"two-sided $p = {p:.3g}$. Residuals identify languages where olfactory "
                 "performance over- or under-shoots its general-capability baseline.}\n")
        fh.write("\\label{tab:mmlu_vs_rata}\n")
        fh.write("\\begin{tabular}{lccccc}\n\\toprule\n")
        fh.write("Language & MMLU (\\%) & RATA F1 (\\%) & Predicted F1 (\\%) & Residual & \\\\\n\\midrule\n")
        for _, r2 in df.sort_values("residual", ascending=False).iterrows():
            fh.write(f"{r2['language']} & {100*r2['mmlu_acc']:.1f} & "
                     f"{100*r2['rata_f1']:.1f} & {100*r2['pred_rata_f1']:.1f} & "
                     f"{100*r2['residual']:+.2f} \\\\\n")
        fh.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    print(f"[written] {out_tex}")


if __name__ == "__main__":
    main()
