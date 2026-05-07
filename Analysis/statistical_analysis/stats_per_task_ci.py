"""
Per-task accuracy with 95% bootstrap CIs and significance vs.\ the top model.

This produces a Table-3-style table where each cell shows
    accuracy [CI_lo, CI_hi] (sig vs. top)
For binary tasks the CI is on the raw accuracy (binomial bootstrap); for
multilabel tasks (RATA, ORA) it is on the per-question mean F1. Significance
is from the same paired test as in stats_model_vs_model.py (McNemar / Wilcoxon),
Holm-corrected across the 20 vs.-top comparisons within each task.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from stats_core import (
    CATEGORY_DISPLAY,
    CATEGORY_ORDER,
    MULTIANSWER_CATS,
    auto_paired_test,
    holm_bonferroni,
    load_all_scores,
    stars,
)

OUT_DIR = Path(__file__).resolve().parent / "stats_outputs"
OUT_DIR.mkdir(exist_ok=True)


def bootstrap_mean_ci(
    x: np.ndarray, n_boot: int = 5000, alpha: float = 0.05, rng_seed: int = 0
) -> tuple[float, float, float]:
    rng = np.random.default_rng(rng_seed)
    n = x.size
    boots = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots[i] = x[idx].mean()
    return (
        float(x.mean()),
        float(np.percentile(boots, 100 * alpha / 2)),
        float(np.percentile(boots, 100 * (1 - alpha / 2))),
    )


def hierarchical_bootstrap_overall(
    task_score_vectors: list[np.ndarray],
    n_boot: int = 5000,
    alpha: float = 0.05,
    rng_seed: int = 0,
) -> tuple[float, float, float]:
    """
    Hierarchical bootstrap CI for the task-unweighted Overall score (mean of
    per-task means). Resamples within each task independently, recomputes each
    task's mean from its resample, then averages task means. This respects the
    paper's task-equal-weighted aggregation rather than the question-weighted
    bootstrap that a flat resample would imply.

    See Bestgen 2022 ("Please don't forget the difference and CI when seeking
    SOTA") and the HELM evaluation methodology. Returns (mean, ci_lo, ci_hi)
    on the original score scale (0..1 if `task_score_vectors` are 0..1).
    """
    rng = np.random.default_rng(rng_seed)
    boots = np.empty(n_boot)
    sizes = [v.size for v in task_score_vectors]
    for b in range(n_boot):
        task_means = []
        for v, n in zip(task_score_vectors, sizes):
            idx = rng.integers(0, n, size=n)
            task_means.append(v[idx].mean())
        boots[b] = float(np.mean(task_means))
    point = float(np.mean([v.mean() for v in task_score_vectors]))
    return (
        point,
        float(np.percentile(boots, 100 * alpha / 2)),
        float(np.percentile(boots, 100 * (1 - alpha / 2))),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", choices=["name", "smiles"], default="name",
                    help="Which prompt column to score: 'name' (compound name, "
                         "answer_to_prompt_2) or 'smiles' (isomeric SMILES, "
                         "answer_to_prompt_1).")
    args = ap.parse_args()
    PROMPT = args.prompt
    SUFFIX = "" if PROMPT == "name" else "_smiles"
    PROMPT_LABEL = "compound name" if PROMPT == "name" else "isomeric SMILES"

    scores, meta = load_all_scores()
    cats_lower = meta["question_category"].astype(str).str.lower().values

    rows = []
    for cat_lower, cat_disp in CATEGORY_DISPLAY.items():
        mask = cats_lower == cat_lower
        n_q = int(mask.sum())

        # Per-model bootstrap mean and CI on this task
        per_model_stats = {}
        for m in scores:
            x = scores[m][PROMPT][mask]
            mean, lo, hi = bootstrap_mean_ci(x)
            per_model_stats[m] = {"mean": mean, "lo": lo, "hi": hi}

        # FIX (point #5): define the LEADER SET as all models whose 95% bootstrap
        # CI includes the empirical leader's point estimate. Comparing every
        # other model against a singleton "top" picked on the same data is
        # anti-conservative because the leader's accuracy is upward-biased under
        # the null. Treating the leader as a tied set instead removes that bias.
        empirical_leader = max(per_model_stats, key=lambda m: per_model_stats[m]["mean"])
        leader_mean = per_model_stats[empirical_leader]["mean"]
        leader_set = [
            m for m in per_model_stats
            if per_model_stats[m]["lo"] <= leader_mean <= per_model_stats[m]["hi"]
        ]

        # Pairwise: each non-leader-set model vs ALL leaders; the cell's p_holm
        # is the MIN over leaders (most generous to the model — i.e. "is this
        # model significantly worse than ANY leader?"). Then Holm-correct over
        # the (n_others) tests.
        non_leaders = [m for m in scores if m not in leader_set]
        raw_p = []
        for m in non_leaders:
            ps = []
            for top_m in leader_set:
                r = auto_paired_test(scores[top_m][PROMPT][mask],
                                     scores[m][PROMPT][mask])
                ps.append(r["p_value"])
            raw_p.append(min(ps))
        adj = holm_bonferroni(raw_p) if raw_p else []
        sig_vs_top = {m: "" for m in leader_set}
        for m, p_h in zip(non_leaders, adj):
            sig_vs_top[m] = stars(p_h) or "ns"

        for m in scores:
            s = per_model_stats[m]
            tag = "(top set)" if m in leader_set else sig_vs_top.get(m, "")
            rows.append({
                "category": cat_disp,
                "n": n_q,
                "model": m,
                "mean_pct": 100 * s["mean"],
                "ci_lo_pct": 100 * s["lo"],
                "ci_hi_pct": 100 * s["hi"],
                "is_top": (m in leader_set),
                "leader_set_size": len(leader_set),
                "sig_vs_top": tag,
            })

    df = pd.DataFrame(rows)
    out_csv = OUT_DIR / f"per_task_accuracy_with_CI{SUFFIX}.csv"
    df.to_csv(out_csv, index=False)
    print(f"[written] {out_csv}")

    # ---- Table-3-styled LaTeX (matches Analysis.ipynb cell 11) ----
    # Build per-(model, category) cell strings and per-task overall scores.
    pivot = {m: {} for m in df["model"].unique()}
    for _, r in df.iterrows():
        pivot[r["model"]][r["category"]] = r

    # Compute per-task task-unweighted overall (mean of 8 task means per model)
    # plus a HIERARCHICAL bootstrap CI on Overall (Bestgen 2022 / HELM-style).
    # The flat 1,010-question bootstrap would be question-weighted; we want a
    # task-equal-weighted CI to match the paper's "unweighted mean of eight
    # task scores" definition.
    overall_by_model: dict[str, float] = {}
    overall_ci_by_model: dict[str, tuple[float, float]] = {}
    for m, by_cat in pivot.items():
        vals = [by_cat[cat]["mean_pct"] for cat in CATEGORY_ORDER if cat in by_cat]
        overall_by_model[m] = float(np.mean(vals)) if vals else float("nan")
        # Per-task per-question score vectors for this model
        task_vecs = []
        for cat_lower, cat_disp in CATEGORY_DISPLAY.items():
            if cat_disp not in by_cat:
                continue
            mask_t = cats_lower == cat_lower
            task_vecs.append(scores[m][PROMPT][mask_t])
        if task_vecs:
            _pt, lo, hi = hierarchical_bootstrap_overall(task_vecs)
            overall_ci_by_model[m] = (100.0 * lo, 100.0 * hi)
        else:
            overall_ci_by_model[m] = (float("nan"), float("nan"))

    # Per-column best / second-best (used for bold-best / underline-second alongside leader-set bold)
    col_best, col_second = {}, {}
    for cat in CATEGORY_ORDER + ["Overall"]:
        if cat == "Overall":
            vals = list(overall_by_model.values())
        else:
            vals = [pivot[m][cat]["mean_pct"] for m in pivot if cat in pivot[m]]
        vals = sorted(set(round(v, 6) for v in vals if v == v), reverse=True)
        col_best[cat] = vals[0] if vals else None
        col_second[cat] = vals[1] if len(vals) > 1 else None

    # Pretty model + reasoning splitter (mirrors Analysis.ipynb cell 11's parse_model_reasoning)
    def parse_model_reasoning(base: str):
        m = base.lower()
        if m.startswith("gemini"):
            mm = re.search(r"(\d{4,6})", base)
            if mm:
                n = int(mm.group(1))
                return "Gemini 2.5 Pro", (f"{n//1000}K" if n >= 1000 else str(n))
            return "Gemini 2.5 Pro", "default"
        if m.startswith("grok_3_mini"):
            return "Grok 3 Mini", base.split("_")[-1]
        if m.startswith("grok_4") or m.startswith("grok4"):
            return "Grok 4.1 Fast", "default"
        if m.startswith("claude_opus_4.6"):
            parts = base.split("_")
            return "Claude Opus 4.6", (parts[-1] if len(parts) > 3 else "—")
        if m.startswith("claude_opus_4.5"):
            return "Claude Opus 4.5", "high"
        if m.startswith("claude_sonnet"):
            return "Claude Sonnet 4.5", "—"
        if m.startswith("deepseek"):
            mm = re.search(r"(\d+k)", base, re.IGNORECASE)
            return "DeepSeek Reasoner", mm.group(0).upper() if mm else "—"
        if m.startswith("gpt_5.2") or m.startswith("gpt-5.2"):
            return "GPT-5.2 Pro", "high"
        if m.startswith("gpt_5_pro") or m.startswith("gpt-5-pro"):
            return "GPT-5 Pro", "high"
        if re.match(r"gpt[_\-]?5", m):
            if "low" in m: return "GPT-5", "low"
            if "high" in m: return "GPT-5", "high"
            return "GPT-5", "default"
        if "oss" in m:
            return "GPT-OSS-120B", "high"
        if m.startswith("o3"):
            return "o3", base.split("_")[-1]
        if m.startswith("o4"):
            return ("o4-mini" if "mini" in base else "o4"), base.split("_")[-1]
        if m.startswith("llama"):
            return "Llama 3.3 70B", "—"
        if m.startswith("olmo"):
            return "OLMo 2 32B", "—"
        if m.startswith("chemllm"):
            return "ChemLLM 20B", "high"
        if m.startswith("llasmol"):
            return "LlaSMol 7B", "—"
        return base.replace("_", " "), "—"

    def family_from_model(model_name: str):
        m = model_name.lower()
        if m.startswith("gpt") or m.startswith("o3") or m.startswith("o4"):
            return "openai"
        if m.startswith("claude"):
            return "claude"
        if m.startswith("gemini"):
            return "gemini"
        if m.startswith("grok"):
            return "grok"
        if m.startswith("deepseek") or m.startswith("llama") or m.startswith("olmo"):
            return "open_source"
        if m.startswith("chemllm") or m.startswith("llasmol"):
            return "chemistry_specialised"
        return "other"

    def fmt_cell(rec: pd.Series, cat: str, ci_mode: str = "full") -> str:
        """ci_mode: 'none' | 'short' | 'full' | 'makecell' | 'tiny_colored' | 'colored_no_ci'"""
        v = rec["mean_pct"]
        if not (v == v):
            return "—"
        v_r = round(float(v), 1)
        lo = rec["ci_lo_pct"]
        hi = rec["ci_hi_pct"]
        sig = rec["sig_vs_top"]
        is_best = col_best[cat] is not None and abs(v - col_best[cat]) < 1e-6
        is_second = (col_second[cat] is not None and abs(v - col_second[cat]) < 1e-6
                     and not is_best)

        if ci_mode in ("none", "colored_no_ci"):
            num = f"{v_r:.1f}"
        elif ci_mode == "short":
            num = f"{v_r:.1f}\\,[{int(round(lo))},{int(round(hi))}]"
        elif ci_mode == "tiny_colored":
            # 1-decimal CI but in \tiny font so the cell stays narrow
            num = f"{v_r:.1f}\\,{{\\tiny[{lo:.1f},{hi:.1f}]}}"
        elif ci_mode == "makecell":
            # primary value gets the bold/underline; CI is on a second line
            pass  # constructed below
        else:  # "full"
            num = f"{v_r:.1f}\\,[{lo:.1f},{hi:.1f}]"

        if ci_mode == "makecell":
            primary = f"{v_r:.1f}"
            if is_best:
                primary = f"\\textbf{{{primary}}}"
            elif is_second:
                primary = f"\\underline{{{primary}}}"
            if sig and sig not in ("(top set)", "ns"):
                primary = primary + f"\\textsuperscript{{{sig}}}"
            ci_line = f"{{\\tiny[{lo:.1f},{hi:.1f}]}}"
            return f"\\makecell{{{primary}\\\\{ci_line}}}"

        if ci_mode in ("tiny_colored", "colored_no_ci"):
            # Color the value itself by p-level. No asterisk superscript.
            color = {"***": "red!85!black",
                      "**":  "orange!90!black",
                      "*":   "olive!80!black"}.get(sig)  # None for ns / leader
            inner = num
            if is_best:
                inner = f"\\textbf{{{inner}}}"
            elif is_second:
                inner = f"\\underline{{{inner}}}"
            if color is not None:
                return f"\\textcolor{{{color}}}{{{inner}}}"
            return inner

        if is_best:
            body = f"\\textbf{{{num}}}"
        elif is_second:
            body = f"\\underline{{{num}}}"
        else:
            body = num
        if sig and sig not in ("(top set)", "ns"):
            body = body + f"\\textsuperscript{{{sig}}}"
        return body

    def fmt_overall(model_name: str) -> str:
        v = overall_by_model.get(model_name, float("nan"))
        if not (v == v):
            return "—"
        v_r = round(float(v), 1)
        if col_best["Overall"] is not None and abs(v - col_best["Overall"]) < 1e-6:
            return f"\\textbf{{{v_r:.1f}}}"
        if col_second["Overall"] is not None and abs(v - col_second["Overall"]) < 1e-6:
            return f"\\underline{{{v_r:.1f}}}"
        return f"{v_r:.1f}"

    # Build rows with parsed (Model, Reasoning, Family)
    row_meta = []
    for raw in pivot:
        model_disp, reasoning = parse_model_reasoning(raw)
        family = family_from_model(model_disp)
        row_meta.append({"raw": raw, "Model": model_disp, "Reasoning": reasoning,
                         "Family": family})
    rm_df = pd.DataFrame(row_meta)
    closed = rm_df[~rm_df["Family"].isin(["open_source", "chemistry_specialised"])].copy()
    open_src = rm_df[rm_df["Family"].isin(["open_source", "chemistry_specialised"])].copy()

    family_order = ["openai", "gemini", "grok", "claude", "other",
                    "open_source", "chemistry_specialised"]
    closed["Family"] = pd.Categorical(closed["Family"], categories=family_order, ordered=True)
    closed = closed.sort_values(["Family", "Model", "Reasoning"]).reset_index(drop=True)
    open_src["Family"] = pd.Categorical(open_src["Family"], categories=family_order, ordered=True)
    open_src = open_src.sort_values(["Family", "Model", "Reasoning"]).reset_index(drop=True)

    def row_to_latex(row, ci_mode: str) -> str:
        raw = row["raw"]
        cells = [
            row["Model"],
            row["Reasoning"],
        ] + [fmt_cell(pivot[raw][cat], cat, ci_mode=ci_mode) for cat in CATEGORY_ORDER] + [
            fmt_overall(raw),
        ]
        return " & ".join(cells) + " \\\\"

    def row_to_latex_merged(row, ci_mode: str) -> str:
        """Variant of row_to_latex that merges Reasoning into the Model column."""
        raw = row["raw"]
        reas = row["Reasoning"]
        merged = (f"{row['Model']} ({reas})" if reas not in ("—", "default", "")
                  else row["Model"])
        cells = [merged] + [fmt_cell(pivot[raw][cat], cat, ci_mode=ci_mode)
                            for cat in CATEGORY_ORDER] + [fmt_overall(raw)]
        return " & ".join(cells) + " \\\\"

    def row_to_latex_stacked_ci(row) -> str:
        """
        Two-row layout per model: top row = accuracy + asterisks, bottom row =
        CIs in \\tiny font. Empty model cell in the second row. Used by V8.
        """
        raw = row["raw"]
        reas = row["Reasoning"]
        merged = (f"{row['Model']} ({reas})" if reas not in ("—", "default", "")
                  else row["Model"])
        # Top row — accuracy with bold/underline + asterisk markers (no CI)
        top_cells = [merged]
        # Bottom row — CI in tiny font, no model label, no significance/bold
        bot_cells = [""]
        for cat in CATEGORY_ORDER:
            rec = pivot[raw][cat]
            v = rec["mean_pct"]
            if not (v == v):
                top_cells.append("—")
                bot_cells.append("")
                continue
            v_r = round(float(v), 1)
            sig = rec["sig_vs_top"]
            is_best = col_best[cat] is not None and abs(v - col_best[cat]) < 1e-6
            is_second = (col_second[cat] is not None
                         and abs(v - col_second[cat]) < 1e-6 and not is_best)
            top = f"{v_r:.1f}"
            if is_best:
                top = f"\\textbf{{{top}}}"
            elif is_second:
                top = f"\\underline{{{top}}}"
            if sig and sig not in ("(top set)", "ns"):
                top = top + f"\\textsuperscript{{{sig}}}"
            top_cells.append(top)
            bot_cells.append(f"{{\\tiny[{rec['ci_lo_pct']:.1f},{rec['ci_hi_pct']:.1f}]}}")
        # Overall — top row gets the (bold/underline) point estimate;
        # bottom row gets the HIERARCHICAL bootstrap CI (Bestgen 2022 / HELM-
        # style) so the CI respects the task-equal-weighted aggregation.
        # No asterisks on Overall (mixed-metric significance is messy).
        top_cells.append(fmt_overall(raw))
        ov_lo, ov_hi = overall_ci_by_model.get(raw, (float("nan"), float("nan")))
        if ov_lo == ov_lo:  # not NaN
            bot_cells.append(f"{{\\tiny[{ov_lo:.1f},{ov_hi:.1f}]}}")
        else:
            bot_cells.append("")
        return (" & ".join(top_cells) + " \\\\\n"
                + " & ".join(bot_cells) + " \\\\[2pt]")

    def emit(tex_path: Path, *, ci_mode: str, font: str, tabcolsep: str,
             wrap_resizebox: bool, ci_caption_blurb: str,
             merge_reasoning: bool = False, sideways: bool = False):
        ncols_total = 10 if merge_reasoning else 11   # used in \multicolumn
        col_spec = ("@{}l|cccc|cc|cc|c@{}" if merge_reasoning
                    else "@{}ll|cccc|cc|cc|c@{}")
        # cmidrule columns shift left by 1 if reasoning is merged
        cmid_a, cmid_b, cmid_c = (("2-5", "6-7", "8-9")
                                   if merge_reasoning else ("3-6", "7-8", "9-10"))
        header_first = ("\\textbf{Model} & " if merge_reasoning
                        else "\\textbf{Model} & \\textbf{Reasoning} & ")
        # padding cell(s) at the start of the multi-column header row
        header_pad = ("& " if merge_reasoning else "& & ")

        if ci_mode == "stacked":
            renderer = lambda row, _ci: row_to_latex_stacked_ci(row)
        elif merge_reasoning:
            renderer = row_to_latex_merged
        else:
            renderer = row_to_latex

        with tex_path.open("w") as fh:
            fh.write("% Auto-generated by stats_per_task_ci.py — do not edit by hand.\n")
            fh.write(f"% Variant: ci_mode={ci_mode}, font={font}, "
                     f"resizebox={wrap_resizebox}, merge_reasoning={merge_reasoning}\n")
            # `sideways=True` rotates the table by 90° using \rotatebox from
            # graphicx (always loaded in NeurIPS) — no `rotating` package needed.
            fh.write("\\begin{table*}[htbp]\n\\centering")
            if sideways:
                fh.write("\n\\rotatebox{90}{%\n\\begin{minipage}{\\textheight}\n"
                         "\\centering")
            if font:
                fh.write(f"\\{font}")
            fh.write("\n")
            fh.write(f"\\setlength{{\\tabcolsep}}{{{tabcolsep}}}\n")
            # The colored-bullet variant supplies its own significance-legend in
            # `ci_caption_blurb` and skips the asterisk-style trailing legend.
            star_legend = (""
                if ci_mode == "tiny_colored" else
                "Superscripts denote Holm-corrected paired-test significance against "
                "the leader set (members of the leader set carry no superscript): "
                "models flagged with stars perform significantly worse than every "
                "model whose 95\\% bootstrap CI overlaps the empirical leader "
                "(McNemar exact for binary tasks, Wilcoxon signed-rank on per-question "
                "F1 for RATA/ORA). \\textsuperscript{***}$p<0.001$, "
                "\\textsuperscript{**}$p<0.01$, \\textsuperscript{*}$p<0.05$. ")
            fh.write("\\caption{Olfactory Perception (OP) benchmark results using "
                     f"{PROMPT_LABEL} prompts. Accuracy (\\%) per task and overall (unweighted mean "
                     f"across the eight task scores). {ci_caption_blurb}"
                     "Abbreviations: \\textbf{OC} = Odor Classification ($n$=175), "
                     "\\textbf{OPD} = Odor Primary Descriptor ($n$=175), \\textbf{OIn} = "
                     "Odor Intensity ($n$=175), \\textbf{OPl} = Odor Pleasantness "
                     "($n$=175), \\textbf{RATA} = Rate-All-That-Apply ($n$=100), "
                     "\\textbf{OS} = Odor Similarity ($n$=100), \\textbf{ORA} = Olfactory "
                     "Receptor Activation ($n$=80), \\textbf{SIT} = Smell Identification "
                     "Test ($n$=30). \\textbf{Bold} = best per column; "
                     "\\underline{underlined} = second best per column. "
                     f"{star_legend}}}\n")
            label = "tab:oi_results" if PROMPT == "name" else "tab:oi_results_smiles"
            fh.write(f"\\label{{{label}}}\n")
            if wrap_resizebox:
                fh.write("\\resizebox{\\textwidth}{!}{%\n")
            fh.write(f"\\begin{{tabular}}{{{col_spec}}}\n\\toprule\n")
            fh.write(f"{header_pad}\\multicolumn{{4}}{{c|}}{{\\textbf{{Simple}} ($N$=700)}} "
                     f"& \\multicolumn{{2}}{{c|}}{{\\textbf{{Intermediate}} ($N$=200)}} "
                     f"& \\multicolumn{{2}}{{c|}}{{\\textbf{{Hard}} ($N$=110)}} & \\\\\n")
            fh.write(f"\\cmidrule(lr){{{cmid_a}}} \\cmidrule(lr){{{cmid_b}}} "
                     f"\\cmidrule(lr){{{cmid_c}}}\n")
            fh.write(header_first + " & ".join(CATEGORY_ORDER) + " & \\textbf{Overall} \\\\\n")
            fh.write("\\midrule\n")
            fh.write(f"\\multicolumn{{{ncols_total}}}{{c}}{{\\cellcolor[HTML]{{FFF3CA}}"
                     "\\textbf{Closed-Source}}} \\\\\n")
            fh.write("\\midrule\n")
            prev_family = None
            for _, r in closed.iterrows():
                cur = r["Family"]
                if prev_family is not None and cur != prev_family:
                    fh.write("\\midrule\n")
                prev_family = cur
                fh.write(renderer(r, ci_mode) + "\n")
            fh.write("\\midrule\n")
            fh.write(f"\\multicolumn{{{ncols_total}}}{{c}}{{\\cellcolor[HTML]{{D9E1F4}}"
                     "\\textbf{Open-Source}}} \\\\\n")
            fh.write("\\midrule\n")
            prev_family = None
            for _, r in open_src.iterrows():
                cur = r["Family"]
                if prev_family is not None and cur != prev_family:
                    fh.write("\\midrule\n")
                prev_family = cur
                fh.write(renderer(r, ci_mode) + "\n")
            fh.write("\\bottomrule\n\\end{tabular}\n")
            if wrap_resizebox:
                fh.write("}\n")
            if sideways:
                fh.write("\\end{minipage}}\n")  # close minipage and rotatebox
            fh.write("\\end{table*}\n")
        print(f"[written] {tex_path}")

    # ---- Variant V1: paper-Table-3 style, NO CIs in cells (recommended) ----
    emit(OUT_DIR / f"per_task_table_v1_no_ci{SUFFIX}.tex",
         ci_mode="none",
         font="small",
         tabcolsep="3pt",
         wrap_resizebox=False,
         ci_caption_blurb="")

    # ---- Variant V2: integer-rounded CIs + Reasoning merged into Model ----
    emit(OUT_DIR / f"per_task_table_v2_short_ci{SUFFIX}.tex",
         ci_mode="short",
         font="scriptsize",
         tabcolsep="2pt",
         wrap_resizebox=False,
         merge_reasoning=True,
         ci_caption_blurb="Each cell shows accuracy followed by its 95\\% bootstrap CI in brackets (integer-rounded). The reasoning configuration is appended in parentheses next to each model name. ")

    # ---- Variant V2-colored: V2 + 1-decimal CI in \tiny font + colored bullets ----
    emit(OUT_DIR / f"per_task_table_v2_colored{SUFFIX}.tex",
         ci_mode="tiny_colored",
         font="scriptsize",
         tabcolsep="2pt",
         wrap_resizebox=False,
         merge_reasoning=True,
         ci_caption_blurb=(
             "Each cell shows accuracy followed by its 95\\% bootstrap CI in "
             "brackets (1-decimal precision, smaller font). The reasoning "
             "configuration is appended in parentheses next to each model name. "
             "Significance against the leader set is encoded by the cell's font "
             "color: \\textcolor{red!85!black}{red} $p<0.001$, "
             "\\textcolor{orange!90!black}{orange} $p<0.01$, "
             "\\textcolor{olive!80!black}{olive} $p<0.05$ "
             "(McNemar exact for binary tasks, Wilcoxon signed-rank on per-question "
             "F1 for RATA/ORA, Holm-corrected within each task). Members of the "
             "leader set are rendered in default text color. "))

    # ---- Variant V3: full CIs but \resizebox to text width ----
    emit(OUT_DIR / f"per_task_table_v3_resizebox{SUFFIX}.tex",
         ci_mode="full",
         font="",       # font is overridden by resizebox
         tabcolsep="2pt",
         wrap_resizebox=True,
         ci_caption_blurb="Each cell shows accuracy followed by its 95\\% bootstrap CI in brackets. ")

    # Keep the previous full-CI single-row file for backward compatibility
    emit(OUT_DIR / f"per_task_table_with_CI{SUFFIX}.tex",
         ci_mode="full",
         font="scriptsize",
         tabcolsep="2pt",
         wrap_resizebox=False,
         ci_caption_blurb="Each cell shows accuracy followed by its 95\\% bootstrap CI in brackets. ")

    # ---- Variant V5: makecell two-line cells (requires \usepackage{makecell}) ----
    emit(OUT_DIR / f"per_task_table_v5_makecell{SUFFIX}.tex",
         ci_mode="makecell",
         font="small",
         tabcolsep="3pt",
         wrap_resizebox=False,
         ci_caption_blurb="Each cell stacks the accuracy (top) and its 95\\% bootstrap CI (below, smaller font). Requires \\texttt{\\textbackslash usepackage\\{makecell\\}} in the preamble. ")

    # ---- Variant V6: most compact main-body version. No CIs in cells, but
    # value itself is colored by p-level. Merged Model+Reasoning column.
    emit(OUT_DIR / f"per_task_table_v6_compact_colored{SUFFIX}.tex",
         ci_mode="colored_no_ci",
         font="small",
         tabcolsep="3pt",
         wrap_resizebox=False,
         merge_reasoning=True,
         ci_caption_blurb=(
             "Cell font color encodes Holm-corrected significance against the leader "
             "set: \\textcolor{red!85!black}{red} $p<0.001$, "
             "\\textcolor{orange!90!black}{orange} $p<0.01$, "
             "\\textcolor{olive!80!black}{olive} $p<0.05$ "
             "(McNemar exact for binary tasks, Wilcoxon signed-rank on per-question "
             "F1 for RATA/ORA). Members of the leader set carry default text color. "
             "95\\% bootstrap CIs are reported in the supplementary table. "))

    # ---- Variant V8: row-stacked CIs (CI on a sub-row in tiny font). ----
    emit(OUT_DIR / f"per_task_table_v8_stacked_ci{SUFFIX}.tex",
         ci_mode="stacked",
         font="small",
         tabcolsep="3pt",
         wrap_resizebox=False,
         merge_reasoning=True,
         ci_caption_blurb=(
             "Each model occupies two rows: the top row carries the accuracy "
             "(with asterisk superscript denoting Holm-corrected significance "
             "against the leader set), and the row immediately below shows the "
             "95\\% bootstrap CI in brackets, in a smaller font. Per-task CIs "
             "are non-parametric bootstrap CIs over the per-question score "
             "vector. The Overall CI is a hierarchical bootstrap that resamples "
             "within each task independently and re-averages the eight per-task "
             "means, respecting the task-equal-weighted aggregation "
             "(Bestgen 2022). Significance: \\textsuperscript{***}$p<0.001$, "
             "\\textsuperscript{**}$p<0.01$, \\textsuperscript{*}$p<0.05$ "
             "(McNemar exact for binary tasks, Wilcoxon signed-rank on per-"
             "question F1 for RATA/ORA, Holm-corrected within each task). "))

    # ---- Variant V7: sideways table for the appendix, full CIs + colors.
    # Requires \usepackage{rotating} in the preamble.
    emit(OUT_DIR / f"per_task_table_v7_sideways_appendix{SUFFIX}.tex",
         ci_mode="tiny_colored",
         font="small",
         tabcolsep="3pt",
         wrap_resizebox=False,
         merge_reasoning=True,
         sideways=True,
         ci_caption_blurb=(
             "Each cell shows accuracy followed by its 95\\% bootstrap CI in "
             "brackets. Cell font color encodes Holm-corrected significance against "
             "the leader set: \\textcolor{red!85!black}{red} $p<0.001$, "
             "\\textcolor{orange!90!black}{orange} $p<0.01$, "
             "\\textcolor{olive!80!black}{olive} $p<0.05$ "
             "(McNemar exact for binary tasks, Wilcoxon signed-rank on per-question "
             "F1 for RATA/ORA). Members of the leader set carry default text color. "
             "Uses \\texttt{\\textbackslash rotatebox\\{90\\}} from \\texttt{graphicx} (no extra package needed for NeurIPS). "))

    # Console: print just the top 6 per task with CIs
    print(f"\nPer-task accuracy with 95% bootstrap CIs (top 6 by mean) — {PROMPT_LABEL} prompt:")
    for cat in CATEGORY_ORDER:
        sub = df[df["category"] == cat].sort_values("mean_pct", ascending=False).head(6)
        print(f"\n[{cat}] n={sub['n'].iloc[0]}")
        for _, r in sub.iterrows():
            line = (f"  {r['model']:<24s} {r['mean_pct']:5.1f} "
                    f"[{r['ci_lo_pct']:5.1f}, {r['ci_hi_pct']:5.1f}]   {r['sig_vs_top']}")
            print(line)


if __name__ == "__main__":
    main()
