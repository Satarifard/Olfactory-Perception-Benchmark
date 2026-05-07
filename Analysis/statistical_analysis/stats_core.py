"""
Shared scoring + paired-comparison utilities for the Olfactory Perception (OP)
benchmark.

The benchmark has 1,010 questions per model × 2 prompt formats (isomeric
SMILES, compound name). Every model is evaluated on the same 1,010 questions
in the same order, so per-question scores form *paired* vectors that allow
McNemar (binary) and Wilcoxon / paired-t (continuous F1) tests across
SMILES-vs-Name comparisons, model-vs-model comparisons, and reasoning-budget
comparisons.

This module re-uses the same answer extraction and any-overlap / multilabel-F1
scoring as Analysis.ipynb, exposed as functions so every downstream stats
script computes identical scores.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

try:
    from scipy import stats as scistats
    _HAS_SCIPY = True
except Exception:
    scistats = None
    _HAS_SCIPY = False


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
# Resolve repo root by walking up from this file's location until we find
# a "Results/benchmark_results" sibling. Works whether stats_core.py lives
# directly under Analysis/ or under a nested subfolder like
# Analysis/statistical_analysis/.
def _find_repo_root() -> Path:
    here = Path(__file__).resolve().parent
    for candidate in (here, *here.parents):
        if (candidate / "Results" / "benchmark_results").is_dir():
            return candidate
    # Fallback: assume parent.parent.parent (script under Analysis/<sub>/)
    return Path(__file__).resolve().parent.parent.parent


_REPO_ROOT = _find_repo_root()
DATA_DIR = _REPO_ROOT / "Results" / "benchmark_results"
PATTERN = "*_OP_Benchmark.csv"

# Categories scored with multilabel F1; everything else uses binary any-overlap
MULTIANSWER_CATS = {"or_activation", "rata"}

CATEGORY_DISPLAY = {
    "odor_classification": "OC",
    "primary_odor_descriptor": "OPD",
    "odor_intensity": "OIn",
    "odor_pleasantness": "OPl",
    "rata": "RATA",
    "mixture_similarity": "OS",
    "or_activation": "ORA",
    "smell_identification": "SIT",
}
CATEGORY_ORDER = ["OC", "OPD", "OIn", "OPl", "RATA", "OS", "ORA", "SIT"]

MIN_REQUIRED_COLS = {
    "question_category",
    "answer",
    "answer_to_prompt_1",
    "answer_to_prompt_2",
    "question_ID",
}

# Reasoning-budget families (within-family paired tests)
REASONING_FAMILIES = {
    "GPT-5": ["GPT_5_low", "GPT_5_high"],
    "Gemini 2.5 Pro": [
        "Gemini_2.5_pro_8192",
        "Gemini_2.5_pro_16000",
        "Gemini_2.5_pro_32768",
    ],
    "Grok 3 Mini": ["Grok_3_mini_low", "Grok_3_mini_high"],
    "DeepSeek Reasoner": ["Deepseek_8K", "Deepseek_16K", "Deepseek_32K"],
    "Claude Opus 4.6": ["Claude_opus_4.6_high", "Claude_opus_4.6_max"],
}


def exclude_models(scores: dict, exclude: list) -> dict:
    """Return a copy of `scores` with the named models removed.

    `exclude` is a list of model identifiers (matching the keys produced by
    `load_all_scores`, i.e. the CSV stems). Empty list / None is a no-op.
    """
    if not exclude:
        return scores
    skip = set(exclude)
    return {m: v for m, v in scores.items() if m not in skip}


# ---------------------------------------------------------------------------
# Token extraction (mirrors Analysis.ipynb cell 1)
# ---------------------------------------------------------------------------
_SPLIT_RE = re.compile(r"[;；\n\r\t]+|,(?!\d)|\s+-\s+|\s+and\s+", flags=re.I)
_NUMERIC_RE = re.compile(r"^[-+]?\d+(?:\.\d+)?$")
_BULLET_RE = re.compile(r"(?m)^\s*(?:[-*•·–—]+|\d+[\).:-])\s+")


def _normalize_token(s: str) -> str:
    if s is None:
        return ""
    s = str(s).strip().strip("[](){}<>\"'` ").lower()
    s = re.sub(r"\s+", " ", s)
    s = s.strip(" \t\n\r-–—•·*").strip(".,:;!?%")
    return s


def split_items(cell: object) -> List[str]:
    if cell is None or (isinstance(cell, float) and np.isnan(cell)):
        return []
    text = str(cell).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return []
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _BULLET_RE.sub("", text)
    parts = [p for p in _SPLIT_RE.split(text) if p is not None]
    cleaned = [_normalize_token(p) for p in parts if _normalize_token(p)]
    cleaned = [t for t in cleaned if t not in {"nan", "none", "null"}]
    cleaned = [t for t in cleaned if not _NUMERIC_RE.fullmatch(t)]
    cleaned = [t for t in cleaned if not t.startswith("desc_count")]
    if not cleaned:
        single = _normalize_token(text)
        if single and single not in {"nan", "none", "null"} and not _NUMERIC_RE.fullmatch(single):
            return [single]
    return cleaned


def any_overlap(pred_items: List[str], truth_items: List[str]) -> bool:
    if not pred_items or not truth_items:
        return False
    return bool(set(pred_items) & set(truth_items))


def f1_multilabel(pred_items: List[str], truth_items: List[str]) -> float:
    P, T = set(pred_items), set(truth_items)
    if not P and not T:
        return 1.0
    tp = len(P & T)
    if tp == 0:
        return 0.0
    fp = len(P - T)
    fn = len(T - P)
    return (2.0 * tp) / (2.0 * tp + fp + fn)


def per_question_score(
    df: pd.DataFrame, pred_col: str, truth_col: str = "answer"
) -> np.ndarray:
    """
    Return a length-N float array of per-question scores.

    For binary categories (any-overlap) the score is 0.0 / 1.0.
    For multilabel categories (RATA, ORA) it is per-question F1 in [0, 1].
    """
    preds = df[pred_col].apply(split_items)
    truths = df[truth_col].apply(split_items)
    cats = df["question_category"].astype(str).str.lower()

    out = np.empty(len(df), dtype=float)
    for i, (p, t, cat) in enumerate(zip(preds, truths, cats)):
        if cat in MULTIANSWER_CATS:
            out[i] = f1_multilabel(p, t)
        else:
            out[i] = 1.0 if any_overlap(p, t) else 0.0
    return out


def task_unweighted_mean(scores: np.ndarray, cats: np.ndarray) -> float:
    """
    Mean of per-task means: matches the paper's Table 3 "Overall" definition
    (Section 4.2: "unweighted arithmetic mean of the eight per-task scores").

    Uses the per-question score vector and the matching category labels;
    smaller tasks (SIT n=30) get equal weight to larger ones (OC n=175).
    """
    s = pd.Series(scores)
    return float(s.groupby(cats).mean().mean())


# ---------------------------------------------------------------------------
# Continuous numerical ratings (OIn / OPl / OS) extraction
# ---------------------------------------------------------------------------
_NUM_TOK_RE = re.compile(r"-?\d+(?:\.\d+)?")


def _extract_two_ratings(cell: object) -> Tuple[float, float]:
    """
    OIn/OPl prompts ask the model to return e.g. "hexan-2-one;72;21".
    Pull out the LAST two numeric tokens as the (rating_1, rating_2) for the
    two stimuli in the question. Returns (nan, nan) if fewer than 2 numbers.
    """
    if cell is None or (isinstance(cell, float) and np.isnan(cell)):
        return (float("nan"), float("nan"))
    nums = _NUM_TOK_RE.findall(str(cell))
    if len(nums) < 2:
        return (float("nan"), float("nan"))
    try:
        return float(nums[-2]), float(nums[-1])
    except ValueError:
        return (float("nan"), float("nan"))


def _extract_one_rating(cell: object) -> float:
    """Mixture-similarity returns e.g. 'Strongly Similar;0.42'. Pull last number."""
    if cell is None or (isinstance(cell, float) and np.isnan(cell)):
        return float("nan")
    nums = _NUM_TOK_RE.findall(str(cell))
    if not nums:
        return float("nan")
    try:
        return float(nums[-1])
    except ValueError:
        return float("nan")


_INTENSITY_RE = re.compile(r"INTENSITY\s*=\s*(-?\d+(?:\.\d+)?)", flags=re.I)
_PLEASANT_RE = re.compile(r"PLEASANTNESS\s*=\s*(-?\d+(?:\.\d+)?)", flags=re.I)
_EXPVAL_RE = re.compile(r"Experimental\s+Values?\s*=\s*(-?\d+(?:\.\d+)?)", flags=re.I)


def _truth_numerics_oin_opl(other_info: object) -> Tuple[float, float]:
    """Pull (rating_1, rating_2) from `other_info` for an OIn or OPl row."""
    if other_info is None or (isinstance(other_info, float) and np.isnan(other_info)):
        return (float("nan"), float("nan"))
    s = str(other_info)
    m_int = _INTENSITY_RE.findall(s)
    m_pl = _PLEASANT_RE.findall(s)
    nums = m_int if m_int else m_pl
    if len(nums) < 2:
        return (float("nan"), float("nan"))
    return (float(nums[0]), float(nums[1]))


def _truth_numeric_os(other_info: object) -> float:
    """Pull experimental perceptual distance from `other_info` for an OS row."""
    if other_info is None or (isinstance(other_info, float) and np.isnan(other_info)):
        return float("nan")
    m = _EXPVAL_RE.search(str(other_info))
    if not m:
        return float("nan")
    return float(m.group(1))


def extract_numeric_ratings(
    df: pd.DataFrame, pred_col: str
) -> Tuple[np.ndarray, np.ndarray]:
    """
    For continuous-rating tasks (OIn, OPl, OS), return aligned (pred, truth)
    numeric arrays for per-question Pearson correlation analysis.

    Predictions are parsed from the model's response (last numeric tokens);
    ground truth comes from the OP benchmark's `other_info` column:
        OIn rows: 'SMILES_1 INTENSITY=...;SMILES_2 INTENSITY=...'
        OPl rows: 'SMILES_1 PLEASANTNESS=...;SMILES_2 PLEASANTNESS=...'
        OS  rows: 'Experimental Values=...'
    """
    cats = df["question_category"].astype(str).str.lower().values
    n = len(df)
    pred_out = np.full(n * 2, np.nan)
    truth_out = np.full(n * 2, np.nan)
    for i in range(n):
        cat = cats[i]
        oi = df.iloc[i]["other_info"] if "other_info" in df.columns else None
        if cat in {"odor_intensity", "odor_pleasantness"}:
            p1, p2 = _extract_two_ratings(df.iloc[i][pred_col])
            t1, t2 = _truth_numerics_oin_opl(oi)
            pred_out[2 * i] = p1
            pred_out[2 * i + 1] = p2
            truth_out[2 * i] = t1
            truth_out[2 * i + 1] = t2
        elif cat == "mixture_similarity":
            pred_out[2 * i] = _extract_one_rating(df.iloc[i][pred_col])
            truth_out[2 * i] = _truth_numeric_os(oi)
    mask = ~np.isnan(pred_out) & ~np.isnan(truth_out)
    return pred_out[mask], truth_out[mask]


def extract_numeric_ratings_per_task(
    df: pd.DataFrame, pred_col: str
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Same as extract_numeric_ratings but separated by task name."""
    out: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for cat in ["odor_intensity", "odor_pleasantness", "mixture_similarity"]:
        mask = df["question_category"].astype(str).str.lower().values == cat
        sub = df[mask]
        p, t = extract_numeric_ratings(sub.reset_index(drop=True), pred_col)
        out[cat] = (p, t)
    return out


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def model_name_from_path(p: Path) -> str:
    name = p.name
    if name.endswith("_OP_Benchmark.csv"):
        return name[: -len("_OP_Benchmark.csv")]
    if name.endswith(".csv"):
        return name[:-4]
    return name


def load_all_scores(
    data_dir: Path = DATA_DIR, pattern: str = PATTERN
) -> Tuple[Dict[str, Dict[str, np.ndarray]], pd.DataFrame]:
    """
    Returns
    -------
    scores : dict
        scores[model][prompt] -> per-question score vector aligned to `meta`.
        prompt is "smiles" (prompt_1) or "name" (prompt_2).
    meta : DataFrame
        question_ID, question_category, with rows ordered as in the per-question vectors.
    """
    files = sorted(data_dir.glob(pattern))
    scores: Dict[str, Dict[str, np.ndarray]] = {}
    meta: pd.DataFrame | None = None

    for fp in files:
        df = pd.read_csv(fp, dtype=str, keep_default_na=True)
        missing = MIN_REQUIRED_COLS - set(df.columns)
        if missing:
            print(f"[skip] {fp.name}: missing {sorted(missing)}")
            continue

        if meta is None:
            meta = df[["question_ID", "question_category"]].reset_index(drop=True)
        else:
            # Verify alignment; assume identical question order across files
            if df["question_ID"].tolist() != meta["question_ID"].tolist():
                # Reorder this df to match the canonical meta order
                df = df.set_index("question_ID").loc[meta["question_ID"]].reset_index()

        s_smiles = per_question_score(df, "answer_to_prompt_1")
        s_name = per_question_score(df, "answer_to_prompt_2")

        mname = model_name_from_path(fp)
        scores[mname] = {"smiles": s_smiles, "name": s_name}

    if meta is None:
        raise RuntimeError(f"No CSV files matched in {data_dir}")
    return scores, meta


# ---------------------------------------------------------------------------
# Statistical tests on paired per-question vectors
# ---------------------------------------------------------------------------
def _is_binary(x: np.ndarray) -> bool:
    return np.all(np.isin(x, [0.0, 1.0]))


def mcnemar_exact(a: np.ndarray, b: np.ndarray) -> Tuple[float, float, int, int]:
    """
    Exact McNemar test for paired binary outcomes.

    a, b : 0/1 arrays of equal length (correct/incorrect on the same questions).
    Returns
    -------
    (p_two_sided, b01_count, b10_count, n_discordant)
        b01 = #(a=0, b=1)   (b better than a)
        b10 = #(a=1, b=0)   (a better than b)
    """
    a = np.asarray(a).astype(int)
    b = np.asarray(b).astype(int)
    b01 = int(np.sum((a == 0) & (b == 1)))
    b10 = int(np.sum((a == 1) & (b == 0)))
    n = b01 + b10
    if n == 0:
        return 1.0, b01, b10, 0
    # Exact two-sided binomial test on min(b01,b10) under H0: p=0.5
    if _HAS_SCIPY:
        # binomtest >= 1.7 returns object with .pvalue
        try:
            res = scistats.binomtest(min(b01, b10), n=n, p=0.5, alternative="two-sided")
            p = float(res.pvalue)
        except AttributeError:
            p = float(scistats.binom_test(min(b01, b10), n=n, p=0.5, alternative="two-sided"))
    else:
        # manual: 2 * P(X <= min(b01,b10))  capped at 1
        from math import comb
        k = min(b01, b10)
        tail = sum(comb(n, i) for i in range(0, k + 1)) / (2 ** n)
        p = min(1.0, 2.0 * tail)
    return p, b01, b10, n


def wilcoxon_paired(a: np.ndarray, b: np.ndarray) -> Tuple[float, float]:
    """
    Two-sided Wilcoxon signed-rank test on paired continuous scores (e.g. F1).
    Returns (statistic, p). If all differences are zero, returns (nan, 1.0).
    """
    diff = np.asarray(a) - np.asarray(b)
    if np.all(diff == 0):
        return np.nan, 1.0
    if not _HAS_SCIPY:
        # Fallback: paired t (sign of diff)
        return paired_t(a, b)
    res = scistats.wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
    return float(res.statistic), float(res.pvalue)


def paired_t(a: np.ndarray, b: np.ndarray) -> Tuple[float, float]:
    """Paired t-test (two-sided)."""
    if not _HAS_SCIPY:
        # Manual
        d = np.asarray(a) - np.asarray(b)
        n = d.size
        if n < 2 or np.std(d, ddof=1) == 0:
            return np.nan, 1.0
        t = np.mean(d) / (np.std(d, ddof=1) / np.sqrt(n))
        # Approx p via normal
        from math import erfc
        p = erfc(abs(t) / np.sqrt(2))
        return float(t), float(p)
    res = scistats.ttest_rel(a, b)
    return float(res.statistic), float(res.pvalue)


def cohens_d_paired(a: np.ndarray, b: np.ndarray) -> float:
    """Paired Cohen's d (mean diff / sd of diff)."""
    d = np.asarray(a) - np.asarray(b)
    sd = np.std(d, ddof=1) if d.size > 1 else 0.0
    if sd == 0:
        return float("inf") if np.mean(d) != 0 else 0.0
    return float(np.mean(d) / sd)


def mcnemar_odds_ratio_ci(
    b01: int, b10: int, alpha: float = 0.05
) -> Tuple[float, float, float]:
    """
    Conditional odds ratio for McNemar's exact test on a paired 2x2 table.

    OR = b01 / b10, where b01 = #(a=0, b=1) and b10 = #(a=1, b=0). 95% CI is
    obtained by inverting the exact (Clopper-Pearson) binomial CI for the
    proportion p = b01 / (b01 + b10) under H0: p = 0.5, then mapping back via
    OR = p / (1 - p). This is the conventional effect size for paired binary
    outcomes; thresholds (small/medium/large) for Cohen's d don't transfer.
    """
    n_disc = b01 + b10
    if n_disc == 0:
        return (float("nan"), float("nan"), float("nan"))
    if b10 == 0:
        return (float("inf"), float("nan"), float("nan"))
    or_point = b01 / b10
    if not _HAS_SCIPY:
        return (or_point, float("nan"), float("nan"))
    # Clopper-Pearson on b01 ~ Binomial(n_disc, p)
    if b01 == 0:
        p_lo = 0.0
    else:
        p_lo = scistats.beta.ppf(alpha / 2, b01, n_disc - b01 + 1)
    if b01 == n_disc:
        p_hi = 1.0
    else:
        p_hi = scistats.beta.ppf(1 - alpha / 2, b01 + 1, n_disc - b01)
    # Map p back to OR = p / (1 - p)
    or_lo = p_lo / (1 - p_lo) if p_lo < 1 else float("inf")
    or_hi = p_hi / (1 - p_hi) if p_hi < 1 else float("inf")
    return (float(or_point), float(or_lo), float(or_hi))


def risk_difference_ci(
    a: np.ndarray, b: np.ndarray, n_boot: int = 5000, alpha: float = 0.05,
    rng_seed: int = 0,
) -> Tuple[float, float, float]:
    """Wrapper around bootstrap_ci_diff; returned in [-1, 1] not pp."""
    md, lo, hi = bootstrap_ci_diff(a, b, n_boot=n_boot, alpha=alpha, rng_seed=rng_seed)
    return float(md), float(lo), float(hi)


# ---------------------------------------------------------------------------
# Steiger's Z for two dependent correlations sharing one variable
# ---------------------------------------------------------------------------
def steiger_z_test(
    r12: float, r13: float, r23: float, n: int
) -> Tuple[float, float]:
    """
    Steiger's Z for H0: r(X, Y1) == r(X, Y2) when Y1 and Y2 are measured on the
    same n cases (here X = human ratings, Y1 = name-prompt predictions,
    Y2 = SMILES-prompt predictions; r23 = corr(Y1, Y2)).

    Returns (Z, two-sided p). Implementation follows Steiger 1980 (eq. 11).
    """
    r12 = float(np.clip(r12, -0.999999, 0.999999))
    r13 = float(np.clip(r13, -0.999999, 0.999999))
    r23 = float(np.clip(r23, -0.999999, 0.999999))
    if n < 4:
        return (float("nan"), float("nan"))
    rm2 = (r12 ** 2 + r13 ** 2) / 2.0
    f = (1.0 - r23) / (2.0 * (1.0 - rm2)) if rm2 < 1 else float("nan")
    h = (1.0 - f * rm2) / (1.0 - rm2) if rm2 < 1 else float("nan")
    denom = (2.0 * (1.0 - r23) * h) / (n - 3)
    if denom <= 0 or np.isnan(denom):
        return (float("nan"), float("nan"))
    z = (np.arctanh(r12) - np.arctanh(r13)) / np.sqrt(denom)
    if not _HAS_SCIPY:
        # normal approximation fallback
        from math import erf, sqrt
        p = 2 * (1 - 0.5 * (1 + erf(abs(z) / sqrt(2))))
    else:
        p = float(2 * (1 - scistats.norm.cdf(abs(z))))
    return float(z), float(p)


def pearson_r_with_ci(
    x: np.ndarray, y: np.ndarray, alpha: float = 0.05
) -> Tuple[float, float, float, int]:
    """Pearson r with Fisher-z 95% CI. Returns (r, lo, hi, n_used)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = ~np.isnan(x) & ~np.isnan(y)
    n = int(mask.sum())
    if n < 4:
        return (float("nan"), float("nan"), float("nan"), n)
    if _HAS_SCIPY:
        r, _ = scistats.pearsonr(x[mask], y[mask])
    else:
        xm = x[mask] - x[mask].mean()
        ym = y[mask] - y[mask].mean()
        r = float((xm * ym).sum() / np.sqrt((xm ** 2).sum() * (ym ** 2).sum()))
    # Fisher z CI
    z = np.arctanh(np.clip(r, -0.999999, 0.999999))
    se = 1.0 / np.sqrt(max(n - 3, 1))
    if _HAS_SCIPY:
        zcrit = float(scistats.norm.ppf(1 - alpha / 2))
    else:
        zcrit = 1.959963984540054
    lo = float(np.tanh(z - zcrit * se))
    hi = float(np.tanh(z + zcrit * se))
    return (float(r), lo, hi, n)


def bootstrap_ci_diff(
    a: np.ndarray,
    b: np.ndarray,
    n_boot: int = 5000,
    alpha: float = 0.05,
    rng_seed: int = 0,
) -> Tuple[float, float, float]:
    """
    Bootstrap CI for the paired mean difference (a − b).

    Returns (mean_diff, lo, hi) on the same scale as the inputs.
    For binary inputs the diff is in [-1, 1] (multiply by 100 for percentage points).
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    diff = a - b
    rng = np.random.default_rng(rng_seed)
    n = diff.size
    boots = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots[i] = diff[idx].mean()
    lo = float(np.percentile(boots, 100 * alpha / 2))
    hi = float(np.percentile(boots, 100 * (1 - alpha / 2)))
    return float(diff.mean()), lo, hi


def permutation_test_paired(
    a: np.ndarray,
    b: np.ndarray,
    n_perm: int = 10000,
    rng_seed: int = 0,
) -> Tuple[float, float]:
    """
    Two-sided sign-flip permutation test on paired differences.
    Useful for small N (e.g. SIT n=30) where asymptotic McNemar is unreliable.

    Returns (mean_diff, p_two_sided).
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    d = a - b
    obs = float(np.mean(d))
    rng = np.random.default_rng(rng_seed)
    n = d.size
    perms = np.empty(n_perm)
    for i in range(n_perm):
        signs = rng.choice([-1.0, 1.0], size=n)
        perms[i] = (d * signs).mean()
    # Two-sided: fraction of |perm| >= |obs|
    p = float((np.abs(perms) >= abs(obs) - 1e-15).mean())
    # Avoid p=0; use (count+1)/(n_perm+1) lower bound
    p = max(p, 1.0 / (n_perm + 1))
    return obs, p


# ---------------------------------------------------------------------------
# Multiple comparison correction
# ---------------------------------------------------------------------------
def holm_bonferroni(pvals: List[float]) -> List[float]:
    """
    Holm–Bonferroni step-down correction.
    Returns adjusted p-values aligned with input order.
    """
    p = np.asarray(pvals, dtype=float)
    n = p.size
    order = np.argsort(p)
    adj = np.empty(n)
    running = 0.0
    for rank, idx in enumerate(order):
        cand = (n - rank) * p[idx]
        running = max(running, cand)
        adj[idx] = min(running, 1.0)
    return adj.tolist()


def benjamini_hochberg(pvals: List[float]) -> List[float]:
    """Benjamini–Hochberg FDR correction."""
    p = np.asarray(pvals, dtype=float)
    n = p.size
    order = np.argsort(p)
    ranked = p[order]
    adj_ranked = np.empty(n)
    prev = 1.0
    for i in range(n - 1, -1, -1):
        val = ranked[i] * n / (i + 1)
        prev = min(prev, val)
        adj_ranked[i] = min(prev, 1.0)
    adj = np.empty(n)
    adj[order] = adj_ranked
    return adj.tolist()


# ---------------------------------------------------------------------------
# Convenience: choose the right test for a (model, category) score vector
# ---------------------------------------------------------------------------
def auto_paired_test(
    a: np.ndarray,
    b: np.ndarray,
    small_n_permutation: bool = True,
    small_n_threshold: int = 30,
) -> Dict[str, float]:
    """
    Pick McNemar (binary) or Wilcoxon (continuous) automatically. For small N
    (e.g. SIT n=30) also report a sign-flip permutation p-value.

    Returns dict with: test, p_value, mean_diff, ci_lo, ci_hi, cohens_d, n,
                       extras (e.g. mcnemar b01/b10 or wilcoxon W stat).
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n = a.size

    out: Dict[str, float] = {"n": n}
    mean_diff, lo, hi = bootstrap_ci_diff(a, b)
    out.update({"mean_diff": mean_diff, "ci_lo": lo, "ci_hi": hi})
    out["cohens_d"] = cohens_d_paired(a, b)

    if _is_binary(a) and _is_binary(b):
        p, b01, b10, n_disc = mcnemar_exact(a, b)
        out.update({
            "test": "mcnemar_exact",
            "p_value": p,
            "mcnemar_b01": b01,
            "mcnemar_b10": b10,
            "n_discordant": n_disc,
        })
    else:
        W, p = wilcoxon_paired(a, b)
        out.update({"test": "wilcoxon_signed_rank", "p_value": p, "wilcoxon_W": W})

    if small_n_permutation and n <= small_n_threshold:
        _, p_perm = permutation_test_paired(a, b)
        out["p_perm"] = p_perm

    return out


def stars(p: float) -> str:
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""
