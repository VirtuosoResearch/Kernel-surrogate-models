# -*- coding: utf-8 -*-
"""
Pipeline: Long-tail analysis over a batch of arrays (NumPy/PyTorch supported).
Focus: "Are there especially large values?" (measured on absolute values).
Outputs:
  - metrics CSV (per array)
  - rank-frequency comparison (log-log)
  - overall CCDF (log-log)
  - overall histogram (log y)
  - boxplot of key metrics across arrays
All plots are saved to disk; nothing is shown interactively.
"""

import os
from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================
# Utilities
# =========================
def _ensure_dir(path: str) -> str:
    """Create directory if it does not exist and return the path."""
    os.makedirs(path, exist_ok=True)
    return path

def _to_abs_1d(x) -> np.ndarray:
    """
    Convert input to a 1D absolute-value NumPy array (float64) and drop NaN/Inf.
    Supports: list/np.ndarray/torch.Tensor.
    """
    if hasattr(x, "detach"):  # torch.Tensor
        x = x.detach().cpu().numpy()
    x = np.asarray(x, dtype=np.float64).ravel()
    x = np.abs(x)
    x = x[np.isfinite(x)]
    return x

def _safe_std(x: np.ndarray) -> float:
    """Sample std with ddof=1; returns 0.0 if not enough data."""
    return float(np.std(x, ddof=1)) if x.size >= 2 else 0.0


# =========================
# Metrics on |x|
# =========================
def tail_share(x: np.ndarray, q: float = 0.95) -> float:
    """
    Top (1-q) share of mass. Example: q=0.95 => Top 5% sum / total sum.
    Returns 0 if empty or total sum == 0.
    """
    if x.size == 0:
        return 0.0
    thr = np.quantile(x, q)
    top = x[x >= thr]
    s = x.sum()
    return float(top.sum() / s) if s > 0 else 0.0

def quantile_ratio(x: np.ndarray, p_high: float = 0.99, p_mid: float = 0.50) -> float:
    """Quantile ratio, e.g., P99/P50. If median==0 and P_high>0, return inf."""
    if x.size == 0:
        return np.nan
    qh = np.quantile(x, p_high)
    qm = np.quantile(x, p_mid)
    return float(qh / qm) if qm > 0 else (np.inf if qh > 0 else 1.0)

def kurtosis_excess(x: np.ndarray) -> float:
    """
    Excess kurtosis = kurtosis - 3. Positive => heavier tails / more outliers.
    Uses Fisher definition with sample std (ddof=1). Handles small/degenerate cases.
    """
    n = x.size
    if n < 3:
        return 0.0
    mu = float(x.mean())
    sd = float(x.std(ddof=1))
    if sd == 0:
        return -3.0
    z4 = np.mean(((x - mu) / sd) ** 4)
    return float(z4 - 3.0)

def gini_coefficient(x: np.ndarray) -> float:
    """
    Gini coefficient on nonnegative data (we use |x| already).
    0 => perfectly equal, 1 => extremely unequal.
    """
    if x.size == 0 or np.all(x == 0):
        return 0.0
    x = np.sort(x)
    n = x.size
    cumx = np.cumsum(x)
    # Gini = 1 + 1/n - 2 * sum(cumx) / (n * sum(x))
    return float(1 + 1/n - 2 * (cumx.sum() / (n * x.sum())))

def long_tail_metrics_abs(arr) -> Dict[str, Any]:
    """
    Compute core long-tail metrics on |arr|.
    Larger values of (CV, Top 5% share, P99/P50, Gini, Excess Kurtosis) => heavier tail.
    """
    x = _to_abs_1d(arr)
    if x.size == 0:
        # Return consistent keys with neutral values for empty input
        return {
            "count": 0, "mean": 0.0, "std": 0.0, "CV": 0.0,
            "Top 5% share": 0.0, "P99/P50": 1.0, "Excess Kurtosis": -3.0,
            "Gini": 0.0, "P90": 0.0, "P95": 0.0, "P99": 0.0, "max": 0.0, "median": 0.0,
        }
    mu = float(x.mean())
    sd = _safe_std(x)
    cv = (sd / mu) if mu > 0 else (np.inf if sd > 0 else 0.0)
    return {
        "count": int(x.size),
        "mean": mu,
        "std": sd,
        "CV": float(cv),
        "Top 5% share": tail_share(x, 0.95),
        "P99/P50": quantile_ratio(x, 0.99, 0.50),
        "Excess Kurtosis": kurtosis_excess(x),
        "Gini": gini_coefficient(x),
        "P90": float(np.quantile(x, 0.90)),
        "P95": float(np.quantile(x, 0.95)),
        "P99": float(np.quantile(x, 0.99)),
        "max": float(x.max()),
        "median": float(np.quantile(x, 0.50)),
    }


# =========================
# Plotting (saved to files)
# =========================
def plot_rank_frequency_all(phi_list: List, out_path: str) -> None:
    """
    Rank-frequency plot (log-log) per array on |x|.
    Compares how long/heavy the tails are across arrays.
    """
    plt.figure()
    for i, arr in enumerate(phi_list):
        x = np.sort(_to_abs_1d(arr))[::-1]
        if x.size == 0:
            continue
        ranks = np.arange(1, x.size + 1)
        plt.loglog(ranks, x, label=f"Array {i}")
    plt.xlabel("Rank (1 = largest)")
    plt.ylabel("|Value|")
    plt.title("Rank vs |Value| (log-log)")
    plt.legend(ncol=2, fontsize=8)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

def plot_ccdf_overall(phi_list: List, out_path: str) -> None:
    """
    Overall CCDF on |x| (log-log) after concatenating all arrays.
    Shows how the tail decays globally.
    """
    all_vals = np.concatenate([_to_abs_1d(a) for a in phi_list]) if len(phi_list) else np.array([])
    if all_vals.size == 0:
        plt.figure(); plt.title("CCDF (no data)")
        plt.savefig(out_path, dpi=300, bbox_inches="tight"); plt.close()
        return
    x = np.sort(all_vals)
    n = x.size
    xs = np.unique(x)
    ccdf = (n - np.searchsorted(x, xs, side='left')) / n
    plt.figure()
    plt.loglog(xs[ccdf > 0], ccdf[ccdf > 0])
    plt.xlabel("t")
    plt.ylabel("P(|X| ≥ t)")
    plt.title("Overall CCDF on |X| (log-log)")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

def plot_hist_overall(phi_list: List, out_path: str, bins: int = 80) -> None:
    """
    Overall histogram of |x| with log-scaled y-axis.
    Helpful to see sparsity in the tail.
    """
    all_vals = np.concatenate([_to_abs_1d(a) for a in phi_list]) if len(phi_list) else np.array([])
    x = _to_abs_1d(all_vals)
    plt.figure()
    if x.size:
        plt.hist(x, bins=bins)
        plt.yscale('log')
    plt.xlabel("|Value|")
    plt.ylabel("Count (log)")
    plt.title("Overall Histogram of |X| (log y)")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

def plot_metrics_box(df: pd.DataFrame, out_path: str,
                     cols=("Gini", "CV", "Top 5% share", "P99/P50")) -> None:
    """
    Boxplot of key long-tail metrics across arrays.
    """
    if df.empty:
        plt.figure(); plt.title("Metrics Boxplot (no data)")
        plt.savefig(out_path, dpi=300, bbox_inches="tight"); plt.close()
        return
    cols = [c for c in cols if c in df.columns]
    plt.figure()
    df[cols].plot(kind="box", vert=True)
    plt.title("Long-tail Metrics Across Arrays")
    plt.ylabel("Value")
    plt.grid(True, axis='y', linestyle='--', alpha=0.4)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

def plot_per_array_rank(phi_list: List, out_dir: str, prefix: str = "rank_") -> None:
    """
    (Optional) Save one rank-frequency plot per array.
    Useful when you want to inspect the most extreme arrays individually.
    """
    for i, arr in enumerate(phi_list):
        x = np.sort(_to_abs_1d(arr))[::-1]
        plt.figure()
        if x.size:
            ranks = np.arange(1, x.size + 1)
            plt.loglog(ranks, x)
        plt.xlabel("Rank (1 = largest)")
        plt.ylabel("|Value|")
        plt.title(f"Array {i}: Rank vs |Value| (log-log)")
        fpath = os.path.join(out_dir, f"{prefix}{i}.png")
        plt.savefig(fpath, dpi=300, bbox_inches="tight")
        plt.close()


# =========================
# Main pipeline
# =========================
def analyze_phi_long_tail(
    phi_list: List,
    save_dir: str = "phi_longtail_report",
    export_csv: bool = True,
    make_per_array_plots: bool = False,
    topk_examples: int = 5,
) -> Tuple[pd.DataFrame, Dict[str, Any], List[int]]:
    """
    Run long-tail analysis on a batch of arrays.

    Args:
        phi_list: List of arrays/tensors.
        save_dir: Output directory for images (and CSV if enabled).
        export_csv: Whether to write per-array metrics to CSV.
        make_per_array_plots: If True, save an individual rank plot per array.
        topk_examples: Select top-k arrays (by P99/P50, then Gini) for inspection.

    Returns:
        df_per_array: DataFrame of per-array metrics (index = array id).
        overall_metrics: Dict of overall metrics on concatenated |phi|.
        topk_idx: List of indices of the top-k “most long-tailed” arrays.
    """
    _ensure_dir(save_dir)

    # 1) Per-array metrics
    rows = []
    for i, arr in enumerate(phi_list):
        m = long_tail_metrics_abs(arr)
        m["Array"] = i
        rows.append(m)
    df = pd.DataFrame(rows).set_index("Array").sort_index()

    # 2) Overall metrics on concatenated |phi|
    all_vals = np.concatenate([_to_abs_1d(a) for a in phi_list]) if len(phi_list) else np.array([])
    overall = long_tail_metrics_abs(all_vals)

    # 3) Key plots
    plot_rank_frequency_all(phi_list, os.path.join(save_dir, "rank_all.png"))
    plot_ccdf_overall(phi_list, os.path.join(save_dir, "overall_ccdf.png"))
    plot_hist_overall(phi_list, os.path.join(save_dir, "hist_overall.png"))
    plot_metrics_box(df, os.path.join(save_dir, "metrics_boxplot.png"))

    if make_per_array_plots:
        _ensure_dir(os.path.join(save_dir, "per_array"))
        plot_per_array_rank(phi_list, os.path.join(save_dir, "per_array"))

    # 4) Export CSV (optional)
    if export_csv:
        df.to_csv(os.path.join(save_dir, "metrics.csv"), index=True)

    # 5) Pick top-k arrays by a robust notion of tail-heaviness
    #    Primary: P99/P50; Secondary: Gini (both higher => heavier tail).
    if not df.empty:
        topk = (df.sort_values(by=["P99/P50", "Gini"], ascending=[False, False])
                  .head(topk_examples).index.tolist())
    else:
        topk = []

    return df, overall, topk


# =========================
# Example usage (uncomment to run)
# =========================
# if __name__ == "__main__":
#     import torch
#     # Build a toy batch with some extreme values
#     phi = [
#         np.random.randn(5000),
#         torch.randn(4000),
#         np.concatenate([np.random.randn(4500), np.array([0, 0, 0, 0])]),
#     ]
#     # Inject some large-magnitude points to simulate a long tail
#     phi[0][::1000] = np.array([50., -120., 200., -300., 500.])[:phi[0][::1000].shape[0]]
#     phi[1][::800] += torch.tensor([80., -250., 400., -700., 900.])[:phi[1][::800].shape[0]]
#
#     df_metrics, overall_metrics, topk_idx = analyze_phi_long_tail(
#         phi,
#         save_dir="phi_longtail_report",
#         export_csv=True,
#         make_per_array_plots=True,
#         topk_examples=3,
#     )
#     print("Per-array metrics:\n", df_metrics.head())
#     print("\nOverall metrics on |phi|:\n", overall_metrics)
#     print("\nMost tail-heavy arrays (indices):", topk_idx)
