"""
seed_level_analysis.py
======================
Reviewer 2/5 compliance: seed-level statistical analysis for PACA-Affinity revision.

Inputs: two JSON files in the same format as multi_seed_results.json produced by
the training script (one for PACA-Affinity, one for the baseline, e.g. PWAA+RPE):

    {
      "seed_results": [
        {"seed": 0,  "paddle": {"MSE":..,"RMSE":..,"MAE":..,"R2":..,"PCC":..},
                      "abbind": {...}, "sabdab": {...}, "skempi": {...}},
        ... 10 entries ...
      ],
      "summary": {...}
    }

Outputs (printed + saved to seed_level_stats.json):
  - per-dataset mean +/- std of each metric across the 10 runs
  - bootstrap 95% CI (percentile, 10,000 resamples) for the MAE difference
  - Wilcoxon signed-rank test on paired runs, Holm-corrected across 4 datasets
  - pooled (sample-size-weighted) results per run
  - rounding-consistency check: every delta% recomputed from DISPLAYED values

Only stdlib + numpy + scipy are required.
"""

import json
import sys
import numpy as np
from scipy.stats import wilcoxon

DATASETS = ["paddle", "abbind", "sabdab", "skempi"]
TEST_SIZES = {"paddle": 342, "abbind": 174, "sabdab": 55, "skempi": 54}  # from manuscript Table 6
METRICS = ["MSE", "RMSE", "MAE", "R2", "PCC"]
N_BOOT = 10000
RNG = np.random.default_rng(2026)


def load_runs(path):
    with open(path) as f:
        data = json.load(f)
    runs = {r["seed"]: r for r in data["seed_results"]}
    return runs


def paired_arrays(runs_a, runs_b, dataset, metric):
    """Per-run metric values for model A and model B on common seeds, paired by seed."""
    common = sorted(set(runs_a) & set(runs_b))
    a = np.array([runs_a[s][dataset][metric] for s in common], dtype=float)
    b = np.array([runs_b[s][dataset][metric] for s in common], dtype=float)
    return common, a, b


def bootstrap_ci_diff(a, b, n_boot=N_BOOT, alpha=0.05):
    """Percentile bootstrap CI of mean(a-b) for paired samples."""
    n = len(a)
    idx = RNG.integers(0, n, size=(n_boot, n))
    diffs = a[idx].mean(axis=1) - b[idx].mean(axis=1)
    lo, hi = np.percentile(diffs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def holm_correction(pvals):
    """Holm step-down correction. Returns adjusted p-values (in original order)."""
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adj_sorted = np.empty(m)
    running_max = 0.0
    for rank, i in enumerate(order):
        val = (m - rank) * p[i]
        running_max = max(running_max, val)
        adj_sorted[rank] = min(running_max, 1.0)
    adj = np.empty(m)
    adj[order] = adj_sorted
    return adj


def fmt(x, nd=4):
    return f"{x:.{nd}f}"


def main(path_a, path_b, name_a="PACA-Affinity", name_b="PWAA+RPE"):
    runs_a, runs_b = load_runs(path_a), load_runs(path_b)
    out = {"datasets": {}, "notes": {}}

    print("=" * 78)
    print(f"Seed-level analysis: {name_a} (n={len(runs_a)} runs) vs {name_b} (n={len(runs_b)} runs)")
    print("=" * 78)

    pvals_raw, pvals_adj_holm = [], []

    for ds in DATASETS:
        common, ma, mb = paired_arrays(runs_a, runs_b, ds, "MAE")
        # Wilcoxon on paired per-run MAE (alternative: 'less' = A has lower MAE)
        diff = ma - mb  # Delta = MAE_A - MAE_B; negative favors model A
        try:
            stat, p_raw = wilcoxon(ma, mb, alternative="less",
                                   zero_method="wilcox", method="exact")
        except ValueError:
            stat, p_raw = 0.0, 1.0  # all differences zero
        ci_lo, ci_hi = bootstrap_ci_diff(ma, mb)
        # matched-pairs rank-biserial correlation (rank-based, Kerby 2014):
        # r = (W- - W+) / (W+ + W-), positive = A tends to have lower MAE
        n = len(diff)
        W_plus = float(stat)                     # sum of ranks of positive diffs
        W_minus = n * (n + 1) / 2 - W_plus       # sum of ranks of negative diffs
        rbc = (W_minus - W_plus) / (W_plus + W_minus)

        rec = {
            "n_runs": len(common),
            "MAE_mean_A": float(ma.mean()), "MAE_std_A": float(ma.std(ddof=1)),
            "MAE_mean_B": float(mb.mean()), "MAE_std_B": float(mb.std(ddof=1)),
            "MAE_diff_mean": float(diff.mean()),
            "MAE_diff_CI95": [ci_lo, ci_hi],
            "wilcoxon_stat": float(stat), "p_raw": float(p_raw),
            "rank_biserial": float(rbc),
            # per-metric mean/std for both models (for full revised Table 5)
            "metrics": {
                m: {
                    "A_mean": float(np.mean([runs_a[s][ds][m] for s in common])),
                    "A_std": float(np.std([runs_a[s][ds][m] for s in common], ddof=1)),
                    "B_mean": float(np.mean([runs_b[s][ds][m] for s in common])),
                    "B_std": float(np.std([runs_b[s][ds][m] for s in common], ddof=1)),
                } for m in METRICS
            },
        }
        out["datasets"][ds] = rec
        pvals_raw.append(float(p_raw))

        print(f"\n[{ds.upper()}]  n_runs={len(common)}  (test n={TEST_SIZES[ds]})")
        for m in METRICS:
            mm = rec["metrics"][m]
            print(f"  {m:5s}  {name_a}: {mm['A_mean']:.4f} +/- {mm['A_std']:.4f}   "
                  f"{name_b}: {mm['B_mean']:.4f} +/- {mm['B_std']:.4f}")
        print(f"  MAE diff mean: {rec['MAE_diff_mean']:+.4f}   "
              f"95% bootstrap CI: [{ci_lo:+.4f}, {ci_hi:+.4f}]")
        print(f"  Wilcoxon stat={stat:.1f}, p(raw)={p_raw:.4g}, rank-biserial={rbc:+.3f}")

    # Holm correction across the 4 datasets
    adj = holm_correction(pvals_raw)
    for ds, p, pa in zip(DATASETS, pvals_raw, adj):
        out["datasets"][ds]["p_holm"] = float(pa)
    print("\nHolm-adjusted p-values:")
    for ds, p, pa in zip(DATASETS, pvals_raw, adj):
        print(f"  {ds:8s} raw={p:.4g}  holm={pa:.4g}")

    # -------- pooled results (sample-size weighted per run) --------
    print("\n[Pooled across datasets] (weighted by per-dataset test size)")
    pooled = {"A": [], "B": []}
    common_seeds = sorted(set(runs_a) & set(runs_b))
    for s in common_seeds:
        tot_n = sum(TEST_SIZES[ds] for ds in DATASETS)
        pa_run = sum(TEST_SIZES[ds] * runs_a[s][ds]["MAE"] for ds in DATASETS) / tot_n
        pb_run = sum(TEST_SIZES[ds] * runs_b[s][ds]["MAE"] for ds in DATASETS) / tot_n
        pooled["A"].append(pa_run)
        pooled["B"].append(pb_run)
    pa, pb = np.array(pooled["A"]), np.array(pooled["B"])
    ci_lo, ci_hi = bootstrap_ci_diff(pa, pb)
    try:
        stat, p_pool = wilcoxon(pa, pb, alternative="less")
    except ValueError:
        stat, p_pool = 0.0, 1.0
    out["pooled"] = {
        "MAE_mean_A": float(pa.mean()), "MAE_std_A": float(pa.std(ddof=1)),
        "MAE_mean_B": float(pb.mean()), "MAE_std_B": float(pb.std(ddof=1)),
        "MAE_diff_CI95": [ci_lo, ci_hi], "wilcoxon_stat": float(stat),
        "p_raw": float(p_pool),
    }
    print(f"  {name_a}: {pa.mean():.4f} +/- {pa.std(ddof=1):.4f}   "
          f"{name_b}: {pb.mean():.4f} +/- {pb.std(ddof=1):.4f}")
    print(f"  diff 95% CI: [{ci_lo:+.4f}, {ci_hi:+.4f}]   Wilcoxon p={p_pool:.4g}")

    # -------- rounding-consistency check (Reviewer 1) --------
    # recompute delta% from DISPLAYED (rounded) MAE values -> the number that
    # must appear in the paper, guaranteeing Table3/Table5 consistency.
    print("\n[Rounding check] delta% recomputed from displayed 4-decimal MAE values")
    for ds in DATASETS:
        ma_d = round(out["datasets"][ds]["MAE_mean_A"], 4)
        mb_d = round(out["datasets"][ds]["MAE_mean_B"], 4)
        delta = (mb_d - ma_d) / mb_d * 100
        out["datasets"][ds]["delta_pct_displayed"] = round(delta, 2)
        print(f"  {ds:8s} {name_a}={ma_d:.4f}  {name_b}={mb_d:.4f}  ->  "
              f"MAE reduction = {delta:.2f}%  (use exactly this value in the paper)")

    with open("seed_level_stats.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nSaved -> seed_level_stats.json")
    print("NOTE: reported values here come from the 10-run means; the representative")
    print("seed (42) single-run values for Table 3 must be taken from seed_results[42],")
    print("and Table 4/5 must be regenerated from this same file - no hand copying.")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python seed_level_analysis.py <paca_results.json> <baseline_results.json>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])