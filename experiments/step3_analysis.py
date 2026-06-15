"""
experiments/step3_analysis.py — Delta v2 Core Analysis.

Key changes from v1:
  1. Four-quadrant framework (replaces orthogonalization)
  2. Disagreement source decomposition (A vs B vs C)
  3. Hodrick (1992) standard errors (alongside Newey-West)

Four Quadrants:
  ┌─────────────────┬──────────────────┐
  │  Concordant      │  Overconfidence   │
  │  Low H, Low D    │  Low H, High D    │
  │  (agreed & sure) │  (confident but   │
  │                  │   disagree)       │
  ├─────────────────┼──────────────────┤
  │  Uncertain       │  Chaotic          │
  │  High H, Low D   │  High H, High D   │
  │  (uniformative)  │  (no consensus)   │
  └─────────────────┴──────────────────┘

Usage:
    python step3_analysis.py --group B --output results/
    python step3_analysis.py --group A B C --compare
"""

import argparse
import json
import numpy as np
from pathlib import Path
from scipy import stats
from scipy.stats import entropy


# ── Entropy & Divergence ──────────────────────────────────────────

def rating_to_probs(rating: float) -> np.ndarray:
    """Convert 1-10 rating to (neg, neu, pos) probability distribution.

    Uses logistic-centered mapping:
      x = (rating - 5.5) / 2.0
      p_neg ∝ exp(-x), p_neu ∝ 1, p_pos ∝ exp(x)
    """
    x = (rating - 5.5) / 2.0
    p_neg = np.exp(-x)
    p_neu = 1.0
    p_pos = np.exp(x)
    total = p_neg + p_neu + p_pos
    return np.array([p_neg / total, p_neu / total, p_pos / total])


def compute_entropy_measures(s: float, t: float, f: float) -> dict:
    """Compute all disagreement measures from three agent ratings.

    Returns dict with:
      - H_smooth: Shannon entropy of average belief distribution
      - JS: Jensen-Shannon divergence
      - D_post: Standard deviation of ratings
      - H_premium: H_smooth orthogonalized (legacy, for comparison)
    """
    sp = rating_to_probs(s)
    tp = rating_to_probs(t)
    fp = rating_to_probs(f)
    avg_p = (sp + tp + fp) / 3

    # H_smooth: entropy of average belief
    h_smooth = entropy(avg_p, base=2)

    # JS: H(avg) - avg(H(individual))
    h_sp = entropy(sp, base=2)
    h_tp = entropy(tp, base=2)
    h_fp = entropy(fp, base=2)
    js = h_smooth - np.mean([h_sp, h_tp, h_fp])

    # D_post: rating dispersion
    d_post = np.std([s, t, f])

    return {
        "H_smooth": h_smooth,
        "JS": js,
        "D_post": d_post,
    }


def assign_quadrant(h_smooth: float, d_post: float,
                    h_median: float, d_median: float) -> str:
    """Assign stock-month to one of four quadrants.

    Medians are computed from the full sample.
    """
    low_h = h_smooth < h_median
    low_d = d_post < d_median

    if low_h and low_d:
        return "Concordant"
    elif low_h and not low_d:
        return "Overconfidence"
    elif not low_h and low_d:
        return "Uncertain"
    else:
        return "Chaotic"


# ── Fama-MacBeth Regression ──────────────────────────────────────

def newey_west_se(betas: np.ndarray, lag: int = 4) -> tuple:
    """Newey-West HAC standard error for Fama-MacBeth coefficients."""
    T = len(betas)
    if T < 2:
        return 0.0, 0.0
    mean_b = np.mean(betas)
    gamma = np.zeros(lag + 1)
    gamma[0] = np.var(betas, ddof=1)
    for j in range(1, lag + 1):
        w = 1 - j / (lag + 1)
        gamma[j] = w * np.mean((betas[j:] - mean_b) * (betas[:-j] - mean_b))
    nw_var = gamma[0] + 2 * sum(gamma[1:])
    se = np.sqrt(max(nw_var / T, 0))
    t_stat = mean_b / se if se > 0 else 0
    return se, t_stat


def hodrick_se(betas: np.ndarray, returns: np.ndarray,
               horizon: int = 1) -> tuple:
    """Hodrick (1992) standard error for overlapping return regressions.

    Corrects for serial correlation induced by overlapping multi-period
    returns, which Newey-West may under-correct.

    Reference: Hodrick, R. (1992). "Dividend Yields and Expected
    Stock Returns: Alternative Procedures for Inference and Measurement
    of Risk." Review of Financial Studies, 5(3), 357-386.
    """
    T = len(betas)
    if T < 2:
        return 0.0, 0.0

    mean_b = np.mean(betas)

    # Hodrick's method: use 1-period ahead prediction errors
    # scaled by the predictor, then sum
    pred_errors = []
    for t in range(T - horizon):
        for h in range(horizon):
            if t + h < T:
                e = returns[t + h] - np.mean(returns)
                pred_errors.append(betas[t] * e)

    # Var of the scaled sum
    if len(pred_errors) < 2:
        return newey_west_se(betas, lag=4)

    hodrick_var = np.var(pred_errors, ddof=1) / T
    se = np.sqrt(max(hodrick_var, 0))
    t_stat = mean_b / se if se > 0 else 0
    return se, t_stat


def fama_macbeth(panel: list, y_col: str, x_cols: list,
                 nw_lag: int = 4, horizon: int = 1,
                 returns_col: str = "ret") -> dict:
    """Fama-MacBeth regression with both NW and Hodrick SEs.

    Returns dict with coefficients, SEs, t-stats for both methods.
    """
    from numpy.linalg import lstsq

    # Collect cross-sectional regressions
    months = sorted(set(row["month"] for row in panel))
    betas = {col: [] for col in x_cols}

    for month in months:
        month_data = [r for r in panel if r["month"] == month]
        if len(month_data) < 10:
            continue

        y = np.array([r[y_col] for r in month_data])
        X = np.column_stack([
            np.ones(len(month_data)),
            *[np.array([r[col] for r in month_data]) for col in x_cols]
        ])

        try:
            result, _, _, _ = lstsq(X, y, rcond=None)
            for i, col in enumerate(x_cols):
                betas[col].append(result[i + 1])
        except Exception:
            continue

    results = {}
    for col in x_cols:
        b = np.array(betas[col])
        if len(b) < 10:
            continue

        mean_b = np.mean(b)

        # Newey-West
        nw_se, nw_t = newey_west_se(b, lag=nw_lag)

        # Hodrick
        returns = np.array([r[returns_col] for r in panel if r["month"] in months])
        hod_se, hod_t = hodrick_se(b, returns, horizon=horizon)

        results[col] = {
            "coef": mean_b,
            "nw_se": nw_se,
            "nw_t": nw_t,
            "hod_se": hod_se,
            "hod_t": hod_t,
            "n_months": len(b),
        }

    return results


# ── Four-Quadrant Analysis ────────────────────────────────────────

def four_quadrant_analysis(panel: list) -> dict:
    """Four-quadrant portfolio analysis.

    For each quadrant, compute:
      - Average next-period return
      - Sharpe ratio
      - t-test vs market
    """
    # Compute medians for quadrant assignment
    h_values = [r["H_smooth"] for r in panel]
    d_values = [r["D_post"] for r in panel]
    h_median = np.median(h_values)
    d_median = np.median(d_values)

    # Assign quadrants
    for row in panel:
        row["quadrant"] = assign_quadrant(
            row["H_smooth"], row["D_post"], h_median, d_median
        )

    # Compute stats per quadrant
    quadrants = ["Concordant", "Overconfidence", "Uncertain", "Chaotic"]
    results = {}

    for q in quadrants:
        q_data = [r for r in panel if r["quadrant"] == q]
        if not q_data:
            continue

        returns = np.array([r["ret"] for r in q_data])
        excess = np.array([r.get("excess_ret", r["ret"]) for r in q_data])

        results[q] = {
            "n_obs": len(q_data),
            "mean_ret": np.mean(returns),
            "std_ret": np.std(returns),
            "sharpe": np.mean(excess) / np.std(excess) if np.std(excess) > 0 else 0,
            "t_vs_zero": stats.ttest_1samp(returns, 0).statistic if len(returns) > 1 else 0,
            "p_vs_zero": stats.ttest_1samp(returns, 0).pvalue if len(returns) > 1 else 1,
        }

    # Long-short: Overconfidence - Concordant (core test)
    oc_rets = np.array([r["ret"] for r in panel if r["quadrant"] == "Overconfidence"])
    co_rets = np.array([r["ret"] for r in panel if r["quadrant"] == "Concordant"])

    if len(oc_rets) > 1 and len(co_rets) > 1:
        diff = np.mean(oc_rets) - np.mean(co_rets)
        se_diff = np.sqrt(np.var(oc_rets)/len(oc_rets) + np.var(co_rets)/len(co_rets))
        t_diff = diff / se_diff if se_diff > 0 else 0

        results["OC_minus_CO"] = {
            "diff": diff,
            "t_stat": t_diff,
            "n_oc": len(oc_rets),
            "n_co": len(co_rets),
        }

    return results


# ── Main ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Delta v2 Core Analysis")
    parser.add_argument("--group", default="B", help="Experiment group")
    parser.add_argument("--input-dir", default="results", help="Input directory")
    parser.add_argument("--output-dir", default="results", help="Output directory")
    parser.add_argument("--compare", action="store_true",
                        help="Compare A vs B vs C")

    args = parser.parse_args()

    # Load ratings
    ratings_path = Path(args.input_dir) / f"agent_ratings_group{args.group}.json"
    if not ratings_path.exists():
        print(f"Ratings not found: {ratings_path}")
        print("Run step2_scoring.py first.")
        return

    with open(ratings_path) as f:
        ratings = json.load(f)

    # Build panel
    panel = []
    for period, tickers in ratings.items():
        for ticker, agent_ratings in tickers.items():
            s = agent_ratings.get("sentiment", {}).get("rating", 5)
            t = agent_ratings.get("technical", {}).get("rating", 5)
            f = agent_ratings.get("fundamental", {}).get("rating", 5)

            measures = compute_entropy_measures(s, t, f)
            measures["ticker"] = ticker
            measures["month"] = period
            measures["sentiment"] = s
            measures["technical"] = t
            measures["fundamental"] = f

            panel.append(measures)

    print(f"Panel: {len(panel)} stock-month observations")

    # Four-quadrant analysis
    q_results = four_quadrant_analysis(panel)
    print("\n=== Four-Quadrant Analysis ===")
    for q, stats_dict in q_results.items():
        print(f"\n  {q}:")
        for k, v in stats_dict.items():
            if isinstance(v, float):
                print(f"    {k}: {v:.4f}")
            else:
                print(f"    {k}: {v}")

    # Fama-MacBeth with dual SEs
    fm_results = fama_macbeth(panel, "ret", ["H_smooth", "JS", "D_post"])
    print("\n=== Fama-MacBeth (NW + Hodrick) ===")
    for var, res in fm_results.items():
        print(f"\n  {var}:")
        print(f"    coef = {res['coef']:.6f}")
        print(f"    NW:   se={res['nw_se']:.6f}  t={res['nw_t']:.2f}")
        print(f"    Hod:  se={res['hod_se']:.6f}  t={res['hod_t']:.2f}")

    # Save
    output = {
        "group": args.group,
        "n_obs": len(panel),
        "quadrants": q_results,
        "fama_macbeth": fm_results,
    }
    out_path = Path(args.output_dir) / f"analysis_group{args.group}.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
