"""
EXPERIMENT 4 — How good are the probability outputs, against the right baseline?

THE DENOMINATOR AGAIN
---------------------
A Brier score of 0.189 against a 0.196 baseline reads as "3% better than doing
nothing", which is what made clean sheets look like the weakest part of the
system. Same error as reading Spearman against 1.0.

Write pi for the TRUE conditional probability of each case. Then

    BS = E[(p - pi)^2] + E[pi(1 - pi)]
       = MSE(p, pi) + pbar(1 - pbar) - Var(pi)

So Var(pi) is the ENTIRE reducible portion: a perfectly calibrated, perfectly
informed model scores pbar(1-pbar) - Var(pi), and everything below that is
coin-flip noise no model removes. The honest score is

    captured = (baseline - BS) / Var(pi)

ESTIMATING Var(pi) IS THE WHOLE DIFFICULTY
------------------------------------------
An earlier version substituted Var(p) — the variance of our OWN predictions —
which assumes the model is calibrated, i.e. assumes the answer. Three estimates
are produced here instead:

  1. Var(p)                naive, circular, retained only as a reference
  2. lower bound           from the floor crossing (see below), assumption-free
  3. debiased resolution   binned, with the within-bin sampling variance removed

THE FLOOR CROSSING IS A BOUND, NOT JUST A SIGN
----------------------------------------------
If BS comes in BELOW the naive floor pbar(1-pbar) - Var(p), rearranging the
identity above gives

    Var(pi) - Var(p) = [floor_est - BS] + MSE(p, pi)

and since MSE >= 0 that is a STRICT LOWER BOUND on how badly the model is
under-dispersed:

    Var(pi) > Var(p) + (floor_est - BS)

The crossing therefore yields a number, not merely the observation that
something is wrong. Because floor_est and BS are computed from the same sample
their errors are correlated, so the difference is bootstrapped on PAIRED
resamples rather than composed from separate standard errors.

THE DIRECT MEASUREMENT IS THE CALIBRATION SLOPE
-----------------------------------------------
Regress y on logit(p). A slope beta > 1 means the true probabilities are more
spread out than the predictions — under-dispersion — and beta is the correction
factor itself. The floor crossing establishes existence; the slope gives
magnitude.

DIRECTION OF THE BIAS, WHICH IS KNOWABLE EVEN WHEN THE LEVEL IS NOT
-------------------------------------------------------------------
Var(pi) sits in the DENOMINATOR of `captured`. Wherever Var(pi) > Var(p), any
figure computed with Var(p) is an UPPER BOUND on the true captured share, not an
approximation of unknown sign.

REMOVING THE BINNING BIAS
-------------------------
The binned resolution term sum_b (n_b/N)(pihat_b - pbar)^2 is inflated by the
sampling noise in each pihat_b. Its expectation carries an extra
(1/N) sum_b pihat_b(1 - pihat_b), which is subtracted. Omitting that subtraction
would inflate Var(pi), deflate `captured`, and reproduce exactly the same class
of error one level up.
"""

from __future__ import annotations

import sqlite3
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fplbrain import backtest, models  # noqa: E402

DB = "data/fpl.db"
SEASON, HIST = "2025-26", ["2024-25"]
EPS = 1e-6
N_BOOT = 2000
N_BINS = 10
RNG = np.random.default_rng(17)


# ---------------------------------------------------------------- estimators


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def calibration_slope(p: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """
    Fit y ~ Bernoulli(sigmoid(a + b * logit(p))).

    b == 1 with a == 0 is perfect calibration.
    b  > 1 means predictions are compressed toward the base rate, i.e. the true
           probabilities are MORE extreme than we say: under-dispersion.
    b  < 1 means over-confidence.
    """
    x = _logit(np.asarray(p, float))
    y = np.asarray(y, float)

    def nll(theta):
        a, b = theta
        z = a + b * x
        # log(1 + exp(z)) computed stably
        return float(np.sum(np.logaddexp(0.0, z) - y * z))

    res = minimize(nll, x0=np.array([0.0, 1.0]), method="BFGS")
    return float(res.x[0]), float(res.x[1])


def var_pi_binned(p: np.ndarray, y: np.ndarray, n_bins: int = N_BINS) -> dict:
    """
    Var(pi) from a binned decomposition, with the within-bin sampling variance
    removed. Binning coarsens, so this remains a LOWER bound on Var(pi).
    """
    p = np.asarray(p, float)
    y = np.asarray(y, float)
    n = len(p)
    # Quantile bins on the prediction, deduplicated for ties.
    edges = np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)))
    idx = np.clip(np.digitize(p, edges[1:-1], right=True), 0, len(edges) - 2)
    pbar = y.mean()

    resolution = 0.0
    within = 0.0
    used = 0
    for b in np.unique(idx):
        m = idx == b
        nb = int(m.sum())
        if nb < 2:
            continue
        pihat = y[m].mean()
        resolution += nb / n * (pihat - pbar) ** 2
        within += pihat * (1 - pihat) / n
        used += 1
    return {"resolution_raw": resolution,
            "within_bin_correction": within,
            "var_pi_debiased": max(resolution - within, 0.0),
            "n_bins_used": used}


def paired_bootstrap_crossing(p: np.ndarray, y: np.ndarray,
                              n_boot: int = N_BOOT) -> tuple[float, float]:
    """
    Bootstrap CI for (floor_est - BS), resampling OBSERVATIONS so both terms move
    together. Their errors are correlated — they share a sample — so separate
    standard errors would not compose.
    """
    p = np.asarray(p, float)
    y = np.asarray(y, float)
    n = len(p)
    out = np.empty(n_boot)
    for i in range(n_boot):
        k = RNG.integers(0, n, n)
        pb, yb = p[k], y[k]
        base = yb.mean() * (1 - yb.mean())
        floor = base - pb.var()
        out[i] = floor - ((pb - yb) ** 2).mean()
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def decompose(p: pd.Series, y: pd.Series, label: str) -> dict:
    p = np.asarray(p, float)
    y = np.asarray(y, float)
    pbar = y.mean()
    baseline = pbar * (1 - pbar)
    bs = float(((p - y) ** 2).mean())
    var_p = float(p.var())

    floor_naive = baseline - var_p
    crossing = floor_naive - bs                     # > 0 => under-dispersed
    lo, hi = paired_bootstrap_crossing(p, y)
    binned = var_pi_binned(p, y)
    a, beta = calibration_slope(p, y)

    # Lower bound on Var(pi) implied by the crossing (valid only when crossing>0)
    var_pi_lb = var_p + max(crossing, 0.0)
    # Best available estimate: the larger of the two lower bounds.
    var_pi_est = max(binned["var_pi_debiased"], var_pi_lb if crossing > 0 else 0.0)

    def cap(vpi):
        return 100 * (baseline - bs) / vpi if vpi > 1e-12 else float("nan")

    return {
        "component": label,
        "n": len(p),
        "predicted_mean": round(float(p.mean()), 4),
        "actual_rate": round(float(pbar), 4),
        "baseline_brier": round(baseline, 4),
        "our_brier": round(bs, 4),
        "naive_pct_better": round(100 * (baseline - bs) / baseline, 1),
        "var_p_naive": round(var_p, 5),
        "crossing": round(crossing, 5),
        "crossing_ci": (round(lo, 5), round(hi, 5)),
        "crossing_significant": bool(lo > 0),
        "calibration_slope": round(beta, 3),
        "var_pi_lower_bound": round(var_pi_lb, 5) if crossing > 0 else None,
        "var_pi_binned_debiased": round(binned["var_pi_debiased"], 5),
        "resolution_raw": round(binned["resolution_raw"], 5),
        "within_bin_correction": round(binned["within_bin_correction"], 5),
        # `captured` falls as Var(pi) rises, so ANY underestimate of Var(pi)
        # inflates it. The binned figure is a lower bound on Var(pi) (binning
        # coarsens away within-bin variation), so captured_using_var_pi is
        # always an UPPER bound. Whether captured_using_var_p is an upper or a
        # lower bound depends on the SIGN of Var(pi) - Var(p), which the
        # crossing and the slope measure rather than assume.
        "captured_using_var_p": round(cap(var_p), 1),
        "var_p_bound_direction": ("upper" if var_pi_est > var_p else "lower"),
        "captured_using_var_pi_UPPER_BOUND": round(cap(var_pi_est), 1),
    }


# ---------------------------------------------------------------- main


if __name__ == "__main__":
    conn = sqlite3.connect(DB)
    df, _ = backtest.run_backtest(conn, SEASON, HIST, start_gw=8, end_gw=38)

    rows = []
    cs = df[df["position"].isin(["GK", "DEF"]) & (df["actual_minutes"] >= 60)]
    rows.append(decompose(cs["p_clean_sheet"], cs["actual_cs"], "clean sheet"))

    dc = df[(df["position"] != "GK") & (df["actual_minutes"] >= 60)].copy()
    dc = dc[dc["actual_dc_actions"].notna()]
    dc["hit"] = (dc["actual_dc_actions"]
                 >= dc["position"].map(models.DEFCON_THRESHOLD)).astype(int)
    rows.append(decompose(dc["p_defcon"], dc["hit"], "DefCon"))

    for r in rows:
        print("=" * 74)
        print(f"{r['component'].upper()}   n={r['n']}")
        print("=" * 74)
        print(f"  predicted mean {r['predicted_mean']}   actual rate {r['actual_rate']}")
        print(f"  baseline Brier {r['baseline_brier']}   ours {r['our_brier']}"
              f"   ({r['naive_pct_better']}% better — the misleading read)")
        print()
        print("  UNDER-DISPERSION")
        sig = "SIGNIFICANT" if r["crossing_significant"] else "not significant"
        print(f"    floor(Var(p)) - Brier      {r['crossing']:+.5f}   "
              f"95% CI {r['crossing_ci']}  [{sig}]")
        if r["crossing"] > 0:
            print(f"    => strict lower bound:     Var(pi) > Var(p) + "
                  f"{r['crossing']:.5f} = {r['var_pi_lower_bound']}")
        print(f"    calibration slope beta     {r['calibration_slope']}"
              f"   (>1 = under-dispersed; beta IS the correction)")
        print()
        print("  Var(pi) ESTIMATES")
        print(f"    Var(p), naive/circular     {r['var_p_naive']}")
        print(f"    binned resolution (raw)    {r['resolution_raw']}")
        print(f"      less within-bin noise    {r['within_bin_correction']}")
        print(f"    = debiased Var(pi)         {r['var_pi_binned_debiased']}")
        print()
        print("  CAPTURED SHARE")
        d = r["var_p_bound_direction"]
        print(f"    using Var(p)   {r['captured_using_var_p']}%"
              f"   <- a {d.upper()} bound (Var(pi) is "
              f"{'>' if d == 'upper' else '<'} Var(p) here)")
        print(f"    using Var(pi)  {r['captured_using_var_pi_UPPER_BOUND']}%"
              f"   <- UPPER bound; binned Var(pi) is itself a lower bound")
        print()

    # Dispersion check across teams, WITH the sampling-noise correction.
    #
    # A raw comparison of modelled vs realised team-level spread is not evidence
    # of anything: each team's realised clean-sheet rate is a proportion over
    # ~30 matches, so its sampling variance p(1-p)/n_team inflates the observed
    # spread. Comparing an unnoised model quantity against a noised empirical
    # one is the SAME bias the within-bin correction above removes, one level up.
    # The independent unit is the MATCH, not the player-row. Every GK and
    # defender from one club in one fixture shares the same clean-sheet outcome
    # (exp2 measures r = 0.51 among same-team defenders), so a team's ~152 rows
    # carry only ~30 independent observations. Using the row count would
    # understate the sampling noise fivefold — §4's correlation finding biting
    # §3's dispersion check.
    team = cs.groupby("team").agg(pred=("p_clean_sheet", "mean"),
                                  actual=("actual_cs", "mean"),
                                  rows=("actual_cs", "size"),
                                  matches=("gw", "nunique"))
    obs_var = float(team.actual.var())
    noise_rows = float((team.actual * (1 - team.actual) / team.rows).mean())
    noise_matches = float((team.actual * (1 - team.actual) / team.matches).mean())
    print("clean-sheet spread across teams (dispersion check)")
    print(f"  modelled                     sd {team.pred.std():.3f}")
    print(f"  realised, raw                sd {np.sqrt(obs_var):.3f}"
          f"   <- inflated by sampling noise")
    print(f"    if rows were independent   noise {noise_rows:.5f} -> "
          f"sd {np.sqrt(max(obs_var - noise_rows, 0)):.3f}   (WRONG: they are not)")
    print(f"    per independent match      noise {noise_matches:.5f} -> "
          f"sd {np.sqrt(max(obs_var - noise_matches, 0)):.3f}   <- compare against modelled")
