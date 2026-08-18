"""
EXPERIMENT 3b — the same comparison, properly powered and PAIRED.

Exp 3 on a single window proves nothing: between-window variance dwarfs any
plausible strategy difference. Here we compare PAIRED differences within each
window, which removes the window effect entirely.

SAMPLE, STATED ACCURATELY
-------------------------
`WINDOWS = [(s, 6) for s in range(10, 33, 3)]` is **8 windows, stride 3, width
6**, covering GW10-36, with **15 resamples** per resampled solve.

Earlier versions of this file claimed "ten overlapping windows" in the docstring
and "12 windows, stride 2" in the comment. Both were wrong; the code has always
produced 8. The published figures were right and only the description lied,
which is its own lesson: a stale comment makes a test look better powered than
it is, and the caveat that follows from n=8 then reads as excessive caution.

DO THE OVERLAPPING WINDOWS BREAK INDEPENDENCE?
----------------------------------------------
Stride 3 with width 6 means adjacent windows share half their gameweeks, so the
worry is that the 8 paired differences are positively autocorrelated at lag 1,
making a plain SE and a Wilcoxon test ANTI-CONSERVATIVE.

MEASURED, THAT WORRY DOES NOT SURVIVE. The overlap does show up in the LEVELS
(lag-1 autocorrelation about +0.18 for naive, +0.20 for shrunk), exactly as the
construction implies. But PAIRING removes the window effect, and the paired
differences come out slightly NEGATIVE (-0.17, -0.07) — no positive dependence,
so the naive SE is not inflated after all.

That is worth stating precisely because the reasoning was sound and the
conclusion was still wrong: window overlap does not automatically contaminate a
paired test, because the pairing is what the overlap acts on. Section 0 of the
output measures it rather than assuming it either way.

Three estimates are produced regardless, so the conclusion does not rest on that
diagnostic being right:

  1. OVERLAPPING, naive SE   — usable, given section 0
  2. NON-OVERLAPPING          — windows 10/16/22/28, stride 6, zero shared
                                gameweeks. Assumption-free, n=4. This is a
                                SUBSET of the eight, so it costs no extra solves.
  3. MOVING-BLOCK BOOTSTRAP   — over the 8 overlapping differences, block length
                                2, which tolerates lag-1 dependence if present.

All three agree, which is the actual reason to trust the finding.

HARD LIMIT: one season of DefCon-era data admits at most ~5 independent
6-gameweek windows. No amount of statistical care fixes that. It needs 2026-27.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis import exp3_robust as e3  # noqa: E402

WINDOWS = [(s, 6) for s in range(10, 33, 3)]   # 8 windows, stride 3, width 6
NON_OVERLAPPING = [10, 16, 22, 28]             # stride 6 -> zero shared gameweeks
N_RESAMPLES = 15
BLOCK = 2                                      # lag-1 dependence -> blocks of 2
N_BOOT = 5000
RNG = np.random.default_rng(3)


def run_windows() -> pd.DataFrame:
    rows = []
    for start, n in WINDOWS:
        hz = list(range(start, start + n))
        proj = e3.projections_for(start, n)
        act = e3.actuals(hz)
        naive = e3.build_naive(proj, hz)
        shrunk = e3.build_shrunk(proj, hz)
        resamp, _ = e3.build_resampled(proj, hz, n_resamples=N_RESAMPLES)
        for r in (naive, shrunk, resamp):
            d = e3.evaluate(r, act, hz)
            d.update(window=start)
            rows.append(d)
        print(f"  GW{start}-{start + n - 1} done", flush=True)
    return pd.DataFrame(rows)


def lag1(x: np.ndarray) -> float:
    if len(x) < 3 or np.std(x) == 0:
        return 0.0
    return float(np.corrcoef(x[:-1], x[1:])[0, 1])


def effective_n(n: int, rho: float) -> float:
    """AR(1) effective sample size. Positive rho shrinks it."""
    rho = float(np.clip(rho, -0.99, 0.99))
    return n * (1 - rho) / (1 + rho)


def block_bootstrap_ci(d: np.ndarray, block: int = BLOCK,
                       n_boot: int = N_BOOT) -> tuple[float, float]:
    """Moving-block bootstrap of the mean, preserving local dependence."""
    n = len(d)
    if n < block + 1:
        return (np.nan, np.nan)
    starts = np.arange(n - block + 1)
    k = int(np.ceil(n / block))
    means = np.empty(n_boot)
    for b in range(n_boot):
        idx = RNG.choice(starts, size=k, replace=True)
        sample = np.concatenate([d[i:i + block] for i in idx])[:n]
        means[b] = sample.mean()
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def report(piv: pd.DataFrame) -> None:
    print("\nPAIRED DIFFERENCES (XI + captain, per 6-GW window), n =", len(piv))
    print(piv.round(0).to_string())

    n = len(piv)
    # For INDEPENDENT draws with an estimated mean, the sample lag-1
    # autocorrelation is biased downward: E[r1] ~ -1/(n-1), and SE(r1) ~ 1/sqrt(n).
    # At n=8 that is -0.143 +/- 0.354, so almost nothing is distinguishable.
    exp_r1 = -1.0 / (n - 1)
    se_r1 = 1.0 / np.sqrt(n)
    print("\n" + "=" * 74)
    print("0. CAN THE AUTOCORRELATION TELL US ANYTHING? — no, at this n")
    print("=" * 74)
    print(f"  white-noise reference at n={n}:  E[r1] = {exp_r1:+.3f}, "
          f"SE(r1) = {se_r1:.3f}")
    for s in ("naive", "shrunk", "resampled"):
        r = lag1(piv[s].to_numpy())
        print(f"    levels      {s:<10} r1 {r:+.3f}   "
              f"({(r - exp_r1) / se_r1:+.1f} SE from white noise)")
    for s in ("shrunk", "resampled"):
        d = (piv[s] - piv["naive"]).dropna().to_numpy()
        r = lag1(d)
        print(f"    differences {s:<10} r1 {r:+.3f}   "
              f"({(r - exp_r1) / se_r1:+.1f} SE from white noise)")
    print("""
  READ THIS CORRECTLY. The differences sit ON the white-noise expectation, so
  there is NO negative dependence to explain — just the standard small-sample
  bias. And the levels, which ARE dependent by construction, land under 1 SE
  from white noise, so the ACF cannot detect the thing we know is there.

  The autocorrelation therefore carries no weight in either direction at n=8.
  The warrant for using the naive SE is section 3: the moving-block bootstrap,
  which tolerates dependence, returns essentially the same interval. Two
  estimators agreeing is the evidence; the ACF is not.""")

    print("\n" + "=" * 74)
    print("1. OVERLAPPING WINDOWS, naive SE")
    print("=" * 74)
    for s in ("shrunk", "resampled"):
        d = (piv[s] - piv["naive"]).dropna().to_numpy()
        se = d.std(ddof=1) / np.sqrt(len(d))
        stat, p = wilcoxon(d) if len(set(d)) > 1 else (np.nan, 1.0)
        print(f"{s:>10} vs naive:  mean {d.mean():+6.1f}  SE {se:4.1f}  "
              f"95% CI [{d.mean() - 1.96 * se:+.1f}, {d.mean() + 1.96 * se:+.1f}]  "
              f"wins {int((d > 0).sum())}/{len(d)}  p={p:.3f}")

    print("\n" + "=" * 74)
    print(f"2. NON-OVERLAPPING WINDOWS {NON_OVERLAPPING} — statistically clean, n=4")
    print("=" * 74)
    sub = piv.loc[[w for w in NON_OVERLAPPING if w in piv.index]]
    for s in ("shrunk", "resampled"):
        d = (sub[s] - sub["naive"]).dropna().to_numpy()
        se = d.std(ddof=1) / np.sqrt(len(d))
        print(f"{s:>10} vs naive:  mean {d.mean():+6.1f}  SE {se:4.1f}  "
              f"95% CI [{d.mean() - 1.96 * se:+.1f}, {d.mean() + 1.96 * se:+.1f}]  "
              f"wins {int((d > 0).sum())}/{len(d)}")

    print("\n" + "=" * 74)
    print(f"3. MOVING-BLOCK BOOTSTRAP over the {len(piv)} overlapping windows "
          f"(block={BLOCK})")
    print("=" * 74)
    for s in ("shrunk", "resampled"):
        d = (piv[s] - piv["naive"]).dropna().to_numpy()
        lo, hi = block_bootstrap_ci(d)
        print(f"{s:>10} vs naive:  mean {d.mean():+6.1f}  95% CI [{lo:+.1f}, {hi:+.1f}]")

    print("\n" + "=" * 74)
    print("4. POWER — what could this design ever detect?")
    print("=" * 74)
    # Two-sided alpha=0.05, 80% power: n = ((z_a + z_b) * SD / delta)^2
    Z = 1.959964 + 0.841621
    print(f"  {'strategy':<12} {'SD':>6} {'SE':>6} {'MDE at n=' + str(n):>12}"
          f" {'n for +2':>10} {'n for +5':>10} {'n for +15':>10}")
    for s in ("shrunk", "resampled"):
        d = (piv[s] - piv["naive"]).dropna().to_numpy()
        sd = d.std(ddof=1)
        se = sd / np.sqrt(len(d))
        need = {delta: int(np.ceil((Z * sd / delta) ** 2)) for delta in (2, 5, 15)}
        print(f"  {s:<12} {sd:>6.1f} {se:>6.1f} {Z * se:>11.1f}p"
              f" {need[2]:>10} {need[5]:>10} {need[15]:>10}")
    print("""
  A trigger below the MDE is a measurement that will never resolve. Note the two
  strategies differ by an order of magnitude: shrinkage moves the squad barely at
  all, so its differences are tight and a small effect is reachable; resampling
  reshuffles the squad wholesale, so its differences are wide and a small effect
  is not reachable at any realistic n.

  Against §7's ceiling of about 5 INDEPENDENT 6-gameweek windows per season, read
  the 'n for' columns in seasons, not in windows.""")

    print("\n" + "=" * 74)
    print("VARIANCE DECOMPOSITION — why none of this resolves")
    print("=" * 74)
    bw = piv.mean(axis=1).std(ddof=1)
    ws = piv.std(axis=1).mean()
    print(f"  between-window SD                  {bw:.1f}")
    print(f"  between-strategy SD within window  {ws:.1f}")
    print(f"  ratio                              {bw / ws:.1f}x")
    print("\n  Which gameweeks you are in dominates which optimiser you use.")


if __name__ == "__main__":
    df = run_windows()
    piv = df.pivot(index="window", columns="strategy", values="total_with_captain")
    report(piv)
