"""
EXPERIMENT 4 — How good are the probability outputs, against the right baseline?

LIMITATIONS.md §3 quoted Brier decompositions that no script in this directory
produced. This is that script.

THE DENOMINATOR AGAIN
---------------------
A Brier score of 0.188 against a 0.194 baseline reads as "3% better than doing
nothing", which is the reading that made clean sheets look like the weakest part
of the system. It is the same error as reading Spearman against 1.0.

For a binary outcome with base rate p̄:

  predicting the base rate every time  ->  Brier = p̄(1 - p̄)
  knowing each case's TRUE probability ->  Brier = p̄(1 - p̄) - Var(p)

So Var(p) — the spread of true probabilities across cases — is the ENTIRE
reducible portion. Everything else is coin-flip noise that no model removes.
The honest score is therefore

  captured = (baseline - ours) / (baseline - floor)

ASSUMPTION, STATED
------------------
Var(p) is unobservable, so it is estimated by the variance of the MODEL's own
predicted probabilities. That assumes the model is roughly calibrated: if it is
over-dispersed the floor is too low and `captured` is understated, if
under-dispersed the reverse. The predicted-vs-actual means printed alongside are
the check on that — if they agree closely, the assumption is reasonable.
"""

from __future__ import annotations

import sqlite3
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fplbrain import backtest, models  # noqa: E402

DB = "data/fpl.db"
SEASON, HIST = "2025-26", ["2024-25"]


def decompose(p: pd.Series, y: pd.Series, label: str) -> dict:
    p = p.astype(float)
    y = y.astype(float)
    base_rate = y.mean()
    baseline = base_rate * (1 - base_rate)
    floor = max(baseline - float(p.var(ddof=0)), 0.0)
    ours = float(((p - y) ** 2).mean())
    reducible = baseline - floor
    captured = (baseline - ours) / reducible if reducible > 1e-12 else float("nan")
    # A `captured` above 100% is not a triumph, it FALSIFIES THE FLOOR. It means
    # our Brier is below a floor that was derived from our own predicted
    # variance, which can only happen if the model is UNDER-DISPERSED — the true
    # probabilities are spread wider than our predictions, so Var(p) was
    # underestimated and the floor set too high. When this fires, the `captured`
    # figure for that component is not interpretable and must not be quoted.
    valid = captured <= 1.0
    return {
        "component": label,
        "n": int(len(p)),
        "predicted_mean": round(float(p.mean()), 4),
        "actual_rate": round(float(base_rate), 4),
        "baseline_brier": round(baseline, 4),
        "floor_brier": round(floor, 4),
        "our_brier": round(ours, 4),
        "naive_pct_better": round(100 * (baseline - ours) / baseline, 1),
        "captured_pct": round(100 * captured, 1) if valid else float("nan"),
        "floor_valid": valid,
        "pred_sd": round(float(p.std(ddof=0)), 4),
    }


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

    res = pd.DataFrame(rows)
    print("BRIER DECOMPOSITION — against the achievable floor, not against 0.5\n")
    print(res.to_string(index=False))
    print("\n`naive_pct_better` is the misleading reading; `captured_pct` is the real one")
    print("— but only where floor_valid is True.")
    for r in rows:
        if not r["floor_valid"]:
            print(f"\n  !! {r['component']}: our Brier {r['our_brier']} is BELOW the "
                  f"estimated floor {r['floor_brier']}.")
            print("     The floor is derived from our own predicted variance, so this")
            print("     means the model is UNDER-DISPERSED: true probabilities vary more")
            print("     than our predictions do. The floor is overstated and no")
            print("     `captured` figure can be quoted for this component.")

    # Does the modelled clean-sheet spread match the real spread across teams?
    # If the model's dispersion is far from reality, the floor above is wrong.
    team = cs.groupby("team").agg(pred=("p_clean_sheet", "mean"),
                                  actual=("actual_cs", "mean"))
    print("\nclean-sheet spread across teams (dispersion check on the floor)")
    print(f"  modelled  {team.pred.min():.3f} to {team.pred.max():.3f}  "
          f"sd {team.pred.std():.3f}")
    print(f"  realised  {team.actual.min():.3f} to {team.actual.max():.3f}  "
          f"sd {team.actual.std():.3f}")
