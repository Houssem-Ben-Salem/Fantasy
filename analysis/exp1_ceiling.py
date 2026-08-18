"""
EXPERIMENT 1 — Where is the ceiling?

A Spearman of 0.33 sounds poor in isolation. But FPL points are a lumpy,
low-count outcome: a player with 0.5 xG in a match either scores or doesn't, and
no model can know which. So the honest question is not "how close to 1.0 are we"
but "how close to the BEST ACHIEVABLE are we".

We estimate the ceiling by building oracles that use information no real
pre-deadline model could possibly have, then measuring where they top out.

  oracle_minutes   our rates, but the player's ACTUAL minutes for that gameweek
  oracle_xg        actual minutes AND actual xG/xA generated in that match,
                   converted to expected points. This knows the player's
                   underlying performance and still cannot know finishing luck.
  oracle_full      as above plus the actual clean sheet and DefCon outcome.
                   Only finishing, bonus and cards remain random.
  actual           the realised points. Spearman 1.0 by construction, a sanity check.

The gap between our model and oracle_xg is ADDRESSABLE modelling error.
The gap between oracle_full and 1.0 is IRREDUCIBLE noise.
Knowing the split tells us where effort is worth spending.

NOTE ON THE SAMPLE. We score only players who featured (minutes >= 1). That is a
deliberate choice and it changes the task: it removes the easy negatives, so the
Spearman here is not directly comparable to `python -m fplbrain.backtest`, which
scores everyone. Both the model and the oracles are scored on the same filtered
sample, so the RATIO — the share of achievable signal captured — is sound, which
is the quantity this experiment exists to produce.
"""

from __future__ import annotations

import sqlite3
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")

# Runnable as `python analysis/exp1_ceiling.py` from the repo root, which puts
# analysis/ on sys.path rather than the root itself.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fplbrain import backtest, models  # noqa: E402

DB = "data/fpl.db"
SEASON, HIST = "2025-26", ["2024-25"]

GOAL_PTS = {"GK": 6, "DEF": 6, "MID": 5, "FWD": 4}
CS_PTS = {"GK": 4, "DEF": 4, "MID": 1, "FWD": 0}


def build() -> pd.DataFrame:
    """
    Walk forward ourselves rather than using run_backtest, because we need the
    projection's COMPONENT columns (exp_goals, exp_minutes, exp_bonus) which the
    backtest frame discards. Same no-look-ahead construction.
    """
    conn = sqlite3.connect(DB)
    frames = []
    gm = None
    for gw in range(8, 39):
        ko = pd.read_sql("SELECT MIN(kickoff_time) k FROM fixtures WHERE season=? AND gw=?",
                         conn, params=(SEASON, gw)).iloc[0]["k"]
        if ko is None:
            continue
        if gm is None or (gw - 8) % 2 == 0:
            gm = backtest._goal_model_as_of(conn, SEASON, HIST, ko)
        proj = models.project_gameweek(
            conn, SEASON, gw,
            gm,
            backtest._minutes_as_of(conn, SEASON, gw),
            backtest._defcon_as_of(conn, SEASON, HIST, gw),
            backtest._rates_as_of(conn, SEASON, HIST, gw))
        frames.append(proj)
    df = pd.concat(frames, ignore_index=True)

    truth = pd.read_sql(
        """SELECT element, gw, minutes, expected_goals xg, expected_assists xa,
                  clean_sheets cs, defensive_contribution dc, goals_scored,
                  assists, bonus, total_points
           FROM player_gw WHERE season=? AND gw>=8""",
        conn, params=(SEASON,))
    m = df.merge(truth, on=["element", "gw"], how="inner", suffixes=("", "_t"))
    m = m[m["minutes"] >= 1].copy()          # score only players who featured
    m["actual_points"] = m["total_points"]
    return m


def add_oracles(m: pd.DataFrame) -> pd.DataFrame:
    pos = m["position"]
    mins_frac = m["minutes"] / 90.0

    # --- oracle_minutes: our modelled rates, actual minutes -------------------
    # Recover our implied per-90 rates from the model's own projection.
    exp_min = m["exp_minutes"].replace(0, np.nan)
    xg90 = m["exp_goals"] / (exp_min / 90.0)
    xa90 = m["exp_assists"] / (exp_min / 90.0)
    played60 = (m["minutes"] >= 60).astype(float)
    any_min = (m["minutes"] > 0).astype(float)

    m["oracle_minutes"] = (
        any_min + played60
        + pos.map(GOAL_PTS).fillna(4) * (xg90.fillna(0) * mins_frac)
        + 3 * (xa90.fillna(0) * mins_frac)
        + pos.map(CS_PTS).fillna(0) * m["p_clean_sheet"] * played60
        + 2 * m["p_defcon"].fillna(0) * played60
        + m["exp_bonus"].fillna(0)
    )

    # --- oracle_xg: actual underlying performance, finishing still unknown ----
    m["oracle_xg"] = (
        any_min + played60
        + pos.map(GOAL_PTS).fillna(4) * m["xg"].fillna(0)
        + 3 * m["xa"].fillna(0)
        + pos.map(CS_PTS).fillna(0) * m["p_clean_sheet"] * played60
        + 2 * m["p_defcon"].fillna(0) * played60
        + m["exp_bonus"].fillna(0)
    )

    # --- oracle_full: also knows clean sheet and DefCon outcomes --------------
    # Note bonus stays MODELLED, not actual — so oracle_full is not a complete
    # oracle, and the 1 - 0.77 residual includes bonus as well as finishing.
    thr = pos.map(models.DEFCON_THRESHOLD)
    dc_hit = (m["dc"].fillna(0) >= thr).astype(float)
    m["oracle_full"] = (
        any_min + played60
        + pos.map(GOAL_PTS).fillna(4) * m["xg"].fillna(0)
        + 3 * m["xa"].fillna(0)
        + pos.map(CS_PTS).fillna(0) * m["cs"].fillna(0)
        + 2 * dc_hit
        + m["exp_bonus"].fillna(0)
    )
    return m


def score(m: pd.DataFrame, cols: dict) -> pd.DataFrame:
    rows = []
    for label, col in cols.items():
        sp, top = [], []
        for _, g in m.groupby("gw"):
            g2 = g[g[col].notna()]
            if len(g2) > 10 and g2[col].nunique() > 1:
                sp.append(spearmanr(g2[col], g2["actual_points"]).statistic)
                if len(g2) >= 20:
                    top.append(g2.nlargest(20, col)["actual_points"].mean())
        rows.append({
            "predictor": label,
            "spearman": round(float(np.mean(sp)), 4),
            "top20_actual": round(float(np.mean(top)), 3),
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    m = add_oracles(build())
    field = m["actual_points"].mean()

    res = score(m, {
        "our model": "exp_points",
        "+ perfect minutes": "oracle_minutes",
        "+ perfect xG/xA": "oracle_xg",
        "+ perfect CS/DefCon": "oracle_full",
        "perfect foresight": "actual_points",
    })
    res["top20_vs_field"] = (res["top20_actual"] - field).round(3)
    print(f"n = {len(m)} player-gameweeks, field mean = {field:.3f} pts\n")
    print(res.to_string(index=False))

    ours = res.loc[res.predictor == "our model", "spearman"].iloc[0]
    ceil = res.loc[res.predictor == "+ perfect CS/DefCon", "spearman"].iloc[0]
    print(f"\naddressable headroom (model -> full oracle): {ceil - ours:.4f}")
    print(f"irreducible noise (full oracle -> 1.0):      {1 - ceil:.4f}")
    print(f"share of ACHIEVABLE signal captured:         {ours / ceil:.1%}")

    # Marginal value of fixing each component, in Spearman points.
    steps = res.set_index("predictor")["spearman"]
    print("\nmarginal gain from perfecting each component:")
    print(f"  minutes            {steps['+ perfect minutes'] - steps['our model']:+.4f}")
    print(f"  xG/xA rates        {steps['+ perfect xG/xA'] - steps['+ perfect minutes']:+.4f}")
    print(f"  clean sheet+DefCon {steps['+ perfect CS/DefCon'] - steps['+ perfect xG/xA']:+.4f}")
