#!/usr/bin/env python3
"""
Out-of-sample test: should a player's attacking returns depend on the fixture?

WRITTEN BEFORE THE FIX, so the fix cannot be tuned to it.

THE DEFECT
----------
models.py's docstring says returns are "goal/assist rates from xG90/xA90,
scaled by opponent strength". The code is

    exp_goals   = xg90 * mins_frac
    exp_assists = xa90 * mins_frac

— no opponent, no venue. Dixon-Coles feeds clean sheets and goals conceded
only, so every attacker's projection is fixture-blind: Calvert-Lewin projected
4.03, 4.04, 4.03, 4.04, 4.03 across five different opponents in GW7-11
2026-27. Captaincy, transfers and Triple Captain timing all inherit it.

THE CANDIDATE
-------------
Under this parameterisation a team's expected goals are
exp(att_team + dfn_opp + home_adv*home), so the fixture's effect on a team's
attack, relative to its average fixture, is

    m = exp(dfn_opp + home_adv*home) / mean_o[ exp(dfn_o) ] * 2 / (1 + exp(home_adv))

with the mean over the current season's other clubs. Candidate:

    exp_goals = xg90 * mins_frac * m**alpha        (same for assists)

alpha = 0 is today's model; alpha = 1 is the full theoretical adjustment.
Below 1 would mean the team-level signal should be damped for individuals.

THE SCORE — isolating the attacking component
---------------------------------------------
Conditioned on the minutes each player ACTUALLY played, so the minutes model's
error (the largest in the system, LIMITATIONS s2) cannot leak into a test of
fixture sensitivity. Poisson log-likelihood of actual goals and actual assists
given xg90 * minutes/90 * m**alpha, summed over every player-gameweek with
minutes > 0. Goal model and rates are rebuilt as of each gameweek's first
kickoff — strictly no look-ahead, using backtest._rates_as_of.

TWO-STAGE
---------
Stage 1 picks alpha on 2025-26 GW8-38. CHOSEN_ALPHA is then frozen in this
file BEFORE stage 2 runs on 2026-27 GW1-5 (`--stage2`), which reports the gain
with a bootstrap over FIXTURES — teammates' goals in one match are not
independent, so resampling player-rows would overstate the precision.
"""

from __future__ import annotations

import sqlite3
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import poisson

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fplbrain import backtest, models  # noqa: E402

DB = ROOT / "data" / "fpl.db"
ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5]
STAGE1 = ("2025-26", ["2024-25"], range(8, 39))
STAGE2 = ("2026-27", ["2024-25", "2025-26"], range(1, 6))
N_BOOT = 4000


def _goal_model(conn, season, hist, cutoff) -> models.DixonColes:
    """As backtest._goal_model_as_of, plus `teams=` so a promoted club with no
    matches sits at league average, exactly as production fit_goal_model does."""
    frames = []
    for s in hist + [season]:
        frames.append(pd.read_sql(
            """SELECT f.kickoff_time, f.home_score, f.away_score,
                      th.short_name AS home_team, ta.short_name AS away_team
               FROM fixtures f
               JOIN teams th ON th.season=f.season AND th.team_id=f.home_team
               JOIN teams ta ON ta.season=f.season AND ta.team_id=f.away_team
               WHERE f.season=? AND f.finished=1 AND f.kickoff_time < ?""",
            conn, params=(s, cutoff)))
    df = pd.concat(frames, ignore_index=True)
    df["kickoff"] = pd.to_datetime(df["kickoff_time"], errors="coerce", utc=True)
    roster = pd.read_sql("SELECT short_name FROM teams WHERE season=?", conn,
                         params=(season,))["short_name"].tolist()
    return models.DixonColes().fit(df.dropna(subset=["kickoff"]), teams=roster)


def fixture_multiplier(gm: models.DixonColes, opp: str, home: bool,
                       roster: list[str], team: str) -> float:
    """Team's expected goals in this fixture relative to its average fixture."""
    others = [o for o in roster if o != team]
    mean_def = np.mean([np.exp(gm.defence[o]) for o in others])
    venue_mean = (1.0 + np.exp(gm.home_adv)) / 2.0
    this = np.exp(gm.defence[opp] + (gm.home_adv if home else 0.0))
    return float(this / (mean_def * venue_mean))


def gameweek_rows(conn, season, hist, gw) -> pd.DataFrame:
    """One row per player-gameweek that played, with as-of rates and m."""
    fx = pd.read_sql(
        """SELECT f.fixture_id, f.kickoff_time, th.short_name home_name, ta.short_name away_name
           FROM fixtures f
           JOIN teams th ON th.season=f.season AND th.team_id=f.home_team
           JOIN teams ta ON ta.season=f.season AND ta.team_id=f.away_team
           WHERE f.season=? AND f.gw=? AND f.finished=1""", conn, params=(season, gw))
    if fx.empty:
        return pd.DataFrame()
    gm = _goal_model(conn, season, hist, fx.kickoff_time.min())
    roster = pd.read_sql("SELECT short_name FROM teams WHERE season=?", conn,
                         params=(season,))["short_name"].tolist()
    rates = backtest._rates_as_of(conn, season, hist, gw)
    # At GW1 there are no current-season rows, the concat includes an empty
    # frame, and pandas types its columns object: values right, dtype wrong,
    # and scipy's logpmf refuses it. backtest never calls this below GW8.
    rates[["xg90", "xa90"]] = rates[["xg90", "xa90"]].astype(float)
    pg = pd.read_sql(
        """SELECT element, fixture_id, was_home, minutes, goals_scored g, assists ast
           FROM player_gw WHERE season=? AND gw=? AND minutes > 0""",
        conn, params=(season, gw))
    d = pg.merge(fx, on="fixture_id").merge(rates[["element", "xg90", "xa90"]], on="element")
    d["team"] = np.where(d.was_home == 1, d.home_name, d.away_name)
    d["opp"] = np.where(d.was_home == 1, d.away_name, d.home_name)
    d["m"] = [fixture_multiplier(gm, o, bool(hm), roster, t)
              for o, hm, t in zip(d.opp, d.was_home, d.team)]
    d["base_g"] = d.xg90 * d.minutes / 90.0
    d["base_a"] = d.xa90 * d.minutes / 90.0
    d["gw"] = gw
    return d


def loglik(d: pd.DataFrame, alpha: float) -> pd.Series:
    """Per-row Poisson log-likelihood of goals + assists."""
    mult = d.m ** alpha
    lg = np.clip(d.base_g * mult, 1e-6, None)
    la = np.clip(d.base_a * mult, 1e-6, None)
    return pd.Series(poisson.logpmf(d.g, lg) + poisson.logpmf(d.ast, la), index=d.index)


def collect(conn, season, hist, gws) -> pd.DataFrame:
    frames = []
    for gw in gws:
        d = gameweek_rows(conn, season, hist, gw)
        if len(d):
            frames.append(d)
            print(f"  {season} GW{gw}: {len(d)} player-games", flush=True)
    return pd.concat(frames, ignore_index=True)


def stage1(conn) -> None:
    season, hist, gws = STAGE1
    print(f"STAGE 1 — tune alpha on {season} GW{gws.start}-{gws.stop - 1}")
    d = collect(conn, season, hist, gws)
    base = loglik(d, 0.0).sum()
    print(f"\n{len(d)} player-games, {d.fixture_id.nunique()} fixtures, "
          f"m ranges {d.m.min():.2f}-{d.m.max():.2f} (sd {d.m.std():.3f})")
    print(f"{'alpha':>6} {'LL':>12} {'vs alpha=0':>11} {'weeks better':>13}")
    for a in ALPHAS:
        ll = loglik(d, a)
        wk = (ll.groupby(d.gw).sum() - loglik(d, 0.0).groupby(d.gw).sum())
        print(f"{a:>6.2f} {ll.sum():>12.2f} {ll.sum() - base:>+11.2f} "
              f"{int((wk > 0).sum()):>6}/{len(wk)}")


# Frozen after stage 1, BEFORE stage 2 was run. Do not change it to fit stage 2.
# Stage 1 (2025-26 GW8-38, 9348 player-games, 310 fixtures), LL gain vs alpha=0:
#   0.25 +8.24 (22/31 weeks)   0.50 +12.79 (21/31)   0.75 +13.71 (21/31)
#   1.00 +11.05 (18/31)        1.25 +4.83 (15/31)    1.50 -4.96 (13/31)
# Peak at 0.75; flat from 0.5 to 1.0. Damping below the theoretical 1.0 is what
# estimation error in as-of team ratings predicts: shrink a noisy multiplier.
CHOSEN_ALPHA = 0.75


def stage2(conn, alpha) -> None:
    season, hist, gws = STAGE2
    print(f"STAGE 2 — confirm alpha={alpha} (frozen) on {season} GW{gws.start}-{gws.stop - 1}")
    d = collect(conn, season, hist, gws)
    gain = loglik(d, alpha) - loglik(d, 0.0)
    by_fix = gain.groupby(d.fixture_id).sum()
    rng = np.random.default_rng(11)
    boot = np.array([rng.choice(by_fix.values, len(by_fix)).sum() for _ in range(N_BOOT)])
    wk = gain.groupby(d.gw).sum()
    print(f"\n{len(d)} player-games over {len(by_fix)} fixtures")
    print(f"  LL gain vs today's fixture-blind model: {gain.sum():+.2f}")
    print(f"  95% fixture-bootstrap interval: [{np.percentile(boot, 2.5):+.2f}, "
          f"{np.percentile(boot, 97.5):+.2f}]   P(gain>0) = {np.mean(boot > 0):.3f}")
    print(f"  weeks better: {int((wk > 0).sum())}/{len(wk)}   per week: "
          + "  ".join(f"GW{g}:{v:+.1f}" for g, v in wk.items()))


if __name__ == "__main__":
    conn = sqlite3.connect(DB)
    if "--stage2" in sys.argv:
        if CHOSEN_ALPHA is None:
            sys.exit("Run stage 1 and freeze CHOSEN_ALPHA first.")
        stage2(conn, CHOSEN_ALPHA)
    else:
        stage1(conn)
