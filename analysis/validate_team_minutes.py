#!/usr/bin/env python3
"""
MINUTES — what was tested in October 2026, what was not shipped, and the one
check to re-run weekly.

Minutes is the largest error source (LIMITATIONS s2). Four candidate fixes were
measured on a replay of the FIXED minutes model (k=0.75, availability flags
excluded because historical flags do not exist), 2025-26 GW2-38 and 2026-27
GW2-5. Three are nulls, recorded so nobody re-runs them:

  1. CLUB MOVERS. check_mover_minutes.py reported movers over-predicted by
     -0.119 vs stayers, CI excluding zero — but it pools four runs of which
     three are the collapsed pre-fix model. Under the fixed model, among
     players predicted to start (p >= 0.5): 2025-26 GW7+ movers -0.060 vs
     stayers -0.073 (movers BETTER); 2026-27 GW2-6 -0.026 vs -0.035. No
     effect. The earlier result was the binary-p_start bug.
  2. SHORT REST. Days since the team's previous league match, 2025-26:
     <=3 days -0.083, 4-5 days -0.072, 6+ days -0.071. One point. No effect.
  3. RECALIBRATION. 2025-26 calibration slope 0.77 (top bin predicts 0.957,
     delivers 0.876) — real over-confidence — but the Murphy reliability term
     caps any recalibration at 1.7% of Brier, and 2026-27's slope is 0.91-0.98,
     so a fix tuned on 2025-26 would over-correct this season. The DefCon
     recalibration failed the same way (LIMITATIONS s3).

  4. TEAM TOTALS — the one with a signal. Exactly ~10.3 players per team play
     60+ minutes (10.29 in 2024-25, 10.28 in 2025-26; goalkeepers 0.99/1.00),
     but nothing makes a team's p_start sum to that, so when a starter is out
     his teammates' probabilities do not rise (the limitation recorded in
     September: Foden out, Cherki unchanged). Candidate: scale each team's
     goalkeepers to the previous season's GK total and outfielders to the rest.
     Parameter-free; targets from the PREVIOUS season only.

       replay 2025-26   Brier +0.96%   team-GW bootstrap [+0.00066, +0.00127]
       replay 2026-27   Brier +0.29%   [-0.00076, +0.00140]  P=0.71
       production GW5   Brier +1.76%   team bootstrap [-0.0010, +0.0043]  P=0.85

     NOT SHIPPED, for a reason the replay cannot fix: it validates the wrong
     regime. Without flags the replay's team sums are 10.8-11.0, so
     normalising scales DOWN. Production applies injury flags first and its
     sums average 9.50 (8.43 COV to 10.79 IPS), so normalising would scale UP
     by as much as 22%. Only production runs test that, and one gameweek is
     not enough. It also moves p_start without exp_minutes, p_60 and
     p_any_minutes, which would have to move together.

DECISION RULE, fixed before more data arrives: re-run this weekly (it scores
every pre-deadline production run in the projections table). Ship when the
pooled production-regime gain's 95% team-gameweek interval excludes zero, or
after 6 production gameweeks with P(gain > 0) >= 0.9 — implemented in
minutes_model so all four minutes quantities scale together. Drop it if the
pooled gain turns negative.

BLOCKED: rotation after European and cup midweeks. Needs dated fixtures for
UCL/UEL/UECL, the EFL Cup and FA Cup, for the CURRENT season (projections need
upcoming dates). Checked 11 Oct 2026: openfootball has the Champions League
for 2024-25/2025-26 only (no 2026-27, no Europa League, no League Cup at the
paths tried); ClubElo's Fixtures API returns "Fixtures API deactivated". A
keyed source (football-data.org, API-Football) would unblock it.

    python analysis/validate_team_minutes.py
"""
from __future__ import annotations

import sqlite3
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fplbrain import scorecard  # noqa: E402

DB = ROOT / "data" / "fpl.db"
SEASON, PREV = "2026-27", "2025-26"


def targets(conn, season: str) -> tuple[float, float]:
    """Mean 60+ minute players per team per MATCH, and of those goalkeepers.
    Grouped by fixture, not gameweek: a double gameweek's two matches in one
    bucket inflated the target from 10.29 to 10.42."""
    t = pd.read_sql(
        """SELECT g.fixture_id, p.position,
                  CASE WHEN g.was_home=1 THEN f.home_team ELSE f.away_team END team,
                  CASE WHEN g.minutes >= 60 THEN 1 ELSE 0 END st
           FROM player_gw g JOIN fixtures f ON f.season=g.season AND f.fixture_id=g.fixture_id
           JOIN players p ON p.season=g.season AND p.element=g.element WHERE g.season=?""",
        conn, params=(season,))
    per = t.groupby(["fixture_id", "team"]).agg(all=("st", "sum"))
    gk = t[t.position == "GK"].groupby(["fixture_id", "team"]).st.sum()
    return float(per["all"].mean()), float(gk.mean())


def normalise(d: pd.DataFrame, t_all: float, t_gk: float) -> pd.Series:
    gk = d.position == "GK"
    s_gk = d[gk].groupby("team").p.transform("sum")
    s_out = d[~gk].groupby("team").p.transform("sum")
    return pd.concat([(d.p[gk] * t_gk / s_gk).clip(0, 1),
                      (d.p[~gk] * (t_all - t_gk) / s_out).clip(0, 1)]).reindex(d.index)


def main() -> None:
    conn = sqlite3.connect(DB)
    t_all, t_gk = targets(conn, PREV)
    runs = scorecard.local_runs(conn, SEASON)
    dl = scorecard.deadlines(conn, SEASON)
    frames = []
    for gw in scorecard.finished_gws(conn, SEASON):
        ok = runs[(runs.created < dl[gw]) & (runs.horizon_start <= gw) & (runs.horizon_end >= gw)]
        ok = ok[ok.created >= scorecard.FIXED_MINUTES]       # production regime = fixed model
        if ok.empty:
            continue
        run = ok.sort_values("created").iloc[-1]
        d = pd.read_sql(
            """SELECT pr.element, pr.p_start p, pl.position, t.short_name team,
                      COALESCE(MAX(CASE WHEN a.minutes >= 60 THEN 1 ELSE 0 END), 0) st
               FROM projections pr JOIN players pl ON pl.season=pr.season AND pl.element=pr.element
               JOIN teams t ON t.season=pl.season AND t.team_id=pl.team_id
               LEFT JOIN player_gw a ON a.season=pr.season AND a.element=pr.element AND a.gw=?
               WHERE pr.run_id=? AND pr.gw=? GROUP BY pr.element""",
            conn, params=(gw, run.run_id, gw))
        d["norm"] = normalise(d, t_all, t_gk)
        d["gw"] = gw
        frames.append(d)
        b0, b1 = ((d.p - d.st) ** 2).mean(), ((d.norm - d.st) ** 2).mean()
        print(f"GW{gw} {run.run_id}: team sums {d.groupby('team').p.sum().mean():.2f} "
              f"(target {t_all:.2f}); Brier {b0:.4f} -> {b1:.4f} ({100 * (b0 - b1) / b0:+.2f}%)")
    if not frames:
        print("no production-regime pre-deadline run has been scored yet")
        return
    d = pd.concat(frames, ignore_index=True)
    gain = ((d.p - d.st) ** 2 - (d.norm - d.st) ** 2).groupby([d.gw, d.team]).sum().values
    rng = np.random.default_rng(4)
    bs = np.array([rng.choice(gain, len(gain)).sum() for _ in range(4000)]) / len(d)
    n_gw = d.gw.nunique()
    lo, hi, p = np.percentile(bs, 2.5), np.percentile(bs, 97.5), np.mean(bs > 0)
    print(f"\npooled over {n_gw} production gameweek(s): Brier gain {gain.sum() / len(d):+.5f}  "
          f"95% [{lo:+.5f}, {hi:+.5f}]  P(gain>0)={p:.2f}")
    verdict = ("SHIP (interval excludes zero)" if lo > 0 else
               "SHIP (6+ gameweeks at P>=0.9)" if n_gw >= 6 and p >= 0.9 else
               "DROP (pooled gain negative)" if gain.sum() < 0 else
               f"WAIT — {n_gw} of 6 gameweeks")
    print(f"decision rule: {verdict}")


if __name__ == "__main__":
    main()
