#!/usr/bin/env python3
"""
Does the minutes model over-predict starts for players who CHANGED CLUB?

THE HYPOTHESIS
--------------
models.minutes_model keys on the stable `code`, so a player's usage carries
across a transfer. If he was nailed on at his old club, the pre-season fallback
(`start_rate` from prior seasons) hands him a high p_start at his new club, where
squad depth may be entirely different. If true, this is systematic — it affects
every transferred player in every squad, not one pick.

WHY THIS FILE EXISTS RATHER THAN AN ANSWER
------------------------------------------
Measured on GW1 alone the effect has the hypothesised sign and a material size,
and is one standard error from zero:

    predicted p_start >= 0.70
      movers   5/8  started = 0.625
      stayers 56/69 started = 0.812
      difference -0.187, SE 0.189  ->  -0.99 SE, 95% CI [-0.557, +0.184]

Eight movers is not a sample. One gameweek is not evidence. LIMITATIONS.md §5
is a catalogue of principled fixes that died on measurement, and "the model
mishandles transfers" would be a fourth if adopted on this.

So the check accumulates. Re-run it after each gameweek; the SE falls as
1/sqrt(k) in gameweeks, so roughly:

    after 1 GW   SE ~0.19   cannot distinguish -0.19 from 0
    after 4 GWs  SE ~0.09   -0.19 would clear 2 SE
    after 8 GWs  SE ~0.07   a -0.10 effect would clear 2 SE

Decide at GW5, not before. If it holds, the fix belongs in the minutes model
(discount prior-club usage by new-club squad depth); if it evaporates, this file
records that it was tested rather than assumed.

METHOD
------
For each completed gameweek, use the projection run that was created BEFORE that
gameweek's first kickoff — the genuine pre-deadline prediction, not a
back-projection. Actuals come from the live API (`event/{gw}/live/`), because the
community mirror has no 2026-27 merged_gw.csv yet.

USAGE
    python analysis/check_mover_minutes.py            # all completed GWs
    python analysis/check_mover_minutes.py --gws 1 2  # specific ones
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fplbrain import sources  # noqa: E402

DB = str(ROOT / "data" / "fpl.db")
SEASON, PREV = "2026-27", "2025-26"
START_MIN = 60
THRESHOLD = 0.70          # only players the model actually expected to start


def _pre_deadline_run(conn, gw: int) -> str | None:
    """The newest projection run created before this gameweek's first kickoff."""
    ko = pd.read_sql(
        "SELECT MIN(kickoff_time) k FROM fixtures WHERE season=? AND gw=?",
        conn, params=(SEASON, gw)).iloc[0]["k"]
    if not ko:
        return None
    r = pd.read_sql(
        """SELECT r.run_id FROM runs r
           WHERE r.notes='projection' AND r.created_at < ?
             AND EXISTS (SELECT 1 FROM projections p
                         WHERE p.run_id=r.run_id AND p.gw=?)
           ORDER BY r.created_at DESC LIMIT 1""",
        conn, params=(ko, gw))
    return None if r.empty else r.iloc[0]["run_id"]


def gameweek_frame(conn, gw: int) -> pd.DataFrame | None:
    run = _pre_deadline_run(conn, gw)
    if run is None:
        return None
    pred = pd.read_sql(
        "SELECT element,p_start FROM projections WHERE run_id=? AND gw=?",
        conn, params=(run, gw))
    cur = pd.read_sql(
        """SELECT p.element,p.code,p.web_name,t.short_name team
           FROM players p JOIN teams t ON t.season=p.season AND t.team_id=p.team_id
           WHERE p.season=?""", conn, params=(SEASON,))
    prev = pd.read_sql(
        """SELECT p.code, t.short_name prev_team
           FROM players p JOIN teams t ON t.season=p.season AND t.team_id=p.team_id
           WHERE p.season=?""", conn, params=(PREV,))
    fetched = sources.live_gameweek(gw)
    if fetched is None:
        print(f"  GW{gw}: live API unavailable — skipped", file=sys.stderr)
        return None
    act = pd.DataFrame([{"element": e["id"], "minutes": e["stats"]["minutes"]}
                        for e in fetched.data["elements"]])
    df = (cur.merge(pred, on="element")
             .merge(act, on="element", how="left")
             .merge(prev, on="code", how="left"))
    df["minutes"] = df["minutes"].fillna(0)
    df["started"] = (df["minutes"] >= START_MIN).astype(int)
    df["moved"] = df["prev_team"].notna() & (df["prev_team"] != df["team"])
    df["no_prev"] = df["prev_team"].isna()
    df["gw"] = gw
    df["run_id"] = run
    return df


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gws", nargs="*", type=int)
    a = ap.parse_args()

    conn = sqlite3.connect(DB)
    gws = a.gws or pd.read_sql(
        "SELECT DISTINCT gw FROM fixtures WHERE season=? AND finished=1 ORDER BY gw",
        conn, params=(SEASON,))["gw"].tolist()
    if not gws:
        gws = [1]

    frames = [f for f in (gameweek_frame(conn, g) for g in gws) if f is not None]
    if not frames:
        print("no gameweeks with both a pre-deadline projection and live actuals",
              file=sys.stderr)
        return 2
    df = pd.concat(frames, ignore_index=True)
    used = sorted(df.gw.unique())
    print(f"gameweeks: {used}   runs: {sorted(df.run_id.unique())}")
    print(f"player-gameweeks: {len(df)}\n")

    print(f"p_start calibration among players predicted to start "
          f"(p_start >= {THRESHOLD})")
    print("(gap < 0 = model OVER-predicted starts)\n")
    hi = df[df.p_start >= THRESHOLD]
    rows = []
    for label, sub in (("moved club", hi[hi.moved]),
                       ("stayed", hi[~hi.moved & ~hi.no_prev]),
                       ("no prior-season row", hi[hi.no_prev])):
        if len(sub) < 3:
            continue
        rows.append({"group": label, "n": len(sub),
                     "mean_pred": round(sub.p_start.mean(), 3),
                     "actual_start_rate": round(sub.started.mean(), 3),
                     "gap": round(sub.started.mean() - sub.p_start.mean(), 3)})
    print(pd.DataFrame(rows).to_string(index=False))

    mv, st = hi[hi.moved], hi[~hi.moved & ~hi.no_prev]
    if len(mv) >= 3 and len(st) >= 3:
        d = mv.started.mean() - st.started.mean()
        se = np.sqrt(mv.started.var(ddof=1) / len(mv) + st.started.var(ddof=1) / len(st))
        print(f"\nmovers - stayers: {d:+.3f}   SE {se:.3f}   {d / se:+.2f} SE"
              f"   95% CI [{d - 1.96 * se:+.3f}, {d + 1.96 * se:+.3f}]")
        verdict = ("SIGNIFICANT — the effect is real; fix belongs in the minutes model"
                   if abs(d) > 1.96 * se else
                   "NOT SIGNIFICANT — do not act on this yet")
        print(f"verdict: {verdict}")
        if abs(d) <= 1.96 * se and abs(d) > 1e-9:
            need = (1.96 * se / abs(d)) ** 2 * len(used)
            print(f"         at this effect size, ~{need:.0f} gameweeks would be "
                  f"needed to resolve it ({len(used)} so far)")
    else:
        print(f"\ntoo few movers ({len(mv)}) to test — accumulate more gameweeks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
