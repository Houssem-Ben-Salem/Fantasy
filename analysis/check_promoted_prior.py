"""
GUARD + MEASUREMENT — the ridge shrinks toward the wrong TARGET for promoted clubs.

`GOAL_RIDGE_LAMBDA` fixed the ±3 pinning by shrinking team ratings toward zero,
i.e. toward LEAGUE AVERAGE. That is the right target for an established club with
a thin recent record. It is the wrong target for a PROMOTED club, whose honest
prior is "worse than average", not "average". The ridge therefore corrected the
variance of the estimate while leaving its centre optimistic.

This is the residual of the bug class, not a new one: §0's fix stopped thin clubs
diverging to the bound; it did not stop four matches from moving a promoted club
to a top-of-the-table rating.

MEASURED, on the only promoted cohort the database can test (2025-26: BUR, LEE,
SUN). Rating after 4 matches vs rating on the full season:

    BUR  dfn  0.207 -> 0.463   (+0.256)
    LEE  dfn  0.131 -> 0.170   (+0.039)
    SUN  dfn -0.162 -> 0.065   (+0.228)
    promoted mean move  +0.174   (positive = worse once more data arrived)
    other 17 mean move  -0.003

n = 3, all three in the same direction. A sign test on three observations gives
p = 0.25, so THIS DOES NOT ESTABLISH THE EFFECT — it establishes the direction
and the mechanism, and it is stated that way deliberately. SUN is the clearest
case and the exact shape of the 2026-27 HUL instance: better than average after
four matches, below average by season's end.

WHAT IT DOES NOT EXPLAIN. In GW5 2026-27 the model put NEW at home to HUL at
1.07 xG against Pinnacle's 2.02. Applying the measured +0.174 to HUL's defence
lifts it to 1.27, and the league-wide level shortfall (§ below) adds 0.17 more,
reaching 1.45 — still 0.57 short, which is 60% of the gap. The promoted-club
prior is real and is NOT the explanation for that fixture. Resist the pull to
close a gap with the nearest available bias.

NOT FIXED, deliberately. A promoted-club prior means shrinking toward a nonzero
target, which changes every rating in the table and needs the same two-stage
out-of-sample validation as `validate_goal_ridge.py` — tune on 2025-26, freeze,
confirm on 2026-27. That could not be done between finding it and a deadline.

Exit 0 = direction unchanged. Exit 1 = promoted clubs now flatter the OTHER way,
which would mean this note is stale and should be rewritten.
"""

from __future__ import annotations

import sqlite3
import sys
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fplbrain.models import DixonColes  # noqa: E402

DB = "data/fpl.db"
PRIOR, CURRENT = "2024-25", "2025-26"
N_MATCHES = 40  # 4 gameweeks


def _matches(conn: sqlite3.Connection, seasons: list[str]) -> pd.DataFrame:
    frames = []
    for s in seasons:
        frames.append(pd.read_sql(
            """SELECT f.kickoff_time, f.home_score, f.away_score,
                      th.short_name AS home_team, ta.short_name AS away_team
               FROM fixtures f
               JOIN teams th ON th.season=f.season AND th.team_id=f.home_team
               JOIN teams ta ON ta.season=f.season AND ta.team_id=f.away_team
               WHERE f.season=? AND f.finished=1""", conn, params=(s,)))
    d = pd.concat(frames, ignore_index=True)
    d["kickoff"] = pd.to_datetime(d["kickoff_time"], errors="coerce", utc=True)
    return d.dropna(subset=["kickoff"]).sort_values("kickoff")


def main() -> int:
    conn = sqlite3.connect(DB)
    teams = lambda s: set(pd.read_sql(  # noqa: E731
        "SELECT short_name FROM teams WHERE season=?", conn, params=(s,))["short_name"])
    promoted = sorted(teams(CURRENT) - teams(PRIOR))
    d = _matches(conn, [PRIOR, CURRENT])
    season_start = pd.Timestamp(f"{CURRENT[:4]}-08-01", tz="UTC")
    cut = d[d.kickoff >= season_start].kickoff.iloc[N_MATCHES - 1]

    roster = sorted(teams(CURRENT))
    early = DixonColes().fit(d[d.kickoff <= cut], teams=roster)
    full = DixonColes().fit(d, teams=roster)

    r = pd.DataFrame({"after4": pd.Series(early.defence),
                      "full": pd.Series(full.defence)}).loc[roster]
    r["move"] = r["full"] - r["after4"]
    r["promoted"] = r.index.isin(promoted)

    print(f"promoted in {CURRENT}: {promoted}")
    print(r[r.promoted].round(3).to_string())
    pm, om = r[r.promoted].move.mean(), r[~r.promoted].move.mean()
    print(f"\n  promoted mean defence move {pm:+.3f}  (positive = worse with more data)")
    print(f"  others   mean defence move {om:+.3f}")
    print(f"  n promoted = {int(r.promoted.sum())}; direction only, not significance.")

    if pm <= om:
        print("\nFAIL: promoted clubs no longer flatter on a 4-match rating. "
              "The note in this file's docstring is stale — rewrite it.")
        return 1
    print("\nOK: direction unchanged (promoted 4-match ratings still flatter).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
