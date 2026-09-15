#!/usr/bin/env python3
"""
GUARD — the minutes model collapses to binary p_start in-season.

THE DEFECT
----------
models.minutes_model has two branches, chosen by whether ANY current-season rows
exist in player_gw:

  pre-season   start_rate averaged over prior seasons, joined on `code`
  in-season    start_rate = mean(minutes >= 60) over current-season rows in the
               lookback window ONLY, with no shrinkage toward the prior

The switch is all-or-nothing. With a single current-season gameweek loaded,
start_rate for every player is exactly 0.0 or 1.0 — one match, read as certainty
— and all of last season's evidence is discarded. It is the same class as the
Dixon-Coles defect in check_team_rating_bounds.py: an unshrunk small-sample
estimate, in the component LIMITATIONS.md §2 measures as 47% of the addressable
gap.

WHY IT PERSISTED
----------------
player_gw is filled only from the community mirror, which had published GW1 and
nothing after it. So once GW1 landed, every refresh rebuilt p_start from that
one stale match while GW2, GW3 and GW4 were played. The live API has those
gameweeks (sources.live_gameweek) but nothing writes them to player_gw.

MEASURED, 2026-09-15
--------------------
p_start across projection runs:

    18 Aug  pre-season branch    138 distinct values   24.9% exactly 0 or 1
    28 Aug  pre-season branch    134 distinct values   29.1%
     3 Sep  in-season, GW1 only    4 distinct values   92.8%
    15 Sep  in-season, GW1 only    5 distinct values   91.7%

Against live-API minutes for GW2-4:

    53 players at p_start 0.0 who started 60+ in >= 2 of GW2-4
       e.g. Cherki (MCI, 28% owned), Konsa, Van de Ven 270/270 minutes
    32 players at p_start 1.0 who started 60+ in <= 1 of GW2-4
       e.g. Senesi and O'Nien, ZERO minutes since GW1

WHAT THIS INVALIDATES
---------------------
Every projection since 3 Sep: top_players, build_squad, plan_transfers and
chip_advice all consume p_start. It also silently explains a transfer verdict —
the model scored Semenyo->Cherki as 3.79 xP vs 1.13 xP for GW4 because Cherki
played 27 minutes in GW1. That verdict is an artefact, not an estimate.

FIXED 2026-09-15 in 87eac4d (Beta prior MINUTES_PRIOR_K=0.75, validated out-of-sample;
and ingest.fill_gameweeks_from_live). This guard is RETAINED: it is what caught the
collapse, and the in-season branch switch is exactly the kind of 'simplification'
that gets reintroduced. Originally it said: NOT FIXED HERE, per the discipline of
not patching mid-gameweek. Two separate
changes are needed and both need out-of-sample validation:
  1. blend current-season start rate with the prior-season rate, weighted by
     current-season appearances (a Beta prior, as defcon_model already does),
     rather than switching branches on the first row;
  2. fill player_gw from the live API when the mirror lags.

EXIT CODES
  0  p_start looks like a probability distribution and player_gw is current
  1  collapse or staleness detected -> do not trust any minutes-driven output
  2  could not check
"""

from __future__ import annotations

import sqlite3
import sys
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DB = str(ROOT / "data" / "fpl.db")
SEASON = "2026-27"
MAX_BINARY_SHARE = 0.50     # pre-season runs sat at ~25-29%
MIN_DISTINCT = 20           # pre-season runs had ~135


def main() -> int:
    try:
        conn = sqlite3.connect(DB)
        run = pd.read_sql(
            """SELECT r.run_id FROM runs r WHERE r.notes='projection'
               AND EXISTS (SELECT 1 FROM projections p WHERE p.run_id=r.run_id)
               ORDER BY r.created_at DESC LIMIT 1""", conn).iloc[0]["run_id"]
        p = pd.read_sql(
            """SELECT p_start FROM projections WHERE run_id=? AND gw=
               (SELECT MIN(gw) FROM projections WHERE run_id=?)""",
            conn, params=(run, run))["p_start"]
        loaded = pd.read_sql(
            "SELECT DISTINCT gw FROM player_gw WHERE season=? ORDER BY gw",
            conn, params=(SEASON,))["gw"].tolist()
        finished = pd.read_sql(
            "SELECT DISTINCT gw FROM fixtures WHERE season=? AND finished=1 ORDER BY gw",
            conn, params=(SEASON,))["gw"].tolist()
    except Exception as e:  # noqa: BLE001
        print(f"!! could not check: {type(e).__name__}: {e}", file=sys.stderr)
        return 2

    binary = float(((p == 0) | (p == 1)).mean())
    distinct = int(p.round(4).nunique())
    missing = sorted(set(finished) - set(loaded))

    print(f"latest projection run: {run}")
    print(f"  p_start: {distinct} distinct values, {binary:.1%} exactly 0 or 1")
    print(f"  finished gameweeks: {finished}")
    print(f"  loaded into player_gw: {loaded}")

    problems = []
    if loaded and (binary > MAX_BINARY_SHARE or distinct < MIN_DISTINCT):
        problems.append(
            f"p_start has COLLAPSED to near-binary ({distinct} distinct, {binary:.0%} "
            f"exactly 0/1). The in-season branch is reading "
            f"{len(loaded)} gameweek(s) as certainty with no prior.")
    if missing:
        problems.append(
            f"player_gw is STALE: gameweeks {missing} are finished but not loaded, "
            f"so minutes are being inferred from GW{max(loaded) if loaded else '-'} "
            f"or earlier.")

    if not problems:
        print("\np_start looks like a probability distribution and player_gw is current")
        return 0
    print()
    for msg in problems:
        print(f"!! {msg}")
    print("   Do not trust top_players, build_squad, plan_transfers or chip_advice:")
    print("   every one of them consumes p_start.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
