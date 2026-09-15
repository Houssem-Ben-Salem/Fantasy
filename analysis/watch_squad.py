#!/usr/bin/env python3
"""
Squad availability monitor — the sensor for the "run the deadline session early"
trigger.

WHY THIS EXISTS
---------------
The plan was: run the GW1 deadline session on Friday morning, earlier only if an
availability flag lands on one of the 15. That trigger had no sensor. `status`
only changes in the database when someone runs an ingest, and between Tuesday and
Friday nobody would — so the trigger could never fire and the calendar would have
decided after all. A trigger with nothing watching it is a pointer to something
that does not exist.

TWO DESIGN RULES, BOTH ABOUT THE SAME FAILURE
---------------------------------------------
1. SILENCE MUST BE EARNED. A monitor that prints nothing when clean is
   indistinguishable from a monitor that crashed, and both read as "no flags".
   So every run appends a timestamped line to the log whether or not anything is
   wrong, and infrastructure failures are LOUD (stderr, non-zero exit) rather
   than quiet. "No alert" then means "checked and clean", not "did not run".

2. CARRY A CONTROL. The league-wide flagged count sits beside your 15. If it
   collapses toward zero the FEED has broken, not the injury situation resolved.
   Measured baseline on 2026-08-18: 103 players with news, 117 with
   chance_of_playing populated, out of 590.

A NOTE ON READING A CLEAN SQUAD
-------------------------------
103 flagged league-wide with none in your 15 is not luck, and should not be read
as evidence the squad is robust. `players.status` feeds the availability
multiplier in models.minutes_model, which drives p_start toward zero for flagged
players, which makes the optimiser avoid them. The squad is SELECTED ON
AVAILABILITY and will look cleaner than a random 15 for structural reasons. That
is correct behaviour. It just is not information.

WHAT THIS CHECK CANNOT SEE — READ BEFORE TRUSTING A CLEAN LOG
------------------------------------------------------------
It watches `players.status`, and that field reports INJURIES. It says nothing
about ROTATION. A player who will be benched and a player who will start 90
minutes are identical to this check, right up until the lineups drop.

So the three decisions actually open on this squad are invisible here:

    Walle Egeli   p_start 0.350   <- the flat prior for NO history, not a read
    Mateta        p_start 0.645   <- lone forward; a rotated one is the big hole
    Calvert-Lewin p_start 0.640   <- first sub

A clean log every night through Friday is genuine reassurance about injuries and
tells you NOTHING about any of the three. That is the boundary of what this data
supports, not a defect in the check — and it is exactly why the deadline session
is a SEARCH task (press conferences, predicted line-ups) rather than a query.

Combined with the selection-on-availability note above: a clean result confirms
nothing has changed SINCE SELECTION. It does not say the squad was well chosen.

EXIT CODES  (so cron/alerting can key on them)
  0  checked, squad clean
  1  checked, one or more of the 15 is flagged  -> run the deadline session now
  2  could not check  -> ingest failed, squad did not resolve, or the feed looks
                         broken. Never confuse this with 0.

USAGE
  python analysis/watch_squad.py              # refresh, check, print, log
  python analysis/watch_squad.py --quiet      # for cron: stdout only on trouble
  python analysis/watch_squad.py --no-ingest  # check the DB as-is, no network
  python analysis/watch_squad.py --db /tmp/x.db --no-ingest   # test a failure path
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

DB = str(ROOT / "data" / "fpl.db")
LOG = ROOT / "analysis" / "squad_watch.log"
SEASON = "2026-27"
HIST = ["2024-25", "2025-26"]

# The squad is read from the latest saved squad_state, NOT hardcoded.
#
# It used to be a hardcoded dict of element ids, written in GW1 and never updated.
# Three transfers later it was still watching Enzo, Semenyo and Mateta — none owned
# any more, Mateta injured, so it would have alarmed daily about a player already
# sold — while ignoring Szoboszlai, Wissa and Cherki, who were. Three of fifteen
# wrong. A sensor pointed at the wrong squad is worse than none: it reports "clean"
# about the wrong fifteen. Element ids are still what is compared, never names
# (two players are called Palmer in 2026-27).
def load_squad(conn) -> dict[int, str]:
    import json
    row = pd.read_sql("SELECT gw, squad FROM squad_state ORDER BY gw DESC LIMIT 1", conn)
    if row.empty:
        raise RuntimeError("no saved squad_state — call save_state before relying on this monitor")
    squad = json.loads(row.iloc[0]["squad"])
    return {int(pl["element"]): str(pl.get("name", pl["element"])) for pl in squad}


# If the league-wide flagged count falls below this, suspect the feed rather than
# a miraculously healthy division. Baseline measured 2026-08-18 was 103.
MIN_PLAUSIBLE_LEAGUE_FLAGS = 10


def _log(line: str) -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def _fail(msg: str, stamp: str) -> int:
    """Infrastructure problems are loud. This must never look like 'clean'."""
    line = f"{stamp} · CHECK FAILED · {msg}"
    _log(line)
    print(f"!! {line}", file=sys.stderr)
    return 2


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quiet", action="store_true",
                    help="suppress stdout when clean (for cron)")
    ap.add_argument("--no-ingest", action="store_true",
                    help="check the database as-is, without refetching")
    ap.add_argument("--verbose", action="store_true",
                    help="show ingest's own warnings (suppressed by default so "
                         "cron does not mail an expected pre-season 404 every run)")
    ap.add_argument("--db", default=DB,
                    help="database to check (for exercising the failure paths "
                         "without touching the real one)")
    a = ap.parse_args()

    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    if not a.no_ingest:
        try:
            import logging
            # Routine warnings here are expected — 2026-27 has no merged_gw.csv
            # until the season starts. Silence them so a nightly cron does not
            # mail a 404 every run, while real failures still raise and are
            # reported loudly below.
            if not a.verbose:
                logging.getLogger("fplbrain").setLevel(logging.ERROR)
            from fplbrain import ingest
            ingest.run(a.db, SEASON, HIST)
        except Exception as e:  # noqa: BLE001
            return _fail(f"ingest raised {type(e).__name__}: {e}", stamp)

    if not Path(a.db).exists():
        return _fail(f"no database at {a.db}", stamp)

    try:
        with sqlite3.connect(a.db) as conn:
            SQUAD = load_squad(conn)
            ids = ",".join(str(i) for i in SQUAD)
            squad = pd.read_sql(
                f"""SELECT element, web_name, status, chance_next_round, news
                    FROM players WHERE season=? AND element IN ({ids})""",
                conn, params=(SEASON,))
            league = pd.read_sql(
                "SELECT status, news, chance_next_round FROM players WHERE season=?",
                conn, params=(SEASON,))
    except Exception as e:  # noqa: BLE001
        return _fail(f"query raised {type(e).__name__}: {e}", stamp)

    # A short squad means the watchlist has drifted from the data — element ids
    # are season-scoped, so this fires loudly rather than silently watching 14.
    if len(squad) != len(SQUAD):
        missing = sorted(set(SQUAD) - set(squad.element))
        return _fail(
            f"resolved {len(squad)}/{len(SQUAD)} squad players; missing ids {missing}",
            stamp)

    news = league["news"].fillna("").str.strip()
    league_flagged = int(((league["status"] != "a") | (news != "")).sum())
    if league_flagged < MIN_PLAUSIBLE_LEAGUE_FLAGS:
        return _fail(
            f"only {league_flagged} flagged players league-wide (expected ~100); "
            f"the availability feed is probably broken, not the league healthy",
            stamp)

    squad_news = squad["news"].fillna("").str.strip()
    flagged = squad[(squad["status"] != "a") | (squad_news != "")]
    clean = len(squad) - len(flagged)

    if flagged.empty:
        line = (f"{stamp} · {clean}/{len(squad)} clean · "
                f"{league_flagged} flagged league-wide")
        _log(line)
        if not a.quiet:
            print(line)
        return 0

    detail = "; ".join(
        f"{SQUAD.get(int(r.element), r.web_name)}"
        f"[{r.status}"
        + (f",{int(r.chance_next_round)}%" if pd.notna(r.chance_next_round) else "")
        + "]"
        for r in flagged.itertuples())
    line = (f"{stamp} · {clean}/{len(squad)} clean · FLAGS: {detail} · "
            f"{league_flagged} flagged league-wide")
    _log(line)

    print(f"** {len(flagged)} of your 15 flagged — run the deadline session now",
          file=sys.stderr)
    print(line)
    for r in flagged.itertuples():
        note = (r.news or "").strip() or "(no note)"
        print(f"   {SQUAD.get(int(r.element), r.web_name):<16} "
              f"status={r.status}  chance={r.chance_next_round}  {note}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
