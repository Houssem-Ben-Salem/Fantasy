"""
Tests for validate() distinguishing "absent by design" from "wrong".

BACKGROUND
----------
validate() used to report `null_defcon` alongside bad_minutes / bad_points /
bad_goals under a single "anomalies" key. For 2024-25 that count is 27,605 —
every row — because the DefCon rule did not exist before 2025-26. A correct
observation, filed under a heading that means corruption.

AGENT.md tells the agent to call refresh_data() at the start of every session,
so this sat in the highest-traffic path in the system. An agent that opens each
session reporting a data integrity problem on correct data burns a turn and
teaches the user to distrust the check — which is worse than not having it.

The three-case split also buys something the blanket count could not express:
PARTIAL nulls are now a real anomaly. A stat present for part of a season and
absent for the rest is a truncated or broken load, and the old report averaged
that signal away into the same number as a legitimately absent column.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd

from fplbrain import ingest

DEFCON_COLS = ["tackles", "recoveries", "clearances_blocks_interceptions",
               "defensive_contribution"]


def _db():
    tmp = tempfile.TemporaryDirectory()
    conn = ingest.connect(Path(tmp.name) / "t.db")
    ingest.init_schema(conn)
    return conn, tmp


def _seed_gw(conn, season: str, n: int, defcon_nulls: int) -> None:
    """n gameweek rows, the first `defcon_nulls` of them with NULL defensive stats."""
    rows = []
    for i in range(n):
        null_it = i < defcon_nulls
        rows.append({
            "season": season, "element": i + 1, "code": 900000 + i, "gw": 1,
            "fixture_id": i + 1, "minutes": 90, "total_points": 5,
            "goals_scored": 0, **{c: (None if null_it else 4) for c in DEFCON_COLS},
        })
    pd.DataFrame(rows).to_sql("player_gw", conn, if_exists="append", index=False)
    conn.commit()


def test_fully_absent_column_is_availability_not_anomaly():
    """The 2024-25 case: 100% null because the stat did not exist that season."""
    conn, tmp = _db()
    with tmp:
        _seed_gw(conn, "2024-25", n=50, defcon_nulls=50)
        r = ingest.validate(conn, "2024-25")

        assert r["anomalies_ok"] is True, (
            f"a wholly absent column was reported as an anomaly: {r['anomalies']}")
        assert not any("defensive_contribution" in k for k in r["anomalies"])
        assert "defensive_contribution" in r["data_availability"]
        assert "2024-25" in r["data_availability"]["defensive_contribution"]


def test_partial_nulls_are_a_real_anomaly():
    """Present for some rows and not others means a broken load, not a missing feature."""
    conn, tmp = _db()
    with tmp:
        _seed_gw(conn, "2025-26", n=100, defcon_nulls=40)
        r = ingest.validate(conn, "2025-26")

        assert r["anomalies_ok"] is False, "partial coverage was not flagged"
        key = "partial_null_defensive_contribution"
        assert key in r["anomalies"]
        assert r["anomalies"][key]["nulls"] == 40
        assert r["anomalies"][key]["pct"] == 40.0
        # Partial absence must NOT be excused as expected.
        assert "defensive_contribution" not in r["data_availability"]


def test_fully_present_column_reports_neither():
    conn, tmp = _db()
    with tmp:
        _seed_gw(conn, "2025-26", n=50, defcon_nulls=0)
        r = ingest.validate(conn, "2025-26")

        assert r["anomalies_ok"] is True
        assert r["data_availability"] == {}
        assert not any("defensive_contribution" in k for k in r["anomalies"])


def test_real_anomalies_still_detected():
    """The split must not have loosened the checks that were already there."""
    conn, tmp = _db()
    with tmp:
        pd.DataFrame([{
            "season": "2025-26", "element": 1, "code": 1, "gw": 1, "fixture_id": 1,
            "minutes": 400,           # impossible
            "total_points": 5, "goals_scored": 0,
            **{c: 4 for c in DEFCON_COLS},
        }]).to_sql("player_gw", conn, if_exists="append", index=False)
        conn.commit()
        r = ingest.validate(conn, "2025-26")

        assert r["anomalies"]["bad_minutes"] == 1
        assert r["anomalies_ok"] is False


def test_case_is_derived_from_data_not_a_hardcoded_season():
    """
    A future season introducing a new stat must classify correctly with no code
    change. Same absence pattern, a season name the code has never heard of.
    """
    conn, tmp = _db()
    with tmp:
        _seed_gw(conn, "2031-32", n=30, defcon_nulls=30)
        r = ingest.validate(conn, "2031-32")

        assert r["anomalies_ok"] is True
        assert "2031-32" in r["data_availability"]["defensive_contribution"]
