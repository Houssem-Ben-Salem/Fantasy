"""
The prospective scorecard must be impossible to flatter.

It is the one test here that no choice of ours could have been tuned to — but
only if it scores what was persisted BEFORE each deadline. A run made after a
deadline knows the team news, sometimes the line-ups; scoring it would quietly
turn the record into hindsight. Likewise the naive baseline may only use
gameweeks before the one being scored.

Hermetic: an in-memory database from schema.sql with two gameweeks, and a
"perfect" run made an hour after GW2's deadline that must be ignored.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone

from fplbrain import ingest, scorecard

S = "2026-27"
GW1 = datetime(2026, 8, 22, 14, 0, tzinfo=timezone.utc)
GW2 = datetime(2026, 8, 29, 14, 0, tzinfo=timezone.utc)
POINTS = {1: {1: 2, 2: 6, 3: 1, 4: 9}, 2: {1: 8, 2: 1, 3: 7, 4: 2}}


def _iso(t: datetime) -> str:
    return t.strftime("%Y-%m-%dT%H:%M:%SZ")


def _db(path: str = ":memory:") -> sqlite3.Connection:
    c = sqlite3.connect(path)
    ingest.init_schema(c)
    for gw, ko in ((1, GW1), (2, GW2)):
        c.execute("INSERT INTO fixtures (season, fixture_id, gw, kickoff_time, home_team, "
                  "away_team, finished) VALUES (?,?,?,?,1,2,1)", (S, gw, gw, _iso(ko)))
        for el, pts in POINTS[gw].items():
            c.execute("INSERT INTO player_gw (season, element, code, gw, fixture_id, was_home, "
                      "total_points, starts, minutes) VALUES (?,?,?,?,?,?,?,1,90)",
                      (S, el, 100 + el, gw, gw, int(el <= 2), pts))

    def run(run_id, created, xp):
        c.execute("INSERT INTO runs (run_id, created_at, season, horizon_start, horizon_end, "
                  "notes, provenance) VALUES (?,?,?,2,3,'projection','{}')",
                  (run_id, created.isoformat(), S))
        for el, v in xp.items():
            c.execute("INSERT INTO projections (season, element, gw, run_id, p_start, exp_points) "
                      "VALUES (?,?,2,?,0.9,?)", (S, el, run_id, v))

    run("proj-before", GW2 - timedelta(hours=3), {1: 1.0, 2: 2.0, 3: 3.0, 4: 4.0})
    run("proj-after", GW2 + timedelta(hours=1), {el: float(p) for el, p in POINTS[2].items()})
    c.commit()
    return c


def test_only_runs_made_before_the_deadline_are_scored():
    per_gw, _ = scorecard.score(_db(), S, include_ci=False)
    row = per_gw[per_gw.gw == 2].iloc[0]
    assert row.run == "proj-before"
    assert row.spearman_all < 1.0          # the hindsight run would score a perfect 1.0


def test_the_naive_baseline_only_sees_earlier_gameweeks():
    act = scorecard.actuals_and_naive(_db(), S, 2).set_index("element")
    assert act.naive_xp.to_dict() == {el: float(p) for el, p in POINTS[1].items()}


def test_the_mcp_tool_reports_the_pre_deadline_run(tmp_path, monkeypatch):
    from fplbrain import mcp_server
    path = tmp_path / "fpl.db"
    _db(str(path)).close()
    monkeypatch.setattr(mcp_server, "DB", str(path))
    monkeypatch.setattr(mcp_server, "SEASON", S)
    out = mcp_server.model_scorecard(include_ci=False)
    assert "proj-before" in out and "proj-after" not in out
