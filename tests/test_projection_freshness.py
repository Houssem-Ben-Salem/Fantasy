"""
Regression tests for v_latest_projections resolving to the wrong run.

BACKGROUND
----------
The view originally selected its run with:

    WHERE pr.run_id = (SELECT run_id FROM runs ORDER BY created_at DESC LIMIT 1)

`runs` holds rows from BOTH ingest and projection runs, and an ingest run has no
projection rows. So whenever an ingest run happened to be newest, the view went
empty while the projections sat in the table unreachable.

scripts/refresh.py hid this: it calls ingest.run() then persist_projections(),
leaving a projection run newest. But mcp_server.refresh_data() called
ingest.run() and nothing else, and AGENT.md instructs the agent to call it at
the start of every session — so the failure was on the routine weekly path, not
an exotic one. Measured on a real database: 3,540 rows before the call, 0 after.

It is another failure that reports success. `run_sql` against the view returns
"_no rows_", which reads as "nothing to report" rather than "the view is
broken", and the other tools kept working because they compute in-process from
_PROJ rather than reading the view.

These tests construct the `runs` and `projections` rows directly so they stay
fast and hermetic — no network, no model fitting.
"""

from __future__ import annotations

import sqlite3
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from fplbrain import ingest


# --------------------------------------------------------------- fixtures


def _fresh_db() -> tuple[sqlite3.Connection, tempfile.TemporaryDirectory]:
    """A schema-initialised DB in a temp dir. Caller keeps the TemporaryDirectory alive."""
    tmp = tempfile.TemporaryDirectory()
    conn = ingest.connect(Path(tmp.name) / "t.db")
    ingest.init_schema(conn)
    return conn, tmp


def _seed_player(conn: sqlite3.Connection, season="2026-27", element=1) -> None:
    """v_latest_projections joins players; without a row the view yields nothing."""
    pd.DataFrame([{
        "season": season, "element": element, "code": 900000 + element,
        "web_name": f"P{element}", "team_id": 1, "position": "MID", "now_cost": 7.5,
    }]).to_sql("players", conn, if_exists="append", index=False)
    pd.DataFrame([{
        "season": season, "team_id": 1, "code": 3, "name": "Alpha", "short_name": "ALP",
    }]).to_sql("teams", conn, if_exists="append", index=False)


def _add_run(conn: sqlite3.Connection, run_id: str, created_at: str, notes: str) -> None:
    conn.execute(
        "INSERT INTO runs (run_id, created_at, season, notes) VALUES (?,?,?,?)",
        (run_id, created_at, "2026-27", notes))
    conn.commit()


def _add_projections(conn: sqlite3.Connection, run_id: str, gws=(1, 2),
                     season="2026-27", element=1, exp_points=5.0) -> None:
    pd.DataFrame([
        {"season": season, "element": element, "gw": gw, "run_id": run_id,
         "p_start": 0.9, "exp_minutes": 80.0, "p_60": 0.85, "exp_goals": 0.3,
         "exp_assists": 0.2, "p_clean_sheet": 0.3, "p_defcon": 0.2,
         "exp_bonus": 0.4, "exp_points": exp_points, "sd_points": 2.0}
        for gw in gws
    ]).to_sql("projections", conn, if_exists="append", index=False)
    conn.commit()


def _view_rows(conn: sqlite3.Connection) -> pd.DataFrame:
    return pd.read_sql("SELECT * FROM v_latest_projections", conn)


# --------------------------------------------------------------- the tests


def test_view_populated_after_projection_run():
    conn, tmp = _fresh_db()
    with tmp:
        _seed_player(conn)
        _add_run(conn, "proj-1", "2026-08-17T10:00:00+00:00", "projection")
        _add_projections(conn, "proj-1", gws=(1, 2))
        assert len(_view_rows(conn)) == 2


def test_ingest_run_after_projection_does_not_empty_the_view():
    """
    THE BUG, in its exact shape.

    An ingest run lands with a LATER created_at than the projection run. Under
    the old SQL the view resolved to the ingest run, which has no projection
    rows, and returned nothing. This is precisely what
    mcp_server.refresh_data() produced on every call.
    """
    conn, tmp = _fresh_db()
    with tmp:
        _seed_player(conn)
        _add_run(conn, "proj-1", "2026-08-17T10:00:00+00:00", "projection")
        _add_projections(conn, "proj-1", gws=(1, 2))
        assert len(_view_rows(conn)) == 2, "precondition failed"

        # ...then an ingest-only refresh, strictly later.
        _add_run(conn, "ingest-1", "2026-08-17T11:00:00+00:00", "ingest")

        rows = _view_rows(conn)
        assert len(rows) == 2, (
            "an ingest run with no projections became the newest run and emptied "
            "the view; it must resolve to the newest run that HAS projections")
        assert set(rows["gw"]) == {1, 2}


def test_view_returns_only_the_newest_projection_run():
    conn, tmp = _fresh_db()
    with tmp:
        _seed_player(conn)
        _add_run(conn, "proj-old", "2026-08-17T10:00:00+00:00", "projection")
        _add_projections(conn, "proj-old", gws=(1, 2), exp_points=1.0)
        _add_run(conn, "proj-new", "2026-08-17T12:00:00+00:00", "projection")
        _add_projections(conn, "proj-new", gws=(3, 4, 5), exp_points=9.0)

        rows = _view_rows(conn)
        assert set(rows["gw"]) == {3, 4, 5}, "stale run leaked into the view"
        assert (rows["exp_points"] == 9.0).all()


def test_view_survives_an_ingest_between_two_projection_runs():
    """Ordering guard: interleaving must not resurrect the older projection run."""
    conn, tmp = _fresh_db()
    with tmp:
        _seed_player(conn)
        _add_run(conn, "proj-old", "2026-08-17T10:00:00+00:00", "projection")
        _add_projections(conn, "proj-old", gws=(1,), exp_points=1.0)
        _add_run(conn, "ingest-1", "2026-08-17T11:00:00+00:00", "ingest")
        _add_run(conn, "proj-new", "2026-08-17T12:00:00+00:00", "projection")
        _add_projections(conn, "proj-new", gws=(7, 8), exp_points=9.0)
        _add_run(conn, "ingest-2", "2026-08-17T13:00:00+00:00", "ingest")

        rows = _view_rows(conn)
        assert set(rows["gw"]) == {7, 8}
        assert (rows["exp_points"] == 9.0).all()


def test_refresh_data_leaves_the_view_populated(monkeypatch, tmp_path):
    """
    The tool must persist projections, not just ingest.

    Network and model fitting are stubbed: this asserts the ORCHESTRATION —
    that refresh_data() persists a projection run and leaves the view readable —
    not the model's numbers, which tests/test_constraints.py covers.
    """
    from fplbrain import mcp_server

    db = tmp_path / "t.db"
    conn = ingest.connect(db)
    ingest.init_schema(conn)
    _seed_player(conn)
    # A fixture so _next_gw() has something to resolve.
    conn.execute(
        "INSERT INTO fixtures (season, fixture_id, gw, home_team, away_team, finished) "
        "VALUES ('2026-27', 1, 1, 1, 1, 0)")
    conn.commit()
    conn.close()

    monkeypatch.setattr(mcp_server, "DB", str(db))

    def fake_ingest_run(db_path, current, hist, strict=True):
        c = ingest.connect(db_path)
        c.execute("INSERT INTO runs (run_id, created_at, season, notes) "
                  "VALUES ('ingest-x', '2026-08-17T11:00:00+00:00', '2026-27', 'ingest')")
        c.commit()
        c.close()
        return {"validation": {}, "crosswalks": {}, "provenance": {},
                "distinctness": {"ok": True, "problems": []}}

    def fake_project_horizon(conn, season, start_gw, n_gw, hist):
        return pd.DataFrame([
            {"element": 1, "gw": gw, "p_start": 0.9, "exp_minutes": 80.0, "p_60": 0.85,
             "exp_goals": 0.3, "exp_assists": 0.2, "p_clean_sheet": 0.3, "p_defcon": 0.2,
             "exp_bonus": 0.4, "exp_points": 5.0, "sd_points": 2.0}
            for gw in range(start_gw, start_gw + n_gw)
        ]), {}

    monkeypatch.setattr(mcp_server.ingest if hasattr(mcp_server, "ingest") else ingest,
                        "run", fake_ingest_run, raising=False)
    monkeypatch.setattr(ingest, "run", fake_ingest_run)
    monkeypatch.setattr(mcp_server.models, "project_horizon", fake_project_horizon)

    # Called with NO arguments deliberately. The discriminator must be the
    # behaviour — whether the view is left populated — not the signature, so
    # that this test fails against the old ingest-only implementation for the
    # right reason rather than on a TypeError.
    out = mcp_server.refresh_data()

    conn = sqlite3.connect(db)
    rows = _view_rows(conn)
    assert len(rows) > 0, (
        "refresh_data() left v_latest_projections EMPTY; it ingested without "
        "persisting projections, so the newest run has no projection rows")
    assert set(rows["gw"]) == {1, 2, 3, 4, 5, 6}, (
        f"expected the default 6-gameweek horizon, got {sorted(rows['gw'])}")
    assert "projections" in out and "run_id" in out
    conn.close()
