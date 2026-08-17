"""
Regression tests for the live-path season-corruption bug.

BACKGROUND
----------
The FPL API's `bootstrap-static` and `/fixtures/` endpoints are current-season-only.
They accept no season parameter and have no historical mode. The original
sources.py threaded a `season` argument into functions that called them anyway, so
loading 2024-25 with the live API reachable wrote CURRENT-season rows into the
database stamped `season='2024-25'`.

It failed upward: the cross-season crosswalk went to a perfect 1.0 (the current
player list matching itself), which looks better than the correct 0.806.

These tests were only possible to write after the bug was found in an environment
where the API WAS reachable. The build environment got a 403, so the broken path
never executed and every published number came from the fallback.

We simulate reachability by monkeypatching sources._get.
"""

from __future__ import annotations

import json
import sqlite3
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from fplbrain import ingest, sources


# --------------------------------------------------------------- fake live API


CURRENT_ELEMENTS = [
    {"id": 1, "code": 900001, "web_name": "CurrentGuy", "first_name": "Cur",
     "second_name": "Rent", "team": 1, "element_type": 3, "now_cost": 75,
     "status": "a", "selected_by_percent": "5.0", "minutes": 0, "starts": 0,
     "total_points": 0, "points_per_game": "0.0", "form": "0.0", "ep_next": "0.0"},
    {"id": 2, "code": 900002, "web_name": "OtherGuy", "first_name": "Oth",
     "second_name": "Er", "team": 2, "element_type": 2, "now_cost": 50,
     "status": "a", "selected_by_percent": "1.0", "minutes": 0, "starts": 0,
     "total_points": 0, "points_per_game": "0.0", "form": "0.0", "ep_next": "0.0"},
]
CURRENT_TEAMS = [
    {"id": 1, "code": 3, "name": "Alpha", "short_name": "ALP", "strength": 4},
    {"id": 2, "code": 7, "name": "Beta", "short_name": "BET", "strength": 3},
]
CURRENT_FIXTURES = [
    {"id": 1, "event": 1, "team_h": 1, "team_a": 2, "team_h_score": None,
     "team_a_score": None, "finished": False, "kickoff_time": "2026-08-21T19:00:00Z",
     "team_h_difficulty": 2, "team_a_difficulty": 5},
]


class _FakeResponse:
    def __init__(self, payload=None, content=b""):
        self._payload = payload
        self.content = content
        self.url = "fake://live"

    def json(self):
        return self._payload

    def raise_for_status(self):
        return None


@pytest.fixture
def live_api(monkeypatch):
    """Make the FPL API appear reachable and record which URLs get requested."""
    calls: list[str] = []

    def fake_get(url, timeout=30):
        calls.append(url)
        if "bootstrap-static" in url:
            return _FakeResponse(payload={
                "elements": CURRENT_ELEMENTS,
                "teams": CURRENT_TEAMS,
                "events": [{"id": 1, "name": "Gameweek 1"}],
            })
        if url.rstrip("/").endswith("/fixtures"):
            return _FakeResponse(payload=CURRENT_FIXTURES)
        if "raw.githubusercontent.com" in url:
            raise RuntimeError("mirror deliberately unavailable in this test")
        raise RuntimeError(f"unexpected url {url}")

    monkeypatch.setattr(sources, "_get", fake_get)
    monkeypatch.setattr(sources, "ALLOW_LIVE", True)
    monkeypatch.setattr(sources, "CURRENT_SEASON", "2026-27")
    with tempfile.TemporaryDirectory() as d:
        monkeypatch.setattr(sources, "CACHE", Path(d))
        yield calls


# --------------------------------------------------------------- the tests


def test_live_used_for_current_season(live_api):
    """The current season SHOULD use the live API — that's the point of it."""
    got = sources.current_players("2026-27")
    assert got.path == "live"
    assert any("bootstrap-static" in u for u in live_api)


def test_live_never_used_for_historical_players(live_api):
    """
    THE BUG. bootstrap-static cannot serve 2024-25, so it must not be called.
    Before the fix this returned current-season players labelled 'live'.
    """
    with pytest.raises(RuntimeError):
        # mirror is down in this test, so with live correctly refused there is
        # no source left — failing loudly is the correct behaviour.
        sources.current_players("2024-25")
    assert not any("bootstrap-static" in u for u in live_api), (
        "bootstrap-static was called for a historical season; it is "
        "current-season-only and will return the wrong data")


def test_live_never_used_for_historical_fixtures(live_api):
    with pytest.raises(RuntimeError):
        sources.current_fixtures("2025-26")
    assert not any(u.rstrip("/").endswith("/fixtures") for u in live_api), (
        "/fixtures/ was called for a historical season")


def test_teams_cache_not_mislabelled_as_live(live_api, monkeypatch):
    """
    A cached file written by the mirror must never be reported as 'live'.
    The original code returned any existing cache tagged 'live' unconditionally.
    """
    sources._write_cache("teams_2026-27", "csv",
                         pd.DataFrame(CURRENT_TEAMS).to_csv(index=False).encode(),
                         "github")
    got = sources.current_teams("2026-27")
    assert got.path != "live", "mirror-sourced cache was reported as live"


def test_teams_cache_from_live_is_labelled_live(live_api):
    sources.current_players("2026-27")          # populates the teams cache via live
    got = sources.current_teams("2026-27")
    assert got.path == "live"


# --------------------------------------------------------------- integrity check


def _seed(conn, season, codes):
    df = pd.DataFrame({
        "season": season,
        "element": range(1, len(codes) + 1),
        "code": codes,
        "web_name": [f"p{c}" for c in codes],
        "team_id": 1,
        "position": "MID",
    })
    df.to_sql("players", conn, if_exists="append", index=False)


def test_integrity_check_catches_identical_seasons():
    with tempfile.TemporaryDirectory() as d:
        db = Path(d) / "t.db"
        conn = ingest.connect(db)
        ingest.init_schema(conn)
        _seed(conn, "2024-25", [1, 2, 3, 4])
        _seed(conn, "2026-27", [1, 2, 3, 4])       # identical -> corruption
        conn.commit()
        rep = ingest.check_season_distinctness(conn, ["2024-25", "2026-27"])
        assert rep["ok"] is False
        assert any("IDENTICAL" in p for p in rep["problems"])


def test_integrity_check_passes_on_distinct_seasons():
    with tempfile.TemporaryDirectory() as d:
        db = Path(d) / "t.db"
        conn = ingest.connect(db)
        ingest.init_schema(conn)
        _seed(conn, "2024-25", list(range(1, 40)))
        _seed(conn, "2026-27", list(range(30, 70)))
        conn.commit()
        rep = ingest.check_season_distinctness(conn, ["2024-25", "2026-27"])
        assert rep["ok"] is True, rep["problems"]


# --------------------------------------------------------------- durability


def test_saved_squad_survives_reingest():
    """
    schema.sql runs on every ingest. squad_state holds USER data, so a DROP
    there would delete the squad the moment the agent called refresh_data().
    """
    with tempfile.TemporaryDirectory() as d:
        db = Path(d) / "t.db"
        conn = ingest.connect(db)
        ingest.init_schema(conn)
        conn.execute(
            "INSERT INTO squad_state (gw, recorded_at, bank, free_transfers, "
            "chips_remaining, squad, notes) VALUES (?,?,?,?,?,?,?)",
            (1, "2026-08-17T00:00:00Z", 0.5, 1, json.dumps(["wildcard"]),
             json.dumps([{"element": 1}]), "my squad"))
        conn.commit()

        ingest.init_schema(conn)   # simulate a later refresh_data()

        rows = pd.read_sql("SELECT * FROM squad_state", conn)
        assert len(rows) == 1, "saved squad was wiped by re-running init_schema"
        assert rows.iloc[0]["notes"] == "my squad"
