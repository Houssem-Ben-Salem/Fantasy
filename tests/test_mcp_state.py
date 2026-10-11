"""
The MCP tools must plan from the truth: the next OPEN gameweek, and the squad
the manager actually owns.

1. `_next_gw()` was "the lowest gameweek with an unfinished fixture". Between a
   deadline and that gameweek's last match — every weekend — that is a week
   that can no longer be changed: on Sunday 11 Oct 2026, with GW6 under way,
   it returned 6, so plan_transfers and chip_advice planned a locked gameweek,
   and scripts/refresh.py projected from it too.
2. plan_transfers and chip_advice took a typed list of 15 names. Five weeks of
   hand-entered squads carried the wrong backup goalkeeper (an Ipswich "Palmer"
   instead of Verbruggen) and a bank estimate GBP0.6m too high. With no names
   they now use the squad synced from the FPL API, selling prices included.
3. sync_squad writes that state.

Hermetic: a temporary database built from schema.sql, synthetic projections,
and canned API payloads. The tools are called exactly as the agent calls them.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from fplbrain import entry, ingest, mcp_server, optimize

SEASON = "2026-27"


def _iso(t: datetime) -> str:
    return t.strftime("%Y-%m-%dT%H:%M:%SZ")


@pytest.fixture
def db(tmp_path, monkeypatch):
    path = tmp_path / "fpl.db"
    with sqlite3.connect(path) as c:
        ingest.init_schema(c)
    monkeypatch.setattr(mcp_server, "DB", str(path))
    monkeypatch.setattr(mcp_server, "SEASON", SEASON)
    return path


def _fixtures(path, rows):
    """rows: (fixture_id, gw, kickoff datetime, finished)"""
    with sqlite3.connect(path) as c:
        c.executemany(
            "INSERT INTO fixtures (season, fixture_id, gw, kickoff_time, home_team, "
            "away_team, finished) VALUES (?,?,?,?,1,2,?)",
            [(SEASON, f, g, _iso(k), fin) for f, g, k, fin in rows])


def test_planning_skips_a_gameweek_whose_deadline_has_passed(db):
    now = datetime.now(timezone.utc)
    _fixtures(db, [(1, 6, now - timedelta(hours=26), 1),     # GW6 under way:
                   (2, 6, now - timedelta(hours=2), 0),      # deadline long gone
                   (3, 6, now + timedelta(days=1), 0),
                   (4, 7, now + timedelta(days=6), 0)])
    assert mcp_server._next_gw() == 7


def test_planning_keeps_a_gameweek_until_its_deadline(db):
    now = datetime.now(timezone.utc)
    _fixtures(db, [(1, 6, now + timedelta(hours=3), 0),      # deadline in 1.5h
                   (2, 7, now + timedelta(days=7), 0)])
    assert mcp_server._next_gw() == 6


# ------------------------------------------------------------ saved squad

SQUAD = list(range(101, 116))
SELL = {e: 5.0 + (e - 100) / 10 for e in SQUAD}


def _save_state(path, gw=7, bank=0.7, ft=2):
    squad = [{"element": e, "selling_price": SELL[e]} for e in SQUAD]
    with sqlite3.connect(path) as c:
        c.execute("INSERT INTO squad_state (gw, recorded_at, bank, free_transfers, "
                  "chips_remaining, squad, notes) VALUES (?,?,?,?,?,?,?)",
                  (gw, "2026-10-11T00:00:00+00:00", bank, ft, "[]", json.dumps(squad), ""))


def _proj(*_a, **_k) -> pd.DataFrame:
    pos = ["GK"] * 2 + ["DEF"] * 5 + ["MID"] * 5 + ["FWD"] * 3
    rows = [dict(element=e, web_name=f"P{e}", position=p, team=f"T{e % 8}", now_cost=5.0,
                 selected_by_percent=1.0, gw=7, exp_points=3.0, p_start=0.9, sd_points=2.0)
            for e, p in zip(SQUAD, pos)]
    return pd.DataFrame(rows)


@pytest.fixture
def captured(db, monkeypatch):
    calls = {}
    monkeypatch.setattr(mcp_server, "_projections", _proj)
    monkeypatch.setattr(mcp_server, "_next_gw", lambda: 7)

    def fake_plan(proj, current_squad, horizon, bank, free_transfers, **kw):
        calls["plan"] = dict(squad=sorted(current_squad), bank=bank,
                             ft=free_transfers, sell=kw.get("selling_prices"))
        return optimize.OptimResult("Optimal", 0.0, pd.DataFrame(), pd.DataFrame(),
                                    pd.DataFrame(), {}, {})

    def fake_chips(proj, squad, gw, **kw):
        calls["chips"] = sorted(squad)
        return {"gw": gw}

    monkeypatch.setattr(optimize, "transfer_plan", fake_plan)
    monkeypatch.setattr(optimize, "evaluate_chips", fake_chips)
    return calls


def test_plan_transfers_uses_the_synced_squad_when_none_is_typed(db, captured):
    _save_state(db)
    mcp_server.plan_transfers("")
    assert captured.get("plan") == dict(squad=SQUAD, bank=0.7, ft=2, sell=SELL)


def test_typed_bank_and_transfers_still_override_the_saved_state(db, captured):
    _save_state(db)
    mcp_server.plan_transfers("", bank=1.5, free_transfers=1)
    assert (captured["plan"]["bank"], captured["plan"]["ft"]) == (1.5, 1)


def test_chip_advice_uses_the_synced_squad_when_none_is_typed(db, captured):
    _save_state(db)
    mcp_server.chip_advice("")
    assert captured.get("chips") == SQUAD


def test_sync_squad_writes_the_state_it_reports(db, monkeypatch):
    from test_entry import _payloads
    payloads = _payloads()
    monkeypatch.setattr(entry, "_get", lambda path: payloads[path])
    out = mcp_server.sync_squad(entry_id=42)
    assert "GW5" in out and "0.7" in out
    with sqlite3.connect(db) as c:
        gw, bank, ft, squad = c.execute(
            "SELECT gw, bank, free_transfers, squad FROM squad_state").fetchone()
    assert (gw, bank, ft, len(json.loads(squad))) == (5, 0.7, 2, 15)


def test_reported_gain_is_the_solver_gain_over_rolling(db, monkeypatch):
    """The headline must be objective(plan) - objective(roll): the number the
    hit policy in prompts/AGENT.md is judged against. It was the sum of raw
    per-pair horizon xP, which counts a benched player's replacement in full
    (bench weight is 0.12) and ignores the decay — GW7 2026-27 printed +16.80
    for a plan worth +7.0 to the solver."""
    _save_state(db)
    monkeypatch.setattr(mcp_server, "_projections", _proj)
    monkeypatch.setattr(mcp_server, "_next_gw", lambda: 7)

    def fake_plan(proj, current_squad, horizon, bank, free_transfers, max_transfers=3, **kw):
        moves = [] if max_transfers == 0 else [
            {"out": "A", "out_xp": 5.0, "in": "B", "in_xp": 35.0, "xp_gain": 30.0}]
        return optimize.OptimResult("Optimal", 100.0 if max_transfers == 0 else 110.0,
                                    pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {}, {},
                                    transfers=moves)

    monkeypatch.setattr(optimize, "transfer_plan", fake_plan)
    out = mcp_server.plan_transfers("")
    assert "+10.00" in out and "+30.00 xP" not in out.split("Gain over rolling")[0]


def test_sync_squad_remembers_the_last_synced_entry(db, monkeypatch):
    """With no id and no FPLBRAIN_ENTRY_ID, reuse the entry the last sync recorded
    — so the MCP config never has to be edited by hand to carry a team id."""
    from test_entry import _payloads
    payloads = _payloads()
    monkeypatch.setattr(entry, "_get", lambda path: payloads[path])
    monkeypatch.delenv("FPLBRAIN_ENTRY_ID", raising=False)
    mcp_server.sync_squad(entry_id=42)
    out = mcp_server.sync_squad()
    assert "(entry 42)" in out


# ------------------------------------------------------------ scenarios and decision notes

def _proj_multi(*_a, **_k) -> pd.DataFrame:
    """The synthetic squad over GW7-11."""
    return pd.concat([_proj().assign(gw=g) for g in range(7, 12)], ignore_index=True)


def test_plan_transfers_can_rule_a_player_out_for_some_weeks(db, monkeypatch):
    """Injury scenarios ("if Gibbs-White misses two weeks") took a Python script
    on 11 Oct 2026; the agent needs them as an argument."""
    _save_state(db)
    seen = {}
    monkeypatch.setattr(mcp_server, "_projections", _proj_multi)
    monkeypatch.setattr(mcp_server, "_next_gw", lambda: 7)

    def fake_plan(proj, current_squad, horizon, bank, free_transfers, max_transfers=3, **kw):
        seen.setdefault("proj", proj)
        return optimize.OptimResult("Optimal", 0.0, pd.DataFrame(), pd.DataFrame(),
                                    pd.DataFrame(), {}, {})

    monkeypatch.setattr(optimize, "transfer_plan", fake_plan)
    out = mcp_server.plan_transfers("", unavailable="101:2")
    p = seen["proj"].set_index(["element", "gw"])
    assert p.loc[(101, 7), "exp_points"] == 0 and p.loc[(101, 8), "p_start"] == 0
    assert p.loc[(101, 9), "exp_points"] == 3.0            # back after two weeks
    assert p.loc[(102, 7), "exp_points"] == 3.0            # nobody else touched
    assert "101" in out or "P101" in out                   # the scenario is stated


def test_a_decision_note_is_appended_without_touching_the_squad(db):
    _save_state(db)
    mcp_server.note_decision("JUDGEMENT: Saka captain over Fernandes on ownership.", gw=7)
    mcp_server.note_decision("Wildcard held: only one player out beyond GW7.", gw=7)
    with sqlite3.connect(db) as c:
        squad, bank, notes = c.execute(
            "SELECT squad, bank, notes FROM squad_state WHERE gw=7").fetchone()
    assert len(json.loads(squad)) == 15 and bank == 0.7
    assert "Saka captain" in notes and "Wildcard held" in notes
