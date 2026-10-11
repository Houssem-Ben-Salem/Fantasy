"""
Reading a manager's real squad from the public FPL API.

Five gameweeks of squad state were entered by hand, and the bank was an
estimate every time: the sessions planned GW7 around GBP1.3m when the real bank
was GBP0.7m, which decides whether Saka is affordable. The public endpoints give
picks, bank, transfers and chips. Two things they do not give are derived, and
these tests pin the rules that derive them:

  * selling price — FPL keeps half of any rise, rounded down to GBP0.1m;
  * free transfers — +1 per gameweek, capped at 5, spent by transfers, kept
    through a Wildcard or Free Hit week.

Hermetic: every HTTP call is answered from canned payloads.
"""
from __future__ import annotations

import pytest

from fplbrain import entry


@pytest.mark.parametrize("purchase, now, expected", [
    (76, 78, 77),    # +0.2 -> keep half: +0.1
    (76, 79, 77),    # +0.3 -> half is 0.15, rounded DOWN to 0.1
    (76, 80, 78),    # +0.4 -> +0.2
    (80, 78, 78),    # a fall is passed on in full
    (60, 60, 60),
])
def test_selling_price_keeps_half_of_a_rise_rounded_down(purchase, now, expected):
    assert entry.selling_price(purchase, now) == expected


def _hist(transfers_by_gw: dict[int, int], upto: int) -> list[dict]:
    return [{"event": g, "event_transfers": transfers_by_gw.get(g, 0)}
            for g in range(1, upto + 1)]


@pytest.mark.parametrize("name, transfers, chips, next_gw, expected", [
    # this user's real 2026-27 history: one transfer in each of GW2-5, none in GW6
    ("one a week, then a roll", {2: 1, 3: 1, 4: 1, 5: 1}, [], 7, 2),
    ("rolling caps at five", {}, [], 9, 5),
    ("a hit leaves nothing to roll", {2: 3}, [], 3, 1),
    ("wildcard keeps saved transfers", {3: 8}, [{"name": "wildcard", "event": 3}], 4, 3),
    ("free hit keeps saved transfers", {3: 6}, [{"name": "freehit", "event": 3}], 4, 3),
])
def test_free_transfers(name, transfers, chips, next_gw, expected):
    assert entry.free_transfers(_hist(transfers, next_gw - 1), chips, next_gw) == expected


# ------------------------------------------------------------- end to end

TEAMS = [{"id": 1, "short_name": "AAA"}, {"id": 2, "short_name": "BBB"}]


def _elements() -> list[dict]:
    """15 players: 2 GK, 5 DEF, 5 MID, 3 FWD; element 15 was bought in GW3."""
    types = [1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4]
    return [{"id": i, "web_name": f"P{i}", "element_type": t, "team": 1 + i % 2,
             "now_cost": 50 + i} for i, t in enumerate(types, start=1)] + \
           [{"id": 99, "web_name": "Sold", "element_type": 4, "team": 1, "now_cost": 60}]


def _payloads(active_chip=None) -> dict[str, object]:
    picks = [{"element": i, "position": i, "multiplier": 2 if i == 8 else 1,
              "is_captain": i == 8, "is_vice_captain": i == 9} for i in range(1, 16)]
    fh_squad = [{"element": 99, "position": 1, "multiplier": 1,
                 "is_captain": False, "is_vice_captain": False}] * 15
    gw1_price = {i: 50 + i - 2 for i in range(1, 16)}   # every held player has risen by 0.2
    p = {
        "bootstrap-static/": {"elements": _elements(), "teams": TEAMS,
                              "events": [{"id": 4, "is_current": True, "is_next": False},
                                         {"id": 5, "is_current": False, "is_next": True}]},
        "entry/42/": {"id": 42, "name": "Test XI", "current_event": 4, "started_event": 1,
                      "last_deadline_bank": 7, "last_deadline_value": 1001},
        "entry/42/history/": {"current": _hist({2: 1, 3: 1}, 4),
                              "chips": [{"name": "freehit", "event": 4}] if active_chip else []},
        "entry/42/transfers/": [{"event": 3, "element_in": 15, "element_in_cost": 61,
                                 "element_out": 99, "element_out_cost": 60}],
        "entry/42/event/4/picks/": {"active_chip": active_chip,
                                    "entry_history": {"event": 4, "bank": 7, "value": 1001},
                                    "picks": fh_squad if active_chip == "freehit" else picks},
        "entry/42/event/3/picks/": {"active_chip": None,
                                    "entry_history": {"event": 3, "bank": 7, "value": 1001},
                                    "picks": picks},
    }
    for i in range(1, 16):
        p[f"element-summary/{i}/"] = {"history": [{"round": 1, "value": gw1_price[i]},
                                                  {"round": 2, "value": gw1_price[i] + 1}]}
    return p


@pytest.fixture
def fake_api(monkeypatch):
    def install(active_chip=None):
        payloads = _payloads(active_chip)
        monkeypatch.setattr(entry, "_get", lambda path: payloads[path])
    return install


def test_state_from_the_api(fake_api):
    fake_api()
    s = entry.entry_state(42)
    assert (s["gw_next"], s["bank"], s["free_transfers"]) == (5, 0.7, 2)
    assert len(s["squad"]) == 15
    sq = {p["element"]: p for p in s["squad"]}
    # bought in GW3 for 6.1, now 6.5: half of +0.4 kept
    assert (sq[15]["purchase_price"], sq[15]["selling_price"]) == (6.1, 6.3)
    # held since GW1 at 4.9 (51 - 2), now 5.1: +0.1 kept
    assert (sq[1]["purchase_price"], sq[1]["selling_price"]) == (4.9, 5.0)
    assert sq[8]["captain"] and sq[9]["vice"]
    assert sorted(s["chips_remaining"]) == ["bench_boost", "free_hit", "triple_captain", "wildcard"]
    assert {"selling_price", "free_transfers"} <= set(s["derived"])


def test_a_free_hit_week_reports_the_squad_that_returns(fake_api):
    """The Free Hit squad is borrowed for one week; planning needs the one that
    comes back, which is the previous gameweek's."""
    fake_api(active_chip="freehit")
    s = entry.entry_state(42)
    assert sorted(p["element"] for p in s["squad"]) == list(range(1, 16))
    assert "free_hit" not in s["chips_remaining"]


def test_a_sync_keeps_the_notes_already_saved_for_that_gameweek(fake_api, tmp_path):
    """squad_state is keyed on gameweek and written with INSERT OR REPLACE, so a
    sync replaced the whole row — including judgement notes the end-of-season
    review scores decisions by. Keep them; replace only the previous sync line."""
    import sqlite3

    from fplbrain import ingest
    conn = sqlite3.connect(tmp_path / "fpl.db")
    ingest.init_schema(conn)
    conn.execute("INSERT INTO squad_state (gw, notes) VALUES (5, ?)",
                 ("JUDGEMENT: started Wissa on the market line, not the model's.",))
    fake_api()
    s = entry.entry_state(42)
    entry.save(conn, s)
    entry.save(conn, s)
    notes = conn.execute("SELECT notes FROM squad_state WHERE gw=5").fetchone()[0]
    assert "JUDGEMENT: started Wissa" in notes
    assert notes.count("Synced from the FPL API") == 1
