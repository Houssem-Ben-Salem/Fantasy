"""
The news log has to make the scout scoreable — and impossible to flatter.

LIMITATIONS s5: transformations of what the model already knows failed three
times; information it cannot see (injuries, press conferences, line-ups) is the
input class that can still add value. Until now that news was searched for at
the deadline and thrown away: unlogged, so never testable. news_claims keeps
every claim with its source and the moment it was recorded, so:

  * it must survive every re-ingest (it is the evidence, like squad_state);
  * a batch with any bad claim records nothing — half-logged news is worse
    than none, because the gaps are invisible later;
  * names resolve through the one resolver ("Palmer" is two players);
  * a claim recorded after a deadline is never scored on that gameweek;
  * the measure that matters is accuracy where the scout DISAGREES with
    persistence ("starts iff started last week") — persistence already gets
    the easy cases right.

Hermetic: a temporary database from schema.sql, two clubs, chosen outcomes.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone

import pytest

from fplbrain import ingest, mcp_server, news

S = "2026-27"
GW5_KO = datetime(2026, 9, 18, 19, 0, tzinfo=timezone.utc)
GW6_KO = datetime(2026, 10, 10, 11, 30, tzinfo=timezone.utc)
DEADLINES = {5: GW5_KO - timedelta(minutes=90), 6: GW6_KO - timedelta(minutes=90)}

# element: (web_name, team_id, position)
PLAYERS = {10: ("Palmer", 1, "MID"), 11: ("Palmer", 2, "GK"), 12: ("Starter", 1, "MID"),
           13: ("Bench", 1, "MID"), 14: ("Injured", 2, "FWD")}
# (gw, element): minutes — Bench rotates IN for GW6, Injured misses it
MINUTES = {(5, 12): 90, (5, 13): 0, (5, 14): 90, (6, 12): 90, (6, 13): 90, (6, 14): 0}


def _iso(t: datetime) -> str:
    return t.strftime("%Y-%m-%dT%H:%M:%SZ")


def _db(path) -> sqlite3.Connection:
    c = sqlite3.connect(path)
    ingest.init_schema(c)
    c.executemany("INSERT INTO teams (season, team_id, short_name) VALUES (?,?,?)",
                  [(S, 1, "CHE"), (S, 2, "IPS")])
    c.executemany("INSERT INTO players (season, element, code, web_name, team_id, position, "
                  "now_cost, status) VALUES (?,?,?,?,?,?,?, 'a')",
                  [(S, e, 100 + e, n, t, p, 5.0) for e, (n, t, p) in PLAYERS.items()])
    for gw, ko in ((5, GW5_KO), (6, GW6_KO)):
        c.execute("INSERT INTO fixtures (season, fixture_id, gw, kickoff_time, home_team, "
                  "away_team, finished) VALUES (?,?,?,?,1,2,1)", (S, gw, gw, _iso(ko)))
    c.executemany("INSERT INTO player_gw (season, element, code, gw, fixture_id, was_home, "
                  "minutes, starts, total_points) VALUES (?,?,?,?,?,?,?,?,0)",
                  [(S, e, 100 + e, gw, gw, int(PLAYERS[e][1] == 1), m, int(m >= 60))
                   for (gw, e), m in MINUTES.items()])
    c.commit()
    return c


def _claim(**kw) -> dict:
    base = dict(gw=6, claim="starts", confidence="rumoured", source_type="predicted_lineup",
                source_name="Example", source_url="https://example.com/x",
                published_at="2026-10-09")
    return {**base, **kw}


@pytest.fixture
def db(tmp_path, monkeypatch):
    path = tmp_path / "fpl.db"
    _db(path).close()
    monkeypatch.setattr(mcp_server, "DB", str(path))
    monkeypatch.setattr(mcp_server, "SEASON", S)
    monkeypatch.setattr(mcp_server, "_next_gw", lambda: 6)
    return path


def _count(path) -> int:
    with sqlite3.connect(path) as c:
        return c.execute("SELECT COUNT(*) FROM news_claims").fetchone()[0]


# ------------------------------------------------------------ recording

def test_a_batch_with_one_bad_claim_records_nothing(db):
    batch = [_claim(player="Starter"), _claim(player="Bench", claim="probably fine")]
    out = mcp_server.record_news(json.dumps(batch))
    assert "Nothing recorded" in out and "probably fine" in out
    assert _count(db) == 0


def test_every_source_needs_a_url_except_an_fpl_flag(db):
    out = mcp_server.record_news(json.dumps([_claim(player="Starter", source_url="")]))
    assert "Nothing recorded" in out
    ok = mcp_server.record_news(json.dumps([_claim(player="Starter", claim="doubtful", chance=75,
                                                    confidence="confirmed", source_type="fpl_flag",
                                                    source_url="")]))
    assert "Nothing recorded" not in ok and _count(db) == 1


def test_an_ambiguous_name_is_refused_and_an_id_is_not(db):
    out = mcp_server.record_news(json.dumps([_claim(player="Palmer")]))
    assert "Nothing recorded" in out and "ambiguous" in out
    assert "Nothing recorded" not in mcp_server.record_news(json.dumps([_claim(player="10")]))


def test_the_log_survives_a_reingest(db):
    mcp_server.record_news(json.dumps([_claim(player="Starter")]))
    with sqlite3.connect(db) as c:
        ingest.init_schema(c)               # what every ingest / refresh_data() runs
    assert _count(db) == 1


def test_the_latest_claim_per_player_is_the_one_shown(db):
    with sqlite3.connect(db) as c:
        news.record(c, S, [{**_claim(claim="starts"), "element": 13, "player": "Bench"}],
                    now=DEADLINES[6] - timedelta(hours=30))
        news.record(c, S, [{**_claim(claim="benched"), "element": 13, "player": "Bench"}],
                    now=DEADLINES[6] - timedelta(hours=2))
        assert news.latest(c, S, 6).set_index("element").loc[13, "claim"] == "benched"


# ------------------------------------------------------------ scoring

def test_scoring_ignores_hindsight_and_measures_disagreement_with_persistence(db):
    before, after = DEADLINES[6] - timedelta(hours=3), DEADLINES[6] + timedelta(hours=1)
    with sqlite3.connect(db) as c:
        news.record(c, S, [{**_claim(claim="starts"), "element": 13, "player": "Bench"},     # rotates in: right
                           {**_claim(claim="out", confidence="confirmed", source_type="presser"),
                            "element": 14, "player": "Injured"},                              # misses: right
                           {**_claim(claim="starts"), "element": 12, "player": "Starter"}],  # agrees: right
                    now=before)
        news.record(c, S, [{**_claim(claim="benched"), "element": 12, "player": "Starter"}],
                    now=after)                                                              # hindsight
        res = news.score(c, S, DEADLINES)
    by = res["by_claim"].set_index("claim")
    assert by.loc["starts", "n"] == 2 and by.loc["starts", "correct"] == 2
    assert by.loc["out", "n"] == 1 and "benched" not in by.index        # the late claim never counts
    assert (res["disagree"]["n"], res["disagree"]["correct"]) == (2, 2)
    assert res["verdict"].startswith("WAIT")                            # 2 of 30 disagreements


def test_wilson_interval():
    assert news.wilson(0, 0) == (0.0, 1.0)
    lo, hi = news.wilson(10, 10)
    assert 0.69 < lo < 0.73 and hi == pytest.approx(1.0)
