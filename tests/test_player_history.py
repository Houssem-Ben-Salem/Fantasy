"""
Regression test: player_detail's season history must join on `code`, not `element`.

`element` is reassigned every season. The original query collected a player's
element ids from each season and matched those numbers in every season, so it
pulled in unrelated players who held the same id elsewhere — a backup goalkeeper
was credited with 18 goals in 2024-25 that belonged to a striker. Hermetic.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd

from fplbrain import ingest
from fplbrain.mcp_server import _season_history


def test_history_does_not_borrow_players_who_shared_an_element_id():
    with tempfile.TemporaryDirectory() as d:
        conn = ingest.connect(Path(d) / "t.db")
        ingest.init_schema(conn)
        # Keeper K (code 1): element 110 in 2026-27, element 141 in 2025-26.
        # Striker S (code 2): element 141 in 2026-27 and element 110 in 2025-26 —
        # i.e. the SAME NUMBERS, swapped. An element join mixes them up.
        pd.DataFrame([
            dict(season="2026-27", element=110, code=1, web_name="K", team_id=1, position="GK"),
            dict(season="2025-26", element=141, code=1, web_name="K", team_id=1, position="GK"),
            dict(season="2026-27", element=141, code=2, web_name="S", team_id=1, position="FWD"),
            dict(season="2025-26", element=110, code=2, web_name="S", team_id=1, position="FWD"),
        ]).to_sql("players", conn, if_exists="append", index=False)
        pd.DataFrame([
            dict(season="2026-27", element=110, code=1, gw=1, fixture_id=1, minutes=90, goals_scored=0),
            dict(season="2025-26", element=141, code=1, gw=1, fixture_id=1, minutes=0, goals_scored=0),
            dict(season="2026-27", element=141, code=2, gw=1, fixture_id=2, minutes=90, goals_scored=1),
            dict(season="2025-26", element=110, code=2, gw=1, fixture_id=2, minutes=90, goals_scored=18),
        ]).to_sql("player_gw", conn, if_exists="append", index=False)
        conn.commit()

        hist = _season_history(conn, "2026-27", 110)      # the keeper
        assert len(hist) == 2, f"expected one row per season, got {len(hist)}:\n{hist}"
        assert hist.goals.sum() == 0, (
            f"the goalkeeper was credited with {int(hist.goals.sum())} goals — "
            "history borrowed a different player who held the same element id")
