"""
Every persisted projection run must say which code and which constants made it.

The prospective scorecard scores what the system said before each deadline. A
run made before 15 Sep 2026 came from a model with binary p_start and pinned
team ratings; one made on 11 Oct from a model with fixture-scaled attack. The
runs table recorded neither, so telling them apart meant guessing from
timestamps. Hermetic: in-memory database from schema.sql.
"""
from __future__ import annotations

import json
import sqlite3

import pandas as pd

from fplbrain import ingest, models


def test_a_run_records_its_code_version_and_constants():
    conn = sqlite3.connect(":memory:")
    ingest.init_schema(conn)
    proj = pd.DataFrame({"element": [1], "gw": [7], "exp_points": [3.0], "p_start": [0.9]})
    run_id = models.persist_projections(conn, proj, "2026-27", {"note": "test"})
    prov = json.loads(conn.execute("SELECT provenance FROM runs WHERE run_id=?",
                                   (run_id,)).fetchone()[0])
    assert prov["note"] == "test"                      # diagnostics still recorded
    assert prov["code_version"]                        # a commit, or "unknown" — never absent
    assert prov["constants"]["ATTACK_FIXTURE_ALPHA"] == models.ATTACK_FIXTURE_ALPHA
    assert prov["constants"]["MINUTES_PRIOR_K"] == models.MINUTES_PRIOR_K
    assert prov["constants"]["GOAL_RIDGE_LAMBDA"] == models.GOAL_RIDGE_LAMBDA
    assert prov["constants"]["XG_WEIGHT"] == models.XG_WEIGHT
