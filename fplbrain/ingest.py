"""
ETL: sources -> SQLite. Idempotent, validated, provenance-tracked.

Run:  python -m fplbrain.ingest --db data/fpl.db --history 2023-24 2024-25 2025-26
"""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import sources

log = logging.getLogger(__name__)

POSITION_MAP = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD", 5: "MNG"}

CURRENT_SEASON = "2026-27"

# Records duplicate rows removed per season during ingest, surfaced by validate().
_DUPES: dict[str, int] = {}


# ---------------------------------------------------------------- helpers


def _num(df: pd.DataFrame, col: str, default=np.nan) -> pd.Series:
    """Coerce a column to numeric, tolerating absence and junk values."""
    if col not in df.columns:
        return pd.Series([default] * len(df), index=df.index, dtype="float64")
    return pd.to_numeric(df[col], errors="coerce")


def _txt(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series([None] * len(df), index=df.index, dtype="object")
    return df[col].astype("object")


def connect(db_path: str | Path) -> sqlite3.Connection:
    p = Path(db_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(p)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_schema(conn: sqlite3.Connection) -> None:
    sql = (Path(__file__).parent / "schema.sql").read_text()
    conn.executescript(sql)
    conn.commit()


# ---------------------------------------------------------------- loaders


def load_teams(conn: sqlite3.Connection, season: str) -> str:
    f = sources.current_teams(season)
    df = f.data
    out = pd.DataFrame(
        {
            "season": season,
            "team_id": _num(df, "id").astype("Int64"),
            "code": _num(df, "code").astype("Int64"),
            "name": _txt(df, "name"),
            "short_name": _txt(df, "short_name"),
            "strength": _num(df, "strength"),
            "str_overall_home": _num(df, "strength_overall_home"),
            "str_overall_away": _num(df, "strength_overall_away"),
            "str_attack_home": _num(df, "strength_attack_home"),
            "str_attack_away": _num(df, "strength_attack_away"),
            "str_defence_home": _num(df, "strength_defence_home"),
            "str_defence_away": _num(df, "strength_defence_away"),
            "elo": np.nan,
            "attack_rating": np.nan,
            "defence_rating": np.nan,
        }
    )
    conn.execute("DELETE FROM teams WHERE season = ?", (season,))
    out.to_sql("teams", conn, if_exists="append", index=False)
    conn.commit()
    log.info("teams %s: %d rows via %s", season, len(out), f.path)
    return f.path


def load_players(conn: sqlite3.Connection, season: str) -> str:
    f = sources.current_players(season)
    df = f.data

    pos = _num(df, "element_type").map(POSITION_MAP)

    out = pd.DataFrame(
        {
            "season": season,
            "element": _num(df, "id").astype("Int64"),
            "code": _num(df, "code").astype("Int64"),
            "web_name": _txt(df, "web_name"),
            "first_name": _txt(df, "first_name"),
            "second_name": _txt(df, "second_name"),
            "team_id": _num(df, "team").astype("Int64"),
            "position": pos,
            "now_cost": _num(df, "now_cost") / 10.0,
            "status": _txt(df, "status"),
            "chance_next_round": _num(df, "chance_of_playing_next_round"),
            "news": _txt(df, "news"),
            "selected_by_percent": _num(df, "selected_by_percent"),
            "minutes": _num(df, "minutes"),
            "starts": _num(df, "starts"),
            "total_points": _num(df, "total_points"),
            "points_per_game": _num(df, "points_per_game"),
            "form": _num(df, "form"),
            "ep_next": _num(df, "ep_next"),
            "xg_per90": _num(df, "expected_goals_per_90"),
            "xa_per90": _num(df, "expected_assists_per_90"),
            "xgi_per90": _num(df, "expected_goal_involvements_per_90"),
            "xgc_per90": _num(df, "expected_goals_conceded_per_90"),
            "defcon_per90": _num(df, "defensive_contribution_per_90"),
            "saves_per90": _num(df, "saves_per_90"),
            "starts_per90": _num(df, "starts_per_90"),
            "penalties_order": _num(df, "penalties_order"),
            "corners_fk_order": _num(df, "corners_and_indirect_freekicks_order"),
            "direct_fk_order": _num(df, "direct_freekicks_order"),
        }
    )
    # Managers were removed as a scoring entity; drop defensively if present.
    out = out[out["position"] != "MNG"]

    conn.execute("DELETE FROM players WHERE season = ?", (season,))
    out.to_sql("players", conn, if_exists="append", index=False)
    conn.commit()
    log.info("players %s: %d rows via %s", season, len(out), f.path)
    return f.path


def load_fixtures(conn: sqlite3.Connection, season: str) -> str:
    f = sources.current_fixtures(season)
    df = f.data
    out = pd.DataFrame(
        {
            "season": season,
            "fixture_id": _num(df, "id").astype("Int64"),
            "gw": _num(df, "event").astype("Int64"),
            "kickoff_time": _txt(df, "kickoff_time"),
            "home_team": _num(df, "team_h").astype("Int64"),
            "away_team": _num(df, "team_a").astype("Int64"),
            "home_score": _num(df, "team_h_score").astype("Int64"),
            "away_score": _num(df, "team_a_score").astype("Int64"),
            "finished": _num(df, "finished").fillna(0).astype(int),
            "fdr_home": _num(df, "team_h_difficulty"),
            "fdr_away": _num(df, "team_a_difficulty"),
        }
    )
    conn.execute("DELETE FROM fixtures WHERE season = ?", (season,))
    out.to_sql("fixtures", conn, if_exists="append", index=False)
    conn.commit()
    log.info("fixtures %s: %d rows via %s", season, len(out), f.path)
    return f.path


def load_gameweeks(conn: sqlite3.Connection, season: str) -> str:
    """
    Load merged_gw.csv for a season. Note: this file has `element` (season-scoped)
    but NOT `code`, so we backfill `code` from the players table where available.
    """
    f = sources.season_gameweeks(season)
    df = f.data

    out = pd.DataFrame(
        {
            "season": season,
            "element": _num(df, "element").astype("Int64"),
            "gw": _num(df, "GW").astype("Int64"),
            "fixture_id": _num(df, "fixture").astype("Int64"),
            "opponent_team": _num(df, "opponent_team").astype("Int64"),
            "was_home": _txt(df, "was_home").astype(str).str.lower().map({"true": 1, "false": 0}),
            "kickoff_time": _txt(df, "kickoff_time"),
            "minutes": _num(df, "minutes"),
            "starts": _num(df, "starts"),
            "total_points": _num(df, "total_points"),
            "goals_scored": _num(df, "goals_scored"),
            "assists": _num(df, "assists"),
            "clean_sheets": _num(df, "clean_sheets"),
            "goals_conceded": _num(df, "goals_conceded"),
            "own_goals": _num(df, "own_goals"),
            "penalties_saved": _num(df, "penalties_saved"),
            "penalties_missed": _num(df, "penalties_missed"),
            "yellow_cards": _num(df, "yellow_cards"),
            "red_cards": _num(df, "red_cards"),
            "saves": _num(df, "saves"),
            "bonus": _num(df, "bonus"),
            "bps": _num(df, "bps"),
            "tackles": _num(df, "tackles"),
            "recoveries": _num(df, "recoveries"),
            "clearances_blocks_interceptions": _num(df, "clearances_blocks_interceptions"),
            "defensive_contribution": _num(df, "defensive_contribution"),
            "expected_goals": _num(df, "expected_goals"),
            "expected_assists": _num(df, "expected_assists"),
            "expected_goal_involvements": _num(df, "expected_goal_involvements"),
            "expected_goals_conceded": _num(df, "expected_goals_conceded"),
            "influence": _num(df, "influence"),
            "creativity": _num(df, "creativity"),
            "threat": _num(df, "threat"),
            "ict_index": _num(df, "ict_index"),
            "value": _num(df, "value") / 10.0,
            "selected": _num(df, "selected"),
            "transfers_in": _num(df, "transfers_in"),
            "transfers_out": _num(df, "transfers_out"),
            "fpl_xp": _num(df, "xP"),
        }
    )

    # Backfill stable `code` from that season's players table if we loaded it.
    codes = pd.read_sql(
        "SELECT element, code FROM players WHERE season = ?", conn, params=(season,)
    )
    if len(codes):
        out = out.merge(codes, on="element", how="left")
    else:
        out["code"] = pd.NA

    out = out.dropna(subset=["element", "gw"])
    out["fixture_id"] = out["fixture_id"].fillna(-1)

    # The community CSVs contain genuine duplicate rows (confirmed in 2025-26:
    # 10 exact duplicates across 2 players). Dedupe on the PK rather than crash,
    # and record how many we dropped so validation can surface it.
    before = len(out)
    out = out.drop_duplicates(subset=["season", "element", "gw", "fixture_id"], keep="first")
    dropped = before - len(out)
    if dropped:
        log.warning("player_gw %s: dropped %d duplicate rows from source", season, dropped)
    _DUPES[season] = dropped

    conn.execute("DELETE FROM player_gw WHERE season = ?", (season,))
    out.to_sql("player_gw", conn, if_exists="append", index=False)
    conn.commit()
    log.info("player_gw %s: %d rows via %s (%d dupes removed)", season, len(out), f.path, dropped)
    return f.path


def build_crosswalk(conn: sqlite3.Connection, hist_season: str, cur_season: str) -> dict:
    """
    Map historical players onto current-season elements.

    FPL reassigns `element` ids every season, so `code` is the only safe join key.
    Historical merged_gw.csv lacks `code`, so we resolve it via that season's
    players_raw.csv. Where that's missing we fall back to fuzzy name matching.
    """
    hist = pd.read_sql(
        "SELECT element, code, web_name, first_name, second_name FROM players WHERE season = ?",
        conn,
        params=(hist_season,),
    )
    cur = pd.read_sql(
        "SELECT element, code, web_name, first_name, second_name FROM players WHERE season = ?",
        conn,
        params=(cur_season,),
    )

    if hist.empty or cur.empty:
        return {"matched_by_code": 0, "matched_by_name": 0, "unmatched": 0,
                "note": "one or both player tables empty"}

    by_code = cur.merge(hist[["code", "element"]], on="code", how="inner",
                        suffixes=("_cur", "_hist"))
    matched_codes = set(by_code["code"])

    remaining = cur[~cur["code"].isin(matched_codes)].copy()
    hist_rem = hist[~hist["code"].isin(matched_codes)].copy()

    def norm(s: pd.Series) -> pd.Series:
        return (s.fillna("").astype(str).str.lower()
                 .str.normalize("NFKD").str.encode("ascii", "ignore").str.decode("ascii")
                 .str.replace(r"[^a-z ]", "", regex=True).str.strip())

    remaining["key"] = norm(remaining["first_name"]) + " " + norm(remaining["second_name"])
    hist_rem["key"] = norm(hist_rem["first_name"]) + " " + norm(hist_rem["second_name"])
    by_name = remaining.merge(hist_rem[["key", "element"]], on="key", how="inner",
                              suffixes=("_cur", "_hist"))

    stats = {
        "current_players": int(len(cur)),
        "historical_players": int(len(hist)),
        "matched_by_code": int(len(by_code)),
        "matched_by_name": int(len(by_name)),
        "unmatched_current": int(len(cur) - len(by_code) - len(by_name)),
        "match_rate": round((len(by_code) + len(by_name)) / max(len(cur), 1), 3),
    }
    log.info("crosswalk %s->%s: %s", hist_season, cur_season, stats)
    return stats


# ---------------------------------------------------------------- validation


def check_season_distinctness(conn: sqlite3.Connection, seasons: list[str]) -> dict:
    """
    Assert that different seasons actually contain different data.

    This exists because of a real bug: the live FPL API's bootstrap-static and
    /fixtures/ endpoints are current-season-only and silently ignore any season
    argument. Fetching them while loading history stamped current-season rows
    with a historical season label. The corruption was invisible in every
    downstream metric except one — the cross-season crosswalk match rate went
    UP, to a perfect 1.0, because the current player list was matching itself.

    A failure that makes a quality metric look better is the dangerous kind, so
    it gets an explicit assertion rather than a hopeful eyeball.
    """
    report: dict = {"ok": True, "problems": []}
    sets: dict = {}
    for s in seasons:
        ids = pd.read_sql(
            "SELECT element, code FROM players WHERE season=? ORDER BY element",
            conn, params=(s,))
        sets[s] = (frozenset(ids["code"].dropna().astype(int)), len(ids))

    names = list(sets)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            sa, na = sets[a]
            sb, nb = sets[b]
            if not sa or not sb:
                continue
            if sa == sb:
                report["ok"] = False
                report["problems"].append(
                    f"players for {a} and {b} are IDENTICAL ({na} rows) — "
                    f"a season-agnostic endpoint was used for a historical load")
            else:
                overlap = len(sa & sb) / max(len(sa | sb), 1)
                if overlap > 0.97:
                    report["ok"] = False
                    report["problems"].append(
                        f"players for {a} and {b} overlap {overlap:.1%}, which is "
                        f"implausibly high for different seasons")

    # A season that has PLAYED gameweeks must have finished fixtures with scores.
    # A season that hasn't kicked off yet legitimately has neither, so we key off
    # player_gw rather than flagging every pre-season database.
    for s in seasons:
        played = pd.read_sql(
            "SELECT COUNT(*) n FROM player_gw WHERE season=?",
            conn, params=(s,)).iloc[0]["n"]
        if not played:
            continue
        row = pd.read_sql(
            "SELECT COUNT(*) n, SUM(finished) fin FROM fixtures WHERE season=?",
            conn, params=(s,)).iloc[0]
        if row["n"] and not row["fin"]:
            report["ok"] = False
            report["problems"].append(
                f"{s}: has {int(played)} gameweek rows but 0 finished fixtures — "
                f"fixtures were loaded from a season-agnostic endpoint; the goal "
                f"model cannot be fitted from this")
    return report


def validate(conn: sqlite3.Connection, season: str) -> dict:
    """Return a report the agent must surface before trusting anything downstream."""
    q = lambda s, p=(): pd.read_sql(s, conn, params=p)  # noqa: E731

    report: dict = {"season": season}

    gws = q("SELECT DISTINCT gw FROM player_gw WHERE season = ? ORDER BY gw", (season,))
    report["gameweeks_present"] = gws["gw"].tolist()
    report["n_gameweeks"] = len(gws)
    report["expected_gameweeks"] = 38
    report["complete"] = len(gws) == 38

    counts = q(
        "SELECT COUNT(*) n, COUNT(DISTINCT element) players FROM player_gw WHERE season = ?",
        (season,),
    ).iloc[0]
    report["rows"] = int(counts["n"])
    report["distinct_players"] = int(counts["players"])

    # Sanity checks that catch real corruption in community CSVs.
    bad = q(
        """SELECT
             SUM(CASE WHEN minutes < 0 OR minutes > 120 THEN 1 ELSE 0 END) bad_minutes,
             SUM(CASE WHEN total_points < -10 OR total_points > 40 THEN 1 ELSE 0 END) bad_points,
             SUM(CASE WHEN goals_scored < 0 OR goals_scored > 6 THEN 1 ELSE 0 END) bad_goals,
             SUM(CASE WHEN defensive_contribution IS NULL THEN 1 ELSE 0 END) null_defcon
           FROM player_gw WHERE season = ?""",
        (season,),
    ).iloc[0]
    report["anomalies"] = {k: (int(v) if pd.notna(v) else None) for k, v in bad.items()}

    report["source_duplicate_rows_removed"] = _DUPES.get(season, 0)

    report["null_code_rows"] = int(
        q("SELECT COUNT(*) n FROM player_gw WHERE season = ? AND code IS NULL", (season,)).iloc[0]["n"]
    )

    top = q(
        """SELECT p.web_name, SUM(g.total_points) pts
           FROM player_gw g JOIN players p ON p.season=g.season AND p.element=g.element
           WHERE g.season = ? GROUP BY g.element ORDER BY pts DESC LIMIT 5""",
        (season,),
    )
    report["top_scorers_sanity_check"] = top.to_dict("records")

    return report


# ---------------------------------------------------------------- orchestrator


def run(db_path: str, current: str = CURRENT_SEASON, history: list[str] | None = None,
        strict: bool = True) -> dict:
    history = history or ["2024-25", "2025-26"]
    conn = connect(db_path)
    init_schema(conn)

    provenance: dict = {"connectivity": sources.probe_connectivity()}

    provenance["teams_current"] = load_teams(conn, current)
    provenance["players_current"] = load_players(conn, current)
    provenance["fixtures_current"] = load_fixtures(conn, current)

    for s in history:
        try:
            provenance[f"teams_{s}"] = load_teams(conn, s)
            provenance[f"players_{s}"] = load_players(conn, s)
            # Historical fixtures carry the match results the goal model fits on.
            provenance[f"fixtures_{s}"] = load_fixtures(conn, s)
            provenance[f"gw_{s}"] = load_gameweeks(conn, s)
        except Exception as e:  # noqa: BLE001
            log.error("failed to load history %s: %s", s, e)
            provenance[f"gw_{s}"] = f"FAILED: {e}"

    # Current season may have played gameweeks already.
    try:
        provenance[f"gw_{current}"] = load_gameweeks(conn, current)
    except Exception:  # noqa: BLE001
        log.info("no gameweek data yet for %s (season not started)", current)
        provenance[f"gw_{current}"] = "none (season not started)"

    reports = {s: validate(conn, s) for s in history}
    crosswalks = {s: build_crosswalk(conn, s, current) for s in history}
    distinctness = check_season_distinctness(conn, history + [current])
    if not distinctness["ok"]:
        for msg in distinctness["problems"]:
            log.error("DATA INTEGRITY: %s", msg)

    conn.execute(
        "INSERT OR REPLACE INTO runs (run_id, created_at, season, notes, provenance) VALUES (?,?,?,?,?)",
        (
            "ingest-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S"),
            datetime.now(timezone.utc).isoformat(),
            current,
            "ingest",
            json.dumps(provenance, default=str),
        ),
    )
    conn.commit()
    conn.close()

    result = {"provenance": provenance, "validation": reports,
              "crosswalks": crosswalks, "distinctness": distinctness}
    if not distinctness["ok"] and strict:
        raise RuntimeError(
            "season data integrity check failed: "
            + "; ".join(distinctness["problems"])
            + " — refusing to return a corrupted database")
    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="data/fpl.db")
    ap.add_argument("--current", default=CURRENT_SEASON)
    ap.add_argument("--history", nargs="*", default=["2024-25", "2025-26"])
    ap.add_argument("--no-strict", action="store_true",
                    help="report integrity problems instead of raising")
    a = ap.parse_args()
    print(json.dumps(run(a.db, a.current, a.history, strict=not a.no_strict),
                     indent=2, default=str))
