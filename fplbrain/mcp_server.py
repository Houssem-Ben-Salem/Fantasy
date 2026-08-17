"""
MCP server — the agent's hands.

Design principles, learned the hard way from feeding LLMs raw sports JSON:

  1. NEVER return raw API payloads. bootstrap-static is 1-3 MB; it will eat the
     context window and the model will still hallucinate. Every tool here
     aggregates first and returns a small markdown table.
  2. Every tool caps its own row count. No unbounded results.
  3. run_sql is read-only and row-limited, so the agent can ask precise questions
     without needing a bespoke tool for each one.
  4. Tools return provenance (when the data was fetched, via which path) so the
     agent can tell the user how stale its answer is.

Install:  pip install "mcp[cli]"
Run:      python -m fplbrain.mcp_server
Claude Desktop config: see README.
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

# The SDK renamed FastMCP -> MCPServer in v2.0. Support both so this file
# doesn't break on an SDK upgrade.
try:
    from mcp.server import MCPServer as _Server  # SDK >= 2.0
except ImportError:  # pragma: no cover
    try:
        from mcp.server.fastmcp import FastMCP as _Server  # SDK 1.x
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            'MCP SDK not found. Install with: pip install "mcp[cli]"'
        ) from exc

from . import models, optimize

DB = os.environ.get("FPLBRAIN_DB", "data/fpl.db")
SEASON = os.environ.get("FPLBRAIN_SEASON", "2026-27")
HIST = ["2024-25", "2025-26"]

mcp = _Server("fplbrain", version="1.0.0")

# Cache projections in-process; recomputing per tool call is wasteful.
_PROJ: dict = {"df": None, "built_at": None, "diagnostics": None}


def _conn() -> sqlite3.Connection:
    if not Path(DB).exists():
        raise RuntimeError(f"database not found at {DB} — run `python -m fplbrain.ingest` first")
    return sqlite3.connect(DB)


def _md(df: pd.DataFrame, max_rows: int = 40) -> str:
    """Markdown table. Compact, and honest about truncation."""
    if df.empty:
        return "_no rows_"
    truncated = len(df) > max_rows
    body = df.head(max_rows).to_markdown(index=False, floatfmt=".2f")
    if truncated:
        body += f"\n\n_({len(df)} rows total, showing {max_rows})_"
    return body


def _projections(start_gw: int, n_gw: int = 6, rebuild: bool = False) -> pd.DataFrame:
    key = (start_gw, n_gw)
    if not rebuild and _PROJ["df"] is not None and _PROJ.get("key") == key:
        return _PROJ["df"]
    with _conn() as c:
        df, diag = models.project_horizon(c, SEASON, start_gw, n_gw, HIST)
    _PROJ.update({"df": df, "built_at": datetime.now(timezone.utc), "diagnostics": diag, "key": key})
    return df


def _next_gw() -> int:
    with _conn() as c:
        row = pd.read_sql(
            "SELECT MIN(gw) g FROM fixtures WHERE season=? AND finished=0 AND gw IS NOT NULL",
            c, params=(SEASON,)).iloc[0]
    return int(row["g"]) if pd.notna(row["g"]) else 1


# ------------------------------------------------------------------ tools


@mcp.tool()
def status() -> str:
    """Database freshness, next gameweek, and which data sources were used. Call this first."""
    with _conn() as c:
        runs = pd.read_sql(
            "SELECT run_id, created_at, notes FROM runs ORDER BY created_at DESC LIMIT 3", c)
        counts = pd.read_sql(
            "SELECT season, COUNT(*) rows FROM player_gw GROUP BY season", c)
        players = pd.read_sql(
            "SELECT COUNT(*) n FROM players WHERE season=?", c, params=(SEASON,)).iloc[0]["n"]
        nxt = pd.read_sql(
            """SELECT MIN(gw) gw, MIN(kickoff_time) ko FROM fixtures
               WHERE season=? AND finished=0 AND gw IS NOT NULL""", c, params=(SEASON,)).iloc[0]
        prov_row = pd.read_sql(
            "SELECT provenance FROM runs WHERE notes='ingest' "
            "ORDER BY created_at DESC LIMIT 1", c)

    # Which path the data came from moves the headline numbers, so say it out
    # loud rather than letting the agent report different figures week to week
    # with no explanation. Read from what ingest persisted — status() must not
    # make a network call.
    path_line = "**source path** unknown (no ingest run recorded)"
    if len(prov_row):
        try:
            prov = json.loads(prov_row.iloc[0]["provenance"])
        except (ValueError, TypeError):
            prov = {}
        if prov.get("players_current") == "live":
            path_line = ("**source path** live FPL API — current season is up to date; "
                         "figures are comparable to previous live runs")
        elif prov.get("players_current"):
            path_line = ("**source path** GitHub mirror snapshot (live API was "
                         "unreachable) — the current-season player list is a periodic "
                         "snapshot, so the player count is lower and the crosswalk rate "
                         "higher than a live run. Expected, not a data problem")

    return (
        f"**season** {SEASON}  |  **players** {players}  |  **next GW** {nxt['gw']} "
        f"(first kickoff {nxt['ko']})\n\n"
        f"{path_line}\n\n"
        f"**historical rows**\n{_md(counts)}\n\n**recent runs**\n{_md(runs)}"
    )


@mcp.tool()
def refresh_data(n_gw: int = 6) -> str:
    """Re-pull all sources, rebuild the database, and persist fresh projections."""
    from . import ingest

    t0 = time.time()
    result = ingest.run(DB, SEASON, HIST)

    # Ingest alone is NOT a refresh. It rebuilds the fact tables and appends an
    # `ingest-*` row to `runs`, leaving the newest run one with no projections.
    # Persisted projections are then stale (or absent on a first run) while the
    # in-process _PROJ cache is fresh — two sources of truth that disagree, and
    # v_latest_projections resolves against the stale one. Project and persist
    # here, exactly as scripts/refresh.py does.
    start_gw = _next_gw()
    with _conn() as c:
        proj, diag = models.project_horizon(c, SEASON, start_gw, n_gw, HIST)
        run_id = models.persist_projections(c, proj, SEASON, diag)
        persisted = pd.read_sql(
            "SELECT COUNT(*) n, COUNT(DISTINCT gw) gws FROM projections WHERE run_id=?",
            c, params=(run_id,)).iloc[0]

    _PROJ["df"] = None
    elapsed = time.time() - t0

    val = result["validation"]
    lines = ["Data refreshed.", ""]
    notes: list[str] = []
    for season, r in val.items():
        # Distinguish "nothing is wrong" from "a stat is absent by design".
        # Printing a dict of zeros, or a 27,605 null count filed as an anomaly,
        # invites the agent to open every session reporting a fault on correct
        # data — which trains the user to ignore the check.
        if r.get("anomalies_ok", True):
            health = "no anomalies"
        else:
            bad = {k: v for k, v in r["anomalies"].items()
                   if (v.get("nulls", 0) if isinstance(v, dict) else v)}
            health = f"**ANOMALIES**: {bad}"
        lines.append(
            f"- **{season}**: {r['rows']} rows, {r['n_gameweeks']}/38 GWs, "
            f"{r.get('source_duplicate_rows_removed', 0)} dupes removed, {health}")
        # Collapse columns that share a reason — four lines saying the same
        # thing is noise at the top of every session. The reason text already
        # names the season.
        by_reason: dict[str, list[str]] = {}
        for col, reason in r.get("data_availability", {}).items():
            by_reason.setdefault(reason, []).append(col)
        for reason, cols in by_reason.items():
            notes.append(f"  - {', '.join(f'`{c}`' for c in cols)}: {reason}")
    if notes:
        lines.append("")
        lines.append("_Expected data gaps (informational, not errors):_")
        lines.extend(notes)
    lines.append("")
    lines.append(
        f"**projections**: run_id `{run_id}`, {int(persisted['n'])} rows across "
        f"{int(persisted['gws'])} gameweeks (GW{start_gw}-{start_gw + n_gw - 1})")
    lines.append(f"crosswalk match rates: " + ", ".join(
        f"{k}={v.get('match_rate')}" for k, v in result["crosswalks"].items()))
    lines.append(f"integrity: distinctness ok={result['distinctness']['ok']}")
    lines.append(f"elapsed: {elapsed:.1f}s")
    lines.append(f"source paths: {json.dumps(result['provenance'], default=str)[:600]}")
    return "\n".join(lines)


@mcp.tool()
def top_players(position: str = "ALL", gw_start: int = 0, n_gw: int = 6,
                max_price: float = 99.0, min_price: float = 0.0,
                sort_by: str = "xp", limit: int = 20) -> str:
    """
    Rank players by projected points over a gameweek horizon.

    position: GK, DEF, MID, FWD, or ALL
    sort_by:  xp (total projected), value (xP per £m), or defcon (defensive contribution rate)
    """
    gw_start = gw_start or _next_gw()
    proj = _projections(gw_start, n_gw)
    agg = proj.groupby(["element", "web_name", "position", "team", "now_cost",
                        "selected_by_percent"], as_index=False).agg(
        xp=("exp_points", "sum"), p_start=("p_start", "mean"),
        p_defcon=("p_defcon", "mean"), p_cs=("p_clean_sheet", "mean"))
    agg["value"] = (agg.xp / agg.now_cost).round(2)
    if position.upper() != "ALL":
        agg = agg[agg.position == position.upper()]
    agg = agg[(agg.now_cost <= max_price) & (agg.now_cost >= min_price)]
    key = {"xp": "xp", "value": "value", "defcon": "p_defcon"}.get(sort_by, "xp")
    agg = agg.nlargest(min(limit, 40), key)
    cols = ["web_name", "position", "team", "now_cost", "xp", "value",
            "p_start", "p_defcon", "selected_by_percent"]
    return (f"Projected GW{gw_start}-{gw_start + n_gw - 1}, sorted by {sort_by}\n\n"
            + _md(agg[cols]))


@mcp.tool()
def player_detail(name: str, n_gw: int = 6) -> str:
    """Deep dive on one player: projections, history, DefCon rate, set-piece duties."""
    gw = _next_gw()
    proj = _projections(gw, n_gw)
    match = proj[proj.web_name.str.contains(name, case=False, na=False)]
    if match.empty:
        return f"No player matching '{name}'."
    element = int(match.iloc[0].element)
    rows = match[match.element == element][
        ["gw", "exp_points", "p_start", "p_clean_sheet", "p_defcon",
         "exp_goals", "exp_assists", "exp_bonus", "n_fixtures"]]

    with _conn() as c:
        meta = pd.read_sql(
            """SELECT web_name, position, now_cost, status, news, chance_next_round,
                      selected_by_percent, penalties_order, corners_fk_order, direct_fk_order
               FROM players WHERE season=? AND element=?""", c, params=(SEASON, element))
        hist = pd.read_sql(
            """SELECT season, apps, minutes, points, goals, assists, xg, xa,
                      defcon_actions, defcon_per90, ppg
               FROM v_season_totals WHERE element IN (
                   SELECT element FROM players WHERE code = (
                       SELECT code FROM players WHERE season=? AND element=?))""",
            c, params=(SEASON, element))
        dcr = pd.read_sql(
            """SELECT season, starts_60, defcon_hits, defcon_rate_per_start
               FROM v_defcon_rate WHERE code = (
                   SELECT code FROM players WHERE season=? AND element=?)""",
            c, params=(SEASON, element))

    out = [f"### {meta.iloc[0].web_name}", "", "**Current**", _md(meta), "",
           "**Projection**", _md(rows), ""]
    if not hist.empty:
        out += ["**History**", _md(hist), ""]
    if not dcr.empty:
        out += ["**DefCon hit rate (per 60+ min start)**", _md(dcr)]
    return "\n".join(out)


@mcp.tool()
def build_squad(budget: float = 100.0, gw_start: int = 0, n_gw: int = 6,
                bench_weight: float = 0.12, locked_players: str = "",
                banned_players: str = "") -> str:
    """
    Solve for the optimal 15 from scratch. Use pre-season or on a wildcard.

    locked_players / banned_players: comma-separated names to force in or out.
    """
    gw_start = gw_start or _next_gw()
    proj = _projections(gw_start, n_gw)
    horizon = list(range(gw_start, gw_start + n_gw))

    def resolve(names: str) -> list[int]:
        out = []
        for nm in [x.strip() for x in names.split(",") if x.strip()]:
            m = proj[proj.web_name.str.contains(nm, case=False, na=False)]
            if not m.empty:
                out.append(int(m.iloc[0].element))
        return out

    res = optimize.initial_squad(
        proj, horizon=horizon, budget=budget, bench_weight=bench_weight,
        locked=resolve(locked_players), banned=resolve(banned_players))

    s = res.squad[["web_name", "position", "team", "now_cost", "horizon_xp",
                   "mean_p_start", "in_xi", "is_captain"]]
    return (f"**{res.summary()}**\n\nHorizon GW{min(horizon)}-{max(horizon)}\n\n"
            + _md(s, 15)
            + f"\n\nStarting XI: {', '.join(res.xi.web_name)}"
            + f"\nBench (order): {', '.join(res.bench.web_name)}"
            + f"\nCaptain: {res.captain.get('web_name')} | Vice: {res.vice.get('web_name')}")


@mcp.tool()
def plan_transfers(squad_names: str, bank: float = 0.0, free_transfers: int = 1,
                   n_gw: int = 5, max_transfers: int = 3) -> str:
    """
    Given your current 15, decide transfers over the coming gameweeks.

    Explicitly prices each extra transfer at -4 and only recommends a hit when the
    horizon gain beats it. squad_names: comma-separated list of 15 player names.
    """
    gw = _next_gw()
    proj = _projections(gw, n_gw)
    ids, unresolved = [], []
    for nm in [x.strip() for x in squad_names.split(",") if x.strip()]:
        m = proj[proj.web_name.str.contains(nm, case=False, na=False)]
        if m.empty:
            unresolved.append(nm)
        else:
            ids.append(int(m.iloc[0].element))
    if unresolved:
        return f"Could not resolve: {unresolved}. Use exact FPL web names."

    res = optimize.transfer_plan(
        proj, ids, horizon=list(range(gw, gw + n_gw)), bank=bank,
        free_transfers=free_transfers, max_transfers=max_transfers)

    if not res.transfers:
        return (f"**No transfer beats rolling.** Objective {res.objective:.2f} xP.\n"
                f"Keep the squad and bank the free transfer.\n\n"
                f"Captain: {res.captain.get('web_name')}")

    lines = [f"**{len(res.transfers)} transfer(s), {res.hits_taken} hit(s)**", ""]
    for t in res.transfers:
        lines.append(f"- OUT {t['out']} ({t['out_xp']} xP) → IN {t['in']} ({t['in_xp']} xP) "
                     f"| net **{t['xp_gain']:+.2f}** over GW{gw}-{gw + n_gw - 1}")
    net = sum(t["xp_gain"] for t in res.transfers) - res.hits_taken * 4
    lines += ["", f"Net after hits: **{net:+.2f} xP**", f"Bank after: £{res.bank:.1f}m",
              f"Captain: {res.captain.get('web_name')}"]
    return "\n".join(lines)


@mcp.tool()
def chip_advice(squad_names: str, gw: int = 0) -> str:
    """Quantify Bench Boost / Triple Captain / Free Hit value for a gameweek."""
    gw = gw or _next_gw()
    proj = _projections(gw, 6)
    ids = []
    for nm in [x.strip() for x in squad_names.split(",") if x.strip()]:
        m = proj[proj.web_name.str.contains(nm, case=False, na=False)]
        if not m.empty:
            ids.append(int(m.iloc[0].element))
    ev = optimize.evaluate_chips(proj, ids, gw)
    return json.dumps(ev, indent=2, default=str)


@mcp.tool()
def fixture_outlook(n_gw: int = 6, limit: int = 20) -> str:
    """
    Model-based fixture difficulty per club, from fitted attack/defence ratings
    rather than the official FDR. Lower opponent strength = greener run.
    """
    gw = _next_gw()
    with _conn() as c:
        gm = models.fit_goal_model(c, HIST + [SEASON])
        fx = pd.read_sql(
            """SELECT f.gw, th.short_name h, ta.short_name a
               FROM fixtures f
               JOIN teams th ON th.season=f.season AND th.team_id=f.home_team
               JOIN teams ta ON ta.season=f.season AND ta.team_id=f.away_team
               WHERE f.season=? AND f.gw>=? AND f.gw<?""",
            c, params=(SEASON, gw, gw + n_gw))
    rows = []
    for _, r in fx.iterrows():
        lh, la = gm.expected_goals(r.h, r.a)
        rows.append({"team": r.h, "gw": r.gw, "opp": r.a, "home": 1,
                     "xg_for": round(lh, 2), "xg_against": round(la, 2)})
        rows.append({"team": r.a, "gw": r.gw, "opp": r.h, "home": 0,
                     "xg_for": round(la, 2), "xg_against": round(lh, 2)})
    d = pd.DataFrame(rows)
    agg = d.groupby("team", as_index=False).agg(
        fixtures=("gw", "count"), xg_for=("xg_for", "sum"), xg_against=("xg_against", "sum"))
    agg["attack_rank"] = agg.xg_for.rank(ascending=False).astype(int)
    agg["defence_rank"] = agg.xg_against.rank(ascending=True).astype(int)
    return (f"Modelled fixture outlook GW{gw}-{gw + n_gw - 1}\n\n"
            + _md(agg.sort_values("xg_against")[
                ["team", "fixtures", "xg_for", "xg_against", "attack_rank", "defence_rank"]], limit))


@mcp.tool()
def run_sql(query: str, limit: int = 50) -> str:
    """
    Read-only SQL against the database. Use for anything the other tools don't cover.

    Tables: players, teams, fixtures, player_gw, projections, runs, squad_state
    Views:  v_season_totals, v_defcon_rate, v_form6, v_latest_projections
    SELECT statements only.
    """
    q = query.strip().rstrip(";")
    lowered = q.lower()
    if not lowered.startswith(("select", "with")):
        return "Only SELECT / WITH queries are permitted."
    for kw in ("insert", "update", "delete", "drop", "alter", "attach", "pragma", "create"):
        if f" {kw} " in f" {lowered} ":
            return f"Blocked keyword: {kw}"
    if " limit " not in lowered:
        q += f" LIMIT {min(limit, 200)}"
    try:
        with _conn() as c:
            df = pd.read_sql(q, c)
        return _md(df, min(limit, 60))
    except Exception as e:  # noqa: BLE001
        return f"SQL error: {e}"


@mcp.tool()
def save_state(gw: int, squad_json: str, bank: float, free_transfers: int,
               chips_remaining: str, notes: str = "") -> str:
    """
    Persist squad state so next week's session starts from truth, not from a
    pasted summary that may have drifted.

    squad_json: [{"element": 123, "purchase_price": 7.5, "selling_price": 7.5}, ...]
    """
    with _conn() as c:
        c.execute(
            """INSERT OR REPLACE INTO squad_state
               (gw, recorded_at, bank, free_transfers, chips_remaining, squad, notes)
               VALUES (?,?,?,?,?,?,?)""",
            (gw, datetime.now(timezone.utc).isoformat(), bank, free_transfers,
             chips_remaining, squad_json, notes))
        c.commit()
    return f"State saved for GW{gw}."


@mcp.tool()
def load_state(gw: int = 0) -> str:
    """Load the most recent saved squad state (or a specific gameweek)."""
    with _conn() as c:
        if gw:
            df = pd.read_sql("SELECT * FROM squad_state WHERE gw=?", c, params=(gw,))
        else:
            df = pd.read_sql("SELECT * FROM squad_state ORDER BY gw DESC LIMIT 1", c)
    return _md(df) if not df.empty else "No saved state."


if __name__ == "__main__":
    mcp.run()
