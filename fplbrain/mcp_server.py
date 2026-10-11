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

from . import entry, models, news, optimize, scorecard

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
    """The next gameweek whose deadline has not passed (see models.next_gameweek)."""
    with _conn() as c:
        return models.next_gameweek(c, SEASON)


def _saved_squad() -> dict | None:
    """The most recent squad_state row, parsed: elements, selling prices, bank,
    free transfers. None when nothing has been saved."""
    with _conn() as c:
        df = pd.read_sql("SELECT * FROM squad_state ORDER BY gw DESC, recorded_at DESC "
                         "LIMIT 1", c)
    if df.empty:
        return None
    r = df.iloc[0]
    squad = json.loads(r["squad"])
    return {"gw": int(r["gw"]), "elements": [int(p["element"]) for p in squad],
            "selling_prices": {int(p["element"]): float(p["selling_price"])
                               for p in squad if p.get("selling_price") is not None},
            "bank": float(r["bank"]), "free_transfers": int(r["free_transfers"])}


# ------------------------------------------------------------------ name lookup
#
# Every tool that takes player names resolves them HERE, and nowhere else.
#
# The original pattern — `web_name.str.contains(nm)` then `.iloc[0]` — fails
# silently in three separate ways, and all three are live in the 2026-27 data:
#
#   1. AMBIGUITY. Two players are named `Palmer` (Cole Palmer, CHE MID £9.5m,
#      and a GK at Ipswich, £4.0m). `.iloc[0]` picks one without saying which,
#      so plan_transfers would optimise around the wrong player and return a
#      perfectly plausible squad. There is no signal that anything went wrong.
#   2. SUBSTRING OVERREACH. `Raya` also matches `Rayan` (BOU). Exact matches
#      must therefore win over substring, or a correct full name stays
#      ambiguous forever.
#   3. SILENT DROPS. build_squad and chip_advice appended only when a match was
#      found, so a typo removed a player from the list. A 14-player squad
#      handed to the optimiser either fails as infeasible or, worse, solves.
#
# Raising with the candidates listed — not just a count — lets an agent fix the
# call on its next attempt instead of guessing again.


class PlayerLookupError(ValueError):
    """A query did not resolve to exactly one player."""


def _describe(rows: pd.DataFrame) -> str:
    return "; ".join(
        f"{r.web_name} ({r.team}, {r.position}, £{r.now_cost}m, id={int(r.element)})"
        for r in rows.drop_duplicates("element").itertuples())


def _resolve_player(proj: pd.DataFrame, query: str) -> int:
    """Resolve one query to a single element id, or raise with the candidates."""
    q = str(query).strip()
    if not q:
        raise PlayerLookupError("empty player name")

    pool = proj.drop_duplicates("element")

    # A bare number is an element id — an unambiguous channel that bypasses
    # name matching entirely. Use it for save_state and for anything scripted.
    if q.isdigit():
        hit = pool[pool.element == int(q)]
        if hit.empty:
            raise PlayerLookupError(f"no player with element id {q}")
        return int(hit.iloc[0].element)

    # regex=False matters: names contain regex metacharacters ("B.Fernandes"),
    # where an unescaped '.' would match any character.
    exact = pool[pool.web_name.str.lower() == q.lower()]
    if len(exact) == 1:
        return int(exact.iloc[0].element)
    if len(exact) > 1:
        raise PlayerLookupError(
            f"'{q}' is ambiguous — {len(exact)} players share that exact name: "
            f"{_describe(exact)}. Pass the element id to disambiguate.")

    part = pool[pool.web_name.str.contains(q, case=False, na=False, regex=False)]
    if part.empty:
        raise PlayerLookupError(f"no player matching '{q}'")
    if len(part) > 1:
        raise PlayerLookupError(
            f"'{q}' is ambiguous — {len(part)} matches: {_describe(part)}. "
            f"Use the exact web_name or the element id.")
    return int(part.iloc[0].element)


def _resolve_players(proj: pd.DataFrame, names: str, expect: int | None = None) -> list[int]:
    """Resolve a comma-separated list. Reports EVERY problem, not just the first."""
    queries = [x.strip() for x in str(names).split(",") if x.strip()]
    if not queries:
        raise PlayerLookupError("no player names supplied")

    ids, problems = [], []
    for q in queries:
        try:
            ids.append(_resolve_player(proj, q))
        except PlayerLookupError as e:
            problems.append(str(e))
    if problems:
        raise PlayerLookupError(
            f"{len(problems)} of {len(queries)} names did not resolve:\n  - "
            + "\n  - ".join(problems))

    # The same player named twice is another route to a short squad.
    dupes = sorted({i for i in ids if ids.count(i) > 1})
    if dupes:
        named = _describe(proj[proj.element.isin(dupes)])
        raise PlayerLookupError(f"the same player was listed more than once: {named}")

    if expect is not None and len(ids) != expect:
        raise PlayerLookupError(
            f"expected {expect} players, got {len(ids)} — check the list is complete")
    return ids


def _season_history(conn: sqlite3.Connection, season: str, element: int) -> pd.DataFrame:
    """
    One player's season totals across every season, joined on the stable `code`.

    NEVER join cross-season on `element`: it is reassigned every season. The
    original query did `WHERE element IN (SELECT element ... WHERE code = X)`, which
    collects this player's element ids from each season and then matches those
    NUMBERS in every season — so it also pulled in whoever else held that id in the
    other seasons. For a backup goalkeeper it returned six different players and
    credited him with 18 goals and 18.58 xG in 2024-25 (those were Wissa's). Every
    player_detail history table was contaminated the same way, while the DefCon
    query beside it already keyed on `code` correctly.
    """
    return pd.read_sql(
        """SELECT season, apps, minutes, points, goals, assists, xg, xa,
                  defcon_actions, defcon_per90, ppg
           FROM v_season_totals
           WHERE code = (SELECT code FROM players WHERE season=? AND element=?)
           ORDER BY season""",
        conn, params=(season, element))


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
    try:
        element = _resolve_player(proj, name)
    except PlayerLookupError as e:
        return f"**Could not resolve '{name}'.**\n{e}"
    match = proj[proj.element == element]
    rows = match[match.element == element][
        ["gw", "exp_points", "p_start", "p_clean_sheet", "p_defcon",
         "exp_goals", "exp_assists", "exp_bonus", "n_fixtures"]]

    with _conn() as c:
        meta = pd.read_sql(
            """SELECT web_name, position, now_cost, status, news, chance_next_round,
                      selected_by_percent, penalties_order, corners_fk_order, direct_fk_order
               FROM players WHERE season=? AND element=?""", c, params=(SEASON, element))
        hist = _season_history(c, SEASON, element)
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

    try:
        locked = _resolve_players(proj, locked_players) if locked_players.strip() else []
        banned = _resolve_players(proj, banned_players) if banned_players.strip() else []
    except PlayerLookupError as e:
        return f"**Could not resolve player names.**\n{e}"

    res = optimize.initial_squad(
        proj, horizon=horizon, budget=budget, bench_weight=bench_weight,
        locked=locked, banned=banned)

    s = res.squad[["web_name", "position", "team", "now_cost", "horizon_xp",
                   "mean_p_start", "in_xi", "is_captain"]]
    return (f"**{res.summary()}**\n\nHorizon GW{min(horizon)}-{max(horizon)}\n\n"
            + _md(s, 15)
            + f"\n\nStarting XI: {', '.join(res.xi.web_name)}"
            + f"\nBench (order): {', '.join(res.bench.web_name)}"
            + f"\nCaptain: {res.captain.get('web_name')} | Vice: {res.vice.get('web_name')}")


@mcp.tool()
def plan_transfers(squad_names: str = "", bank: float | None = None,
                   free_transfers: int | None = None,
                   n_gw: int = 5, max_transfers: int = 3, unavailable: str = "") -> str:
    """
    Given your current 15, decide transfers over the coming gameweeks.

    Explicitly prices each extra transfer at -4 and only recommends a hit when the
    horizon gain beats it.

    squad_names: leave EMPTY to use the squad saved by sync_squad() — real squad,
      real bank, derived selling prices and free transfers. Typing 15 names is the
      fallback, and hand-entered squads have been wrong before (a wrong backup
      keeper for five weeks). bank / free_transfers override the saved values.
    unavailable: injury scenarios, e.g. "Gibbs-White:2, 229:5" — each player
      (name or element id) projects zero for that many gameweeks from the next
      one (no number = the whole horizon). Run one plan per scenario and present
      the decision as "if X is out -> A; if fit -> B".
    """
    gw = _next_gw()
    proj = _projections(gw, n_gw)
    selling, note = None, ""
    if unavailable.strip():
        proj = proj.copy()
        out = []
        for item in unavailable.split(","):
            who, _, weeks = item.strip().partition(":")
            try:
                el = _resolve_player(proj, who.strip())
            except PlayerLookupError as e:
                return f"**Could not resolve an unavailable player.**\n{e}"
            n = int(weeks) if weeks.strip() else n_gw
            hit = (proj.element == el) & proj.gw.between(gw, gw + n - 1)
            proj.loc[hit, ["exp_points", "p_start"]] = [0.0, 0.0]
            name = proj.loc[proj.element == el, "web_name"].iloc[0]
            out.append(f"{name} ({el}) out GW{gw}-{gw + n - 1}")
        note = "Scenario: " + "; ".join(out) + ".\n\n"
    if not squad_names.strip():
        st = _saved_squad()
        if st is None:
            return "No saved squad. Call sync_squad() first, or pass squad_names."
        ids, selling = st["elements"], st["selling_prices"] or None
        bank = st["bank"] if bank is None else bank
        free_transfers = st["free_transfers"] if free_transfers is None else free_transfers
        note += (f"Squad: saved state for GW{st['gw']}"
                + ("" if st["gw"] == gw else f" — **planning GW{gw}; re-run sync_squad()**")
                + ".\n\n")
    else:
        # expect=15: a squad short by one silently changes the problem the solver
        # is given, and it will happily return an answer for the wrong squad.
        try:
            ids = _resolve_players(proj, squad_names, expect=15)
        except PlayerLookupError as e:
            return f"**Could not resolve your squad.**\n{e}"
    bank = 0.0 if bank is None else bank
    free_transfers = 1 if free_transfers is None else free_transfers

    res = optimize.transfer_plan(
        proj, ids, horizon=list(range(gw, gw + n_gw)), bank=bank,
        free_transfers=free_transfers, max_transfers=max_transfers,
        selling_prices=selling)

    if not res.transfers:
        return note + (f"**No transfer beats rolling.** Objective {res.objective:.2f} xP.\n"
                f"Keep the squad and bank the free transfer.\n\n"
                f"Captain: {res.captain.get('web_name')}")

    # The decision number is the solver's gain over rolling: decayed, bench at
    # bench_weight, hits priced. Summing per-pair raw xP instead counted a
    # benched player's replacement in full — GW7 2026-27 printed +16.80 for a
    # plan worth +7.0 — and the hit policy in prompts/AGENT.md is judged on it.
    roll = optimize.transfer_plan(
        proj, ids, horizon=list(range(gw, gw + n_gw)), bank=bank,
        free_transfers=free_transfers, max_transfers=0, selling_prices=selling)
    gain = res.objective - roll.objective

    lines = [note + f"**{len(res.transfers)} transfer(s), {res.hits_taken} hit(s)** — "
             f"gain over rolling **{gain:+.2f} xP** (GW{gw}-{gw + n_gw - 1}, after hits)", ""]
    for t in res.transfers:
        lines.append(f"- OUT {t['out']} → IN {t['in']}  "
                     f"(raw GW{gw}-{gw + n_gw - 1} xP {t['out_xp']} → {t['in_xp']})")
    lines += ["", "Gain over rolling is the number to judge a hit by; the raw per-player "
              "xP ignores bench weighting and the horizon decay.",
              f"Bank after: £{res.bank:.1f}m", f"Captain: {res.captain.get('web_name')}"]
    return "\n".join(lines)


@mcp.tool()
def chip_advice(squad_names: str = "", gw: int = 0) -> str:
    """Quantify Bench Boost / Triple Captain / Free Hit value for a gameweek.
    squad_names: leave EMPTY to use the squad saved by sync_squad()."""
    gw = gw or _next_gw()
    proj = _projections(gw, 6)
    if not squad_names.strip():
        st = _saved_squad()
        if st is None:
            return "No saved squad. Call sync_squad() first, or pass squad_names."
        ids = st["elements"]
    else:
        # Bench Boost value is a sum over the squad, so a dropped name understates
        # the chip rather than failing — silently, and in the direction of not
        # playing it. expect=15 makes that impossible.
        try:
            ids = _resolve_players(proj, squad_names, expect=15)
        except PlayerLookupError as e:
            return f"**Could not resolve your squad.**\n{e}"
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


@mcp.tool()
def note_decision(note: str, gw: int = 0) -> str:
    """
    Append a dated note to the saved squad state for a gameweek (default: the
    next open one) without touching the squad, bank or transfers. Use it for
    every judgement call that departs from the model, with the reason — those
    notes are what the end-of-season review scores judgement by, and the next
    sync_squad() keeps them.
    """
    gw = gw or _next_gw()
    with _conn() as c:
        row = c.execute("SELECT notes FROM squad_state WHERE gw=?", (gw,)).fetchone()
        if row is None:
            return f"No saved state for GW{gw}. Call sync_squad() first."
        stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        notes = ((row[0] or "").rstrip() + f"\n\n[{stamp}] {note.strip()}").strip()
        c.execute("UPDATE squad_state SET notes=? WHERE gw=?", (notes, gw))
        c.commit()
    return f"Noted for GW{gw}."


@mcp.tool()
def model_scorecard(last_n: int = 8, include_ci: bool = True) -> str:
    """
    How the projections persisted BEFORE each deadline did against what happened,
    next to a naive "points so far" baseline — the weekly model review. Scored
    only on runs that could not be changed after the fact (local refreshes, and
    CI's Friday exports when include_ci). spearman_start, among players who
    actually started, isolates the points model from minutes: it is the open
    question, and the pooled interval says how far it is from being decided.
    """
    with _conn() as c:
        t, pooled = scorecard.score(c, SEASON, include_ci=include_ci)
    if t.empty:
        return "No finished gameweek has a pre-deadline run yet."
    cols = ["gw", "source", "run", "model", "n", "spearman_all", "naive_spearman_all",
            "spearman_start", "naive_spearman_start", "brier", "naive_brier"]
    recent = t[t.gw > t.gw.max() - last_n][cols].round(3)
    return ("**Per gameweek** — the last run before each deadline\n\n"
            + _md(recent, max_rows=2 * last_n)
            + "\n\n**Pooled, model minus naive** — bootstrap over (team, gameweek)\n\n"
            + _md(pooled.round(3), max_rows=20)
            + "\n\nUntil spearman_start's interval excludes zero, the points model has not "
              "been shown to beat 'points so far'; minutes carries spearman_all.")


def _players_frame() -> pd.DataFrame:
    """Current players in the shape _resolve_player and _describe expect."""
    with _conn() as c:
        return pd.read_sql(
            """SELECT p.element, p.web_name, t.short_name AS team, p.position, p.now_cost
               FROM players p JOIN teams t ON t.season=p.season AND t.team_id=p.team_id
               WHERE p.season=?""", c, params=(SEASON,))


@mcp.tool()
def record_news(claims_json: str, recorded_by: str = "news-scout") -> str:
    """
    Log team-news claims so they can be scored later. All or nothing: if any
    claim is invalid, none is recorded and every problem is listed.

    claims_json: a JSON list of objects —
      player        name or element id (resolved like every other tool)
      claim         out | doubtful | available | starts | benched
      confidence    confirmed (club/manager/FPL flag) | rumoured (journalist, predicted line-up)
      source_type   presser | club | journalist | predicted_lineup | fpl_flag | newsletter | other
      source_url    required unless source_type is fpl_flag
      source_name, published_at (ISO date), quote, chance (0-100), gw (default: next open)

    Claims inform the human-facing plan only, until news_scorecard() shows they
    beat persistence where they disagree with it (fplbrain/news.py).
    """
    try:
        raw = json.loads(claims_json)
    except json.JSONDecodeError as e:
        return f"**Nothing recorded.** Not valid JSON: {e}"
    raw = [raw] if isinstance(raw, dict) else raw
    pool, gw0, errors, claims = _players_frame(), _next_gw(), [], []
    for i, c in enumerate(raw):
        try:
            el = _resolve_player(pool, str(c.get("player", "")))
        except PlayerLookupError as e:
            errors.append(f"claim {i + 1}: {e}")
            continue
        try:
            gw = int(c.get("gw") or gw0)
        except (TypeError, ValueError):
            gw = c.get("gw")
        claims.append({**c, "element": el, "gw": gw,
                       "player": pool.loc[pool.element == el, "web_name"].iloc[0]})
    if not errors:
        try:
            with _conn() as conn:
                news.record(conn, SEASON, claims, recorded_by)
        except news.ClaimError as e:
            errors = list(e.args[0])
    if errors:
        return ("**Nothing recorded.** Fix these and resend the whole batch:\n"
                + "\n".join(f"- {x}" for x in errors))
    df = pd.DataFrame(claims)[["player", "gw", "claim", "confidence", "source_type"]]
    return f"Recorded {len(claims)} claim(s).\n\n" + _md(df)


@mcp.tool()
def news_for(gw: int = 0, squad_only: bool = True) -> str:
    """The latest recorded claim per player for a gameweek (default: the next
    open one), for the synced squad unless squad_only is False."""
    gw = gw or _next_gw()
    with _conn() as c:
        d = news.latest(c, SEASON, gw)
    st = _saved_squad() if squad_only else None
    if st:
        d = d[d.element.isin(st["elements"])]
    if d.empty:
        return f"No news recorded for GW{gw}{' for your squad' if st else ''}."
    cols = ["player", "claim", "chance", "confidence", "source_type", "source_name",
            "published_at", "recorded_at", "source_url"]
    return f"Latest claim per player for GW{gw}:\n\n" + _md(d[cols], max_rows=40)


@mcp.tool()
def news_scorecard() -> str:
    """How recorded claims did against what happened — the latest claim per player
    logged before each deadline — and the verdict against persistence."""
    with _conn() as c:
        res = news.score(c, SEASON, scorecard.deadlines(c, SEASON))
    if res["rows"].empty:
        return ("No claim was recorded before a finished gameweek's deadline yet, so there is "
                f"nothing to score. {res['verdict']}")
    dis = res["disagree"]
    db = res["doubtful"]
    return ("**Claims scored on what happened**\n\n" + _md(res["by_claim"].round(3))
            + f"\n\n**Where claims disagreed with persistence:** {dis['correct']}/{dis['n']} right, "
              f"95% [{dis['lo']:.2f}, {dis['hi']:.2f}]."
            + (f" Doubtful-with-chance Brier {db['brier']:.3f} over {db['n']}." if db["n"] else "")
            + f"\n\n{res['verdict']}")


def _last_synced_entry() -> int:
    """The entry id the most recent API sync recorded in squad_state notes, so
    the team id never has to be hand-edited into the MCP server config."""
    import re
    with _conn() as c:
        rows = c.execute("SELECT notes FROM squad_state WHERE notes LIKE ? "
                         "ORDER BY recorded_at DESC LIMIT 1",
                         (entry.SYNC_PREFIX + ", entry %",)).fetchall()
    m = re.match(re.escape(entry.SYNC_PREFIX) + r", entry (\d+)", rows[0][0]) if rows else None
    return int(m.group(1)) if m else 0


@mcp.tool()
def sync_squad(entry_id: int = 0, save: bool = True) -> str:
    """
    Read the manager's real squad, bank, free transfers and chips from the public
    FPL API (no login) and save it as squad_state for the gameweek being planned.
    Call this at the start of every session instead of trusting a pasted squad.

    entry_id: the FPL team id; 0 uses FPLBRAIN_ENTRY_ID, else the entry the last
      sync recorded — so after one sync, no argument is needed.
    Selling prices and free transfers are DERIVED from the transfer history
    (fplbrain/entry.py) and labelled so. Transfers already made for the next
    deadline are invisible until it passes.
    """
    eid = (entry_id or int(os.environ.get("FPLBRAIN_ENTRY_ID", "0") or 0)
           or _last_synced_entry())
    if not eid:
        return ("No entry id: pass entry_id once (later syncs remember it), or set "
                "FPLBRAIN_ENTRY_ID in the MCP server env.")
    try:
        s = entry.entry_state(eid)
    except Exception as exc:  # network, 404, API shape change
        return f"FPL API unavailable for entry {eid}: {exc}. Fall back to load_state()."
    if save:
        with _conn() as c:
            entry.save(c, s)
    df = pd.DataFrame(s["squad"])[["name", "pos", "team", "now_cost", "purchase_price",
                                   "selling_price", "acquired", "captain", "vice"]]
    head = (f"**{s['entry_name']}** (entry {eid}), planning GW{s['gw_next']} — "
            f"bank £{s['bank']:.1f}m · free transfers {s['free_transfers']} (derived) · "
            f"selling value £{s['selling_value']:.1f}m (derived) · chips: "
            f"{', '.join(s['chips_remaining']) or 'none'}"
            + (f" · saved as squad_state GW{s['gw_next']}" if save else ""))
    return head + "\n\n" + _md(df, max_rows=15)


if __name__ == "__main__":
    mcp.run()
