#!/usr/bin/env python3
"""
Build the data files behind the fplbrain Tracker (a Claude dashboard artifact).

The tracker draws every number from one of these files; Claude uploads them to
the artifact after running this script and nothing here publishes anything.
Each file comes from the same code the MCP tools use — the synced squad
(fplbrain/entry.py), the persisted projections, the xG-weighted goal model, the
transfer optimiser, the news log and the prospective scorecard — so the tracker
and the tools can never disagree about a number.

Output: data/tracker/*.json (git-ignored: it holds the manager's squad).

    python scripts/build_tracker.py [--entry ID] [--out data/tracker]

The entry id defaults to FPLBRAIN_ENTRY_ID, else the entry the last API sync
recorded in squad_state.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fplbrain import entry, models, news, optimize, scorecard  # noqa: E402

API = "https://fantasy.premierleague.com/api"
SEASON = os.environ.get("FPLBRAIN_SEASON", "2026-27")
HORIZON = 6
FIRST_HALF_LAST_GW = 19
CHIP_NAMES = {"wildcard": "Wildcard", "free_hit": "Free Hit", "bench_boost": "Bench Boost",
              "triple_captain": "Triple Captain"}


def _get(path: str):
    r = requests.get(f"{API}/{path}", timeout=30, headers={"User-Agent": "fplbrain tracker"})
    r.raise_for_status()
    return r.json()


def _entry_id(conn, given: int) -> int:
    if given:
        return given
    if os.environ.get("FPLBRAIN_ENTRY_ID"):
        return int(os.environ["FPLBRAIN_ENTRY_ID"])
    row = conn.execute("SELECT notes FROM squad_state WHERE notes LIKE ? ORDER BY recorded_at DESC "
                       "LIMIT 1", (entry.SYNC_PREFIX + ", entry %",)).fetchone()
    m = re.match(re.escape(entry.SYNC_PREFIX) + r", entry (\d+)", row[0]) if row else None
    if not m:
        sys.exit("no entry id: pass --entry, set FPLBRAIN_ENTRY_ID, or sync_squad() once")
    return int(m.group(1))


def _round(df: pd.DataFrame, n: int = 2) -> list[dict]:
    """Rows as JSON-ready dicts: floats rounded, NaN as null."""
    out = df.copy()
    for c in out.select_dtypes("number").columns:
        out[c] = out[c].round(n)
    return json.loads(out.to_json(orient="records"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--entry", type=int, default=0)
    ap.add_argument("--db", default=os.environ.get("FPLBRAIN_DB", str(ROOT / "data" / "fpl.db")))
    ap.add_argument("--out", default=str(ROOT / "data" / "tracker"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(a.db)
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    files: dict[str, list[dict]] = {}

    # ---- the real squad, synced now
    eid = _entry_id(conn, a.entry)
    st = entry.entry_state(eid)
    entry.save(conn, st)
    gw = st["gw_next"]
    hz = list(range(gw, gw + HORIZON))
    sq = pd.DataFrame(st["squad"])
    ids = sq.element.tolist()
    sell = dict(zip(sq.element, sq.selling_price))

    boot = _get("bootstrap-static/")
    events = pd.DataFrame(boot["events"])
    team_name = {t["id"]: t["short_name"] for t in boot["teams"]}
    deadline = events.loc[events.id == gw, "deadline_time"].iloc[0]

    # ---- projections, fixtures and the goal model for the coming gameweeks
    proj, _ = models.project_horizon(conn, SEASON, gw, HORIZON, ["2024-25", "2025-26"])
    fx = pd.read_sql("""SELECT f.gw, th.short_name h, ta.short_name a FROM fixtures f
                        JOIN teams th ON th.season=f.season AND th.team_id=f.home_team
                        JOIN teams ta ON ta.season=f.season AND ta.team_id=f.away_team
                        WHERE f.season=? AND f.gw BETWEEN ? AND ?""",
                     conn, params=(SEASON, hz[0], hz[-1]))
    legs = pd.concat([fx.rename(columns={"h": "team", "a": "opp"}).assign(venue="H"),
                      fx.rename(columns={"a": "team", "h": "opp"}).assign(venue="A")])
    gm = models.fit_goal_model(conn, ["2024-25", "2025-26", SEASON])
    xg = [gm.expected_goals(t, o) if v == "H" else gm.expected_goals(o, t)[::-1]
          for t, o, v in zip(legs.team, legs.opp, legs.venue)]
    legs["xg_for"] = [f for f, _ in xg]
    legs["xg_against"] = [g for _, g in xg]
    legs["cs"] = [models.np.exp(-g) for _, g in xg]
    files["fixtures"] = _round(legs.sort_values(["team", "gw"])[
        ["team", "gw", "opp", "venue", "xg_for", "xg_against", "cs"]], 3)

    flags = pd.read_sql("SELECT element, status, chance_next_round chance, news FROM players "
                        "WHERE season=?", conn, params=(SEASON,))
    # One label per club and gameweek, so a double gameweek can't duplicate a player's row.
    opp = (legs.assign(opp=legs.opp + " (" + legs.venue + ")")
               .groupby(["team", "gw"], as_index=False).opp.agg(" + ".join))
    p = proj[proj.element.isin(ids)].merge(opp, on=["team", "gw"], how="left")
    p["opp"] = p.opp.fillna("—")
    files["squad_xp"] = _round(p[["element", "web_name", "position", "gw", "opp", "exp_points"]]
                               .rename(columns={"web_name": "player", "position": "pos",
                                                "exp_points": "xp"}))

    nxt = proj[(proj.gw == gw) & proj.element.isin(ids)].copy()
    xi = optimize.best_legal_xi(nxt, "exp_points")
    tot = p.groupby("element").exp_points.sum().rename("xp_6gw")
    s = (sq.merge(nxt[["element", "p_start", "exp_points", "selected_by_percent"]], on="element")
           .merge(tot, on="element").merge(flags, on="element", how="left"))
    s["in_model_xi"] = s.element.isin(xi.element)
    cap = xi.sort_values("exp_points", ascending=False).iloc[0].element
    s["model_captain"] = s.element == cap
    order = {"GK": 0, "DEF": 1, "MID": 2, "FWD": 3}
    s = s.assign(_o=s.pos.map(order)).sort_values(["_o", "exp_points"], ascending=[True, False])
    files["squad"] = _round(s.rename(columns={
        "name": "player", "now_cost": "price", "selling_price": "sell", "exp_points": "xp_next",
        "selected_by_percent": "owned"})[
        ["element", "player", "pos", "team", "price", "sell", "status", "chance", "news",
         "p_start", "xp_next", "xp_6gw", "owned", "in_model_xi", "model_captain"]])

    # ---- the plan, under each injury scenario the flags and the news log raise
    claims = news.latest(conn, SEASON, gw)
    doubt = sorted(set(s.loc[s.status.isin(["d", "i", "s", "u", "n"]), "element"])
                   | set(claims.loc[claims.claim.isin(["out", "doubtful"]) & claims.element.isin(ids),
                                    "element"]))
    names = dict(zip(s["element"], s["name"]))
    club = dict(zip(proj.element, proj.team))
    ft = int(st["free_transfers"])

    def plan(out_ids: list[int], weeks: int) -> dict:
        pr = proj.copy()
        pr.loc[pr.element.isin(out_ids) & pr.gw.between(gw, gw + weeks - 1),
               ["exp_points", "p_start"]] = [0.0, 0.0]
        args = dict(horizon=hz[:5], bank=st["bank"], free_transfers=ft, selling_prices=sell)
        roll = optimize.transfer_plan(pr, ids, max_transfers=0, **args)
        best = optimize.transfer_plan(pr, ids, max_transfers=ft, **args)
        hit = optimize.transfer_plan(pr, ids, max_transfers=ft + 1, **args)
        # The buy's club disambiguates names: "Thomas" is two players in 2026-27.
        moves = "; ".join(f"{t['out']} → {t['in']} ({club.get(t['in_element'], '?')})"
                          for t in best.transfers) or "Roll"
        return dict(moves=moves, gain=best.objective - roll.objective,
                    hit_extra=hit.objective - best.objective,
                    captain=best.captain.get("web_name"),
                    buys=[t["in_element"] for t in best.transfers])

    scen = [("A", "As flagged now", [], 1)]
    scen += [(f"B{i}", f"{names[e]} out 2 GWs", [e], 2) for i, e in enumerate(doubt, start=1)]
    if len(doubt) > 1:
        scen.append(("C", "All doubtful out 2 GWs", doubt, 2))
    rows, buys = [], set()
    for sid, label, out_ids, weeks in scen:
        r = plan(out_ids, weeks)
        buys |= set(r.pop("buys"))
        rows.append(dict(id=sid, scenario=label, **r))
    files["plan"] = _round(pd.DataFrame(rows))

    cand = proj[(proj.gw == gw) & proj.element.isin(ids + sorted(buys))].merge(
        opp, on=["team", "gw"], how="left")
    cand["opp"] = cand.opp.fillna("—")
    cand["in_squad"] = cand.element.isin(ids)
    files["captaincy"] = _round(cand.sort_values("exp_points", ascending=False).head(6)[
        ["element", "web_name", "team", "opp", "exp_points", "p_start", "selected_by_percent",
         "in_squad"]].rename(columns={"web_name": "player", "exp_points": "xp",
                                      "selected_by_percent": "owned"}))

    # ---- the manager's season
    hist = _get(f"entry/{eid}/history/")
    h = pd.DataFrame(hist["current"]).merge(
        events[["id", "average_entry_score", "finished"]].rename(columns={"id": "event"}),
        on="event", how="left")
    h["value"] = h["value"] / 10
    h["bank"] = h["bank"] / 10
    files["history"] = _round(h.rename(columns={
        "event": "gw", "points": "points", "total_points": "total", "overall_rank": "overall_rank",
        "rank": "gw_rank", "points_on_bench": "bench", "event_transfers": "transfers",
        "event_transfers_cost": "hit", "average_entry_score": "average"})[
        ["gw", "points", "average", "total", "overall_rank", "gw_rank", "bench", "transfers",
         "hit", "value", "bank", "finished"]])
    tr = pd.DataFrame(_get(f"entry/{eid}/transfers/"))
    if len(tr):
        el = {e["id"]: e["web_name"] for e in boot["elements"]}
        files["transfers"] = _round(pd.DataFrame({
            "gw": tr.event, "out": tr.element_out.map(el), "in": tr.element_in.map(el),
            "out_price": tr.element_out_cost / 10, "in_price": tr.element_in_cost / 10,
            "made_at": tr.time.str.slice(0, 16)}).sort_values("gw", ascending=False))
    else:
        files["transfers"] = []

    # ---- news, the model's record, deadlines
    c = claims[claims.element.isin(ids + sorted(buys))]
    files["news"] = _round(c[["element", "player", "claim", "chance", "confidence", "source_type",
                              "source_name", "published_at", "source_url"]]
                           .rename(columns={"source_url": "url"}))
    _, pooled = scorecard.score(conn, SEASON, include_ci=False)
    if len(pooled):
        pooled = pooled.assign(check=pooled.metric.map({
            "spearman_all": "Ranking, all players", "spearman_start": "Ranking, starters only"}))
        files["model_record"] = _round(pooled[["check", "source", "model", "gws",
                                               "model_minus_naive", "lo", "hi",
                                               "p_model_better"]], 3)
    else:
        files["model_record"] = []
    ns = news.score(conn, SEASON, scorecard.deadlines(conn, SEASON))
    files["news_record"] = _round(pd.DataFrame([{**ns["disagree"], "needed": news.MIN_DISAGREEMENTS,
                                                 "verdict": ns["verdict"]}]), 3)
    up = events[(events.id >= gw) & (events.id <= max(FIRST_HALF_LAST_GW, gw))]
    files["deadlines"] = [
        {"gw": int(e.id), "deadline_utc": e.deadline_time,
         "note": ("First-half chips expire at this deadline" if e.id == FIRST_HALF_LAST_GW
                  else "Free Hit window (festive run)" if 17 <= e.id <= FIRST_HALF_LAST_GW
                  else "")}
        for e in up.itertuples()]

    files["status"] = [{
        "team": st["entry_name"], "gw": gw, "deadline_utc": deadline,
        "free_transfers": ft, "bank": st["bank"], "selling_value": st["selling_value"],
        "squad_value": round(st["squad_value"], 1),
        "chips_left": ", ".join(CHIP_NAMES.get(x, x) for x in st["chips_remaining"]) or "None",
        "chips_left_n": len(st["chips_remaining"]),
        "gws_to_chip_expiry": max(FIRST_HALF_LAST_GW - gw + 1, 0) if gw <= FIRST_HALF_LAST_GW else 0,
        "synced_at": st["fetched_at"], "built_at": now}]

    # The hash lets a refresh re-upload only the files whose content changed.
    for name, rows in files.items():
        body = json.dumps(rows, ensure_ascii=False).encode()
        (out / f"{name}.json").write_bytes(body)
        print(f"  {name:<13} {len(rows):>4} rows  sha256 {hashlib.sha256(body).hexdigest()[:16]}")
    print(f"wrote {len(files)} files to {out} for GW{gw}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
