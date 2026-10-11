"""
Read a manager's real squad from the public FPL API.

Five gameweeks of squad state were entered by hand, and the bank was an
estimate every time — GW7 2026-27 was planned around GBP1.3m when the real bank
was GBP0.7m, which decides whether a GBP9.5m player is affordable. The entry
endpoints are public (no login) and give picks, bank, transfers and chips.

Two numbers they do NOT give are derived here, and every state this module
returns lists them under "derived" so nobody mistakes them for API facts:

  * selling price — FPL keeps half of any rise, rounded down to GBP0.1m:
        selling = purchase + (now - purchase) // 2   if the price rose, else now
    Purchase price is the cost of the latest transfer in, or the GW1 price for a
    player held since the start (prices do not move before the first deadline).
  * free transfers — 1 for the week after the squad was created, then +1 per
    gameweek capped at 5, spent by transfers made (hits leave nothing to roll),
    and kept through a Wildcard or Free Hit week. A one-off top-up (2025-26 gave
    every manager five in GW16) would not be seen — check the app if the season
    announces one.

Transfers made for the NEXT deadline are invisible until that deadline passes;
the state is the squad as of the last one.

    python -m fplbrain.entry --id 5391324 [--save]
    FPLBRAIN_ENTRY_ID=5391324 python -m fplbrain.entry --save
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
from datetime import datetime, timezone

import requests

API = "https://fantasy.premierleague.com/api"
MAX_FT = 5
POSITIONS = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}
# API chip names -> the names squad_state and the optimiser use
CHIPS = {"wildcard": "wildcard", "freehit": "free_hit", "bboost": "bench_boost",
         "3xc": "triple_captain"}
FIRST_HALF_LAST_GW = 19   # each chip is available once in GW1-19 and once in GW20-38


def _get(path: str) -> object:
    """One GET against the public API. Isolated so tests can answer it."""
    r = requests.get(f"{API}/{path}", timeout=30,
                     headers={"User-Agent": "fplbrain (squad sync)"})
    r.raise_for_status()
    return r.json()


def selling_price(purchase: int, now: int) -> int:
    """In tenths of GBP1m, as the API reports prices."""
    return purchase + (now - purchase) // 2 if now > purchase else now


def free_transfers(history: list[dict], chips: list[dict], next_gw: int,
                   started_event: int = 1) -> int:
    """Free transfers available at `next_gw`'s deadline, replayed from history."""
    chip_week = {c["event"]: c["name"] for c in chips}
    ft = 0
    for row in sorted(history, key=lambda r: r["event"]):
        g = row["event"]
        if g >= next_gw:
            break
        if g == started_event:          # squad creation: unlimited, then one
            ft = 1
        elif chip_week.get(g) in ("wildcard", "freehit"):
            ft = min(MAX_FT, ft + 1)    # the chip week's transfers are free; saved ones kept
        else:
            ft = min(MAX_FT, max(ft - row["event_transfers"], 0) + 1)
    return ft


def chips_remaining(chips: list[dict], next_gw: int) -> list[str]:
    half = range(1, FIRST_HALF_LAST_GW + 1) if next_gw <= FIRST_HALF_LAST_GW \
        else range(FIRST_HALF_LAST_GW + 1, 39)
    used = {CHIPS.get(c["name"], c["name"]) for c in chips if c["event"] in half}
    return [c for c in CHIPS.values() if c not in used]


def entry_state(entry_id: int) -> dict:
    boot = _get("bootstrap-static/")
    el = {e["id"]: e for e in boot["elements"]}
    teams = {t["id"]: t["short_name"] for t in boot["teams"]}
    nxt = next((e["id"] for e in boot["events"] if e.get("is_next")), None)

    info = _get(f"entry/{entry_id}/")
    cur = info["current_event"]
    hist = _get(f"entry/{entry_id}/history/")
    gw_next = nxt or cur + 1

    picks = _get(f"entry/{entry_id}/event/{cur}/picks/")
    bank = picks["entry_history"]["bank"]
    if picks.get("active_chip") == "freehit" and cur > info.get("started_event", 1):
        # The Free Hit squad is borrowed for one week; the previous one returns.
        picks = {**_get(f"entry/{entry_id}/event/{cur - 1}/picks/"),
                 "entry_history": picks["entry_history"]}

    bought = {}
    for t in sorted(_get(f"entry/{entry_id}/transfers/"), key=lambda t: t["event"]):
        bought[t["element_in"]] = (t["element_in_cost"], t["event"])

    squad = []
    for p in sorted(picks["picks"], key=lambda p: p["position"]):
        e = el[p["element"]]
        if p["element"] in bought:
            purchase, how = bought[p["element"]][0], f"GW{bought[p['element']][1]} transfer"
        else:
            h = _get(f"element-summary/{p['element']}/")["history"]
            purchase = next(r["value"] for r in h if r["round"] == info.get("started_event", 1))
            how = "initial squad"
        squad.append({
            "element": p["element"], "name": e["web_name"], "pos": POSITIONS[e["element_type"]],
            "team": teams[e["team"]], "now_cost": e["now_cost"] / 10,
            "purchase_price": purchase / 10,
            "selling_price": selling_price(purchase, e["now_cost"]) / 10,
            "acquired": how, "captain": p["is_captain"], "vice": p["is_vice_captain"],
        })

    return {
        "entry_id": entry_id, "entry_name": info.get("name"),
        "gw_current": cur, "gw_next": gw_next,
        "bank": bank / 10,
        "squad_value": sum(p["now_cost"] for p in squad),
        "selling_value": round(sum(p["selling_price"] for p in squad) + bank / 10, 1),
        "free_transfers": free_transfers(hist["current"], hist.get("chips", []), gw_next,
                                         info.get("started_event", 1)),
        "chips_used": hist.get("chips", []),
        "chips_remaining": chips_remaining(hist.get("chips", []), gw_next),
        "squad": squad,
        "derived": ["selling_price", "selling_value", "free_transfers"],
        "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


SYNC_PREFIX = "Synced from the FPL API"


def save(conn: sqlite3.Connection, s: dict) -> None:
    """
    Write as the squad_state row for the NEXT gameweek — the one being planned.

    The table is keyed on gameweek and written with INSERT OR REPLACE, so notes
    already saved for that week (judgement calls the end-of-season review scores
    decisions by) are carried over; only a previous sync line is replaced.
    """
    line = (f"{SYNC_PREFIX}, entry {s['entry_id']} ({s['entry_name']}), "
            f"{s['fetched_at']}. Squad and bank are API facts as of the GW{s['gw_current']} "
            f"deadline; {', '.join(s['derived'])} are DERIVED (fplbrain/entry.py). "
            f"Transfers made for GW{s['gw_next']} are not visible until its deadline.")
    row = conn.execute("SELECT notes FROM squad_state WHERE gw=?", (s["gw_next"],)).fetchone()
    kept = "\n\n".join(p for p in (row[0] or "").split("\n\n")
                         if p.strip() and not p.startswith(SYNC_PREFIX)) if row else ""
    notes = line + (f"\n\n{kept}" if kept else "")
    conn.execute(
        """INSERT OR REPLACE INTO squad_state
           (gw, recorded_at, bank, free_transfers, chips_remaining, squad, notes)
           VALUES (?,?,?,?,?,?,?)""",
        (s["gw_next"], s["fetched_at"], s["bank"], s["free_transfers"],
         json.dumps(s["chips_remaining"]), json.dumps(s["squad"]), notes))
    conn.commit()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--id", type=int, default=int(os.environ.get("FPLBRAIN_ENTRY_ID", 0)))
    ap.add_argument("--db", default=os.environ.get("FPLBRAIN_DB", "data/fpl.db"))
    ap.add_argument("--save", action="store_true", help="write squad_state for the next GW")
    a = ap.parse_args()
    if not a.id:
        ap.error("pass --id or set FPLBRAIN_ENTRY_ID")
    s = entry_state(a.id)
    print(f"{s['entry_name']} (entry {s['entry_id']}) — planning GW{s['gw_next']}: "
          f"bank £{s['bank']:.1f}m, FT {s['free_transfers']} (derived), "
          f"selling value £{s['selling_value']:.1f}m (derived), chips {s['chips_remaining']}")
    for p in s["squad"]:
        flag = " (C)" if p["captain"] else " (V)" if p["vice"] else ""
        print(f"  {p['pos']:<3} {p['name']:<14} {p['team']}  now £{p['now_cost']:.1f}  "
              f"paid £{p['purchase_price']:.1f}  sell £{p['selling_price']:.1f}  "
              f"[{p['acquired']}]{flag}")
    if a.save:
        with sqlite3.connect(a.db) as conn:
            save(conn, s)
        print(f"saved as squad_state GW{s['gw_next']}")


if __name__ == "__main__":
    main()
