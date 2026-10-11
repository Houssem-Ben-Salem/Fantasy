#!/usr/bin/env python3
"""
SessionStart status for fplbrain: what a session should know before anything
else, because each of these went unnoticed for weeks in 2026-27 —

  * the next deadline and how long is left (GW6 was missed with 2 FTs in hand)
  * whether CI is green, and any open [fplbrain] alert issues (CI was red for
    17 days and nobody knew)
  * how old the saved squad is and whether it came from the API (a hand-typed
    squad carried the wrong backup keeper for five weeks)
  * whether the git guardrails are switched on in this clone

Every source is optional: no database, no network or no git just drops a line.
Stdlib only; network calls time out in 3s; always exits 0.
"""
from __future__ import annotations

import json
import re
import sqlite3
import subprocess
import sys
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DB = ROOT / "data" / "fpl.db"
LEAD = timedelta(minutes=90)          # deadline = first kickoff - 90 min
NOW = datetime.now(timezone.utc)


def _ago(t: datetime) -> str:
    s = (NOW - t).total_seconds()
    if s < 3600:
        return f"{s / 60:.0f}m ago"
    return f"{s / 3600:.0f}h ago" if s < 172800 else f"{s / 86400:.0f}d ago"


def _until(t: datetime) -> str:
    s = (t - NOW).total_seconds()
    d, h = int(s // 86400), int(s % 86400 // 3600)
    return f"{d}d {h}h" if d else f"{h}h {int(s % 3600 // 60)}m"


def deadline_line(conn) -> tuple[str, int | None]:
    rows = conn.execute("SELECT gw, MIN(kickoff_time) FROM fixtures WHERE season=(SELECT MAX(season) "
                        "FROM fixtures) AND gw IS NOT NULL AND kickoff_time IS NOT NULL "
                        "GROUP BY gw ORDER BY gw").fetchall()
    for gw, k in rows:
        dl = datetime.fromisoformat(k.replace("Z", "+00:00")) - LEAD
        if dl > NOW:
            urgent = " — UNDER 24 HOURS" if dl - NOW < timedelta(hours=24) else ""
            return f"GW{gw} deadline {dl:%a %d %b %H:%M} UTC, in {_until(dl)}{urgent}", gw
    return "no upcoming deadline in the fixture list", None


def squad_line(conn, next_gw: int | None) -> str:
    r = conn.execute("SELECT gw, recorded_at, bank, free_transfers, chips_remaining, notes "
                     "FROM squad_state ORDER BY gw DESC, recorded_at DESC LIMIT 1").fetchone()
    if not r:
        return "no saved squad — run sync_squad()"
    gw, rec, bank, ft, chips, notes = r
    src = "synced from the FPL API" if (notes or "").startswith("Synced from the FPL API") else "HAND-ENTERED"
    when = _ago(datetime.fromisoformat(rec)) if rec else "unknown time"
    stale = f" — STALE: it is for GW{gw}, planning GW{next_gw}; run sync_squad()" \
        if next_gw and gw != next_gw else ""
    chips = json.loads(chips) if chips else []
    left = f", {19 - next_gw + 1} GWs until first-half chips expire" if next_gw and next_gw <= 19 and chips else ""
    return (f"squad GW{gw} {src} {when}: bank £{bank:.1f}m, {ft} FT, "
            f"chips {', '.join(chips) or 'none'}{left}{stale}")


def _slug() -> str | None:
    url = subprocess.run(["git", "-C", str(ROOT), "remote", "get-url", "origin"],
                         capture_output=True, text=True, timeout=3).stdout.strip()
    m = re.search(r"github\.com[:/]([^/]+/[^/]+?)(?:\.git)?$", url)
    return m.group(1) if m else None


def _get(url: str):
    req = urllib.request.Request(url, headers={"Accept": "application/vnd.github+json",
                                               "User-Agent": "fplbrain-session-status"})
    with urllib.request.urlopen(req, timeout=3) as r:
        return json.load(r)


def github_lines(slug: str) -> list[str]:
    out = []
    # Both workflows: if the brief workflow itself breaks, its failure alert may
    # break with it (same gh call), so a red run must be visible from here too.
    for wf in ("refresh-fpl-data", "deadline-brief"):
        runs = _get(f"https://api.github.com/repos/{slug}/actions/workflows/"
                    f"{wf}.yml/runs?per_page=1").get("workflow_runs", [])
        if runs:
            r = runs[0]
            state = r["conclusion"] or r["status"]
            mark = "green" if state == "success" else f"{state.upper()} — {r['html_url']}"
            when = _ago(datetime.fromisoformat(r["created_at"].replace("Z", "+00:00")))
            out.append(f"CI {wf}: {mark} ({when})")
    issues = [i for i in _get(f"https://api.github.com/repos/{slug}/issues?state=open&per_page=20")
              if i["title"].startswith("[fplbrain]") and "pull_request" not in i]
    for i in issues[:3]:
        out.append(f"open alert #{i['number']}: {i['title']}")
    return out


def main() -> int:
    lines = []
    next_gw = None
    try:
        with sqlite3.connect(f"file:{DB}?mode=ro", uri=True) as conn:
            dl, next_gw = deadline_line(conn)
            lines.append(dl)
            lines.append(squad_line(conn, next_gw))
    except Exception as e:  # no DB yet, schema drift, locked file
        lines.append(f"database unavailable ({type(e).__name__}) — run `make ingest`")
    try:
        slug = _slug()
        if slug:
            lines += github_lines(slug)
    except Exception:
        lines.append("GitHub status unavailable (offline or rate-limited)")
    try:
        hp = subprocess.run(["git", "-C", str(ROOT), "config", "--get", "core.hooksPath"],
                            capture_output=True, text=True, timeout=3).stdout.strip()
        if hp != ".githooks":
            lines.append("git guardrails OFF in this clone — enable: git config core.hooksPath .githooks")
    except Exception:
        pass
    text = "fplbrain status\n" + "\n".join(f"- {l}" for l in lines)
    short = [l.split(",")[0] if l.startswith(("GW", "squad")) else l for l in lines]
    short = [re.sub(r" — https://\S+", "", l) for l in short]
    print(json.dumps({"systemMessage": "fplbrain · " + " · ".join(short[:3]),
                      "hookSpecificOutput": {"hookEventName": "SessionStart",
                                             "additionalContext": text}}))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        sys.exit(0)
