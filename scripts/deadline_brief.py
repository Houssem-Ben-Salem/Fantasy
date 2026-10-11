#!/usr/bin/env python3
"""
Build and post the deadline brief as a GitHub issue, or the final-call comment.

Runs in CI after scripts/refresh.py, on the actions scripts/deadline_window.py
chose. The reminder is the part that must never fail — GW6 2026-27 was missed —
so every other section is built independently and a failure shows inline
instead of killing the post.

The brief hides a snapshot of the squad's injury flags; the final call, ~12h
before the deadline, diffs against it, because news from the Thursday and
Friday press conferences is the information the model cannot infer
(LIMITATIONS s5) and the reason to look again.

Needs FPLBRAIN_ENTRY_ID (the repository variable FPL_ENTRY_ID) for anything
squad-specific; without it the brief is a deadline reminder plus model picks.

    python scripts/deadline_brief.py --actions '<json>'      # as CI runs it
    python scripts/deadline_brief.py --dry-run [--final]     # print, post nothing
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
os.environ.setdefault("FPLBRAIN_DB", str(ROOT / "data" / "fpl.db"))

import deadline_window  # noqa: E402
import gh_issues  # noqa: E402

FLAGS_RE = re.compile(r"<!-- fplbrain:flags (\{.*?\}) -->", re.S)
FIRST_HALF_LAST_GW = 19


# ------------------------------------------------------------ pure pieces

def flags_snapshot(rows: list[dict]) -> dict[str, str]:
    """element -> 'status|chance|news' for the players in `rows`."""
    return {str(r["element"]): f"{r['status']}|{'' if r['chance'] is None else int(r['chance'])}|{r['news'] or ''}"
            for r in rows}


def diff_flags(before: dict[str, str], after: dict[str, str], names: dict[str, str]) -> list[str]:
    out = []
    for el, now in after.items():
        was = before.get(el)
        if was is not None and was != now:
            s0, c0, _ = was.split("|", 2)
            s1, c1, n1 = now.split("|", 2)
            out.append(f"**{names.get(el, el)}**: {s0}{('/' + c0 + '%') if c0 else ''} → "
                       f"{s1}{('/' + c1 + '%') if c1 else ''}" + (f" — {n1}" if n1 else ""))
    return out


def countdown(deadline: datetime, now: datetime) -> str:
    s = (deadline - now).total_seconds()
    d, h, m = int(s // 86400), int(s % 86400 // 3600), int(s % 3600 // 60)
    return f"{d}d {h}h" if d else f"{h}h {m}m"


# ------------------------------------------------------------ sections

def _safe(fn, *a) -> str:
    try:
        return fn(*a)
    except Exception as e:  # one broken section must not cost the reminder
        return f"_{fn.__name__.replace('_', ' ')} unavailable: {type(e).__name__}: {e}_"


def _conn():
    return sqlite3.connect(os.environ["FPLBRAIN_DB"])


def squad_rows(elements: list[int]) -> list[dict]:
    q = ("SELECT p.element, p.web_name name, p.position pos, t.short_name team, p.status, "
         "p.chance_next_round chance, p.news, p.selected_by_percent own FROM players p "
         "JOIN teams t ON t.season=p.season AND t.team_id=p.team_id "
         "WHERE p.season=? AND p.element IN (%s)" % ",".join("?" * len(elements)))
    with _conn() as c:
        c.row_factory = sqlite3.Row
        return [dict(r) for r in c.execute(q, [os.environ.get("FPLBRAIN_SEASON", "2026-27"), *elements])]


def squad_section(state: dict, rows: list[dict], gw: int) -> str:
    flagged = [r for r in rows if r["status"] != "a" or (r["news"] or "").strip()]
    chips = state["chips_remaining"]
    left = (f" · {FIRST_HALF_LAST_GW - gw + 1} gameweeks until the first-half chips expire"
            if gw <= FIRST_HALF_LAST_GW and chips else "")
    lines = [f"**Bank £{state['bank']:.1f}m · {state['free_transfers']} free transfer(s)** (derived) · "
             f"chips: {', '.join(chips) or 'none'}{left}", ""]
    if flagged:
        lines.append("Flagged in your squad:")
        lines += [f"- **{r['name']}** ({r['team']}) — {r['status']}"
                  + (f" {int(r['chance'])}%" if r['chance'] is not None else "")
                  + (f": {r['news']}" if r['news'] else "") for r in flagged]
    else:
        lines.append("No injury or suspension flags in your squad.")
    return "\n".join(lines)


def plan_section() -> str:
    from fplbrain import mcp_server
    return mcp_server.plan_transfers()


def captain_section(elements: list[int] | None, gw: int) -> str:
    from fplbrain import mcp_server
    proj = mcp_server._projections(gw, 1)
    p = proj[proj.gw == gw]
    if elements:
        p = p[p.element.isin(elements)]
    top = p.sort_values("exp_points", ascending=False).head(5)
    who = "your squad" if elements else "all players (no squad configured)"
    rows = [f"| {r.web_name} | {r.team} | {r.exp_points:.2f} | {r.selected_by_percent:.1f}% |"
            for r in top.itertuples()]
    return (f"Top projected for GW{gw}, {who}. Ownership matters for overall rank: a captain "
            "most of the field also owns protects rank; a low-owned one gains it.\n\n"
            "| player | team | xP | owned |\n|---|---|---|---|\n" + "\n".join(rows))


def chips_section(gw: int) -> str:
    from fplbrain import mcp_server
    ev = json.loads(mcp_server.chip_advice())
    return (f"Bench Boost would add **{ev.get('bench_boost_gain', 0):.1f}**, Triple Captain "
            f"**{ev.get('triple_captain_gain', 0):.1f}** this week. Policy before GW19 "
            "(prompts/AGENT.md): no doubles or blanks are scheduled, so each chip goes in its best "
            "remaining week — a wildcard when two or more players are wrong for two-plus weeks, "
            "Free Hit in GW17-19, Bench Boost after the wildcard, Triple Captain re-assessed in November.")


def health_section() -> str:
    from fplbrain import scorecard
    with _conn() as c:
        _, pooled = scorecard.score(c, os.environ.get("FPLBRAIN_SEASON", "2026-27"), include_ci=False)
        run = c.execute("SELECT run_id, provenance FROM runs WHERE run_id LIKE 'proj-%' "
                        "ORDER BY created_at DESC LIMIT 1").fetchone()
    prov = json.loads(run[1]) if run and run[1] else {}
    lines = [f"Projections: `{run[0] if run else 'none'}` from code `{prov.get('code_version', '?')}`."]
    if not pooled.empty:
        st = pooled[pooled.metric == "spearman_start"].copy()
        st["last"] = st.gws.map(lambda g: max(int(x) for x in str(g).split(",")))
        # the model that scored the most recent gameweek, preferring local runs
        r = st.sort_values(["last", "source"], ascending=[False, False]).iloc[0]
        lines.append(f"Prospective record, points among starters vs a naive baseline: "
                     f"{r.model_minus_naive:+.3f} [{r.lo:+.3f}, {r.hi:+.3f}] over GW{r.gws} — "
                     + ("still undecided." if r.lo <= 0 <= r.hi else "decided."))
    return " ".join(lines)


# ------------------------------------------------------------ assembly

def build(gw: int, deadline: datetime, final: bool, previous_body: str = "") -> str:
    now = datetime.now(timezone.utc)
    head = (f"## {'Final call — ' if final else ''}GW{gw} deadline: "
            f"{deadline:%a %d %b %H:%M} UTC — in {countdown(deadline, now)}\n")
    parts = [head]
    entry_id = int(os.environ.get("FPLBRAIN_ENTRY_ID") or 0)
    elements, rows = None, []
    if entry_id:
        try:
            from fplbrain import entry
            state = entry.entry_state(entry_id)
            with _conn() as c:
                entry.save(c, state)
            elements = [p["element"] for p in state["squad"]]
            rows = squad_rows(elements)
            if final:
                m = FLAGS_RE.search(previous_body)
                changes = diff_flags(json.loads(m.group(1)) if m else {}, flags_snapshot(rows),
                                     {str(r["element"]): r["name"] for r in rows})
                parts.append("### Since the brief\n" + ("\n".join(f"- {c}" for c in changes)
                             if changes else "No change in your players' flags."))
            parts.append("### Your squad\n" + _safe(squad_section, state, rows, gw))
            parts.append("### Model plan\n" + _safe(plan_section))
        except Exception as e:
            parts.append(f"_Squad sync failed ({type(e).__name__}: {e}); squad sections skipped._")
    else:
        parts.append("_No squad configured: set the repository variable `FPL_ENTRY_ID` "
                     "(Settings → Secrets and variables → Actions → Variables) for squad-specific advice._")
    parts.append("### Captaincy\n" + _safe(captain_section, elements, gw))
    if not final:
        parts.append("### Chips\n" + _safe(chips_section, gw))
        parts.append("### Model health\n" + _safe(health_section))
    parts.append("---\nFor injury news and a red-team review, open Claude Code in the repo and run "
                 "`/deadline-brief`. " + ("" if final else "A final call with refreshed flags follows "
                                                         "about 12 hours before the deadline."))
    if rows:
        parts.append(f"<!-- fplbrain:flags {json.dumps(flags_snapshot(rows))} -->")
    if final:
        parts.append(deadline_window.FINAL_MARK)
    return "\n\n".join(parts)


def _live_deadline() -> tuple[int, datetime]:
    req = urllib.request.Request("https://fantasy.premierleague.com/api/bootstrap-static/",
                                 headers={"User-Agent": "fplbrain deadline-brief"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return deadline_window.next_deadline(json.load(r)["events"], datetime.now(timezone.utc))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--actions", default="[]")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--final", action="store_true", help="with --dry-run: render a final call")
    a = ap.parse_args()
    if a.dry_run:
        gw, dl = _live_deadline()
        print(build(gw, dl, a.final))
        return 0
    for act in json.loads(a.actions):
        dl = datetime.fromisoformat(act["deadline"])
        if act["do"] == "brief":
            body = build(act["gw"], dl, act["final"])
            print(gh_issues.create(deadline_window.title(act["gw"]), body))
        elif act["do"] == "preview":
            # Manual delivery test: the full brief, built on the runner, shown on
            # the run's summary page and sent once as an issue that closes itself.
            body = build(act["gw"], dl, False)
            summary = os.environ.get("GITHUB_STEP_SUMMARY")
            if summary:
                with open(summary, "a") as fh:
                    fh.write(body + "\n")
            url = gh_issues.create(deadline_window.preview_title(act["gw"]),
                                   "Delivery test, closed automatically. If this reached your "
                                   "inbox, briefs will too.\n\n" + body)
            number = int(url.rstrip("/").rsplit("/", 1)[-1])
            gh_issues.close(number, "Delivery test complete — closing.")
            print(f"preview sent and closed: {url}")
        elif act["do"] == "final":
            issue = next(i for i in gh_issues.list_alerts() if i["number"] == act["number"])
            gh_issues.comment(act["number"], build(act["gw"], dl, True, issue.get("body") or ""))
            print(f"final call posted on #{act['number']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
