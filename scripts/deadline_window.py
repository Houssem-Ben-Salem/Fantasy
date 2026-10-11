#!/usr/bin/env python3
"""
Decide what the deadline-brief workflow should do on this run.

GW6 2026-27 was missed with two free transfers in hand. This runs every two
hours from CI (which can reach the FPL API) and decides, from the next
deadline and the [fplbrain] issues that already exist:

  brief   no brief issue for the coming gameweek and the deadline is within 48h
          (if within 12h already, the brief doubles as the final call)
  final   a brief exists, the deadline is within 12h, no final call posted yet:
          a comment with refreshed injury flags and plan — late news is the
          news that matters
  close   briefs for gameweeks whose deadline has passed

Stdlib only and fast, so the common case — nothing to do — finishes before
any dependency is installed. GitHub's scheduler can run hours late, which is
why the windows are wide.

    python3 scripts/deadline_window.py [--github-output $GITHUB_OUTPUT]
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gh_issues  # noqa: E402

BRIEF_LEAD = timedelta(hours=48)
# 12h, not 8h: for a Saturday 10:00 UTC deadline (06:00 in the owner's EDT) an
# 8h final call lands overnight; 12h lands Friday ~18:00 local, after the
# Friday press conferences, with an evening left to act.
FINAL_LEAD = timedelta(hours=12)
FINAL_MARK = "<!-- fplbrain:final-call -->"


def title(gw: int) -> str:
    return f"{gh_issues.PREFIX} GW{gw} deadline brief"


def next_deadline(events: list[dict], now: datetime) -> tuple[int, datetime] | None:
    for e in sorted(events, key=lambda e: e["id"]):
        dl = datetime.fromisoformat(e["deadline_time"].replace("Z", "+00:00"))
        if dl > now:
            return e["id"], dl
    return None


def decide(now: datetime, gw: int, deadline: datetime, issues: list[dict]) -> list[dict]:
    actions = [{"do": "close", "number": i["number"], "title": i["title"]}
               for i in issues if i["state"] == "OPEN" and i["title"].startswith(f"{gh_issues.PREFIX} GW")
               and i["title"].endswith("deadline brief") and i["title"] != title(gw)]
    left = deadline - now
    mine = [i for i in issues if i["title"] == title(gw)]
    if left <= timedelta(0):
        return actions
    if not mine and left <= BRIEF_LEAD:
        actions.append({"do": "brief", "gw": gw, "deadline": deadline.isoformat(),
                        "final": left <= FINAL_LEAD})
    elif mine and left <= FINAL_LEAD and not any(FINAL_MARK in t for t in gh_issues.texts(mine[0])):
        actions.append({"do": "final", "gw": gw, "deadline": deadline.isoformat(),
                        "number": mine[0]["number"]})
    return actions


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--github-output")
    a = ap.parse_args()
    now = datetime.now(timezone.utc)
    req = urllib.request.Request("https://fantasy.premierleague.com/api/bootstrap-static/",
                                 headers={"User-Agent": "fplbrain deadline-brief"})
    with urllib.request.urlopen(req, timeout=60) as r:
        events = json.load(r)["events"]
    nd = next_deadline(events, now)
    actions = decide(now, nd[0], nd[1], gh_issues.list_alerts()) if nd else []
    for act in [x for x in actions if x["do"] == "close"]:
        gh_issues.close(act["number"], "Deadline passed — closing this brief.")
        print(f"closed #{act['number']} {act['title']}")
    work = [x for x in actions if x["do"] in ("brief", "final")]
    print(f"next deadline: GW{nd[0]} {nd[1]:%a %d %b %H:%M} UTC" if nd else "no upcoming deadline",
          "| work:", json.dumps(work) or "none")
    if a.github_output:
        with open(a.github_output, "a") as fh:
            fh.write(f"work={'true' if work else 'false'}\n")
            fh.write(f"actions={json.dumps(work)}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
