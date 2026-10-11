#!/usr/bin/env python3
"""
Open, update and close the "[fplbrain] CI failing: <workflow>" issue.

CI failed every day for 17 days (25 Sep - 11 Oct 2026) and nobody knew: a red
run on a schedule notifies no one. Now the first failure opens an issue (GitHub
emails it), later failures add a comment, and the first green run closes it.

    python3 scripts/ci_alert.py fail --workflow refresh-fpl-data --run-url URL
    python3 scripts/ci_alert.py ok   --workflow refresh-fpl-data --run-url URL

Stdlib + gh only: it must work when `pip install` is the step that broke.
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gh_issues  # noqa: E402


def title(workflow: str) -> str:
    return f"{gh_issues.PREFIX} CI failing: {workflow}"


def plan(state: str, workflow: str, issues: list[dict], run_url: str, now: str) -> tuple:
    """('create', title, body) | ('comment', n, body) | ('close', n, body) | ('none',)"""
    open_ = [i for i in issues if i["title"] == title(workflow) and i["state"] == "OPEN"]
    if state == "fail":
        if open_:
            return ("comment", open_[0]["number"], f"Still failing at {now}: {run_url}")
        return ("create", title(workflow),
                f"The scheduled `{workflow}` run failed at {now}.\n\nRun: {run_url}\n\n"
                "Job logs need repository-admin access to read through the API; to triage, "
                "reproduce locally (CLAUDE.md: a worktree of origin/main, a fresh Python 3.12 "
                "venv resolving requirements.txt, then refresh -> tests -> export). This issue "
                "closes itself on the next green run.")
    if state == "ok" and open_:
        return ("close", open_[0]["number"], f"Recovered: green run at {now} — {run_url}")
    return ("none",)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("state", choices=["fail", "ok"])
    ap.add_argument("--workflow", required=True)
    ap.add_argument("--run-url", required=True)
    a = ap.parse_args()
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    act = plan(a.state, a.workflow, gh_issues.list_alerts(), a.run_url, now)
    if act[0] == "create":
        print(gh_issues.create(act[1], act[2]))
    elif act[0] == "comment":
        gh_issues.comment(act[1], act[2])
    elif act[0] == "close":
        gh_issues.close(act[1], act[2])
    print(f"ci_alert: {act[0]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
