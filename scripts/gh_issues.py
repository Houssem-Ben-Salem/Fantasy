"""
Thin wrapper over the `gh` CLI for the alert issues, stdlib only, so the alert
steps work even when `pip install` is what failed.

DELIVERY. An issue posted by the workflow bot is emailed only to people watching
the repository with email on for watched activity — the first delivery test
(issue #1, 11 Oct 2026) reached GitHub and never reached the owner's inbox. So
every alert is ASSIGNED to the owner and @-MENTIONS them: both are direct
"participating" notifications, which GitHub emails by default whatever the
watch settings. The owner comes from GITHUB_REPOSITORY_OWNER, which Actions
sets on every run; outside Actions nobody is mentioned.
"""
from __future__ import annotations

import json
import os
import subprocess

PREFIX = "[fplbrain]"


def _gh(*args: str) -> str:
    return subprocess.run(["gh", *args], capture_output=True, text=True, check=True).stdout


def list_alerts() -> list[dict]:
    """Every [fplbrain] issue, open or closed, with bodies and comments."""
    out = _gh("issue", "list", "--state", "all", "--limit", "100",
              "--json", "number,title,state,body,comments")
    return [i for i in json.loads(out) if i["title"].startswith(PREFIX)]


def texts(issue: dict) -> list[str]:
    return [issue.get("body") or ""] + [c.get("body") or "" for c in issue.get("comments", [])]


def _owner() -> str:
    return os.environ.get("GITHUB_REPOSITORY_OWNER", "").strip()


def _cc(body: str) -> str:
    return f"{body}\n\ncc @{_owner()}" if _owner() else body


def create(title: str, body: str) -> str:
    args = ["issue", "create", "--title", title, "--body", _cc(body)]
    if _owner():
        try:
            return _gh(*args, "--assignee", _owner()).strip()
        except subprocess.CalledProcessError:
            pass  # assignment refused: the mention still notifies; never lose the alert
    return _gh(*args).strip()


def comment(number: int, body: str) -> None:
    _gh("issue", "comment", str(number), "--body", _cc(body))


def close(number: int, body: str) -> None:
    _gh("issue", "close", str(number), "--comment", body)
