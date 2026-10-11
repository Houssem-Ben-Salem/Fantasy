#!/usr/bin/env python3
"""
PreToolUse guard for Bash: keep an agent from walking around the git hooks.

The rules themselves live in .githooks/ (nothing under data/, tests before a
commit, no history-rewriting pushes) and apply to everyone. A git hook can be
skipped, though — `--no-verify`, `git -c core.hooksPath=...`, unsetting the
hooks path — so this denies those, plus force pushes (refused here with a
clear reason before the attempt) and `git add -f` into data/.

Reads the hook payload on stdin; prints a PreToolUse decision on stdout.
Stdlib only, so it runs under any python3.
"""
from __future__ import annotations

import json
import os
import re
import shlex
import sys

SEP = re.compile(r"\s*(?:&&|\|\||;|\||\n)\s*")
VALUE_FLAGS = {"-m", "-F", "-C", "-c", "-t", "--message", "--file", "--author",
               "--date", "--template", "--reuse-message", "--reedit-message"}
FORCE_PUSH = {"--force", "-f", "--force-if-includes"}


def _segments(cmd: str):
    for part in SEP.split(cmd):
        try:
            toks = shlex.split(part)
        except ValueError:
            toks = part.split()
        while toks and re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", toks[0]):   # VAR=x git ...
            toks = toks[1:]
        if toks:
            yield toks


def _git(toks: list[str]):
    """(global options, subcommand args) for a git invocation, else None."""
    if os.path.basename(toks[0]) != "git":
        return None
    i, glob = 1, []
    while i < len(toks) and toks[i].startswith("-"):
        if toks[i] in ("-C", "-c", "--git-dir", "--work-tree") and i + 1 < len(toks):
            glob += toks[i:i + 2]
            i += 2
        else:
            glob.append(toks[i])
            i += 1
    return glob, toks[i:]


def _short_flags(args: list[str]) -> set[str]:
    """Letters set in short-flag clusters, skipping option values (-m "msg")."""
    out, skip = set(), False
    for a in args:
        if skip:
            skip = False
            continue
        if a in VALUE_FLAGS:
            skip = True
            continue
        if a.startswith("-") and not a.startswith("--") and len(a) > 1:
            for ch in a[1:]:
                if ch in "mFCct":          # value flag: the rest is its value
                    break
                out.add(ch)
    return out


def _data_path(p: str) -> bool:
    p = p.lstrip("./") if p not in (".", "./") else p
    return p in (".", "./", ":/", "*", "-A", "--all") or p == "data" or p.startswith("data/")


def verdict(command: str) -> str | None:
    """A reason to deny, or None to allow."""
    for toks in _segments(command):
        parsed = _git(toks)
        if parsed is None:
            continue
        glob, args = parsed
        if any("core.hookspath" in g.lower() for g in glob):
            return "Overriding core.hooksPath on the command line skips the repository's git hooks."
        if not args:
            continue
        sub, rest = args[0], args[1:]
        if sub in ("commit", "push", "merge", "rebase", "am", "cherry-pick") and "--no-verify" in rest:
            return f"`git {sub} --no-verify` skips the repository's git hooks (data/ rule, tests, no rewrites)."
        if sub == "commit" and "n" in _short_flags(rest):
            return "`git commit -n` is --no-verify: it skips the pre-commit hook."
        if sub == "push":
            if FORCE_PUSH & set(rest) or any(a.startswith("--force-with-lease") for a in rest) \
                    or "f" in _short_flags(rest) or any(a.startswith("+") for a in rest):
                return ("Force-pushing rewrites remote history (and the CI bot pushes to main too). "
                        "Fetch and rebase instead; the pre-push hook would refuse it anyway.")
        if sub == "config" and any("core.hookspath" in a.lower() for a in rest):
            vals = [a for a in rest if not a.startswith("-") and "hookspath" not in a.lower()]
            if "--unset" in rest or "--unset-all" in rest or vals != [".githooks"]:
                return "core.hooksPath must stay `.githooks`; changing it disables the repository's git hooks."
        if sub == "add" and ({"-f", "--force"} & set(rest) or "f" in _short_flags(rest)):
            if any(_data_path(a) for a in rest if not a.startswith("-") or a in ("-A", "--all")):
                return "`git add -f` would stage ignored files under data/, which is never versioned."
    return None


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except json.JSONDecodeError:
        return 0
    reason = verdict(payload.get("tool_input", {}).get("command", ""))
    if reason:
        print(json.dumps({"hookSpecificOutput": {
            "hookEventName": "PreToolUse", "permissionDecision": "deny",
            "permissionDecisionReason": f"fplbrain guardrail: {reason}"}}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
