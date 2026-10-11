"""
Agent and skill definitions must stay loadable and point at tools that exist.

Claude Code skips a definition whose frontmatter it cannot read, without an
error anyone sees, and a subagent granted a tool that was renamed simply cannot
call it. Both would fail silently in the middle of a deadline week.
"""
from __future__ import annotations

from pathlib import Path

from fplbrain import mcp_server

ROOT = Path(__file__).resolve().parents[1]


def _frontmatter(path: Path) -> dict[str, str]:
    text = path.read_text()
    assert text.startswith("---\n"), f"{path}: no frontmatter"
    block = text.split("---\n", 2)[1]
    out = {}
    for line in block.splitlines():
        key, _, value = line.partition(":")
        out[key.strip()] = value.strip()
    return out


def test_agents_are_loadable_and_their_fplbrain_tools_exist():
    agents = sorted((ROOT / ".claude" / "agents").glob("*.md"))
    assert agents, "no agent definitions found"
    for path in agents:
        fm = _frontmatter(path)
        assert fm.get("name") == path.stem and fm.get("description"), path
        tools = [t.strip() for t in fm.get("tools", "").split(",") if t.strip()]
        for t in tools:
            if t.startswith("mcp__fplbrain__"):
                assert callable(getattr(mcp_server, t.removeprefix("mcp__fplbrain__"), None)), \
                    f"{path.name} grants {t}, which mcp_server does not define"


def test_skills_are_loadable_and_their_linked_files_exist():
    skills = sorted((ROOT / ".claude" / "skills").glob("*/SKILL.md"))
    assert skills, "no skills found"
    for path in skills:
        fm = _frontmatter(path)
        assert fm.get("name") == path.parent.name and fm.get("description"), path
        body = path.read_text()
        for link in [s.split("](", 1)[1].split(")", 1)[0] for s in body.split("[")[1:] if "](" in s]:
            if not link.startswith(("http", "#")):
                assert (path.parent / link).exists(), f"{path}: broken link {link}"
