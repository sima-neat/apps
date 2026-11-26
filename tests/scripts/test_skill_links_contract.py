"""Contract tests for the skill links that Claude Code and Codex discover."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest


APPS_ROOT = Path(__file__).resolve().parents[2]
SKILL_PREFIX_BY_TREE = {"agents/skills": "apps-", "skills": "sima-"}
LINK_FOLDERS = (".agents/skills", ".claude/skills")


def skill_tree_by_name() -> dict[str, str]:
    return {
        skill.name: tree
        for tree in SKILL_PREFIX_BY_TREE
        for skill in (APPS_ROOT / tree).iterdir()
        if skill.is_dir()
    }


def tracked_mode(path: str) -> str:
    entry = subprocess.check_output(
        ["git", "ls-files", "--stage", "--", path], cwd=APPS_ROOT, text=True
    )
    return entry.split(" ", 1)[0]


def test_claude_md_only_imports_agents_md():
    assert (APPS_ROOT / "CLAUDE.md").read_text(encoding="utf-8") == "@AGENTS.md\n"


@pytest.mark.parametrize("tree", SKILL_PREFIX_BY_TREE)
def test_skill_names_carry_their_tree_prefix(tree):
    prefix = SKILL_PREFIX_BY_TREE[tree]
    for skill in (APPS_ROOT / tree).iterdir():
        if not skill.is_dir():
            continue
        assert skill.name.startswith(prefix), f"{tree}/{skill.name}: name must start with {prefix}"
        skill_md = (skill / "SKILL.md").read_text(encoding="utf-8")
        assert f"\nname: {skill.name}\n" in skill_md, f"{tree}/{skill.name}/SKILL.md: name must match the folder"


@pytest.mark.parametrize("link_folder", LINK_FOLDERS)
def test_link_folder_holds_one_link_per_skill(link_folder):
    skills = skill_tree_by_name()
    entries = {entry.name for entry in (APPS_ROOT / link_folder).iterdir()}
    assert entries == set(skills), f"{link_folder}: entries must match the skill folders"

    for name, tree in skills.items():
        path = f"{link_folder}/{name}"
        link = APPS_ROOT / path
        assert tracked_mode(path) == "120000", f"{path}: must be a tracked symlink, not a copy"
        assert link.readlink() == Path("../..", tree, name), f"{path}: must link to ../../{tree}/{name}"
        assert (link / "SKILL.md").is_file(), f"{path}/SKILL.md: link target has no skill"
