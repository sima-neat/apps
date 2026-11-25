"""Contract tests for the files agents read: AGENTS.md, agents/, the customer skills, and their discovery links.

AGENTS.md is loaded into every agent session and routes to agents/ and every skill. Harnesses discover skills
through one link per skill folder in each of SKILL_DISCOVERY_FOLDERS, so each skill keeps one copy. sima-cli ships
skills/ alone, so a customer skill links only its own files and public URLs.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import yaml

APPS_ROOT = Path(__file__).resolve().parents[2]
AGENTS = APPS_ROOT / "AGENTS.md"
# AGENTS.md is loaded into every agent session, so its length is a cost on every task.
AGENTS_LINE_BUDGET = 110
# Claude Code reads skills from .claude/skills/, and Codex from .agents/skills/.
SKILL_DISCOVERY_FOLDERS = (".claude/skills", ".agents/skills")
FRONTMATTER = re.compile(r"\A---\n(.*?)\n---\n", re.DOTALL)
# The target of a Markdown link or image, inline as ](target) or in a reference definition as [label]: target.
MARKDOWN_TARGET = re.compile(r"\]\(\s*<?([^)\s>]+)|^ {0,3}\[[^]]+\]:\s*<?([^\s>]+)", re.MULTILINE)
# A path into the files written for this repository's contributors: agents/, AGENTS.md, or the cheat sheet, as a
# relative path or at the start of a path in a GitHub URL of this repository. A folder named agents inside another
# path, such as targets/codex/agents/, is not one.
CONTRIBUTOR_FILE_REFERENCE = re.compile(
    r"(?:github\.com/sima-neat/apps/(?:blob|tree|raw)/[^/\s]+/)(?:agents(?![\w.-])|AGENTS\.md|docs/neat-api-cheat-sheet)"
    r"|(?<![\w./-])(?:\.\./)*(?:agents/|AGENTS\.md|docs/neat-api-cheat-sheet)"
)


def _skills() -> list[Path]:
    """Each skill folder: this repository's under agents/skills/, and customers' under skills/."""
    skills = [*APPS_ROOT.glob("agents/skills/*/SKILL.md"), *APPS_ROOT.glob("skills/*/SKILL.md")]
    return sorted(skill.parent for skill in skills)


def _agent_documents() -> list[Path]:
    return [AGENTS, *sorted((APPS_ROOT / "agents").rglob("*.md")), *sorted((APPS_ROOT / "skills").rglob("*.md"))]


def _slug(heading: str) -> str:
    """The anchor GitHub gives a Markdown heading."""
    text = re.sub(r"[`*_]", "", heading.strip().lower())
    return re.sub(r"[^\w\- ]", "", text).replace(" ", "-")


def _anchors(document: Path) -> set[str]:
    text = re.sub(r"```.*?```", "", document.read_text(encoding="utf-8"), flags=re.DOTALL)
    return {_slug(match.group(1)) for match in re.finditer(r"^#{1,6} +(.+?) *#*$", text, re.MULTILINE)}


def _local_links(document: Path) -> list[tuple[str, Path, str]]:
    """Each relative link in the document as (target, resolved path, anchor)."""
    links = []
    for match in MARKDOWN_TARGET.finditer(document.read_text(encoding="utf-8")):
        target = match.group(1) or match.group(2)
        if "://" in target or target.startswith(("mailto:", "<", "$")):
            continue
        path, _, anchor = target.partition("#")
        links.append((target, (document.parent / path).resolve() if path else document.resolve(), anchor))
    return links


def _frontmatter(skill: Path) -> dict:
    match = FRONTMATTER.match(skill.read_text(encoding="utf-8"))
    fields = yaml.safe_load(match.group(1)) if match else None
    return fields if isinstance(fields, dict) else {}


def _is_relative_link_to(entry: Path, folder: Path) -> bool:
    return entry.is_symlink() and not Path(os.readlink(entry)).is_absolute() and entry.resolve() == folder.resolve()


def test_links_and_anchors_resolve_in_agent_documents() -> None:
    failures = []
    for document in _agent_documents():
        for target, path, anchor in _local_links(document):
            if not path.exists():
                failures.append(f"{document.relative_to(APPS_ROOT)} -> {target}: no such file")
            elif anchor and path.suffix == ".md" and anchor not in _anchors(path):
                failures.append(f"{document.relative_to(APPS_ROOT)} -> {target}: no such heading")

    assert not failures, "\n".join(failures)


def test_agents_index_links_every_agents_file_and_skill() -> None:
    indexed = {path for _, path, _ in _local_links(AGENTS)}
    files = [*sorted((APPS_ROOT / "agents").glob("*.md")), *sorted((APPS_ROOT / "agents" / "references").glob("*.md"))]
    skills = [folder / "SKILL.md" for folder in _skills()]
    missing = [page.relative_to(APPS_ROOT).as_posix() for page in [*files, *skills] if page.resolve() not in indexed]

    assert not missing, "link from AGENTS.md: " + ", ".join(missing)
    assert len(AGENTS.read_text(encoding="utf-8").splitlines()) <= AGENTS_LINE_BUDGET


def test_skills_name_their_folder_and_describe_when_to_use_them() -> None:
    failures = []
    for folder in _skills():
        fields = _frontmatter(folder / "SKILL.md")
        name = folder.relative_to(APPS_ROOT).as_posix()
        if fields.get("name") != folder.name:
            failures.append(f"{name}/SKILL.md: name must be {folder.name}")
        if not isinstance(fields.get("description"), str) or not fields["description"].strip():
            failures.append(f"{name}/SKILL.md: no description in its YAML frontmatter")

    assert not failures, "\n".join(failures)


def test_customer_skills_stand_alone() -> None:
    failures = []
    for skill in sorted((APPS_ROOT / "skills").glob("*/")):
        for document in sorted(path for path in skill.rglob("*") if path.is_file() and path.suffix in {".md", ".yml"}):
            name = document.relative_to(APPS_ROOT).as_posix()
            text = document.read_text(encoding="utf-8")
            failures += [f"{name} names {match.group(0)}" for match in CONTRIBUTOR_FILE_REFERENCE.finditer(text)]
            if document.suffix == ".md":
                failures += [
                    f"{name} -> {target}"
                    for target, path, _ in _local_links(document)
                    if not path.is_relative_to(skill.resolve())
                ]

    assert not failures, "skills/ ships alone, so link only its own files and public URLs:\n" + "\n".join(failures)


def test_skill_discovery_folders_link_each_skill_once() -> None:
    skills = {folder.name: folder for folder in _skills()}
    failures = []
    for discovery in SKILL_DISCOVERY_FOLDERS:
        folder = APPS_ROOT / discovery
        entries = {entry.name: entry for entry in folder.iterdir()} if folder.is_dir() else {}
        failures += [f"{discovery}/{name}: missing" for name in sorted(skills.keys() - entries.keys())]
        failures += [f"{discovery}/{name}: names no skill" for name in sorted(entries.keys() - skills.keys())]
        failures += [
            f"{discovery}/{name}: must be a relative link to {skills[name].relative_to(APPS_ROOT)}"
            for name, entry in sorted(entries.items())
            if name in skills and not _is_relative_link_to(entry, skills[name])
        ]

    assert not failures, "each skill folder needs one relative link in each discovery folder:\n" + "\n".join(failures)
