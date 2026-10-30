"""Guard: tracked text files and paths must not name the user's separate project.

The token is assembled at runtime so this file does not contain it. Future
users of the environment are called agentic control (see AGENTS.md).
"""
import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TOKEN = "R" + "S" + "I"
# Standalone token only, any case: "VERSION" or "PERSISTENT" contain the
# letters but are not the acronym.
PATTERN = re.compile(rb"(?<![A-Za-z0-9])" + TOKEN.encode() + rb"(?![A-Za-z0-9])", re.IGNORECASE)


def _tracked_files():
    try:
        out = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, check=True, capture_output=True).stdout
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("not a git checkout")
    return [name for name in out.decode().split("\0") if name]


def test_pattern_matches_token_but_not_words_containing_its_letters():
    assert PATTERN.search(TOKEN.encode())
    assert PATTERN.search(f"no {TOKEN.lower()}-style agent".encode())
    for word in (b"VERSION", b"PERSISTENT", b"CRITERIA_VERSIONS", b"VERSION_ID"):
        assert not PATTERN.search(word)


def test_no_tracked_text_file_or_path_contains_the_token():
    hits = []
    for name in _tracked_files():
        if PATTERN.search(name.encode()):
            hits.append(f"{name}: path")
        path = ROOT / name
        if not path.is_file():
            continue
        data = path.read_bytes()
        if b"\0" in data[:8192]:
            continue  # binary (media, archives)
        for lineno, line in enumerate(data.splitlines(), 1):
            if PATTERN.search(line):
                hits.append(f"{name}:{lineno}")
    assert not hits, "forbidden project acronym in tracked files: " + ", ".join(hits[:50])
