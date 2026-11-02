"""The reduced homepage must preserve the full research/media archive."""
import hashlib
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_homepage_has_only_five_visible_v7_previews():
    source = (ROOT / "README.md").read_text()
    manifest = json.loads((ROOT / "docs/evidence/three_global/presentation_002.json").read_text())
    images = re.findall(r"!\[[^\]]*\]\(([^\s)]+\.gif)\)", source)
    assert len(images) == len(set(images)) == manifest["readme_gif_count"] == 5
    assert set(images) == set(manifest["readme_gifs"])
    assert all("v7" in name or "walkthrough" in name for name in images)
    targets = re.findall(r"\]\(([^\s)]+)\)", source)
    assert not any("compare" in name or re.search(r"v5[^/]*\.(gif|mp4)$", name) for name in targets)
    assert "v5 / Law 8 remains the default" in source
    assert "general TRIM\nsupport is not claimed" in source
    assert "HISTORICAL_CAMERA_MEDIA.md" in source


def test_v7_preview_reuse_preserves_original_media_and_source_manifest():
    manifest = json.loads((ROOT / "docs/evidence/three_global/presentation_002.json").read_text())
    digest = lambda name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
    original = manifest["source_manifest"]
    assert digest(original["path"]) == original["sha256"]
    assert len(manifest["preserved_media"]) == 56
    assert all(digest(name) == expected for name, expected in manifest["preserved_media"].items())
    catalog = json.loads((ROOT / original["path"]).read_text())
    assert len(catalog["episodes"]) + len(catalog["comparisons"]) == manifest["full_mp4_count"] == 47
    for derived in manifest["gif_derivations"]:
        row = next(row for row in catalog["episodes"] if row["id"] == derived["id"])
        assert digest(derived["gif"]) == derived["gif_sha256"] == row["receipt"]["outputs"]["demo.gif"]
        assert digest(derived["mp4"]) == derived["mp4_sha256"] == row["published"][derived["mp4"]]
        assert derived["command"] == row["receipt"]["gif_command"]
        assert derived["speed"] == row["receipt"]["gif_speed"]
    assert manifest["simulator_runs"] == manifest["camera_replays"] == manifest["new_encoding_runs"] == 0
