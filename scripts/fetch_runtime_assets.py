#!/usr/bin/env python3
"""Fetch/verify only the pinned official USD composition used by both runtimes."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_file(url, path, expected_sha256, expected_bytes=None):
    path = Path(path)
    if path.is_file():
        if sha256(path) != expected_sha256:
            raise RuntimeError(f"Existing file differs from pinned SHA256; preserved: {path}")
        return "existing_verified"
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name+".rcs-partial")
    request = urllib.request.Request(url, headers={"User-Agent": "RoboticCuttingSim-bootstrap/1"})
    with urllib.request.urlopen(request, timeout=120) as response, partial.open("wb") as stream:
        shutil.copyfileobj(response, stream, length=1024*1024)
    if expected_bytes is not None and partial.stat().st_size != expected_bytes:
        raise RuntimeError(f"Downloaded size differs from pin: {url}; kept{partial}")
    if sha256(partial) != expected_sha256:
        raise RuntimeError(f"Downloaded SHA256 differs from pin: {url}; kept{partial}")
    partial.replace(path)
    return "official_download_verified"


def fetch_assets(cache=None, manifest=None):
    cache = Path(cache or ROOT/"artifacts/workcell_asset_cache").resolve()
    manifest = Path(manifest or ROOT/"configs/runtime_assets.lock.json")
    lock = json.loads(manifest.read_text())
    records = []
    for record in lock["files"]:
        relative = PurePosixPath(record["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Unsafe pinned asset path: {relative}")
        path = cache/relative
        if not path.resolve().is_relative_to(cache):
            raise ValueError(f"Asset path escapes through a symlink: {path}")
        source = fetch_file(lock["source"]+"/"+str(relative), path, record["sha256"], record["bytes"])
        records.append({**record, "source": source})
    receipt = {"status": "PASS_ASSET_FETCH", "cache": str(cache), "manifest_sha256": sha256(manifest),
               "asset_root": lock["source"], "files": records,
               "scope": lock["scope"], "gpu_validation": "NOT_RUN"}
    cache.mkdir(parents=True, exist_ok=True)
    # Preserve the cache's original fetch receipt; each bootstrap also records
    # the complete current verification result in its new output directory.
    receipt_path = cache/"fetch_receipt.json"
    if not receipt_path.exists():
        receipt_path.write_text(json.dumps(receipt, indent=2)+"\n")
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", type=Path)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    print(json.dumps(fetch_assets(args.asset_root, args.manifest), indent=2))
