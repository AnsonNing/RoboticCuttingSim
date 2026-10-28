#!/usr/bin/env bash
set -euo pipefail

task_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
node_dir="$task_root/.deps/remote-viewer-node"
cache_dir="$task_root/.deps/remote-viewer"
node_version="v22.23.4"
node_archive="node-${node_version}-linux-x64.tar.xz"
node_sha256="1805bcd35e6815559bd2eb97db438d3fdb0eb3a5e6a977b7c1badca76337e9ce"

if [[ "$(uname -s)" != Linux || "$(uname -m)" != x86_64 ]]; then
    echo 'The pinned portable Node bundle requires Linux x86_64.' >&2
    exit 1
fi
mkdir -p "$cache_dir" "$node_dir"
if [[ ! -x "$node_dir/bin/node" ]]; then
    curl --fail --location --retry 3 --proto '=https' \
        "https://nodejs.org/dist/${node_version}/${node_archive}" -o "$cache_dir/$node_archive"
    printf '%s  %s\n' "$node_sha256" "$cache_dir/$node_archive" | sha256sum --check
    tar -xJf "$cache_dir/$node_archive" -C "$node_dir" --strip-components=1
fi
if [[ "$("$node_dir/bin/node" --version)" != "$node_version" ]]; then
    echo "Existing repository-local Node differs from pinned $node_version." >&2
    exit 1
fi
export PATH="$node_dir/bin:$PATH"
export NPM_CONFIG_CACHE="$cache_dir/npm-cache"
cd "$task_root/viewer"
npm ci --ignore-scripts --no-audit --no-fund
npm run build
echo 'Viewer build ready. Start: python3 scripts/serve_remote_web_viewer.py from the repository root.'
