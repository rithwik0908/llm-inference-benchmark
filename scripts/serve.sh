#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"
export PYTHONPATH="$repo_dir/src${PYTHONPATH:+:$PYTHONPATH}"
exec python3 -m inference_bench.server "$@"
