#!/usr/bin/env bash
set -euo pipefail
demo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$demo_dir/.cache/uv}"
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-$demo_dir/.cache/python}"
export HF_HOME="${HF_HOME:-$demo_dir/.cache/huggingface}"
export HF_HUB_DISABLE_TELEMETRY=1
exec uv run --python 3.13 --project "$demo_dir" "$demo_dir/demo.py" "$@"
