#!/usr/bin/env bash
# The standard blog build (owner steering 2026-08-08 16:37Z): render
# every generated page, then mdbook build. Use THIS instead of a bare
# `mdbook build` at session close — it keeps the Queue page (a VIEW of
# fontaine/queue.json) from going stale. Push to the Space stays a
# separate step (upload_folder via huggingface_hub; memory file
# blog-space-push).
set -euo pipefail
cd "$(dirname "$0")/../.."

uv run python fontaine/scripts/queue_page.py
(cd fontaine/blog && mdbook build)
echo "blog built (queue page + mdbook)"
