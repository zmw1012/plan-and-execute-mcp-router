#!/usr/bin/env bash
# Download the pinned ETOM checkout into this worktree. Does not modify official files.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$ROOT/data/etom/raw"
URL="https://github.com/snooow1029/ETOM.git"
SHA="790f28727f567b88918420d163c524d46644a694"
mkdir -p "$ROOT/data/etom"
if [[ ! -d "$DEST/.git" ]]; then
  git clone "$URL" "$DEST"
fi
git -C "$DEST" fetch --tags origin
git -C "$DEST" checkout --detach "$SHA"
echo "ETOM checkout ready at $SHA"
echo "Raw files stay under data/etom/raw and are gitignored."
