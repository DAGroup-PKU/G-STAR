#!/usr/bin/env bash
# Kept for earlier commands; the launcher is distributed_train.sh in the repository root.
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/distributed_train.sh" "$@"
