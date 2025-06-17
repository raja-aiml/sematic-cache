#!/usr/bin/env bash
# Wrapper script to maintain backward compatibility
# Calls the actual cluster.sh script in the scripts/ directory

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$SCRIPT_DIR/scripts/cluster.sh" "$@"