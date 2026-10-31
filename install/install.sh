#!/bin/sh
# No toolchains are installed by this launcher. See install/INSTALL.md.
set -eu
if ! command -v python3 >/dev/null 2>&1; then
    printf '%s\n' 'Python 3.9+ is required. Install Python explicitly, then rerun this installer.' >&2
    exit 1
fi
SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
exec python3 "$SCRIPT_DIR/source_install.py" "$@"
