#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

# 优先使用当前目录下的虚拟环境
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
fi

PYTHON_BIN="${PYTHON_BIN:-python3}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    PYTHON_BIN="python"
fi

"$PYTHON_BIN" main.py
