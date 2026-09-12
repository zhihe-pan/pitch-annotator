#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

# 优先使用当前目录下的虚拟环境。直接指向 venv 里的 Python，避免 conda/base
# 环境中的 python3 抢先被解析，导致找不到 PySide6。
if [ -x "venv/bin/python" ]; then
    PYTHON_BIN="$DIR/venv/bin/python"
else
    PYTHON_BIN="${PYTHON_BIN:-python3}"
    if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
        PYTHON_BIN="python"
    fi
fi

"$PYTHON_BIN" main.py
