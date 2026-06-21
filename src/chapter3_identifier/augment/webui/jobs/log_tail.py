from __future__ import annotations

import re
from pathlib import Path

_TQDM_LINE_RE = re.compile(r".*\d+%\|")


def _sanitize_log_text(text: str) -> str:
    if not text:
        return text
    lines: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.split("\r")[-1].rstrip()
        if not line:
            continue
        if lines and _TQDM_LINE_RE.match(line) and _TQDM_LINE_RE.match(lines[-1]):
            lines[-1] = line
        else:
            lines.append(line)
    return "\n".join(lines)


def read_log_tail_text(log_path: str | Path | None, max_chars: int = 8000) -> str:
    if not log_path:
        return ""
    path = Path(log_path)
    if not path.exists():
        return ""
    size = path.stat().st_size
    if size <= 0:
        return ""
    read_bytes = min(size, max(max_chars * 4, 4096))
    with open(path, "rb") as f:
        f.seek(max(0, size - read_bytes))
        chunk = f.read()
        tail_text = chunk.decode("utf-8", errors="replace")
        if len(tail_text) <= max_chars:
            return _sanitize_log_text(tail_text.strip())

        # Keep both the beginning and end of long logs.
        marker = "\n... [日志中间已省略] ...\n"
        head_chars = max(600, int(max_chars * 0.3))
        tail_chars = max(1200, max_chars - head_chars - len(marker))
        head_read_bytes = min(size, max(head_chars * 4, 4096))
        f.seek(0)
        head_chunk = f.read(head_read_bytes)
        head_text = head_chunk.decode("utf-8", errors="replace")

    head_text = _sanitize_log_text(head_text.strip())[:head_chars]
    tail_text = _sanitize_log_text(tail_text.strip())
    tail_text = tail_text[-tail_chars:] if len(tail_text) > tail_chars else tail_text
    return f"{head_text}{marker}{tail_text}"
