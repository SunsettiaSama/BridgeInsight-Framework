from __future__ import annotations

import os
from typing import Optional, Sequence


def extract_metadata_from_path(file_path: str) -> tuple[str, str, str]:
    path_parts = file_path.replace("\\", "/").split("/")
    filename = path_parts[-1]
    filename_without_ext = os.path.splitext(filename)[0]
    sensor_time_parts = filename_without_ext.rsplit("_", 1)

    if len(sensor_time_parts) == 2:
        sensor_id = sensor_time_parts[0]
        time_stamp = sensor_time_parts[1]
    else:
        sensor_id = filename_without_ext
        time_stamp = "000000"

    if len(path_parts) >= 3:
        month = path_parts[-3]
        day = path_parts[-2]
    else:
        month = "01"
        day = "01"

    if len(time_stamp) >= 6:
        time_str = f"{time_stamp[0:2]}:{time_stamp[2:4]}:{time_stamp[4:6]}"
    else:
        time_str = "00:00:00"

    date_str = f"{month}/{day}"
    return sensor_id, date_str, time_str


def format_sample_title(
    kind: str,
    sensor_id: str,
    file_path: Optional[str],
    window_index: int,
    timestamp: Optional[Sequence[int]] = None,
) -> str:
    if file_path:
        sid, date_str, time_str = extract_metadata_from_path(file_path)
        sensor_id = sid or sensor_id
    elif timestamp and len(timestamp) >= 3:
        month, day, hour = timestamp[:3]
        date_str = f"{int(month):02d}/{int(day):02d}"
        time_str = f"{int(hour):02d}:00:00"
    else:
        date_str = "--/--"
        time_str = "--:--:--"

    sid = sensor_id or "Unknown"
    return f"{kind} · {sid} @ {date_str} {time_str} (Window {window_index})"
