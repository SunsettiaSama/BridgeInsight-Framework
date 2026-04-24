import sys
import tkinter as tk
from pathlib import Path

import yaml

project_root = Path(__file__).parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.visualize_tools.annotation_tools.annotation import AnnotationWindowGUI

_CONFIG_PATH = project_root / "config" / "visualize_tools" / "annotation_gui.yaml"


def _load_config() -> dict:
    with open(_CONFIG_PATH, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f) or {}


def _build_preset(cfg: dict) -> dict:
    sensor_raw = cfg.get('sensor_ids')
    if isinstance(sensor_raw, list):
        sensor_ids = [s.strip() for s in sensor_raw if s]
    elif isinstance(sensor_raw, str) and sensor_raw.strip():
        sensor_ids = [s.strip() for s in sensor_raw.split(',') if s.strip()]
    else:
        sensor_ids = []

    review_cfg = cfg.get('review') or {}

    return {
        'mode':                    cfg.get('mode', 'extreme'),
        'annotation_result_path':  cfg.get('annotation_result_path'),
        'date_start':              str(cfg['date_start']) if cfg.get('date_start') else None,
        'date_end':                str(cfg['date_end']) if cfg.get('date_end') else None,
        'sensor_ids':              sensor_ids,
        'rms_threshold':           cfg.get('rms_threshold'),
        'amplitude_threshold':     cfg.get('amplitude_threshold'),
        'review_allow_edit':       review_cfg.get('allow_edit', True),
        'review_filter_annotation': review_cfg.get('filter_annotation') or None,
    }


def main():
    cfg = _load_config()
    preset = _build_preset(cfg)

    mode = preset['mode']
    print(f"[annotation_gui] 启动模式: {mode}")
    if mode == 'review':
        edit_label = "允许修改" if preset['review_allow_edit'] else "只读"
        print(f"[annotation_gui] 复盘模式 → {edit_label}")

    root = tk.Tk()
    AnnotationWindowGUI(
        root,
        save_result_path=preset['annotation_result_path'],
        preset_config=preset,
    )
    root.mainloop()


if __name__ == '__main__':
    main()
