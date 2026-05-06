from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

from .pseudo_labeler import PseudoLabel

logger = logging.getLogger(__name__)


class AnnotationBuilder:
    """
    将金标注与伪标签合并，写入 AnnotationDataset 兼容的 JSON 文件。

    输出 JSON 格式（与 config/train/datasets/annotation_dataset.yaml 对齐）：
    [
        {
            "sample_id":     "<str>",
            "annotation":    <int>,       # class_id (0=Normal, 1=VIV, ...)
            "file_path":     "<str>",     # inplane/outplane VIC 文件路径
            "window_index":  <int>,
            "data_type":     "vic",
            "is_pseudo":     false/true,
            "pseudo_source": null/"mecc"/"dl"/"both",
            "confidence":    <float>
        },
        ...
    ]

    金标注条目直接原样复制，伪标签条目附加 is_pseudo / pseudo_source / confidence 字段。
    """

    def __init__(self, gold_annotation_path: str):
        """
        Parameters
        ----------
        gold_annotation_path : str
            现有金标注 JSON 文件路径（由人工标注工具产生）
        """
        self.gold_path    = gold_annotation_path
        self._gold_entries: List[dict] = self._load_gold(gold_annotation_path)

    @staticmethod
    def _load_gold(path: str) -> List[dict]:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        logger.info(f"金标注加载完成：{len(data)} 条 | 路径={path}")
        return data

    def build(
        self,
        pseudo_labels:   List[PseudoLabel],
        sample_metadata: Dict,
        output_path:     str,
        direction:       str = "inplane",
    ) -> int:
        """
        合并金标注和伪标签，写入 output_path。

        Parameters
        ----------
        pseudo_labels    : PseudoLabeler.generate() 的输出
        sample_metadata  : {sample_idx (int or str): {inplane_file_path, window_idx, ...}}
                           来自 FullDatasetRunner.save_predictions() 写入 JSON 的 sample_metadata 字段
        output_path      : 输出 JSON 路径
        direction        : "inplane" | "outplane"，决定使用哪个 file_path 字段

        Returns
        -------
        int  实际新增的伪标签条数
        """
        file_key = f"{direction}_file_path"
        entries  = [dict(entry) for entry in self._gold_entries]
        added    = 0
        skipped  = 0

        for pl in pseudo_labels:
            meta = (
                sample_metadata.get(str(pl.sample_idx))
                or sample_metadata.get(pl.sample_idx)
            )
            if meta is None:
                skipped += 1
                continue

            file_path = meta.get(file_key)
            if not file_path:
                skipped += 1
                continue

            window_idx = meta.get("window_idx", meta.get("window_index", 0))

            entries.append({
                "sample_id":     f"pseudo_{pl.sample_idx}_{pl.source}",
                "annotation":    pl.class_id,
                "file_path":     file_path,
                "window_index":  window_idx,
                "data_type":     "vic",
                "is_pseudo":     True,
                "pseudo_source": pl.source,
                "confidence":    round(pl.confidence, 6),
            })
            added += 1

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(entries, f, ensure_ascii=False)

        logger.info(
            f"标注文件已写入：{output_path} | "
            f"金标注={len(self._gold_entries)} | 伪标签={added} | 跳过={skipped}"
        )
        return added
