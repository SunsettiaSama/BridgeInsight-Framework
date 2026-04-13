import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import json
import logging
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

logger = logging.getLogger(__name__)


def _make_extractor(enable_denoise: bool = False, freq_threshold=None):
    """延迟导入 VICWindowExtractor，避免模块级与预处理子包的硬耦合。"""
    from src.data_processer.preprocess.get_data_vib import VICWindowExtractor  # noqa: PLC0415
    return VICWindowExtractor(enable_denoise=enable_denoise, freq_threshold=freq_threshold)


# ---------------------------------------------------------------------------
# 轻量推理数据集（必须在模块顶层定义，保证 Windows spawn 多进程可 pickle）
# ---------------------------------------------------------------------------

class _InplaneWindowDataset(Dataset):
    """
    仅加载面内振动窗口，用于 DL 推理阶段。

    与 StayCableVib2023Dataset 解耦：只持有 _SampleRecord 列表和基本参数，
    不依赖完整数据集实例，从而支持 DataLoader num_workers > 0。
    """

    def __init__(
        self,
        records,                        # List[_SampleRecord]
        window_size: int,
        enable_denoise: bool = False,
        original_indices: Optional[List[int]] = None,
        freq_threshold: Optional[float] = None,
        extractor=None,                 # 可注入的 VICWindowExtractor，默认延迟构造
    ):
        self._records     = records
        self._window_size = window_size
        self._extractor   = (
            extractor
            if extractor is not None
            else _make_extractor(enable_denoise=enable_denoise, freq_threshold=freq_threshold)
        )
        # 原始样本索引：用于在过滤后仍能映射回 dataset._samples 的位置
        self._original_indices = (
            original_indices if original_indices is not None
            else list(range(len(records)))
        )

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        rec    = self._records[idx]
        signal = self._extractor.extract_window(
            rec.inplane_meta.get("file_path"),
            rec.window_idx,
            self._window_size,
            metadata=rec.inplane_meta,
        )
        return torch.from_numpy(signal).float(), self._original_indices[idx]


# ---------------------------------------------------------------------------
# 全量数据集识别器
# ---------------------------------------------------------------------------

class FullDatasetRunner:
    """
    对 StayCableVib2023Dataset 的全量样本执行 DL 识别。

    设计原则
    --------
    - 与数据集解耦：通过 `dataset._samples` / `dataset.config` 访问必要信息，
      不在 FullDatasetRunner 内部存储数据集引用。
    - 多进程加速：DataLoader 的 `num_workers` 并行加载 VIC 文件，
      `batch_size` 控制单次 GPU 推理量。
    - 结果格式：`run()` 返回 ``{sample_idx: predicted_label}``；
      `to_file_indexed()` 将其转换为按文件组织的窗口预测列表。
    """

    def __init__(
        self,
        identifier,           # DLVibrationIdentifier
        batch_size: int = 256,
        num_workers: int = 4,
    ):
        self.identifier   = identifier
        self.batch_size   = batch_size
        self.num_workers  = num_workers

    # ------------------------------------------------------------------ #
    # 主接口                                                                #
    # ------------------------------------------------------------------ #

    def run(self, dataset) -> Dict[int, int]:
        """
        对数据集所有样本执行推理。

        Parameters
        ----------
        dataset : StayCableVib2023Dataset

        Returns
        -------
        Dict[int, int]
            ``{sample_idx: predicted_label}``，覆盖数据集全量样本。
        """
        # 延迟构造 extractor，统一传入预验证与推理 Dataset，避免重复实例化
        extractor = _make_extractor(
            enable_denoise=dataset.config.enable_denoise,
            freq_threshold=getattr(dataset.config, "denoise_freq_threshold", None),
        )

        # 预验证：读取每个面内文件的真实长度，丢弃超出范围的不完整窗口
        valid_records, valid_orig_indices = self._validate_records(
            dataset._samples, dataset.config.window_size, extractor=extractor
        )

        pred_ds = _InplaneWindowDataset(
            records          = valid_records,
            window_size      = dataset.config.window_size,
            enable_denoise   = dataset.config.enable_denoise,
            original_indices = valid_orig_indices,
            freq_threshold   = getattr(dataset.config, "denoise_freq_threshold", None),
            extractor        = extractor,
        )

        loader_kwargs: dict = dict(
            batch_size         = self.batch_size,
            shuffle            = False,
            num_workers        = self.num_workers,
            pin_memory         = (self.identifier.device.type == "cuda"),
            persistent_workers = (self.num_workers > 0),
        )
        if self.num_workers > 0:
            loader_kwargs["prefetch_factor"] = 2

        loader      = DataLoader(pred_ds, **loader_kwargs)
        predictions: Dict[int, int] = {}
        total       = len(pred_ds)

        logger.info(
            f"开始全量 DL 识别：{total} 个样本（原始 {len(dataset._samples)} 个，"
            f"预验证丢弃 {len(dataset._samples) - total} 个不完整窗口）| "
            f"batch={self.batch_size} | workers={self.num_workers} | "
            f"device={self.identifier.device}"
        )

        with tqdm(total=total, desc="DL全量识别", unit="win") as pbar:
            for signals, indices in loader:
                preds = self.identifier.predict_batch(signals)
                for pred, idx in zip(preds, indices.numpy()):
                    predictions[int(idx)] = int(pred)
                pbar.update(len(indices))

        logger.info(f"识别完成，共 {len(predictions)} 条结果")
        return predictions

    # ------------------------------------------------------------------ #
    # 预验证                                                                #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _validate_records(
        records, window_size: int, extractor=None
    ) -> Tuple[list, List[int]]:
        """
        读取每个面内 VIC 文件的真实数据长度，丢弃末尾不完整窗口。

        对同一文件只读一次，通过字典缓存长度。
        返回 (有效记录列表, 对应的原始索引列表)。

        Parameters
        ----------
        extractor : 可注入的 VICWindowExtractor 实例；为 None 时延迟构造（enable_denoise=False）。
        """
        if extractor is None:
            extractor = _make_extractor(enable_denoise=False)
        file_length_cache: Dict[str, int] = {}

        for rec in records:
            fp = rec.inplane_meta.get("file_path")
            if fp and fp not in file_length_cache:
                vic_data = extractor.unpacker.VIC_DATA_Unpack(str(fp))
                file_length_cache[fp] = len(vic_data)
                del vic_data

        valid_records: list = []
        valid_orig_indices: List[int] = []
        skipped = 0

        for orig_idx, rec in enumerate(records):
            fp         = rec.inplane_meta.get("file_path")
            actual_len = file_length_cache.get(fp, 0)
            end_idx    = (rec.window_idx + 1) * window_size
            if actual_len > 0 and end_idx <= actual_len:
                valid_records.append(rec)
                valid_orig_indices.append(orig_idx)
            else:
                skipped += 1

        if skipped:
            logger.warning(
                f"预验证：{skipped} 个窗口索引超出文件实际长度（window_size={window_size}），已跳过"
            )

        return valid_records, valid_orig_indices

    # ------------------------------------------------------------------ #
    # 结果格式转换                                                           #
    # ------------------------------------------------------------------ #

    @staticmethod
    def to_file_indexed(
        predictions: Dict[int, int],
        dataset,
    ) -> Dict[str, List[int]]:
        """
        将 ``{sample_idx: label}`` 转换为按文件组织的窗口预测列表。

        返回结构
        --------
        ``{"{inplane_sensor_id}_{month}_{day}_{hour}": [pred_win0, pred_win1, ...]}``

        列表下标 = window_idx，值 = 预测类别（-1 表示该窗口无预测结果）。
        """
        key_max: Dict[str, int] = {}
        for rec in dataset._samples:
            m, d, h = rec.timestamp_key
            key     = f"{rec.cable_pair[0]}_{m}_{d}_{h}"
            if key not in key_max or rec.window_idx > key_max[key]:
                key_max[key] = rec.window_idx

        file_indexed: Dict[str, List[int]] = {
            k: [-1] * (v + 1) for k, v in key_max.items()
        }

        for idx, pred in predictions.items():
            rec     = dataset._samples[idx]
            m, d, h = rec.timestamp_key
            key     = f"{rec.cable_pair[0]}_{m}_{d}_{h}"
            file_indexed[key][rec.window_idx] = pred

        return file_indexed

    # ------------------------------------------------------------------ #
    # 缓存 IO                                                               #
    # ------------------------------------------------------------------ #

    @staticmethod
    def save_predictions(
        path: str,
        predictions: Dict[int, int],
        dataset,
        model_info: str = "",
    ) -> None:
        """
        将识别结果持久化为 JSON。

        JSON 结构
        ---------
        .. code-block:: json

            {
              "metadata": {
                "created_at": "...",
                "num_samples": 12345,
                "num_classes": 4,
                "model_info":  "..."
              },
              "predictions": {"0": 1, "1": 0, ...},
              "sample_metadata": {
                "0": {
                  "cable_pair": ["sensor_in", "sensor_out"],
                  "timestamp": [month, day, hour],
                  "window_idx": 0,
                  "inplane_sensor_id": "...",
                  "outplane_sensor_id": "...",
                  "missing_rate_in": 0.01,
                  "missing_rate_out": 0.02,
                  "has_wind": true
                },
                ...
              },
              "by_file": {
                "ST-VIC-C34-101-01_9_1_0": [0, 0, 1, 0, ...],
                ...
              }
            }

        - ``predictions``       : 平铺格式，键为样本索引字符串，值为预测类别。
        - ``sample_metadata``   : 每个样本的完整元数据（用于追溯和统计）。
        - ``by_file``           : 按（传感器, 月, 日, 时）分组的窗口预测列表，
                                  列表下标即 window_idx。
        """
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

        # 只为有预测结果的样本构建元数据（而非全量 dataset._samples）
        sample_metadata = {}
        for idx in predictions.keys():
            if idx < len(dataset._samples):
                rec = dataset._samples[idx]
                sample_metadata[str(idx)] = {
                    "cable_pair":          list(rec.cable_pair),
                    "timestamp":           list(rec.timestamp_key),  # (month, day, hour)
                    "window_idx":          rec.window_idx,
                    "inplane_sensor_id":   rec.inplane_meta.get("sensor_id"),
                    "outplane_sensor_id":  rec.outplane_meta.get("sensor_id"),
                    "inplane_file_path":   rec.inplane_meta.get("file_path"),
                    "outplane_file_path":  rec.outplane_meta.get("file_path"),
                    "missing_rate_in":     rec.inplane_meta.get("missing_rate"),
                    "missing_rate_out":    rec.outplane_meta.get("missing_rate"),
                    "has_wind":            rec.wind_meta is not None,
                }
            else:
                logger.warning(f"样本索引 {idx} 超出数据集范围（数据集共 {len(dataset._samples)} 个样本）")

        # 计算数据集指纹（用于后续验证一致性）
        dataset_fingerprint = dataset._compute_fingerprint()

        payload = {
            "metadata": {
                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "num_samples": len(predictions),
                "num_classes": getattr(dataset, "_num_classes", 4),
                "model_info":  model_info,
                "dataset_fingerprint_hash": dataset._fingerprint_hash(dataset_fingerprint),
            },
            "predictions": {str(k): int(v) for k, v in predictions.items()},
            "sample_metadata": sample_metadata,
            "by_file":      FullDatasetRunner.to_file_indexed(predictions, dataset),
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)

        logger.info(f"识别结果已保存：{path}（{len(predictions)} 条）")


    @staticmethod
    def load_predictions(path: str) -> Dict[int, int]:
        """
        从 JSON 缓存加载平铺格式识别结果。

        Returns
        -------
        Dict[int, int]  ``{sample_idx: predicted_label}``
        """
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        predictions = {int(k): int(v) for k, v in payload["predictions"].items()}
        meta        = payload.get("metadata", {})
        logger.info(
            f"识别结果已加载：{path} | "
            f"样本数={len(predictions)} | "
            f"生成时间={meta.get('created_at', 'unknown')}"
        )
        return predictions
