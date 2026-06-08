import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import json
import logging
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

logger = logging.getLogger(__name__)

_NORMAL_LABEL = 0


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
        records,
        window_size: int,
        enable_denoise: bool = False,
        original_indices: Optional[List[int]] = None,
        freq_threshold: Optional[float] = None,
        extractor=None,
    ):
        self._records        = records
        self._window_size    = window_size
        self._enable_denoise = enable_denoise
        self._freq_threshold = freq_threshold
        self._extractor      = None
        self._original_indices = (
            original_indices if original_indices is not None
            else list(range(len(records)))
        )

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        if self._extractor is None:
            self._extractor  = _make_extractor(
                enable_denoise=self._enable_denoise,
                freq_threshold=self._freq_threshold,
            )
            self._cache_path: Optional[str] = None
            self._cache_data                = None

        rec       = self._records[idx]
        file_path = rec.inplane_meta.get("file_path")

        if file_path != self._cache_path:
            self._cache_data = self._extractor.load_file(file_path)
            self._cache_path = file_path

        signal = self._extractor.extract_window_from_data(
            self._cache_data,
            rec.window_idx,
            self._window_size,
            metadata=rec.inplane_meta,
            file_path=file_path,
        )
        return torch.from_numpy(signal).float(), self._original_indices[idx]


class _OutplaneWindowDataset(Dataset):
    """
    仅加载面外振动窗口，用于 DL 推理阶段。

    与 _InplaneWindowDataset 结构相同，仅读取 outplane_meta。
    """

    def __init__(
        self,
        records,
        window_size: int,
        enable_denoise: bool = False,
        original_indices: Optional[List[int]] = None,
        freq_threshold: Optional[float] = None,
        extractor=None,
    ):
        self._records        = records
        self._window_size    = window_size
        self._enable_denoise = enable_denoise
        self._freq_threshold = freq_threshold
        self._extractor      = None
        self._original_indices = (
            original_indices if original_indices is not None
            else list(range(len(records)))
        )

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        if self._extractor is None:
            self._extractor  = _make_extractor(
                enable_denoise=self._enable_denoise,
                freq_threshold=self._freq_threshold,
            )
            self._cache_path: Optional[str] = None
            self._cache_data                = None

        rec       = self._records[idx]
        file_path = rec.outplane_meta.get("file_path")

        if file_path != self._cache_path:
            self._cache_data = self._extractor.load_file(file_path)
            self._cache_path = file_path

        signal = self._extractor.extract_window_from_data(
            self._cache_data,
            rec.window_idx,
            self._window_size,
            metadata=rec.outplane_meta,
            file_path=file_path,
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

    def run(
        self, dataset
    ) -> Tuple[Dict[int, int], Dict[int, int], Dict[int, int]]:
        """
        对数据集所有样本同时执行面内、面外推理，并合并结果。

        Parameters
        ----------
        dataset : StayCableVib2023Dataset

        Returns
        -------
        Tuple[Dict[int, int], Dict[int, int], Dict[int, int]]
            (merged_predictions, inplane_predictions, outplane_predictions)

            - merged_predictions  : 任一方向非随机振动即取该特殊类别（面内优先）
            - inplane_predictions : 面内识别结果 ``{sample_idx: label}``
            - outplane_predictions: 面外识别结果 ``{sample_idx: label}``
        """
        freq_threshold = getattr(dataset.config, "denoise_freq_threshold", None)
        extractor = _make_extractor(
            enable_denoise=dataset.config.enable_denoise,
            freq_threshold=freq_threshold,
        )
        window_size    = dataset.config.window_size
        enable_denoise = dataset.config.enable_denoise
        n_orig         = len(dataset._samples)

        # ---- 面内推理 ---------------------------------------------------- #
        in_recs, in_idxs = self._validate_records(
            dataset._samples, window_size, extractor=extractor
        )
        in_recs, in_idxs = self._sort_by_meta_path(in_recs, in_idxs, "inplane")

        in_ds = _InplaneWindowDataset(
            records          = in_recs,
            window_size      = window_size,
            enable_denoise   = enable_denoise,
            original_indices = in_idxs,
            freq_threshold   = freq_threshold,
        )
        logger.info(
            f"[面内] 开始推理：{len(in_ds)} 个窗口"
            f"（丢弃 {n_orig - len(in_ds)} 个不完整窗口）| "
            f"batch={self.batch_size} | workers={self.num_workers} | "
            f"device={self.identifier.device}"
        )
        inplane_predictions = self._run_inference_loop(in_ds, "DL面内识别")
        del in_ds, in_recs, in_idxs

        # ---- 面外推理 ---------------------------------------------------- #
        out_recs, out_idxs = self._validate_outplane_records(
            dataset._samples, window_size, extractor=extractor
        )
        out_recs, out_idxs = self._sort_by_meta_path(out_recs, out_idxs, "outplane")

        out_ds = _OutplaneWindowDataset(
            records          = out_recs,
            window_size      = window_size,
            enable_denoise   = enable_denoise,
            original_indices = out_idxs,
            freq_threshold   = freq_threshold,
        )
        logger.info(
            f"[面外] 开始推理：{len(out_ds)} 个窗口"
            f"（{n_orig - len(out_ds)} 个窗口无有效面外文件，视为随机振动）| "
            f"batch={self.batch_size} | workers={self.num_workers} | "
            f"device={self.identifier.device}"
        )
        outplane_predictions = self._run_inference_loop(out_ds, "DL面外识别")
        del out_ds, out_recs, out_idxs
        del extractor

        # ---- 合并 --------------------------------------------------------- #
        merged_predictions = self._merge_predictions(
            inplane_predictions, outplane_predictions
        )

        import gc  # noqa: PLC0415
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info(
            f"识别完成 | 面内={len(inplane_predictions)} | "
            f"面外={len(outplane_predictions)} | 合并={len(merged_predictions)}"
        )
        return merged_predictions, inplane_predictions, outplane_predictions

    # ------------------------------------------------------------------ #
    # 概率推理接口                                                          #
    # ------------------------------------------------------------------ #

    def run_with_proba(
        self, dataset
    ) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray]]:
        """
        对数据集所有样本执行面内、面外 softmax 概率推理。

        Parameters
        ----------
        dataset : StayCableVib2023Dataset

        Returns
        -------
        Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray]]
            (inplane_probas, outplane_probas)
            每条为 {sample_idx: np.ndarray(shape=[num_classes], dtype=float32)}
        """
        freq_threshold = getattr(dataset.config, "denoise_freq_threshold", None)
        extractor = _make_extractor(
            enable_denoise=dataset.config.enable_denoise,
            freq_threshold=freq_threshold,
        )
        window_size    = dataset.config.window_size
        enable_denoise = dataset.config.enable_denoise

        # ---- 面内概率推理 ------------------------------------------------ #
        in_recs, in_idxs = self._validate_records(
            dataset._samples, window_size, extractor=extractor
        )
        in_recs, in_idxs = self._sort_by_meta_path(in_recs, in_idxs, "inplane")

        in_ds = _InplaneWindowDataset(
            records          = in_recs,
            window_size      = window_size,
            enable_denoise   = enable_denoise,
            original_indices = in_idxs,
            freq_threshold   = freq_threshold,
        )
        logger.info(
            f"[面内概率] 开始推理：{len(in_ds)} 个窗口 | "
            f"batch={self.batch_size} | workers={self.num_workers} | "
            f"device={self.identifier.device}"
        )
        inplane_probas = self._run_proba_loop(in_ds, "DL面内概率推理")
        del in_ds, in_recs, in_idxs

        # ---- 面外概率推理 ------------------------------------------------ #
        out_recs, out_idxs = self._validate_outplane_records(
            dataset._samples, window_size, extractor=extractor
        )
        out_recs, out_idxs = self._sort_by_meta_path(out_recs, out_idxs, "outplane")

        out_ds = _OutplaneWindowDataset(
            records          = out_recs,
            window_size      = window_size,
            enable_denoise   = enable_denoise,
            original_indices = out_idxs,
            freq_threshold   = freq_threshold,
        )
        logger.info(
            f"[面外概率] 开始推理：{len(out_ds)} 个窗口 | "
            f"batch={self.batch_size} | workers={self.num_workers} | "
            f"device={self.identifier.device}"
        )
        outplane_probas = self._run_proba_loop(out_ds, "DL面外概率推理")
        del out_ds, out_recs, out_idxs, extractor

        import gc  # noqa: PLC0415
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info(
            f"概率推理完成 | 面内={len(inplane_probas)} | 面外={len(outplane_probas)}"
        )
        return inplane_probas, outplane_probas

    # ------------------------------------------------------------------ #
    # 通用推理循环                                                          #
    # ------------------------------------------------------------------ #

    def _run_inference_loop(self, pred_ds: Dataset, desc: str) -> Dict[int, int]:
        """对给定 Dataset 执行 DataLoader 推理循环，返回 {original_idx: label}。"""
        loader_kwargs: dict = dict(
            batch_size         = self.batch_size,
            shuffle            = False,
            num_workers        = self.num_workers,
            pin_memory         = (self.identifier.device.type == "cuda"),
            persistent_workers = (self.num_workers > 0),
        )
        if self.num_workers > 0:
            loader_kwargs["prefetch_factor"] = 1

        loader      = DataLoader(pred_ds, **loader_kwargs)
        predictions: Dict[int, int] = {}

        with tqdm(total=len(pred_ds), desc=desc, unit="win") as pbar:
            for signals, indices in loader:
                preds = self.identifier.predict_batch(signals)
                for pred, idx in zip(preds, indices.numpy()):
                    predictions[int(idx)] = int(pred)
                pbar.update(len(indices))
                del signals, preds

        del loader
        return predictions

    def _run_proba_loop(self, pred_ds: Dataset, desc: str) -> Dict[int, np.ndarray]:
        """对给定 Dataset 执行 DataLoader 概率推理循环，返回 {original_idx: proba_array}。"""
        loader_kwargs: dict = dict(
            batch_size         = self.batch_size,
            shuffle            = False,
            num_workers        = self.num_workers,
            pin_memory         = (self.identifier.device.type == "cuda"),
            persistent_workers = (self.num_workers > 0),
        )
        if self.num_workers > 0:
            loader_kwargs["prefetch_factor"] = 1

        loader  = DataLoader(pred_ds, **loader_kwargs)
        probas: Dict[int, np.ndarray] = {}

        with tqdm(total=len(pred_ds), desc=desc, unit="win") as pbar:
            for signals, indices in loader:
                batch_probas = self.identifier.predict_batch_proba(signals)
                for proba, idx in zip(batch_probas, indices.numpy()):
                    probas[int(idx)] = proba
                pbar.update(len(indices))
                del signals, batch_probas

        del loader
        return probas

    # ------------------------------------------------------------------ #
    # 合并规则                                                              #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _merge_predictions(
        inplane: Dict[int, int],
        outplane: Dict[int, int],
    ) -> Dict[int, int]:
        """
        合并面内、面外预测结果。

        规则：任一方向非随机振动（label != _NORMAL_LABEL）则取该特殊类别；
        若两者均为特殊振动，面内优先。
        面外缺失的样本仅以面内结果为准；面内缺失的样本（极少）以面外为准。
        """
        all_indices = set(inplane.keys()) | set(outplane.keys())
        merged: Dict[int, int] = {}
        for idx in all_indices:
            in_pred  = inplane.get(idx, _NORMAL_LABEL)
            out_pred = outplane.get(idx, _NORMAL_LABEL)
            if in_pred != _NORMAL_LABEL:
                merged[idx] = in_pred
            elif out_pred != _NORMAL_LABEL:
                merged[idx] = out_pred
            else:
                merged[idx] = _NORMAL_LABEL
        return merged

    # ------------------------------------------------------------------ #
    # 排序辅助                                                              #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _sort_by_meta_path(
        records: list, orig_indices: List[int], direction: str
    ) -> Tuple[list, List[int]]:
        """按文件路径排序，使同文件窗口连续，提升单文件缓存命中率。"""
        attr = f"{direction}_meta"
        sorted_pairs = sorted(
            zip(records, orig_indices),
            key=lambda p: getattr(p[0], attr).get("file_path", ""),
        )
        return [p[0] for p in sorted_pairs], [p[1] for p in sorted_pairs]

    # ------------------------------------------------------------------ #
    # 预验证                                                                #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _validate_records(
        records, window_size: int, extractor=None
    ) -> Tuple[list, List[int]]:
        """
        读取每个面内 VIC 文件的真实数据长度，丢弃末尾不完整窗口。
        返回 (有效记录列表, 对应的原始索引列表)。
        """
        if extractor is None:
            extractor = _make_extractor(enable_denoise=False)
        file_length_cache: Dict[str, int] = {}

        seen: set = set()
        unique_fps_dedup = []
        for rec in records:
            fp = rec.inplane_meta.get("file_path")
            if fp and fp not in seen:
                seen.add(fp)
                unique_fps_dedup.append(fp)

        with tqdm(unique_fps_dedup, desc="[面内] 预验证文件长度", unit="file") as pbar:
            for fp in pbar:
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
                f"[面内] 预验证：{skipped} 个窗口超出文件实际长度（window_size={window_size}），已跳过"
            )

        return valid_records, valid_orig_indices

    @staticmethod
    def _validate_outplane_records(
        records, window_size: int, extractor=None
    ) -> Tuple[list, List[int]]:
        """
        读取每个面外 VIC 文件的真实数据长度，丢弃无效路径及末尾不完整窗口。
        没有有效面外文件的记录直接忽略（推理时缺失值将视为随机振动）。
        返回 (有效记录列表, 对应的原始索引列表)。
        """
        if extractor is None:
            extractor = _make_extractor(enable_denoise=False)
        file_length_cache: Dict[str, int] = {}

        seen: set = set()
        unique_fps_dedup = []
        for rec in records:
            fp = rec.outplane_meta.get("file_path") if rec.outplane_meta else None
            if fp and fp not in seen:
                seen.add(fp)
                unique_fps_dedup.append(fp)

        with tqdm(unique_fps_dedup, desc="[面外] 预验证文件长度", unit="file") as pbar:
            for fp in pbar:
                vic_data = extractor.unpacker.VIC_DATA_Unpack(str(fp))
                file_length_cache[fp] = len(vic_data)
                del vic_data

        valid_records: list = []
        valid_orig_indices: List[int] = []
        skipped_invalid = 0
        skipped_len     = 0

        for orig_idx, rec in enumerate(records):
            fp = rec.outplane_meta.get("file_path") if rec.outplane_meta else None
            if not fp:
                skipped_invalid += 1
                continue
            actual_len = file_length_cache.get(fp, 0)
            end_idx    = (rec.window_idx + 1) * window_size
            if actual_len > 0 and end_idx <= actual_len:
                valid_records.append(rec)
                valid_orig_indices.append(orig_idx)
            else:
                skipped_len += 1

        if skipped_invalid:
            logger.info(f"[面外] {skipped_invalid} 条记录无有效面外文件路径，将视为随机振动")
        if skipped_len:
            logger.warning(
                f"[面外] {skipped_len} 个窗口超出文件实际长度（window_size={window_size}），已跳过"
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
        inplane_predictions: Optional[Dict[int, int]] = None,
        outplane_predictions: Optional[Dict[int, int]] = None,
    ) -> None:
        """
        将识别结果持久化为 JSON。

        JSON 结构
        ---------
        - ``predictions``          : 合并后的最终预测（任一方向特殊振动即取之）。
        - ``predictions_inplane``  : 仅面内识别结果（可选）。
        - ``predictions_outplane`` : 仅面外识别结果（可选）。
        - ``sample_metadata``      : 每个样本的完整元数据（用于追溯和统计）。
        - ``by_file``              : 按（传感器, 月, 日, 时）分组的合并预测列表。
        """
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

        all_indices = set(predictions.keys())
        if inplane_predictions:
            all_indices |= set(inplane_predictions.keys())
        if outplane_predictions:
            all_indices |= set(outplane_predictions.keys())

        sample_metadata = {}
        for idx in all_indices:
            if idx < len(dataset._samples):
                rec = dataset._samples[idx]
                sample_metadata[str(idx)] = {
                    "cable_pair":          list(rec.cable_pair),
                    "timestamp":           list(rec.timestamp_key),
                    "window_idx":          rec.window_idx,
                    "inplane_sensor_id":   rec.inplane_meta.get("sensor_id"),
                    "outplane_sensor_id":  rec.outplane_meta.get("sensor_id") if rec.outplane_meta else None,
                    "inplane_file_path":   rec.inplane_meta.get("file_path"),
                    "outplane_file_path":  rec.outplane_meta.get("file_path") if rec.outplane_meta else None,
                    "missing_rate_in":     rec.inplane_meta.get("missing_rate"),
                    "missing_rate_out":    rec.outplane_meta.get("missing_rate") if rec.outplane_meta else None,
                    "has_wind":            rec.wind_meta is not None,
                }
            else:
                logger.warning(f"样本索引 {idx} 超出数据集范围（数据集共 {len(dataset._samples)} 个样本）")

        dataset_fingerprint = dataset._compute_fingerprint()

        payload = {
            "metadata": {
                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "num_samples": len(predictions),
                "num_classes": getattr(dataset, "_num_classes", 4),
                "model_info":  model_info,
                "dataset_fingerprint_hash": dataset._fingerprint_hash(dataset_fingerprint),
            },
            "predictions":          {str(k): int(v) for k, v in predictions.items()},
            "predictions_inplane":  {str(k): int(v) for k, v in (inplane_predictions or {}).items()},
            "predictions_outplane": {str(k): int(v) for k, v in (outplane_predictions or {}).items()},
            "sample_metadata":      sample_metadata,
            "by_file":              FullDatasetRunner.to_file_indexed(predictions, dataset),
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)

        logger.info(
            f"识别结果已保存：{path} | 合并={len(predictions)} | "
            f"面内={len(inplane_predictions or {})} | 面外={len(outplane_predictions or {})}"
        )


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

    @staticmethod
    def load_result(result_path: str) -> Dict:
        """
        加载完整识别结果 JSON，自动补全缺失的 sample_metadata。

        若同目录存在 ``<stem>_enriched<suffix>`` 文件则优先加载，
        否则在原文件基础上补全后写入 enriched 文件供后续复用。

        Parameters
        ----------
        result_path : str
            识别结果 JSON 路径（由 save_predictions() 生成）

        Returns
        -------
        Dict
            含 predictions / sample_metadata / by_file / metadata 等字段的字典
        """
        from pathlib import Path as _Path  # noqa: PLC0415

        result_p     = _Path(result_path)
        enriched_path = result_p.with_name(result_p.stem + "_enriched" + result_p.suffix)

        if enriched_path.exists():
            logger.info(f"检测到已补全文件，直接加载：{enriched_path}")
            with open(enriched_path, "r", encoding="utf-8") as f:
                return json.load(f)

        logger.info(f"加载识别结果：{result_path}")
        with open(result_path, "r", encoding="utf-8") as f:
            result = json.load(f)

        logger.info(
            f"  - 样本总数: {result['metadata']['num_samples']} | "
            f"类别数: {result['metadata']['num_classes']} | "
            f"生成时间: {result['metadata']['created_at']}"
        )

        if "sample_metadata" not in result or not result["sample_metadata"]:
            logger.warning("识别结果缺少 sample_metadata，将自动补全...")
            result = FullDatasetRunner._enrich_with_dataset_metadata(result)
            with open(enriched_path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            logger.info(f"补全结果已保存至：{enriched_path}")

        return result

    @staticmethod
    def _enrich_with_dataset_metadata(result: Dict) -> Dict:
        """
        从当前数据集配置为识别结果补全 sample_metadata 字段。

        仅在识别结果不含 sample_metadata 时调用（向后兼容旧格式）。
        """
        import yaml  # noqa: PLC0415
        from pathlib import Path as _Path  # noqa: PLC0415
        from src.config.data_processer.datasets.StayCableVib2023Dataset.StayCableVib2023Config import (  # noqa: PLC0415
            StayCableVib2023Config,
        )
        from src.data_processer.datasets.data_factory import get_dataset  # noqa: PLC0415

        project_root        = _Path(__file__).parent.parent.parent.parent
        dataset_config_path = project_root / "config" / "train" / "datasets" / "total_staycable_vib.yaml"

        with open(dataset_config_path, "r", encoding="utf-8") as f:
            dataset_config_dict = yaml.safe_load(f)

        dataset = get_dataset(StayCableVib2023Config(**dataset_config_dict))
        logger.info(f"数据集加载完成（{len(dataset)} 个样本）")

        fp_hash = dataset._fingerprint_hash(dataset._compute_fingerprint())
        original_hash = result.get("metadata", {}).get("dataset_fingerprint_hash")
        if original_hash and original_hash != fp_hash:
            logger.warning(
                f"数据集指纹不匹配！识别时：{original_hash}  当前：{fp_hash}\n"
                f"可能导致样本索引对应错误，请检查数据集配置是否变化。"
            )

        predictions     = {int(k): int(v) for k, v in result["predictions"].items()}
        sample_metadata = {}
        missing         = 0
        for idx in predictions:
            if idx < len(dataset._samples):
                rec = dataset._samples[idx]
                sample_metadata[str(idx)] = {
                    "cable_pair":         list(rec.cable_pair),
                    "timestamp":          list(rec.timestamp_key),
                    "window_idx":         rec.window_idx,
                    "inplane_sensor_id":  rec.inplane_meta.get("sensor_id"),
                    "outplane_sensor_id": rec.outplane_meta.get("sensor_id"),
                    "inplane_file_path":  rec.inplane_meta.get("file_path"),
                    "outplane_file_path": rec.outplane_meta.get("file_path"),
                    "missing_rate_in":    rec.inplane_meta.get("missing_rate"),
                    "missing_rate_out":   rec.outplane_meta.get("missing_rate"),
                    "has_wind":           rec.wind_meta is not None,
                }
            else:
                logger.warning(f"样本索引 {idx} 超出数据集范围（共 {len(dataset._samples)} 个）")
                missing += 1

        if missing:
            logger.warning(f"共 {missing} 个样本索引超出范围")

        result["sample_metadata"]                          = sample_metadata
        result["metadata"]["enrichment_note"]              = "sample_metadata 自动补全"
        result["metadata"]["dataset_fingerprint_hash"]     = fp_hash
        logger.info(f"补全完成，共 {len(sample_metadata)} 个样本")
        return result
