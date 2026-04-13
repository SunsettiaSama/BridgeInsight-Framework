import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

import hashlib
import json
import logging
import numpy as np
import torch
from torch.utils.data import Dataset, Subset
from typing import Any, Dict, List, Optional, Tuple

from src.config.data_processer.datasets.StayCableVib2023Dataset.StayCableVib2023Config import StayCableVib2023Config
from src.data_processer.preprocess.get_data_vib import VICWindowExtractor
from src.data_processer.preprocess.get_data_wind import parse_single_metadata_to_wind_data

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 类型别名
# ---------------------------------------------------------------------------
_TimestampKey = Tuple[int, int, int]   # (month, day, hour)


# ---------------------------------------------------------------------------
# 时间戳工具
# ---------------------------------------------------------------------------

def _make_timestamp_key(record: Dict) -> Optional[_TimestampKey]:
    """从元数据记录提取 (month, day, hour) key；缺字段时返回 None。"""
    m, d, h = record.get("month"), record.get("day"), record.get("hour")
    if m is None or d is None or h is None:
        return None
    return (int(m), int(d), int(h))


# ---------------------------------------------------------------------------
# 样本索引项（内存中保持轻量，原始数据按需加载）
# ---------------------------------------------------------------------------

class _SampleRecord:
    """
    一个样本的完整索引信息。

    三路元数据对应关系
    ------------------
    inplane_meta  : cable_pair[0] 传感器 + 当前 (month, day, hour)
    outplane_meta : cable_pair[1] 传感器 + 相同 (month, day, hour)
    wind_meta     : 全局风站    + 相同 (month, day, hour)

    三路均通过时间戳精确对齐后才构成一条有效样本；
    wind_meta 允许为 None（当 require_wind_alignment=False 时）。
    window_idx 标识该小时内第几个时间窗口（0-based）。
    """

    __slots__ = (
        "inplane_meta",
        "outplane_meta",
        "wind_meta",
        "window_idx",
        "cable_pair",
        "timestamp_key",
        "cable_pair_idx",   # 在 config.cable_pairs 中的位置，用于时序排序
    )

    def __init__(
        self,
        inplane_meta: Dict,
        outplane_meta: Dict,
        wind_meta: Optional[Dict],
        window_idx: int,
        cable_pair: Tuple[str, str],
        timestamp_key: _TimestampKey,
        cable_pair_idx: int = 0,
    ):
        self.inplane_meta   = inplane_meta
        self.outplane_meta  = outplane_meta
        self.wind_meta      = wind_meta
        self.window_idx     = window_idx
        self.cable_pair     = cable_pair
        self.timestamp_key  = timestamp_key
        self.cable_pair_idx = cable_pair_idx

    # ---- 序列化（用于缓存 JSON）----

    def to_dict(self) -> Dict:
        return {
            "inplane_meta":   self.inplane_meta,
            "outplane_meta":  self.outplane_meta,
            "wind_meta":      self.wind_meta,
            "window_idx":     self.window_idx,
            "cable_pair":     list(self.cable_pair),
            "timestamp_key":  list(self.timestamp_key),
            "cable_pair_idx": self.cable_pair_idx,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "_SampleRecord":
        return cls(
            inplane_meta   = d["inplane_meta"],
            outplane_meta  = d["outplane_meta"],
            wind_meta      = d.get("wind_meta"),
            window_idx     = d["window_idx"],
            cable_pair     = tuple(d["cable_pair"]),
            timestamp_key  = tuple(d["timestamp_key"]),
            cable_pair_idx = d.get("cable_pair_idx", 0),
        )


# ---------------------------------------------------------------------------
# 主数据集
# ---------------------------------------------------------------------------

class StayCableVib2023Dataset(Dataset):
    # 样本索引缓存版本号：窗口计算逻辑变更时递增，强制废弃旧缓存
    _INDEX_CACHE_VERSION = 2

    """
    苏通大桥拉索振动 2023 数据集

    设计目标
    --------
    面向全年统计分析与识别，而非仅服务于 ML 训练。
    支持：
    - 多根拉索（面内+面外传感器对）同时加载
    - 按时间戳对齐振动与风数据（三路严格对应）
    - 样本按时间序排列（time_ordered=True），保留时序连续性
    - 样本索引缓存（含配置指纹校验），避免重复排序的高开销
    - 训练/验证划分（随机或时序）以及全量数据集接口

    __getitem__ 返回结构
    --------------------
    {
        "inplane":  Tensor (window_size, 1)    面内振动加速度
        "outplane": Tensor (window_size, 1)    面外振动加速度
        "wind":     Dict | None                {wind_speed, wind_direction, wind_attack_angle}（ndarray）
        "metadata": Dict                       见下方说明
    }

    metadata 字段
    -------------
    cable_pair          : (面内传感器ID, 面外传感器ID)
    timestamp           : (month, day, hour)
    window_idx          : 当前小时内的窗口编号（0-based）
    inplane_sensor_id   : 面内传感器 ID
    outplane_sensor_id  : 面外传感器 ID
    inplane_file_path   : 面内 VIC 文件路径
    outplane_file_path  : 面外 VIC 文件路径
    missing_rate_in     : 面内数据缺失率
    missing_rate_out    : 面外数据缺失率
    has_wind            : 是否有对应风数据
    """

    def __init__(self, config: StayCableVib2023Config):
        self.config = config

        self._vib_extractor = VICWindowExtractor(
            enable_denoise=config.enable_denoise,
            freq_threshold=getattr(config, "denoise_freq_threshold", None),
        )
        self._file_length_cache: Dict[str, int] = {}

        # ---- 加载原始元数据 ----
        logger.info(f"加载振动元数据：{config.vib_metadata_path}")
        self._vib_meta_all: List[Dict] = _load_json(config.vib_metadata_path)
        logger.info(f"振动元数据共 {len(self._vib_meta_all)} 条")

        self._wind_meta_all: List[Dict] = []
        if config.wind_metadata_path is not None:
            logger.info(f"加载风参数元数据：{config.wind_metadata_path}")
            raw_wind = _load_json(config.wind_metadata_path)
            logger.info(f"风参数元数据共 {len(raw_wind)} 条（过滤前）")
            if config.wind_sensor_ids is not None:
                allowed = set(config.wind_sensor_ids)
                self._wind_meta_all = [r for r in raw_wind if r.get("sensor_id") in allowed]
                logger.info(
                    f"风传感器过滤：保留 {len(self._wind_meta_all)} 条"
                    f"（指定传感器：{config.wind_sensor_ids}）"
                )
            else:
                self._wind_meta_all = raw_wind
                logger.info("wind_sensor_ids=None，使用全量风元数据")

        # ---- 构建/加载样本索引 ----
        self._samples: List[_SampleRecord] = self._init_sample_index()
        logger.info(
            f"数据集就绪：{len(self._samples)} 个样本，"
            f"{len(config.cable_pairs)} 根拉索，"
            f"time_ordered={config.time_ordered}"
        )

        # ---- 划分训练/验证集 ----
        self._train_indices, self._val_indices = self._build_split_indices()

        # ---- DL 预识别结果（可选）----
        self._sample_predictions: Dict[int, int] = {}
        self._is_dl_identified: bool = False
        if config.predictions_cache_path and Path(config.predictions_cache_path).exists():
            self._load_predictions_from_cache(config.predictions_cache_path)

    # =====================================================================
    # 公开接口
    # =====================================================================

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> Dict:
        rec = self._samples[idx]
        inplane_data  = self._load_vib(rec.inplane_meta,  rec.window_idx)
        outplane_data = self._load_vib(rec.outplane_meta, rec.window_idx)
        wind_data     = self._load_wind(rec.wind_meta)

        return {
            "inplane":  torch.from_numpy(inplane_data).float(),
            "outplane": torch.from_numpy(outplane_data).float(),
            "wind":     wind_data,
            "metadata": {
                "cable_pair":          rec.cable_pair,
                "timestamp":           rec.timestamp_key,
                "window_idx":          rec.window_idx,
                "inplane_sensor_id":   rec.inplane_meta.get("sensor_id"),
                "outplane_sensor_id":  rec.outplane_meta.get("sensor_id"),
                "inplane_file_path":   rec.inplane_meta.get("file_path"),
                "outplane_file_path":  rec.outplane_meta.get("file_path"),
                "missing_rate_in":     rec.inplane_meta.get("missing_rate"),
                "missing_rate_out":    rec.outplane_meta.get("missing_rate"),
                "has_wind":            rec.wind_meta is not None,
                "dl_label":            self._sample_predictions.get(idx),
                "is_dl_identified":    self._is_dl_identified,
            },
        }

    def get_train_dataset(self) -> Subset:
        """训练集子集（随机或时序划分，取决于 split_by_time）。"""
        return Subset(self, self._train_indices)

    def get_val_dataset(self) -> Subset:
        """验证集子集。"""
        return Subset(self, self._val_indices)

    def get_full_dataset(self) -> "StayCableVib2023Dataset":
        """
        返回自身（全量数据集）。

        当 time_ordered=True 时，迭代顺序即时间顺序，
        适合按时序逐窗口应用识别方法、统计分布等。
        """
        return self

    # =====================================================================
    # DL 预识别结果管理
    # =====================================================================

    def apply_predictions(self, predictions: Dict[int, int]) -> None:
        """
        将 FullDatasetRunner.run() 的识别结果写入数据集。

        写入后 __getitem__ 返回的 metadata 中将包含：
          - ``dl_label``         : int，预测类别 ID（0~3）
          - ``is_dl_identified`` : True

        Parameters
        ----------
        predictions : Dict[int, int]
            ``{sample_idx: predicted_label}``
        """
        self._sample_predictions = predictions
        self._is_dl_identified   = True
        logger.info(f"DL 识别结果已应用，{len(predictions)} 个样本已标记")

    def save_predictions(self, path: str, model_info: str = "") -> None:
        """
        将当前识别结果持久化为 JSON 缓存文件。

        若尚未执行识别（is_dl_identified=False），则不写入。
        """
        if not self._is_dl_identified:
            logger.warning("尚未应用识别结果，跳过保存")
            return
        from src.identifier.deeplearning_methods.full_dataset_runner import FullDatasetRunner
        FullDatasetRunner.save_predictions(path, self._sample_predictions, self, model_info)

    def _load_predictions_from_cache(self, path: str) -> None:
        """从 JSON 缓存加载识别结果并应用。"""
        from src.identifier.deeplearning_methods.full_dataset_runner import FullDatasetRunner
        predictions = FullDatasetRunner.load_predictions(path)
        self.apply_predictions(predictions)
        logger.info(f"已从缓存自动加载 DL 识别结果：{path}")

    def get_metadata_list(self) -> List[Dict]:
        """
        返回所有样本的轻量元数据列表（不触发数据 I/O）。

        适合在计算前先做统计分析（样本数/拉索分布/时间覆盖等）。
        """
        return [
            {
                "cable_pair":         rec.cable_pair,
                "timestamp":          rec.timestamp_key,
                "window_idx":         rec.window_idx,
                "inplane_sensor_id":  rec.inplane_meta.get("sensor_id"),
                "outplane_sensor_id": rec.outplane_meta.get("sensor_id"),
                "has_wind":           rec.wind_meta is not None,
                "missing_rate_in":    rec.inplane_meta.get("missing_rate"),
                "missing_rate_out":   rec.outplane_meta.get("missing_rate"),
            }
            for rec in self._samples
        ]

    # =====================================================================
    # 样本索引：构建 / 缓存 / 校验
    # =====================================================================

    def _init_sample_index(self) -> List[_SampleRecord]:
        """
        样本索引入口：
          1. 若 use_cache=True，尝试读取并校验缓存
          2. 缓存有效则直接返回；否则重建（含可选时序排序）
          3. 重建后写入缓存供下次使用
        """
        if self.config.use_cache:
            cache_path = self._resolve_cache_path()
            fingerprint = self._compute_fingerprint()
            cached = self._try_load_cache(cache_path, fingerprint)
            if cached is not None:
                logger.info(f"缓存命中，直接加载（{len(cached)} 个样本）：{cache_path}")
                return cached

        samples = self._build_sample_index()

        if self.config.use_cache:
            self._save_cache(cache_path, fingerprint, samples)

        return samples

    def _build_sample_index(self) -> List[_SampleRecord]:
        """
        从原始元数据构建样本索引（核心逻辑，耗时主要来自时序排序）。

        三路对应规则
        ------------
        Step 1. 振动元数据按 sensor_id 分组，每组再按 (month, day, hour) 建立查找表
        Step 2. 风参数元数据按 (month, day, hour) 建立全局查找表
        Step 3. 遍历每根拉索对：
                  - 取面内/面外传感器共有时间戳
                  - 每个时间戳查找对应风数据（可选严格对齐）
                  - 每个时间戳按 actual_length // window_size 展开为多个窗口样本
        Step 4. 若 time_ordered=True，按 (month, day, hour, window_idx, cable_pair_idx) 排序
        """
        # --- Step 1: 振动分组 ---
        vib_by_sensor: Dict[str, Dict[_TimestampKey, Dict]] = {}
        for rec in self._vib_meta_all:
            sid = rec.get("sensor_id")
            key = _make_timestamp_key(rec)
            if sid is None or key is None:
                logger.debug(f"跳过缺少 sensor_id/时间字段的振动记录：{rec.get('file_path')}")
                continue
            vib_by_sensor.setdefault(sid, {})[key] = rec

        # --- Step 2: 风数据查找表 ---
        # 若指定了 wind_sensor_ids，按列表顺序决定同一时间戳的优先传感器（靠前优先）
        wind_by_ts: Dict[_TimestampKey, Dict] = {}
        if self.config.wind_sensor_ids is not None:
            priority = {sid: i for i, sid in enumerate(self.config.wind_sensor_ids)}
            for rec in self._wind_meta_all:
                key = _make_timestamp_key(rec)
                if key is None:
                    continue
                existing = wind_by_ts.get(key)
                if existing is None:
                    wind_by_ts[key] = rec
                else:
                    cur_pri = priority.get(rec.get("sensor_id"), len(priority))
                    old_pri = priority.get(existing.get("sensor_id"), len(priority))
                    if cur_pri < old_pri:
                        wind_by_ts[key] = rec
        else:
            for rec in self._wind_meta_all:
                key = _make_timestamp_key(rec)
                if key is not None:
                    wind_by_ts[key] = rec

        # --- Step 3: 枚举样本 ---
        samples: List[_SampleRecord] = []

        for pair_idx, (in_id, out_id) in enumerate(self.config.cable_pairs):
            in_lookup  = vib_by_sensor.get(in_id,  {})
            out_lookup = vib_by_sensor.get(out_id, {})

            if not in_lookup:
                logger.warning(f"面内传感器 {in_id} 无有效元数据，跳过")
                continue
            if not out_lookup:
                logger.warning(f"面外传感器 {out_id} 无有效元数据，跳过")
                continue

            common_ts = sorted(set(in_lookup) & set(out_lookup))
            logger.info(f"拉索 ({in_id}, {out_id})：共有时间戳 {len(common_ts)} 个")

            mr_threshold = self.config.missing_rate_threshold
            skipped_no_wind = 0
            skipped_missing  = 0
            for ts_key in common_ts:
                in_meta   = in_lookup[ts_key]
                out_meta  = out_lookup[ts_key]
                wind_meta = wind_by_ts.get(ts_key)

                if self.config.require_wind_alignment and wind_meta is None:
                    skipped_no_wind += 1
                    continue

                if mr_threshold is not None:
                    in_rate  = in_meta.get("missing_rate", 0.0) or 0.0
                    out_rate = out_meta.get("missing_rate", 0.0) or 0.0
                    if in_rate > mr_threshold or out_rate > mr_threshold:
                        skipped_missing += 1
                        continue

                num_win = self._estimate_num_windows(in_meta)
                for win_idx in range(num_win):
                    samples.append(
                        _SampleRecord(
                            inplane_meta   = in_meta,
                            outplane_meta  = out_meta,
                            wind_meta      = wind_meta,
                            window_idx     = win_idx,
                            cable_pair     = (in_id, out_id),
                            timestamp_key  = ts_key,
                            cable_pair_idx = pair_idx,
                        )
                    )

            if skipped_no_wind:
                logger.info(
                    f"  ({in_id}, {out_id})：{skipped_no_wind} 个时间戳因缺少风数据跳过"
                )
            if skipped_missing:
                logger.info(
                    f"  ({in_id}, {out_id})：{skipped_missing} 个时间戳因缺失率超标跳过"
                )

        logger.info(f"索引构建完成，共 {len(samples)} 个样本（排序前）")

        # --- Step 4: 可选时序排序（耗时操作）---
        if self.config.time_ordered:
            logger.info("开始按时间戳排序样本索引（time_ordered=True）……")
            samples.sort(
                key=lambda r: (r.timestamp_key[0], r.timestamp_key[1],
                               r.timestamp_key[2], r.window_idx, r.cable_pair_idx)
            )
            logger.info("时序排序完成")

        return samples

    def _estimate_num_windows(self, in_meta: Dict) -> int:
        """
        计算面内文件的完整窗口数。

        优先通过读取真实文件获取精确长度，确保不产生越界窗口。
        结果经 _file_length_cache 缓存，同一文件只读一次。
        """
        file_path = in_meta.get("file_path")
        if file_path:
            actual = self._get_inplane_file_length(file_path)
        else:
            actual = int(in_meta.get("actual_length") or 0)
        if actual > 0:
            return max(1, actual // self.config.window_size)
        return 1

    def _get_inplane_file_length(self, file_path: str) -> int:
        """读取面内 VIC 文件的实际数据长度（带实例级缓存）。"""
        if file_path not in self._file_length_cache:
            vic_data = self._vib_extractor.unpacker.VIC_DATA_Unpack(str(file_path))
            self._file_length_cache[file_path] = len(vic_data)
            del vic_data
        return self._file_length_cache[file_path]

    # =====================================================================
    # 缓存：路径 / 指纹 / 读写
    # =====================================================================

    def _resolve_cache_path(self) -> Path:
        """解析缓存文件路径：优先使用配置值，否则放在 vib_metadata 同目录。"""
        if self.config.cache_path is not None:
            p = Path(self.config.cache_path)
            if p.is_dir() or not p.suffix:
                return p / "staycable_vib2023_index_cache.json"
            return p
        return Path(self.config.vib_metadata_path).parent / "staycable_vib2023_index_cache.json"

    def _compute_fingerprint(self) -> Dict[str, Any]:
        """
        计算当前配置的缓存指纹。

        纳入指纹的字段
        --------------
        - cable_pairs（排序，避免顺序不同误判）
        - vib_metadata_path / wind_metadata_path
        - window_size / require_wind_alignment
        - time_ordered（有序/无序缓存不互通）
        - 两份元数据的记录数（快速一致性检查，不做全文 hash）

        不纳入指纹的字段
        ----------------
        - enable_denoise：只影响数据内容，不影响样本索引
        - split_ratio / split_by_time / split_seed：划分逻辑在索引加载后再计算
        """
        return {
            "index_cache_version":     self._INDEX_CACHE_VERSION,
            "cable_pairs":             sorted([list(p) for p in self.config.cable_pairs]),
            "vib_metadata_path":       str(self.config.vib_metadata_path),
            "wind_metadata_path":      str(self.config.wind_metadata_path),
            "wind_sensor_ids":         sorted(self.config.wind_sensor_ids) if self.config.wind_sensor_ids is not None else None,
            "window_size":             self.config.window_size,
            "require_wind_alignment":  self.config.require_wind_alignment,
            "missing_rate_threshold":  self.config.missing_rate_threshold,
            "time_ordered":            self.config.time_ordered,
            "vib_record_count":        len(self._vib_meta_all),
            "wind_record_count":       len(self._wind_meta_all),
        }

    @staticmethod
    def _fingerprint_hash(fp: Dict) -> str:
        """将指纹转成稳定的 SHA-256 hex string，方便日志追踪。"""
        raw = json.dumps(fp, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def _try_load_cache(
        self, cache_path: Path, fingerprint: Dict
    ) -> Optional[List[_SampleRecord]]:
        """
        尝试加载缓存并校验指纹。

        校验逻辑
        --------
        1. 缓存文件不存在 → 返回 None
        2. 读取 JSON，比对 fingerprint 字典是否完全一致
        3. 不一致（含 time_ordered 不同）→ 警告并返回 None，调用方将重建并覆盖
        4. 一致 → 反序列化并返回
        """
        if not cache_path.exists():
            logger.info(f"缓存文件不存在，将重新构建：{cache_path}")
            return None

        logger.info(f"发现缓存文件，校验指纹……：{cache_path}")
        raw = _load_json(str(cache_path))
        cached_fp = raw.get("fingerprint", {})

        if cached_fp != fingerprint:
            logger.warning(
                "缓存指纹不一致，将重建并覆盖缓存。\n"
                f"  期望指纹 hash：{self._fingerprint_hash(fingerprint)}\n"
                f"  缓存指纹 hash：{self._fingerprint_hash(cached_fp)}\n"
                f"  差异字段：{_dict_diff(cached_fp, fingerprint)}"
            )
            return None

        samples = [_SampleRecord.from_dict(d) for d in raw["samples"]]
        logger.info(
            f"缓存指纹校验通过（hash={self._fingerprint_hash(fingerprint)}），"
            f"加载 {len(samples)} 个样本"
        )
        return samples

    def _save_cache(
        self, cache_path: Path, fingerprint: Dict, samples: List[_SampleRecord]
    ) -> None:
        """将样本索引序列化并写入缓存文件。"""
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "fingerprint": fingerprint,
            "fingerprint_hash": self._fingerprint_hash(fingerprint),
            "num_samples": len(samples),
            "samples": [s.to_dict() for s in samples],
        }
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)
        logger.info(
            f"样本索引已缓存（{len(samples)} 条，"
            f"hash={self._fingerprint_hash(fingerprint)}）：{cache_path}"
        )

    # =====================================================================
    # 训练/验证集划分
    # =====================================================================

    def _build_split_indices(self) -> Tuple[List[int], List[int]]:
        """
        构建训练/验证集索引。

        split_ratio == -1
            不划分，全量数据集（训练集为全部索引，验证集为空）

        split_by_time=True
            按样本在列表中的位置顺序划分：前 split_ratio 为训练集。
            若同时设置 time_ordered=True，则训练集覆盖时间轴前段，
            验证集覆盖后段，符合时序预测的评估范式。

        split_by_time=False
            随机打乱后按比例划分（传统 ML 划分）。
        """
        n = len(self._samples)

        # ---- 全量数据集模式（不划分）----
        if self.config.split_ratio == -1:
            train_idx = list(range(n))
            val_idx = []
            logger.info(f"数据集模式：全量数据（不划分），共 {n} 个样本")
            return train_idx, val_idx

        # ---- 正常划分模式 ----
        train_size = int(n * self.config.split_ratio)

        if self.config.split_by_time:
            train_idx = list(range(train_size))
            val_idx   = list(range(train_size, n))
        else:
            rng = np.random.default_rng(self.config.split_seed)
            perm = rng.permutation(n).tolist()
            train_idx = sorted(perm[:train_size])
            val_idx   = sorted(perm[train_size:])

        logger.info(
            f"数据集划分（split_by_time={self.config.split_by_time}）："
            f"训练集 {len(train_idx)} / 验证集 {len(val_idx)}"
        )
        return train_idx, val_idx

    # =====================================================================
    # 数据加载
    # =====================================================================

    def _load_vib(self, meta: Dict, window_idx: int) -> np.ndarray:
        """按需加载单个振动窗口，shape = (window_size, 1)。"""
        return self._vib_extractor.extract_window(
            meta.get("file_path"),
            window_idx,
            self.config.window_size,
            metadata=meta,
        )

    def _load_wind(self, meta: Optional[Dict]) -> Optional[Dict]:
        """
        按需加载风参数数据。

        返回 {wind_speed, wind_direction, wind_attack_angle}（ndarray）或 None。
        """
        if meta is None:
            return None
        result = parse_single_metadata_to_wind_data(meta, enable_denoise=self.config.enable_denoise)
        return result.get("data")


# ---------------------------------------------------------------------------
# 模块级工具
# ---------------------------------------------------------------------------

def _load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _dict_diff(a: Dict, b: Dict) -> Dict:
    """返回两个字典中值不同的键及其对应值，便于缓存不一致时定位原因。"""
    all_keys = set(a) | set(b)
    return {
        k: {"cached": a.get(k), "current": b.get(k)}
        for k in all_keys
        if a.get(k) != b.get(k)
    }
