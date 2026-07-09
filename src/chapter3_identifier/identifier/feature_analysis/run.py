"""
全量识别结果后处理流水线

功能
----
1. 加载已有的 enriched 识别结果 JSON（含 sample_metadata + predictions）
2. 预计算各时间戳的多传感器风统计量（均值/标准差/紊流度）
3. 对每个样本并行计算特征（由配置开关控制）：
   - PSD 前 N 阶主导模态（频率 + 功率）
   - 谱熵 / 谱带宽 / 主频能量占比
   - 时域统计：RMS / 峭度 / 偏度 / 波峰因子 / 过零率
   - 面内外耦合：互相关 / 椭圆率 / 相干性 / 相位差
   - 折减风速（需配置拉索外径）
4. 按振动类别（可选再按传感器）分割保存

使用方式
--------
    python -m src.identifier.feature_analysis.run \
        --result   results/identification_result/res_cnn_full_dataset_*_enriched.json \
        --wind     results/metadata/wind_metadata_filtered.json \
        --output   results/enriched_stats \
        --config   config/identifier/feature_analysis/default.yaml
"""

import sys
import json
import logging
import argparse
import types
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

_SRC_DIR = Path(__file__).resolve().parent.parent.parent.parent
_REPO_ROOT = _SRC_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

if "src.identifier" not in sys.modules:
    _identifier_pkg = types.ModuleType("src.identifier")
    _identifier_pkg.__path__ = [str(_SRC_DIR / "chapter3_identifier" / "identifier")]
    sys.modules["src.identifier"] = _identifier_pkg

from src.data_processer.preprocess.get_data_vib import VICWindowExtractor
from src.config.identifier.feature_analysis.config import (
    ProcessFullDataConfig,
    load_config,
)
from src.identifier.feature_analysis._modal import (
    compute_psd_top_modes,
    compute_spectral_features,
)
from src.identifier.feature_analysis._signal import compute_time_stats
from src.identifier.feature_analysis._coupling import compute_cross_coupling
from src.identifier.feature_analysis._wind import (
    build_wind_lookup,
    compute_wind_stats_by_timestamp,
    get_wind_stats_for_sample,
    load_wind_metadata,
    compute_reduced_velocity,
)
from src.identifier.feature_analysis._splitter import save_class_results

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 全局配置（多进程 worker 通过 initializer 注入）
# ---------------------------------------------------------------------------
_worker_cfg: Optional[ProcessFullDataConfig] = None


def _init_worker(cfg: ProcessFullDataConfig) -> None:
    global _worker_cfg
    _worker_cfg = cfg


# ---------------------------------------------------------------------------
# 多进程 Worker（顶层函数，保证 Windows spawn 可 pickle）
# ---------------------------------------------------------------------------

def _worker_compute_features(args: Tuple) -> Optional[Dict]:
    """
    对单个样本加载信号并计算所有已开启特征。
    返回特征字典，越界时返回 None。

    去噪：由 VICWindowExtractor 统一处理（分层策略），
    metadata 为 None 时自动降级为实时 FFT 计算主频。
    """
    cfg = _worker_cfg
    sample_idx, in_path, out_path, window_idx = args

    extractor = VICWindowExtractor(
        enable_denoise=cfg.enable_denoise,
        freq_threshold=cfg.denoise_freq_threshold,
    )

    def _load(path: str) -> Optional[np.ndarray]:
        # metadata=None：dominant_freq_per_window 不可用时走实时 FFT fallback
        signal = extractor.extract_window(path, window_idx, cfg.window_size, metadata=None)
        return signal.ravel() if signal is not None else None

    in_raw  = _load(in_path)
    out_raw = _load(out_path)

    if in_raw is None or out_raw is None:
        return None

    result: Dict = {"sample_idx": sample_idx}

    # ---- PSD 主导模态 ----
    if cfg.enable_psd_modes:
        result["psd_inplane"]  = compute_psd_top_modes(
            in_raw,  cfg.fs, cfg.psd_n_modes, cfg.psd_nperseg, cfg.psd_min_peak_distance_hz
        )
        result["psd_outplane"] = compute_psd_top_modes(
            out_raw, cfg.fs, cfg.psd_n_modes, cfg.psd_nperseg, cfg.psd_min_peak_distance_hz
        )

    # ---- 谱统计特征 ----
    if cfg.enable_spectral_features:
        result["spectral_inplane"]  = compute_spectral_features(
            in_raw,  cfg.fs, cfg.psd_nperseg, cfg.psd_min_peak_distance_hz, cfg.psd_n_modes
        )
        result["spectral_outplane"] = compute_spectral_features(
            out_raw, cfg.fs, cfg.psd_nperseg, cfg.psd_min_peak_distance_hz, cfg.psd_n_modes
        )

    # ---- 时域统计 ----
    if cfg.enable_time_stats:
        result["time_stats_inplane"]  = compute_time_stats(in_raw)
        result["time_stats_outplane"] = compute_time_stats(out_raw)

    # ---- 面内外耦合 ----
    if cfg.enable_cross_coupling:
        result["cross_coupling"] = compute_cross_coupling(
            in_raw, out_raw, cfg.fs, cfg.psd_nperseg
        )

    return result


def _iter_batches(items: List[Tuple], batch_size: int) -> List[Tuple]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def _assemble_enriched_record(
    idx: int,
    pred_cls: int,
    meta: Dict,
    feats: Dict,
    wind_stats_by_ts: Dict,
    cfg: ProcessFullDataConfig,
) -> Dict:
    timestamp = meta.get("timestamp", [])
    wind_stats = (
        get_wind_stats_for_sample(timestamp, wind_stats_by_ts)
        if cfg.enable_wind_stats else []
    )

    record: Dict = {
        "sample_idx":         idx,
        "predicted_class":    pred_cls,
        "cable_pair":         meta.get("cable_pair"),
        "timestamp":          timestamp,
        "window_idx":         meta.get("window_idx"),
        "inplane_sensor_id":  meta.get("inplane_sensor_id"),
        "outplane_sensor_id": meta.get("outplane_sensor_id"),
        "inplane_file_path":  meta.get("inplane_file_path"),
        "outplane_file_path": meta.get("outplane_file_path"),
        "missing_rate_in":    meta.get("missing_rate_in"),
        "missing_rate_out":   meta.get("missing_rate_out"),
        "has_wind":           meta.get("has_wind", False),
    }

    if cfg.enable_psd_modes:
        record["psd_inplane"] = feats.get("psd_inplane")
        record["psd_outplane"] = feats.get("psd_outplane")

    if cfg.enable_spectral_features:
        record["spectral_inplane"] = feats.get("spectral_inplane")
        record["spectral_outplane"] = feats.get("spectral_outplane")

    if cfg.enable_time_stats:
        record["time_stats_inplane"] = feats.get("time_stats_inplane")
        record["time_stats_outplane"] = feats.get("time_stats_outplane")

    if cfg.enable_cross_coupling:
        record["cross_coupling"] = feats.get("cross_coupling")

    if cfg.enable_wind_stats:
        record["wind_stats"] = wind_stats

    if cfg.enable_reduced_velocity and cfg.cable_diameter_map:
        inplane_sensor_id = meta.get("inplane_sensor_id", "")
        cable_diameter = cfg.cable_diameter_map.get(inplane_sensor_id)
        if cable_diameter is None:
            logger.warning(
                f"样本 {idx}：传感器 '{inplane_sensor_id}' 不在 cable_diameter_map 中，"
                "跳过折减风速计算"
            )
            record["reduced_velocity"] = []
        else:
            vr_list = []
            psd_in = record.get("psd_inplane") or {}
            freqs = psd_in.get("frequencies", [])
            f1 = freqs[0] if freqs else None
            for ws in wind_stats:
                U = ws.get("mean_wind_speed")
                if f1 and U is not None:
                    vr = compute_reduced_velocity(U, f1, cable_diameter)
                    vr_list.append({
                        "sensor_id": ws.get("sensor_id", ""),
                        "reduced_velocity": vr,
                    })
            record["reduced_velocity"] = vr_list

    return record


# ---------------------------------------------------------------------------
# 主流水线
# ---------------------------------------------------------------------------

def run(
    result_path: str,
    wind_metadata_path: str,
    output_dir: str,
    cfg: Optional[ProcessFullDataConfig] = None,
    config_yaml: Optional[str] = None,
    limit: Optional[int] = None,
    excluded_sensor_ids: Optional[set[str]] = None,
) -> None:
    if cfg is None:
        cfg = load_config(config_yaml)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("配置摘要：")
    for field, val in cfg.dict().items():
        logger.info(f"  {field}: {val}")

    # ------------------------------------------------------------------
    # 步骤 1：加载识别结果
    # ------------------------------------------------------------------
    logger.info("=" * 72)
    logger.info("步骤 1/5  加载识别结果")
    logger.info("=" * 72)

    with open(result_path, "r", encoding="utf-8") as f:
        result = json.load(f)

    predictions: Dict[str, int]    = result.get("predictions", {})
    sample_metadata: Dict[str, Dict] = result.get("sample_metadata", {})
    if excluded_sensor_ids:
        keep_keys = [
            key for key in predictions
            if (
                sample_metadata.get(str(key), {}).get("inplane_sensor_id") not in excluded_sensor_ids
                and sample_metadata.get(str(key), {}).get("outplane_sensor_id") not in excluded_sensor_ids
            )
        ]
        removed = len(predictions) - len(keep_keys)
        predictions = {key: predictions[key] for key in keep_keys}
        sample_metadata = {key: sample_metadata[key] for key in keep_keys if key in sample_metadata}
        logger.info(f"已剔除传感器样本：{removed} 条")
    if limit is not None and int(limit) > 0:
        keys = sorted(predictions.keys(), key=lambda k: int(k))[: int(limit)]
        predictions = {key: predictions[key] for key in keys}
        sample_metadata = {key: sample_metadata[key] for key in keys if key in sample_metadata}
        logger.info(f"应用 limit={limit}，保留 {len(predictions)} 条")
    logger.info(f"预测结果：{len(predictions)} 条  |  元数据：{len(sample_metadata)} 条")

    # ------------------------------------------------------------------
    # 步骤 2：预计算风统计量
    # ------------------------------------------------------------------
    wind_stats_by_ts: Dict = {}
    if cfg.enable_wind_stats:
        logger.info("=" * 72)
        logger.info("步骤 2/5  预计算风统计量（所有传感器×时间戳）")
        logger.info("=" * 72)

        wind_meta_list = load_wind_metadata(wind_metadata_path)
        logger.info(f"风元数据共 {len(wind_meta_list)} 条记录")
        wind_lookup = build_wind_lookup(wind_meta_list)
        logger.info(f"去重后共 {len(wind_lookup)} 个时间戳")
        wind_stats_by_ts = compute_wind_stats_by_timestamp(wind_lookup)
    else:
        logger.info("步骤 2/5  风统计量计算已关闭（enable_wind_stats=False），跳过")

    # ------------------------------------------------------------------
    # 步骤 3：并行计算振动特征
    # ------------------------------------------------------------------
    logger.info("=" * 72)
    logger.info(f"步骤 3/5  并行计算振动特征（{cfg.n_workers} 进程，batch_size={cfg.batch_size}）")
    logger.info("=" * 72)

    worker_args: List[Tuple] = [
        (int(idx_str),
         meta["inplane_file_path"],
         meta["outplane_file_path"],
         meta["window_idx"])
        for idx_str, meta in sample_metadata.items()
        if meta.get("inplane_file_path") and meta.get("outplane_file_path")
           and meta.get("window_idx") is not None
    ]
    logger.info(f"共 {len(worker_args)} 个样本待处理")

    skipped = 0
    processed = 0
    batches = _iter_batches(worker_args, int(cfg.batch_size))
    saved_paths: Dict[int, Path] = {}

    def _save_batch(batch_idx: int, batch_features: Dict[int, Dict], batch_args: List[Tuple]) -> None:
        batch_ids = {int(args[0]) for args in batch_args}
        enriched_samples: List[Dict] = []
        for idx_str, pred_cls in predictions.items():
            idx = int(idx_str)
            if idx not in batch_ids:
                continue
            meta = sample_metadata.get(idx_str)
            if meta is None:
                continue
            record = _assemble_enriched_record(
                idx=idx,
                pred_cls=pred_cls,
                meta=meta,
                feats=batch_features.get(idx, {}),
                wind_stats_by_ts=wind_stats_by_ts,
                cfg=cfg,
            )
            enriched_samples.append(record)

        if not enriched_samples:
            logger.info(f"batch {batch_idx} 无可保存样本")
            return

        suffix = f"_batch_{batch_idx:05d}"
        saved = save_class_results(
            enriched_samples=enriched_samples,
            output_dir=str(out_dir),
            source_result_path=str(Path(result_path).resolve()),
            wind_metadata_path=str(Path(wind_metadata_path).resolve()) if cfg.enable_wind_stats else "",
            split_by_sensor=cfg.split_by_sensor,
            file_suffix=suffix,
        )
        saved_paths.update(saved)
        logger.info(f"batch {batch_idx} 保存完成：{len(enriched_samples)} 条，suffix={suffix}")

    if cfg.n_workers <= 0:
        _init_worker(cfg)
        for batch_idx, batch in enumerate(batches, start=1):
            logger.info(f"处理 batch {batch_idx}：{len(batch)} 条")
            batch_features: Dict[int, Dict] = {}
            for args in tqdm(
                batch,
                total=len(batch),
                desc=f"特征计算 batch {batch_idx}",
                unit="样本",
                dynamic_ncols=True,
            ):
                r = _worker_compute_features(args)
                processed += 1
                if r is None:
                    skipped += 1
                    continue
                idx = r.pop("sample_idx")
                batch_features[idx] = r
            _save_batch(batch_idx, batch_features, batch)
            logger.info(f"batch {batch_idx} 完成：累计 {processed}/{len(worker_args)}")
    else:
        with Pool(
            processes=cfg.n_workers,
            initializer=_init_worker,
            initargs=(cfg,),
        ) as pool:
            for batch_idx, batch in enumerate(batches, start=1):
                logger.info(f"处理 batch {batch_idx}：{len(batch)} 条")
                batch_features: Dict[int, Dict] = {}
                for r in tqdm(
                    pool.imap_unordered(_worker_compute_features, batch),
                    total=len(batch),
                    desc=f"特征计算 batch {batch_idx}",
                    unit="样本",
                    dynamic_ncols=True,
                ):
                    processed += 1
                    if r is None:
                        skipped += 1
                        continue
                    idx = r.pop("sample_idx")
                    batch_features[idx] = r
                _save_batch(batch_idx, batch_features, batch)
                logger.info(f"batch {batch_idx} 完成：累计 {processed}/{len(worker_args)}")

    logger.info(f"特征计算完成：{processed - skipped} 条成功，{skipped} 条越界跳过")

    logger.info("=" * 72)
    logger.info(f"全部完成！共生成 {len(saved_paths)} 个类别目录/文件")
    logger.info(f"输出目录：{out_dir}")
    logger.info("=" * 72)


# ---------------------------------------------------------------------------
# CLI 入口
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="全量识别结果后处理：多特征计算 + 类别分割保存"
    )
    parser.add_argument("--result",  required=True,  help="enriched 识别结果 JSON 路径")
    parser.add_argument("--wind",    required=True,  help="风元数据 JSON 路径")
    parser.add_argument("--output",  required=True,  help="输出目录")
    parser.add_argument(
        "--config", default=None,
        help="配置 YAML 路径（默认读取 config/identifier/feature_analysis/default.yaml）",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run(
        result_path=args.result,
        wind_metadata_path=args.wind,
        output_dir=args.output,
        config_yaml=args.config,
    )


if __name__ == "__main__":
    main()
