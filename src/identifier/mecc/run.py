import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import logging
import yaml
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

from src.config.data_processer.datasets.StayCableVib2023Dataset.StayCableVib2023Config import (
    StayCableVib2023Config,
)
from src.data_processer.datasets.data_factory import get_dataset
from src.data_processer.preprocess.get_data_vib import VICWindowExtractor
from src.identifier.cable_analysis_methods.base_mode_calculator import Cal_Mount
from src.identifier.cable_analysis_methods.mecc import Abnormal_Vibration_Filter
from src.identifier.deeplearning_methods.full_dataset_runner import FullDatasetRunner

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

_DEFAULT_CONFIG = (
    Path(__file__).parent.parent.parent.parent
    / "config"
    / "identifier"
    / "mecc"
    / "default.yaml"
)


def _load_run_config(config_path: Path) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# 多进程 Worker（顶层函数，保证 Windows spawn 可 pickle）
# ---------------------------------------------------------------------------

def _process_file_group_worker(
    args: Tuple[str, List[Tuple[int, int, Dict]], Dict[str, float], dict, int, int, bool]
) -> Dict[int, int]:
    """
    单文件组 MECC 分类 Worker。

    每个 Worker 在子进程内独立实例化 VICWindowExtractor 和 Abnormal_Vibration_Filter，
    加载 VIC 文件一次后顺序处理该文件的所有窗口。

    Parameters
    ----------
    args : (file_path, windows, f0_cache, mecc_params, window_size, fs, viv_only)
        windows : [(orig_idx, window_idx, meta_dict), ...]
            meta_dict 仅保留 sensor_id，避免 _SampleRecord 跨进程传输
    """
    file_path, windows, f0_cache, mecc_params, window_size, fs, viv_only = args

    extractor = VICWindowExtractor(enable_denoise=False)
    mecc_filter = Abnormal_Vibration_Filter(fs=fs)

    vic_data = extractor.load_file(file_path)

    results: Dict[int, int] = {}
    for orig_idx, window_idx, meta_dict in windows:
        sensor_id = meta_dict.get("sensor_id")
        f0 = f0_cache.get(sensor_id)
        if f0 is None:
            continue

        signal = extractor.extract_window_from_data(
            vic_data,
            window_idx,
            window_size,
            metadata=meta_dict,
            file_path=file_path,
        )
        if signal is None:
            continue

        # extract_window_from_data 返回 (N, 1) 列向量（为 PyTorch 设计），
        # MECC 的 welch 需要 1D 数组，否则沿 axis=-1 计算长度为 1 的谱，结果错误。
        signal = signal.squeeze()

        pred = mecc_filter.classify_vibration(signal, f0, **mecc_params)

        if viv_only:
            if pred == 1:
                results[orig_idx] = 1
        else:
            results[orig_idx] = int(pred)

    return results


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------

def _build_f0_cache(samples) -> Dict[str, float]:
    sensor_ids: set = set()
    for rec in samples:
        for attr in ("inplane_meta", "outplane_meta"):
            meta = getattr(rec, attr, None)
            if meta:
                sid = meta.get("sensor_id")
                if sid:
                    sensor_ids.add(sid)

    cache: Dict[str, float] = {}
    for sid in sorted(sensor_ids):
        mount = Cal_Mount.from_sensor(sid)
        cache[sid] = mount.inplane_mode(1)
        logger.info(f"  {sid} → f0 = {cache[sid]:.4f} Hz")
    return cache


def _group_by_file(
    samples, direction: str
) -> Dict[str, List[Tuple[int, int, dict]]]:
    """
    按 VIC 文件路径分组，每组内只保留轻量元数据，避免 _SampleRecord 跨进程序列化。

    Returns
    -------
    {file_path: [(orig_idx, window_idx, {"sensor_id": ...}), ...]}
    """
    attr = f"{direction}_meta"
    groups: Dict[str, List] = defaultdict(list)
    for orig_idx, rec in enumerate(samples):
        meta = getattr(rec, attr)
        if meta is None:
            continue
        fp = meta.get("file_path")
        if not fp:
            continue
        groups[fp].append((
            orig_idx,
            rec.window_idx,
            {"sensor_id": meta.get("sensor_id"), "file_path": fp},
        ))
    return dict(groups)


def _run_mecc_parallel(
    samples,
    direction: str,
    f0_cache: Dict[str, float],
    mecc_params: dict,
    window_size: int,
    fs: int,
    num_workers: int,
    viv_only: bool,
    desc: str,
) -> Dict[int, int]:
    """
    并行 MECC 识别：按文件分组，每个子进程负责一个 VIC 文件的所有窗口。
    """
    file_groups = _group_by_file(samples, direction)
    n_files = len(file_groups)
    n_windows = sum(len(v) for v in file_groups.values())
    logger.info(f"[{direction}] {n_files} 个文件，{n_windows} 个窗口，workers={num_workers}")

    task_args = [
        (fp, windows, f0_cache, mecc_params, window_size, fs, viv_only)
        for fp, windows in file_groups.items()
    ]

    predictions: Dict[int, int] = {}
    skipped = 0

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(_process_file_group_worker, arg): arg[0] for arg in task_args}
        with tqdm(total=n_windows, desc=desc, unit="win") as pbar:
            for future in as_completed(futures):
                fp = futures[future]
                file_result = future.result()
                predictions.update(file_result)
                n = len(file_groups[fp])
                pbar.update(n)
                if len(file_result) < n:
                    skipped += n - len(file_result)

    if skipped:
        logger.warning(f"[{direction}] {skipped} 个窗口因信号无效、f0 缺失或非目标类别被跳过")

    return predictions


# ---------------------------------------------------------------------------
# 主工作流
# ---------------------------------------------------------------------------

def main(config_path: Optional[Path] = None):
    project_root = Path(__file__).parent.parent.parent.parent
    config_path = Path(config_path) if config_path else _DEFAULT_CONFIG

    logger.info(f"加载 MECC 识别配置：{config_path}")
    cfg = _load_run_config(config_path)

    # ---- 解析配置 ----
    dataset_config_path = project_root / cfg["dataset_config"]
    fs            = int(cfg.get("fs", 50))
    num_workers   = int(cfg.get("num_workers", 8))
    viv_only      = bool(cfg.get("viv_only", True))
    output_dir    = project_root / cfg.get("output_dir", "results/identification_result")
    mecc_params   = {k: v for k, v in cfg["mecc"].items()}

    suffix    = "viv_only" if viv_only else "full"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"mecc_{suffix}_{timestamp}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # 1. 加载数据集
    # -------------------------------------------------------------------------
    logger.info("=" * 80)
    logger.info("步骤 1/4: 加载 StayCable_Vib2023 数据集")
    logger.info("=" * 80)

    with open(dataset_config_path, "r", encoding="utf-8") as f:
        dataset_config = StayCableVib2023Config(**yaml.safe_load(f))

    dataset = get_dataset(dataset_config)
    logger.info(f"数据集加载完成，共 {len(dataset)} 个样本")

    # -------------------------------------------------------------------------
    # 2. 预计算各传感器对应拉索基频 f0
    # -------------------------------------------------------------------------
    logger.info("=" * 80)
    logger.info("步骤 2/4: 预计算各拉索基频 f0")
    logger.info("=" * 80)

    f0_cache = _build_f0_cache(dataset._samples)
    logger.info(f"基频缓存完成，共 {len(f0_cache)} 个传感器")

    # -------------------------------------------------------------------------
    # 3. 并行 MECC 识别（面内 + 面外）
    # -------------------------------------------------------------------------
    logger.info("=" * 80)
    logger.info("步骤 3/4: 执行全量 MECC 识别（多进程）")
    logger.info(f"  识别目标：{'仅 VIV（label=1）' if viv_only else '全量结果（0=一般振动, 1=VIV）'}")
    logger.info(f"  MECC 参数：{mecc_params}")
    logger.info(f"  workers  ：{num_workers}")
    logger.info("=" * 80)

    window_size = dataset_config.window_size

    inplane_predictions = _run_mecc_parallel(
        dataset._samples, "inplane",
        f0_cache, mecc_params, window_size, fs, num_workers, viv_only,
        "MECC 面内识别",
    )
    logger.info(f"[面内] 识别完成，共 {len(inplane_predictions)} 个窗口")

    outplane_predictions = _run_mecc_parallel(
        dataset._samples, "outplane",
        f0_cache, mecc_params, window_size, fs, num_workers, viv_only,
        "MECC 面外识别",
    )
    logger.info(f"[面外] 识别完成，共 {len(outplane_predictions)} 个窗口")

    # -------------------------------------------------------------------------
    # 4. 合并预测结果并保存
    # -------------------------------------------------------------------------
    logger.info("=" * 80)
    logger.info("步骤 4/4: 合并识别结果并保存")
    logger.info("=" * 80)

    merged_predictions = FullDatasetRunner._merge_predictions(
        inplane_predictions, outplane_predictions
    )
    logger.info(
        f"合并完成 | 合并={len(merged_predictions)} | "
        f"面内={len(inplane_predictions)} | 面外={len(outplane_predictions)}"
    )

    viv_mode_note = "仅保存 VIV 阳性结果（label=1）" if viv_only else "全量结果（0=一般振动, 1=VIV）"
    model_info = (
        f"MECC | {viv_mode_note} | "
        f"sigma_0={mecc_params['sigma_0']}, "
        f"freq_min={mecc_params['freq_min']} Hz, "
        f"k_viv={mecc_params['k_viv']}, C_viv={mecc_params['C_viv']}"
    )

    FullDatasetRunner.save_predictions(
        path=str(output_path),
        predictions=merged_predictions,
        dataset=dataset,
        model_info=model_info,
        inplane_predictions=inplane_predictions,
        outplane_predictions=outplane_predictions,
    )

    logger.info("=" * 80)
    logger.info("全量 MECC 识别工作流执行完成！")
    logger.info(f"结果路径：{output_path}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
