from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import json
import logging
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import yaml
from tqdm import tqdm

from src.config.data_processer.datasets.StayCableVib2023Dataset.StayCableVib2023Config import (
    StayCableVib2023Config,
)
from src.data_processer.datasets.data_factory import get_dataset
from src.data_processer.preprocess.get_data_vib import VICWindowExtractor
from src.identifier.physics.base_mode_calculator import Cal_Mount
from src.identifier.physics.mecc import Abnormal_Vibration_Filter
from src.identifier.dl.identifier import DLVibrationIdentifier
from src.identifier.dl.runner import FullDatasetRunner
from src.training.deep_learning.scripts.res_cnn import train_res_cnn

from .pseudo_labeler import PseudoLabeler
from .annotation_builder import AnnotationBuilder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

_PROJECT_ROOT   = Path(__file__).parent.parent.parent.parent
_DEFAULT_CONFIG = _PROJECT_ROOT / "config" / "identifier" / "co_training" / "default.yaml"


# ---------------------------------------------------------------------------
# MECC 打分 Worker（顶层函数，保证 Windows spawn 可 pickle）
# ---------------------------------------------------------------------------

def _mecc_score_worker(
    args: Tuple[str, List[Tuple[int, int, Dict]], Dict[str, float], dict, dict, int, int]
) -> Dict[int, Tuple[int, float, Optional[float]]]:
    """
    单文件组 MECC 打分 Worker，返回 {orig_idx: (label, mecc_score, f_major)}。

    与 src/identifier/mecc/run.py 的 _process_file_group_worker 结构相同，
    但调用 classify_vibration(return_score=True) 以获取连续 mecc 分值。
    """
    file_path, windows, f0_cache, mecc_params, welch_params, window_size, fs = args

    extractor   = VICWindowExtractor(enable_denoise=False)
    mecc_filter = Abnormal_Vibration_Filter(fs=fs, **welch_params)
    vic_data    = extractor.load_file(file_path)

    results: Dict[int, Tuple[int, float, Optional[float]]] = {}
    for orig_idx, window_idx, meta_dict in windows:
        sensor_id = meta_dict.get("sensor_id")
        f0 = f0_cache.get(sensor_id)
        if f0 is None:
            continue

        sig = extractor.extract_window_from_data(
            vic_data, window_idx, window_size,
            metadata=meta_dict, file_path=file_path,
        )
        if sig is None:
            continue

        sig = sig.squeeze()
        label, mecc_score, f_major = mecc_filter.classify_vibration(
            sig, f0, return_score=True, **mecc_params
        )
        results[orig_idx] = (label, float(mecc_score), f_major)

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
        mount     = Cal_Mount.from_sensor(sid)
        cache[sid] = mount.inplane_mode(1)
        logger.info(f"  {sid} → f0 = {cache[sid]:.4f} Hz")
    return cache


def _group_by_file(
    samples, direction: str
) -> Dict[str, List[Tuple[int, int, dict]]]:
    attr   = f"{direction}_meta"
    groups: Dict[str, List] = defaultdict(list)
    for orig_idx, rec in enumerate(samples):
        meta = getattr(rec, attr, None)
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


def _run_mecc_score_parallel(
    samples,
    direction:   str,
    f0_cache:    Dict[str, float],
    mecc_params: dict,
    welch_params: dict,
    window_size: int,
    fs:          int,
    num_workers: int,
) -> Dict[int, Tuple[int, float, Optional[float]]]:
    """并行 MECC 打分：按文件分组，每个子进程处理一个 VIC 文件的所有窗口。"""
    file_groups = _group_by_file(samples, direction)
    n_windows   = sum(len(v) for v in file_groups.values())
    logger.info(
        f"[MECC 打分 {direction}] {len(file_groups)} 个文件, "
        f"{n_windows} 个窗口, workers={num_workers}"
    )

    task_args = [
        (fp, windows, f0_cache, mecc_params, welch_params, window_size, fs)
        for fp, windows in file_groups.items()
    ]

    scores: Dict[int, Tuple[int, float, Optional[float]]] = {}
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(_mecc_score_worker, arg): arg[0]
            for arg in task_args
        }
        with tqdm(total=n_windows, desc=f"MECC 打分 [{direction}]", unit="win") as pbar:
            for future in as_completed(futures):
                fp     = futures[future]
                result = future.result()
                scores.update(result)
                pbar.update(len(file_groups[fp]))

    return scores


def _build_gold_indices(
    gold_annotation_path: str,
    samples,
    direction: str = "inplane",
) -> set:
    """
    从金标注文件中解析 (file_path, window_index) 对，映射到数据集 sample_idx 集合。

    用于在生成伪标签时排除已有人工标注的样本。
    """
    with open(gold_annotation_path, "r", encoding="utf-8") as f:
        gold_entries = json.load(f)

    gold_pairs: set = set()
    for entry in gold_entries:
        fp  = entry.get("file_path", "")
        idx = int(entry.get("window_index", entry.get("window_idx", 0)))
        gold_pairs.add((str(Path(fp).resolve()), idx))

    attr = f"{direction}_meta"
    gold_indices: set = set()
    for orig_idx, rec in enumerate(samples):
        meta = getattr(rec, attr, None)
        if meta is None:
            continue
        fp  = meta.get("file_path", "")
        widx = getattr(rec, "window_idx", 0)
        if (str(Path(fp).resolve()), widx) in gold_pairs:
            gold_indices.add(orig_idx)

    logger.info(
        f"金标注映射完成：{len(gold_entries)} 条标注 → "
        f"{len(gold_indices)} 个数据集样本索引"
    )
    return gold_indices


# ---------------------------------------------------------------------------
# 单轮协同训练
# ---------------------------------------------------------------------------

def run_one_round(
    round_idx:            int,
    identifier:           DLVibrationIdentifier,
    dataset,
    f0_cache:             Dict[str, float],
    mecc_params:          dict,
    welch_params:         dict,
    gold_annotation_path: str,
    output_annotation_dir: str,
    pseudo_labeler:       PseudoLabeler,
    annotation_builder:   AnnotationBuilder,
    num_workers:          int,
    fs:                   int,
    dataset_config_path:  str,
    model_config_path:    str,
    best_params:          dict,
    epochs:               int,
    training_output_dir:  str,
) -> Tuple[int, str]:
    """
    执行单轮协同训练，返回 (新增伪标签数, 本轮标注文件路径)。

    流程
    ----
    1. 用当前 DL 模型对全量数据集进行概率推理（面内）
    2. 并行 MECC 打分（面内），获取连续 mecc_score
    3. PseudoLabeler 结合两者生成伪标签（排除金标注样本）
    4. AnnotationBuilder 合并金标注 + 伪标签，写入新标注 JSON
    5. 用新标注训练 DL（覆盖 identifier 不修改，等待外层加载新 checkpoint）
    """
    logger.info("=" * 70)
    logger.info(f"协同训练 Round {round_idx} 开始")
    logger.info("=" * 70)

    window_size = dataset.config.window_size

    # 步骤 1：DL 概率推理（面内）
    logger.info(f"[Round {round_idx}] 步骤1：DL 概率推理")
    runner = FullDatasetRunner(identifier, batch_size=256, num_workers=0)
    inplane_probas, _ = runner.run_with_proba(dataset)

    # 步骤 2：MECC 打分（面内）
    logger.info(f"[Round {round_idx}] 步骤2：MECC 并行打分")
    mecc_scores = _run_mecc_score_parallel(
        dataset._samples, "inplane",
        f0_cache, mecc_params, welch_params, window_size, fs, num_workers,
    )

    # 步骤 3：确定金标注样本索引
    gold_indices = _build_gold_indices(gold_annotation_path, dataset._samples, "inplane")

    # 步骤 4：生成伪标签
    logger.info(f"[Round {round_idx}] 步骤3：生成伪标签")
    pseudo_labels = pseudo_labeler.generate(mecc_scores, inplane_probas, gold_indices)

    # 步骤 5：构建本轮标注文件
    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    anno_path   = os.path.join(
        output_annotation_dir, f"round_{round_idx:02d}_{timestamp}.json"
    )
    os.makedirs(output_annotation_dir, exist_ok=True)

    sample_metadata: dict = {}
    for orig_idx, rec in enumerate(dataset._samples):
        in_meta  = rec.inplane_meta  or {}
        out_meta = rec.outplane_meta or {}
        sample_metadata[str(orig_idx)] = {
            "inplane_file_path":  in_meta.get("file_path"),
            "outplane_file_path": out_meta.get("file_path"),
            "window_idx":         rec.window_idx,
        }

    added = annotation_builder.build(
        pseudo_labels   = pseudo_labels,
        sample_metadata = sample_metadata,
        output_path     = anno_path,
        direction       = "inplane",
    )
    logger.info(f"[Round {round_idx}] 伪标签 {added} 条 → {anno_path}")

    # 步骤 6：用增强标注训练 DL
    logger.info(f"[Round {round_idx}] 步骤4：训练 DL（annotation={anno_path}）")
    round_output = os.path.join(training_output_dir, f"round_{round_idx:02d}")
    os.makedirs(round_output, exist_ok=True)

    train_res_cnn(
        dataset_config_path = dataset_config_path,
        model_config_path   = model_config_path,
        best_params         = best_params,
        epochs              = epochs,
        output_dir          = round_output,
    )

    return added, anno_path


# ---------------------------------------------------------------------------
# 主工作流
# ---------------------------------------------------------------------------

def main(config_path: Optional[Path] = None):
    config_path = Path(config_path) if config_path else _DEFAULT_CONFIG

    logger.info(f"加载协同训练配置：{config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # ---- 路径配置 ----
    dataset_config_path  = str(_PROJECT_ROOT / cfg["dataset_config"])
    gold_annotation_path = str(_PROJECT_ROOT / cfg["gold_annotation_path"])
    mecc_config_path     = str(_PROJECT_ROOT / cfg.get("mecc_config", "config/identifier/mecc/default.yaml"))
    dl_checkpoint_path   = str(_PROJECT_ROOT / cfg["dl_checkpoint_path"])
    dl_model_config_path = str(_PROJECT_ROOT / cfg["dl_model_config_path"])
    training_best_params_path = str(_PROJECT_ROOT / cfg["training_best_params_path"])
    output_annotation_dir = str(_PROJECT_ROOT / cfg.get("pseudo_label_dir", "results/co_training/pseudo_labels"))
    training_output_dir  = str(_PROJECT_ROOT / cfg.get("training_output_dir", "results/co_training/training"))

    # ---- 超参 ----
    max_rounds     = int(cfg.get("max_rounds", 5))
    converge_delta = int(cfg.get("converge_delta", 50))
    epochs         = int(cfg.get("epochs", 100))
    num_workers    = int(cfg.get("num_workers", 8))
    fs             = int(cfg.get("fs", 50))
    dl_model_type  = cfg.get("dl_model_type", "res_cnn")

    # ---- MECC / Welch 参数 ----
    with open(mecc_config_path, "r", encoding="utf-8") as f:
        mecc_cfg = yaml.safe_load(f)
    mecc_params  = {k: v for k, v in mecc_cfg["mecc"].items()}
    welch_params = {k: int(v) for k, v in mecc_cfg.get("welch", {}).items()}

    # ---- 置信度阈值 ----
    pseudo_labeler = PseudoLabeler(
        mecc_conf_viv = float(cfg.get("mecc_conf_viv", 0.15)),
        mecc_conf_nor = float(cfg.get("mecc_conf_nor", 0.50)),
        dl_conf_viv   = float(cfg.get("dl_conf_viv", 0.92)),
        dl_conf_nor   = float(cfg.get("dl_conf_nor", 0.92)),
    )
    annotation_builder = AnnotationBuilder(gold_annotation_path)

    # ---- 训练超参 ----
    with open(training_best_params_path, "r", encoding="utf-8") as f:
        best_params = json.load(f).get("best_params", {})

    # ========== Step 1: 加载数据集 ==========
    logger.info("步骤 1: 加载 StayCable_Vib2023 数据集")
    with open(dataset_config_path, "r", encoding="utf-8") as f:
        dataset_config = StayCableVib2023Config(**yaml.safe_load(f))
    dataset = get_dataset(dataset_config)
    logger.info(f"数据集加载完成，共 {len(dataset)} 个样本")

    # ========== Step 2: 预计算 f0 ==========
    logger.info("步骤 2: 预计算各拉索基频 f0")
    f0_cache = _build_f0_cache(dataset._samples)

    # ========== Step 3: 主迭代循环 ==========
    for round_idx in range(1, max_rounds + 1):
        # 加载当前最新 DL checkpoint
        identifier = DLVibrationIdentifier.from_checkpoint(
            checkpoint_path   = dl_checkpoint_path,
            model_type        = dl_model_type,
            model_config_path = dl_model_config_path,
        )

        added, anno_path = run_one_round(
            round_idx             = round_idx,
            identifier            = identifier,
            dataset               = dataset,
            f0_cache              = f0_cache,
            mecc_params           = mecc_params,
            welch_params          = welch_params,
            gold_annotation_path  = gold_annotation_path,
            output_annotation_dir = output_annotation_dir,
            pseudo_labeler        = pseudo_labeler,
            annotation_builder    = annotation_builder,
            num_workers           = num_workers,
            fs                    = fs,
            dataset_config_path   = anno_path,
            model_config_path     = dl_model_config_path,
            best_params           = best_params,
            epochs                = epochs,
            training_output_dir   = training_output_dir,
        )

        # 更新 checkpoint 路径为本轮产出（取 best_model.pth）
        round_ckpt = os.path.join(training_output_dir, f"round_{round_idx:02d}", "best_model.pth")
        if os.path.exists(round_ckpt):
            dl_checkpoint_path = round_ckpt
            logger.info(f"DL checkpoint 已更新 → {dl_checkpoint_path}")

        logger.info(f"Round {round_idx} 完成：新增伪标签 {added} 条")
        if added < converge_delta:
            logger.info(
                f"新增伪标签 {added} < converge_delta={converge_delta}，协同训练收敛，停止迭代"
            )
            break

    logger.info("=" * 70)
    logger.info("协同训练工作流结束")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()

