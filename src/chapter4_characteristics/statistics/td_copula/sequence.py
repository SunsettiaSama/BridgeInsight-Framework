"""序列提取与 lag-1 相邻窗配对（独立缓存，不碰静态 modes npz）。"""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

from src.chapter4_characteristics.settings import (
    CLASS_DIRS,
    CLASS_LABELS,
    get_copula_dir,
    load_config,
)
from src.chapter4_characteristics.statistics.mode_extract import (
    DEFAULT_FS,
    DEFAULT_FREQ_LIMIT_HZ,
    DEFAULT_NFFT,
    DEFAULT_RANDOM_SEED,
    DEFAULT_WINDOW_SIZE,
    _collect_jobs_from_index,
    _iter_index_json_files,
    _slice_window,
    extract_welch_ranked_modes,
)
from src.data_processer.io_unpacker import UNPACK

DEFAULT_ENERGY_TOP_K = 4
DEFAULT_TD_MAX_SAMPLES = 8000


def get_td_dir(cfg: dict) -> Path:
    return get_copula_dir(cfg) / "td"


def td_seq_path(cfg: dict, class_id: int) -> Path:
    return get_td_dir(cfg) / f"class_{class_id}_td_seq.npz"


def core_var_names(energy_top_k: int) -> list[str]:
    names = ["freq_in_1", "freq_out_1"]
    for k in range(1, energy_top_k + 1):
        names.append(f"energy_in_{k}")
    for k in range(1, energy_top_k + 1):
        names.append(f"energy_out_{k}")
    return names


def _td_extract_config(
    class_id: int,
    energy_top_k: int,
    n_modes: int,
    nfft: int,
    max_samples: int,
    fs: float,
    window_size: int,
    freq_limit_hz: float,
    random_seed: int,
) -> dict:
    return {
        "class_id": int(class_id),
        "class_label": CLASS_DIRS[class_id],
        "energy_top_k": int(energy_top_k),
        "n_modes": int(n_modes),
        "nfft": int(nfft),
        "max_samples": int(max_samples),
        "fs": float(fs),
        "window_size": int(window_size),
        "freq_limit_hz": float(freq_limit_hz),
        "random_seed": int(random_seed),
        "pipeline": "td_copula_sequence",
        "energy_reference": "welch_nfft128_ranked_bin_linear_sum_relative_to_full_spectrum",
        "data_source": "raw_vic_indexed_by_enriched",
    }


def _contiguous_runs(windows: set[int]) -> list[list[int]]:
    ws = sorted(windows)
    if not ws:
        return []
    runs: list[list[int]] = [[ws[0]]]
    for w in ws[1:]:
        if w == runs[-1][-1] + 1:
            runs[-1].append(w)
        else:
            runs.append([w])
    return runs


def _count_lag1_in_pool(by_pair: dict[tuple[str, str], set[int]]) -> int:
    n = 0
    for windows in by_pair.values():
        for run in _contiguous_runs(windows):
            if len(run) >= 2:
                n += len(run) - 1
    return n


def _collect_sequence_jobs(
    class_id: int,
    cfg: dict,
    max_samples: int,
    random_seed: int,
) -> list[tuple[str, str, int]]:
    """只抽取连续 window_idx 段上的窗口，保证充足 lag-1 对。"""
    json_files = _iter_index_json_files(class_id, cfg)
    rng = np.random.default_rng(random_seed)
    order = np.arange(len(json_files))
    rng.shuffle(order)

    by_pair: dict[tuple[str, str], set[int]] = defaultdict(set)
    n_scanned = 0
    # 目标：至少约 max_samples 个连续窗（lag1 ≈ max_samples-#runs）
    lag1_target = max(max_samples - 1, max_samples // 2)
    for file_i in order:
        json_file = json_files[int(file_i)]
        with open(json_file, "r", encoding="utf-8") as f:
            payload = json.load(f)
        n_scanned += 1
        for sample in payload.get("samples", []):
            in_path = sample.get("inplane_file_path")
            out_path = sample.get("outplane_file_path")
            window_idx = sample.get("window_idx")
            if in_path is None or out_path is None or window_idx is None:
                continue
            by_pair[(str(in_path), str(out_path))].add(int(window_idx))
        if n_scanned % 500 == 0:
            n_lag1 = _count_lag1_in_pool(by_pair)
            print(
                f"  [td] 索引进度 {n_scanned}/{len(json_files)}，"
                f"池内 lag1≈{n_lag1}"
            )
        if _count_lag1_in_pool(by_pair) >= lag1_target:
            break

    if not by_pair:
        return []

    # 收集长度≥2 的连续段，按长度降序填充
    run_jobs: list[tuple[int, str, str, list[int]]] = []
    for in_p, out_p in by_pair:
        for run in _contiguous_runs(by_pair[(in_p, out_p)]):
            if len(run) >= 2:
                run_jobs.append((len(run), in_p, out_p, run))
    # 同长度段内打散，再按长度降序
    rng.shuffle(run_jobs)
    run_jobs.sort(key=lambda x: x[0], reverse=True)

    jobs: list[tuple[str, str, int]] = []
    n_lag1 = 0
    for _, in_p, out_p, run in run_jobs:
        for w in run:
            jobs.append((in_p, out_p, w))
        n_lag1 += len(run) - 1
        if len(jobs) >= max_samples:
            break

    print(
        f"  [td] 索引扫描 {n_scanned}/{len(json_files)}，"
        f"连续段窗口 {len(jobs)}（估计 lag1≈{n_lag1}，"
        f"{len({p for p, _, _ in jobs})} 个面内文件）"
    )
    return jobs


def _to_core_row(
    freq_in: np.ndarray,
    energy_in: np.ndarray,
    freq_out: np.ndarray,
    energy_out: np.ndarray,
    energy_top_k: int,
) -> np.ndarray:
    parts = [
        freq_in[0:1],
        freq_out[0:1],
        energy_in[:energy_top_k],
        energy_out[:energy_top_k],
    ]
    return np.concatenate(parts).astype(np.float64)


def build_sequence_arrays(
    class_id: int,
    cfg: dict,
    energy_top_k: int,
    n_modes: int,
    max_samples: int,
    fs: float,
    nfft: int,
    window_size: int,
    freq_limit_hz: float,
    random_seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """返回 features (n,d), in_paths, out_paths, window_idx。"""
    jobs = _collect_sequence_jobs(
        class_id, cfg, max_samples=max_samples, random_seed=random_seed
    )
    if not jobs:
        # 回退：与静态相同的随机池（仍写出 path/window，可能 lag1 较少）
        jobs = _collect_jobs_from_index(
            class_id, cfg, max_samples=max_samples, random_seed=random_seed
        )
    print(f"  [td] 待处理窗口：{len(jobs)}（nfft={nfft}，K={n_modes}，Ke={energy_top_k}）")

    by_path: dict[str, list[tuple[int, int, str]]] = defaultdict(list)
    for i, (in_path, out_path, window_idx) in enumerate(jobs):
        by_path[in_path].append((window_idx, i, "in"))
        by_path[out_path].append((window_idx, i, "out"))

    freq_in_rows: list[np.ndarray | None] = [None] * len(jobs)
    energy_in_rows: list[np.ndarray | None] = [None] * len(jobs)
    freq_out_rows: list[np.ndarray | None] = [None] * len(jobs)
    energy_out_rows: list[np.ndarray | None] = [None] * len(jobs)

    unpacker = UNPACK(init_path=False)
    n_files = len(by_path)
    for file_i, (path, items) in enumerate(by_path.items(), start=1):
        if not Path(path).exists():
            continue
        raw = np.asarray(unpacker.VIC_DATA_Unpack(str(path)), dtype=np.float64)
        for window_idx, job_i, side in items:
            sig = _slice_window(raw, window_idx, window_size)
            if sig is None:
                continue
            extracted = extract_welch_ranked_modes(
                sig, n_modes=n_modes, fs=fs, nfft=nfft, freq_limit_hz=freq_limit_hz
            )
            if extracted is None:
                continue
            freqs, energies = extracted
            if side == "in":
                freq_in_rows[job_i] = freqs
                energy_in_rows[job_i] = energies
            else:
                freq_out_rows[job_i] = freqs
                energy_out_rows[job_i] = energies
        if file_i % 200 == 0 or file_i == n_files:
            print(f"    [td] 已解包文件 {file_i}/{n_files}")

    keep: list[int] = []
    for i in range(len(jobs)):
        if (
            freq_in_rows[i] is not None
            and energy_in_rows[i] is not None
            and freq_out_rows[i] is not None
            and energy_out_rows[i] is not None
        ):
            keep.append(i)

    if not keep:
        raise ValueError("td_copula：未提取到有效序列样本")

    features = np.asarray(
        [
            _to_core_row(
                freq_in_rows[i],
                energy_in_rows[i],
                freq_out_rows[i],
                energy_out_rows[i],
                energy_top_k,
            )
            for i in keep
        ],
        dtype=np.float64,
    )
    in_paths = np.asarray([jobs[i][0] for i in keep], dtype=object)
    out_paths = np.asarray([jobs[i][1] for i in keep], dtype=object)
    window_idx = np.asarray([jobs[i][2] for i in keep], dtype=np.int32)
    print(f"  [td] 有效序列样本：{len(keep)}，d={features.shape[1]}")
    return features, in_paths, out_paths, window_idx


def save_td_seq(
    path: Path,
    features: np.ndarray,
    in_paths: np.ndarray,
    out_paths: np.ndarray,
    window_idx: np.ndarray,
    var_names: list[str],
    config: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        config_json=np.asarray(json.dumps(config, ensure_ascii=False)),
        var_names_json=np.asarray(json.dumps(var_names, ensure_ascii=False)),
        features=np.asarray(features, dtype=np.float64),
        in_paths=in_paths,
        out_paths=out_paths,
        window_idx=np.asarray(window_idx, dtype=np.int32),
        created_at=np.asarray(datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
    )
    print(f"  [td] 写出序列缓存：{path}")


def load_td_seq(
    path: Path,
    expected_config: Optional[dict] = None,
) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"td 序列缓存不存在：{path}；请先运行 td_copula sequence")
    data = np.load(path, allow_pickle=True)
    saved = json.loads(str(data["config_json"]))
    if expected_config is not None and saved != expected_config:
        raise ValueError(f"td 序列缓存参数不匹配：{path}")
    return {
        "config": saved,
        "var_names": json.loads(str(data["var_names_json"])),
        "features": np.asarray(data["features"], dtype=np.float64),
        "in_paths": np.asarray(data["in_paths"], dtype=object),
        "out_paths": np.asarray(data["out_paths"], dtype=object),
        "window_idx": np.asarray(data["window_idx"], dtype=np.int32),
    }


def build_lag1_pairs(
    features: np.ndarray,
    in_paths: np.ndarray,
    out_paths: np.ndarray,
    window_idx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """返回 (X_t, X_{t+1})，各自 shape (n_pairs, d)。"""
    groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    for i in range(len(window_idx)):
        groups[(str(in_paths[i]), str(out_paths[i]))].append(i)

    xt_rows: list[np.ndarray] = []
    xt1_rows: list[np.ndarray] = []
    for _, idxs in groups.items():
        idxs = sorted(idxs, key=lambda i: int(window_idx[i]))
        for a, b in zip(idxs[:-1], idxs[1:]):
            if int(window_idx[b]) == int(window_idx[a]) + 1:
                xt_rows.append(features[a])
                xt1_rows.append(features[b])

    if not xt_rows:
        raise ValueError(
            "td_copula：未找到 window_idx 连续的相邻窗对；"
            "请增大 max_samples 或检查 enriched 索引是否含连续窗"
        )
    return (
        np.asarray(xt_rows, dtype=np.float64),
        np.asarray(xt1_rows, dtype=np.float64),
    )


def extract_class_sequence(
    class_id: int,
    config_path: str | None = None,
    refresh: bool = False,
    max_samples: int | None = None,
) -> Path:
    cfg = load_config(config_path)
    td = cfg.get("td_copula") or {}
    energy_top_k = int(td.get("energy_top_k", cfg.get("td_energy_top_k", DEFAULT_ENERGY_TOP_K)))
    n_modes = energy_top_k  # 核心特征只需前 Ke 阶（含 f1）
    nfft = int(td.get("nfft", cfg.get("copula_nfft", DEFAULT_NFFT)))
    max_samples = int(
        max_samples
        if max_samples is not None
        else td.get("max_samples", cfg.get("td_max_samples", DEFAULT_TD_MAX_SAMPLES))
    )
    fs = float(cfg.get("fs", DEFAULT_FS))
    window_size = int(cfg.get("window_size", DEFAULT_WINDOW_SIZE))
    freq_limit_hz = float(cfg.get("freq_max_hz", DEFAULT_FREQ_LIMIT_HZ))
    random_seed = int(td.get("rng_seed", cfg.get("copula_rng_seed", DEFAULT_RANDOM_SEED)))

    out_path = td_seq_path(cfg, class_id)
    extract_cfg = _td_extract_config(
        class_id,
        energy_top_k,
        n_modes,
        nfft,
        max_samples,
        fs,
        window_size,
        freq_limit_hz,
        random_seed,
    )
    var_names = core_var_names(energy_top_k)

    if out_path.exists() and not refresh:
        try:
            load_td_seq(out_path, expected_config=extract_cfg)
            print(f"  [td] 已存在且参数匹配，跳过：{out_path}")
            return out_path
        except ValueError:
            print(f"  [td] 缓存参数不匹配，将重算：{out_path}")

    print("=" * 80)
    print(
        f"[td] 序列提取 class={class_id} ({CLASS_LABELS.get(class_id, '?')}) "
        f"Ke={energy_top_k} nfft={nfft}"
    )
    print("=" * 80)

    features, in_paths, out_paths, window_idx = build_sequence_arrays(
        class_id,
        cfg,
        energy_top_k=energy_top_k,
        n_modes=n_modes,
        max_samples=max_samples,
        fs=fs,
        nfft=nfft,
        window_size=window_size,
        freq_limit_hz=freq_limit_hz,
        random_seed=random_seed,
    )
    save_td_seq(out_path, features, in_paths, out_paths, window_idx, var_names, extract_cfg)
    return out_path
