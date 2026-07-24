from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.signal import welch

from src.chapter4_characteristics.settings import (
    CLASS_DIRS,
    CLASS_LABELS,
    get_copula_dir,
    get_enriched_dir,
    load_config,
)
from src.data_processer.io_unpacker import UNPACK

DEFAULT_N_MODES = 24
DEFAULT_NFFT = 128
DEFAULT_FS = 50.0
DEFAULT_WINDOW_SIZE = 3000
DEFAULT_FREQ_LIMIT_HZ = 25.0
DEFAULT_MAX_SAMPLES = 20_000
DEFAULT_RANDOM_SEED = 42


def _mode_data_source(class_id: int) -> str:
    if class_id == 2:
        return "rwiv_202409_train_val_plus_dl"
    return "raw_vic_indexed_by_enriched"


def modes_cache_path(cfg: dict, class_id: int, n_modes: int = DEFAULT_N_MODES, nfft: int = DEFAULT_NFFT) -> Path:
    return get_copula_dir(cfg) / "modes" / f"class_{class_id}_modes_nfft{nfft}_k{n_modes}.npz"


def _extract_config(
    class_id: int,
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
        "n_modes": int(n_modes),
        "nfft": int(nfft),
        "max_samples": int(max_samples),
        "fs": float(fs),
        "window_size": int(window_size),
        "freq_limit_hz": float(freq_limit_hz),
        "random_seed": int(random_seed),
        "energy_reference": "welch_nfft128_ranked_bin_linear_sum_relative_to_full_spectrum",
        "data_source": _mode_data_source(class_id),
    }


def build_var_names(n_modes: int) -> list[str]:
    names: list[str] = []
    for k in range(1, n_modes + 1):
        names.append(f"freq_in_{k}")
    for k in range(1, n_modes + 1):
        names.append(f"energy_in_{k}")
    for k in range(1, n_modes + 1):
        names.append(f"freq_out_{k}")
    for k in range(1, n_modes + 1):
        names.append(f"energy_out_{k}")
    return names


def matrix_from_arrays(
    freq_in: np.ndarray,
    energy_in: np.ndarray,
    freq_out: np.ndarray,
    energy_out: np.ndarray,
) -> np.ndarray:
    return np.column_stack([freq_in, energy_in, freq_out, energy_out])


def _slice_window(raw: np.ndarray, window_idx: int, window_size: int) -> np.ndarray | None:
    start = window_idx * window_size
    end = start + window_size
    if end > len(raw):
        return None
    return raw[start:end]


def extract_welch_ranked_modes(
    signal: np.ndarray,
    n_modes: int,
    fs: float = DEFAULT_FS,
    nfft: int = DEFAULT_NFFT,
    freq_limit_hz: float = DEFAULT_FREQ_LIMIT_HZ,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Welch：全谱频点按 PSD 降序，能量 / 全谱线性总功率；跳过 f=0。"""
    sig = np.asarray(signal, dtype=np.float64).ravel()
    if sig.size < nfft:
        return None

    f, psd = welch(
        sig,
        fs=fs,
        nperseg=nfft // 2,
        noverlap=nfft // 4,
        nfft=nfft,
        scaling="density",
    )
    mask = f <= freq_limit_hz
    f = f[mask]
    psd = psd[mask]
    if len(psd) < n_modes:
        return None

    total_power = float(np.sum(psd))
    if total_power <= 0:
        return None

    order = np.argsort(psd)[::-1]
    f_modes: list[float] = []
    e_modes: list[float] = []
    for idx in order:
        freq = float(f[idx])
        energy = float(psd[idx] / total_power)
        if freq <= 0.0 or energy <= 0.0:
            continue
        if not np.isfinite(freq) or not np.isfinite(energy):
            continue
        f_modes.append(freq)
        e_modes.append(energy)
        if len(f_modes) >= n_modes:
            break

    if len(f_modes) < n_modes:
        return None
    return np.asarray(f_modes, dtype=np.float64), np.asarray(e_modes, dtype=np.float64)


def _iter_index_json_files(class_id: int, cfg: dict) -> list[Path]:
    """只读 enriched 索引文件列表；不触发 compact / enrich。

    优先读 batch（class_0 多为未整理 batch）；无 batch 时回退 canonical。
    """
    from src.chapter4_characteristics.feature_analysis._compactor import (
        list_batch_json_files,
        list_canonical_json_files,
        parse_batch_sensor_id,
    )
    from src.chapter4_characteristics.feature_analysis.entry import EXCLUDED_C34_ANOMALY_SENSOR_IDS

    class_dir = get_enriched_dir(cfg) / CLASS_DIRS[class_id]
    if not class_dir.exists():
        raise FileNotFoundError(f"enriched 类别目录不存在：{class_dir}")

    excluded = EXCLUDED_C34_ANOMALY_SENSOR_IDS
    batch_files = [
        p for p in list_batch_json_files(class_dir)
        if parse_batch_sensor_id(p.stem) not in excluded
    ]
    if batch_files:
        return batch_files

    json_files = list_canonical_json_files(class_dir, excluded_sensor_ids=excluded)
    if not json_files:
        raise FileNotFoundError(f"目录下无 JSON 文件：{class_dir}")
    return json_files


def _collect_jobs_from_index(
    class_id: int,
    cfg: dict,
    max_samples: int,
    random_seed: int,
) -> list[tuple[str, str, int]]:
    """
    流式扫描 enriched 索引，按 (面内,面外) VIC 路径集中收集窗口。
    先积累足够路径池，再按文件整组抽取，减少解包文件数。
    """
    json_files = _iter_index_json_files(class_id, cfg)
    rng = np.random.default_rng(random_seed)
    order = np.arange(len(json_files))
    rng.shuffle(order)

    by_pair: dict[tuple[str, str], list[int]] = defaultdict(list)
    n_scanned = 0
    pool_target = max(max_samples * 3, max_samples)
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
            by_pair[(str(in_path), str(out_path))].append(int(window_idx))
        n_pool = sum(len(v) for v in by_pair.values())
        if n_scanned % 500 == 0:
            print(f"  索引进度 {n_scanned}/{len(json_files)}，池内窗口 {n_pool}")
        if n_pool >= pool_target:
            break

    n_pool = sum(len(v) for v in by_pair.values())
    if n_pool == 0:
        return []

    pairs = list(by_pair.keys())
    rng.shuffle(pairs)
    jobs: list[tuple[str, str, int]] = []
    for in_p, out_p in pairs:
        windows = list(dict.fromkeys(by_pair[(in_p, out_p)]))
        rng.shuffle(windows)
        for w in windows:
            jobs.append((in_p, out_p, w))
            if len(jobs) >= max_samples:
                print(
                    f"  索引扫描 {n_scanned}/{len(json_files)}，池 {n_pool}，"
                    f"按 VIC 路径集中抽样 {max_samples}"
                    f"（{len({p for p, _, _ in jobs})} 个面内文件）"
                )
                return jobs

    print(f"  索引扫描完成：文件 {n_scanned}，窗口 {len(jobs)}")
    return jobs


def _collect_rwiv_jobs(max_samples: int, random_seed: int) -> list[tuple[str, str, int]]:
    from src.figure_paintings.figs_for_thesis.Chapter4._rwiv_pipeline import (
        load_rwiv_samples_for_figures,
    )

    samples = load_rwiv_samples_for_figures()
    rng = np.random.default_rng(random_seed)
    order = np.arange(len(samples))
    rng.shuffle(order)

    jobs: list[tuple[str, str, int]] = []
    for sample_i in order:
        sample = samples[int(sample_i)]
        in_path = sample.get("inplane_file_path")
        out_path = sample.get("outplane_file_path")
        window_idx = sample.get("window_idx")
        if in_path is None or out_path is None or window_idx is None:
            continue
        jobs.append((str(in_path), str(out_path), int(window_idx)))
        if len(jobs) >= max_samples:
            break

    print(
        f"  RWIV 合并样本池：候选 {len(samples)}，"
        f"按 random_seed={random_seed} 抽取 {len(jobs)}"
    )
    return jobs


def build_mode_arrays(
    class_id: int,
    cfg: dict,
    n_modes: int,
    max_samples: int,
    fs: float,
    nfft: int,
    window_size: int,
    freq_limit_hz: float,
    random_seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if class_id == 2:
        jobs = _collect_rwiv_jobs(max_samples=max_samples, random_seed=random_seed)
    else:
        jobs = _collect_jobs_from_index(
            class_id, cfg, max_samples=max_samples, random_seed=random_seed
        )
    print(f"  待处理窗口：{len(jobs)}（nfft={nfft}，K={n_modes}，面内+面外）")

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
            print(f"    已解包文件 {file_i}/{n_files}")

    keep = [
        i
        for i in range(len(jobs))
        if (
            freq_in_rows[i] is not None
            and energy_in_rows[i] is not None
            and freq_out_rows[i] is not None
            and energy_out_rows[i] is not None
        )
    ]
    if not keep:
        raise ValueError("未提取到有效面内+面外 Welch 全谱排序模态样本")

    print(f"  有效模态样本：{len(keep)}")
    return (
        np.asarray([freq_in_rows[i] for i in keep], dtype=np.float64),
        np.asarray([energy_in_rows[i] for i in keep], dtype=np.float64),
        np.asarray([freq_out_rows[i] for i in keep], dtype=np.float64),
        np.asarray([energy_out_rows[i] for i in keep], dtype=np.float64),
    )


def save_modes(
    path: Path,
    freq_in: np.ndarray,
    energy_in: np.ndarray,
    freq_out: np.ndarray,
    energy_out: np.ndarray,
    config: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        config_json=np.asarray(json.dumps(config, ensure_ascii=False)),
        freq_in=np.asarray(freq_in, dtype=np.float64),
        energy_in=np.asarray(energy_in, dtype=np.float64),
        freq_out=np.asarray(freq_out, dtype=np.float64),
        energy_out=np.asarray(energy_out, dtype=np.float64),
        created_at=np.asarray(datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
    )
    print(f"  写出模态缓存：{path}")


def load_modes(
    path: Path,
    expected_config: Optional[dict] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    if not path.exists():
        raise FileNotFoundError(f"模态缓存不存在：{path}；请先运行 copula extract")
    data = np.load(path, allow_pickle=False)
    saved = json.loads(str(data["config_json"]))
    if expected_config is not None and saved != expected_config:
        raise ValueError(f"模态缓存参数不匹配：{path}")
    return (
        np.asarray(data["freq_in"], dtype=np.float64),
        np.asarray(data["energy_in"], dtype=np.float64),
        np.asarray(data["freq_out"], dtype=np.float64),
        np.asarray(data["energy_out"], dtype=np.float64),
        saved,
    )


def extract_class_modes(
    class_id: int,
    config_path: str | None = None,
    refresh: bool = False,
    n_modes: int | None = None,
    max_samples: int | None = None,
) -> Path:
    cfg = load_config(config_path)
    n_modes = int(n_modes if n_modes is not None else cfg.get("copula_n_modes", DEFAULT_N_MODES))
    nfft = int(cfg.get("copula_nfft", DEFAULT_NFFT))
    max_samples = int(
        max_samples if max_samples is not None else cfg.get("copula_max_samples", DEFAULT_MAX_SAMPLES)
    )
    fs = float(cfg.get("fs", DEFAULT_FS))
    window_size = int(cfg.get("window_size", DEFAULT_WINDOW_SIZE))
    freq_limit_hz = float(cfg.get("freq_max_hz", DEFAULT_FREQ_LIMIT_HZ))
    random_seed = int(cfg.get("copula_rng_seed", DEFAULT_RANDOM_SEED))

    out_path = modes_cache_path(cfg, class_id, n_modes=n_modes, nfft=nfft)
    extract_cfg = _extract_config(
        class_id, n_modes, nfft, max_samples, fs, window_size, freq_limit_hz, random_seed
    )

    if out_path.exists() and not refresh:
        try:
            load_modes(out_path, expected_config=extract_cfg)
            print(f"  已存在且参数匹配，跳过：{out_path}")
            return out_path
        except ValueError:
            print(f"  缓存参数不匹配，将重算：{out_path}")

    print("=" * 80)
    print(f"模态重算 class={class_id} ({CLASS_LABELS.get(class_id, '?')}) K={n_modes} nfft={nfft}")
    print("=" * 80)

    freq_in, energy_in, freq_out, energy_out = build_mode_arrays(
        class_id,
        cfg,
        n_modes=n_modes,
        max_samples=max_samples,
        fs=fs,
        nfft=nfft,
        window_size=window_size,
        freq_limit_hz=freq_limit_hz,
        random_seed=random_seed,
    )
    save_modes(out_path, freq_in, energy_in, freq_out, energy_out, extract_cfg)
    return out_path


def extract_all_classes(
    config_path: str | None = None,
    refresh: bool = False,
    class_ids: list[int] | None = None,
    max_samples: int | None = None,
) -> list[Path]:
    ids = class_ids if class_ids is not None else [0, 1, 2, 3]
    paths: list[Path] = []
    for class_id in ids:
        paths.append(
            extract_class_modes(
                class_id,
                config_path=config_path,
                refresh=refresh,
                max_samples=max_samples,
            )
        )
    return paths
