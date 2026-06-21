from __future__ import annotations

import logging
import math
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from src.chapter3_identifier.augment.annotation.split import load_or_create_split
from src.chapter3_identifier.augment.annotation.gold_index import annotation_key
from src.chapter3_identifier.augment.features.context_window import extract_context_window
from src.chapter3_identifier.augment.features.spectrum import compute_psd_vector
from src.data_processer.preprocess.get_data_vib import VICWindowExtractor

logger = logging.getLogger(__name__)

DEFAULT_CONTEXT_INPUT_SIZE = 3000
DEFAULT_CONTEXT_TOTAL_SECONDS = 1500.0
DEFAULT_CONTEXT_ALLOW_CROSS_FILE = True


def context_side_windows(context_total_seconds: float, window_size: int, fs: float) -> int:
    window_seconds = int(window_size) / float(fs) if fs > 0 else 60.0
    total_windows = max(1, math.ceil(float(context_total_seconds) / window_seconds))
    if total_windows % 2 == 0:
        total_windows += 1
    return (total_windows - 1) // 2


def resample_or_pad_signal(signal: np.ndarray, target_size: int) -> np.ndarray:
    target = max(1, int(target_size))
    arr = np.asarray(signal, dtype=np.float32).reshape(-1)
    if int(arr.size) == target:
        return arr.astype(np.float32, copy=False)
    if int(arr.size) == 0:
        return np.zeros(target, dtype=np.float32)
    if int(arr.size) < target:
        out = np.zeros(target, dtype=np.float32)
        out[: int(arr.size)] = arr
        return out
    src_x = np.linspace(0.0, 1.0, int(arr.size), dtype=np.float32)
    dst_x = np.linspace(0.0, 1.0, target, dtype=np.float32)
    return np.interp(dst_x, src_x, arr).astype(np.float32)


def load_context_array(
    file_path: str,
    window_index: int,
    window_size: int,
    fs: float,
    context_input_size: int = DEFAULT_CONTEXT_INPUT_SIZE,
    context_total_seconds: float = DEFAULT_CONTEXT_TOTAL_SECONDS,
    allow_cross_file: bool = DEFAULT_CONTEXT_ALLOW_CROSS_FILE,
    extractor: VICWindowExtractor | None = None,
) -> np.ndarray:
    side_windows = context_side_windows(context_total_seconds, window_size, fs)
    ctx = extract_context_window(
        file_path=file_path,
        window_index=int(window_index),
        window_size=int(window_size),
        fs=float(fs),
        before=side_windows,
        after=side_windows,
        extractor=extractor,
        allow_cross_file=bool(allow_cross_file),
    )
    return resample_or_pad_signal(ctx.signal, context_input_size)


def _load_sample_arrays(
    file_path: str,
    window_index: int,
    label: int,
    window_size: int,
    fs: float,
    nfft: int,
    freq_max_hz: float,
    enable_denoise: bool,
) -> Tuple[np.ndarray, np.ndarray, int]:
    extractor = VICWindowExtractor(enable_denoise=enable_denoise)
    data = extractor.load_file(file_path)
    sig = extractor.extract_window_from_data(
        data,
        window_index,
        window_size,
        metadata={"window_index": window_index},
        file_path=file_path,
    )
    if sig is None:
        raise ValueError(f"无法提取窗口：{file_path} idx={window_index}")
    time_sig = np.asarray(sig, dtype=np.float32).reshape(-1)
    psd = compute_psd_vector(time_sig, fs=fs, nfft=nfft, freq_max_hz=freq_max_hz)
    return time_sig, psd, int(label)


def _preload_worker(args: Tuple[int, dict, int, float, int, float, bool]) -> Tuple[int, np.ndarray, np.ndarray, int]:
    ds_idx, entry, window_size, fs, nfft, freq_max_hz, enable_denoise = args
    label = int(entry.get("annotation", entry.get("class_id", 0)))
    time_sig, psd, label = _load_sample_arrays(
        entry["file_path"],
        int(entry.get("window_index", 0)),
        label,
        window_size,
        fs,
        nfft,
        freq_max_hz,
        enable_denoise,
    )
    return ds_idx, time_sig, psd, label


class DualStreamDataset(Dataset):
    def __init__(
        self,
        entries: List[dict],
        indices: Optional[List[int]] = None,
        window_size: int = 3000,
        fs: float = 50.0,
        nfft: int = 2048,
        freq_max_hz: float = 25.0,
        enable_denoise: bool = False,
        enable_long_context: bool = False,
        context_input_size: int = DEFAULT_CONTEXT_INPUT_SIZE,
        context_total_seconds: float = DEFAULT_CONTEXT_TOTAL_SECONDS,
        context_allow_cross_file: bool = DEFAULT_CONTEXT_ALLOW_CROSS_FILE,
        enable_preload_cache: bool = True,
        preload_num_workers: int = 4,
        show_preload_progress: bool = True,
    ):
        self.entries = entries
        self.indices = indices if indices is not None else list(range(len(entries)))
        self.window_size = window_size
        self.fs = fs
        self.nfft = nfft
        self.freq_max_hz = freq_max_hz
        self.enable_denoise = enable_denoise
        self.extractor = VICWindowExtractor(enable_denoise=enable_denoise)
        self._cache_path: Optional[str] = None
        self._cache_data = None
        self._preloaded: Optional[List[Tuple[np.ndarray, np.ndarray, int]]] = None

        if enable_preload_cache:
            self._preloaded = self._preload_all(
                num_workers=preload_num_workers,
                show_progress=show_preload_progress,
            )

    def __len__(self) -> int:
        return len(self.indices)

    def _load_window(self, file_path: str, window_index: int) -> np.ndarray:
        if file_path != self._cache_path:
            self._cache_data = self.extractor.load_file(file_path)
            self._cache_path = file_path
        sig = self.extractor.extract_window_from_data(
            self._cache_data,
            window_index,
            self.window_size,
            metadata={"window_index": window_index},
            file_path=file_path,
        )
        if sig is None:
            raise ValueError(f"无法提取窗口：{file_path} idx={window_index}")
        return np.asarray(sig, dtype=np.float32).reshape(-1)

    def _preload_all(
        self,
        num_workers: int = 4,
        show_progress: bool = True,
    ) -> List[Tuple[np.ndarray, np.ndarray, int]]:
        n = len(self.indices)
        logger.info(f"预加载 DualStream 样本：{n} 条（workers={num_workers}）")
        start = time.time()
        preloaded: List[Optional[Tuple[np.ndarray, np.ndarray, int]]] = [None] * n

        tasks = [
            (
                ds_idx,
                self.entries[self.indices[ds_idx]],
                self.window_size,
                self.fs,
                self.nfft,
                self.freq_max_hz,
                self.enable_denoise,
            )
            for ds_idx in range(n)
        ]

        if num_workers > 1 and n > 1:
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                futures = {executor.submit(_preload_worker, task): task[0] for task in tasks}
                iterator = as_completed(futures)
                if show_progress:
                    from tqdm import tqdm

                    iterator = tqdm(iterator, total=n, desc="预加载时序+频谱", unit="样本", ncols=100)
                for future in iterator:
                    ds_idx, time_sig, psd, label = future.result()
                    preloaded[ds_idx] = (time_sig, psd, label)
        else:
            indices = range(n)
            if show_progress:
                from tqdm import tqdm

                indices = tqdm(indices, desc="预加载时序+频谱", unit="样本", ncols=100)
            for ds_idx in indices:
                entry = self.entries[self.indices[ds_idx]]
                label = int(entry.get("annotation", entry.get("class_id", 0)))
                time_sig, psd, label = _load_sample_arrays(
                    entry["file_path"],
                    int(entry.get("window_index", 0)),
                    label,
                    self.window_size,
                    self.fs,
                    self.nfft,
                    self.freq_max_hz,
                    self.enable_denoise,
                )
                preloaded[ds_idx] = (time_sig, psd, label)

        missing = [i for i, item in enumerate(preloaded) if item is None]
        if missing:
            raise RuntimeError(f"预加载失败：{len(missing)} 个样本，例如 idx={missing[:5]}")

        elapsed = time.time() - start
        mem_mb = n * (self.window_size + 1025) * 4 / (1024 * 1024)
        logger.info(f"预加载完成：{n} 样本，耗时 {elapsed:.1f}s，约 {mem_mb:.1f} MB")
        return [item for item in preloaded if item is not None]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self._preloaded is not None:
            time_sig, psd, label = self._preloaded[idx]
        else:
            entry = self.entries[self.indices[idx]]
            file_path = entry["file_path"]
            window_index = int(entry.get("window_index", 0))
            label = int(entry.get("annotation", entry.get("class_id", 0)))
            time_sig = self._load_window(file_path, window_index)
            psd = compute_psd_vector(
                time_sig, fs=self.fs, nfft=self.nfft, freq_max_hz=self.freq_max_hz
            )

        time_tensor = torch.from_numpy(time_sig).float().unsqueeze(-1)
        psd_tensor = torch.from_numpy(psd).float().unsqueeze(-1)
        label_tensor = torch.tensor(label, dtype=torch.long)
        return time_tensor, psd_tensor, label_tensor


def _valid_label(label: object, num_classes: int) -> int | None:
    if label is None:
        return None
    value = int(label)
    if 0 <= value < int(num_classes):
        return value
    return None


def build_round2_pair_entries(
    entries: List[dict],
    pair_hints: Optional[Dict[Tuple[str, int], dict]] = None,
    num_classes: int = 4,
    enable_prediction_fill: bool = True,
    enable_gold_fill: bool = False,
    require_bidirectional_labels: bool = False,
    raise_on_empty: bool = True,
) -> Tuple[List[dict], dict]:
    merged_by_key: Dict[Tuple[str, int], dict] = {}
    direct_ann_by_key: Dict[Tuple[str, int], tuple[int, str]] = {}
    outplane_path_by_in_key: Dict[Tuple[str, int], str] = {}

    for entry in entries:
        fp = entry.get("file_path")
        wi = int(entry.get("window_index", 0))
        if not fp:
            continue
        key = annotation_key(fp, wi)
        merged_by_key[key] = dict(entry)
        ann = _valid_label(entry.get("annotation"), num_classes)
        if ann is None:
            continue
        if bool(entry.get("is_manual")):
            direct_ann_by_key[key] = (ann, "manual")
        elif key not in direct_ann_by_key:
            direct_ann_by_key[key] = (ann, "gold")
        out_fp = entry.get("outplane_file_path")
        if out_fp:
            outplane_path_by_in_key[key] = str(out_fp)

    hints = pair_hints or {}
    paired: Dict[Tuple[str, int], dict] = {}
    source_names = ("manual", "gold", "gold_fill", "prediction")
    source_counter = {f"{direction}_{source}": 0 for direction in ("inplane", "outplane") for source in source_names}

    def fill_missing_from_gold(
        in_pair: tuple[int, str] | None,
        out_pair: tuple[int, str] | None,
    ) -> tuple[tuple[int, str] | None, tuple[int, str] | None]:
        if require_bidirectional_labels or not enable_gold_fill:
            return in_pair, out_pair
        if in_pair is None and out_pair is not None and out_pair[1] == "gold":
            return (out_pair[0], "gold_fill"), out_pair
        if out_pair is None and in_pair is not None and in_pair[1] == "gold":
            return in_pair, (in_pair[0], "gold_fill")
        return in_pair, out_pair

    def fill_missing_from_prediction(
        hint: dict,
        in_pair: tuple[int, str] | None,
        out_pair: tuple[int, str] | None,
    ) -> tuple[tuple[int, str] | None, tuple[int, str] | None]:
        if require_bidirectional_labels or not enable_prediction_fill:
            return in_pair, out_pair
        if in_pair is None:
            in_label = _valid_label(hint.get("inplane_prediction"), num_classes)
            if in_label is not None:
                in_pair = (in_label, "prediction")
        if out_pair is None:
            out_label = _valid_label(hint.get("outplane_prediction"), num_classes)
            if out_label is not None:
                out_pair = (out_label, "prediction")
        return in_pair, out_pair

    def add_pair(
        in_key: Tuple[str, int],
        in_fp: str,
        out_fp: str,
        wi: int,
        in_pair: tuple[int, str],
        out_pair: tuple[int, str],
    ) -> None:
        source_counter[f"inplane_{in_pair[1]}"] += 1
        source_counter[f"outplane_{out_pair[1]}"] += 1
        paired[in_key] = {
            "inplane_file_path": str(in_fp),
            "outplane_file_path": str(out_fp),
            "window_index": wi,
            "inplane_annotation": int(in_pair[0]),
            "outplane_annotation": int(out_pair[0]),
            "inplane_label_source": in_pair[1],
            "outplane_label_source": out_pair[1],
        }

    for in_key, hint in hints.items():
        in_fp = str(hint.get("inplane_file_path", ""))
        wi = int(hint.get("window_index", 0))
        out_fp = hint.get("outplane_file_path")
        if not in_fp or not out_fp:
            continue
        out_key = annotation_key(str(out_fp), wi)
        in_pair = direct_ann_by_key.get(in_key)
        out_pair = direct_ann_by_key.get(out_key)
        if in_pair is None and out_pair is None:
            continue
        in_pair, out_pair = fill_missing_from_gold(in_pair, out_pair)
        in_pair, out_pair = fill_missing_from_prediction(hint, in_pair, out_pair)
        if in_pair is None or out_pair is None:
            continue
        add_pair(in_key, in_fp, str(out_fp), wi, in_pair, out_pair)

    for in_key, in_entry in merged_by_key.items():
        in_fp = in_entry.get("file_path")
        wi = int(in_entry.get("window_index", 0))
        out_fp = in_entry.get("outplane_file_path") or outplane_path_by_in_key.get(in_key)
        if not in_fp or not out_fp or in_key in paired:
            continue
        out_key = annotation_key(str(out_fp), wi)
        in_pair = direct_ann_by_key.get(in_key)
        out_pair = direct_ann_by_key.get(out_key)
        in_pair, out_pair = fill_missing_from_gold(in_pair, out_pair)
        if in_pair is None or out_pair is None:
            continue
        add_pair(in_key, str(in_fp), str(out_fp), wi, in_pair, out_pair)

    result = list(paired.values())
    if not result and raise_on_empty:
        raise ValueError("round2 训练需要同时包含面内/面外文件与可用标签（人工/金标/预测补全）")
    stats = {
        "pair_total": len(result),
        "enable_prediction_fill": bool(enable_prediction_fill),
        "enable_gold_fill": bool(enable_gold_fill),
        "require_bidirectional_labels": bool(require_bidirectional_labels),
        **source_counter,
    }
    return result, stats


def _label_counts(entries: List[dict], indices: List[int], field: str) -> dict[int, int]:
    counts: dict[int, int] = {}
    for idx in indices:
        label = int(entries[idx][field])
        counts[label] = int(counts.get(label, 0)) + 1
    return dict(sorted(counts.items()))


class Round2PairDataset(Dataset):
    def __init__(
        self,
        entries: List[dict],
        indices: Optional[List[int]] = None,
        window_size: int = 3000,
        fs: float = 50.0,
        nfft: int = 2048,
        freq_max_hz: float = 25.0,
        enable_denoise: bool = False,
        enable_long_context: bool = False,
        context_input_size: int = DEFAULT_CONTEXT_INPUT_SIZE,
        context_total_seconds: float = DEFAULT_CONTEXT_TOTAL_SECONDS,
        context_allow_cross_file: bool = DEFAULT_CONTEXT_ALLOW_CROSS_FILE,
        enable_preload_cache: bool = True,
        preload_num_workers: int = 4,
        show_preload_progress: bool = True,
    ):
        self.entries = entries
        self.indices = indices if indices is not None else list(range(len(entries)))
        self.window_size = window_size
        self.fs = fs
        self.nfft = nfft
        self.freq_max_hz = freq_max_hz
        self.enable_denoise = enable_denoise
        self.enable_long_context = bool(enable_long_context)
        self.context_input_size = int(context_input_size)
        self.context_total_seconds = float(context_total_seconds)
        self.context_allow_cross_file = bool(context_allow_cross_file)
        self.extractor = VICWindowExtractor(enable_denoise=enable_denoise)
        self._cache_path: Optional[str] = None
        self._cache_data = None
        self._preloaded: Optional[List[tuple]] = None
        self.round2_preload_total = int(len(self.indices))
        self.round2_bad_total = 0
        self.round2_bad_rate = 0.0
        self.round2_bad_reason_counts: Dict[str, int] = {}
        self.round2_bad_examples: List[str] = []

        if enable_preload_cache:
            self._preloaded = self._preload_all(
                num_workers=preload_num_workers,
                show_progress=show_preload_progress,
            )

    def __len__(self) -> int:
        return len(self.indices)

    def _load_window(self, file_path: str, window_index: int) -> np.ndarray:
        if file_path != self._cache_path:
            self._cache_data = self.extractor.load_file(file_path)
            self._cache_path = file_path
        sig = self.extractor.extract_window_from_data(
            self._cache_data,
            window_index,
            self.window_size,
            metadata={"window_index": window_index},
            file_path=file_path,
        )
        if sig is None:
            raise ValueError(f"无法提取窗口：{file_path} idx={window_index}")
        return np.asarray(sig, dtype=np.float32).reshape(-1)

    def _preload_pair(self, entry: dict) -> tuple:
        wi = int(entry.get("window_index", 0))
        in_time = self._load_window(entry["inplane_file_path"], wi)
        out_time = self._load_window(entry["outplane_file_path"], wi)
        in_psd = compute_psd_vector(in_time, fs=self.fs, nfft=self.nfft, freq_max_hz=self.freq_max_hz)
        out_psd = compute_psd_vector(out_time, fs=self.fs, nfft=self.nfft, freq_max_hz=self.freq_max_hz)
        in_label = int(entry["inplane_annotation"])
        out_label = int(entry["outplane_annotation"])
        if self.enable_long_context:
            in_context = load_context_array(
                entry["inplane_file_path"],
                wi,
                self.window_size,
                self.fs,
                context_input_size=self.context_input_size,
                context_total_seconds=self.context_total_seconds,
                allow_cross_file=self.context_allow_cross_file,
                extractor=self.extractor,
            )
            out_context = load_context_array(
                entry["outplane_file_path"],
                wi,
                self.window_size,
                self.fs,
                context_input_size=self.context_input_size,
                context_total_seconds=self.context_total_seconds,
                allow_cross_file=self.context_allow_cross_file,
                extractor=self.extractor,
            )
            return in_time, in_psd, in_context, out_time, out_psd, out_context, in_label, out_label
        return in_time, in_psd, out_time, out_psd, in_label, out_label

    def _preload_worker(
        self,
        args: Tuple[int, dict],
    ) -> Tuple[int, Optional[tuple], str]:
        ds_idx, entry = args
        wi = int(entry.get("window_index", 0))
        payload = _load_round2_pair_arrays(
            inplane_file_path=entry["inplane_file_path"],
            outplane_file_path=entry["outplane_file_path"],
            window_index=wi,
            window_size=self.window_size,
            fs=self.fs,
            nfft=self.nfft,
            freq_max_hz=self.freq_max_hz,
            in_label=int(entry["inplane_annotation"]),
            out_label=int(entry["outplane_annotation"]),
            enable_denoise=self.enable_denoise,
            enable_long_context=self.enable_long_context,
            context_input_size=self.context_input_size,
            context_total_seconds=self.context_total_seconds,
            context_allow_cross_file=self.context_allow_cross_file,
        )
        pair_payload, reason = payload
        return ds_idx, pair_payload, reason

    def _preload_all(
        self,
        num_workers: int = 4,
        show_progress: bool = True,
    ) -> List[tuple]:
        n = len(self.indices)
        feature_name = "六特征" if self.enable_long_context else "四特征"
        logger.info(f"预加载配对双头样本：{n} 条（{feature_name}, workers={num_workers}）")
        preloaded: List[Optional[tuple]] = [None] * n
        dropped_refs: List[str] = []
        dropped_reason_counts: Dict[str, int] = {}
        tasks = [(ds_idx, self.entries[self.indices[ds_idx]]) for ds_idx in range(n)]

        if num_workers > 1 and n > 1:
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                futures = {executor.submit(_preload_round2_worker, task): task[0] for task in tasks}
                iterator = as_completed(futures)
                if show_progress:
                    from tqdm import tqdm

                    iterator = tqdm(iterator, total=n, desc=f"预加载配对双头 {feature_name}", unit="样本", ncols=100)
                for future in iterator:
                    ds_idx, payload, reason = future.result()
                    if payload is None:
                        entry = self.entries[self.indices[ds_idx]]
                        dropped_reason_counts[reason] = int(dropped_reason_counts.get(reason, 0)) + 1
                        dropped_refs.append(
                            f"{reason}: in={entry.get('inplane_file_path', '?')} out={entry.get('outplane_file_path', '?')} idx={int(entry.get('window_index', 0))}"
                        )
                        continue
                    preloaded[ds_idx] = payload
        else:
            indices = range(n)
            if show_progress:
                from tqdm import tqdm

                indices = tqdm(indices, desc=f"预加载配对双头 {feature_name}", unit="样本", ncols=100)
            for ds_idx in indices:
                entry = self.entries[self.indices[ds_idx]]
                payload = _load_round2_pair_arrays(
                    inplane_file_path=entry["inplane_file_path"],
                    outplane_file_path=entry["outplane_file_path"],
                    window_index=int(entry.get("window_index", 0)),
                    window_size=self.window_size,
                    fs=self.fs,
                    nfft=self.nfft,
                    freq_max_hz=self.freq_max_hz,
                    in_label=int(entry["inplane_annotation"]),
                    out_label=int(entry["outplane_annotation"]),
                    enable_denoise=self.enable_denoise,
                    enable_long_context=self.enable_long_context,
                    context_input_size=self.context_input_size,
                    context_total_seconds=self.context_total_seconds,
                    context_allow_cross_file=self.context_allow_cross_file,
                )
                pair_payload, reason = payload
                if pair_payload is None:
                    dropped_reason_counts[reason] = int(dropped_reason_counts.get(reason, 0)) + 1
                    dropped_refs.append(
                        f"{reason}: in={entry.get('inplane_file_path', '?')} out={entry.get('outplane_file_path', '?')} idx={int(entry.get('window_index', 0))}"
                    )
                    continue
                preloaded[ds_idx] = pair_payload

        kept_ds_indices = [i for i, item in enumerate(preloaded) if item is not None]
        bad_total = int(len(dropped_refs))
        self.round2_preload_total = int(n)
        self.round2_bad_total = bad_total
        self.round2_bad_rate = float(bad_total / max(n, 1))
        self.round2_bad_reason_counts = dict(sorted(dropped_reason_counts.items()))
        self.round2_bad_examples = dropped_refs[:10]
        if dropped_refs:
            logger.warning(
                "配对双头预加载坏样本率 %.2f%% (%s/%s)，原因分布=%s，示例=%s",
                self.round2_bad_rate * 100.0,
                bad_total,
                n,
                self.round2_bad_reason_counts,
                self.round2_bad_examples[:3],
            )
        if not kept_ds_indices:
            raise RuntimeError("配对双头预加载失败：无可用样本（窗口提取全部失败）")
        self.indices = [self.indices[i] for i in kept_ds_indices]
        return [preloaded[i] for i in kept_ds_indices if preloaded[i] is not None]

    def __getitem__(self, idx: int):
        if self._preloaded is not None:
            payload = self._preloaded[idx]
        else:
            entry = self.entries[self.indices[idx]]
            payload = self._preload_pair(entry)

        if self.enable_long_context:
            in_time, in_psd, in_context, out_time, out_psd, out_context, in_label, out_label = payload
            in_context_t = torch.from_numpy(in_context).float().unsqueeze(-1)
            out_context_t = torch.from_numpy(out_context).float().unsqueeze(-1)
        else:
            in_time, in_psd, out_time, out_psd, in_label, out_label = payload

        in_time_t = torch.from_numpy(in_time).float().unsqueeze(-1)
        in_psd_t = torch.from_numpy(in_psd).float().unsqueeze(-1)
        out_time_t = torch.from_numpy(out_time).float().unsqueeze(-1)
        out_psd_t = torch.from_numpy(out_psd).float().unsqueeze(-1)
        in_label_t = torch.tensor(in_label, dtype=torch.long)
        out_label_t = torch.tensor(out_label, dtype=torch.long)
        if self.enable_long_context:
            return in_time_t, in_psd_t, in_context_t, out_time_t, out_psd_t, out_context_t, in_label_t, out_label_t
        return in_time_t, in_psd_t, out_time_t, out_psd_t, in_label_t, out_label_t


def build_dataloaders(
    entries: List[dict],
    split_path: str,
    batch_size: int = 16,
    train_val_ratio: float = 0.8,
    random_seed: int = 42,
    window_size: int = 3000,
    fs: float = 50.0,
    nfft: int = 2048,
    freq_max_hz: float = 25.0,
    enable_preload_cache: bool = True,
    preload_num_workers: int = 4,
    show_preload_progress: bool = True,
    num_workers: int = 0,
):
    from torch.utils.data import DataLoader

    train_idx, val_idx = load_or_create_split(
        entries, split_path, train_val_ratio=train_val_ratio, random_seed=random_seed
    )
    preload_kwargs = {
        "window_size": window_size,
        "fs": fs,
        "nfft": nfft,
        "freq_max_hz": freq_max_hz,
        "enable_preload_cache": enable_preload_cache,
        "preload_num_workers": preload_num_workers,
        "show_preload_progress": show_preload_progress,
    }
    train_ds = DualStreamDataset(entries, train_idx, **preload_kwargs)
    val_ds = DualStreamDataset(
        entries,
        val_idx,
        enable_preload_cache=enable_preload_cache,
        preload_num_workers=preload_num_workers,
        show_preload_progress=False,
        window_size=window_size,
        fs=fs,
        nfft=nfft,
        freq_max_hz=freq_max_hz,
    )
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )
    return train_loader, val_loader


def _load_round2_pair_arrays(
    inplane_file_path: str,
    outplane_file_path: str,
    window_index: int,
    window_size: int,
    fs: float,
    nfft: int,
    freq_max_hz: float,
    in_label: int,
    out_label: int,
    enable_denoise: bool,
    enable_long_context: bool = False,
    context_input_size: int = DEFAULT_CONTEXT_INPUT_SIZE,
    context_total_seconds: float = DEFAULT_CONTEXT_TOTAL_SECONDS,
    context_allow_cross_file: bool = DEFAULT_CONTEXT_ALLOW_CROSS_FILE,
) -> Tuple[Optional[tuple], str]:
    extractor = VICWindowExtractor(enable_denoise=enable_denoise)

    in_data = extractor.load_file(inplane_file_path)
    in_sig = extractor.extract_window_from_data(
        in_data,
        window_index,
        window_size,
        metadata={"window_index": window_index},
        file_path=inplane_file_path,
    )
    if in_sig is None:
        return None, "inplane_window_missing"

    out_data = extractor.load_file(outplane_file_path)
    out_sig = extractor.extract_window_from_data(
        out_data,
        window_index,
        window_size,
        metadata={"window_index": window_index},
        file_path=outplane_file_path,
    )
    if out_sig is None:
        return None, "outplane_window_missing"

    in_time = np.asarray(in_sig, dtype=np.float32).reshape(-1)
    out_time = np.asarray(out_sig, dtype=np.float32).reshape(-1)
    in_psd = compute_psd_vector(in_time, fs=fs, nfft=nfft, freq_max_hz=freq_max_hz)
    out_psd = compute_psd_vector(out_time, fs=fs, nfft=nfft, freq_max_hz=freq_max_hz)
    if not enable_long_context:
        return (in_time, in_psd, out_time, out_psd, int(in_label), int(out_label)), "ok"

    in_context = load_context_array(
        inplane_file_path,
        window_index,
        window_size,
        fs,
        context_input_size=context_input_size,
        context_total_seconds=context_total_seconds,
        allow_cross_file=context_allow_cross_file,
        extractor=extractor,
    )
    out_context = load_context_array(
        outplane_file_path,
        window_index,
        window_size,
        fs,
        context_input_size=context_input_size,
        context_total_seconds=context_total_seconds,
        allow_cross_file=context_allow_cross_file,
        extractor=extractor,
    )
    return (
        in_time,
        in_psd,
        in_context,
        out_time,
        out_psd,
        out_context,
        int(in_label),
        int(out_label),
    ), "ok"


def _preload_round2_worker(
    args: Tuple[int, dict],
) -> Tuple[int, Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int]], str]:
    ds_idx, entry = args
    wi = int(entry.get("window_index", 0))
    payload, reason = _load_round2_pair_arrays(
        inplane_file_path=entry["inplane_file_path"],
        outplane_file_path=entry["outplane_file_path"],
        window_index=wi,
        window_size=int(entry.get("_window_size", 3000)),
        fs=float(entry.get("_fs", 50.0)),
        nfft=int(entry.get("_nfft", 2048)),
        freq_max_hz=float(entry.get("_freq_max_hz", 25.0)),
        in_label=int(entry["inplane_annotation"]),
        out_label=int(entry["outplane_annotation"]),
        enable_denoise=bool(entry.get("_enable_denoise", False)),
        enable_long_context=bool(entry.get("_enable_long_context", False)),
        context_input_size=int(entry.get("_context_input_size", DEFAULT_CONTEXT_INPUT_SIZE)),
        context_total_seconds=float(entry.get("_context_total_seconds", DEFAULT_CONTEXT_TOTAL_SECONDS)),
        context_allow_cross_file=bool(entry.get("_context_allow_cross_file", DEFAULT_CONTEXT_ALLOW_CROSS_FILE)),
    )
    return ds_idx, payload, reason


def build_round2_dataloaders(
    entries: List[dict],
    split_path: str,
    round_idx: int | None = None,
    batch_size: int = 16,
    train_val_ratio: float = 0.8,
    random_seed: int = 42,
    window_size: int = 3000,
    fs: float = 50.0,
    nfft: int = 2048,
    freq_max_hz: float = 25.0,
    enable_denoise: bool = False,
    enable_long_context: bool = False,
    context_input_size: int = DEFAULT_CONTEXT_INPUT_SIZE,
    context_total_seconds: float = DEFAULT_CONTEXT_TOTAL_SECONDS,
    context_allow_cross_file: bool = DEFAULT_CONTEXT_ALLOW_CROSS_FILE,
    enable_preload_cache: bool = True,
    preload_num_workers: int = 4,
    show_preload_progress: bool = True,
    num_workers: int = 0,
):
    from torch.utils.data import DataLoader

    augmented = []
    for entry in entries:
        row = dict(entry)
        row["_window_size"] = int(window_size)
        row["_fs"] = float(fs)
        row["_nfft"] = int(nfft)
        row["_freq_max_hz"] = float(freq_max_hz)
        row["_enable_denoise"] = bool(enable_denoise)
        row["_enable_long_context"] = bool(enable_long_context)
        row["_context_input_size"] = int(context_input_size)
        row["_context_total_seconds"] = float(context_total_seconds)
        row["_context_allow_cross_file"] = bool(context_allow_cross_file)
        augmented.append(row)

    train_idx, val_idx = load_or_create_split(
        augmented, split_path, train_val_ratio=train_val_ratio, random_seed=random_seed
    )
    round_note = f"round {int(round_idx)} " if round_idx is not None else ""
    logger.info(
        "%s配对双头训练 split：strategy=stratified_resample, label_strategy=joint_inplane_outplane, train=%s, val=%s, ratio=%.3f, seed=%s",
        round_note,
        len(train_idx),
        len(val_idx),
        float(train_val_ratio),
        int(random_seed),
    )
    logger.info(
        "%s配对双头 split 标签分布：in_train=%s, in_val=%s, out_train=%s, out_val=%s",
        round_note,
        _label_counts(augmented, train_idx, "inplane_annotation"),
        _label_counts(augmented, val_idx, "inplane_annotation"),
        _label_counts(augmented, train_idx, "outplane_annotation"),
        _label_counts(augmented, val_idx, "outplane_annotation"),
    )
    preload_kwargs = {
        "window_size": window_size,
        "fs": fs,
        "nfft": nfft,
        "freq_max_hz": freq_max_hz,
        "enable_denoise": enable_denoise,
        "enable_long_context": enable_long_context,
        "context_input_size": context_input_size,
        "context_total_seconds": context_total_seconds,
        "context_allow_cross_file": context_allow_cross_file,
        "enable_preload_cache": enable_preload_cache,
        "preload_num_workers": preload_num_workers,
        "show_preload_progress": show_preload_progress,
    }
    train_ds = Round2PairDataset(augmented, train_idx, **preload_kwargs)
    val_ds = Round2PairDataset(
        augmented,
        val_idx,
        window_size=window_size,
        fs=fs,
        nfft=nfft,
        freq_max_hz=freq_max_hz,
        enable_denoise=enable_denoise,
        enable_long_context=enable_long_context,
        context_input_size=context_input_size,
        context_total_seconds=context_total_seconds,
        context_allow_cross_file=context_allow_cross_file,
        enable_preload_cache=enable_preload_cache,
        preload_num_workers=preload_num_workers,
        show_preload_progress=False,
    )
    if len(train_ds) < len(train_idx) or len(val_ds) < len(val_idx):
        logger.warning(
            "%s配对双头数据集过滤了无法提取窗口的样本：train %s->%s, val %s->%s",
            round_note,
            len(train_idx),
            len(train_ds),
            len(val_idx),
            len(val_ds),
        )
    logger.info(
        "%s配对双头 bad sample stats | train: %.2f%% (%s/%s) %s | val: %.2f%% (%s/%s) %s",
        round_note,
        float(getattr(train_ds, "round2_bad_rate", 0.0)) * 100.0,
        int(getattr(train_ds, "round2_bad_total", 0)),
        int(getattr(train_ds, "round2_preload_total", len(train_ds))),
        getattr(train_ds, "round2_bad_reason_counts", {}),
        float(getattr(val_ds, "round2_bad_rate", 0.0)) * 100.0,
        int(getattr(val_ds, "round2_bad_total", 0)),
        int(getattr(val_ds, "round2_preload_total", len(val_ds))),
        getattr(val_ds, "round2_bad_reason_counts", {}),
    )
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )
    return train_loader, val_loader
