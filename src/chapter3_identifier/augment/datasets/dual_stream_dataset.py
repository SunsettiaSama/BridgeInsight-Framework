from __future__ import annotations

import logging
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from src.chapter3_identifier.augment.annotation.split import load_or_create_split
from src.chapter3_identifier.augment.features.spectrum import compute_psd_vector
from src.data_processer.preprocess.get_data_vib import VICWindowExtractor

logger = logging.getLogger(__name__)


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


def build_round2_pair_entries(entries: List[dict]) -> List[dict]:
    paired: Dict[Tuple[str, int], dict] = {}
    for entry in entries:
        in_fp = entry.get("file_path")
        out_fp = entry.get("outplane_file_path")
        wi = int(entry.get("window_index", 0))
        fallback_ann = entry.get("annotation")
        in_ann = entry.get("inplane_annotation", fallback_ann)
        out_ann = entry.get("outplane_annotation", fallback_ann)
        if in_fp and out_fp and in_ann is not None and out_ann is not None:
            paired[(in_fp, wi)] = {
                "inplane_file_path": in_fp,
                "outplane_file_path": out_fp,
                "window_index": wi,
                "inplane_annotation": int(in_ann),
                "outplane_annotation": int(out_ann),
            }
    result = list(paired.values())
    if not result:
        raise ValueError("round2 训练需要同时包含面内/面外文件与标注的数据")
    return result


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
        self._preloaded: Optional[
            List[
                Tuple[
                    np.ndarray,
                    np.ndarray,
                    np.ndarray,
                    np.ndarray,
                    int,
                    int,
                ]
            ]
        ] = None

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

    def _preload_pair(self, entry: dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int]:
        wi = int(entry.get("window_index", 0))
        in_time = self._load_window(entry["inplane_file_path"], wi)
        out_time = self._load_window(entry["outplane_file_path"], wi)
        in_psd = compute_psd_vector(in_time, fs=self.fs, nfft=self.nfft, freq_max_hz=self.freq_max_hz)
        out_psd = compute_psd_vector(out_time, fs=self.fs, nfft=self.nfft, freq_max_hz=self.freq_max_hz)
        in_label = int(entry["inplane_annotation"])
        out_label = int(entry["outplane_annotation"])
        return in_time, in_psd, out_time, out_psd, in_label, out_label

    def _preload_worker(
        self,
        args: Tuple[int, dict],
    ) -> Tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int]:
        ds_idx, entry = args
        wi = int(entry.get("window_index", 0))
        in_time, in_psd, out_time, out_psd, in_label, out_label = _load_round2_pair_arrays(
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
        )
        return ds_idx, in_time, in_psd, out_time, out_psd, in_label, out_label

    def _preload_all(
        self,
        num_workers: int = 4,
        show_progress: bool = True,
    ) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int]]:
        n = len(self.indices)
        logger.info(f"预加载 round2 双向样本：{n} 条（workers={num_workers}）")
        preloaded: List[
            Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int]]
        ] = [None] * n
        tasks = [(ds_idx, self.entries[self.indices[ds_idx]]) for ds_idx in range(n)]

        if num_workers > 1 and n > 1:
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                futures = {executor.submit(_preload_round2_worker, task): task[0] for task in tasks}
                iterator = as_completed(futures)
                if show_progress:
                    from tqdm import tqdm

                    iterator = tqdm(iterator, total=n, desc="预加载 round2 四特征", unit="样本", ncols=100)
                for future in iterator:
                    ds_idx, in_time, in_psd, out_time, out_psd, in_label, out_label = future.result()
                    preloaded[ds_idx] = (in_time, in_psd, out_time, out_psd, in_label, out_label)
        else:
            indices = range(n)
            if show_progress:
                from tqdm import tqdm

                indices = tqdm(indices, desc="预加载 round2 四特征", unit="样本", ncols=100)
            for ds_idx in indices:
                entry = self.entries[self.indices[ds_idx]]
                preloaded[ds_idx] = self._preload_pair(entry)

        missing = [i for i, item in enumerate(preloaded) if item is None]
        if missing:
            raise RuntimeError(f"round2 预加载失败：{len(missing)} 个样本，例如 idx={missing[:5]}")
        return [item for item in preloaded if item is not None]

    def __getitem__(self, idx: int):
        if self._preloaded is not None:
            in_time, in_psd, out_time, out_psd, in_label, out_label = self._preloaded[idx]
        else:
            entry = self.entries[self.indices[idx]]
            in_time, in_psd, out_time, out_psd, in_label, out_label = self._preload_pair(entry)

        in_time_t = torch.from_numpy(in_time).float().unsqueeze(-1)
        in_psd_t = torch.from_numpy(in_psd).float().unsqueeze(-1)
        out_time_t = torch.from_numpy(out_time).float().unsqueeze(-1)
        out_psd_t = torch.from_numpy(out_psd).float().unsqueeze(-1)
        in_label_t = torch.tensor(in_label, dtype=torch.long)
        out_label_t = torch.tensor(out_label, dtype=torch.long)
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
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int]:
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
        raise ValueError(f"无法提取面内窗口：{inplane_file_path} idx={window_index}")

    out_data = extractor.load_file(outplane_file_path)
    out_sig = extractor.extract_window_from_data(
        out_data,
        window_index,
        window_size,
        metadata={"window_index": window_index},
        file_path=outplane_file_path,
    )
    if out_sig is None:
        raise ValueError(f"无法提取面外窗口：{outplane_file_path} idx={window_index}")

    in_time = np.asarray(in_sig, dtype=np.float32).reshape(-1)
    out_time = np.asarray(out_sig, dtype=np.float32).reshape(-1)
    in_psd = compute_psd_vector(in_time, fs=fs, nfft=nfft, freq_max_hz=freq_max_hz)
    out_psd = compute_psd_vector(out_time, fs=fs, nfft=nfft, freq_max_hz=freq_max_hz)
    return in_time, in_psd, out_time, out_psd, int(in_label), int(out_label)


def _preload_round2_worker(
    args: Tuple[int, dict],
) -> Tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int]:
    ds_idx, entry = args
    wi = int(entry.get("window_index", 0))
    in_time, in_psd, out_time, out_psd, in_label, out_label = _load_round2_pair_arrays(
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
    )
    return ds_idx, in_time, in_psd, out_time, out_psd, in_label, out_label


def build_round2_dataloaders(
    entries: List[dict],
    split_path: str,
    batch_size: int = 16,
    train_val_ratio: float = 0.8,
    random_seed: int = 42,
    window_size: int = 3000,
    fs: float = 50.0,
    nfft: int = 2048,
    freq_max_hz: float = 25.0,
    enable_denoise: bool = False,
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
        augmented.append(row)

    train_idx, val_idx = load_or_create_split(
        augmented, split_path, train_val_ratio=train_val_ratio, random_seed=random_seed
    )
    preload_kwargs = {
        "window_size": window_size,
        "fs": fs,
        "nfft": nfft,
        "freq_max_hz": freq_max_hz,
        "enable_denoise": enable_denoise,
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
        enable_preload_cache=enable_preload_cache,
        preload_num_workers=preload_num_workers,
        show_preload_progress=False,
    )
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )
    return train_loader, val_loader
