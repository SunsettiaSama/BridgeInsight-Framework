from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from src.chapter1_identifier.augment.annotation.split import load_or_create_split
from src.chapter1_identifier.augment.features.spectrum import compute_psd_vector
from src.data_processer.preprocess.get_data_vib import VICWindowExtractor

logger = logging.getLogger(__name__)


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
    ):
        self.entries = entries
        self.indices = indices if indices is not None else list(range(len(entries)))
        self.window_size = window_size
        self.fs = fs
        self.nfft = nfft
        self.freq_max_hz = freq_max_hz
        self.extractor = VICWindowExtractor(enable_denoise=enable_denoise)
        self._cache_path: Optional[str] = None
        self._cache_data = None

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

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        entry = self.entries[self.indices[idx]]
        file_path = entry["file_path"]
        window_index = int(entry.get("window_index", 0))
        label = int(entry.get("annotation", entry.get("class_id", 0)))

        time_sig = self._load_window(file_path, window_index)
        psd = compute_psd_vector(time_sig, fs=self.fs, nfft=self.nfft, freq_max_hz=self.freq_max_hz)

        time_tensor = torch.from_numpy(time_sig).float().unsqueeze(-1)
        psd_tensor = torch.from_numpy(psd).float().unsqueeze(-1)
        label_tensor = torch.tensor(label, dtype=torch.long)
        return time_tensor, psd_tensor, label_tensor


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
):
    from torch.utils.data import DataLoader

    train_idx, val_idx = load_or_create_split(
        entries, split_path, train_val_ratio=train_val_ratio, random_seed=random_seed
    )
    train_ds = DualStreamDataset(
        entries, train_idx, window_size=window_size, fs=fs, nfft=nfft, freq_max_hz=freq_max_hz
    )
    val_ds = DualStreamDataset(
        entries, val_idx, window_size=window_size, fs=fs, nfft=nfft, freq_max_hz=freq_max_hz
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader
