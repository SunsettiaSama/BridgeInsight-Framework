from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.chapter1_identifier.augment.features.spectrum import compute_psd_vector
from src.chapter1_identifier.augment.infer.identifier import DualStreamIdentifier
from src.identifier.dl.runner import FullDatasetRunner

logger = logging.getLogger(__name__)

_NORMAL_LABEL = 0


class _DirectionWindowDataset(Dataset):
    def __init__(
        self,
        records,
        direction: str,
        window_size: int,
        enable_denoise: bool = False,
        original_indices: Optional[List[int]] = None,
    ):
        from src.identifier.dl.runner import _make_extractor

        self._records = records
        self._direction = direction
        self._window_size = window_size
        self._enable_denoise = enable_denoise
        self._original_indices = original_indices or list(range(len(records)))
        self._extractor = None
        self._cache_path = None
        self._cache_data = None

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, idx: int):
        if self._extractor is None:
            from src.identifier.dl.runner import _make_extractor
            self._extractor = _make_extractor(enable_denoise=self._enable_denoise)

        rec = self._records[idx]
        meta_key = f"{self._direction}_meta"
        meta = getattr(rec, meta_key) or {}
        file_path = meta.get("file_path")
        if file_path != self._cache_path:
            self._cache_data = self._extractor.load_file(file_path)
            self._cache_path = file_path
        signal = self._extractor.extract_window_from_data(
            self._cache_data,
            rec.window_idx,
            self._window_size,
            metadata=meta,
            file_path=file_path,
        )
        return torch.from_numpy(signal).float(), self._original_indices[idx]


class DualStreamInferenceRunner:
    def __init__(
        self,
        identifier: DualStreamIdentifier,
        batch_size: int = 256,
        num_workers: int = 0,
        fs: float = 50.0,
        nfft: int = 2048,
        freq_max_hz: float = 25.0,
    ):
        self.identifier = identifier
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.fs = fs
        self.nfft = nfft
        self.freq_max_hz = freq_max_hz

    def _run_direction_proba(self, dataset, direction: str) -> Dict[int, np.ndarray]:
        from src.identifier.dl.runner import _make_extractor

        window_size = dataset.config.window_size
        enable_denoise = dataset.config.enable_denoise
        extractor = _make_extractor(enable_denoise=enable_denoise)

        if direction == "inplane":
            recs, idxs = FullDatasetRunner._validate_records(
                dataset._samples, window_size, extractor=extractor
            )
        else:
            recs, idxs = FullDatasetRunner._validate_outplane_records(
                dataset._samples, window_size, extractor=extractor
            )

        ds = _DirectionWindowDataset(
            records=recs,
            direction=direction,
            window_size=window_size,
            enable_denoise=enable_denoise,
            original_indices=idxs,
        )

        probas: Dict[int, np.ndarray] = {}
        batch_times: List[torch.Tensor] = []
        batch_idxs: List[int] = []

        def flush():
            if not batch_times:
                return
            time_batch = torch.stack(batch_times)
            psd_list = []
            for t in time_batch:
                psd = compute_psd_vector(
                    t.numpy(), fs=self.fs, nfft=self.nfft, freq_max_hz=self.freq_max_hz
                )
                psd_list.append(torch.from_numpy(psd).float())
            max_len = max(p.shape[0] for p in psd_list)
            padded = []
            for p in psd_list:
                if p.shape[0] < max_len:
                    pad = torch.zeros(max_len - p.shape[0])
                    p = torch.cat([p, pad], dim=0)
                padded.append(p)
            psd_batch = torch.stack(padded).unsqueeze(-1)
            time_batch = time_batch.unsqueeze(-1)
            proba_batch = self.identifier.predict_proba_batch(time_batch, psd_batch)
            for orig_idx, proba in zip(batch_idxs, proba_batch):
                probas[orig_idx] = proba
            batch_times.clear()
            batch_idxs.clear()

        for time_tensor, orig_idx in tqdm(ds, desc=f"DualStream {direction} 推理"):
            batch_times.append(time_tensor)
            batch_idxs.append(orig_idx)
            if len(batch_times) >= self.batch_size:
                flush()
        flush()
        return probas

    def run_with_proba(self, dataset) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray]]:
        inplane = self._run_direction_proba(dataset, "inplane")
        outplane = self._run_direction_proba(dataset, "outplane")
        return inplane, outplane

    @staticmethod
    def merge_proba_predictions(
        inplane: Dict[int, np.ndarray],
        outplane: Dict[int, np.ndarray],
    ) -> Dict[int, Tuple[int, np.ndarray]]:
        merged: Dict[int, Tuple[int, np.ndarray]] = {}
        all_idx = set(inplane.keys()) | set(outplane.keys())
        for idx in all_idx:
            in_p = inplane.get(idx)
            out_p = outplane.get(idx)
            if in_p is None and out_p is None:
                continue
            if in_p is None:
                pred = int(out_p.argmax())
                merged[idx] = (pred, out_p)
                continue
            if out_p is None:
                pred = int(in_p.argmax())
                merged[idx] = (pred, in_p)
                continue
            in_pred = int(in_p.argmax())
            out_pred = int(out_p.argmax())
            if in_pred != _NORMAL_LABEL:
                merged[idx] = (in_pred, in_p)
            elif out_pred != _NORMAL_LABEL:
                merged[idx] = (out_pred, out_p)
            else:
                merged[idx] = (_NORMAL_LABEL, (in_p + out_p) / 2.0)
        return merged
