from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.chapter3_identifier.augment.features.spectrum import compute_psd_vector
from src.chapter3_identifier.augment.infer.batch_ops import AsyncPsdPipeline, normalize_time_batch
from src.chapter3_identifier.augment.infer.identifier import DualStreamIdentifier
from src.chapter3_identifier.identifier.dl.progress import iter_progress
from src.identifier.dl.runner import FullDatasetRunner

logger = logging.getLogger(__name__)

_NORMAL_LABEL = 0


def _collate_windows(batch):
    times, idxs = zip(*batch)
    return torch.stack(times), list(idxs)


class _DirectionWindowDataset(Dataset):
    def __init__(
        self,
        records,
        direction: str,
        window_size: int,
        enable_denoise: bool = False,
        original_indices: Optional[List[int]] = None,
    ):
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
        time_sig = torch.from_numpy(np.asarray(signal, dtype=np.float32).reshape(-1)).float()
        return time_sig.unsqueeze(-1), self._original_indices[idx]


class _PairedWindowDataset(Dataset):
    def __init__(
        self,
        records,
        window_size: int,
        enable_denoise: bool = False,
        original_indices: Optional[List[int]] = None,
    ):
        self._records = records
        self._window_size = window_size
        self._enable_denoise = enable_denoise
        self._original_indices = original_indices or list(range(len(records)))
        self._extractor = None
        self._cache_path = None
        self._cache_data = None

    def __len__(self) -> int:
        return len(self._records)

    def _load_signal(self, file_path: str, window_idx: int, metadata: dict) -> torch.Tensor:
        if file_path != self._cache_path:
            self._cache_data = self._extractor.load_file(file_path)
            self._cache_path = file_path
        signal = self._extractor.extract_window_from_data(
            self._cache_data,
            window_idx,
            self._window_size,
            metadata=metadata,
            file_path=file_path,
        )
        if signal is None:
            raise ValueError(f"无法提取窗口：{file_path} idx={window_idx}")
        arr = np.asarray(signal, dtype=np.float32).reshape(-1)
        return torch.from_numpy(arr).float().unsqueeze(-1)

    def __getitem__(self, idx: int):
        if self._extractor is None:
            from src.identifier.dl.runner import _make_extractor

            self._extractor = _make_extractor(enable_denoise=self._enable_denoise)

        rec = self._records[idx]
        in_meta = rec.inplane_meta or {}
        out_meta = rec.outplane_meta or {}
        in_time = self._load_signal(in_meta.get("file_path"), rec.window_idx, in_meta)
        out_time = self._load_signal(out_meta.get("file_path"), rec.window_idx, out_meta)
        return in_time, out_time, self._original_indices[idx]


class DualStreamInferenceRunner:
    def __init__(
        self,
        identifier: DualStreamIdentifier,
        batch_size: int = 256,
        num_workers: int = 4,
        psd_workers: int = 2,
        fs: float = 50.0,
        nfft: int = 2048,
        freq_max_hz: float = 25.0,
    ):
        self.identifier = identifier
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.psd_workers = psd_workers
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

        loader = DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=_collate_windows,
            pin_memory=torch.cuda.is_available(),
        )

        probas: Dict[int, np.ndarray] = {}
        pipeline = AsyncPsdPipeline(
            fs=self.fs,
            nfft=self.nfft,
            freq_max_hz=self.freq_max_hz,
            psd_workers=self.psd_workers,
        )

        predict_fn = self.identifier.predict_proba_batch

        for time_batch, idx_batch in iter_progress(
            loader, total=len(loader), desc=f"DualStream {direction} 推理"
        ):
            time_batch = normalize_time_batch(time_batch)
            if pipeline.has_pending():
                pipeline.consume(predict_fn, probas)
            pipeline.submit(time_batch, idx_batch)

        if pipeline.has_pending():
            pipeline.consume(predict_fn, probas)

        pipeline.close()
        return probas

    def run_with_proba(self, dataset) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray]]:
        if self.identifier.model_type == "quad_stream_dual_head":
            return self._run_joint_proba(dataset)
        inplane = self._run_direction_proba(dataset, "inplane")
        outplane = self._run_direction_proba(dataset, "outplane")
        return inplane, outplane

    def _run_joint_proba(self, dataset) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray]]:
        from src.identifier.dl.runner import _make_extractor

        window_size = dataset.config.window_size
        enable_denoise = dataset.config.enable_denoise
        extractor = _make_extractor(enable_denoise=enable_denoise)
        _, in_idxs = FullDatasetRunner._validate_records(dataset._samples, window_size, extractor=extractor)
        _, out_idxs = FullDatasetRunner._validate_outplane_records(
            dataset._samples, window_size, extractor=extractor
        )
        valid_idx = sorted(set(in_idxs) & set(out_idxs))
        valid_records = [dataset._samples[i] for i in valid_idx]
        ds = _PairedWindowDataset(
            records=valid_records,
            window_size=window_size,
            enable_denoise=enable_denoise,
            original_indices=valid_idx,
        )
        loader = DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

        in_probas: Dict[int, np.ndarray] = {}
        out_probas: Dict[int, np.ndarray] = {}
        for in_time_batch, out_time_batch, idx_batch in iter_progress(
            loader, total=len(loader), desc="round2 联合双向推理"
        ):
            in_time_batch = normalize_time_batch(in_time_batch)
            out_time_batch = normalize_time_batch(out_time_batch)

            in_psd = []
            out_psd = []
            in_np = in_time_batch.squeeze(-1).numpy()
            out_np = out_time_batch.squeeze(-1).numpy()
            for row in in_np:
                in_psd.append(
                    compute_psd_vector(
                        row,
                        fs=self.fs,
                        nfft=self.nfft,
                        freq_max_hz=self.freq_max_hz,
                    )
                )
            for row in out_np:
                out_psd.append(
                    compute_psd_vector(
                        row,
                        fs=self.fs,
                        nfft=self.nfft,
                        freq_max_hz=self.freq_max_hz,
                    )
                )
            in_psd_batch = torch.from_numpy(np.asarray(in_psd, dtype=np.float32)).unsqueeze(-1)
            out_psd_batch = torch.from_numpy(np.asarray(out_psd, dtype=np.float32)).unsqueeze(-1)
            in_batch_proba, out_batch_proba = self.identifier.predict_dual_proba_batch(
                in_time_batch,
                in_psd_batch,
                out_time_batch,
                out_psd_batch,
            )
            for idx, in_p, out_p in zip(idx_batch, in_batch_proba, out_batch_proba):
                in_probas[int(idx)] = in_p
                out_probas[int(idx)] = out_p
        return in_probas, out_probas

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
