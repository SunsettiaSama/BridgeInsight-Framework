from __future__ import annotations

from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable, Deque, List, Optional, Tuple

import numpy as np
import torch

from src.chapter3_identifier.augment.features.spectrum import compute_psd_vector, psd_bin_count


def normalize_time_batch(time_batch: torch.Tensor) -> torch.Tensor:
    """与训练 DualStreamDataset 一致：(B, L, 1)。"""
    if time_batch.ndim == 2:
        return time_batch.unsqueeze(-1)
    if time_batch.ndim == 3 and time_batch.shape[-1] == 1:
        return time_batch
    if time_batch.ndim == 3 and time_batch.shape[1] == 1:
        return time_batch.transpose(1, 2)
    raise ValueError(f"无法规范化时序 batch，shape={tuple(time_batch.shape)}")


def normalize_psd_batch(psd_batch: torch.Tensor) -> torch.Tensor:
    """与训练 DualStreamDataset 一致：(B, psd_bins, 1)。"""
    if psd_batch.ndim == 2:
        return psd_batch.unsqueeze(-1)
    if psd_batch.ndim == 3 and psd_batch.shape[-1] == 1:
        return psd_batch
    if psd_batch.ndim == 3 and psd_batch.shape[1] == 1:
        return psd_batch.transpose(1, 2)
    raise ValueError(f"无法规范化 PSD batch，shape={tuple(psd_batch.shape)}")


def compute_psd_batch(
    time_batch: torch.Tensor,
    fs: float,
    nfft: int,
    freq_max_hz: float,
) -> torch.Tensor:
    time_batch = normalize_time_batch(time_batch)
    flat = time_batch.squeeze(-1).numpy()
    n_bins = psd_bin_count(fs=fs, nfft=nfft, freq_max_hz=freq_max_hz)
    psd_arr = np.empty((flat.shape[0], n_bins), dtype=np.float32)
    for i in range(flat.shape[0]):
        psd_arr[i] = compute_psd_vector(flat[i], fs=fs, nfft=nfft, freq_max_hz=freq_max_hz)
    return torch.from_numpy(psd_arr).unsqueeze(-1)


class AsyncPsdPipeline:
    """
    异步 Welch PSD：当前 batch 的频谱在后台线程计算，与下一 batch 的 DataLoader I/O 重叠。

    CUDA 说明：Welch 仍在 CPU（scipy）；GPU 仅跑 ResCNN forward。
    若需 GPU 频谱，可用 torch.stft 对齐训练频谱后再 non_blocking 传入 device，
    但需单独验证与 scipy.welch 数值一致性，此处暂不实现。
    """

    def __init__(
        self,
        fs: float,
        nfft: int,
        freq_max_hz: float,
        psd_workers: int = 2,
    ):
        self.fs = fs
        self.nfft = nfft
        self.freq_max_hz = freq_max_hz
        self._executor = ThreadPoolExecutor(
            max_workers=max(1, psd_workers),
            thread_name_prefix="infer-psd",
        )
        self._time_batch: Optional[torch.Tensor] = None
        self._idxs: Optional[List[int]] = None
        self._psd_future: Optional[Future] = None

    def submit(self, time_batch: torch.Tensor, idxs: List[int]) -> None:
        time_batch = normalize_time_batch(time_batch)
        self._time_batch = time_batch
        self._idxs = list(idxs)
        self._psd_future = self._executor.submit(
            compute_psd_batch,
            time_batch,
            self.fs,
            self.nfft,
            self.freq_max_hz,
        )

    def has_pending(self) -> bool:
        return self._psd_future is not None

    def consume(
        self,
        predict_fn,
        probas: dict,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self._psd_future is None or self._time_batch is None or self._idxs is None:
            raise RuntimeError("AsyncPsdPipeline 无待处理 batch")
        psd_batch = normalize_psd_batch(self._psd_future.result())
        time_batch = self._time_batch
        idxs = self._idxs
        proba_batch = predict_fn(time_batch, psd_batch)
        for orig_idx, proba in zip(idxs, proba_batch):
            probas[orig_idx] = proba
        self._time_batch = None
        self._idxs = None
        self._psd_future = None
        return time_batch, psd_batch

    def close(self) -> None:
        self._executor.shutdown(wait=True)


@dataclass
class PreparedJointBatch:
    idx_batch: List[int]
    in_time_batch: torch.Tensor
    out_time_batch: torch.Tensor
    in_psd_batch: torch.Tensor
    out_psd_batch: torch.Tensor
    in_context_batch: Optional[torch.Tensor] = None
    out_context_batch: Optional[torch.Tensor] = None
    wind_features_batch: Optional[torch.Tensor] = None
    window_slice_time_s: float = 0.0
    context_load_time_s: float = 0.0
    psd_time_s: float = 0.0


class AsyncJointBatchPipeline:
    """
    异步联合 batch 准备：short window + long context + PSD 在后台线程计算，
    主线程与 GPU forward 重叠下一 batch 的数据准备。
    """

    def __init__(
        self,
        prepare_fn: Callable[[List[int]], PreparedJointBatch],
        workers: int = 2,
        queue_depth: int = 2,
    ):
        self._prepare_fn = prepare_fn
        self._executor = ThreadPoolExecutor(
            max_workers=max(1, int(workers)),
            thread_name_prefix="infer-joint-batch",
        )
        self._queue: Deque[Future] = deque()
        self._max_depth = max(1, int(queue_depth))

    def submit(self, ds_indices: List[int]) -> None:
        if len(self._queue) >= self._max_depth:
            raise RuntimeError(f"AsyncJointBatchPipeline 队列已满：depth={self._max_depth}")
        self._queue.append(self._executor.submit(self._prepare_fn, list(ds_indices)))

    def has_pending(self) -> bool:
        return len(self._queue) > 0

    def consume(self) -> PreparedJointBatch:
        if not self._queue:
            raise RuntimeError("AsyncJointBatchPipeline 无待处理 batch")
        batch = self._queue.popleft().result()
        return batch

    def close(self) -> None:
        while self._queue:
            self._queue.popleft().result()
        self._executor.shutdown(wait=True)
