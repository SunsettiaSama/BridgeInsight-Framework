from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from src.chapter1_identifier.augment.features.spectrum import compute_psd_vector
from src.data_processer.preprocess.get_data_vib import VICWindowExtractor

_LABEL_NAMES = ["Normal", "VIV", "RWIV", "Transition"]


class GoldReferenceFinder:
    def __init__(
        self,
        gold_entries: List[dict],
        window_size: int = 3000,
        fs: float = 50.0,
        nfft: int = 2048,
        freq_max_hz: float = 25.0,
    ):
        self.entries = gold_entries
        self.window_size = window_size
        self.fs = fs
        self.nfft = nfft
        self.freq_max_hz = freq_max_hz
        self._vectors: Optional[np.ndarray] = None
        self._extractor = VICWindowExtractor(enable_denoise=False)
        self._cache_path: Optional[str] = None
        self._cache_data = None

    def _load_window(self, file_path: str, window_index: int) -> np.ndarray:
        if file_path != self._cache_path:
            self._cache_data = self._extractor.load_file(file_path)
            self._cache_path = file_path
        sig = self._extractor.extract_window_from_data(
            self._cache_data,
            window_index,
            self.window_size,
            metadata={"window_index": window_index},
            file_path=file_path,
        )
        return np.asarray(sig, dtype=np.float32).reshape(-1)

    def _psd_vector(self, file_path: str, window_index: int) -> np.ndarray:
        signal = self._load_window(file_path, window_index)
        return compute_psd_vector(signal, fs=self.fs, nfft=self.nfft, freq_max_hz=self.freq_max_hz)

    def _ensure_index(self) -> None:
        if self._vectors is not None:
            return
        vectors = []
        for entry in self.entries:
            fp = entry.get("file_path")
            wi = int(entry.get("window_index", 0))
            vectors.append(self._psd_vector(fp, wi))
        self._vectors = np.stack(vectors, axis=0)

    @staticmethod
    def _cosine_similarity(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
        q = query / (np.linalg.norm(query) + 1e-8)
        m = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-8)
        return m @ q

    def find_topk(self, file_path: str, window_index: int, topk: int = 3) -> List[Dict]:
        self._ensure_index()
        query = self._psd_vector(file_path, window_index)
        scores = self._cosine_similarity(query, self._vectors)
        order = np.argsort(scores)[::-1][:topk]
        results = []
        for rank, idx in enumerate(order, start=1):
            entry = self.entries[int(idx)]
            label = int(entry.get("annotation", entry.get("class_id", 0)))
            results.append(
                {
                    "rank": rank,
                    "similarity": float(scores[int(idx)]),
                    "label": label,
                    "label_name": _LABEL_NAMES[label] if 0 <= label < len(_LABEL_NAMES) else str(label),
                    "file_path": entry.get("file_path"),
                    "window_index": int(entry.get("window_index", 0)),
                    "sensor_id": entry.get("sensor_id"),
                    "time": entry.get("time"),
                }
            )
        return results
