from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from src.chapter1_identifier.augment._bootstrap import ensure_paths
from src.chapter1_identifier.augment.features.spectrum import compute_psd_vector
from src.chapter1_identifier.augment.models.dual_stream_res_cnn import DualStreamResCNN
from src.chapter1_identifier.augment.settings import DualStreamResCNNConfig, load_dual_stream_model_config

ensure_paths()


class DualStreamIdentifier:
    NUM_CLASSES = 4
    LABEL_NAMES = {0: "Normal", 1: "VIV", 2: "RWIV", 3: "Transition"}

    def __init__(
        self,
        model: DualStreamResCNN,
        device: Optional[str] = None,
        fs: float = 50.0,
        nfft: int = 2048,
        freq_max_hz: float = 25.0,
    ):
        self.model = model
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model.to(self.device)
        self.model.eval()
        self.fs = fs
        self.nfft = nfft
        self.freq_max_hz = freq_max_hz

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        model_config_path: Optional[str] = None,
        device: Optional[str] = None,
        fs: float = 50.0,
        nfft: int = 2048,
        freq_max_hz: float = 25.0,
    ) -> "DualStreamIdentifier":
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        cfg_dict = ckpt.get("config")
        if cfg_dict is None:
            cfg_dict = load_dual_stream_model_config(model_config_path)
            from src.chapter1_identifier.augment.features.spectrum import psd_bin_count
            cfg_dict.setdefault("spec_branch", {})["input_size"] = psd_bin_count(fs, nfft, freq_max_hz)
        model_cfg = DualStreamResCNNConfig.from_dict(cfg_dict)
        model = DualStreamResCNN(model_cfg)
        model.load_state_dict(ckpt["model_state_dict"])
        return cls(model, device=device, fs=fs, nfft=nfft, freq_max_hz=freq_max_hz)

    def _prepare_batch(self, time_np: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        time_np = np.asarray(time_np, dtype=np.float32).reshape(-1)
        psd = compute_psd_vector(time_np, fs=self.fs, nfft=self.nfft, freq_max_hz=self.freq_max_hz)
        time_t = torch.from_numpy(time_np).float().view(1, -1, 1).to(self.device)
        psd_t = torch.from_numpy(psd).float().view(1, -1, 1).to(self.device)
        return time_t, psd_t

    @torch.no_grad()
    def predict_proba(self, time_signal: np.ndarray) -> np.ndarray:
        time_t, psd_t = self._prepare_batch(time_signal)
        logits = self.model(time_t, psd_t)
        proba = F.softmax(logits, dim=1).cpu().numpy()[0]
        return proba.astype(np.float32)

    @torch.no_grad()
    def predict_batch(self, batch_time: torch.Tensor, batch_psd: torch.Tensor) -> np.ndarray:
        batch_time = batch_time.to(self.device)
        batch_psd = batch_psd.to(self.device)
        logits = self.model(batch_time, batch_psd)
        return logits.argmax(dim=1).cpu().numpy()

    @torch.no_grad()
    def predict_proba_batch(self, batch_time: torch.Tensor, batch_psd: torch.Tensor) -> np.ndarray:
        batch_time = batch_time.to(self.device)
        batch_psd = batch_psd.to(self.device)
        logits = self.model(batch_time, batch_psd)
        return F.softmax(logits, dim=1).cpu().numpy()
