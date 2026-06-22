from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from src.chapter3_identifier.augment._bootstrap import ensure_paths
from src.chapter3_identifier.augment.datasets.dual_stream_dataset import (
    DEFAULT_CONTEXT_ALLOW_CROSS_FILE,
    DEFAULT_CONTEXT_INPUT_SIZE,
    DEFAULT_CONTEXT_TOTAL_SECONDS,
)
from src.chapter3_identifier.augment.features.spectrum import compute_psd_vector
from src.chapter3_identifier.augment.models.dual_stream_res_cnn import DualStreamResCNN
from src.chapter3_identifier.augment.models.quad_stream_dual_head_res_cnn import (
    QuadStreamDualHeadContextResCNN,
    QuadStreamDualHeadResCNN,
    QuadStreamSerialContextDualHeadResCNN,
)
from src.chapter3_identifier.augment.labels import get_label_names
from src.chapter3_identifier.augment.settings import (
    BranchConfig,
    DualStreamResCNNConfig,
    load_dual_stream_model_config,
)

ensure_paths()


class DualStreamIdentifier:
    def __init__(
        self,
        model: DualStreamResCNN,
        device: Optional[str] = None,
        fs: float = 50.0,
        nfft: int = 2048,
        freq_max_hz: float = 25.0,
        label_names: Optional[list[str]] = None,
    ):
        self.model = model
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model.to(self.device)
        self.model.eval()
        self.fs = fs
        self.nfft = nfft
        self.freq_max_hz = freq_max_hz
        self.label_names = label_names or get_label_names()
        self.num_classes = len(self.label_names)
        self.model_type = "dual_stream_single_head"
        if isinstance(self.model, QuadStreamSerialContextDualHeadResCNN):
            self.model_type = "quad_stream_serial_context_dual_head"
        elif isinstance(self.model, QuadStreamDualHeadContextResCNN):
            self.model_type = "quad_stream_dual_head_context"
        elif isinstance(self.model, QuadStreamDualHeadResCNN):
            self.model_type = "quad_stream_dual_head"
        self.context_input_size = DEFAULT_CONTEXT_INPUT_SIZE
        self.context_total_seconds = DEFAULT_CONTEXT_TOTAL_SECONDS
        self.context_allow_cross_file = DEFAULT_CONTEXT_ALLOW_CROSS_FILE
        self.context_mode = "short_only"
        self.wind_feature_dim = int(getattr(self.model, "wind_feature_dim", 0))

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        model_config_path: Optional[str] = None,
        device: Optional[str] = None,
        fs: float = 50.0,
        nfft: int = 2048,
        freq_max_hz: float = 25.0,
        label_names: Optional[list[str]] = None,
    ) -> "DualStreamIdentifier":
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        cfg_dict = ckpt.get("config")
        if cfg_dict is None:
            cfg_dict = load_dual_stream_model_config(model_config_path)
            from src.chapter3_identifier.augment.features.spectrum import psd_bin_count
            cfg_dict.setdefault("spec_branch", {})["input_size"] = psd_bin_count(fs, nfft, freq_max_hz)
        model_type = str(cfg_dict.get("model_type", "dual_stream_single_head"))
        wind_feature_dim = int(cfg_dict.get("wind_feature_dim", 0))
        if model_type == "quad_stream_dual_head":
            time_cfg = BranchConfig(**cfg_dict["time_branch"])
            spec_cfg = BranchConfig(**cfg_dict["spec_branch"])
            model = QuadStreamDualHeadResCNN(
                time_branch_cfg=time_cfg,
                spec_branch_cfg=spec_cfg,
                num_classes=int(cfg_dict.get("num_classes", 4)),
                fusion_hidden_dim=int(cfg_dict.get("fusion_hidden_dim", 128)),
                fusion_dropout=float(cfg_dict.get("fusion_dropout", 0.1)),
                cross_attn_heads=int(cfg_dict.get("cross_attn_heads", 4)),
                wind_feature_dim=wind_feature_dim,
            )
        elif model_type in {"quad_stream_dual_head_context", "quad_stream_serial_context_dual_head"}:
            time_cfg = BranchConfig(**cfg_dict["time_branch"])
            spec_cfg = BranchConfig(**cfg_dict["spec_branch"])
            context_cfg = BranchConfig(**cfg_dict["context_branch"])
            model_cls = (
                QuadStreamSerialContextDualHeadResCNN
                if model_type == "quad_stream_serial_context_dual_head"
                else QuadStreamDualHeadContextResCNN
            )
            model = model_cls(
                time_branch_cfg=time_cfg,
                spec_branch_cfg=spec_cfg,
                context_branch_cfg=context_cfg,
                num_classes=int(cfg_dict.get("num_classes", 4)),
                fusion_hidden_dim=int(cfg_dict.get("fusion_hidden_dim", 128)),
                fusion_dropout=float(cfg_dict.get("fusion_dropout", 0.1)),
                cross_attn_heads=int(cfg_dict.get("cross_attn_heads", 4)),
                wind_feature_dim=wind_feature_dim,
            )
        else:
            model_cfg = DualStreamResCNNConfig.from_dict(cfg_dict)
            model = DualStreamResCNN(model_cfg)
        model.load_state_dict(ckpt["model_state_dict"])
        identifier = cls(
            model,
            device=device,
            fs=fs,
            nfft=nfft,
            freq_max_hz=freq_max_hz,
            label_names=label_names,
        )
        identifier.context_input_size = int(cfg_dict.get("context_input_size", DEFAULT_CONTEXT_INPUT_SIZE))
        identifier.context_total_seconds = float(cfg_dict.get("context_total_seconds", DEFAULT_CONTEXT_TOTAL_SECONDS))
        identifier.context_allow_cross_file = bool(
            cfg_dict.get("context_allow_cross_file", DEFAULT_CONTEXT_ALLOW_CROSS_FILE)
        )
        uses_context = model_type in {"quad_stream_dual_head_context", "quad_stream_serial_context_dual_head"}
        identifier.context_mode = str(cfg_dict.get("context_mode", "short_long" if uses_context else "short_only"))
        identifier.wind_feature_dim = int(wind_feature_dim)
        return identifier

    def _prepare_batch(self, time_np: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        time_np = np.asarray(time_np, dtype=np.float32).reshape(-1)
        psd = compute_psd_vector(time_np, fs=self.fs, nfft=self.nfft, freq_max_hz=self.freq_max_hz)
        time_t = torch.from_numpy(time_np).float().view(1, -1, 1).to(self.device)
        psd_t = torch.from_numpy(psd).float().view(1, -1, 1).to(self.device)
        return time_t, psd_t

    @torch.no_grad()
    def predict_proba(self, time_signal: np.ndarray) -> np.ndarray:
        if self.model_type != "dual_stream_single_head":
            raise ValueError("当前 checkpoint 为双头模型，请调用双向批量预测接口")
        time_t, psd_t = self._prepare_batch(time_signal)
        logits = self.model(time_t, psd_t)
        proba = F.softmax(logits, dim=1).cpu().numpy()[0]
        return proba.astype(np.float32)

    @torch.no_grad()
    def predict_batch(self, batch_time: torch.Tensor, batch_psd: torch.Tensor) -> np.ndarray:
        if self.model_type != "dual_stream_single_head":
            raise ValueError("当前 checkpoint 为双头模型，请调用双向批量预测接口")
        batch_time = batch_time.to(self.device)
        batch_psd = batch_psd.to(self.device)
        logits = self.model(batch_time, batch_psd)
        return logits.argmax(dim=1).cpu().numpy()

    @torch.no_grad()
    def predict_proba_batch(self, batch_time: torch.Tensor, batch_psd: torch.Tensor) -> np.ndarray:
        batch_time = batch_time.to(self.device, non_blocking=True)
        batch_psd = batch_psd.to(self.device, non_blocking=True)
        if self.model_type != "dual_stream_single_head":
            raise ValueError("当前 checkpoint 为双头模型，请调用 predict_dual_proba_batch")
        logits = self.model(batch_time, batch_psd)
        return F.softmax(logits, dim=1).cpu().numpy()

    @torch.no_grad()
    def predict_dual_proba_batch(
        self,
        in_time: torch.Tensor,
        in_psd: torch.Tensor,
        out_time: torch.Tensor,
        out_psd: torch.Tensor,
        in_context: torch.Tensor | None = None,
        out_context: torch.Tensor | None = None,
        wind_features: torch.Tensor | None = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if self.model_type not in {"quad_stream_dual_head", "quad_stream_dual_head_context", "quad_stream_serial_context_dual_head"}:
            raise ValueError("当前 checkpoint 不是双头模型，不能做联合双向预测")
        in_time = in_time.to(self.device, non_blocking=True)
        in_psd = in_psd.to(self.device, non_blocking=True)
        out_time = out_time.to(self.device, non_blocking=True)
        out_psd = out_psd.to(self.device, non_blocking=True)
        if wind_features is not None:
            wind_features = wind_features.to(self.device, non_blocking=True)
        if self.model_type in {"quad_stream_dual_head_context", "quad_stream_serial_context_dual_head"}:
            use_context = self.context_mode == "short_long"
            if use_context:
                if in_context is None or out_context is None:
                    raise ValueError("context_mode=short_long 需要 in_context/out_context")
                in_context = in_context.to(self.device, non_blocking=True)
                out_context = out_context.to(self.device, non_blocking=True)
            in_logits, out_logits = self.model(
                in_time,
                in_psd,
                in_context,
                out_time,
                out_psd,
                out_context,
                wind_features=wind_features,
                use_context=use_context,
            )
        else:
            in_logits, out_logits = self.model(
                in_time,
                in_psd,
                out_time,
                out_psd,
                wind_features=wind_features,
            )
        in_proba = F.softmax(in_logits, dim=1).cpu().numpy()
        out_proba = F.softmax(out_logits, dim=1).cpu().numpy()
        return in_proba, out_proba
