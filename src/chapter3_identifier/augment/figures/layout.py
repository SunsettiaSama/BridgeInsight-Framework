from __future__ import annotations

from copy import deepcopy

DEFAULT_LAYOUT_PROFILE = "wide_fill_v3"

_LAYOUT_PRESETS = {
    "wide_fill_v3": {
        "sample_sizes": {
            "timeseries": [6.8, 1.9],
            "spectrum": [6.8, 1.35],
            "trajectory": [4.0, 4.0],
            "prediction": [8.2, 1.7],
            "square": [4.0, 4.0],
            "wind_timeseries": [8.2, 1.9],
        },
        "context_sizes": {
            "timeseries": [16.0, 4.0],
            "spectrogram": [4.0, 4.0],
        },
        "ui": {
            "context_grid_ratio": "36%",
            "feature_grid_columns": "minmax(0, 2fr) minmax(0, 2fr) minmax(0, 1fr) minmax(0, 1fr)",
            "summary_grid_columns": "minmax(0, 4fr) minmax(0, 2fr)",
            "context_grid_columns": "minmax(0, 2fr) minmax(0, 2fr) minmax(0, 1fr) minmax(0, 1fr)",
        },
    },
    "wide_fill_v2": {
        "sample_sizes": {
            "timeseries": [6.8, 1.9],
            "spectrum": [6.8, 1.6],
            "trajectory": [4.0, 4.0],
            "prediction": [8.2, 1.7],
            "square": [4.0, 4.0],
            "wind_timeseries": [8.2, 1.9],
        },
        "context_sizes": {
            "timeseries": [8.0, 4.0],
            "spectrogram": [8.0, 4.0],
        },
        "ui": {
            "context_grid_ratio": "31%",
            "feature_grid_columns": "minmax(0, 2fr) minmax(0, 2fr) minmax(0, 1fr) minmax(0, 1fr)",
            "summary_grid_columns": "minmax(0, 4fr) minmax(0, 2fr)",
            "context_grid_columns": "minmax(0, 2fr) minmax(0, 2fr) minmax(0, 1fr)",
        },
    },
    "wide_fill_v1": {
        "sample_sizes": {
            "timeseries": [6.8, 1.9],
            "spectrum": [6.8, 1.6],
            "trajectory": [4.0, 4.0],
            "prediction": [8.2, 1.7],
            "square": [4.0, 4.0],
            "wind_timeseries": [8.2, 1.9],
        },
        "context_sizes": {
            "timeseries": [15.0, 3.0],
            "spectrogram": [12.0, 3.0],
        },
        "ui": {
            "context_grid_ratio": "31%",
            "feature_grid_columns": "minmax(0, 2fr) minmax(0, 2fr) minmax(0, 1fr) minmax(0, 1fr)",
            "summary_grid_columns": "minmax(0, 4fr) minmax(0, 2fr)",
            "context_grid_columns": "minmax(0, 2fr) minmax(0, 2fr) minmax(0, 1fr)",
        },
    },
    "default": {
        "sample_sizes": {
            "timeseries": [4.0, 2.0],
            "spectrum": [4.0, 2.0],
            "trajectory": [4.0, 4.0],
            "prediction": [4.0, 2.5],
            "square": [4.0, 4.0],
            "wind_timeseries": [8.0, 2.2],
        },
        "context_sizes": {
            "timeseries": [8.0, 2.2],
            "spectrogram": [8.0, 2.2],
        },
        "ui": {
            "context_grid_ratio": "31%",
            "feature_grid_columns": "minmax(0, 2fr) minmax(0, 2fr) minmax(0, 1fr) minmax(0, 1fr)",
            "summary_grid_columns": "minmax(0, 4fr) minmax(0, 2fr)",
            "context_grid_columns": "minmax(0, 2fr) minmax(0, 2fr) minmax(0, 1fr)",
        },
    },
}


def _resolve_profile(layout_profile: str) -> dict:
    key = layout_profile if layout_profile in _LAYOUT_PRESETS else DEFAULT_LAYOUT_PROFILE
    return _LAYOUT_PRESETS[key]


def sample_fig_size(layout_profile: str, kind: str) -> tuple[float, float]:
    profile = _resolve_profile(layout_profile)
    sizes = profile["sample_sizes"]
    fallback = _LAYOUT_PRESETS[DEFAULT_LAYOUT_PROFILE]["sample_sizes"]
    size = sizes.get(kind, fallback.get(kind, [4.0, 2.0]))
    return float(size[0]), float(size[1])


def context_fig_size(layout_profile: str, kind: str) -> tuple[float, float]:
    profile = _resolve_profile(layout_profile)
    sizes = profile["context_sizes"]
    fallback = _LAYOUT_PRESETS[DEFAULT_LAYOUT_PROFILE]["context_sizes"]
    size = sizes.get(kind, fallback.get("timeseries", [8.0, 2.2]))
    return float(size[0]), float(size[1])


def layout_protocol(layout_profile: str) -> dict:
    profile = _resolve_profile(layout_profile)
    sample_sizes = profile["sample_sizes"]
    context_sizes = profile["context_sizes"]

    wide_w, wide_h = sample_sizes["spectrum"]
    long_w, long_h = sample_sizes["timeseries"]
    square_w, square_h = sample_sizes.get("square", [4.0, 4.0])
    wind_ts_w, wind_ts_h = sample_sizes.get("wind_timeseries", sample_sizes["timeseries"])
    pred_w, pred_h = sample_sizes["prediction"]
    ctx_w, ctx_h = context_sizes["timeseries"]
    ctx_sp_w, ctx_sp_h = context_sizes["spectrogram"]

    return {
        "layout_profile": layout_profile if layout_profile in _LAYOUT_PRESETS else DEFAULT_LAYOUT_PROFILE,
        "sample_sizes": deepcopy(sample_sizes),
        "context_sizes": deepcopy(context_sizes),
        "slot_aspect_ratios": {
            "wide": float(wide_w) / max(float(wide_h), 1e-6),
            "long": float(long_w) / max(float(long_h), 1e-6),
            "square": float(square_w) / max(float(square_h), 1e-6),
            "wind_timeseries": float(wind_ts_w) / max(float(wind_ts_h), 1e-6),
            "prediction": float(pred_w) / max(float(pred_h), 1e-6),
            "context_timeseries": float(ctx_w) / max(float(ctx_h), 1e-6),
            "context_spectrogram": float(ctx_sp_w) / max(float(ctx_sp_h), 1e-6),
        },
        "ui": deepcopy(profile.get("ui", {})),
    }
