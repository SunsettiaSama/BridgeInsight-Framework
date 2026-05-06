"""
共享管线模块：VIV 样本加载、统计量计算、风数据对齐。
供 fig4_19 到 fig4_28 统一 import 使用。

数据流：
  DL 结果 JSON  → load_latest_result → get_viv_samples
               → load_enriched_stats（从 enriched_stats 加载统计量）
  MECC 结果 JSON → load_latest_result → get_viv_samples
               → compute_signal_stats（从原始信号实时计算统计量）
"""

import sys
import json
import numpy as np
from pathlib import Path
from scipy import signal as scipy_signal
from scipy import stats as scipy_stats

_project_root = Path(__file__).parent.parent.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.data_processer.io_unpacker import UNPACK
from src.identifier.deeplearning_methods import FullDatasetRunner


# ==================== 颜色常量（供各脚本统一引用） ====================
from src.figure_paintings.figs_for_thesis.config import (
    VIV_INPLANE_COLOR, VIV_OUTPLANE_COLOR,
)
DL_INPLANE_COLOR    = VIV_INPLANE_COLOR    # '#8074C8' 深紫
DL_OUTPLANE_COLOR   = VIV_OUTPLANE_COLOR   # '#E3625D' 珊瑚红
MECC_INPLANE_COLOR  = '#2CA02C'             # 草绿
MECC_OUTPLANE_COLOR = '#FF7F0E'             # 橙


# ==================== 信号处理常量 ====================
_FS          = 50.0
_WINDOW_SIZE = 3000
_NFFT        = 2048
_FREQ_LIMIT  = 25.0
_VIV_CLASS   = 1


# ==================== 结果路径 ====================
def default_dl_result_glob(project_root: Path) -> Path:
    return project_root / "results" / "identification_result" / "res_cnn_full_dataset_*.json"


def default_mecc_result_glob(project_root: Path) -> Path:
    return project_root / "results" / "identification_result_mecc_viv" / "mecc_viv_only_*.json"


def default_enriched_dir(project_root: Path) -> Path:
    return project_root / "results" / "enriched_stats" / "class_1_viv"


# ==================== 结果加载 ====================
def load_latest_result(full_glob: Path) -> dict:
    """加载最新结果 JSON（根据 full_glob 中的目录和 glob 模式自动匹配）。"""
    parent  = full_glob.parent
    pattern = full_glob.name
    if not parent.exists():
        raise FileNotFoundError(f"结果目录不存在：{parent}")
    files = sorted(parent.glob(pattern))
    if not files:
        raise FileNotFoundError(f"在 {parent} 中未找到匹配 {pattern!r} 的文件")
    print(f"  加载识别结果：{files[-1].name}")
    return FullDatasetRunner.load_result(str(files[-1]))


def get_viv_samples(result: dict, max_n: int = 0, seed: int = 42) -> list:
    """从 result 中筛选 VIV 样本（label=1），可选随机抽样。"""
    predictions     = {int(k): int(v) for k, v in result["predictions"].items()}
    sample_metadata = result.get("sample_metadata", {})

    viv_samples = []
    for idx, pred_label in predictions.items():
        if pred_label != _VIV_CLASS:
            continue
        meta = sample_metadata.get(str(idx))
        if meta is None:
            continue
        in_path  = meta.get("inplane_file_path")
        out_path = meta.get("outplane_file_path")
        if not in_path or not out_path:
            continue
        viv_samples.append({
            "idx":                idx,
            "window_idx":         meta["window_idx"],
            "inplane_sensor_id":  meta.get("inplane_sensor_id", ""),
            "outplane_sensor_id": meta.get("outplane_sensor_id", ""),
            "inplane_file_path":  in_path,
            "outplane_file_path": out_path,
            "timestamp":          meta.get("timestamp", []),
        })

    if max_n and len(viv_samples) > max_n:
        rng    = np.random.default_rng(seed)
        chosen = rng.choice(len(viv_samples), size=max_n, replace=False)
        viv_samples = [viv_samples[i] for i in sorted(chosen.tolist())]

    return viv_samples


# ==================== 原始信号统计 ====================
def _load_window(file_path: str, window_idx: int, unpacker) -> np.ndarray | None:
    path = Path(file_path)
    if not path.exists():
        return None
    raw   = np.array(unpacker.VIC_DATA_Unpack(str(path)))
    start = window_idx * _WINDOW_SIZE
    end   = start + _WINDOW_SIZE
    if end > len(raw):
        return None
    return raw[start:end]


def _welch_stats(sig: np.ndarray):
    """计算 PSD 并返回：主频、主频能量占比、归一化 cumsum。"""
    f, psd = scipy_signal.welch(
        sig, fs=_FS,
        nperseg=_NFFT // 2, noverlap=_NFFT // 4, nfft=_NFFT,
    )
    mask    = f <= _FREQ_LIMIT
    f_lim   = f[mask]
    psd_lim = psd[mask]
    if len(psd_lim) == 0 or psd_lim.sum() == 0:
        return None, None, None

    dom_idx  = int(np.argmax(psd_lim))
    dom_freq = float(f_lim[dom_idx])
    half_w   = 3
    lo       = max(0, dom_idx - half_w)
    hi       = min(len(psd_lim) - 1, dom_idx + half_w)
    dom_energy = float(psd_lim[lo:hi + 1].sum() / psd_lim.sum())

    arr_sorted = np.sort(psd_lim)[::-1]
    total      = arr_sorted.sum()
    cumsum     = np.cumsum(arr_sorted) / total if total > 0 else None

    return dom_freq, dom_energy, cumsum


def compute_signal_stats(samples: list, source: str = "") -> dict:
    """
    从原始信号计算 RMS、峭度、主频/能量、累积能量等统计量。
    返回 dict 与 load_enriched_stats 返回结构相同。
    """
    unpacker = UNPACK(init_path=False)

    rms_pairs  = []
    kurt_in    = []
    kurt_out   = []
    freq_in    = []
    freq_out   = []
    energy_in  = []
    energy_out = []
    cumsum_in  = []
    cumsum_out = []

    total = len(samples)
    for i, s in enumerate(samples):
        if (i + 1) % 500 == 0:
            print(f"  [{source}] 处理 {i + 1}/{total}...")

        sig_i = _load_window(s["inplane_file_path"],  s["window_idx"], unpacker)
        sig_o = _load_window(s["outplane_file_path"], s["window_idx"], unpacker)

        rms_i = rms_o = None

        if sig_i is not None and len(sig_i) >= 64:
            rms_i = float(np.sqrt(np.mean(sig_i ** 2)))
            kurt_in.append(float(scipy_stats.kurtosis(sig_i, fisher=True)))
            fi, ei, cs_i = _welch_stats(sig_i)
            if fi is not None:
                freq_in.append(fi)
                energy_in.append(ei)
                if cs_i is not None:
                    cumsum_in.append(cs_i)

        if sig_o is not None and len(sig_o) >= 64:
            rms_o = float(np.sqrt(np.mean(sig_o ** 2)))
            kurt_out.append(float(scipy_stats.kurtosis(sig_o, fisher=True)))
            fo, eo, cs_o = _welch_stats(sig_o)
            if fo is not None:
                freq_out.append(fo)
                energy_out.append(eo)
                if cs_o is not None:
                    cumsum_out.append(cs_o)

        if rms_i is not None and rms_o is not None:
            rms_pairs.append((rms_i, rms_o))

    rms_arr = np.array(rms_pairs, dtype=np.float64) if rms_pairs else np.empty((0, 2))
    return {
        "inplane_rms":    rms_arr[:, 0] if rms_arr.shape[0] > 0 else np.array([]),
        "outplane_rms":   rms_arr[:, 1] if rms_arr.shape[0] > 0 else np.array([]),
        "kurtosis_in":    np.array(kurt_in,    dtype=np.float64),
        "kurtosis_out":   np.array(kurt_out,   dtype=np.float64),
        "dom_freq_in":    np.array(freq_in,    dtype=np.float64),
        "dom_freq_out":   np.array(freq_out,   dtype=np.float64),
        "dom_energy_in":  np.array(energy_in,  dtype=np.float64),
        "dom_energy_out": np.array(energy_out, dtype=np.float64),
        "cumsum_in":      cumsum_in,
        "cumsum_out":     cumsum_out,
    }


# ==================== enriched_stats 加载 ====================
def load_enriched_stats(enriched_dir: Path) -> dict:
    """
    从 enriched_stats 目录加载所有统计量（DL-VIV 数据源）。
    返回结构与 compute_signal_stats 相同。
    """
    if not enriched_dir.exists():
        raise FileNotFoundError(f"enriched_stats 目录不存在：{enriched_dir}")
    json_files = sorted(enriched_dir.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"目录下无 JSON 文件：{enriched_dir}")

    rms_pairs  = []
    kurt_in    = []
    kurt_out   = []
    freq_in    = []
    freq_out   = []
    energy_in  = []
    energy_out = []
    cumsum_in  = []
    cumsum_out = []

    for jf in json_files:
        with open(jf, "r", encoding="utf-8") as f:
            data = json.load(f)
        for sample in data["samples"]:
            ts_in   = sample.get("time_stats_inplane")   or {}
            ts_out  = sample.get("time_stats_outplane")  or {}
            psd_in  = sample.get("psd_inplane")          or {}
            psd_out = sample.get("psd_outplane")         or {}
            spec_in = sample.get("spectral_inplane")     or {}
            spec_out= sample.get("spectral_outplane")    or {}

            rms_i = ts_in.get("rms")
            rms_o = ts_out.get("rms")
            if rms_i is not None and rms_o is not None:
                rms_pairs.append((rms_i, rms_o))

            k_i = ts_in.get("kurtosis")
            k_o = ts_out.get("kurtosis")
            if k_i is not None:
                kurt_in.append(k_i)
            if k_o is not None:
                kurt_out.append(k_o)

            freqs_i  = psd_in.get("frequencies")
            powers_i = psd_in.get("powers")
            e_i      = spec_in.get("dominant_mode_energy_ratio")
            if freqs_i and powers_i and e_i is not None:
                dom_idx = int(np.argmax(powers_i))
                freq_in.append(freqs_i[dom_idx])
                energy_in.append(e_i)

            freqs_o  = psd_out.get("frequencies")
            powers_o = psd_out.get("powers")
            e_o      = spec_out.get("dominant_mode_energy_ratio")
            if freqs_o and powers_o and e_o is not None:
                dom_idx = int(np.argmax(powers_o))
                freq_out.append(freqs_o[dom_idx])
                energy_out.append(e_o)

            if powers_i and len(powers_i) > 0:
                arr = np.array(powers_i, dtype=np.float64)
                arr_sorted = np.sort(arr)[::-1]
                total = arr_sorted.sum()
                if total > 0:
                    cumsum_in.append(np.cumsum(arr_sorted) / total)

            if powers_o and len(powers_o) > 0:
                arr = np.array(powers_o, dtype=np.float64)
                arr_sorted = np.sort(arr)[::-1]
                total = arr_sorted.sum()
                if total > 0:
                    cumsum_out.append(np.cumsum(arr_sorted) / total)

    rms_arr = np.array(rms_pairs, dtype=np.float64) if rms_pairs else np.empty((0, 2))
    return {
        "inplane_rms":    rms_arr[:, 0] if rms_arr.shape[0] > 0 else np.array([]),
        "outplane_rms":   rms_arr[:, 1] if rms_arr.shape[0] > 0 else np.array([]),
        "kurtosis_in":    np.array(kurt_in,    dtype=np.float64),
        "kurtosis_out":   np.array(kurt_out,   dtype=np.float64),
        "dom_freq_in":    np.array(freq_in,    dtype=np.float64),
        "dom_freq_out":   np.array(freq_out,   dtype=np.float64),
        "dom_energy_in":  np.array(energy_in,  dtype=np.float64),
        "dom_energy_out": np.array(energy_out, dtype=np.float64),
        "cumsum_in":      cumsum_in,
        "cumsum_out":     cumsum_out,
    }


# ==================== 风数据对齐 ====================
def build_enriched_lookup(enriched_dir: Path) -> dict:
    """
    从 enriched_stats 构建 (sensor_id, window_idx) → {wind_stats, rms_in, rms_out} 查找表。
    用于为 MECC 样本匹配风数据（fig4_27/28）。
    """
    lookup: dict[tuple, dict] = {}
    for jf in sorted(enriched_dir.glob("*.json")):
        sensor_id = jf.stem
        with open(jf, "r", encoding="utf-8") as f:
            data = json.load(f)
        for sample in data["samples"]:
            wi     = sample.get("window_idx")
            ws     = sample.get("wind_stats") or []
            ts_in  = sample.get("time_stats_inplane")  or {}
            ts_out = sample.get("time_stats_outplane") or {}
            if wi is not None and ws:
                lookup[(sensor_id, wi)] = {
                    "wind_stats": ws,
                    "rms_in":    ts_in.get("rms"),
                    "rms_out":   ts_out.get("rms"),
                }
    return lookup


def load_mecc_wind_by_sensor(
    mecc_viv_samples: list,
    enriched_lookup:  dict,
    sensor_groups_map: dict,
) -> dict:
    """
    将 MECC-VIV 样本按传感器分组，并从 enriched_lookup 匹配风速 + RMS 数据。
    sensor_groups_map: {location_name: 'sensor_id.json'} 与 fig4_27 Config.SENSOR_GROUPS 相同。
    返回：{location_name: {"wind_speeds": ndarray, "rms_inplane": ndarray, "rms_outplane": ndarray}}
    """
    fname_to_name = {v.replace(".json", ""): k for k, v in sensor_groups_map.items()}

    accum: dict[str, dict[str, list]] = {}
    for s in mecc_viv_samples:
        sid = s.get("inplane_sensor_id", "")
        wi  = s.get("window_idx")
        info = enriched_lookup.get((sid, wi))
        if info is None:
            continue
        ws    = info["wind_stats"]
        rms_i = info["rms_in"]
        rms_o = info["rms_out"]
        if not ws or rms_i is None or rms_o is None:
            continue

        speeds = [w["mean_wind_speed"] for w in ws if w.get("mean_wind_speed") is not None]
        if not speeds:
            continue

        loc = fname_to_name.get(sid, sid)
        if loc not in accum:
            accum[loc] = {"wind_speeds": [], "rms_inplane": [], "rms_outplane": []}
        accum[loc]["wind_speeds"].append(float(np.mean(speeds)))
        accum[loc]["rms_inplane"].append(float(rms_i))
        accum[loc]["rms_outplane"].append(float(rms_o))

    return {
        loc: {k: np.array(v, dtype=np.float64) for k, v in d.items()}
        for loc, d in accum.items()
    }


def load_mecc_wind_dir_by_sensor(
    mecc_viv_samples: list,
    enriched_lookup:  dict,
    sensor_groups_map: dict,
) -> dict:
    """
    将 MECC-VIV 样本按传感器分组，匹配风向 + 紊流度 + RMS（fig4_28 用）。
    返回：{location_name: {"wind_dirs", "turb_intens", "rms_in", "rms_out"}}
    """
    fname_to_name = {v.replace(".json", ""): k for k, v in sensor_groups_map.items()}

    accum: dict[str, dict[str, list]] = {}
    for s in mecc_viv_samples:
        sid = s.get("inplane_sensor_id", "")
        wi  = s.get("window_idx")
        info = enriched_lookup.get((sid, wi))
        if info is None:
            continue
        ws    = info["wind_stats"]
        rms_i = info["rms_in"]
        rms_o = info["rms_out"]
        if not ws or rms_i is None or rms_o is None:
            continue

        d_list  = [w["mean_wind_direction"] for w in ws if w.get("mean_wind_direction") is not None]
        ti_list = [w["turbulence_intensity"] for w in ws if w.get("turbulence_intensity") is not None]
        if not d_list:
            continue

        corrected = (360.0 - float(np.mean(d_list))) % 360.0
        ti_mean   = float(np.nanmean(ti_list)) if ti_list else np.nan

        loc = fname_to_name.get(sid, sid)
        if loc not in accum:
            accum[loc] = {"wind_dirs": [], "turb_intens": [], "rms_in": [], "rms_out": []}
        accum[loc]["wind_dirs"].append(corrected)
        accum[loc]["turb_intens"].append(ti_mean)
        accum[loc]["rms_in"].append(float(rms_i))
        accum[loc]["rms_out"].append(float(rms_o))

    return {
        loc: {k: np.array(v, dtype=np.float64) for k, v in d.items()}
        for loc, d in accum.items()
    }
