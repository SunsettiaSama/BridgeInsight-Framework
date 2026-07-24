"""探索图：由单测点加速度反演轴向应力幅。

流程：实测加速度 → 频域积分位移 → 128 阶振型单点伪逆反演 → 动态轴力 → 雨流应力幅。
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal as scipy_signal
import yaml

project_root = Path(__file__).parent.parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.chapter3_identifier.identifier.physics.base_mode_calculator import (  # noqa: E402
    Cal_Mount,
    parse_mount_point_id,
)
from src.chapter4_characteristics._bootstrap import ensure_paths  # noqa: E402
from src.data_processer.io_unpacker import UNPACK  # noqa: E402

ensure_paths()

from src.figure_paintings.figs_for_thesis.Chapter4._viv_pipeline import (  # noqa: E402
    get_viv_samples as _pipeline_get_viv_samples,
    load_dl_result,
)
from src.figure_paintings.figs_for_thesis.config import (  # noqa: E402
    ABOVE_THRESHOLD_COLOR,
    CN_FONT,
    ENG_FONT,
    FONT_SIZE,
    VIV_INPLANE_COLOR,
)
from src.visualize_tools.web_dashboard import push as web_push  # noqa: E402


@dataclass(frozen=True)
class ModalResponse:
    order: int
    frequency_hz: float
    phi_at_sensor: float
    displacement_at_sensor_m: np.ndarray
    modal_coordinate_m: np.ndarray


@dataclass(frozen=True)
class StressProcessResult:
    sample: dict
    time_s: np.ndarray
    effective_time_s: np.ndarray
    acceleration_ms2: np.ndarray
    displacement_m: np.ndarray
    psd_freq_hz: np.ndarray
    psd_acc: np.ndarray
    modes: list[ModalResponse]
    axial_stress_mpa: np.ndarray
    stress_amplitudes_mpa: np.ndarray
    stress_counts: np.ndarray
    sensor_position_ratio: float
    integration_lowcut_hz: float
    static_stress_mpa: float
    dynamic_stress_mean_mpa: float


@dataclass(frozen=True)
class FrequencyStressResult:
    sample: dict
    time_s: np.ndarray
    effective_time_s: np.ndarray
    frequency_time_s: np.ndarray
    base_frequency_hz: np.ndarray
    dominant_peak_frequency_hz: np.ndarray
    ridge_stress_mpa: np.ndarray
    stress_amplitudes_mpa: np.ndarray
    stress_counts: np.ndarray
    psd_freq_hz: np.ndarray
    psd_acc: np.ndarray
    spectrogram_freq_hz: np.ndarray
    spectrogram_time_s: np.ndarray
    spectrogram_power: np.ndarray
    dominant_frequency_hz: float
    static_base_frequency_hz: float
    static_stress_mpa: float
    dynamic_stress_mean_mpa: float
    sensor_position_ratio: float


@dataclass(frozen=True)
class ModalFlowResult:
    sample: dict
    modal_time_s: np.ndarray
    modal_frequency_hz: np.ndarray
    modal_order: np.ndarray
    modal_power: np.ndarray
    spectrogram_freq_hz: np.ndarray
    spectrogram_time_s: np.ndarray
    spectrogram_power: np.ndarray
    static_base_frequency_hz: float
    dominant_frequency_hz: float


@dataclass(frozen=True)
class ModalFlowTensionResult:
    sample: dict
    aggregation_window_second: float
    time_s: np.ndarray
    base_frequency_hz: np.ndarray
    stress_mpa: np.ndarray
    stress_amplitudes_mpa: np.ndarray
    stress_counts: np.ndarray
    valid_point_count: np.ndarray
    static_base_frequency_hz: float
    static_stress_mpa: float
    dynamic_stress_mean_mpa: float


class Config:
    FS = 50.0
    WINDOW_SIZE = 3000
    ANALYSIS_DURATION_SECOND = 3000.0
    EDGE_TRIM_SECOND = 300.0
    RANDOM_SEED = 17

    PREFERRED_CABLE_IDS = ("C18-101", "C18-102", "C34-101", "C34-102", "C34-302")
    CANDIDATE_SCAN_LIMIT = 30
    NORMAL_CLASS_ID = 0

    SENSOR_POSITION_RATIO_BY_CODE = {
        "1": 0.25,
        "2": 0.50,
        "3": 0.75,
    }
    DEFAULT_SENSOR_POSITION_RATIO = 0.50

    LOWCUT_FIRST_MODE_FACTOR = 1.0
    PSD_NPERSEG = 8192
    N_MODES = 128
    PSD_MODE_LABEL_MAX_HZ = 10.0
    PSD_MODE_LABEL_STEP = 4

    TF_WINDOW_SECOND = 60.0
    TF_STEP_SAMPLE = 1
    TF_NFFT = 2048
    TF_WELCH_NPERSEG = 2048
    TF_FREQ_MIN_HZ = 0.2
    TF_FREQ_MAX_HZ = 10.0
    TF_PEAK_PROMINENCE_RATIO = 0.025
    TF_PEAK_DISTANCE_HZ = 0.12
    TF_GAP_MIN_RATIO = 0.45
    TF_GAP_MAX_RATIO = 1.75
    TF_NEARBY_GAP_COUNT = 10
    TF_SPECTROGRAM_PLOT_STRIDE = 250
    MODAL_FLOW_STEP_SAMPLE = 1
    MODAL_FLOW_SPECTROGRAM_PLOT_STRIDE = 250
    MODAL_FLOW_FREQ_TOLERANCE_RATIO = 0.16
    MODAL_FLOW_MIN_ORDER = 1
    MODAL_FLOW_MAX_ORDER = 40
    MODAL_FLOW_MAX_PEAKS_PER_WINDOW = 18
    MODAL_FLOW_TENSION_WINDOW_SECONDS = (0.02, 1.0)

    FIG_SIZE = (16, 12)
    LINE_WIDTH = 1.15
    GRID_COLOR = "gray"
    GRID_ALPHA = 0.25
    GRID_LINESTYLE = "--"
    ACC_COLOR = "#4D4D4D"
    DISP_COLOR = VIV_INPLANE_COLOR
    STRESS_COLOR = ABOVE_THRESHOLD_COLOR
    MODE_COLOR = "#992224"
    HIST_COLOR = "#8074C8"
    WEB_PAGE = "fig4_x 应力幅获取流程"
    SNAPSHOT_PATH = (
        project_root
        / "results"
        / "chapter4_characteristics"
        / "figure_snapshots"
        / "fig4_x_stress_amplitude_process.png"
    )
    FREQUENCY_SNAPSHOT_PATH = (
        project_root
        / "results"
        / "chapter4_characteristics"
        / "figure_snapshots"
        / "fig4_x_stress_amplitude_frequency_route.png"
    )
    NORMAL_FREQUENCY_SNAPSHOT_PATH = (
        project_root
        / "results"
        / "chapter4_characteristics"
        / "figure_snapshots"
        / "fig4_x_stress_amplitude_normal_frequency_route.png"
    )
    MODAL_FLOW_SNAPSHOT_PATH = (
        project_root
        / "results"
        / "chapter4_characteristics"
        / "figure_snapshots"
        / "fig4_x_stress_amplitude_modal_flow.png"
    )
    MODAL_FLOW_TENSION_SNAPSHOT_PATH = (
        project_root
        / "results"
        / "chapter4_characteristics"
        / "figure_snapshots"
        / "fig4_x_stress_amplitude_modal_flow_tension.png"
    )


def _load_raw(file_path: str, unpacker: UNPACK) -> np.ndarray:
    return np.asarray(unpacker.VIC_DATA_Unpack(file_path), dtype=np.float64)


def _load_window(file_path: str, window_idx: int, unpacker: UNPACK) -> np.ndarray:
    raw = np.asarray(unpacker.VIC_DATA_Unpack(file_path), dtype=np.float64)
    start = window_idx * Config.WINDOW_SIZE
    end = start + Config.WINDOW_SIZE
    if end > len(raw):
        raise ValueError(f"窗口越界：window_idx={window_idx}, data_len={len(raw)}")
    return raw[start:end]


def _analysis_bounds(window_idx: int) -> tuple[int, int]:
    start = window_idx * Config.WINDOW_SIZE
    length = int(round(Config.ANALYSIS_DURATION_SECOND * Config.FS))
    return start, start + length


def _load_analysis_segment(file_path: str, window_idx: int, unpacker: UNPACK) -> np.ndarray:
    raw = _load_raw(file_path, unpacker)
    start, end = _analysis_bounds(window_idx)
    if end > len(raw):
        raise ValueError(
            f"3000s 分析段越界：window_idx={window_idx}, "
            f"need_end={end}, data_len={len(raw)}"
        )
    return raw[start:end]


def _cable_id_from_sensor(sensor_id: str) -> str:
    mount_point = parse_mount_point_id(sensor_id)
    if mount_point is None:
        raise ValueError(f"无法从传感器编号解析测点：{sensor_id}")
    return mount_point


def _sensor_position_ratio(sensor_id: str) -> float:
    cable_id = _cable_id_from_sensor(sensor_id)
    code = cable_id.split("-")[1][0]
    return Config.SENSOR_POSITION_RATIO_BY_CODE.get(
        code,
        Config.DEFAULT_SENSOR_POSITION_RATIO,
    )


def _load_mount(sensor_id: str) -> Cal_Mount:
    cable_id = _cable_id_from_sensor(sensor_id)
    params_path = project_root / "config" / "cables" / "cable_params.yaml"
    if not params_path.exists():
        raise FileNotFoundError(f"拉索参数配置不存在：{params_path}")

    with open(params_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}

    params = (config.get("cables") or {}).get(cable_id)
    if params is None:
        raise KeyError(f"拉索参数配置中缺少测点：{cable_id}")

    missing = [
        key
        for key in ("length", "force", "alphac_deg", "E", "A", "m")
        if params.get(key) in (None, "???")
    ]
    if missing:
        raise ValueError(f"测点 {cable_id} 缺少物理参数：{missing}")

    return Cal_Mount(
        length=float(params["length"]),
        force=float(params["force"]),
        alphac=float(params["alphac_deg"]) * np.pi / 180.0,
        A=float(params["A"]),
        m=float(params["m"]),
        E=float(params["E"]),
    )


def _select_sample(samples: list[dict], unpacker: UNPACK) -> dict:
    preferred = []
    for sample in samples:
        cable_id = _cable_id_from_sensor(sample["inplane_sensor_id"])
        if cable_id in Config.PREFERRED_CABLE_IDS:
            preferred.append(sample)

    candidates = preferred if preferred else samples
    if not candidates:
        raise ValueError("没有可用于应力反演探索的 VIV 样本")

    scan_count = min(Config.CANDIDATE_SCAN_LIMIT, len(candidates))
    rng = np.random.default_rng(Config.RANDOM_SEED)
    scan_indices = np.sort(rng.choice(len(candidates), size=scan_count, replace=False))

    best_sample: dict | None = None
    best_rms = -np.inf
    for idx in scan_indices:
        sample = candidates[int(idx)]
        raw = _load_raw(sample["inplane_file_path"], unpacker)
        start, end = _analysis_bounds(sample["window_idx"])
        if end > len(raw):
            continue
        acc = raw[start:start + Config.WINDOW_SIZE]
        rms = float(np.sqrt(np.mean((acc - np.mean(acc)) ** 2)))
        if rms > best_rms:
            best_sample = sample
            best_rms = rms

    if best_sample is None:
        raise ValueError("扫描候选中没有足够长度的 3000s 连续实测数据")

    print(
        f"  选中样本：{best_sample['inplane_sensor_id']} "
        f"win={best_sample['window_idx']}，扫描 RMS 最大值={best_rms:.4g}"
    )
    return best_sample


def _get_samples_by_class(result: dict, class_id: int) -> list[dict]:
    predictions = {int(k): int(v) for k, v in result["predictions"].items()}
    sample_metadata = result.get("sample_metadata", {})

    samples = []
    for idx, pred_label in predictions.items():
        if pred_label != class_id:
            continue
        meta = sample_metadata.get(str(idx))
        if meta is None:
            continue
        in_path = meta.get("inplane_file_path")
        out_path = meta.get("outplane_file_path")
        if not in_path or not out_path:
            continue
        samples.append(
            {
                "idx": idx,
                "window_idx": meta["window_idx"],
                "inplane_sensor_id": meta.get("inplane_sensor_id", ""),
                "outplane_sensor_id": meta.get("outplane_sensor_id", ""),
                "inplane_file_path": in_path,
                "outplane_file_path": out_path,
                "timestamp": meta.get("timestamp", []),
            }
        )
    return samples


def _frequency_domain_integrate(
    acceleration: np.ndarray,
    fs: float,
    lowcut_hz: float,
) -> np.ndarray:
    acc = acceleration - np.mean(acceleration)
    spectrum = np.fft.rfft(acc)
    freqs = np.fft.rfftfreq(len(acc), d=1.0 / fs)
    omega = 2.0 * np.pi * freqs

    disp_spectrum = np.zeros_like(spectrum, dtype=np.complex128)
    mask = freqs >= lowcut_hz
    disp_spectrum[mask] = -spectrum[mask] / (omega[mask] ** 2)
    displacement = np.fft.irfft(disp_spectrum, n=len(acc))
    return displacement - np.mean(displacement)


def _extract_modal_responses(
    displacement_at_sensor: np.ndarray,
    mount: Cal_Mount,
    sensor_position_ratio: float,
) -> list[ModalResponse]:
    orders = np.arange(1, Config.N_MODES + 1, dtype=np.float64)
    phi_values = np.sin(orders * np.pi * sensor_position_ratio)
    phi_norm_square = float(np.dot(phi_values, phi_values))
    if phi_norm_square <= 0.0:
        raise ValueError("测点在 128 阶振型基底中全部落在节点，无法反演")

    weights = phi_values / phi_norm_square
    responses: list[ModalResponse] = []
    for idx, order_value in enumerate(orders):
        order = int(order_value)
        frequency_hz = float(mount.inplane_mode(order))
        phi = float(phi_values[idx])
        modal_coordinate = weights[idx] * displacement_at_sensor
        responses.append(
            ModalResponse(
                order=order,
                frequency_hz=frequency_hz,
                phi_at_sensor=phi,
                displacement_at_sensor_m=phi * modal_coordinate,
                modal_coordinate_m=modal_coordinate,
            )
        )
    print(
        f"  128 阶振型伪逆：sum(phi_n^2)={phi_norm_square:.3f}，"
        "未做逐模态带通"
    )
    return responses


def _axial_stress_mpa(modes: list[ModalResponse], mount: Cal_Mount) -> np.ndarray:
    modal_sum = np.zeros_like(modes[0].modal_coordinate_m, dtype=np.float64)
    for response in modes:
        modal_sum += (response.order ** 2) * (response.modal_coordinate_m ** 2)

    dynamic_force = (
        mount.E
        * mount.A
        * (np.pi ** 2)
        / (4.0 * (mount.length ** 2))
        * modal_sum
    )
    return (mount.force + dynamic_force) / mount.A / 1e6


def _turning_points(series: np.ndarray) -> np.ndarray:
    values = np.asarray(series, dtype=np.float64)
    if len(values) < 3:
        return values

    keep = np.ones(len(values), dtype=bool)
    keep[1:] = np.diff(values) != 0.0
    values = values[keep]
    if len(values) < 3:
        return values

    turning = [values[0]]
    for idx in range(1, len(values) - 1):
        prev_value = values[idx - 1]
        this_value = values[idx]
        next_value = values[idx + 1]
        if (this_value - prev_value) * (next_value - this_value) <= 0.0:
            turning.append(this_value)
    turning.append(values[-1])
    return np.asarray(turning, dtype=np.float64)


def _rainflow_amplitudes(series: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    reversals = _turning_points(series)
    if len(reversals) < 2:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)

    stack: list[float] = []
    ranges: list[float] = []
    counts: list[float] = []
    for value in reversals:
        stack.append(float(value))
        while len(stack) >= 3:
            previous_range = abs(stack[-2] - stack[-3])
            latest_range = abs(stack[-1] - stack[-2])
            if latest_range < previous_range:
                break
            if len(stack) == 3:
                ranges.append(previous_range)
                counts.append(0.5)
                stack.pop(0)
            else:
                ranges.append(previous_range)
                counts.append(1.0)
                del stack[-3:-1]

    for idx in range(len(stack) - 1):
        ranges.append(abs(stack[idx + 1] - stack[idx]))
        counts.append(0.5)

    amplitudes = np.asarray(ranges, dtype=np.float64) / 2.0
    cycle_counts = np.asarray(counts, dtype=np.float64)
    mask = amplitudes > 0.0
    return amplitudes[mask], cycle_counts[mask]


def compute_stress_process(sample: dict, unpacker: UNPACK) -> StressProcessResult:
    sensor_id = sample["inplane_sensor_id"]
    mount = _load_mount(sensor_id)
    sensor_position_ratio = _sensor_position_ratio(sensor_id)
    acceleration = _load_analysis_segment(
        sample["inplane_file_path"],
        sample["window_idx"],
        unpacker,
    )
    acceleration = acceleration - np.mean(acceleration)
    first_mode_hz = float(mount.inplane_mode(1))
    integration_lowcut_hz = first_mode_hz * Config.LOWCUT_FIRST_MODE_FACTOR
    displacement = _frequency_domain_integrate(
        acceleration,
        Config.FS,
        integration_lowcut_hz,
    )

    freq_hz, psd_acc = scipy_signal.welch(
        acceleration,
        fs=Config.FS,
        nperseg=Config.PSD_NPERSEG,
        noverlap=Config.PSD_NPERSEG // 2,
    )
    modes = _extract_modal_responses(displacement, mount, sensor_position_ratio)
    axial_stress_mpa = _axial_stress_mpa(modes, mount)
    trim_n = int(round(Config.EDGE_TRIM_SECOND * Config.FS))
    if trim_n * 2 >= len(axial_stress_mpa):
        raise ValueError("首尾裁剪长度超过分析段长度")

    effective_stress_mpa = axial_stress_mpa[trim_n:-trim_n]
    amplitudes, counts = _rainflow_amplitudes(effective_stress_mpa)

    static_stress_mpa = mount.force / mount.A / 1e6
    dynamic_stress_mean_mpa = float(np.mean(effective_stress_mpa - static_stress_mpa))
    time_s = np.arange(len(acceleration), dtype=np.float64) / Config.FS
    effective_time_s = time_s[trim_n:-trim_n]

    return StressProcessResult(
        sample=sample,
        time_s=time_s,
        effective_time_s=effective_time_s,
        acceleration_ms2=acceleration,
        displacement_m=displacement,
        psd_freq_hz=freq_hz,
        psd_acc=psd_acc,
        modes=modes,
        axial_stress_mpa=axial_stress_mpa,
        stress_amplitudes_mpa=amplitudes,
        stress_counts=counts,
        sensor_position_ratio=sensor_position_ratio,
        integration_lowcut_hz=integration_lowcut_hz,
        static_stress_mpa=static_stress_mpa,
        dynamic_stress_mean_mpa=dynamic_stress_mean_mpa,
    )


def _dominant_frequency_in_band(freq_hz: np.ndarray, power: np.ndarray) -> float:
    mask = (freq_hz >= Config.TF_FREQ_MIN_HZ) & (freq_hz <= Config.TF_FREQ_MAX_HZ)
    if not np.any(mask):
        raise ValueError("指定频带内没有可用 PSD 频点")
    band_freq = freq_hz[mask]
    band_power = power[mask]
    peak_idx = int(np.argmax(band_power))
    return float(band_freq[peak_idx])


def _nearest_taut_string_mode(mount: Cal_Mount, frequency_hz: float) -> int:
    orders = np.arange(1, Config.N_MODES + 1, dtype=np.float64)
    theoretical = orders / (2.0 * mount.length) * np.sqrt(mount.force / mount.m)
    idx = int(np.argmin(np.abs(theoretical - frequency_hz)))
    return int(orders[idx])


def _static_base_frequency(mount: Cal_Mount) -> float:
    return 1.0 / (2.0 * mount.length) * np.sqrt(mount.force / mount.m)


def _parabolic_peak_frequency(freq_hz: np.ndarray, power: np.ndarray, peak_idx: int) -> float:
    if peak_idx <= 0 or peak_idx >= len(power) - 1:
        return float(freq_hz[peak_idx])

    y0 = float(power[peak_idx - 1])
    y1 = float(power[peak_idx])
    y2 = float(power[peak_idx + 1])
    denom = y0 - 2.0 * y1 + y2
    if denom == 0.0:
        return float(freq_hz[peak_idx])

    delta = 0.5 * (y0 - y2) / denom
    df = float(freq_hz[1] - freq_hz[0])
    return float(freq_hz[peak_idx] + delta * df)


def _estimate_base_frequency_from_gaps(
    freq_hz: np.ndarray,
    power: np.ndarray,
    expected_base_hz: float,
) -> tuple[float, float]:
    mask = (freq_hz >= Config.TF_FREQ_MIN_HZ) & (freq_hz <= Config.TF_FREQ_MAX_HZ)
    band_freq = freq_hz[mask]
    band_power = power[mask]
    if len(band_freq) < 3:
        raise ValueError("频谱频点不足，无法估计基频")

    distance_bins = max(1, int(round(Config.TF_PEAK_DISTANCE_HZ / (band_freq[1] - band_freq[0]))))
    peaks, _ = scipy_signal.find_peaks(
        band_power,
        distance=distance_bins,
        prominence=float(np.max(band_power)) * Config.TF_PEAK_PROMINENCE_RATIO,
    )
    if len(peaks) < 2:
        main_idx = int(np.argmax(band_power))
        return expected_base_hz, float(band_freq[main_idx])

    main_peak_idx = int(peaks[np.argmax(band_power[peaks])])
    peak_order = int(np.where(peaks == main_peak_idx)[0][0])
    gap_count = min(Config.TF_NEARBY_GAP_COUNT, len(peaks) - 1)
    left_gap_count = min(gap_count // 2, peak_order)
    right_gap_count = min(gap_count - left_gap_count, len(peaks) - 1 - peak_order)
    left_gap_count = min(gap_count - right_gap_count, peak_order)

    selected_start = peak_order - left_gap_count
    selected_end = peak_order + right_gap_count + 1
    selected = peaks[selected_start:selected_end].tolist()
    peak_freqs = np.array(
        [
            _parabolic_peak_frequency(band_freq, band_power, idx)
            for idx in selected
        ],
        dtype=np.float64,
    )
    peak_powers = np.array([band_power[idx] for idx in selected], dtype=np.float64)

    main_local = int(np.argmax(peak_powers))
    main_freq = float(peak_freqs[main_local])

    gaps_arr = np.diff(peak_freqs)
    valid = (
        (gaps_arr >= expected_base_hz * Config.TF_GAP_MIN_RATIO)
        & (gaps_arr <= expected_base_hz * Config.TF_GAP_MAX_RATIO)
    )
    if not np.any(valid):
        return expected_base_hz, main_freq
    return float(np.mean(gaps_arr[valid])), main_freq


def compute_frequency_stress_process(
    sample: dict,
    unpacker: UNPACK,
) -> FrequencyStressResult:
    sensor_id = sample["inplane_sensor_id"]
    mount = _load_mount(sensor_id)
    sensor_position_ratio = _sensor_position_ratio(sensor_id)
    acceleration = _load_analysis_segment(
        sample["inplane_file_path"],
        sample["window_idx"],
        unpacker,
    )
    acceleration = acceleration - np.mean(acceleration)

    freq_hz, psd_acc = scipy_signal.welch(
        acceleration,
        fs=Config.FS,
        nperseg=Config.PSD_NPERSEG,
        noverlap=Config.PSD_NPERSEG // 2,
    )
    dominant_freq_hz = _dominant_frequency_in_band(freq_hz, psd_acc)
    mode_order = _nearest_taut_string_mode(mount, dominant_freq_hz)
    static_base_hz = _static_base_frequency(mount)

    window_n = int(round(Config.TF_WINDOW_SECOND * Config.FS))
    step = Config.TF_STEP_SAMPLE
    starts = np.arange(0, len(acceleration) - window_n + 1, step, dtype=np.int64)
    spec_time_s = (starts + window_n / 2.0) / Config.FS
    spec_freq_hz: np.ndarray | None = None
    spec_columns: list[np.ndarray] = []
    spec_plot_times: list[float] = []
    base_frequency_hz: list[float] = []
    dominant_peak_frequency_hz: list[float] = []

    for col_idx, start in enumerate(starts):
        segment = acceleration[start:start + window_n]
        seg_freq_hz, seg_power = scipy_signal.welch(
            segment,
            fs=Config.FS,
            window="hann",
            nperseg=Config.TF_WELCH_NPERSEG,
            noverlap=Config.TF_WELCH_NPERSEG // 2,
            nfft=Config.TF_NFFT,
            detrend="constant",
            scaling="density",
        )
        if spec_freq_hz is None:
            spec_freq_hz = seg_freq_hz
        f1_hz, main_peak_hz = _estimate_base_frequency_from_gaps(
            seg_freq_hz,
            seg_power,
            static_base_hz,
        )
        if col_idx % Config.TF_SPECTROGRAM_PLOT_STRIDE == 0:
            spec_columns.append(seg_power)
            spec_plot_times.append(float(spec_time_s[col_idx]))
        base_frequency_hz.append(f1_hz)
        dominant_peak_frequency_hz.append(main_peak_hz)

    if spec_freq_hz is None:
        raise ValueError("没有可用的滑动窗口频谱")

    spec_power = np.stack(spec_columns, axis=1)
    spec_plot_time_s = np.asarray(spec_plot_times, dtype=np.float64)
    base_frequency_arr = np.asarray(base_frequency_hz, dtype=np.float64)
    dominant_peak_arr = np.asarray(dominant_peak_frequency_hz, dtype=np.float64)

    tension_n = 4.0 * mount.m * (mount.length ** 2) * (base_frequency_arr ** 2)
    ridge_stress_mpa = tension_n / mount.A / 1e6

    effective_mask = (
        (spec_time_s >= Config.EDGE_TRIM_SECOND)
        & (spec_time_s <= Config.ANALYSIS_DURATION_SECOND - Config.EDGE_TRIM_SECOND)
    )
    if not np.any(effective_mask):
        raise ValueError("时频脊线没有落在首尾裁剪后的有效时间段内")
    effective_stress_mpa = ridge_stress_mpa[effective_mask]
    amplitudes, counts = _rainflow_amplitudes(effective_stress_mpa)

    static_stress_mpa = mount.force / mount.A / 1e6
    dynamic_stress_mean_mpa = float(np.mean(effective_stress_mpa - static_stress_mpa))
    time_s = np.arange(len(acceleration), dtype=np.float64) / Config.FS

    print(
        f"  张紧弦频率路线：整段主峰={dominant_freq_hz:.3f} Hz，"
        f"主峰附近 {Config.TF_NEARBY_GAP_COUNT} 个 gap 平均估计基频，"
        f"静态 f1={static_base_hz:.3f} Hz，"
        f"60s 窗 / {Config.TF_STEP_SAMPLE} 点步长 / nfft={Config.TF_NFFT}"
    )

    return FrequencyStressResult(
        sample=sample,
        time_s=time_s,
        effective_time_s=spec_time_s[effective_mask],
        frequency_time_s=spec_time_s,
        base_frequency_hz=base_frequency_arr,
        dominant_peak_frequency_hz=dominant_peak_arr,
        ridge_stress_mpa=ridge_stress_mpa,
        stress_amplitudes_mpa=amplitudes,
        stress_counts=counts,
        psd_freq_hz=freq_hz,
        psd_acc=psd_acc,
        spectrogram_freq_hz=spec_freq_hz,
        spectrogram_time_s=spec_plot_time_s,
        spectrogram_power=spec_power,
        dominant_frequency_hz=dominant_freq_hz,
        static_base_frequency_hz=static_base_hz,
        static_stress_mpa=static_stress_mpa,
        dynamic_stress_mean_mpa=dynamic_stress_mean_mpa,
        sensor_position_ratio=sensor_position_ratio,
    )


def compute_modal_flow(sample: dict, unpacker: UNPACK) -> ModalFlowResult:
    sensor_id = sample["inplane_sensor_id"]
    mount = _load_mount(sensor_id)
    acceleration = _load_analysis_segment(
        sample["inplane_file_path"],
        sample["window_idx"],
        unpacker,
    )
    acceleration = acceleration - np.mean(acceleration)

    freq_hz, psd_acc = scipy_signal.welch(
        acceleration,
        fs=Config.FS,
        nperseg=Config.PSD_NPERSEG,
        noverlap=Config.PSD_NPERSEG // 2,
    )
    dominant_freq_hz = _dominant_frequency_in_band(freq_hz, psd_acc)
    static_base_hz = _static_base_frequency(mount)

    window_n = int(round(Config.TF_WINDOW_SECOND * Config.FS))
    starts = np.arange(
        0,
        len(acceleration) - window_n + 1,
        Config.MODAL_FLOW_STEP_SAMPLE,
        dtype=np.int64,
    )
    spec_time_s = (starts + window_n / 2.0) / Config.FS
    spec_freq_hz: np.ndarray | None = None
    spec_columns: list[np.ndarray] = []
    spec_plot_times: list[float] = []
    modal_time: list[float] = []
    modal_freq: list[float] = []
    modal_order: list[int] = []
    modal_power: list[float] = []

    for col_idx, start in enumerate(starts):
        segment = acceleration[start:start + window_n]
        seg_freq_hz, seg_power = scipy_signal.welch(
            segment,
            fs=Config.FS,
            window="hann",
            nperseg=Config.TF_WELCH_NPERSEG,
            noverlap=Config.TF_WELCH_NPERSEG // 2,
            nfft=Config.TF_NFFT,
            detrend="constant",
            scaling="density",
        )
        if spec_freq_hz is None:
            spec_freq_hz = seg_freq_hz
        if col_idx % Config.MODAL_FLOW_SPECTROGRAM_PLOT_STRIDE == 0:
            spec_columns.append(seg_power)
            spec_plot_times.append(float(spec_time_s[col_idx]))

        band_mask = (
            (seg_freq_hz >= Config.TF_FREQ_MIN_HZ)
            & (seg_freq_hz <= Config.TF_FREQ_MAX_HZ)
        )
        band_freq = seg_freq_hz[band_mask]
        band_power = seg_power[band_mask]
        if len(band_freq) < 3:
            continue

        distance_bins = max(
            1,
            int(round(Config.TF_PEAK_DISTANCE_HZ / (band_freq[1] - band_freq[0]))),
        )
        peaks, _ = scipy_signal.find_peaks(
            band_power,
            distance=distance_bins,
            prominence=float(np.max(band_power)) * Config.TF_PEAK_PROMINENCE_RATIO,
        )
        if len(peaks) == 0:
            continue

        ranked_peaks = sorted(
            peaks.tolist(),
            key=lambda idx: float(band_power[idx]),
            reverse=True,
        )[: Config.MODAL_FLOW_MAX_PEAKS_PER_WINDOW]
        time_value = float(spec_time_s[col_idx])
        for peak_idx in ranked_peaks:
            peak_freq = _parabolic_peak_frequency(band_freq, band_power, peak_idx)
            order = int(round(peak_freq / static_base_hz))
            if order < Config.MODAL_FLOW_MIN_ORDER or order > Config.MODAL_FLOW_MAX_ORDER:
                continue
            expected_freq = order * static_base_hz
            if abs(peak_freq - expected_freq) > Config.MODAL_FLOW_FREQ_TOLERANCE_RATIO * static_base_hz:
                continue

            modal_time.append(time_value)
            modal_freq.append(peak_freq)
            modal_order.append(order)
            modal_power.append(float(band_power[peak_idx]))

    if spec_freq_hz is None:
        raise ValueError("没有可用于模态流识别的滑动窗口频谱")

    print(
        f"  模态流识别：识别点={len(modal_freq)}，"
        f"阶次范围={Config.MODAL_FLOW_MIN_ORDER}-{Config.MODAL_FLOW_MAX_ORDER}，"
        f"步长={Config.MODAL_FLOW_STEP_SAMPLE} 点"
    )

    return ModalFlowResult(
        sample=sample,
        modal_time_s=np.asarray(modal_time, dtype=np.float64),
        modal_frequency_hz=np.asarray(modal_freq, dtype=np.float64),
        modal_order=np.asarray(modal_order, dtype=np.int64),
        modal_power=np.asarray(modal_power, dtype=np.float64),
        spectrogram_freq_hz=spec_freq_hz,
        spectrogram_time_s=np.asarray(spec_plot_times, dtype=np.float64),
        spectrogram_power=np.stack(spec_columns, axis=1),
        static_base_frequency_hz=static_base_hz,
        dominant_frequency_hz=dominant_freq_hz,
    )


def compute_modal_flow_tension(
    result: ModalFlowResult,
    aggregation_window_second: float,
) -> ModalFlowTensionResult:
    sensor_id = result.sample["inplane_sensor_id"]
    mount = _load_mount(sensor_id)
    window_second = aggregation_window_second
    centers = np.arange(
        window_second / 2.0,
        Config.ANALYSIS_DURATION_SECOND,
        window_second,
        dtype=np.float64,
    )
    half_window = window_second / 2.0

    base_frequency = np.full(len(centers), np.nan, dtype=np.float64)
    valid_count = np.zeros(len(centers), dtype=np.int64)
    candidate_f1 = result.modal_frequency_hz / result.modal_order
    weights = np.maximum(result.modal_power, 0.0)
    order = np.argsort(result.modal_time_s)
    modal_time_s = result.modal_time_s[order]
    candidate_f1 = candidate_f1[order]
    weights = weights[order]

    for idx, center in enumerate(centers):
        left = np.searchsorted(modal_time_s, center - half_window, side="left")
        right = np.searchsorted(modal_time_s, center + half_window, side="left")
        if right <= left:
            continue

        f1_values = candidate_f1[left:right]
        point_weights = weights[left:right]
        valid = np.isfinite(f1_values) & (point_weights > 0.0)
        if not np.any(valid):
            continue

        base_frequency[idx] = float(np.average(f1_values[valid], weights=point_weights[valid]))
        valid_count[idx] = int(np.sum(valid))

    valid_freq = np.isfinite(base_frequency)
    if not np.any(valid_freq):
        raise ValueError("模态流 1s 聚合未得到任何有效基频")

    base_frequency = np.interp(
        centers,
        centers[valid_freq],
        base_frequency[valid_freq],
    )
    stress_mpa = 4.0 * mount.m * (mount.length ** 2) * (base_frequency ** 2) / mount.A / 1e6

    effective_mask = (
        (centers >= Config.EDGE_TRIM_SECOND)
        & (centers <= Config.ANALYSIS_DURATION_SECOND - Config.EDGE_TRIM_SECOND)
    )
    effective_stress = stress_mpa[effective_mask]
    amplitudes, counts = _rainflow_amplitudes(effective_stress)

    static_stress_mpa = mount.force / mount.A / 1e6
    dynamic_stress_mean_mpa = float(np.mean(effective_stress - static_stress_mpa))
    print(
        f"  模态流索力：{aggregation_window_second:g}s 聚合点={len(centers)}，"
        f"有效原始秒={int(np.sum(valid_freq))}，"
        f"动态应力均值={dynamic_stress_mean_mpa:.6f} MPa"
    )

    return ModalFlowTensionResult(
        sample=result.sample,
        aggregation_window_second=aggregation_window_second,
        time_s=centers,
        base_frequency_hz=base_frequency,
        stress_mpa=stress_mpa,
        stress_amplitudes_mpa=amplitudes,
        stress_counts=counts,
        valid_point_count=valid_count,
        static_base_frequency_hz=result.static_base_frequency_hz,
        static_stress_mpa=static_stress_mpa,
        dynamic_stress_mean_mpa=dynamic_stress_mean_mpa,
    )


def _format_sample_label(result: StressProcessResult) -> str:
    sample = result.sample
    sensor_id = sample["inplane_sensor_id"]
    timestamp = sample.get("timestamp", [])
    time_str = "-".join(str(x) for x in timestamp) if timestamp else "unknown time"
    cable_id = _cable_id_from_sensor(sensor_id)
    visible_modes = [
        item
        for item in result.modes
        if item.frequency_hz <= Config.PSD_MODE_LABEL_MAX_HZ
    ]
    return (
        f"{cable_id} 面内测点 x/L≈{result.sensor_position_ratio:.2f}；"
        f"{time_str} win={sample['window_idx']}；"
        f"128 阶振型伪逆，PSD 中显示前 {len(visible_modes)} 阶"
    )


def _style_axis(ax: plt.Axes) -> None:
    ax.grid(
        True,
        color=Config.GRID_COLOR,
        alpha=Config.GRID_ALPHA,
        linestyle=Config.GRID_LINESTYLE,
        linewidth=0.7,
    )
    ax.tick_params(axis="both", labelsize=FONT_SIZE - 8)


def _shade_trimmed_edges(ax: plt.Axes, result: StressProcessResult) -> None:
    left_end = Config.EDGE_TRIM_SECOND
    right_start = Config.ANALYSIS_DURATION_SECOND - Config.EDGE_TRIM_SECOND
    ax.axvspan(0.0, left_end, color="#D0D0D0", alpha=0.28, linewidth=0)
    ax.axvspan(
        right_start,
        Config.ANALYSIS_DURATION_SECOND,
        color="#D0D0D0",
        alpha=0.28,
        linewidth=0,
    )
    ax.axvline(result.effective_time_s[0], color="#808080", linewidth=0.8, linestyle="--")
    ax.axvline(result.effective_time_s[-1], color="#808080", linewidth=0.8, linestyle="--")


def plot_stress_process(result: StressProcessResult) -> plt.Figure:
    fig = plt.figure(figsize=Config.FIG_SIZE)
    gs = fig.add_gridspec(3, 2, height_ratios=[1.0, 1.0, 1.1])
    ax_acc = fig.add_subplot(gs[0, 0])
    ax_disp = fig.add_subplot(gs[0, 1])
    ax_psd = fig.add_subplot(gs[1, :])
    ax_stress = fig.add_subplot(gs[2, 0])
    ax_hist = fig.add_subplot(gs[2, 1])

    ax_acc.plot(
        result.time_s,
        result.acceleration_ms2,
        color=Config.ACC_COLOR,
        linewidth=Config.LINE_WIDTH,
    )
    ax_acc.set_title("1 加速度实测窗口", fontproperties=CN_FONT, fontsize=FONT_SIZE - 5)
    ax_acc.set_xlabel("时间 (s)", fontproperties=CN_FONT, fontsize=FONT_SIZE - 7)
    ax_acc.set_ylabel(r"加速度 ($m/s^2$)", fontproperties=CN_FONT, fontsize=FONT_SIZE - 7)
    _shade_trimmed_edges(ax_acc, result)
    _style_axis(ax_acc)

    ax_disp.plot(
        result.time_s,
        result.displacement_m * 1000.0,
        color=Config.DISP_COLOR,
        linewidth=Config.LINE_WIDTH,
    )
    ax_disp.set_title("2 频域积分位移", fontproperties=CN_FONT, fontsize=FONT_SIZE - 5)
    ax_disp.set_xlabel("时间 (s)", fontproperties=CN_FONT, fontsize=FONT_SIZE - 7)
    ax_disp.set_ylabel("位移 (mm)", fontproperties=CN_FONT, fontsize=FONT_SIZE - 7)
    _shade_trimmed_edges(ax_disp, result)
    _style_axis(ax_disp)

    ax_psd.semilogy(
        result.psd_freq_hz,
        result.psd_acc,
        color=Config.ACC_COLOR,
        linewidth=Config.LINE_WIDTH,
        label="实测加速度 PSD",
    )
    visible_modes = [
        mode
        for mode in result.modes
        if mode.frequency_hz <= Config.PSD_MODE_LABEL_MAX_HZ
    ]
    for mode in visible_modes:
        ax_psd.axvline(
            mode.frequency_hz,
            color=Config.MODE_COLOR,
            linewidth=0.9,
            alpha=0.45,
        )
        if mode.order == 1 or mode.order % Config.PSD_MODE_LABEL_STEP == 0:
            ax_psd.text(
                mode.frequency_hz,
                np.nanmax(result.psd_acc) * 0.65,
                f"{mode.order}",
                ha="center",
                va="bottom",
                fontproperties=ENG_FONT,
                fontsize=FONT_SIZE - 9,
                color=Config.MODE_COLOR,
            )
    ax_psd.set_xlim(0.0, min(10.0, Config.FS / 2.0))
    ax_psd.set_title("3 128 阶理论振型基底", fontproperties=CN_FONT, fontsize=FONT_SIZE - 5)
    ax_psd.set_xlabel("频率 (Hz)", fontproperties=CN_FONT, fontsize=FONT_SIZE - 7)
    ax_psd.set_ylabel("PSD", fontproperties=ENG_FONT, fontsize=FONT_SIZE - 7)
    _style_axis(ax_psd)

    dynamic = result.axial_stress_mpa - result.static_stress_mpa
    ax_stress.plot(
        result.time_s,
        dynamic,
        color=Config.STRESS_COLOR,
        linewidth=Config.LINE_WIDTH,
    )
    ax_stress.axhline(
        result.dynamic_stress_mean_mpa,
        color=Config.ACC_COLOR,
        linewidth=1.0,
        linestyle="--",
        alpha=0.8,
        label="均值",
    )
    ax_stress.set_title("4 动态轴向应力时程", fontproperties=CN_FONT, fontsize=FONT_SIZE - 5)
    ax_stress.set_xlabel("时间 (s)", fontproperties=CN_FONT, fontsize=FONT_SIZE - 7)
    ax_stress.set_ylabel("动态应力 (MPa)", fontproperties=CN_FONT, fontsize=FONT_SIZE - 7)
    _shade_trimmed_edges(ax_stress, result)
    _style_axis(ax_stress)

    if len(result.stress_amplitudes_mpa) > 0:
        ax_hist.hist(
            result.stress_amplitudes_mpa,
            bins=32,
            weights=result.stress_counts,
            color=Config.HIST_COLOR,
            alpha=0.78,
            edgecolor="white",
            linewidth=0.5,
        )
    ax_hist.set_title("5 雨流应力幅分布", fontproperties=CN_FONT, fontsize=FONT_SIZE - 5)
    ax_hist.set_xlabel("应力幅 (MPa)", fontproperties=CN_FONT, fontsize=FONT_SIZE - 7)
    ax_hist.set_ylabel("循环计数", fontproperties=CN_FONT, fontsize=FONT_SIZE - 7)
    _style_axis(ax_hist)

    fig.suptitle(
        "单测点加速度到轴向应力幅的探索性反演",
        fontproperties=CN_FONT,
        fontsize=FONT_SIZE + 1,
        y=0.985,
    )
    fig.text(
        0.5,
        0.945,
        _format_sample_label(result),
        ha="center",
        va="top",
        fontproperties=CN_FONT,
        fontsize=FONT_SIZE - 8,
        color="#404040",
    )
    fig.text(
        0.99,
        0.012,
        (
            f"静应力={result.static_stress_mpa:.1f} MPa；"
            f"动态均值={result.dynamic_stress_mean_mpa:.4f} MPa；"
            f"3000 s 段，首尾各裁 {Config.EDGE_TRIM_SECOND:.0f} s；"
            f"仅滤去一阶以下：f<{result.integration_lowcut_hz:.2f} Hz"
        ),
        ha="right",
        va="bottom",
        fontproperties=CN_FONT,
        fontsize=FONT_SIZE - 10,
        color="#404040",
    )
    fig.subplots_adjust(left=0.075, right=0.975, bottom=0.075, top=0.91, hspace=0.50, wspace=0.25)
    return fig


def plot_frequency_stress_process(
    result: FrequencyStressResult,
    title: str = "基于张紧弦基频 gap 的应力幅探索",
) -> plt.Figure:
    fig = plt.figure(figsize=Config.FIG_SIZE)
    gs = fig.add_gridspec(3, 2, height_ratios=[1.1, 1.0, 1.1])
    ax_spec = fig.add_subplot(gs[0, :])
    ax_freq = fig.add_subplot(gs[1, 0])
    ax_psd = fig.add_subplot(gs[1, 1])
    ax_stress = fig.add_subplot(gs[2, 0])
    ax_hist = fig.add_subplot(gs[2, 1])

    freq_mask = result.spectrogram_freq_hz <= Config.TF_FREQ_MAX_HZ
    power_db = 10.0 * np.log10(result.spectrogram_power[freq_mask, :] + np.finfo(float).tiny)
    mesh = ax_spec.pcolormesh(
        result.spectrogram_time_s,
        result.spectrogram_freq_hz[freq_mask],
        power_db,
        shading="auto",
        cmap="viridis",
    )
    ax_spec.plot(
        result.frequency_time_s,
        result.base_frequency_hz,
        color="#E3625D",
        linewidth=1.0,
        label="主峰附近 gap 平均基频",
    )
    ax_spec.axhline(
        result.static_base_frequency_hz,
        color="white",
        linewidth=1.0,
        linestyle="--",
        alpha=0.85,
    )
    _shade_trimmed_edges(ax_spec, result)
    ax_spec.set_title("1 60s 窗 / 1 点步长频谱图", fontproperties=CN_FONT, fontsize=FONT_SIZE - 5)
    ax_spec.set_xlabel("时间 (s)", fontproperties=CN_FONT, fontsize=FONT_SIZE - 7)
    ax_spec.set_ylabel("频率 (Hz)", fontproperties=CN_FONT, fontsize=FONT_SIZE - 7)
    ax_spec.legend(prop=CN_FONT, fontsize=FONT_SIZE - 9, loc="upper right", frameon=False)
    _style_axis(ax_spec)
    cbar = fig.colorbar(mesh, ax=ax_spec, pad=0.01)
    cbar.set_label("PSD (dB)", fontproperties=ENG_FONT, fontsize=FONT_SIZE - 9)
    cbar.ax.tick_params(labelsize=FONT_SIZE - 10)

    ax_freq.plot(
        result.frequency_time_s,
        result.base_frequency_hz,
        color=Config.DISP_COLOR,
        linewidth=Config.LINE_WIDTH,
    )
    _shade_trimmed_edges(ax_freq, result)
    ax_freq.set_title("2 主峰附近 10 个 gap 平均估计基频 f1(t)", fontproperties=CN_FONT, fontsize=FONT_SIZE - 5)
    ax_freq.set_xlabel("时间 (s)", fontproperties=CN_FONT, fontsize=FONT_SIZE - 7)
    ax_freq.set_ylabel("频率 (Hz)", fontproperties=CN_FONT, fontsize=FONT_SIZE - 7)
    _style_axis(ax_freq)

    ax_psd.semilogy(
        result.psd_freq_hz,
        result.psd_acc,
        color=Config.ACC_COLOR,
        linewidth=Config.LINE_WIDTH,
    )
    ax_psd.axvline(
        result.dominant_frequency_hz,
        color=Config.MODE_COLOR,
        linewidth=1.3,
        linestyle="--",
        label=f"主峰 {result.dominant_frequency_hz:.2f} Hz",
    )
    ax_psd.set_xlim(0.0, Config.TF_FREQ_MAX_HZ)
    ax_psd.set_title("3 整段 PSD 与全局主峰", fontproperties=CN_FONT, fontsize=FONT_SIZE - 5)
    ax_psd.set_xlabel("频率 (Hz)", fontproperties=CN_FONT, fontsize=FONT_SIZE - 7)
    ax_psd.set_ylabel("PSD", fontproperties=ENG_FONT, fontsize=FONT_SIZE - 7)
    ax_psd.legend(prop=ENG_FONT, fontsize=FONT_SIZE - 9, loc="upper right", frameon=False)
    _style_axis(ax_psd)

    dynamic_stress = result.ridge_stress_mpa - result.static_stress_mpa
    ax_stress.plot(
        result.frequency_time_s,
        dynamic_stress,
        color=Config.STRESS_COLOR,
        linewidth=Config.LINE_WIDTH,
    )
    ax_stress.axhline(
        result.dynamic_stress_mean_mpa,
        color=Config.ACC_COLOR,
        linewidth=1.0,
        linestyle="--",
        alpha=0.8,
    )
    _shade_trimmed_edges(ax_stress, result)
    ax_stress.set_title(r"4 $T=4mL^2f_1^2$ 动态应力", fontproperties=CN_FONT, fontsize=FONT_SIZE - 5)
    ax_stress.set_xlabel("时间 (s)", fontproperties=CN_FONT, fontsize=FONT_SIZE - 7)
    ax_stress.set_ylabel("动态应力 (MPa)", fontproperties=CN_FONT, fontsize=FONT_SIZE - 7)
    _style_axis(ax_stress)

    if len(result.stress_amplitudes_mpa) > 0:
        ax_hist.hist(
            result.stress_amplitudes_mpa,
            bins=36,
            weights=result.stress_counts,
            color=Config.HIST_COLOR,
            alpha=0.78,
            edgecolor="white",
            linewidth=0.5,
        )
    ax_hist.set_title("5 雨流应力幅分布", fontproperties=CN_FONT, fontsize=FONT_SIZE - 5)
    ax_hist.set_xlabel("应力幅 (MPa)", fontproperties=CN_FONT, fontsize=FONT_SIZE - 7)
    ax_hist.set_ylabel("循环计数", fontproperties=CN_FONT, fontsize=FONT_SIZE - 7)
    _style_axis(ax_hist)

    sample = result.sample
    sensor_id = sample["inplane_sensor_id"]
    cable_id = _cable_id_from_sensor(sensor_id)
    timestamp = sample.get("timestamp", [])
    time_str = "-".join(str(x) for x in timestamp) if timestamp else "unknown time"
    fig.suptitle(
        title,
        fontproperties=CN_FONT,
        fontsize=FONT_SIZE + 1,
        y=0.985,
    )
    fig.text(
        0.5,
        0.945,
        (
            f"{cable_id} 面内测点；{time_str} win={sample['window_idx']}；"
            f"整段主峰={result.dominant_frequency_hz:.3f} Hz；"
            f"静态基频={result.static_base_frequency_hz:.3f} Hz；"
            f"每窗以主峰附近 {Config.TF_NEARBY_GAP_COUNT} 个 gap 平均估计 f1(t)"
        ),
        ha="center",
        va="top",
        fontproperties=CN_FONT,
        fontsize=FONT_SIZE - 8,
        color="#404040",
    )
    fig.text(
        0.99,
        0.012,
        (
            f"静应力={result.static_stress_mpa:.1f} MPa；"
            f"动态均值={result.dynamic_stress_mean_mpa:.4f} MPa；"
            f"3000 s 段，首尾各裁 {Config.EDGE_TRIM_SECOND:.0f} s"
        ),
        ha="right",
        va="bottom",
        fontproperties=CN_FONT,
        fontsize=FONT_SIZE - 10,
        color="#404040",
    )
    fig.subplots_adjust(left=0.075, right=0.975, bottom=0.075, top=0.91, hspace=0.55, wspace=0.25)
    return fig


def _shade_standard_edges(ax: plt.Axes) -> None:
    left_end = Config.EDGE_TRIM_SECOND
    right_start = Config.ANALYSIS_DURATION_SECOND - Config.EDGE_TRIM_SECOND
    ax.axvspan(0.0, left_end, color="#D0D0D0", alpha=0.28, linewidth=0)
    ax.axvspan(
        right_start,
        Config.ANALYSIS_DURATION_SECOND,
        color="#D0D0D0",
        alpha=0.28,
        linewidth=0,
    )
    ax.axvline(left_end, color="#808080", linewidth=0.8, linestyle="--")
    ax.axvline(right_start, color="#808080", linewidth=0.8, linestyle="--")


def plot_modal_flow(result: ModalFlowResult) -> plt.Figure:
    fig = plt.figure(figsize=(16, 11))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.2, 1.0])
    ax_spec = fig.add_subplot(gs[0, :])
    ax_order = fig.add_subplot(gs[1, 0])
    ax_hist = fig.add_subplot(gs[1, 1])

    freq_mask = result.spectrogram_freq_hz <= Config.TF_FREQ_MAX_HZ
    power_db = 10.0 * np.log10(result.spectrogram_power[freq_mask, :] + np.finfo(float).tiny)
    mesh = ax_spec.pcolormesh(
        result.spectrogram_time_s,
        result.spectrogram_freq_hz[freq_mask],
        power_db,
        shading="auto",
        cmap="viridis",
    )

    if len(result.modal_frequency_hz) > 0:
        scatter = ax_spec.scatter(
            result.modal_time_s,
            result.modal_frequency_hz,
            c=result.modal_order,
            s=5,
            cmap="turbo",
            alpha=0.65,
            linewidths=0,
        )
        cbar = fig.colorbar(scatter, ax=ax_spec, pad=0.01)
        cbar.set_label("阶次 n", fontproperties=CN_FONT, fontsize=FONT_SIZE - 9)
        cbar.ax.tick_params(labelsize=FONT_SIZE - 10)

    for order in range(Config.MODAL_FLOW_MIN_ORDER, Config.MODAL_FLOW_MAX_ORDER + 1):
        freq = order * result.static_base_frequency_hz
        if freq > Config.TF_FREQ_MAX_HZ:
            break
        ax_spec.axhline(freq, color="white", linewidth=0.35, alpha=0.35)

    _shade_standard_edges(ax_spec)
    ax_spec.set_title("1 模态流识别：谱峰落在静态基频整数倍附近", fontproperties=CN_FONT, fontsize=FONT_SIZE - 5)
    ax_spec.set_xlabel("时间 (s)", fontproperties=CN_FONT, fontsize=FONT_SIZE - 7)
    ax_spec.set_ylabel("频率 (Hz)", fontproperties=CN_FONT, fontsize=FONT_SIZE - 7)
    _style_axis(ax_spec)
    bg_cbar = fig.colorbar(mesh, ax=ax_spec, pad=0.08)
    bg_cbar.set_label("PSD (dB)", fontproperties=ENG_FONT, fontsize=FONT_SIZE - 9)
    bg_cbar.ax.tick_params(labelsize=FONT_SIZE - 10)

    if len(result.modal_order) > 0:
        ax_order.scatter(
            result.modal_time_s,
            result.modal_order,
            c=10.0 * np.log10(result.modal_power + np.finfo(float).tiny),
            s=5,
            cmap="magma",
            alpha=0.7,
            linewidths=0,
        )
    _shade_standard_edges(ax_order)
    ax_order.set_title("2 识别阶次随时间分布", fontproperties=CN_FONT, fontsize=FONT_SIZE - 5)
    ax_order.set_xlabel("时间 (s)", fontproperties=CN_FONT, fontsize=FONT_SIZE - 7)
    ax_order.set_ylabel("阶次 n", fontproperties=CN_FONT, fontsize=FONT_SIZE - 7)
    ax_order.set_ylim(Config.MODAL_FLOW_MIN_ORDER - 1, Config.MODAL_FLOW_MAX_ORDER + 1)
    _style_axis(ax_order)

    if len(result.modal_order) > 0:
        bins = np.arange(
            Config.MODAL_FLOW_MIN_ORDER - 0.5,
            Config.MODAL_FLOW_MAX_ORDER + 1.5,
            1.0,
        )
        ax_hist.hist(
            result.modal_order,
            bins=bins,
            color=Config.HIST_COLOR,
            alpha=0.78,
            edgecolor="white",
            linewidth=0.5,
        )
    ax_hist.set_title("3 各阶识别次数", fontproperties=CN_FONT, fontsize=FONT_SIZE - 5)
    ax_hist.set_xlabel("阶次 n", fontproperties=CN_FONT, fontsize=FONT_SIZE - 7)
    ax_hist.set_ylabel("识别次数", fontproperties=CN_FONT, fontsize=FONT_SIZE - 7)
    _style_axis(ax_hist)

    sample = result.sample
    sensor_id = sample["inplane_sensor_id"]
    cable_id = _cable_id_from_sensor(sensor_id)
    timestamp = sample.get("timestamp", [])
    time_str = "-".join(str(x) for x in timestamp) if timestamp else "unknown time"
    fig.suptitle(
        "张紧弦模态流识别探索",
        fontproperties=CN_FONT,
        fontsize=FONT_SIZE + 1,
        y=0.985,
    )
    fig.text(
        0.5,
        0.945,
        (
            f"{cable_id} 面内测点；{time_str} win={sample['window_idx']}；"
            f"静态基频={result.static_base_frequency_hz:.3f} Hz；"
            f"整段主峰={result.dominant_frequency_hz:.3f} Hz；"
            f"识别点={len(result.modal_order)}"
        ),
        ha="center",
        va="top",
        fontproperties=CN_FONT,
        fontsize=FONT_SIZE - 8,
        color="#404040",
    )
    fig.subplots_adjust(left=0.075, right=0.94, bottom=0.075, top=0.91, hspace=0.42, wspace=0.24)
    return fig


def plot_modal_flow_tension(result: ModalFlowTensionResult) -> plt.Figure:
    fig = plt.figure(figsize=Config.FIG_SIZE)
    gs = fig.add_gridspec(3, 2, height_ratios=[1.0, 1.0, 1.1])
    ax_f1 = fig.add_subplot(gs[0, :])
    ax_count = fig.add_subplot(gs[1, 0])
    ax_stress = fig.add_subplot(gs[1, 1])
    ax_hist = fig.add_subplot(gs[2, :])

    ax_f1.plot(
        result.time_s,
        result.base_frequency_hz,
        color=Config.DISP_COLOR,
        linewidth=Config.LINE_WIDTH,
    )
    ax_f1.axhline(
        result.static_base_frequency_hz,
        color=Config.MODE_COLOR,
        linestyle="--",
        linewidth=1.0,
        alpha=0.8,
        label="静态基频",
    )
    _shade_standard_edges(ax_f1)
    ax_f1.set_title("1 模态流 1s 聚合基频 f1(t)", fontproperties=CN_FONT, fontsize=FONT_SIZE - 5)
    ax_f1.set_xlabel("时间 (s)", fontproperties=CN_FONT, fontsize=FONT_SIZE - 7)
    ax_f1.set_ylabel("基频 (Hz)", fontproperties=CN_FONT, fontsize=FONT_SIZE - 7)
    ax_f1.legend(prop=CN_FONT, fontsize=FONT_SIZE - 9, loc="upper right", frameon=False)
    _style_axis(ax_f1)

    ax_count.plot(
        result.time_s,
        result.valid_point_count,
        color=Config.ACC_COLOR,
        linewidth=0.9,
    )
    _shade_standard_edges(ax_count)
    ax_count.set_title("2 每秒参与估计的模态点数", fontproperties=CN_FONT, fontsize=FONT_SIZE - 5)
    ax_count.set_xlabel("时间 (s)", fontproperties=CN_FONT, fontsize=FONT_SIZE - 7)
    ax_count.set_ylabel("点数", fontproperties=CN_FONT, fontsize=FONT_SIZE - 7)
    _style_axis(ax_count)

    dynamic_stress = result.stress_mpa - result.static_stress_mpa
    ax_stress.plot(
        result.time_s,
        dynamic_stress,
        color=Config.STRESS_COLOR,
        linewidth=Config.LINE_WIDTH,
    )
    ax_stress.axhline(
        result.dynamic_stress_mean_mpa,
        color=Config.ACC_COLOR,
        linewidth=1.0,
        linestyle="--",
        alpha=0.8,
    )
    _shade_standard_edges(ax_stress)
    ax_stress.set_title(r"3 $T=4mL^2f_1^2$ 动态应力", fontproperties=CN_FONT, fontsize=FONT_SIZE - 5)
    ax_stress.set_xlabel("时间 (s)", fontproperties=CN_FONT, fontsize=FONT_SIZE - 7)
    ax_stress.set_ylabel("动态应力 (MPa)", fontproperties=CN_FONT, fontsize=FONT_SIZE - 7)
    _style_axis(ax_stress)

    if len(result.stress_amplitudes_mpa) > 0:
        ax_hist.hist(
            result.stress_amplitudes_mpa,
            bins=42,
            weights=result.stress_counts,
            color=Config.HIST_COLOR,
            alpha=0.78,
            edgecolor="white",
            linewidth=0.5,
        )
    ax_hist.set_title("4 雨流应力幅分布", fontproperties=CN_FONT, fontsize=FONT_SIZE - 5)
    ax_hist.set_xlabel("应力幅 (MPa)", fontproperties=CN_FONT, fontsize=FONT_SIZE - 7)
    ax_hist.set_ylabel("循环计数", fontproperties=CN_FONT, fontsize=FONT_SIZE - 7)
    _style_axis(ax_hist)

    sample = result.sample
    sensor_id = sample["inplane_sensor_id"]
    cable_id = _cable_id_from_sensor(sensor_id)
    timestamp = sample.get("timestamp", [])
    time_str = "-".join(str(x) for x in timestamp) if timestamp else "unknown time"
    fig.suptitle(
        "基于模态流 1s 聚合的索力/应力探索",
        fontproperties=CN_FONT,
        fontsize=FONT_SIZE + 1,
        y=0.985,
    )
    fig.text(
        0.5,
        0.945,
        (
            f"{cable_id} 面内测点；{time_str} win={sample['window_idx']}；"
            f"静态基频={result.static_base_frequency_hz:.3f} Hz；"
            f"1s 聚合；首尾各裁 {Config.EDGE_TRIM_SECOND:.0f} s"
        ),
        ha="center",
        va="top",
        fontproperties=CN_FONT,
        fontsize=FONT_SIZE - 8,
        color="#404040",
    )
    fig.text(
        0.99,
        0.012,
        (
            f"静应力={result.static_stress_mpa:.1f} MPa；"
            f"动态均值={result.dynamic_stress_mean_mpa:.4f} MPa"
        ),
        ha="right",
        va="bottom",
        fontproperties=CN_FONT,
        fontsize=FONT_SIZE - 10,
        color="#404040",
    )
    fig.subplots_adjust(left=0.075, right=0.975, bottom=0.075, top=0.91, hspace=0.48, wspace=0.25)
    return fig


def plot_unified_viv_stress_exploration(
    modal_flow: ModalFlowResult,
    tension_fast: ModalFlowTensionResult,
    tension_slow: ModalFlowTensionResult,
) -> plt.Figure:
    fig = plt.figure(figsize=Config.FIG_SIZE)
    gs = fig.add_gridspec(3, 2, height_ratios=[1.15, 1.0, 1.05])
    ax_spec = fig.add_subplot(gs[0, :])
    ax_f1 = fig.add_subplot(gs[1, 0])
    ax_count = fig.add_subplot(gs[1, 1])
    ax_stress = fig.add_subplot(gs[2, 0])
    ax_hist = fig.add_subplot(gs[2, 1])

    freq_mask = modal_flow.spectrogram_freq_hz <= Config.TF_FREQ_MAX_HZ
    power_db = 10.0 * np.log10(
        modal_flow.spectrogram_power[freq_mask, :] + np.finfo(float).tiny
    )
    mesh = ax_spec.pcolormesh(
        modal_flow.spectrogram_time_s,
        modal_flow.spectrogram_freq_hz[freq_mask],
        power_db,
        shading="auto",
        cmap="viridis",
    )
    max_order_to_draw = min(
        Config.MODAL_FLOW_MAX_ORDER,
        int(Config.TF_FREQ_MAX_HZ / modal_flow.static_base_frequency_hz),
    )
    for order in range(Config.MODAL_FLOW_MIN_ORDER, max_order_to_draw + 1):
        freq = order * modal_flow.static_base_frequency_hz
        ax_spec.axhline(freq, color="white", linewidth=0.45, alpha=0.42)
        if order == 1 or order % 4 == 0:
            ax_spec.text(
                Config.ANALYSIS_DURATION_SECOND + 12.0,
                freq,
                f"{order}",
                ha="left",
                va="center",
                fontproperties=ENG_FONT,
                fontsize=FONT_SIZE - 11,
                color=Config.MODE_COLOR,
                clip_on=False,
            )

    _shade_standard_edges(ax_spec)
    ax_spec.set_xlim(0.0, Config.ANALYSIS_DURATION_SECOND)
    ax_spec.set_ylim(0.0, Config.TF_FREQ_MAX_HZ)
    ax_spec.set_title("1 时频谱与整数倍模态峰", fontproperties=CN_FONT, fontsize=FONT_SIZE - 5)
    ax_spec.set_xlabel("时间 (s)", fontproperties=CN_FONT, fontsize=FONT_SIZE - 7)
    ax_spec.set_ylabel("频率 (Hz)", fontproperties=CN_FONT, fontsize=FONT_SIZE - 7)
    _style_axis(ax_spec)
    cbar = fig.colorbar(mesh, ax=ax_spec, pad=0.012)
    cbar.set_label("PSD (dB)", fontproperties=ENG_FONT, fontsize=FONT_SIZE - 9)
    cbar.ax.tick_params(labelsize=FONT_SIZE - 10)

    ax_f1.plot(
        tension_fast.time_s,
        tension_fast.base_frequency_hz,
        color=Config.DISP_COLOR,
        linewidth=0.55,
        alpha=0.42,
        label=f"{tension_fast.aggregation_window_second:g}s 聚合",
    )
    ax_f1.plot(
        tension_slow.time_s,
        tension_slow.base_frequency_hz,
        color=Config.STRESS_COLOR,
        linewidth=Config.LINE_WIDTH,
        alpha=0.95,
        label=f"{tension_slow.aggregation_window_second:g}s 聚合",
    )
    ax_f1.axhline(
        tension_slow.static_base_frequency_hz,
        color=Config.MODE_COLOR,
        linewidth=1.0,
        linestyle="--",
        alpha=0.8,
    )
    _shade_standard_edges(ax_f1)
    ax_f1.set_title("2 模态流聚合基频对比", fontproperties=CN_FONT, fontsize=FONT_SIZE - 5)
    ax_f1.set_xlabel("时间 (s)", fontproperties=CN_FONT, fontsize=FONT_SIZE - 7)
    ax_f1.set_ylabel("基频 (Hz)", fontproperties=CN_FONT, fontsize=FONT_SIZE - 7)
    ax_f1.legend(prop=CN_FONT, fontsize=FONT_SIZE - 9, loc="upper right", frameon=False)
    _style_axis(ax_f1)

    ax_count.plot(
        tension_fast.time_s,
        tension_fast.valid_point_count,
        color=Config.ACC_COLOR,
        linewidth=0.45,
        alpha=0.42,
        label=f"{tension_fast.aggregation_window_second:g}s 聚合",
    )
    ax_count.plot(
        tension_slow.time_s,
        tension_slow.valid_point_count,
        color=Config.MODE_COLOR,
        linewidth=Config.LINE_WIDTH,
        alpha=0.88,
        label=f"{tension_slow.aggregation_window_second:g}s 聚合",
    )
    _shade_standard_edges(ax_count)
    ax_count.set_title("3 聚合窗内模态点数", fontproperties=CN_FONT, fontsize=FONT_SIZE - 5)
    ax_count.set_xlabel("时间 (s)", fontproperties=CN_FONT, fontsize=FONT_SIZE - 7)
    ax_count.set_ylabel("点数", fontproperties=CN_FONT, fontsize=FONT_SIZE - 7)
    ax_count.legend(prop=CN_FONT, fontsize=FONT_SIZE - 9, loc="upper right", frameon=False)
    _style_axis(ax_count)

    dynamic_fast = tension_fast.stress_mpa - tension_fast.static_stress_mpa
    dynamic_slow = tension_slow.stress_mpa - tension_slow.static_stress_mpa
    ax_stress.plot(
        tension_fast.time_s,
        dynamic_fast,
        color=Config.DISP_COLOR,
        linewidth=0.55,
        alpha=0.42,
        label=f"{tension_fast.aggregation_window_second:g}s 聚合",
    )
    ax_stress.plot(
        tension_slow.time_s,
        dynamic_slow,
        color=Config.STRESS_COLOR,
        linewidth=Config.LINE_WIDTH,
        alpha=0.95,
        label=f"{tension_slow.aggregation_window_second:g}s 聚合",
    )
    ax_stress.axhline(
        tension_slow.dynamic_stress_mean_mpa,
        color=Config.ACC_COLOR,
        linewidth=1.0,
        linestyle="--",
        alpha=0.8,
    )
    _shade_standard_edges(ax_stress)
    ax_stress.set_title(r"4 $T=4mL^2f_1^2$ 动态应力", fontproperties=CN_FONT, fontsize=FONT_SIZE - 5)
    ax_stress.set_xlabel("时间 (s)", fontproperties=CN_FONT, fontsize=FONT_SIZE - 7)
    ax_stress.set_ylabel("动态应力 (MPa)", fontproperties=CN_FONT, fontsize=FONT_SIZE - 7)
    ax_stress.legend(prop=CN_FONT, fontsize=FONT_SIZE - 9, loc="upper right", frameon=False)
    _style_axis(ax_stress)

    if len(tension_fast.stress_amplitudes_mpa) > 0:
        ax_hist.hist(
            tension_fast.stress_amplitudes_mpa,
            bins=42,
            weights=tension_fast.stress_counts,
            color=Config.DISP_COLOR,
            alpha=0.38,
            edgecolor="white",
            linewidth=0.4,
            label=f"{tension_fast.aggregation_window_second:g}s 聚合",
        )
    if len(tension_slow.stress_amplitudes_mpa) > 0:
        ax_hist.hist(
            tension_slow.stress_amplitudes_mpa,
            bins=42,
            weights=tension_slow.stress_counts,
            color=Config.HIST_COLOR,
            alpha=0.62,
            edgecolor="white",
            linewidth=0.5,
            label=f"{tension_slow.aggregation_window_second:g}s 聚合",
        )
    ax_hist.set_title("5 雨流应力幅分布", fontproperties=CN_FONT, fontsize=FONT_SIZE - 5)
    ax_hist.set_xlabel("应力幅 (MPa)", fontproperties=CN_FONT, fontsize=FONT_SIZE - 7)
    ax_hist.set_ylabel("循环计数", fontproperties=CN_FONT, fontsize=FONT_SIZE - 7)
    ax_hist.legend(prop=CN_FONT, fontsize=FONT_SIZE - 9, loc="upper right", frameon=False)
    _style_axis(ax_hist)

    sample = modal_flow.sample
    sensor_id = sample["inplane_sensor_id"]
    cable_id = _cable_id_from_sensor(sensor_id)
    timestamp = sample.get("timestamp", [])
    time_str = "-".join(str(x) for x in timestamp) if timestamp else "unknown time"
    fig.suptitle(
        "VIV 模态流基频与应力幅探索",
        fontproperties=CN_FONT,
        fontsize=FONT_SIZE + 1,
        y=0.985,
    )
    fig.text(
        0.5,
        0.945,
        (
            f"{cable_id} 面内测点；{time_str} win={sample['window_idx']}；"
            f"静态基频={tension_slow.static_base_frequency_hz:.3f} Hz；"
            f"模态点={len(modal_flow.modal_order)}；0.02s 与 1s 聚合对比"
        ),
        ha="center",
        va="top",
        fontproperties=CN_FONT,
        fontsize=FONT_SIZE - 8,
        color="#404040",
    )
    fig.text(
        0.99,
        0.012,
        (
            f"静应力={tension_slow.static_stress_mpa:.1f} MPa；"
            f"动态均值：0.02s={tension_fast.dynamic_stress_mean_mpa:.4f} MPa，"
            f"1s={tension_slow.dynamic_stress_mean_mpa:.4f} MPa；"
            f"首尾各裁 {Config.EDGE_TRIM_SECOND:.0f} s"
        ),
        ha="right",
        va="bottom",
        fontproperties=CN_FONT,
        fontsize=FONT_SIZE - 10,
        color="#404040",
    )
    fig.subplots_adjust(left=0.075, right=0.955, bottom=0.075, top=0.91, hspace=0.50, wspace=0.24)
    return fig


def main() -> None:
    print("=" * 80)
    print("探索图：单测点加速度 → 轴向应力幅")
    print("=" * 80)

    print("\n[步骤1] 加载 DL-VIV 实测窗口...")
    dl_result = load_dl_result()
    all_samples = _pipeline_get_viv_samples(dl_result)
    print(f"[OK] DL-VIV 样本：{len(all_samples)} 个")

    print("\n[步骤2] 选择一个代表性实测样本...")
    unpacker = UNPACK(init_path=False)
    sample = _select_sample(all_samples, unpacker)

    print("\n[步骤3] 执行 VIV 模态流识别...")
    modal_flow_result = compute_modal_flow(sample, unpacker)

    print("\n[步骤4] 基于模态流执行 0.02s 与 1s 聚合索力/应力计算...")
    modal_flow_tension_fast = compute_modal_flow_tension(
        modal_flow_result,
        aggregation_window_second=Config.MODAL_FLOW_TENSION_WINDOW_SECONDS[0],
    )
    modal_flow_tension_slow = compute_modal_flow_tension(
        modal_flow_result,
        aggregation_window_second=Config.MODAL_FLOW_TENSION_WINDOW_SECONDS[1],
    )

    print("\n[步骤5] 绘制统一应力幅探索图...")
    modal_flow_tension_figure = plot_unified_viv_stress_exploration(
        modal_flow_result,
        modal_flow_tension_fast,
        modal_flow_tension_slow,
    )
    Config.MODAL_FLOW_TENSION_SNAPSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    modal_flow_tension_figure.savefig(
        Config.MODAL_FLOW_TENSION_SNAPSHOT_PATH,
        dpi=300,
        bbox_inches="tight",
    )
    print(f"[OK] 已保存：{Config.MODAL_FLOW_TENSION_SNAPSHOT_PATH}")

    web_push(
        modal_flow_tension_figure,
        page=Config.WEB_PAGE,
        slot=0,
        title="VIV 模态流基频与应力幅",
        page_cols=1,
    )
    plt.close(modal_flow_tension_figure)
    print(f"[OK] 已推送到 WebUI：{Config.WEB_PAGE}")


if __name__ == "__main__":
    main()
