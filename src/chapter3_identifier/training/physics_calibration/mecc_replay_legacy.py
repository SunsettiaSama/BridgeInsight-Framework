"""用 idx→annotation 的金标（与旧搜参一致）复跑 k=2,C=0.3,sigma=0.1。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import numpy as np
import yaml

# 须在 AnnotationDataset 之前导入 mecc，避免 src/statistics 遮蔽标准库 statistics
from src.training.physics_calibration import mecc as mecc_mod

from src.config.data_processer.datasets.AnnotationDataset.AnnotationDatasetConfig import AnnotationDatasetConfig
from src.data_processer.datasets.AnnotationDataset.AnnotationDataset import AnnotationDataset
from src.identifier.physics.base_mode_calculator import Cal_Mount

LEGACY = dict(sigma_0=0.1, k_viv=2.0, C_viv=0.3)


def _legacy_f0_scalar() -> float:
    mount = Cal_Mount()
    inplane_modes, _ = mount.base_modes()
    return float(inplane_modes[0])


def _load_val_binary_from_idx_map():
    with open(mecc_mod.CONFIG_PATH, encoding="utf-8") as f:
        cfg_dict = yaml.safe_load(f)
    cfg_dict["auto_split"] = True
    cfg_dict["split_ratio"] = mecc_mod.VAL_RATIO
    cfg_dict["enable_preload_cache"] = False
    cfg_dict["show_preload_progress"] = False
    dataset = AnnotationDataset(AnnotationDatasetConfig(**cfg_dict))
    val_ds = dataset.get_val_dataset()

    rows = []
    for i in range(len(val_ds)):
        orig_idx = val_ds.indices[i]
        anno = dataset._idx_to_annotation[orig_idx]
        label = int(anno["class_id"])
        if label not in mecc_mod.LABEL_IDS:
            continue
        data, _ = val_ds[i]
        sid = mecc_mod._sensor_id_from_annotation(anno)
        rows.append((data.numpy(), label, sid))

    f0_cache = mecc_mod._build_f0_cache({sid for _, _, sid in rows})
    features = np.stack([r[0] for r in rows])
    labels = np.array([r[1] for r in rows])
    f0s = np.array([f0_cache[r[2]] for r in rows], dtype=np.float64)
    return features, labels, f0s, dataset


def _report(title, metrics):
    cm = metrics["confusion_matrix"]
    w = metrics["weighted"]
    viv = metrics["per_class"]["VIV"]
    print(f"\n=== {title} ===")
    print(f"weighted F1={w['F1']:.4f}")
    print(f"VIV R={viv['Recall']:.4f}  P={viv['Precision']:.4f}  F1={viv['F1']:.4f}")
    print(f"CM: TN={cm[0][0]} FP={cm[0][1]} FN={cm[1][0]} TP={cm[1][1]}")


def main():
    features, labels, f0s_per, _ = _load_val_binary_from_idx_map()
    freq_p95 = mecc_mod.compute_freq_p95()
    f0_legacy = _legacy_f0_scalar()
    f0s_legacy = np.full(len(labels), f0_legacy)

    print(f"验证集（idx 金标）: N={len(labels)}  Normal={(labels==0).sum()}  VIV={(labels==1).sum()}")
    print(f"legacy f0 (base_modes[0]) = {f0_legacy:.4f} Hz")

    m, _ = mecc_mod._calc_metrics(
        LEGACY["sigma_0"], LEGACY["k_viv"], LEGACY["C_viv"],
        features, labels, f0s_legacy, freq_p95,
    )
    _report("历史参数 k=2,C=0.3,sigma=0.1 + 全局 f0=0.4897（旧代码口径）", m)

    m2, _ = mecc_mod._calc_metrics(
        LEGACY["sigma_0"], LEGACY["k_viv"], LEGACY["C_viv"],
        features, labels, f0s_per, freq_p95,
    )
    _report("同参数 + 每测点 inplane_mode(1) f0", m2)


if __name__ == "__main__":
    main()
