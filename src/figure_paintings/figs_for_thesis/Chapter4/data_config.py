"""Chapter4 绘图数据路径配置：legacy / chapter4 双链路切换。"""

from pathlib import Path

# 切换数据源："legacy" | "chapter4"
DATA_SOURCE = "chapter4"

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent

LEGACY = {
    "dl_result_glob": "results/identification_result/res_cnn_full_dataset_*.json",
    "enriched_root": "results/enriched_stats",
    "mecc_result_glob": "results/identification_result_mecc_viv/mecc_viv_only_*.json",
}

CHAPTER4 = {
    "runtime_config_path": None,
    "predictions_enriched": "results/chapter4_characteristics/inference/predictions_enriched_exclude_c34_201_202_301.json",
    "predictions_enriched_raw": "results/chapter4_characteristics/inference/predictions_enriched.json",
    "enriched_root": "results/chapter4_characteristics/enriched",
    "mecc_result_glob": "results/identification_result_mecc_viv/mecc_viv_only_*.json",
}

EXCLUDED_SENSOR_IDS = {
    "ST-VIC-C34-201-01",
    "ST-VIC-C34-201-02",
    "ST-VIC-C34-202-01",
    "ST-VIC-C34-202-02",
    "ST-VIC-C34-301-01",
    "ST-VIC-C34-301-02",
}

# 风-振关系图测点（不含已排除的 C34-201/202/301）
SENSOR_GROUPS_WIND = {
    "C18 边跨": "ST-VIC-C18-101-01.json",
    "C34 边跨": "ST-VIC-C34-101-01.json",
    "C34 中跨": "ST-VIC-C34-102-01.json",
}

CLASS_DIR_NAMES = {
    0: "class_0_normal",
    1: "class_1_viv",
    2: "class_2_rwiv",
    3: "class_3_transition",
}
