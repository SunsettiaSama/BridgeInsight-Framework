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
    "predictions_enriched": "results/chapter4_characteristics/inference/predictions_enriched.json",
    "enriched_root": "results/chapter4_characteristics/enriched",
    "mecc_result_glob": "results/identification_result_mecc_viv/mecc_viv_only_*.json",
}

CLASS_DIR_NAMES = {
    0: "class_0_normal",
    1: "class_1_viv",
    2: "class_2_rwiv",
    3: "class_3_transition",
}
