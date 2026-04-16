# Autor@ 猫毛

from pathlib import Path
from src.identifier.feature_analysis.run import run as feature_analysis_run
from src.config.identifier.feature_analysis.config import ProcessFullDataConfig, load_config
from src.config.data_processer.preprocess.preprocess_config import load_preprocess_config

_PROJECT_ROOT = Path(__file__).parent

# =============================================================================
# 全量识别结果后处理：特征提取 + 元数据建立
# =============================================================================

class ProcessFullData:

    @staticmethod
    def build_metadata(config_yaml: str = None) -> None:
        result_dir = _PROJECT_ROOT / "results" / "identification_result"
        enriched_files = sorted(result_dir.glob("res_cnn_full_dataset_*_enriched.json"))
        if not enriched_files:
            raise FileNotFoundError(
                f"未找到 enriched 识别结果文件，请先运行 enrich_results_with_metadata.py\n"
                f"搜索路径：{result_dir}"
            )
        result_path = enriched_files[-1]

        _, wind_cfg = load_preprocess_config()
        wind_metadata_path = wind_cfg.filter_result_path

        output_dir = _PROJECT_ROOT / "results" / "enriched_stats"

        cfg = load_config(config_yaml)

        print(f"[ProcessFullData] 识别结果：{result_path.name}")
        print(f"[ProcessFullData] 风元数据：{wind_metadata_path}")
        print(f"[ProcessFullData] 输出目录：{output_dir}")

        feature_analysis_run(
            result_path=str(result_path),
            wind_metadata_path=str(wind_metadata_path),
            output_dir=str(output_dir),
            cfg=cfg,
        )


# =============================================================================

if __name__ == "__main__":
    ProcessFullData.build_metadata()

