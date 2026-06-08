import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

import logging
import time

# 导入各个模块的主函数
from src.training.deep_learning.scripts.search_hyperparams import main_search
from src.training.deep_learning.scripts.mlp import main as main_mlp
from src.training.deep_learning.scripts.rnn import main as main_rnn
from src.training.deep_learning.scripts.lstm import main as main_lstm
from src.training.deep_learning.scripts.cnn import main as main_cnn

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_all_pipelines():
    """
    整合所有模型的训练流程：
    1. 超参数搜索
    2. MLP训练
    3. RNN训练
    4. LSTM训练
    5. CNN训练
    """
    
    logger.info("=" * 80)
    logger.info("开始完整的模型训练流程")
    logger.info("=" * 80)
    
    try:
        # 1. 超参数搜索
        logger.info("\n" + "=" * 80)
        logger.info("【步骤1/5】超参数搜索")
        logger.info("=" * 80)
        start_time = time.time()
        main_search()
        elapsed_time = time.time() - start_time
        logger.info(f"超参数搜索完成，耗时：{elapsed_time:.2f}秒")
        
        # 2. MLP训练
        logger.info("\n" + "=" * 80)
        logger.info("【步骤2/5】MLP模型训练")
        logger.info("=" * 80)
        start_time = time.time()
        main_mlp()
        elapsed_time = time.time() - start_time
        logger.info(f"MLP训练完成，耗时：{elapsed_time:.2f}秒")
        
        # 3. RNN训练
        logger.info("\n" + "=" * 80)
        logger.info("【步骤3/5】RNN模型训练")
        logger.info("=" * 80)
        start_time = time.time()
        main_rnn()
        elapsed_time = time.time() - start_time
        logger.info(f"RNN训练完成，耗时：{elapsed_time:.2f}秒")
        
        # 4. LSTM训练
        logger.info("\n" + "=" * 80)
        logger.info("【步骤4/5】LSTM模型训练")
        logger.info("=" * 80)
        start_time = time.time()
        main_lstm()
        elapsed_time = time.time() - start_time
        logger.info(f"LSTM训练完成，耗时：{elapsed_time:.2f}秒")
        
        # 5. CNN训练
        logger.info("\n" + "=" * 80)
        logger.info("【步骤5/5】CNN模型训练")
        logger.info("=" * 80)
        start_time = time.time()
        main_cnn()
        elapsed_time = time.time() - start_time
        logger.info(f"CNN训练完成，耗时：{elapsed_time:.2f}秒")
        
        # 训练完成
        logger.info("\n" + "=" * 80)
        logger.info("【完成】所有模型训练流程已完成！")
        logger.info("=" * 80)
        logger.info("\n生成的结果文件位置：")
        logger.info("  - 超参数搜索：results/training_result/deep_learning_module/search_best_hyperparams/")
        logger.info("  - MLP训练：results/training_result/deep_learning_module/mlp/mlp_train_result.json")
        logger.info("  - RNN训练：results/training_result/deep_learning_module/rnn/rnn_train_result.json")
        logger.info("  - LSTM训练：results/training_result/deep_learning_module/lstm/lstm_train_result.json")
        logger.info("  - CNN训练：results/training_result/deep_learning_module/cnn/cnn_train_result.json")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"训练流程出错：{e}", exc_info=True)
        raise


if __name__ == "__main__":
    run_all_pipelines()

