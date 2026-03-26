"""
模型注册表：统一管理模型Config类和模型类的映射关系
- 模型Config类注册表：config_type → 对应的Config类
- 模型类注册表：config_type → 对应的模型类
"""
from typing import Type, Dict
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config.base_config import BaseConfig

# --------------------------
# 模型Config类注册表：管理模型配置
# --------------------------

from src.config.deep_learning_module.models.unet import UNetConfig
from src.config.deep_learning_module.models.mlp import SimpleMLPConfig
from src.config.deep_learning_module.models.cnn import CNNConfig
# 目前LSTM配置还未调整，先注释掉
from src.config.deep_learning_module.models.lstm import LSTMConfig
from src.config.deep_learning_module.models.rnn import RNNConfig

MODEL_CONFIG_REGISTRY: Dict[str, Type[BaseConfig]] = {
    # 格式："model_type" → 对应的模型Config类
    "unet": UNetConfig,
    "mlp": SimpleMLPConfig,
    "lstm": LSTMConfig,
    "cnn": CNNConfig,
    "rnn": RNNConfig,
}

# --------------------------
# 模型类注册表：管理模型实现
# --------------------------

# 导入模型类（这里需要根据实际项目结构调整导入路径）
# 假设模型存放在 src/deep_learning_module/models/ 下
try:
    from src.deep_learning_module.models.unet import UNet
    from src.deep_learning_module.models.mlp import MLP
    from src.deep_learning_module.models.lstm import LSTM
    from src.deep_learning_module.models.cnn import CNN
    from src.deep_learning_module.models.rnn import RNN
except ImportError as e:
    import warnings
    warnings.warn(f"模型导入失败，部分模型可能不可用：{e}")
    UNet = None
    MLP = None
    LSTM = None
    CNN = None
    RNN = None

MODEL_CLASS_REGISTRY: Dict[str, Type] = {
    # 格式："model_type" → 对应的模型类（与MODEL_CONFIG_REGISTRY的key一一对应）
    "unet": UNet,
    "mlp": MLP,
    "lstm": LSTM,
    "cnn": CNN,
    "rnn": RNN,
}
