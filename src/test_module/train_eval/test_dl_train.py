import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.config.train_eval.deep_learning_module.base_config import BaseConfig
from src.training.deep_learning.trainer.sft import SFTTrainer


class SimpleFullyConnectedNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int):
        super(SimpleFullyConnectedNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


class SimpleDataset(Dataset):
    def __init__(self, num_samples: int = 100, input_dim: int = 50, num_classes: int = 3):
        self.num_samples = num_samples
        self.input_dim = input_dim
        self.num_classes = num_classes
        
        self.X = torch.randn(num_samples, input_dim)
        self.y = torch.randint(0, num_classes, (num_samples,))
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


class TrainerConfig(BaseConfig):
    epochs: int = 5
    batch_size: int = 16
    learning_rate: float = 1e-3
    optimizer: str = "adam"
    device: str = "cpu"
    output_dir: str = "./output"
    best_model_metric: str = "accuracy"
    use_mixed_precision: bool = False
    mixed_precision_type: str = "float16"
    use_distributed: bool = False
    use_tensorboard: bool = False
    tensorboard_log_dir: str = "./logs"
    resume_from_checkpoint: None = None
    loss_type: str = "cross_entropy"
    
    num_workers: int = 0
    pin_memory: bool = False
    shuffle: bool = True
    drop_last: bool = True
    
    use_tensorboard: bool = False
    dist_port: int = 29500
    find_unused_parameters: bool = False
    
    fix_feature_extractor: bool = False
    freeze_layer_prefixes: list = []
    feature_extractor_attrs: list = []
    
    pretrained_weight_path: str = None
    head_param_prefixes: list = []
    load_pretrained_head: bool = False
    
    weight_decay: float = 1e-4
    exclude_bias_from_weight_decay: bool = False
    head_lr_scale: float = 1.0
    
    scheduler: str = "constant"
    
    focal_gamma: float = 2.0
    num_classes: int = 3
    task_type: str = "classification"
    

def test_sft_trainer():
    print("=" * 60)
    print("开始测试SFTTrainer")
    print("=" * 60)
    
    config = TrainerConfig()
    
    print(f"配置创建成功")
    print(f"  - 训练轮数: {config.epochs}")
    print(f"  - 批次大小: {config.batch_size}")
    print(f"  - 学习率: {config.learning_rate}")
    print(f"  - 优化器: {config.optimizer}")
    print()
    
    train_dataset = SimpleDataset(num_samples=100, input_dim=50, num_classes=3)
    val_dataset = SimpleDataset(num_samples=30, input_dim=50, num_classes=3)
    
    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    
    print(f"数据集创建成功")
    print(f"  - 训练样本: {len(train_dataset)}")
    print(f"  - 验证样本: {len(val_dataset)}")
    print()
    
    model = SimpleFullyConnectedNet(input_dim=50, hidden_dim=128, num_classes=3)
    print(f"模型创建成功: {model.__class__.__name__}")
    print()
    
    try:
        trainer = SFTTrainer(
            config=config,
            model=model,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader
        )
        print("SFTTrainer初始化成功")
        print()
        
        print("=" * 60)
        print("开始训练")
        print("=" * 60)
        
        trainer.train(
            num_epochs=config.epochs,
            model=model,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader
        )
        
        print("=" * 60)
        print("训练完成！")
        print("=" * 60)
        
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_sft_trainer()
