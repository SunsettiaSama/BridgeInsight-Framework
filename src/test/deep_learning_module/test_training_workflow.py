import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.config.deep_learning_module.models.SimpleMLPConfig import SimpleMLPConfig, DropoutConfig
from src.config.train_eval.deep_learning_module.sft import SFTTrainerConfig
from src.deep_learning_module.model_factory import get_model
from src.train_eval.deep_learning_module.sft import SFTTrainer
from src.config.data_processer.datasets.AnnotationDataset.AnnotationDatasetConfig import AnnotationDatasetConfig
from src.data_processer.datasets.data_factory import get_dataset


def test_simple_training_workflow():
    print("="*70)
    print("开始：简单训练流程测试（MLP + Annotation 数据集）")
    print("="*70)
    
    print("\n[1/5] 准备数据集...")
    dataset_config = AnnotationDatasetConfig(
        data_dir = r'F:\Research\Vibration Characteristics In Cable Vibration\data\2024September\SuTong\VIC',
        annotation_file="./results/dataset_annotation/annotation_results.json",
        task_type="classification",
    )
    
    dataset = get_dataset(dataset_config)
    train_dataset = dataset.get_train_dataset()
    val_dataset = dataset.get_val_dataset()
    
    sample_data, sample_label = train_dataset[0]
    input_shape = sample_data.shape
    num_classes = dataset.get_num_classes()
    
    print(f"✓ 数据集加载成功")
    print(f"  - 训练集大小：{len(train_dataset)}")
    print(f"  - 验证集大小：{len(val_dataset)}")
    print(f"  - 输入形状：{input_shape}")
    print(f"  - 类别数：{num_classes}")
    
    print("\n[2/5] 创建数据加载器...")
    train_loader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True,
        num_workers=0
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=0
    )
    print(f"✓ 数据加载器创建成功")
    print(f"  - 训练批次数：{len(train_loader)}")
    print(f"  - 验证批次数：{len(val_loader)}")
    
    print("\n[3/5] 创建 MLP 模型...")
    model_config = SimpleMLPConfig(
        input_shape=input_shape,
        hidden_dims=[128, 64],
        num_classes=num_classes,
        task_type="classification",
        activation_type="ReLU",
        dropout=DropoutConfig(enable=True, prob=0.2)
    )
    
    model = get_model(model_config)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"✓ MLP 模型创建成功")
    print(f"  - 模型类型：{model.__class__.__name__}")
    print(f"  - 总参数数：{total_params:,}")
    print(f"  - 可训练参数：{trainable_params:,}")
    
    print("\n[4/5] 创建训练配置...")
    train_config = SFTTrainerConfig(
        epochs=2,
        batch_size=16,
        learning_rate=0.001,
        optimizer="AdamW",
        device="cuda" if torch.cuda.is_available() else "cpu",
        output_dir="./test_output",
        loss_type="CrossEntropyLoss",
        sft_task_type="classification",
        best_model_metric="accuracy",
        save_best_model=True,
        save_freq=1,
        use_tensorboard=False,
        save_log_file=True,
        log_freq=2
    )
    
    print(f"✓ 训练配置创建成功")
    print(f"  - 设备：{train_config.device}")
    print(f"  - 优化器：{train_config.optimizer}")
    print(f"  - 学习率：{train_config.learning_rate}")
    
    print("\n[5/5] 开始训练...")
    trainer = SFTTrainer(config=train_config)
    trainer.train(
        model=model,
        train_dataloader=train_loader,
        val_dataloader=val_loader
    )
    
    print(f"\n✓ 训练完成！")
    print(f"  - 最优指标：{trainer.best_metric:.4f}")
    print(f"  - 最优轮次：第 {trainer.best_epoch + 1} 轮")
    print(f"  - 模型保存位置：{train_config.output_dir}")
    
    return True


def test_model_forward_pass():
    print("\n" + "="*70)
    print("额外测试：模型前向传播验证")
    print("="*70)
    
    batch_size = 4
    seq_len = 300
    feat_dim = 5
    num_classes = 3
    
    x = torch.randn(batch_size, seq_len, feat_dim)
    
    model_config = SimpleMLPConfig(
        input_shape=(seq_len, feat_dim),
        hidden_dims=[128, 64],
        num_classes=num_classes,
        task_type="classification",
        dropout=DropoutConfig(enable=False)
    )
    model = get_model(model_config)
    
    with torch.no_grad():
        output = model(x)
    
    print(f"✓ 前向传播成功")
    print(f"  - 输入形状：{x.shape}")
    print(f"  - 输出形状：{output.shape}")
    print(f"  - 预期输出形状：({batch_size}, {num_classes})")
    
    assert output.shape == (batch_size, num_classes), "输出形状不匹配"
    print(f"✓ 输出形状验证通过")
    
    return True


if __name__ == "__main__":
    print("\n" + "█"*70)
    print("█" + " "*68 + "█")
    print("█" + "  简单训练流程测试".center(68) + "█")
    print("█" + " "*68 + "█")
    print("█"*70 + "\n")
    
    forward_pass_ok = test_model_forward_pass()
    training_ok = test_simple_training_workflow()
    
    print("\n" + "="*70)
    print("测试总结")
    print("="*70)
    print(f"✓ 前向传播测试：{'通过' if forward_pass_ok else '失败'}")
    print(f"✓ 训练流程测试：{'通过' if training_ok else '失败'}")
    
    if forward_pass_ok and training_ok:
        print("\n🎉 所有测试通过！训练框架可以正常运行")
        exit(0)
    else:
        print("\n⚠️  部分测试失败，请查看上面的错误信息")
        exit(1)
