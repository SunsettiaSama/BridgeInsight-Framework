"""
模型工厂测试脚本：验证通过Config创建模型的完整流程

测试覆盖（16个测试）：
1. MLP模型工厂
2. U-Net分割任务（2D输入）
3. U-Net分类任务（2D输入）
4. U-Net回归任务（2D输入）
5. U-Net时序输入（1D处理）
6. LSTM分类任务
7. LSTM Seq2Seq任务
8. LSTM Regression任务
9. CNN图像分类
10. CNN时序分类
11. CNN回归任务
12. RNN分类任务
13. RNN Seq2Seq任务
14. RNN Regression任务
15. Config序列化与反序列化
16. 异常处理验证

每个测试验证：
- Config创建和参数验证
- 通过工厂创建模型实例
- 模型前向传播
- 参数数量统计
- 输出形状验证
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import torch
import torch.nn as nn
from src.config.deep_learning_module.models.SimpleMLPConfig import SimpleMLPConfig
from src.config.deep_learning_module.models.unet import UNetConfig
from src.config.deep_learning_module.models.lstm import LSTMConfig
from src.config.deep_learning_module.models.cnn import CNNConfig, ConvConfig, FCConfig, DropoutConfig, PoolConfig
from src.config.deep_learning_module.models.rnn import RNNConfig
from src.deep_learning_module.model_factory import get_model


def print_section(title: str):
    """打印分隔符"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def count_parameters(model: nn.Module) -> tuple:
    """统计模型参数"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def test_mlp_model():
    """测试MLP模型工厂"""
    print_section("测试1：MLP模型")
    
    try:
        # 创建MLP Config
        config = SimpleMLPConfig(
            input_shape=(300, 5),
            hidden_dims=[128, 64],
            num_classes=3,
            task_type="classification",
            activation_type="ReLU",
            dropout={"enable": True, "prob": 0.2}
        )
        
        print(f"✓ MLP Config创建成功")
        print(f"  - 输入形状: {config.input_shape}")
        print(f"  - 隐藏层: {config.hidden_dims}")
        print(f"  - 类别数: {config.num_classes}")
        print(f"  - 激活函数: {config.activation_type}")
        print(f"  - 任务类型: {config.task_type}")
        
        # 通过工厂创建模型
        model = get_model(config)
        print(f"\n✓ MLP模型创建成功: {model.__class__.__name__}")
        
        # 统计参数
        total, trainable = count_parameters(model)
        print(f"  - 总参数数: {total:,}")
        print(f"  - 可训练参数: {trainable:,}")
        
        # 前向传播测试
        x = torch.randn(16, 300, 5)  # batch_size=16, seq_len=300, feat_dim=5
        with torch.no_grad():
            output = model(x)
        print(f"\n✓ 前向传播成功")
        print(f"  - 输入形状: {x.shape}")
        print(f"  - 输出形状: {output.shape}")
        
        # 验证输出维度
        assert output.shape == (16, 3), f"输出形状错误！期望(16, 3)，实际{output.shape}"
        print(f"  - 输出维度验证: ✓ 通过")
        
        return True
        
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_unet_model():
    """测试U-Net模型工厂"""
    print_section("测试2：U-Net模型（分割任务）")
    
    try:
        # 创建U-Net分割配置
        config = UNetConfig(
            in_channels=1,
            num_classes=5,
            task_type="segmentation"
        )
        
        print(f"✓ U-Net分割 Config创建成功")
        print(f"  - 输入通道数: {config.in_channels}")
        print(f"  - 输出类别数: {config.num_classes}")
        print(f"  - 网络类型: {config.network_type}")
        print(f"  - 任务类型: {config.task_type}")
        
        # 通过工厂创建模型
        model = get_model(config)
        print(f"\n✓ U-Net模型创建成功: {model.__class__.__name__}")
        
        # 统计参数
        total, trainable = count_parameters(model)
        print(f"  - 总参数数: {total:,}")
        print(f"  - 可训练参数: {trainable:,}")
        
        # 前向传播测试
        x = torch.randn(4, 1, 256, 256)  # batch_size=4, 1 channel, 256x256
        with torch.no_grad():
            output = model(x)
        print(f"\n✓ 前向传播成功")
        print(f"  - 输入形状: {x.shape}")
        print(f"  - 输出形状: {output.shape}")
        
        # 验证输出维度（分割任务输出空间维度与输入相同）
        assert output.shape == (4, 5, 256, 256), f"输出形状错误！期望(4, 5, 256, 256)，实际{output.shape}"
        print(f"  - 输出维度验证: ✓ 通过")
        
        return True
        
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_unet_classification():
    """测试U-Net分类模型工厂"""
    print_section("测试3：U-Net模型（分类任务）")
    
    try:
        # 创建U-Net分类配置
        config = UNetConfig(
            in_channels=3,
            num_classes=10,
            task_type="classification"
        )
        
        print(f"✓ U-Net分类 Config创建成功")
        print(f"  - 输入通道数: {config.in_channels}")
        print(f"  - 输出类别数: {config.num_classes}")
        print(f"  - 任务类型: {config.task_type}")
        
        # 通过工厂创建模型
        model = get_model(config)
        print(f"\n✓ U-Net模型创建成功: {model.__class__.__name__}")
        
        # 统计参数
        total, trainable = count_parameters(model)
        print(f"  - 总参数数: {total:,}")
        print(f"  - 可训练参数: {trainable:,}")
        
        # 前向传播测试
        x = torch.randn(8, 3, 224, 224)  # batch_size=8, 3 channels, 224x224
        with torch.no_grad():
            output = model(x)
        print(f"\n✓ 前向传播成功")
        print(f"  - 输入形状: {x.shape}")
        print(f"  - 输出形状: {output.shape}")
        
        # 验证输出维度（分类任务输出为(B, num_classes)）
        assert output.shape == (8, 10), f"输出形状错误！期望(8, 10)，实际{output.shape}"
        print(f"  - 输出维度验证: ✓ 通过")
        
        return True
        
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_unet_regression():
    """测试U-Net回归模型工厂"""
    print_section("测试4：U-Net模型（回归任务）")
    
    try:
        # 创建U-Net回归配置
        config = UNetConfig(
            in_channels=1,
            num_classes=1,
            task_type="regression",
            regression_output_dim=5
        )
        
        print(f"✓ U-Net回归 Config创建成功")
        print(f"  - 输入通道数: {config.in_channels}")
        print(f"  - 输出维度: {config.regression_output_dim}")
        print(f"  - 任务类型: {config.task_type}")
        
        # 通过工厂创建模型
        model = get_model(config)
        print(f"\n✓ U-Net模型创建成功: {model.__class__.__name__}")
        
        # 统计参数
        total, trainable = count_parameters(model)
        print(f"  - 总参数数: {total:,}")
        print(f"  - 可训练参数: {trainable:,}")
        
        # 前向传播测试
        x = torch.randn(6, 1, 128, 128)  # batch_size=6, 1 channel, 128x128
        with torch.no_grad():
            output = model(x)
        print(f"\n✓ 前向传播成功")
        print(f"  - 输入形状: {x.shape}")
        print(f"  - 输出形状: {output.shape}")
        
        # 验证输出维度（回归任务输出为(B, regression_output_dim)）
        assert output.shape == (6, 5), f"输出形状错误！期望(6, 5)，实际{output.shape}"
        print(f"  - 输出维度验证: ✓ 通过")
        
        return True
        
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_unet_timeseries():
    """测试U-Net时序输入"""
    print_section("测试5：U-Net模型（时序输入）")
    
    try:
        # 创建U-Net时序配置
        config = UNetConfig(
            in_channels=5,
            num_classes=3,
            task_type="classification",
            support_timeseries=True
        )
        
        print(f"✓ U-Net时序 Config创建成功")
        print(f"  - 输入通道数（特征维度）: {config.in_channels}")
        print(f"  - 输出类别数: {config.num_classes}")
        print(f"  - 支持时序: {config.support_timeseries}")
        print(f"  - 任务类型: {config.task_type}")
        
        # 通过工厂创建模型
        model = get_model(config)
        print(f"\n✓ U-Net模型创建成功: {model.__class__.__name__}")
        
        # 统计参数
        total, trainable = count_parameters(model)
        print(f"  - 总参数数: {total:,}")
        print(f"  - 可训练参数: {trainable:,}")
        
        # 前向传播测试 - 时序格式
        x = torch.randn(10, 300, 5)  # (batch_size, time_steps, features)
        with torch.no_grad():
            output = model(x)
        print(f"\n✓ 前向传播成功")
        print(f"  - 输入形状: {x.shape}")
        print(f"  - 输出形状: {output.shape}")
        
        # 验证输出维度（分类任务输出为(B, num_classes)）
        assert output.shape == (10, 3), f"输出形状错误！期望(10, 3)，实际{output.shape}"
        print(f"  - 输出维度验证: ✓ 通过")
        
        return True
        
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_config_to_dict_to_model():
    """测试Config序列化和反序列化"""
    print_section("测试6：Config序列化与反序列化")
    
    try:
        # 创建原始Config
        original_config = SimpleMLPConfig(
            input_shape=(200, 10),
            hidden_dims=[256, 128, 64],
            num_classes=5,
            task_type="classification"
        )
        
        print(f"✓ 原始Config创建成功")
        
        # 序列化为字典
        config_dict = original_config.to_dict()
        print(f"✓ Config序列化为字典: {len(config_dict)} 个字段")
        
        # 从字典反序列化
        restored_config = SimpleMLPConfig.from_dict(config_dict)
        print(f"✓ Config从字典反序列化成功")
        
        # 通过恢复的Config创建模型
        model = get_model(restored_config)
        print(f"\n✓ 模型创建成功: {model.__class__.__name__}")
        
        # 验证参数一致
        total1, _ = count_parameters(get_model(original_config))
        total2, _ = count_parameters(model)
        assert total1 == total2, f"参数数不一致！原始:{total1}, 恢复:{total2}"
        print(f"  - 参数数一致验证: ✓ 通过 ({total1:,} 参数)")
        
        return True
        
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_invalid_config():
    """测试异常处理"""
    print_section("测试7：异常处理")
    
    try:
        # 测试无效的激活函数
        try:
            config = SimpleMLPConfig(
                input_shape=(300, 5),
                hidden_dims=[128],
                activation_type="InvalidActivation"
            )
            print(f"✗ 应该抛出异常但没有")
            return False
        except ValueError as e:
            print(f"✓ 成功捕获Config校验异常:")
            print(f"  - 异常信息: {str(e)[:80]}...")
        
        # 测试通过工厂创建未注册的Config
        try:
            class UnregisteredConfig:
                pass
            
            get_model(UnregisteredConfig())
            print(f"✗ 应该抛出异常但没有")
            return False
        except (ValueError, TypeError) as e:
            print(f"\n✓ 成功捕获工厂异常:")
            print(f"  - 异常信息: {str(e)[:80]}...")
        
        return True
        
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_lstm_classification():
    """测试LSTM分类模型"""
    print_section("测试8：LSTM模型（分类任务）")
    
    try:
        config = LSTMConfig(
            input_size=5,
            num_classes=3,
            task_type="classification",
            hidden_size=64,
            num_layers=2,
            bidirectional=False
        )
        
        print(f"✓ LSTM分类 Config创建成功")
        print(f"  - 输入特征维度: {config.input_size}")
        print(f"  - 输出类别数: {config.num_classes}")
        print(f"  - 隐藏层维度: {config.hidden_size}")
        print(f"  - LSTM层数: {config.num_layers}")
        print(f"  - 任务类型: {config.task_type}")
        
        model = get_model(config)
        print(f"\n✓ LSTM模型创建成功: {model.__class__.__name__}")
        
        total, trainable = count_parameters(model)
        print(f"  - 总参数数: {total:,}")
        print(f"  - 可训练参数: {trainable:,}")
        
        x = torch.randn(16, 100, 5)
        with torch.no_grad():
            output = model(x)
        print(f"\n✓ 前向传播成功")
        print(f"  - 输入形状: {x.shape}")
        print(f"  - 输出形状: {output.shape}")
        
        assert output.shape == (16, 3), f"输出形状错误！期望(16, 3)，实际{output.shape}"
        print(f"  - 输出维度验证: ✓ 通过")
        
        return True
        
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_lstm_seq2seq():
    """测试LSTM序列到序列模型"""
    print_section("测试9：LSTM模型（Seq2Seq任务）")
    
    try:
        config = LSTMConfig(
            input_size=8,
            num_classes=4,
            task_type="seq2seq",
            hidden_size=32,
            num_layers=1,
            bidirectional=True
        )
        
        print(f"✓ LSTM Seq2Seq Config创建成功")
        print(f"  - 输入特征维度: {config.input_size}")
        print(f"  - 输出特征维度: {config.num_classes}")
        print(f"  - 隐藏层维度: {config.hidden_size}")
        print(f"  - 双向LSTM: {config.bidirectional}")
        print(f"  - 任务类型: {config.task_type}")
        
        model = get_model(config)
        print(f"\n✓ LSTM模型创建成功: {model.__class__.__name__}")
        
        total, trainable = count_parameters(model)
        print(f"  - 总参数数: {total:,}")
        print(f"  - 可训练参数: {trainable:,}")
        
        x = torch.randn(8, 50, 8)
        with torch.no_grad():
            output = model(x)
        print(f"\n✓ 前向传播成功")
        print(f"  - 输入形状: {x.shape}")
        
        if isinstance(output, tuple):
            out_tensor, hidden_state = output
            print(f"  - 输出形状: {out_tensor.shape}")
            assert out_tensor.shape[0] == 8 and out_tensor.shape[2] == 4, f"输出形状错误！期望(8, *, 4)，实际{out_tensor.shape}"
        else:
            print(f"  - 输出形状: {output.shape}")
            assert output.shape[0] == 8 and output.shape[2] == 4, f"输出形状错误！期望(8, *, 4)，实际{output.shape}"
        print(f"  - 输出维度验证: ✓ 通过")
        
        return True
        
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_lstm_regression():
    """测试LSTM Regression模型（回归风格输出）"""
    print_section("测试10：LSTM模型（Regression任务）")
    
    try:
        config = LSTMConfig(
            input_size=6,
            num_classes=2,
            task_type="regression",
            predict_seq_len=4,
            hidden_size=48,
            num_layers=2
        )
        
        print(f"✓ LSTM Regression Config创建成功")
        print(f"  - 输入特征维度: {config.input_size}")
        print(f"  - 输出特征维度: {config.num_classes}")
        print(f"  - 预测步数: {config.predict_seq_len}")
        print(f"  - 隐藏层维度: {config.hidden_size}")
        print(f"  - 任务类型: {config.task_type}")
        
        model = get_model(config)
        print(f"\n✓ LSTM模型创建成功: {model.__class__.__name__}")
        
        total, trainable = count_parameters(model)
        print(f"  - 总参数数: {total:,}")
        print(f"  - 可训练参数: {trainable:,}")
        
        x = torch.randn(12, 100, 6)
        with torch.no_grad():
            output = model(x)
        print(f"\n✓ 前向传播成功")
        print(f"  - 输入形状: {x.shape}")
        print(f"  - 输出形状: {output.shape}")
        
        assert output.shape == (12, 4, 2), f"输出形状错误！期望(12, 4, 2)，实际{output.shape}"
        print(f"  - 输出维度验证: ✓ 通过")
        
        return True
        
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_cnn_image_classification():
    """测试CNN图像分类模型"""
    print_section("测试11：CNN模型（图像分类）")
    
    try:
        config = CNNConfig(
            in_channels=3,
            input_type="image",
            task_type="classification",
            input_size=(32, 32),
            num_classes=10,
            conv=ConvConfig(conv_channels=[32, 64, 128]),
            fc=FCConfig(fc_hidden_dims=[256, 128]),
            dropout=DropoutConfig(enable=True, prob=0.3)
        )
        
        print(f"✓ CNN图像分类 Config创建成功")
        print(f"  - 输入通道数: {config.in_channels}")
        print(f"  - 输入尺寸: {config.input_size}")
        print(f"  - 输出类别数: {config.num_classes}")
        print(f"  - 卷积通道: {config.conv.conv_channels}")
        print(f"  - 任务类型: {config.task_type}")
        
        model = get_model(config)
        print(f"\n✓ CNN模型创建成功: {model.__class__.__name__}")
        
        total, trainable = count_parameters(model)
        print(f"  - 总参数数: {total:,}")
        print(f"  - 可训练参数: {trainable:,}")
        
        x = torch.randn(8, 3, 32, 32)
        with torch.no_grad():
            output = model(x)
        print(f"\n✓ 前向传播成功")
        print(f"  - 输入形状: {x.shape}")
        print(f"  - 输出形状: {output.shape}")
        
        assert output.shape == (8, 10), f"输出形状错误！期望(8, 10)，实际{output.shape}"
        print(f"  - 输出维度验证: ✓ 通过")
        
        return True
        
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_cnn_timeseries_classification():
    """测试CNN时序分类模型"""
    print_section("测试12：CNN模型（时序分类）")
    
    try:
        config = CNNConfig(
            in_channels=2,
            input_type="timeseries",
            task_type="classification",
            input_size=128,
            num_classes=5,
            conv=ConvConfig(conv_channels=[16, 32], kernel_size=3, padding=1),
            pool=PoolConfig(pool_type="avg", pool_kernel_size=2, pool_stride=2),
            fc=FCConfig(fc_hidden_dims=[64]),
            activation_type="leaky_relu"
        )
        
        print(f"✓ CNN时序分类 Config创建成功")
        print(f"  - 输入通道数: {config.in_channels}")
        print(f"  - 输入长度: {config.input_size}")
        print(f"  - 输出类别数: {config.num_classes}")
        print(f"  - 卷积通道: {config.conv.conv_channels}")
        print(f"  - 激活函数: {config.activation_type}")
        
        model = get_model(config)
        print(f"\n✓ CNN模型创建成功: {model.__class__.__name__}")
        
        total, trainable = count_parameters(model)
        print(f"  - 总参数数: {total:,}")
        print(f"  - 可训练参数: {trainable:,}")
        
        x = torch.randn(16, 2, 128)
        with torch.no_grad():
            output = model(x)
        print(f"\n✓ 前向传播成功")
        print(f"  - 输入形状: {x.shape}")
        print(f"  - 输出形状: {output.shape}")
        
        assert output.shape == (16, 5), f"输出形状错误！期望(16, 5)，实际{output.shape}"
        print(f"  - 输出维度验证: ✓ 通过")
        
        return True
        
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_cnn_regression():
    """测试CNN回归模型"""
    print_section("测试13：CNN模型（回归任务）")
    
    try:
        config = CNNConfig(
            in_channels=1,
            input_type="timeseries",
            task_type="regression",
            input_size=256,
            num_outputs=2,
            conv=ConvConfig(conv_channels=[8, 16], kernel_size=5, padding=2),
            pool=PoolConfig(pool_type="max", pool_kernel_size=4, pool_stride=4),
            fc=FCConfig(fc_hidden_dims=[32]),
            dropout=DropoutConfig(enable=False)
        )
        
        print(f"✓ CNN回归 Config创建成功")
        print(f"  - 输入通道数: {config.in_channels}")
        print(f"  - 输入长度: {config.input_size}")
        print(f"  - 输出维度: {config.num_outputs}")
        print(f"  - 卷积通道: {config.conv.conv_channels}")
        print(f"  - 任务类型: {config.task_type}")
        
        model = get_model(config)
        print(f"\n✓ CNN模型创建成功: {model.__class__.__name__}")
        
        total, trainable = count_parameters(model)
        print(f"  - 总参数数: {total:,}")
        print(f"  - 可训练参数: {trainable:,}")
        
        x = torch.randn(10, 1, 256)
        with torch.no_grad():
            output = model(x)
        print(f"\n✓ 前向传播成功")
        print(f"  - 输入形状: {x.shape}")
        print(f"  - 输出形状: {output.shape}")
        
        assert output.shape == (10, 2), f"输出形状错误！期望(10, 2)，实际{output.shape}"
        print(f"  - 输出维度验证: ✓ 通过")
        
        return True
        
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_rnn_classification():
    """测试RNN分类模型"""
    print_section("测试14：RNN模型（分类任务）")
    
    try:
        config = RNNConfig(
            input_size=7,
            num_output_dim=4,
            task_type="classification",
            hidden_size=64,
            num_layers=2,
            bidirectional=False,
            nonlinearity="tanh"
        )
        
        print(f"✓ RNN分类 Config创建成功")
        print(f"  - 输入特征维度: {config.input_size}")
        print(f"  - 输出类别数: {config.num_output_dim}")
        print(f"  - 隐藏层维度: {config.hidden_size}")
        print(f"  - RNN层数: {config.num_layers}")
        print(f"  - 激活函数: {config.nonlinearity}")
        
        model = get_model(config)
        print(f"\n✓ RNN模型创建成功: {model.__class__.__name__}")
        
        total, trainable = count_parameters(model)
        print(f"  - 总参数数: {total:,}")
        print(f"  - 可训练参数: {trainable:,}")
        
        x = torch.randn(12, 80, 7)
        with torch.no_grad():
            output = model(x)
        print(f"\n✓ 前向传播成功")
        print(f"  - 输入形状: {x.shape}")
        print(f"  - 输出形状: {output.shape}")
        
        assert output.shape == (12, 4), f"输出形状错误！期望(12, 4)，实际{output.shape}"
        print(f"  - 输出维度验证: ✓ 通过")
        
        return True
        
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_rnn_seq2seq():
    """测试RNN Seq2Seq模型"""
    print_section("测试15：RNN模型（Seq2Seq任务）")
    
    try:
        config = RNNConfig(
            input_size=6,
            num_output_dim=3,
            task_type="seq2seq",
            hidden_size=48,
            num_layers=1,
            bidirectional=True,
            nonlinearity="relu"
        )
        
        print(f"✓ RNN Seq2Seq Config创建成功")
        print(f"  - 输入特征维度: {config.input_size}")
        print(f"  - 输出特征维度: {config.num_output_dim}")
        print(f"  - 隐藏层维度: {config.hidden_size}")
        print(f"  - 双向RNN: {config.bidirectional}")
        print(f"  - 任务类型: {config.task_type}")
        
        model = get_model(config)
        print(f"\n✓ RNN模型创建成功: {model.__class__.__name__}")
        
        total, trainable = count_parameters(model)
        print(f"  - 总参数数: {total:,}")
        print(f"  - 可训练参数: {trainable:,}")
        
        x = torch.randn(10, 60, 6)
        with torch.no_grad():
            output = model(x)
        print(f"\n✓ 前向传播成功")
        print(f"  - 输入形状: {x.shape}")
        
        if isinstance(output, tuple):
            out_tensor, h_n = output
            print(f"  - 输出形状: {out_tensor.shape}")
            assert out_tensor.shape[0] == 10 and out_tensor.shape[2] == 3, f"输出形状错误！期望(10, *, 3)，实际{out_tensor.shape}"
        else:
            print(f"  - 输出形状: {output.shape}")
            assert output.shape[0] == 10 and output.shape[2] == 3, f"输出形状错误！期望(10, *, 3)，实际{output.shape}"
        print(f"  - 输出维度验证: ✓ 通过")
        
        return True
        
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_rnn_Regression():
    """测试RNN回归模型（Regression任务）"""
    print_section("测试16：RNN模型（回归/Regression任务）")
    
    try:
        config = RNNConfig(
            input_size=5,
            num_output_dim=2,
            task_type="regression",
            predict_seq_len=3,
            hidden_size=32,
            num_layers=1,
            nonlinearity="tanh"
        )
        
        print(f"✓ RNN回归 Config创建成功")
        print(f"  - 输入特征维度: {config.input_size}")
        print(f"  - 输出特征维度: {config.num_output_dim}")
        print(f"  - 预测步数: {config.predict_seq_len}")
        print(f"  - 隐藏层维度: {config.hidden_size}")
        print(f"  - 任务类型: {config.task_type}")
        
        model = get_model(config)
        print(f"\n✓ RNN模型创建成功: {model.__class__.__name__}")
        
        total, trainable = count_parameters(model)
        print(f"  - 总参数数: {total:,}")
        print(f"  - 可训练参数: {trainable:,}")
        
        x = torch.randn(14, 50, 5)
        with torch.no_grad():
            output = model(x)
        print(f"\n✓ 前向传播成功")
        print(f"  - 输入形状: {x.shape}")
        print(f"  - 输出形状: {output.shape}")
        
        assert output.shape == (14, 3, 2), f"输出形状错误！期望(14, 3, 2)，实际{output.shape}"
        print(f"  - 输出维度验证: ✓ 通过")
        
        return True
        
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """运行所有测试"""
    print("\n")
    print("*" * 70)
    print("  模型工厂完整测试套件")
    print("*" * 70)
    
    results = []
    results.append(("MLP模型工厂", test_mlp_model()))
    results.append(("U-Net分割任务", test_unet_model()))
    results.append(("U-Net分类任务", test_unet_classification()))
    results.append(("U-Net回归任务", test_unet_regression()))
    results.append(("U-Net时序输入", test_unet_timeseries()))
    results.append(("LSTM分类任务", test_lstm_classification()))
    results.append(("LSTM Seq2Seq", test_lstm_seq2seq()))
    results.append(("LSTM Regression", test_lstm_regression()))
    results.append(("CNN图像分类", test_cnn_image_classification()))
    results.append(("CNN时序分类", test_cnn_timeseries_classification()))
    results.append(("CNN回归任务", test_cnn_regression()))
    results.append(("RNN分类任务", test_rnn_classification()))
    results.append(("RNN Seq2Seq", test_rnn_seq2seq()))
    results.append(("RNN Regression", test_rnn_Regression()))
    results.append(("Config序列化", test_config_to_dict_to_model()))
    results.append(("异常处理", test_invalid_config()))
    
    # 总结
    print_section("测试总结")
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"{name:20s}: {status}")
    
    print(f"\n总体结果: {passed}/{total} 测试通过")
    
    if passed == total:
        print("🎉 所有测试通过！模型工厂可正常使用。")
        return 0
    else:
        print("⚠️  部分测试失败，请检查错误信息。")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
