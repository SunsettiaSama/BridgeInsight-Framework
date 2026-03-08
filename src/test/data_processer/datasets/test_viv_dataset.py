import pytest
import torch
import numpy as np
from pathlib import Path
import tempfile
import scipy.io as sio
from src.config.data_processer.datasets.VIV2NumClassification.VIVTimeseriesClassificationDataset import (
    VIVTimeSeriesClassificationDatasetConfig
)
from src.data_processer.datasets.data_factory import get_dataset
from torch.utils.data import DataLoader


class TestVIVDatasetLoading:
    """测试VIV数据集加载功能"""
    
    @pytest.fixture
    def temp_dataset_dir(self):
        """创建临时测试数据集"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            # 创建train/val/test子目录
            (tmpdir / "train").mkdir()
            (tmpdir / "val").mkdir()
            (tmpdir / "test").mkdir()
            
            # 创建测试.mat文件
            for split, num_samples in [("train", 10), ("val", 3), ("test", 2)]:
                split_dir = tmpdir / split
                for i in range(num_samples):
                    data = np.random.randn(1000, 10).astype(np.float32)
                    label = np.random.randint(0, 10)
                    sio.savemat(
                        split_dir / f"sample_{i}.mat",
                        {"data": data, "label": label}
                    )
            
            yield tmpdir
    
    def test_config_creation(self):
        """测试配置类创建"""
        config = VIVTimeSeriesClassificationDatasetConfig(
            data_dir="./data",
            batch_size=32,
            fix_seq_len=1000,
            normalize=True
        )
        
        assert config.data_dir == "./data"
        assert config.batch_size == 32
        assert config.fix_seq_len == 1000
        assert config.normalize is True
    
    def test_dataset_creation_with_official_split(self, temp_dataset_dir):
        """测试使用官方划分创建数据集"""
        config = VIVTimeSeriesClassificationDatasetConfig(
            data_dir=str(temp_dataset_dir),
            batch_size=4,
            use_official_split=True,
            fix_seq_len=1000,
            normalize=False,
            shuffle=False
        )
        
        # 通过工厂创建数据集
        dataset = get_dataset(config)
        
        # 验证数据集创建成功
        assert dataset is not None
        
        # 获取各集合
        train_dataset = dataset.get_train_dataset()
        val_dataset = dataset.get_val_dataset()
        test_dataset = dataset.get_test_dataset()
        
        assert len(train_dataset) == 10
        assert len(val_dataset) == 3
        assert len(test_dataset) == 2
    
    def test_dataset_getitem(self, temp_dataset_dir):
        """测试数据集单样本获取"""
        config = VIVTimeSeriesClassificationDatasetConfig(
            data_dir=str(temp_dataset_dir),
            batch_size=4,
            use_official_split=True,
            fix_seq_len=1000,
            normalize=False
        )
        
        dataset = get_dataset(config)
        train_dataset = dataset.get_train_dataset()
        
        # 获取单样本
        data, label = train_dataset[0]
        
        # 验证数据格式
        assert isinstance(data, torch.Tensor)
        assert isinstance(label, torch.Tensor)
        assert data.shape == (1000, 10)
        assert label.dim() == 0  # 标量
    
    def test_dataloader_creation(self, temp_dataset_dir):
        """测试DataLoader创建"""
        config = VIVTimeSeriesClassificationDatasetConfig(
            data_dir=str(temp_dataset_dir),
            batch_size=4,
            use_official_split=True,
            fix_seq_len=1000,
            normalize=False,
            shuffle=True,
            num_workers=0
        )
        
        dataset = get_dataset(config)
        train_dataset = dataset.get_train_dataset()
        train_loader = DataLoader(train_dataset, batch_size=4, shuffle=False)
        
        # 验证DataLoader
        batch_count = 0
        for data, labels in train_loader:
            batch_count += 1
            assert data.shape[0] in [4, 10 % 4]  # batch_size或剩余样本
            assert data.shape[1:] == (1000, 10)
            assert labels.shape[0] == data.shape[0]
        
        assert batch_count > 0
    
    def test_sequence_normalization(self, temp_dataset_dir):
        """测试序列归一化"""
        config = VIVTimeSeriesClassificationDatasetConfig(
            data_dir=str(temp_dataset_dir),
            batch_size=4,
            use_official_split=True,
            fix_seq_len=1000,
            normalize=True,
            normalize_type="z-score",
            shuffle=False,
            num_workers=0
        )
        
        dataset = get_dataset(config)
        train_dataset = dataset.get_train_dataset()
        
        # 获取样本并检查值范围
        for i in range(min(3, len(train_dataset))):
            data, _ = train_dataset[i]
            # 归一化后数据应该在合理范围内
            assert torch.isfinite(data).all()


class TestVIVDatasetTraining:
    """测试VIV数据集在训练流程中的应用"""
    
    @pytest.fixture
    def temp_dataset_dir(self):
        """创建临时测试数据集"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            (tmpdir / "train").mkdir()
            (tmpdir / "val").mkdir()
            
            # 创建足够的训练数据
            for split, num_samples in [("train", 20), ("val", 5)]:
                split_dir = tmpdir / split
                for i in range(num_samples):
                    data = np.random.randn(1000, 10).astype(np.float32)
                    label = np.random.randint(0, 3)  # 3分类
                    sio.savemat(
                        split_dir / f"sample_{i}.mat",
                        {"data": data, "label": label}
                    )
            
            yield tmpdir
    
    def test_lstm_training_loop(self, temp_dataset_dir):
        """测试LSTM训练循环"""
        import torch.nn as nn
        
        config = VIVTimeSeriesClassificationDatasetConfig(
            data_dir=str(temp_dataset_dir),
            batch_size=4,
            use_official_split=True,
            fix_seq_len=1000,
            normalize=True,
            batch_first=True,
            shuffle=True,
            num_workers=0
        )
        
        dataset = get_dataset(config)
        train_loader = DataLoader(
            dataset.get_train_dataset(),
            batch_size=4,
            shuffle=True
        )
        val_loader = DataLoader(
            dataset.get_val_dataset(),
            batch_size=4,
            shuffle=False
        )
        
        # 创建简单的LSTM模型
        model = nn.LSTM(
            input_size=10,
            hidden_size=32,
            num_layers=1,
            batch_first=True
        )
        lstm_output = nn.Linear(32, 3)  # 3分类
        criterion = nn.CrossEntropyLoss()
        
        # 模拟训练步骤
        model.train()
        for batch_idx, (data, labels) in enumerate(train_loader):
            if batch_idx >= 2:  # 仅测试2个batch
                break
            
            # LSTM前向传播
            lstm_out, _ = model(data)
            last_output = lstm_out[:, -1, :]  # 取最后一步
            logits = lstm_output(last_output)
            loss = criterion(logits, labels)
            
            # 验证loss值
            assert torch.isfinite(loss)
            assert loss.item() > 0
    
    def test_batch_processing_consistency(self, temp_dataset_dir):
        """测试批处理的一致性"""
        config = VIVTimeSeriesClassificationDatasetConfig(
            data_dir=str(temp_dataset_dir),
            batch_size=4,
            use_official_split=True,
            fix_seq_len=1000,
            normalize=False,
            shuffle=False,
            num_workers=0
        )
        
        dataset = get_dataset(config)
        train_loader = DataLoader(
            dataset.get_train_dataset(),
            batch_size=4,
            shuffle=False
        )
        
        # 连续两次迭代应该得到相同的数据（不打乱的情况）
        batches_1 = []
        batches_2 = []
        
        for data, labels in train_loader:
            batches_1.append((data.clone(), labels.clone()))
        
        for data, labels in train_loader:
            batches_2.append((data.clone(), labels.clone()))
        
        # 验证两次迭代的数据一致
        assert len(batches_1) == len(batches_2)
        for (d1, l1), (d2, l2) in zip(batches_1, batches_2):
            assert torch.allclose(d1, d2)
            assert torch.allclose(l1.float(), l2.float())


class TestVIV2NumClasses:
    """测试VIV二分类数据集"""
    
    @pytest.fixture
    def temp_binary_dataset_dir(self):
        """创建包含多类标签的临时数据集"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            (tmpdir / "train").mkdir()
            
            # 创建含有标签0、1、2的数据
            for i in range(10):
                data = np.random.randn(1000, 10).astype(np.float32)
                label = i % 3  # 标签：0, 1, 2, 0, 1, 2, ...
                sio.savemat(
                    tmpdir / "train" / f"sample_{i}.mat",
                    {"data": data, "label": label}
                )
            
            yield tmpdir
    
    def test_binary_label_filtering(self, temp_binary_dataset_dir):
        """测试二分类标签过滤（标签1被删除，2转为1）"""
        from src.data_processer.datasets.VIV2NumClassification.VIVTimeseriesClassificationDataset_2_num_classes import (
            VIVTimeSeriesClassificationDataset2NumClasses
        )
        
        config = VIVTimeSeriesClassificationDatasetConfig(
            data_dir=str(temp_binary_dataset_dir),
            batch_size=4,
            use_official_split=False,
            shuffle=False,
            num_workers=0
        )
        
        dataset = VIVTimeSeriesClassificationDataset2NumClasses(config)
        
        # 收集所有标签
        all_labels = []
        for i in range(len(dataset)):
            _, label = dataset[i]
            all_labels.append(label.item())
        
        # 验证：不存在标签1
        assert 1 not in all_labels
        # 验证：只有标签0和1（转换后）
        unique_labels = set(all_labels)
        assert unique_labels.issubset({0, 1})


class TestDatasetConfig:
    """测试数据集配置校验"""
    
    def test_config_merge_dict(self):
        """测试配置字典合并"""
        config = VIVTimeSeriesClassificationDatasetConfig(
            data_dir="./data",
            batch_size=8
        )
        
        config.merge_dict({
            "batch_size": 16,
            "shuffle": False
        })
        
        assert config.batch_size == 16
        assert config.shuffle is False
    
    def test_split_ratio_validation(self):
        """测试划分比例校验"""
        # 有效的配置
        config = VIVTimeSeriesClassificationDatasetConfig(
            data_dir="./data",
            split_ratio=0.7,
            test_ratio=0.15
        )
        
        # 无效的配置应该抛出异常
        with pytest.raises(ValueError):
            VIVTimeSeriesClassificationDatasetConfig(
                data_dir="./data",
                split_ratio=0.8,
                test_ratio=0.3  # 超过1.0
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
