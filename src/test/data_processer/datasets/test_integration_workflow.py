"""
完整的数据集工作流测试

测试VIV多分类、VIV二分类和标注数据集的完整工作流：
1. 配置创建
2. 数据集加载
3. DataLoader迭代
4. 模型训练循环
"""

import pytest
import json
import numpy as np
import scipy.io as sio
from pathlib import Path
import tempfile
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.config.data_processer.datasets.VIV2NumClassification.VIVTimeseriesClassificationDataset import (
    VIVTimeSeriesClassificationDatasetConfig
)
from src.data_processer.datasets.data_factory import get_dataset
from src.config.data_processer.datasets.DatasetsFromAnnotation.AnnotationDatasetConfig import (
    AnnotationDatasetConfig
)


class TestCompleteWorkflow:
    """完整工作流测试"""
    
    @pytest.fixture
    def viv_dataset_dir(self):
        """创建VIV数据集"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            for split in ["train", "val", "test"]:
                split_dir = tmpdir / split
                split_dir.mkdir()
                
                num_samples = {"train": 20, "val": 5, "test": 3}[split]
                for i in range(num_samples):
                    data = np.random.randn(1000, 10).astype(np.float32)
                    label = np.random.randint(0, 5)
                    sio.savemat(
                        split_dir / f"sample_{i}.mat",
                        {"data": data, "label": label}
                    )
            
            yield tmpdir
    
    @pytest.fixture
    def annotation_dataset_dir(self):
        """创建标注数据集"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            data_dir = tmpdir / "data"
            data_dir.mkdir()
            
            annotations = []
            for i in range(20):
                data = np.random.randn(800, 8).astype(np.float32)
                mat_file = data_dir / f"sample_{i:02d}.mat"
                sio.savemat(mat_file, {"data": data, "label": 0})
                
                annotations.append({
                    "sample_id": f"sample_{i:02d}",
                    "file_path": str(mat_file),
                    "annotation": "异常" if i % 2 else "正常"
                })
            
            annotation_file = tmpdir / "annotations.json"
            with open(annotation_file, 'w', encoding='utf-8') as f:
                json.dump(annotations, f, ensure_ascii=False)
            
            yield tmpdir, annotation_file
    
    def test_viv_complete_workflow(self, viv_dataset_dir):
        """测试VIV数据集完整工作流"""
        # Step 1: 创建配置
        config = VIVTimeSeriesClassificationDatasetConfig(
            data_dir=str(viv_dataset_dir),
            batch_size=8,
            use_official_split=True,
            fix_seq_len=1000,
            normalize=True,
            normalize_type="z-score",
            shuffle=False,
            num_workers=0
        )
        
        # Step 2: 创建数据集
        dataset = get_dataset(config)
        train_dataset = dataset.get_train_dataset()
        val_dataset = dataset.get_val_dataset()
        test_dataset = dataset.get_test_dataset()
        
        # Step 3: 创建DataLoader
        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
        
        # Step 4: 验证数据加载
        train_batches = list(train_loader)
        assert len(train_batches) > 0
        
        first_batch_data, first_batch_labels = train_batches[0]
        assert first_batch_data.shape[1:] == (1000, 10)
        assert first_batch_labels.shape[0] == first_batch_data.shape[0]
        
        # Step 5: 模拟训练循环
        model = nn.Sequential(
            nn.Linear(1000 * 10, 128),
            nn.ReLU(),
            nn.Linear(128, 5)  # 5分类
        )
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        model.train()
        for batch_idx, (data, labels) in enumerate(train_loader):
            if batch_idx >= 1:  # 仅测试1个batch
                break
            
            # 展平数据用于全连接网络
            data_flat = data.view(data.shape[0], -1)
            logits = model(data_flat)
            loss = criterion(logits, labels)
            
            assert torch.isfinite(loss)
            assert loss.item() > 0
        
        # Step 6: 验证测试集
        model.eval()
        with torch.no_grad():
            test_correct = 0
            test_total = 0
            for data, labels in test_loader:
                data_flat = data.view(data.shape[0], -1)
                outputs = model(data_flat)
                _, predicted = torch.max(outputs, 1)
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()
    
    def test_annotation_complete_workflow(self, annotation_dataset_dir):
        """测试标注数据集完整工作流"""
        tmpdir, annotation_file = annotation_dataset_dir
        
        # Step 1: 创建配置
        config = AnnotationDatasetConfig(
            data_dir=str(tmpdir),
            annotation_file=str(annotation_file),
            batch_size=8,
            enable_label_mapping=True,
            label_to_class={"正常": 0, "异常": 1},
            fix_seq_len=800,
            normalize=True,
            normalize_type="z-score",
            shuffle=False,
            num_workers=0
        )
        
        # Step 2: 创建数据集
        dataset = get_dataset(config)
        
        # Step 3: 创建DataLoader
        loader = DataLoader(dataset, batch_size=8, shuffle=False)
        
        # Step 4: 验证数据加载
        batches = list(loader)
        assert len(batches) > 0
        
        first_batch_data, first_batch_labels = batches[0]
        assert first_batch_data.shape[1:] == (800, 8)
        
        # Step 5: 验证标签映射
        all_labels = []
        for _, labels in loader:
            all_labels.extend(labels.tolist())
        
        unique_labels = set(all_labels)
        assert unique_labels.issubset({0, 1})  # 只有两个类别
        
        # Step 6: 模拟LSTM训练
        lstm_model = nn.LSTM(
            input_size=8,
            hidden_size=32,
            batch_first=True
        )
        classifier = nn.Linear(32, 2)  # 二分类
        criterion = nn.CrossEntropyLoss()
        
        lstm_model.train()
        classifier.train()
        
        for data, labels in loader:
            lstm_out, (h_n, c_n) = lstm_model(data)
            last_hidden = h_n.squeeze(0)  # (batch, hidden_size)
            logits = classifier(last_hidden)
            loss = criterion(logits, labels)
            
            assert torch.isfinite(loss)
    
    def test_multi_epoch_training(self, viv_dataset_dir):
        """测试多轮训练"""
        config = VIVTimeSeriesClassificationDatasetConfig(
            data_dir=str(viv_dataset_dir),
            batch_size=4,
            use_official_split=True,
            fix_seq_len=1000,
            normalize=False,
            shuffle=True,
            num_workers=0
        )
        
        dataset = get_dataset(config)
        train_loader = DataLoader(dataset.get_train_dataset(), batch_size=4, shuffle=True)
        
        model = nn.Linear(1000 * 10, 5)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        
        losses = []
        
        # 模拟3个epoch
        for epoch in range(3):
            epoch_loss = 0.0
            batch_count = 0
            
            for data, labels in train_loader:
                data_flat = data.view(data.shape[0], -1)
                
                optimizer.zero_grad()
                outputs = model(data_flat)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                batch_count += 1
            
            avg_loss = epoch_loss / batch_count if batch_count > 0 else 0
            losses.append(avg_loss)
        
        # 验证losses都是有效的数值
        assert all(torch.isfinite(torch.tensor(l)) for l in losses)
    
    def test_batch_consistency_across_epochs(self, viv_dataset_dir):
        """测试跨epoch的数据一致性"""
        config = VIVTimeSeriesClassificationDatasetConfig(
            data_dir=str(viv_dataset_dir),
            batch_size=4,
            use_official_split=True,
            fix_seq_len=1000,
            normalize=False,
            shuffle=False,  # 不打乱
            num_workers=0
        )
        
        dataset = get_dataset(config)
        train_dataset = dataset.get_train_dataset()
        loader = DataLoader(train_dataset, batch_size=4, shuffle=False)
        
        # 收集第一个epoch的数据
        epoch1_batches = [(d.clone(), l.clone()) for d, l in loader]
        
        # 收集第二个epoch的数据
        epoch2_batches = [(d.clone(), l.clone()) for d, l in loader]
        
        # 验证一致性
        assert len(epoch1_batches) == len(epoch2_batches)
        
        for (d1, l1), (d2, l2) in zip(epoch1_batches, epoch2_batches):
            assert torch.allclose(d1, d2)
            assert torch.allclose(l1.float(), l2.float())


class TestErrorHandling:
    """测试错误处理"""
    
    def test_missing_annotation_file(self):
        """测试缺失的标注文件"""
        with pytest.raises(FileNotFoundError):
            config = AnnotationDatasetConfig(
                data_dir="./data",
                annotation_file="nonexistent.json"
            )
    
    def test_invalid_annotation_file_format(self):
        """测试无效的标注文件格式"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # 创建非JSON文件
            invalid_file = Path(tmpdir) / "annotations.txt"
            invalid_file.write_text("invalid data")
            
            with pytest.raises(ValueError):
                config = AnnotationDatasetConfig(
                    data_dir=tmpdir,
                    annotation_file=str(invalid_file)
                )
    
    def test_conflicting_label_filters(self):
        """测试冲突的标签过滤"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # 创建有效的标注文件
            annotation_file = Path(tmpdir) / "annotations.json"
            annotation_file.write_text(json.dumps([]))
            
            with pytest.raises(ValueError):
                config = AnnotationDatasetConfig(
                    data_dir=tmpdir,
                    annotation_file=str(annotation_file),
                    include_labels=["正常"],
                    exclude_labels=["正常"]  # 冲突
                )
    
    def test_missing_label_to_class_mapping(self):
        """测试缺失标签映射"""
        with tempfile.TemporaryDirectory() as tmpdir:
            annotation_file = Path(tmpdir) / "annotations.json"
            annotation_file.write_text(json.dumps([]))
            
            with pytest.raises(ValueError):
                config = AnnotationDatasetConfig(
                    data_dir=tmpdir,
                    annotation_file=str(annotation_file),
                    enable_label_mapping=True,
                    label_to_class=None  # 缺失
                )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
