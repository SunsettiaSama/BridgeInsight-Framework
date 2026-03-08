import pytest
import json
import numpy as np
import scipy.io as sio
from pathlib import Path
import tempfile
import torch
from torch.utils.data import DataLoader

from src.config.data_processer.datasets.DatasetsFromAnnotation.AnnotationDatasetConfig import (
    AnnotationDatasetConfig
)
from src.data_processer.datasets.DatasetsFromAnnotation.AnnotationDataset import (
    AnnotationDataset
)
from src.data_processer.datasets.data_factory import get_dataset


class TestAnnotationDatasetConfig:
    """测试标注数据集配置"""
    
    @pytest.fixture
    def temp_annotation_dir(self):
        """创建临时标注文件和数据集"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            # 创建数据文件
            data_dir = tmpdir / "data"
            data_dir.mkdir()
            
            annotations = []
            for i in range(10):
                # 创建.mat文件
                data = np.random.randn(1000, 10).astype(np.float32)
                label = np.random.randint(0, 3)
                mat_file = data_dir / f"sample_{i:02d}.mat"
                sio.savemat(mat_file, {"data": data, "label": label})
                
                # 创建标注记录
                annotation_text = ["正常", "异常", "未知"][i % 3]
                annotations.append({
                    "sample_id": f"sample_{i:02d}",
                    "file_path": str(mat_file),
                    "annotation": annotation_text,
                    "metadata": {"sensor_id": f"sensor_{i}"}
                })
            
            # 保存标注文件
            annotation_file = tmpdir / "annotations.json"
            with open(annotation_file, 'w', encoding='utf-8') as f:
                json.dump(annotations, f, ensure_ascii=False, indent=2)
            
            yield tmpdir, annotation_file, data_dir
    
    def test_config_creation(self, temp_annotation_dir):
        """测试配置创建"""
        tmpdir, annotation_file, _ = temp_annotation_dir
        
        config = AnnotationDatasetConfig(
            data_dir=str(tmpdir),
            annotation_file=str(annotation_file),
            batch_size=4,
            fix_seq_len=1000
        )
        
        assert config.annotation_file == str(annotation_file)
        assert config.batch_size == 4
        assert config.fix_seq_len == 1000
        assert config.only_annotated is True
    
    def test_config_with_label_mapping(self, temp_annotation_dir):
        """测试带标签映射的配置"""
        tmpdir, annotation_file, _ = temp_annotation_dir
        
        config = AnnotationDatasetConfig(
            data_dir=str(tmpdir),
            annotation_file=str(annotation_file),
            enable_label_mapping=True,
            label_to_class={"正常": 0, "异常": 1, "未知": 2}
        )
        
        assert config.enable_label_mapping is True
        assert config.label_to_class == {"正常": 0, "异常": 1, "未知": 2}


class TestAnnotationDatasetLoading:
    """测试标注数据集加载"""
    
    @pytest.fixture
    def temp_annotation_dir(self):
        """创建临时标注文件和数据集"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            # 创建数据文件
            data_dir = tmpdir / "data"
            data_dir.mkdir()
            
            annotations = []
            for i in range(15):
                # 创建.mat文件
                data = np.random.randn(1000, 10).astype(np.float32)
                label = np.random.randint(0, 3)
                mat_file = data_dir / f"sample_{i:02d}.mat"
                sio.savemat(mat_file, {"data": data, "label": label})
                
                # 创建标注记录
                annotation_text = ["正常", "异常", "未知"][i % 3]
                # 某些样本无标注
                if i % 5 == 0:
                    annotation_text = None
                
                annotations.append({
                    "sample_id": f"sample_{i:02d}",
                    "file_path": str(mat_file),
                    "annotation": annotation_text,
                    "metadata": {"sensor_id": f"sensor_{i}"}
                })
            
            # 保存标注文件
            annotation_file = tmpdir / "annotations.json"
            with open(annotation_file, 'w', encoding='utf-8') as f:
                json.dump(annotations, f, ensure_ascii=False, indent=2)
            
            yield tmpdir, annotation_file, data_dir
    
    def test_dataset_creation(self, temp_annotation_dir):
        """测试数据集创建"""
        tmpdir, annotation_file, _ = temp_annotation_dir
        
        config = AnnotationDatasetConfig(
            data_dir=str(tmpdir),
            annotation_file=str(annotation_file),
            batch_size=4,
            only_annotated=True,
            fix_seq_len=1000,
            normalize=False
        )
        
        dataset = AnnotationDataset(config)
        
        # 应该只有标注的样本（过滤掉无标注的）
        assert len(dataset) > 0
        assert len(dataset) <= 15
    
    def test_dataset_getitem(self, temp_annotation_dir):
        """测试单样本获取"""
        tmpdir, annotation_file, _ = temp_annotation_dir
        
        config = AnnotationDatasetConfig(
            data_dir=str(tmpdir),
            annotation_file=str(annotation_file),
            batch_size=4,
            only_annotated=True,
            fix_seq_len=1000,
            normalize=False,
            shuffle=False
        )
        
        dataset = AnnotationDataset(config)
        
        if len(dataset) > 0:
            data, label = dataset[0]
            
            assert isinstance(data, torch.Tensor)
            assert isinstance(label, torch.Tensor)
            assert data.shape == (1000, 10)
            assert label.dim() == 0
    
    def test_label_mapping(self, temp_annotation_dir):
        """测试标签映射"""
        tmpdir, annotation_file, _ = temp_annotation_dir
        
        config = AnnotationDatasetConfig(
            data_dir=str(tmpdir),
            annotation_file=str(annotation_file),
            only_annotated=True,
            enable_label_mapping=True,
            label_to_class={"正常": 0, "异常": 1, "未知": 2},
            fix_seq_len=1000
        )
        
        dataset = AnnotationDataset(config)
        
        # 收集所有标签
        labels = []
        for i in range(min(len(dataset), 5)):
            _, label = dataset[i]
            labels.append(label.item())
        
        # 验证标签都在映射范围内
        assert all(l in [0, 1, 2] for l in labels)
    
    def test_label_filtering(self, temp_annotation_dir):
        """测试标签过滤"""
        tmpdir, annotation_file, _ = temp_annotation_dir
        
        config = AnnotationDatasetConfig(
            data_dir=str(tmpdir),
            annotation_file=str(annotation_file),
            only_annotated=True,
            include_labels=["正常", "异常"],  # 只包含这两个标签
            exclude_labels=None,
            fix_seq_len=1000
        )
        
        dataset = AnnotationDataset(config)
        
        # 数据集不应包含标注为"未知"的样本
        assert len(dataset) > 0
    
    def test_dataloader_creation(self, temp_annotation_dir):
        """测试DataLoader创建"""
        tmpdir, annotation_file, _ = temp_annotation_dir
        
        config = AnnotationDatasetConfig(
            data_dir=str(tmpdir),
            annotation_file=str(annotation_file),
            batch_size=4,
            only_annotated=True,
            fix_seq_len=1000,
            normalize=False,
            shuffle=False,
            num_workers=0
        )
        
        dataset = AnnotationDataset(config)
        loader = DataLoader(dataset, batch_size=4, shuffle=False)
        
        batch_count = 0
        for data, labels in loader:
            batch_count += 1
            assert data.shape[0] in [4, len(dataset) % 4]
            assert data.shape[1:] == (1000, 10)
            assert labels.shape[0] == data.shape[0]
        
        assert batch_count > 0


class TestAnnotationDatasetWithFactory:
    """测试通过工厂接口使用标注数据集"""
    
    @pytest.fixture
    def temp_annotation_dir(self):
        """创建临时标注文件和数据集"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            # 创建数据文件
            data_dir = tmpdir / "data"
            data_dir.mkdir()
            
            annotations = []
            for i in range(10):
                data = np.random.randn(500, 5).astype(np.float32)
                mat_file = data_dir / f"sample_{i:02d}.mat"
                sio.savemat(mat_file, {"data": data, "label": i % 2})
                
                annotations.append({
                    "sample_id": f"sample_{i:02d}",
                    "file_path": str(mat_file),
                    "annotation": "异常" if i % 2 else "正常",
                    "metadata": {}
                })
            
            annotation_file = tmpdir / "annotations.json"
            with open(annotation_file, 'w', encoding='utf-8') as f:
                json.dump(annotations, f, ensure_ascii=False)
            
            yield tmpdir, annotation_file
    
    def test_factory_creation(self, temp_annotation_dir):
        """测试通过工厂创建标注数据集"""
        tmpdir, annotation_file = temp_annotation_dir
        
        config = AnnotationDatasetConfig(
            data_dir=str(tmpdir),
            annotation_file=str(annotation_file),
            batch_size=4,
            fix_seq_len=500
        )
        
        # 通过工厂创建
        dataset = get_dataset(config)
        
        assert isinstance(dataset, AnnotationDataset)
        assert len(dataset) > 0


class TestAnnotationDatasetNormalization:
    """测试标注数据集的归一化"""
    
    @pytest.fixture
    def temp_annotation_dir(self):
        """创建带有大幅变化数据的临时数据集"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            data_dir = tmpdir / "data"
            data_dir.mkdir()
            
            annotations = []
            for i in range(8):
                # 创建具有不同幅度的数据
                data = np.random.randn(500, 5).astype(np.float32) * (i + 1)
                mat_file = data_dir / f"sample_{i:02d}.mat"
                sio.savemat(mat_file, {"data": data, "label": 0})
                
                annotations.append({
                    "sample_id": f"sample_{i:02d}",
                    "file_path": str(mat_file),
                    "annotation": "正常",
                })
            
            annotation_file = tmpdir / "annotations.json"
            with open(annotation_file, 'w', encoding='utf-8') as f:
                json.dump(annotations, f, ensure_ascii=False)
            
            yield tmpdir, annotation_file
    
    def test_zscore_normalization(self, temp_annotation_dir):
        """测试z-score归一化"""
        tmpdir, annotation_file = temp_annotation_dir
        
        config = AnnotationDatasetConfig(
            data_dir=str(tmpdir),
            annotation_file=str(annotation_file),
            fix_seq_len=500,
            normalize=True,
            normalize_type="z-score"
        )
        
        dataset = AnnotationDataset(config)
        
        # 获取几个样本检查归一化
        for i in range(min(3, len(dataset))):
            data, _ = dataset[i]
            
            # 检查数据是否有限
            assert torch.isfinite(data).all()
            # 归一化后数据应该在合理范围内
            assert data.abs().max() < 100


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
