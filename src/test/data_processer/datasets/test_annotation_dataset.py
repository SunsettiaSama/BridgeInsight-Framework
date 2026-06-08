"""
标注数据集简单测试
验证能否正确返回分类和回归样本
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

import json
import tempfile
from src.data_processer.datasets.AnnotationDataset.AnnotationDataset import AnnotationDataset


class SimpleConfig:
    """简单的配置类"""
    def __init__(self, annotation_file, data_dir, **kwargs):
        self.annotation_file = annotation_file
        self.data_dir = data_dir
        self.task_type = kwargs.get('task_type', 'classification')
        self.data_format = kwargs.get('data_format', 'npy')
        self.window_size = kwargs.get('window_size', 3000)
        self.cache_max_items = kwargs.get('cache_max_items', 100)
        self.fix_seq_len = kwargs.get('fix_seq_len', None)
        self.normalize = kwargs.get('normalize', False)
        self.normalize_type = kwargs.get('normalize_type', 'z-score')
        self.enable_label_mapping = kwargs.get('enable_label_mapping', True)
        self.label_to_class = kwargs.get('label_to_class', {"0": 0, "1": 1, "2": 2})
        self.unknown_label_class = -1
        self.only_annotated = True
        self.include_labels = None
        self.exclude_labels = None
        self.look_back = kwargs.get('look_back', 100)
        self.forecast_steps = kwargs.get('forecast_steps', 50)
        self.regression_stride = kwargs.get('regression_stride', 10)
        self.pad_mode = "zero"
        self.trunc_mode = "tail"
        self.sample_id_field = "sample_id"
        self.annotation_field = "annotation"
        self.data_path_field = "file_path"
        self.dataset_type = "custom"
        self.auto_split = False
        self.use_official_split = False
        self.split_ratio = 0.8
        self.test_ratio = None
        self.split_seed = 42
        self.batch_size = 8
        self.shuffle = False
        self.num_workers = 0
        self.pin_memory = False
        self.drop_last = False
        self.prefetch_factor = 2
        self.max_samples = None
        self.mean = [0.0]
        self.std = [1.0]
        self.resize_size = None
        self.keep_aspect_ratio = False
        self.train_aug = False
        self.hflip_prob = 0.0
        self.vflip_prob = 0.0
        self.rotate_angle = 0
        self.cache_in_memory = False
        self.cache_dir = None
        self.use_dist_sampler = False


def load_annotations(num_samples=10):
    """加载真实标注文件，返回前num_samples条"""
    annotation_file = Path(__file__).parent.parent.parent.parent.parent / "results" / "dataset_annotation" / "annotation_results.json"
    
    if not annotation_file.exists():
        print(f"标注文件不存在: {annotation_file}")
        return None
    
    try:
        with open(annotation_file, 'r', encoding='utf-8') as f:
            annotations = json.load(f)
        
        # 过滤出存在的文件
        valid = []
        for ann in annotations:
            file_path = ann.get("file_path")
            if file_path and Path(file_path).exists():
                valid.append(ann)
                if len(valid) >= num_samples:
                    break
        
        print(f"✓ 加载 {len(valid)} 条标注 (共{len(annotations)}条)")
        return valid
    
    except Exception as e:
        print(f"加载标注失败: {e}")
        return None


def test_classification():
    """测试分类任务"""
    print("\n" + "="*60)
    print("测试1: 分类任务")
    print("="*60)
    
    try:
        # 加载标注
        annotations = load_annotations(num_samples=10)
        if not annotations:
            print("✗ 无效的标注数据")
            return False
        
        # 创建临时标注文件
        temp_dir = tempfile.mkdtemp()
        temp_annotation = Path(temp_dir) / "anno.json"
        with open(temp_annotation, 'w', encoding='utf-8') as f:
            json.dump(annotations, f, ensure_ascii=False)
        
        # 获取数据目录
        data_dir = str(Path(annotations[0]["file_path"]).parent.parent)
        
        # 创建数据集
        config = SimpleConfig(
            annotation_file=str(temp_annotation),
            data_dir=data_dir,
            task_type="classification",
            window_size=3000,
        )
        
        dataset = AnnotationDataset(config)
        print(f"✓ 数据集创建成功: {len(dataset)} 样本")
        
        # 获取一个样本
        if len(dataset) > 0:
            data, label = dataset[0]
            print(f"✓ 分类样本: 数据shape={data.shape}, 标签={label.item()}")
            assert data.ndim == 2, "数据应为2D"
            assert label.ndim == 0, "标签应为标量"
            
            # 再测试一个
            data2, label2 = dataset[1]
            print(f"✓ 第二个样本: 数据shape={data2.shape}, 标签={label2.item()}")
        
        # 清理
        import shutil
        shutil.rmtree(temp_dir)
        return True
    
    except Exception as e:
        print(f"✗ 失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_regression():
    """测试回归任务"""
    print("\n" + "="*60)
    print("测试2: 回归任务")
    print("="*60)
    
    try:
        # 加载标注
        annotations = load_annotations(num_samples=15)
        if not annotations:
            print("✗ 无效的标注数据")
            return False
        
        # 创建临时标注文件
        temp_dir = tempfile.mkdtemp()
        temp_annotation = Path(temp_dir) / "anno.json"
        with open(temp_annotation, 'w', encoding='utf-8') as f:
            json.dump(annotations, f, ensure_ascii=False)
        
        # 获取数据目录
        data_dir = str(Path(annotations[0]["file_path"]).parent.parent)
        
        # 创建数据集
        config = SimpleConfig(
            annotation_file=str(temp_annotation),
            data_dir=data_dir,
            task_type="regression",
            window_size=3000,
            look_back=100,
            forecast_steps=50,
            regression_stride=10,
        )
        
        dataset = AnnotationDataset(config)
        print(f"✓ 数据集创建成功: {len(dataset)} 样本")
        
        # 获取一个样本
        if len(dataset) > 0:
            input_data, output_data = dataset[0]
            print(f"✓ 回归样本: input_shape={input_data.shape}, output_shape={output_data.shape}")
            assert input_data.shape[0] == 100, f"输入应为100步，实际{input_data.shape[0]}"
            assert output_data.shape[0] == 50, f"输出应为50步，实际{output_data.shape[0]}"
            
            # 再测试一个
            input_data2, output_data2 = dataset[1]
            print(f"✓ 第二个样本: input_shape={input_data2.shape}, output_shape={output_data2.shape}")
        
        # 清理
        import shutil
        shutil.rmtree(temp_dir)
        return True
    
    except Exception as e:
        print(f"✗ 失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_cache():
    """测试缓存机制"""
    print("\n" + "="*60)
    print("测试3: 缓存机制")
    print("="*60)
    
    try:
        # 加载标注
        annotations = load_annotations(num_samples=5)
        if not annotations:
            print("✗ 无效的标注数据")
            return False
        
        # 创建临时标注文件
        temp_dir = tempfile.mkdtemp()
        temp_annotation = Path(temp_dir) / "anno.json"
        with open(temp_annotation, 'w', encoding='utf-8') as f:
            json.dump(annotations, f, ensure_ascii=False)
        
        # 获取数据目录
        data_dir = str(Path(annotations[0]["file_path"]).parent.parent)
        
        # 创建数据集
        config = SimpleConfig(
            annotation_file=str(temp_annotation),
            data_dir=data_dir,
            task_type="classification",
            window_size=3000,
            cache_max_items=50,
        )
        
        dataset = AnnotationDataset(config)
        print(f"✓ 缓存已启用: max_items={dataset.vic_cache.max_items if dataset.vic_cache else 'N/A'}")
        
        # 多次访问同一样本
        for i in range(3):
            data, label = dataset[0]
        
        # 打印缓存统计
        if dataset.vic_cache:
            stats = dataset.vic_cache.get_stats()
            print(f"✓ 缓存统计: 命中={stats['hits']}, 未命中={stats['misses']}, 命中率={stats['hit_rate']:.1f}%")
        
        # 清理
        import shutil
        shutil.rmtree(temp_dir)
        return True
    
    except Exception as e:
        print(f"✗ 失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("\n" + "*"*60)
    print("  标注数据集测试 - 简化版")
    print("*"*60)
    
    results = []
    results.append(("分类任务", test_classification()))
    results.append(("回归任务", test_regression()))
    results.append(("缓存机制", test_cache()))
    
    print("\n" + "="*60)
    print("测试结果汇总")
    print("="*60)
    
    for name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"  {name:15s}: {status}")
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    print(f"\n总体: {passed}/{total} 测试通过")
    
    if passed == total:
        print("✓ 所有测试通过！")
        return 0
    else:
        print(f"✗ {total - passed} 个测试失败")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
