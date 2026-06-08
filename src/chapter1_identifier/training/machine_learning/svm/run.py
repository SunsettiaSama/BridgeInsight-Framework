import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import logging
import sys
import os
from pathlib import Path
from .train import train_svm
from .eval import infer_svm
from config.machine_learning_module.svm.workflow_config import SVMWorkflowConfig


class SVMWorkflow:
    """SVM 完整工作流类"""
    
    def __init__(self, config=None):
        """
        初始化工作流
        :param config: SVMWorkflowConfig 或其他配置对象
        """
        self.config = config or SVMWorkflowConfig()
        self.logger = self._setup_logging()
        self.train_result = None
        self.eval_result = None
    
    def _setup_logging(self):
        """设置日志系统"""
        logger = logging.getLogger('SVMWorkflow')
        logger.setLevel(self.config.LOG_LEVEL)
        
        # 清除已有的处理器
        logger.handlers.clear()
        
        # 日志格式化器
        formatter = logging.Formatter(self.config.LOG_FORMAT)
        
        # 控制台处理器
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # 文件处理器
        if self.config.LOG_FILE:
            file_handler = logging.FileHandler(self.config.LOG_FILE, encoding='utf-8')
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        
        return logger
    
    def run(self, train_dataloader=None, val_dataloader=None, infer_dataloader=None, infer_has_label=True):
        """
        执行完整的SVM工作流
        :param train_dataloader: 训练集DataLoader（训练阶段需要）
        :param val_dataloader: 验证集DataLoader（可选）
        :param infer_dataloader: 推理集DataLoader（推理阶段需要）
        :param infer_has_label: 推理数据是否有标签
        :return: 包含训练和推理结果的字典
        """
        self.logger.info("=" * 50)
        self.logger.info(f"开始执行SVM工作流 - 模式: {self.config.MODE}")
        self.logger.info("=" * 50)
        
        results = {}
        
        # 训练阶段
        if self.config.ENABLE_TRAIN:
            self.logger.info("开始训练阶段...")
            if train_dataloader is None:
                self.logger.error("训练模式下，train_dataloader 不能为空")
                raise ValueError("train_dataloader is required for training")
            
            self.train_result = self._run_training(train_dataloader, val_dataloader)
            results['train'] = self.train_result
            self.logger.info("训练阶段完成")
        
        # 推理阶段
        if self.config.ENABLE_EVAL:
            self.logger.info("开始推理阶段...")
            if infer_dataloader is None:
                self.logger.error("推理模式下，infer_dataloader 不能为空")
                raise ValueError("infer_dataloader is required for evaluation")
            
            self.eval_result = self._run_evaluation(infer_dataloader, infer_has_label)
            results['eval'] = self.eval_result
            self.logger.info("推理阶段完成")
        
        self.logger.info("=" * 50)
        self.logger.info("SVM工作流执行完成")
        self.logger.info("=" * 50)
        
        return results
    
    def _run_training(self, train_dataloader, val_dataloader=None):
        """
        执行训练阶段
        :param train_dataloader: 训练集DataLoader
        :param val_dataloader: 验证集DataLoader
        :return: 训练结果
        """
        self.logger.info("正在运行SVM训练...")
        
        tc = self.config.TRAIN_CONFIG
        result = train_svm(
            train_dataloader, val_dataloader,
            model_save_path=tc.get('model_save_path'),
            result_save_path=tc.get('result_save_path')
        )
        
        self.logger.info(f"训练集准确率: {result['train_metrics']['accuracy']:.4f}")
        if result['val_metrics']:
            self.logger.info(f"验证集准确率: {result['val_metrics']['accuracy']:.4f}")
        
        self.logger.info(f"模型参数: {result['model_params']}")
        
        return result
    
    def _run_evaluation(self, infer_dataloader, has_label=True):
        """
        执行推理阶段
        :param infer_dataloader: 推理集DataLoader
        :param has_label: 推理数据是否有标签
        :return: 推理结果
        """
        self.logger.info("正在运行SVM推理...")
        
        ec = self.config.EVAL_CONFIG
        model_path = ec.get('model_load_path', 'results/classification_results/machine_learning/svm/svm_model.pkl')
        infer_result_path = ec.get('result_path')
        
        result = infer_svm(infer_dataloader, model_path, has_label, infer_result_path=infer_result_path)
        
        self.logger.info(f"推理样本数: {result['sample_num']}")
        self.logger.info(f"类别数: {result['class_num']}")
        
        if has_label and 'accuracy' in result['metrics']:
            self.logger.info(f"推理准确率: {result['metrics']['accuracy']:.4f}")
        
        return result
    
    def get_results(self):
        """获取完整的工作流结果"""
        return {
            'train': self.train_result,
            'eval': self.eval_result
        }


def run_svm_workflow(
    train_dataloader=None,
    val_dataloader=None,
    infer_dataloader=None,
    config=None,
    infer_has_label=True,
    enable_train=True,
    enable_eval=True
):
    """
    便捷函数：直接运行SVM工作流
    
    :param train_dataloader: 训练集DataLoader
    :param val_dataloader: 验证集DataLoader（可选）
    :param infer_dataloader: 推理集DataLoader
    :param config: 工作流配置（如果为None，使用默认配置）
    :param infer_has_label: 推理数据是否有标签
    :param enable_train: 是否启用训练
    :param enable_eval: 是否启用推理
    :return: 工作流结果字典
    """
    if config is None:
        config = SVMWorkflowConfig()
    
    # 根据参数更新配置
    config.ENABLE_TRAIN = enable_train
    config.ENABLE_EVAL = enable_eval
    
    workflow = SVMWorkflow(config)
    results = workflow.run(
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        infer_dataloader=infer_dataloader,
        infer_has_label=infer_has_label
    )
    
    return results


if __name__ == "__main__":
    # 示例用法
    from torch.utils.data import Dataset, DataLoader
    import torch
    
    class MockDataset(Dataset):
        def __init__(self, size=1000, num_classes=10, feature_dim=10):
            self.size = size
            self.num_classes = num_classes
            self.feature_dim = feature_dim
        
        def __len__(self):
            return self.size
        
        def __getitem__(self, idx):
            data = torch.randn(self.feature_dim)
            label = torch.randint(0, self.num_classes, (1,)).item()
            return data, label
    
    # 创建DataLoader
    train_dataset = MockDataset(1000)
    val_dataset = MockDataset(200)
    infer_dataset = MockDataset(300)
    
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    infer_dataloader = DataLoader(infer_dataset, batch_size=32, shuffle=False)
    
    # 运行工作流（训练和推理）
    config = SVMWorkflowConfig()
    config.MODE = 'train_eval'
    config.ENABLE_TRAIN = True
    config.ENABLE_EVAL = True
    
    results = run_svm_workflow(
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        infer_dataloader=infer_dataloader,
        config=config,
        infer_has_label=True
    )
    
    print("\n工作流执行完成!")
    print(f"训练结果: {results.get('train')}")
    print(f"推理结果: {results.get('eval')}")
