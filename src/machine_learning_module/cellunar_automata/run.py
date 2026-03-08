import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import logging
import os
from .train import train_ca_classifier
from .eval import infer_ca_classifier


class CAWorkflow:
    """元胞自动机完整工作流类"""
    
    def __init__(self, config=None):
        """
        初始化工作流
        :param config: 工作流配置对象（如果为None，使用默认参数）
        """
        self.config = config or {}
        self.logger = self._setup_logging()
        self.train_result = None
        self.eval_result = None
    
    def _setup_logging(self):
        """设置日志系统"""
        logger = logging.getLogger('CAWorkflow')
        logger.setLevel(logging.INFO)
        
        logger.handlers.clear()
        
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        return logger
    
    def run(self, train_dataloader=None, val_dataloader=None, infer_dataloader=None, infer_has_label=True):
        """
        执行完整的元胞自动机工作流
        :param train_dataloader: 训练集DataLoader（训练阶段需要）
        :param val_dataloader: 验证集DataLoader（可选）
        :param infer_dataloader: 推理集DataLoader（推理阶段需要）
        :param infer_has_label: 推理数据是否有标签
        :return: 包含训练和推理结果的字典
        """
        self.logger.info("=" * 50)
        self.logger.info("开始执行元胞自动机工作流")
        self.logger.info("=" * 50)
        
        results = {}
        
        enable_train = self.config.get('enable_train', True)
        enable_eval = self.config.get('enable_eval', True)
        
        if enable_train:
            self.logger.info("开始训练阶段...")
            if train_dataloader is None:
                self.logger.error("训练模式下，train_dataloader 不能为空")
                raise ValueError("train_dataloader is required for training")
            
            self.train_result = self._run_training(train_dataloader, val_dataloader)
            results['train'] = self.train_result
            self.logger.info("训练阶段完成")
        
        if enable_eval:
            self.logger.info("开始推理阶段...")
            if infer_dataloader is None:
                self.logger.error("推理模式下，infer_dataloader 不能为空")
                raise ValueError("infer_dataloader is required for evaluation")
            
            self.eval_result = self._run_evaluation(infer_dataloader, infer_has_label)
            results['eval'] = self.eval_result
            self.logger.info("推理阶段完成")
        
        self.logger.info("=" * 50)
        self.logger.info("元胞自动机工作流执行完成")
        self.logger.info("=" * 50)
        
        return results
    
    def _run_training(self, train_dataloader, val_dataloader=None):
        """
        执行训练阶段
        :param train_dataloader: 训练集DataLoader
        :param val_dataloader: 验证集DataLoader
        :return: 训练结果
        """
        self.logger.info("正在运行元胞自动机训练...")
        
        _base = "results/classification_results/machine_learning/ca"
        tc = self.config.get('TRAIN_CONFIG', {
            'template_save_path': f'{_base}/ca_class_templates.pkl',
            'result_save_path': f'{_base}/ca_e2e_train_result.json'
        })
        result = train_ca_classifier(
            train_dataloader,
            template_save_path=tc.get('template_save_path'),
            result_save_path=tc.get('result_save_path')
        )
        
        self.logger.info(f"训练样本数: {result.get('n_samples', 'N/A')}")
        self.logger.info(f"类别数: {result.get('n_class', 'N/A')}")
        
        self.logger.info(f"CA参数 - 网格大小: {result.get('ca_grid', 'N/A')}, 演化步数: {result.get('ca_steps', 'N/A')}")
        
        return result
    
    def _run_evaluation(self, infer_dataloader, has_label=True):
        """
        执行推理阶段
        :param infer_dataloader: 推理集DataLoader
        :param has_label: 推理数据是否有标签
        :return: 推理结果
        """
        self.logger.info("正在运行元胞自动机推理...")
        
        _base = "results/classification_results/machine_learning/ca"
        ec = self.config.get('EVAL_CONFIG', {
            'template_load_path': f'{_base}/ca_class_templates.pkl',
            'result_path': f'{_base}/ca_e2e_infer_result.json'
        })
        result = infer_ca_classifier(
            infer_dataloader, has_label,
            template_load_path=ec.get('template_load_path'),
            infer_result_path=ec.get('result_path')
        )
        
        self.logger.info(f"推理样本数: {result.get('total', 'N/A')}")
        
        if has_label and 'accuracy' in result:
            self.logger.info(f"推理准确率: {result['accuracy']:.4f}")
        
        return result
    
    def get_results(self):
        """获取完整的工作流结果"""
        return {
            'train': self.train_result,
            'eval': self.eval_result
        }


def run_ca_workflow(
    train_dataloader=None,
    val_dataloader=None,
    infer_dataloader=None,
    config=None,
    infer_has_label=True,
    enable_train=True,
    enable_eval=True
):
    """
    便捷函数：直接运行元胞自动机工作流
    
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
        config = {}
    
    config['enable_train'] = enable_train
    config['enable_eval'] = enable_eval
    
    workflow = CAWorkflow(config)
    results = workflow.run(
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        infer_dataloader=infer_dataloader,
        infer_has_label=infer_has_label
    )
    
    return results


if __name__ == "__main__":
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
    
    train_dataset = MockDataset(1000)
    val_dataset = MockDataset(200)
    infer_dataset = MockDataset(300)
    
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    infer_dataloader = DataLoader(infer_dataset, batch_size=32, shuffle=False)
    
    config = {'enable_train': True, 'enable_eval': True}
    
    results = run_ca_workflow(
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        infer_dataloader=infer_dataloader,
        config=config,
        infer_has_label=True
    )
    
    print("\n工作流执行完成!")
    print(f"训练结果: {results.get('train')}")
    print(f"推理结果: {results.get('eval')}")
