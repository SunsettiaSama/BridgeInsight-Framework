import sys
import os
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from machine_learning_module.cellunar_automata.run import run_ca_workflow


class SimpleDataset(Dataset):
    def __init__(self, size=500, num_classes=3, feature_dim=20):
        self.size = size
        self.num_classes = num_classes
        self.feature_dim = feature_dim
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        data = torch.randn(self.feature_dim)
        label = torch.randint(0, self.num_classes, (1,)).item()
        return data, label


def test_ca_workflow():
    print("\n" + "="*50)
    print("元胞自动机工作流测试开始")
    print("="*50)
    
    train_dataset = SimpleDataset(500)
    val_dataset = SimpleDataset(100)
    infer_dataset = SimpleDataset(150)
    
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
    
    print("\n" + "-"*50)
    print("元胞自动机工作流测试结果")
    print("-"*50)
    
    if 'train' in results and results['train']:
        train_result = results['train']
        print(f"✓ 训练完成")
        print(f"  - 训练样本数: {train_result.get('n_samples', 'N/A')}")
        print(f"  - 类别数: {train_result.get('n_class', 'N/A')}")
        print(f"  - CA网格大小: {train_result.get('ca_grid', 'N/A')}")
        print(f"  - CA演化步数: {train_result.get('ca_steps', 'N/A')}")
    
    if 'eval' in results and results['eval']:
        eval_result = results['eval']
        print(f"✓ 推理完成")
        print(f"  - 推理样本数: {eval_result.get('total', 'N/A')}")
        if 'accuracy' in eval_result:
            print(f"  - 推理准确率: {eval_result['accuracy']:.4f}")
    
    print("\n" + "="*50)
    print("元胞自动机工作流测试成功")
    print("="*50)


if __name__ == "__main__":
    test_ca_workflow()
