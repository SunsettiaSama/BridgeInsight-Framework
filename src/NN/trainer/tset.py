import torch
import yaml
import os
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import KFold
from tqdm import tqdm
# 请确保MyDataset和EfficientViT已正确导入
from .datasets import MyDataset
from .EfficientViT.classification.model.efficientvit import EfficientViT

def load_config(config_path):
    """加载YAML配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    # 参数校验与默认值补充
    config['training']['device'] = torch.device('cuda' if (torch.cuda.is_available() and config['training']['device'] == 'auto') else config['training']['device'])
    # 确保模型保存目录存在
    os.makedirs(config['paths']['model_save_dir'], exist_ok=True)
    return config

def train(config_path='config.yaml'):
    # 加载配置文件
    cfg = load_config(config_path)
    
    # 从配置中提取参数
    k_folds = cfg['cross_validation']['k_folds']
    batch_size = cfg['data']['batch_size']
    drop_last = cfg['data']['drop_last']
    data_dir = cfg['data']['data_dir']
    img_dir = cfg['data']['img_dir']
    num_epochs = cfg['training']['num_epochs']
    device = cfg['training']['device']
    model_save_interval = cfg['training']['model_save_interval']
    lr = cfg['optimizer']['lr']
    optimizer_type = cfg['optimizer']['type']
    img_size = cfg['model']['img_size']
    num_classes = cfg['model']['num_classes']
    tensorboard_log_dir = cfg['paths']['tensorboard_log_dir']
    model_save_dir = cfg['paths']['model_save_dir']

    # 初始化数据集、KFold、TensorBoard
    dataset = MyDataset(data_dir=data_dir, img_dir=img_dir)
    kfold = KFold(n_splits=k_folds, shuffle=True)
    writer = SummaryWriter(tensorboard_log_dir)

    # 初始化模型、损失函数、优化器
    Net = EfficientViT(img_size=img_size, num_classes=num_classes).to(device)
    criterion = CrossEntropyLoss().to(device)
    # 适配不同优化器（扩展方便）
    if optimizer_type == 'Adam':
        optimizer = Adam(Net.parameters(), lr=lr)
    else:
        raise ValueError(f"暂不支持优化器类型：{optimizer_type}")

    # 开始K折交叉验证训练
    for fold, (train_ids, val_ids) in enumerate(kfold.split(dataset)):
        print(f'FOLD {fold}')
        print('--------------------------------')
        
        # 创建采样器和数据加载器
        train_subsampler = SubsetRandomSampler(train_ids)
        val_subsampler = SubsetRandomSampler(val_ids)
        train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_subsampler, drop_last=drop_last)
        val_loader = DataLoader(dataset, batch_size=batch_size, sampler=val_subsampler, drop_last=drop_last)
        
        # 单折训练
        for epoch in range(num_epochs):
            print(f'Epoch {epoch + 1}/{num_epochs} --------------------------------------------------')
            
            # 训练阶段
            Net.train()
            train_total, train_correct = 0, 0
            print('--Training--')
            with tqdm(total=len(train_loader)) as pbar:
                for i, (images, labels) in enumerate(train_loader):
                    images, labels = images.to(device), labels.to(device)
                    optimizer.zero_grad()
                    outputs = Net(images)
                    predicted = torch.argmax(outputs, dim=1)
                    loss = criterion(outputs, labels)

                    loss.backward()
                    optimizer.step()

                    train_total += labels.size(0)
                    train_correct += (predicted == labels).sum().item()
                    pbar.update(1)
            train_acc = 100 * train_correct / train_total
            train_loss = loss.item()
            print(f'Train Accuracy: {train_acc:.2f}%, Loss: {train_loss:.4f}')
            # 记录训练日志
            writer.add_scalar(tag=f'Loss/train FOLD {fold}', scalar_value=train_loss, global_step=epoch)
            writer.add_scalar(tag=f'Accuracy/train FOLD {fold}', scalar_value=train_acc, global_step=epoch)

            # 验证阶段（修正原脚本缩进错误）
            Net.eval()
            val_total, val_correct = 0, 0
            val_loss = 0.0
            print('--Test--')
            with torch.no_grad():
                for i, (images, labels) in enumerate(val_loader):
                    images, labels = images.to(device), labels.to(device)
                    outputs = Net(images)
                    batch_loss = criterion(outputs, labels)
                    predicted = torch.argmax(outputs, dim=1)

                    # 累加验证集的total、correct和loss
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
                    val_loss += batch_loss.item() * labels.size(0)  # 按样本数加权平均
            # 计算验证集平均loss和准确率
            val_acc = 100 * val_correct / val_total
            val_loss_avg = val_loss / val_total
            print(f'Test Accuracy: {val_acc:.2f}%, Loss: {val_loss_avg:.4f}')
            # 记录验证日志
            writer.add_scalar(tag=f'Loss/val FOLD {fold}', scalar_value=val_loss_avg, global_step=epoch)
            writer.add_scalar(tag=f'Accuracy/val FOLD {fold}', scalar_value=val_acc, global_step=epoch)

            # 保存模型
            if (epoch + 1) % model_save_interval == 0:
                model_path = os.path.join(model_save_dir, f'ECA_Net_fold{fold}_epoch{epoch+1}.pth')
                torch.save(Net.state_dict(), model_path)
                print(f'Model saved to {model_path}')

    writer.close()
    return

if __name__ == '__main__':
    # 可指定自定义配置文件路径，默认读取当前目录的config.yaml
    train(config_path='config.yaml')