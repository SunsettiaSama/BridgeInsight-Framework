from datasets import *
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter
from .EfficientViT.classification.model.efficientvit import EfficientViT
from tqdm import tqdm
from torch import nn
import torch
from .datasets import MyDataset

from ..visualize_tools.utils import PlotLib

def eval_model():
    
    from NN.EfficientViT.classification.model.efficientvit import EfficientViT
    from NN.datasets import MyDataset
    from tqdm import tqdm
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    Net = EfficientViT(img_size = 512, num_classes = 3).to(device)
    Net.load_state_dict(
        torch.load(r'F:\Research\My_Thesis\Vibration Characteristics In Cable Vibration\NN\models\V5\\ECA_Net_fold4.pth', map_location = device)
        )
    Net.eval()

    dataset = MyDataset(
        data_dir = r'F:\Research\My_Thesis\Vibration Characteristics In Cable Vibration\NN\datasets\dev\data\\', 
        img_dir = r'F:\Research\My_Thesis\Vibration Characteristics In Cable Vibration\NN\datasets\dev\img\\'
    )

    True_Label = []
    Predicted_Label = []
    labels = ['Normal Vibration', 'VIV']

    def int2label(num):
        if num == 0:
            label = 'Normal Vibration'
        elif num == 1:
            label = 'VIV'
        elif num == 2:
            label = 'VIV'
        return label
    
    VIV_lis = []
    with tqdm(total = len(dataset)) as pbar:
        for i, (img, label) in enumerate(dataset):

            img = img.to(device).unsqueeze(0)
            True_Label.append(int2label(label))
            predict_int = torch.argmax(Net(img))
            Predicted_Label.append(int2label(predict_int))

            pbar.update(1)

            
            if predict_int == 2:
                VIV_lis.append(img[0, :, :, :].cpu().numpy())
            
    # 绘制混淆矩阵
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix
    def plot_confusion_matrix(y_true, y_pred, labels, normalize=False):
        # 生成混淆矩阵
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        
        if normalize:
            # 归一化处理
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("显示归一化的混淆矩阵")
        else:
            print('显示未归一化的混淆矩阵')

        # 绘制混淆矩阵
        fig, ax = plt.subplots()
        fmt = '.2f' if normalize else 'd'
        sns.heatmap(cm, annot=True, fmt=fmt, cmap='Pastel2', xticklabels=labels, yticklabels=labels)
        ax.set_title('Confusion Matrix')
        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')

        return fig, ax 

    
    fig, ax = plot_confusion_matrix(True_Label, Predicted_Label, labels = labels, normalize = True)
    ploter = PlotLib()
    ploter.figs.append(fig)

    # 绘制VIV真值结果
    for img_array in VIV_lis:
        fig, ax = plt.subplots()
        ax.imshow(np.transpose(img_array, (1, 2, 0)))

        ax.grid(False)
        ax.axis('off')

        ploter.figs.append(fig)
        plt.close()


    ploter.show()

    return 
