from torch.utils.data import Dataset, DataLoader
from scipy.io import loadmat
import os
import numpy as np
from PIL import Image
import torch
from pathlib import Path
from torchvision.transforms import ToTensor



# 因为要采用img、所以需要改一下对应图像路径
class MyDataset(Dataset):
    def __init__(self, data_dir = './datasets/train/data/', img_dir = './datasets/train/img/'):
        self.current_directory = data_dir
        self.img_dir = img_dir
        with os.scandir(self.current_directory) as entries:
            self.paths = [entry.path for entry in entries if entry.is_file()]

        self.totensor = ToTensor()
        return 
    

    def __getitem__(self, index):
        # 获取label
        path = self.paths[index]
        label, data = [(key, value) for key, value in loadmat(path).items() if isinstance(value, np.ndarray)][0]
        output = int(label)
        input = Image.open(self.img_dir + Path(path).parts[-1][:-4] + '.png')
        input = self.totensor(input)[:3]
        return input, output


    def __len__(self):
        return len(self.paths)
    
if __name__ == '__main__':
    datas = MyDataset()
    print(datas[0])
    print(datas[0][0].shape)