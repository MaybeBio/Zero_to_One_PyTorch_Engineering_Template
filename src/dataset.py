# src/dataset.py
import torch
import numpy as np
import nnfs
from nnfs.datasets import spiral_data
from torch.utils.data import Dataset, DataLoader, random_split

# nnfs.init() # 通常在主程序入口调用

class SpiralDataset(Dataset):
    """
    Description
    ---------
    自定义 Dataset 类：负责从数据源读取并封装单个样本;
    必须继承 torch.utils.data.Dataset, 实现 __len__ 和 __getitem__,
    对于大数据集, __getitem__ 只在需要时加载单个文件 (Lazy Loading)，节省内存
    """
    def __init__(self, samples=100, classes=3):
        """
        Description
        ---------
        构造函数：初始化数据集
        
        Args
        ---------
        samples : int
            每个类别的样本数
        classes : int
            类别数
        
        Returns
        ---------
        None
        """
         
        # 实际项目中，这里通常接收文件路径 list，而不是直接生成数据
        X, y = spiral_data(samples=samples, classes=classes)
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        # 实际项目中，这里可能包含实时的数据增强 (Augmentation)
        # 例如: torchvision.transforms
        return self.X[idx], self.y[idx]

def get_dataloader(samples, classes, batch_size=32, shuffle=True, val_split=0.0):
    """
    Description
    ---------
    构建并返回 DataLoader, DataLoader负责batch批量加载数据/切分/shuffle打乱/多进程预取等
    
    Args
    ---------
    samples : int
        每个类别的样本数
    classes : int
        类别数
    batch_size : int or None
        批量大小; None 表示全量梯度下降
    shuffle : bool
        是否打乱数据顺序
    val_split : float
        验证集比例 (0.0 ~ 1.0)
    
    Returns
    ---------
    DataLoader or (DataLoader, DataLoader)
        Train DataLoader, [Val DataLoader]


    Notes
    ---------
    - 1. 不只返回一个 loader, 通常需要 Train/Val/Test split, 返回多个dataloader
    - 2. 使用 num_workers 进行多进程预取: 待补充
    - 3. 使用 pin_memory 加速 Host-to-Device 传输 (如果用 GPU): 待补充
    """
    full_dataset = SpiralDataset(samples, classes)
    
    # 全量梯度下降情况
    if batch_size is None:
        batch_size = len(full_dataset)
        
    if val_split > 0:
        val_size = int(len(full_dataset) * val_split)
        train_size = len(full_dataset) - val_size
        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
        # 验证集通常不shuffle
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        return train_loader, val_loader
    else:
        # 或者不显示给出shuffle参数, 在具体示例时再分别调整修改
        return DataLoader(full_dataset, batch_size=batch_size, shuffle=shuffle)

