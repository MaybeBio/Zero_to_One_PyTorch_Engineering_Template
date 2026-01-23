# src/utils.py
import os
import yaml
import random
import logging
import numpy as np
import matplotlib.pyplot as plt
import torch

def setup_logging(log_file='training.log'):
    """
    Description
    -----------
    配置 logging, 让实验过程中的关键信息(如训练进度、评估结果等)同时记录到控制台和文件中,
    方便实时查看和后续回朔查看
    - 同时输出到控制台(Console)和文件(File)
    - 格式: [时间] [级别] 消息
    
    Args
    -----
    log_file : str
        日志文件路径
    """
    logging.basicConfig(
        level=logging.INFO, # 日志级别, 只显示 INFO 及以上级别的日志
        format='[%(asctime)s] [%(levelname)s] %(message)s', # 日志格式
        handlers=[
            logging.StreamHandler(), # 控制台输出
            logging.FileHandler(log_file) # 文件输出
        ]
    )

def seed_everything(seed=2026):
    """
    Description
    -----------
    固定所有随机种子，保证实验可复现 (Reproducibility)
    
    Args
    -----
    seed : int
        随机种子
    """
    # 固定python的random库的随机性(如shuffle等)
    random.seed(seed)
    # 固定python哈希随机性(如dict的遍历顺序)
    os.environ['PYTHONHASHSEED'] = str(seed)
    # 固定numpy库的随机性(如np.random.shuffle等, 随机数生成等)
    np.random.seed(seed)
    # 固定pytorch核心随机性(cpu/单卡场景下的model参数初始化/dropout等)
    torch.manual_seed(seed)
    # 固定pytorch单张gpu的随机性(确保单卡随机操作一致)
    torch.cuda.manual_seed(seed)
    # 固定pytorch多张gpu的随机性(确保多卡随机操作一致)
    torch.cuda.manual_seed_all(seed)
    # 固定 cuDNN 库（GPU 加速库）的算法确定性（避免非确定性算法导致结果波动）
    # 可能会稍微降低性能，但保证确定性（cuDNN算法选择）
    torch.backends.cudnn.deterministic = True
    # 关闭 cuDNN 算法自动调优（避免动态选算法带来的随机性）
    torch.backends.cudnn.benchmark = False

def load_config(config_path, defaults=None):
    """
    Description
    -----------
    加载 YAML 配置文件并与 defaults 递归合并
    
    Args
    -----
    config_path : str
        配置文件路径
    defaults : dict, optional
        默认配置字典，用于补全 config 中缺失的项
    
    Returns
    -------
    dict
        合并后的配置字典
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
        
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f) or {}

    # 递归合并输入的配置字典和默认的配置字典
    # 仅作为补全使用，config 中已有的键值不被覆盖
    # 仅作为示例，实际项目中可根据需要调整合并逻辑
    def recursive_merge(default_dict, new_dict):
        if not isinstance(default_dict, dict) or not isinstance(new_dict, dict):
            return new_dict
        result = default_dict.copy()
        for k, v in new_dict.items():
            if k in result and isinstance(result[k], dict) and isinstance(v, dict):
                result[k] = recursive_merge(result[k], v)
            else:
                result[k] = v
        return result
        
    if defaults:
        return recursive_merge(defaults, config)
                        
    return config

def plot_history(history, save_path=None, show=True):
    """
    Description
    -----------
    绘制训练过程中的损失和准确率曲线 (包含训练集和验证集)

    Args
    -----
    history : dict
        训练历史记录，包含 'loss', 'acc', 'val_loss', 'val_acc'
    save_path : str, optional
        图片保存路径，如果为 None 则不保存
    show : bool
        是否调用 plt.show() 显示图片 (在无头服务器上设为 False)
    """
    # 辅助转换函数
    def to_cpu_list(data):
        return [x if isinstance(x, (int, float)) else x.item() for x in data]

    loss = to_cpu_list(history.get('loss', []))
    val_loss = to_cpu_list(history.get('val_loss', []))
    acc = to_cpu_list(history.get('acc', []))
    val_acc = to_cpu_list(history.get('val_acc', []))

    plt.figure(figsize=(12, 5))
    
    # 绘制 Loss
    plt.subplot(1, 2, 1)
    if loss: plt.plot(loss, label='Train Loss')
    if val_loss: plt.plot(val_loss, label='Val Loss', linestyle='--')
    plt.title('Loss Curve')
    plt.xlabel('Epochs')
    plt.legend()
    
    # 绘制 Accuracy
    plt.subplot(1, 2, 2)
    if acc: plt.plot(acc, label='Train Acc', color='orange')
    if val_acc: plt.plot(val_acc, label='Val Acc', color='red', linestyle='--')
    plt.title('Accuracy Curve')
    plt.xlabel('Epochs')
    plt.legend()
    
    if save_path:
        # 自动创建父目录
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Figure saved to {save_path}")
        
    if show:
        plt.show()
    plt.close() # 释放资源
