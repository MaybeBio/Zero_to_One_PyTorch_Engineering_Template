# src/predict.py
import torch
import numpy as np

def predict(model, X, device='cpu'):
    """
    Description
    -----------
    推理逻辑, 在测试/预测阶段使用

    对比：
      - 简易版：直接 model(X) 得到预测结果
      - 工业版：
        1. model.eval() 关闭 Dropout/BatchNorm 随机性
        2. torch.no_grad() 关闭梯度计算引擎，节省显存并加速
        3. 处理 device (GPU -> CPU) 和 tensor -> numpy 转换
        4. 输入类型兼容 (NumPy, List, Tensor)

    Args
    ----
    model : torch.nn.Module
        训练好的模型
    X : np.ndarray or list or torch.Tensor
        原始数据, 输入特征数据，形状为 (num_samples, num_features), 即 batch 数据
    device : str
        运行设备，'cpu' 或 'cuda'

    Notes
    -----
    关于 @torch.no_grad() 和 with torch.no_grad():
    - @torch.no_grad(): 装饰器，作用于整个函数。适合纯评估/推理函数，代码更整洁。
    - with torch.no_grad(): 上下文管理器，作用于代码块。适合在函数内部只想对部分代码关闭梯度计算的情况，
      或者显式强调某段代码段是推理核心。
    - 两者在功能上是等价的。本函数使用 with 语句是为了强调数据搬运与推理计算的边界, 与函数整体逻辑区分开来, 参考trainer中的evaluate函数
    """
    
    # 切换评估模式, 这会固定 BatchNorm 的 running_stats 和禁用 Dropout
    # 防止推理结果不稳定
    model.eval()
    
    # 防御性编程：处理不同的输入类型 (支持 NumPy 数组、PyTorch 张量和列表)
    if isinstance(X, np.ndarray):
        X_tensor = torch.from_numpy(X).float()
    elif isinstance(X, torch.Tensor):
        X_tensor = X.float()
    else:
        # List or other iterables
        X_tensor = torch.tensor(X, dtype=torch.float32)
        
    X_tensor = X_tensor.to(device)
    
    # 推理阶段不需要计算梯度, 测试/预测阶段使用
    # 关闭 autograd 追踪，节省显存并加速前向推理（不需要梯度）
    # 上下文管理器，禁止梯度计算
    with torch.no_grad(): 
        logits = model(X_tensor)
        # 如果需要概率，手动加 Softmax (因为模型输出是 logits)
        probs = torch.softmax(logits, dim=1)
        # 获取最大概率的类别索引
        predictions = torch.argmax(probs, dim=1)
        
    # 将结果从 GPU 移动到 CPU 并转换为 numpy 数组返回, 方便后续与sklearn等后续cpu上处理库兼容
    return predictions.cpu().numpy()

