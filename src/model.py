# src/model.py
import torch
import torch.nn as nn

class UniversalMLP(nn.Module):
    """
    Description
    ------------
    通用多层感知机 (MLP) 模板, 适用于分类/回归任务, 可用于进一步扩展(适用于tabular数据等)
    

    Notes
    ------------
    - 1. 不变的地方: 只写init定义层, forward定义流向, backward自动完成自动微分处理
    - 2. 支持字符串指定激活函数, 也就是添加多种激活函数选择
      - 增加权重初始化 (_init_weights)
    """
    def __init__(self, input_dim, hidden_dims, output_dim, activation='relu', dropout_rate=0.0):
        """ 
        Description
        ------------
        初始化多层感知机, 动态构建隐藏层
        
        Args
        -----
        input_dim : int
            输入特征维度
        hidden_dims : list of int
            隐藏层维度列表, 每个元素代表一层的神经元数量
        output_dim : int
            输出维度 (类别数)
        activation : str or nn.Module
            激活函数模块名称('relu', 'tanh', 'sigmoid', 'leaky_relu') 或 nn.Module 类, 默认使用 ReLU
        dropout_rate : float
            Dropout 比例, 默认为 0.0 (不使用 Dropout)
        
        Returns
        -------
        None

        """
        
        # 调用父类构造函数
        super(UniversalMLP, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        # 激活函数选择：支持字符串或类
        def get_activation(act):
            if isinstance(act, str):
                act = act.lower()
                if act == 'relu': return nn.ReLU()
                if act == 'tanh': return nn.Tanh()
                if act == 'sigmoid': return nn.Sigmoid()
                if act == 'leaky_relu': return nn.LeakyReLU()
                raise ValueError(f"Unsupported activation string: {act}")
            # 检查是否是元类(类的类), 或者是否继承自nn.Module
            elif isinstance(act, type) and issubclass(act, nn.Module):
                return act() # 实例化
            else:
                return act # 假设已经是实例，注意深拷贝问题，但在 Sequential 中主要关注每层是否独立

        # 动态构建隐藏层
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            # BatchNorm 通常放在 Activation 之前 (ResNet v1) 或之后 (ResNet v2)，这里演示放在前面, 也就是Linear和Activation之间
            # layers.append(nn.BatchNorm1d(h_dim)) 
            
            # 添加激活函数和Dropout
            # 确保每次都生成一个新的 Activation 实例, 为每一层创建独立的实例, 不要把同一个activation实例重复使用在多层
            layers.append(get_activation(activation))
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
                
            # 更新前一层维度    
            prev_dim = h_dim
            
        # 输出层 (不加激活，因为 CrossEntropyLoss 包含 Softmax)
        layers.append(nn.Linear(prev_dim, output_dim))
        
        # 将列表转为 Sequential 容器
        # 将各模块注册为 Module 的子模块, 以便参数能被正确识别和更新
        self.network = nn.Sequential(*layers)
        
        # 显示调用初始化
        # 对全连接线性层权重进行初始化, 默认初始化可能导致训练不稳定/梯度消失/爆炸
        # 我们只有全连接层, 所以这里只处理 nn.Linear; 我们只用 ReLU 激活函数, 所以选择 Kaiming 初始化
        self.network.apply(self._init_weights)
        
    def _init_weights(self, m):
        """
        Description
        -----------
        权重初始化方法, Kaiming / Xavier 初始化，比默认初始化收敛更快

        Args
        ----
        m : nn.Module
            模块实例, 通常是 nn.Linear, nn.Conv2d 等

        Notes
        -----
        - 1. 初始化没做好容易梯度消失/爆炸，导致训练失败
        - 2. Kaiming 初始化适合 ReLU 激活函数, Xavier 初始化适合 Sigmoid/Tanh 激活函数
        - 3. 因为我们整个网络只包含全连接层, 所以这里只处理 nn.Linear, 忽略其他类型模块; 然后因为我们使用的是ReLU激活函数,
        所以我们选择 Kaiming 初始化方法
        """
        # 只初始化全连接线性层
        if isinstance(m, nn.Linear):
            # Kaiming He 初始化 (适合 ReLU)
            # Kaiming正态分布初始化, 针对ReLU类激活函数设计的权重初始化方法, 避免深层网络训练时出现梯度消失/爆炸问题
            # fan_out 保证反向传播时梯度方差的量级稳定（输出维度决定初始化范围）
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                # 对全连接层偏置初始化为0
                nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
        """
        Description
        -----------
        前向传播逻辑

        Args
        ----
        x : torch.Tensor
            输入特征张量, 形状为 (batch_size, input_dim), 即 (批大小, 输入特征维度)
        
        Returns
        -------
        torch.Tensor
            输出张量, 形状为 (batch_size, output_dim), 即 (批大小, 输出维度/类别数)
        """
        return self.network(x)
