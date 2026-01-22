# src/trainer.py
import torch
import torch.nn as nn
import torch.optim as optim
import logging
import os

# 获取 logger
logger = logging.getLogger(__name__)

class EarlyStopping:
    """
    Description
    -----------
    早停 (Early Stopping) 逻辑封装, 当验证集损失在 patience 个 epoch 内没有降低时，停止训练

    """
    def __init__(self, patience=5, min_delta=0, path='best_model.pth'):
        """  
        Description
        -----------
        初始化早停参数

        Args
        -----
        patience : int
            容忍的最大不提升 epoch 数 (轮数, 即多少个 epoch 内验证集损失没有降低则停止训练)
        min_delta : float
            最小提升幅度 (即验证集损失必须降低至少 min_delta 才算提升)
        path : str
            最佳模型保存路径
        
        Returns
        -------
        None
        
        """
        self.patience = patience
        self.min_delta = min_delta
        self.path = path
        # 没有提升的 epoch 计数器
        self.counter = 0
        # 最佳损失初始化为 None
        self.best_loss = None
        # 是否触发早停
        self.early_stop = False

    def __call__(self, val_loss, trainer, epoch=None):
        if self.best_loss is None:
            self.best_loss = val_loss
            # 首次记录，保存为最佳模型
            trainer.save_checkpoint(self.path, is_best=True, epoch=epoch)
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            logger.info(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            # 发现更优模型，保存
            trainer.save_checkpoint(self.path, is_best=True, epoch=epoch)
            self.counter = 0

class Trainer:
    """
    Description
    -----------
    训练器 (Trainer) 逻辑封装, 包含训练循环、评估、保存模型等功能, 也就是model实际训练流程封装
    
    Notes
    -----------
    - 1. 改进的地方：
      - 使用 logging 替代 print, 方便日志管理, 相比于print, logging 可以灵活配置输出格式和级别, 
      可以将日志文件同时输出到控制台+文件, 便于调试和记录训练过程
      - 增加 save_checkpoint 和 load_checkpoint
      - 支持 验证集评估 (evaluate)
      - 支持 早停 (Early Stopping) 逻辑
    """
    def __init__(self, model, optimizer=None, criterion=None, device='cpu', save_dir='checkpoints'):
        """  
        Description
        -----------
        初始化训练器

        Args
        -----
        model : torch.nn.Module
            待训练的模型
        optimizer : torch.optim.Optimizer, optional
            优化器实例, 如果为 None 则使用 Adam 优化器
        criterion : torch.nn.Module, optional
            损失函数实例, 如果为 None 则使用 CrossEntropyLoss, 即默认为分类任务使用的交叉熵损失函数
        device : str, optional
            运行设备, 'cpu' 或 'cuda'
        save_dir : str, optional
            模型检查点保存目录, 检查点checkpoint数据, 包含模型权重和训练过程中的上下文信息, 目的是让训练可以诶断点续训或复用中间状态

        Returns
        -------
        None    
        """
        
        self.model = model.to(device)
        self.device = device
        # 默认使用交叉熵损失 (分类任务标准)
        self.criterion = criterion if criterion else nn.CrossEntropyLoss()
        # 默认使用 Adam
        self.optimizer = optimizer if optimizer else optim.Adam(model.parameters(), lr=0.001)
        # 初始化历史记录，确保即使 resume 也能接续
        self.history = {'loss': [], 'acc': [], 'val_loss': [], 'val_acc': []}
        self.save_dir = save_dir
        
        # 确保保存目录存在
        os.makedirs(self.save_dir, exist_ok=True)

    def train_epoch(self, dataloader):
        """
        Description
        -----------
        训练单个 Epoch (遍历所有 Batch)

        Args
        -----
        dataloader : torch.utils.data.DataLoader
            训练数据的 DataLoader
        
        Returns
        -------
        avg_loss : float
            平均损失
        avg_acc : float
            平均准确率
        """
        # 开启训练模式 (启用 Dropout/BatchNorm), 注意与后面的model.eval()区分, 后者是评估模式
        self.model.train() 
        total_loss = 0
        correct = 0
        total = 0
        
        for X_batch, y_batch in dataloader:
            # 1. 搬运数据到 GPU/CPU
            X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
            
            # 2. 梯度清零
            self.optimizer.zero_grad()
            
            # 3. 前向传播
            outputs = self.model(X_batch)
            
            # 4. 计算损失
            loss = self.criterion(outputs, y_batch)
            
            # 5. 反向传播
            loss.backward()
            
            # 6. 更新参数
            self.optimizer.step()
            
            # 统计
            total_loss += loss.item()
            predicted = torch.argmax(outputs, dim=1)
            # 或者如下, 返回(value, index)
            # _, predicted = torch.max(outputs.data, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()
            
        avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0
        avg_acc = correct / total if total > 0 else 0
        return avg_loss, avg_acc

    @torch.no_grad()
    def evaluate(self, dataloader):
        """
        Description
        -----------
        验证集评估函数, 使用 @torch.no_grad() 装饰器自动关闭梯度计算，节省显存
        """
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        for X_batch, y_batch in dataloader:
            X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
            outputs = self.model(X_batch)
            loss = self.criterion(outputs, y_batch)
            total_loss += loss.item()
            predicted = torch.argmax(outputs, dim=1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()
            
        avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0
        avg_acc = correct / total if total > 0 else 0
        return avg_loss, avg_acc

    def save_checkpoint(self, path, epoch=None, is_best=False):
        """
        Description
        -----------
        保存model检查点, 包含模型权重和优化器状态等信息, 以便于断点续训或复用模型

        Args
        -----
            path: 检查点文件保存路径 (如果是相对路径，则相对于 self.save_dir)
            epoch: 当前轮数 (用于断点续传)
            is_best: 是否是最佳模型标记
        """
        # 如果 path 是文件名，则拼接 save_dir
        if not os.path.isabs(path) and os.path.dirname(path) == '':
            path = os.path.join(self.save_dir, path)
            
        state = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history
        }
        torch.save(state, path)
        logger.info(f"Checkpoint saved to {path}" + (f" (Epoch {epoch})" if epoch else ""))

    def load_checkpoint(self, path):
        """
        Description
        -----------
        加载模型检查点, 恢复模型权重和优化器状态

        Args
        ----
        path : str
            检查点文件路径

                
        Returns
        -------
            start_epoch: 恢复后的起始 epoch (下一轮从 start_epoch 开始)
        """
        if not os.path.exists(path):
            logger.warning(f"Checkpoint file not found: {path} - Starting from scratch.")
            return 1
            
        logger.info(f"Loading checkpoint from {path}...")
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if self.optimizer and 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
        # 恢复 history
        if 'history' in checkpoint:
            self.history = checkpoint['history']
            
        # 获取保存时的 epoch，下一轮从 epoch + 1 开始
        ckpt_epoch = checkpoint.get('epoch')
        if ckpt_epoch is None:
            ckpt_epoch = 0 # 防御性编程：如果是 None，则假设从头开始（或这是一个纯权重文件）
        start_epoch = ckpt_epoch + 1
        return start_epoch

    def fit(self, dataloader, epochs, val_dataloader=None, print_every=10, patience=None, resume_from=None):
        """   
        Description
        -----------
        训练模型主函数, 包含多个 Epoch 的训练循环, 可选验证集评估

        Args
        ----
        dataloader : torch.utils.data.DataLoader
            训练数据的 DataLoader
        epochs : int
            训练轮数
        val_dataloader : torch.utils.data.DataLoader, optional
            验证数据的 DataLoader, 如果提供则在每个 epoch 训练结束后同时进行评估
        print_every : int
            每隔多少个 epoch 打印一次日志信息
        patience : int, optional
            早停 Patience (需要 val_dataloader)
        resume_from : str, optional
            检查点路径，用于恢复训练
        
        Returns
        -------
        history : dict
            训练历史记录，包含 'loss' 和 'acc' (以及验证集的 'val_loss' 和 'val_acc' 如果提供了验证集)
        """
        start_epoch = 1
        # 1. 断点续传逻辑
        
        if resume_from:
            start_epoch = self.load_checkpoint(resume_from)
            
        if start_epoch > epochs:
            logger.info(f"Training already completed (Current Epoch {start_epoch-1} >= Target {epochs}).")
            return self.history

        logger.info(f"Start training on {self.device} from epoch {start_epoch} to {epochs}")
        
        best_acc = 0.0
        # 尝试从历史中恢复 best_acc，避免逻辑中断
        if self.history.get('val_acc'):
             best_acc = max(self.history['val_acc'])

        # 初始化早停 (注意传入完整路径)
        early_stopping_path = os.path.join(self.save_dir, 'best_model.pth')
        early_stopping = EarlyStopping(patience=patience, path=early_stopping_path) if patience else None
        
        for epoch in range(start_epoch, epochs + 1):
            loss, acc = self.train_epoch(dataloader)
            self.history['loss'].append(loss)
            self.history['acc'].append(acc)
            
            val_msg = ""
            if val_dataloader:
                val_loss, val_acc = self.evaluate(val_dataloader)
                self.history['val_loss'].append(val_loss)
                self.history['val_acc'].append(val_acc)
                val_msg = f" | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
                
                # 早停逻辑 (监控 Val Loss)
                if early_stopping:
                    early_stopping(val_loss, self, epoch=epoch)
                    if early_stopping.early_stop:
                        logger.info("Early stopping triggered")
                        break
                else:
                     # 如果没有 early stopping，手动保存 val acc 最高的
                    if val_acc > best_acc:
                        best_acc = val_acc
                        self.save_checkpoint('best_model.pth', epoch=epoch, is_best=True)

            # 定期保存 "最新" 的检查点 (覆盖式，用于断点续传)
            self.save_checkpoint('last_checkpoint.pth', epoch=epoch, is_best=False)

            if epoch % print_every == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                logger.info(f"Epoch {epoch}/{epochs} | Loss: {loss:.4f} | Acc: {acc:.4f} | LR: {current_lr}{val_msg}")
                
        # 训练结束保存最后一个模型 (可以作为归档)
        self.save_checkpoint('final_model.pth', epoch=epochs, is_best=False)
        return self.history 
