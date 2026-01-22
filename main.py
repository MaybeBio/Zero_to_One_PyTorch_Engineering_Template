# main.py
import argparse
import logging
import torch
import torch.optim as optim
import nnfs # 这个只是为了生成示例数据导入的一个特殊的库, 实际项目中不一定需要
from src.utils import load_config, plot_history, setup_logging, seed_everything
from src.dataset import get_dataloader
from src.model import UniversalMLP # 这里导入我们的model定义
from src.trainer import Trainer
# from src.predict import predict

def parse_args():
    parser = argparse.ArgumentParser(description="A very simple PyTorch Neural Network Project")
    parser.add_argument('--config', type=str, default='configs/config.yaml', help='Path to config file')
    return parser.parse_args()

def main():
    args = parse_args()

    # 1. 基础设施初始化
    setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("Project initialized.")
    
    # 加载配置
    cfg = load_config(args.config)
    seed_everything(cfg['training'].get('seed', 2026))

    # 初始化 nnfs (仅用于生成示例数据), 这个库在实际项目中不一定需要
    nnfs.init()
    
    # 自动选择设备 (GPU 优先)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # 2. 准备数据
    # 工业级：通常会有 Train/Val Split
    # 注意：这里我们演示 batch_size 的使用. 如果在 yaml 里 batch_size 设为 null，则代码 logic 需要处理
    batch_size = cfg['data'].get('batch_size', None)
    
    # 获取训练集和验证集 DataLoader
    # 解释: 这里的划分是在 Dataset 层面进行的，与 batch_size 无关。
    # 我们先将总数据随机划分为 训练集 和 验证集，然后分别封装成 DataLoader。
    # 这样可以在每个 Epoch 结束时用验证集评估模型性能，监控过拟合.
    # 训练集和验证集就划分1次, 后续每一次epoch就是拿这个训练集一直在shuffle取batch训练, 然后每个batch拿固定的这个验证集评估
    train_loader, val_loader = get_dataloader(
        samples=cfg['data']['samples'], 
        classes=cfg['data']['classes'],
        batch_size=batch_size,
        shuffle=True,
        val_split=0.2  # 划分 20% 作为验证集
    )

    # 3. 构建模型 (动态结构)
    model = UniversalMLP(
        input_dim=cfg['model']['input_dim'],
        hidden_dims=cfg['model']['hidden_dims'], 
        output_dim=cfg['model']['output_dim'],
        activation=cfg['model'].get('activation', 'relu'),
        dropout_rate=cfg['model']['dropout_rate']
    )
    logger.info(f"Model structure:\n{model}")
    
    # 4. 定义优化器
    # weight_decay (权重衰减): 即 L2 正则化项.
    # 作用: 限制权重数值的大小，防止模型过拟合. 值越大，惩罚越强
    optimizer = optim.Adam(
        model.parameters(), 
        lr=cfg['training']['learning_rate'], 
        weight_decay=float(cfg['training']['weight_decay'])
    )
    
    # 5. 初始化训练器并开始训练
    trainer = Trainer(
        model, 
        optimizer, 
        device=device,
        save_dir=cfg['training'].get('save_dir', 'checkpoints')
    )
    
    logger.info("Start Training...")
    history = trainer.fit(
        train_loader, 
        epochs=cfg['training']['epochs'], 
        val_dataloader=val_loader, # 传入验证集
        print_every=cfg['training']['print_every']
    )
    
    # 6. 可视化并保存结果
    plot_history(history, save_path=f"{cfg['training'].get('save_dir', 'checkpoints')}/training_curve.png", show=False)
    logger.info("Training Finished.")

if __name__ == "__main__":
    main()

