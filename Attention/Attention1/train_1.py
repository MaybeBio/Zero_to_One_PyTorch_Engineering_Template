
import argparse
import yaml # pip install pyyaml
import torch
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping # 保存模型、监控学习率、早停
from lightning.pytorch.loggers import TensorBoardLogger
import os

# IDRinter modules
from .model import IDRinter_Seq
from .dataset import PPIDataModule

def main(args):
    # 1. Load Configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set seed for reproducibility
    L.seed_everything(config.get('seed', 2026))
    
    # 2. Prepare DataModule
    print("Loading Data...")
    dm = PPIDataModule(
        data_path=config['data']['path'], 
        batch_size=config['training']['batch_size'],
        num_workers=config['training'].get('num_workers', 4)
    )
    
    # 3. Model Initialization
    print("Initializing Model...")
    model = IDRinter_Seq(
        esm_model_path=config['model'].get('esm_path', None),
        num_decoder_layers=config['model'].get('layers', 2),
        embed_dim=config['model'].get('embed_dim', 1280),
        num_heads=config['model'].get('heads', 8),
        dropout=config['model'].get('dropout', 0.1),
        learning_rate=float(config['training']['lr'])
    )
    
    # 4. Usage of Callbacks
    callbacks = [
        # Checkpoint: Save top 3 models based on val_loss
        ModelCheckpoint(
            dirpath=os.path.join(config['logging']['save_dir'], 'checkpoints'),
            filename='{epoch}-{val_loss:.4f}-{val_acc:.4f}',
            monitor='val_loss',
            mode='min',
            save_top_k=3,
            save_last=True
        ),
        # LR Monitor: Log learning rate changes
        LearningRateMonitor(logging_interval='step'),
        # Early Stopping: Stop if val_loss doesn't improve for 'patience' epochs
        EarlyStopping(
            monitor='val_loss',
            patience=config['training'].get('patience', 10),
            mode='min'
        )
    ]
    
    # 5. Logger
    logger = TensorBoardLogger(
        save_dir=config['logging']['save_dir'],
        name=config['logging']['name']
    )
    
    # 6. Trainer
    trainer = L.Trainer(
        max_epochs=config['training']['epochs'],
        accelerator='auto', # auto-detect GPU/CPU
        devices='auto',
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=10,
        precision=config['training'].get('precision', '16-mixed') # Mixed precision for speed
    )
    
    # 7. Start Training
    print("Starting Training...")
    trainer.fit(model, datamodule=dm)
    
    print("Training Complete!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train IDR-DomINTER Sequence Model")
    parser.add_argument('--config', type=str, default='configs/seq_model.yaml', help='Path to config file')
    args = parser.parse_args()
    main(args)
