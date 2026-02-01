import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from .feature.sequence_embed import SeqEmbedModel

class IDRDecoderLayer(nn.Module):
    # 原名 TransformerWebBlock
    """
    对应 PyTorch 官方的 TransformerDecoderLayer。
    这是解码器的"单层"定义。
    Standard Transformer Decoder Layer (modified: no masked self-attention).
    """
    def __init__(self, d_model=1280, nhead=8, dropout=0.1):
        super().__init__()
        # 1. Cross-Attention (核心交互: 本蛋白 Looking at 对方蛋白)
        self.cross_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, batch_first=True, dropout=dropout)
        
        # Layer Normalization 1
        self.norm1 = nn.LayerNorm(d_model)
        
        # 2. Feed Forward Network (FFN: 增加非线性变换能力)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4), # 1280 -> 5120
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model)  # 5120 -> 1280
        )
        # Layer Normalization 2
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt, memory, memory_key_padding_mask=None):
        """
        Args:
            tgt: Target (Query) - 本蛋白的特征 (希望被更新)
            memory: Memory (Key/Value) - 对方蛋白的特征 (提供上下文信息)
            memory_key_padding_mask: 对方蛋白的Pad Mask
        """
        # 1. Cross Attention: tgt 看着 memory 更新自己
        # attn_out: [Batch, Len_Q, Dim]
        attn_out, _ = self.cross_attn(query=tgt, key=memory, value=memory, key_padding_mask=memory_key_padding_mask)
        
        # Residual + Norm
        tgt = self.norm1(tgt + self.dropout(attn_out)) 
        
        # 2. FFN
        ffn_out = self.ffn(tgt)
        
        # Residual + Norm
        tgt = self.norm2(tgt + self.dropout(ffn_out))
        return tgt

class IDRDecoder(nn.Module):
    # 原名 CrossAttentionModule
    """
    对应 PyTorch 官方的 TransformerDecoder。
    这是解码器的"整体"定义，负责堆叠层数。
    """
    def __init__(self, d_model=1280, num_layers=2):
        super().__init__()
        # 堆叠积木: num_layers 层 IDRDecoderLayer
        self.layers = nn.ModuleList([
            IDRDecoderLayer(d_model=d_model) 
            for _ in range(num_layers)
        ])
    
    def forward(self, tgt, memory, memory_mask=None):
        output = tgt
        # 逐层流转: Layer 1 的输出 -> Layer 2 的输入
        # Memory (对方特征) 始终不变，作为参考系
        for layer in self.layers:
            output = layer(tgt, memory, memory_key_padding_mask=memory_mask)
        return output

class DeepIDRSeq(pl.LightningModule):
    """
    DeepIDR - Sequence View Model
    
    Architecture / 架构流程:
    1. Encoder: ESM-2 (Frozen) -> 理解单体全长序列
    2. Region Extraction -> 聚焦互作区域
    3. Decoder: IDRDecoder (Cross-Attention) -> 理解相互作用 (A关注B, B关注A)
    4. Classifier -> 综合判别
    """
    def __init__(self, learning_rate=1e-4, esm_weights_path=None):
        super().__init__()
        self.save_hyperparameters()
        
        # 1. Sequence Embedding (Encoder)
        self.embedding_model = SeqEmbedModel(esm_weights_path)
        # Freeze ESM (冻结Encoder)
        for param in self.embedding_model.parameters():
            param.requires_grad = False
        
        embed_dim = self.embedding_model.get_embedding_dim()
        
        # 2. Decoder (Interaction Modules)
        # A2B Decdoer: A 是 Target (被更新), B 是 Memory (被查询)
        self.decoder_a2b = IDRDecoder(d_model=embed_dim, num_layers=2)
        # B2A Decoder: B 是 Target (被更新), A 是 Memory (被查询)
        self.decoder_b2a = IDRDecoder(d_model=embed_dim, num_layers=2)
        
        # 3. Classifier Head (分类头)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim * 4, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(), 
            nn.Dropout(0.3),
            nn.Linear(1024, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 1)
        )
        
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, seq_a, region_a, seq_b, region_b):
        """
        Args:
           seq_a: Protein A 全长序列列表
           region_a: Protein A 关注区域 (start, end) 列表
           seq_b: ...
           region_b: ...
        """
        # ================= Step 1: Encoder (单体特征提取) =================
        full_feat_a, _ = self.embedding_model(seq_a)
        full_feat_b, _ = self.embedding_model(seq_b)
        
        # ================= Step 2: Region Extraction (聚焦) =================
        feat_a, mask_a = self.embedding_model.extract_region_embeddings(full_feat_a, region_a)
        feat_b, mask_b = self.embedding_model.extract_region_embeddings(full_feat_b, region_b)
        
        # 准备 Padding Mask (注意取反)
        pad_mask_a = ~mask_a
        pad_mask_b = ~mask_b
        
        # ================= Step 3: Decoder Interaction (双向交互) =================
        # Decoder A: A 看着 B 思考
        inter_a = self.decoder_a2b(tgt=feat_a, memory=feat_b, memory_mask=pad_mask_b)
        
        # Decoder B: B 看着 A 思考
        inter_b = self.decoder_b2a(tgt=feat_b, memory=feat_a, memory_mask=pad_mask_a)
        
        # ================= Step 4: Pooling & Classification =================
        def masked_mean(tensor, mask):
            mask_broadcast = mask.unsqueeze(-1).float()
            sum_pooled = (tensor * mask_broadcast).sum(dim=1)
            count = mask_broadcast.sum(dim=1).clamp(min=1e-9)
            return sum_pooled / count

        pool_a = masked_mean(feat_a, mask_a)
        pool_b = masked_mean(feat_b, mask_b)
        pool_inter_a = masked_mean(inter_a, mask_a)
        pool_inter_b = masked_mean(inter_b, mask_b)
        
        combined = torch.cat([pool_a, pool_b, pool_inter_a, pool_inter_b], dim=1)
        logits = self.classifier(combined)
        return logits

    def training_step(self, batch, batch_idx):
        # 训练步: 解包数据 -> 前向传播 -> 计算Loss
        # Batch unpacking depends on dataset implementation
        # Assuming batch = (seq_a_list, region_a_list, seq_b_list, region_b_list, labels)
        seq_a, region_a, seq_b, region_b, labels = batch
        logits = self(seq_a, region_a, seq_b, region_b)
        loss = self.criterion(logits.squeeze(), labels.float())
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        # 验证步: 计算Loss 和 Accuracy
        seq_a, region_a, seq_b, region_b, labels = batch
        logits = self(seq_a, region_a, seq_b, region_b)
        loss = self.criterion(logits.squeeze(), labels.float()) # 确保labels转为float
        
        # 计算准确率 (Sigmoid > 0.5 为正类)
        acc = ((torch.sigmoid(logits.squeeze()) > 0.5) == labels).float().mean()
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss"
            }
        }
