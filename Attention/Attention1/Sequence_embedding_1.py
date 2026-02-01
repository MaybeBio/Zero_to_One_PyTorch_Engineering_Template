# feature/seq_embed.py
# Task: 提取序列的embedding

import torch 
import torch.nn as nn
import esm # LLM model 蛋白质序列大语言模型, pip install fair-esm
from typing import *

class SeqEmbedModel(nn.Module):
    """ 
    Description
    -----------
    序列embedding提取模块
    generate embedding for protein sequence
    
    Notes
    -----
    - 1, 我们这里先使用esm2_t33_650M_UR50D预训练模型做嵌入, 后续考虑用ESM-2/3等更大的model ,参考https://huggingface.co/facebook/esm2_t33_650M_UR50D
    """
    
    def __init__(self, esm_model_path):
        """  
        Description
        -----------
        加载预训练的ESM-2模型(650M)
        Load ESM-2 pretrained model, we use esm2_t33_650M_UR50D here
        
        Args
        ----
            esm_model_path: esm预训练大模型权重路径 
        """
        
        super().__init__()
        
        # 加载预训练的ESM-2模型, 可以预先下载保存好权重文件, 如果没有就重新下载
        # 重新下载的话每次都很麻烦, 但是会避免一些版本兼容性问题
        # 如果要加载预先下载的模型, 也就是使用load_model_and_alphabet_local函数的话, 需要在/miniconda3/envs/ml_base/lib/python3.13/site-packages/esm/pretrained.py 
        # 也就是该model所在环境中该包源码文件pretrained.py中, 在line  70处修改torch.load的参数, 增加weights_only=False, 否则会报错
        # 如果是预先下载保存的话, 要下载2个文件, 一个是模型权重esm2_t33_650M_UR50D.pt, 另一个是回归权重(contact-regression) esm2_t33_650M_UR50D-contact-regression.pt
        
        # 下载参考https://github.com/facebookresearch/esm#available-models
        # load ESM-2 here
        self.esm_model, self.alphabet = esm.pretrained.load_model_and_alphabet_local(esm_model_path) if esm_model_path else esm.pretrained.esm2_t33_650M_UR50D()
        
        # "字符表" 对象alphabet调用的批量数据转换函数, 将序列转换为model推理/训练需要的tensor格式
        self.batch_converter = self.alphabet.get_batch_converter()
        
        # 冻结预训练model 
        # Disable dropout for deterministic results
        self.esm_model.eval()
        
        
        
    def forward(self, seqs: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """   
        Description
        -----------
        预训练model前向推理, 提取序列embedding
        Forward process of ESM-2 model to generate sequence embedding
        
        Args
        ----
            seqs: List[str], 序列列表, 每个元素是一个蛋白质序列字符串
        """   
        
        # refer https://github.com/facebookresearch/esm?tab=readme-ov-file#getting-started-with-this-repo-
        # prepare data like (lablel, str): [("protein1", "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"),("protein3",  "K A <mask> I S Q"), ...]
        # (lablel, str) same above, token 是转换后的tensor, token才是我们想要的embedding
        batch_labels, batch_strs, batch_tokens = self.batch_converter([ (str(i), seq) for i, seq in enumerate(seqs) ])
        # 计算每个序列的真实长度(不含padding)
        batch_lens = (batch_tokens != self.alphabet.padding_idx).sum(1)
        
        # move inputs to the same device as model
        # 将输入数据移动到和模型相同的设备上
        # 首先是获取model所在的设备, next获取迭代器的初始值, 所在的device
        device = next(self.esm_model.parameters()).device
        # 输入数据移动到该设备上
        batch_tokens = batch_tokens.to(device)
        
        # Extract pre-residue representations 
        with torch.no_grad():
            # 通过预训练模型前向推理, ?self.esm_model查看参数
            # 这里的返回值代表每一层的输出, 我们取最后一层(.num_layers)的输出
            # 此时的results 是1个字典['logits', 'representations', 'attentions', 'contacts']
            results = self.esm_model(batch_tokens, repr_layers=[self.esm_model.num_layers], return_contacts=True)
        # token_repr为[batch_size, seq_len, representation_dim] shape的tensor
        # 批量样本数, 批量统一序列长度（含特殊 token+padding）, ESM-2的残基级特征维度
        token_repr = results["representations"][self.esm_model.num_layers]
        
        
        # Generate per-sequence representations via averaging
        # NOTE: token 0 is always a special beginning-of-sequence token, so the first residue is token 1
        # 对每个序列的残基级特征进行平均池化, 得到序列级别的特征表示
        seq_repr = []
        for i, tokens_len in enumerate(batch_lens):
            # 排除掉padding和特殊token的影响, 只对真实残基部分进行平均池化
            # token_repr[i, 1:tokens_len-1] 代表第i个序列的真实残基部分的特征表示, shape为[res_len, representation_dim]
            # mean(0) 对第0维度/行0行沿着行对每一列feature进行平均池化, 得到shape为[representation_dim]的序列级别特征表示, 即ESM-2的1280维embedding
            seq_repr.append(token_repr[i, 1:tokens_len-1].mean(0))
            
        # List of attention maps from each layer    
        # 获取Attention Map
        attentions = results["attentions"]    
        
        # 返回残基级表示和注意力图
        return token_repr, attentions   
