import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, HeteroConv, Linear, GATConv
from torch_geometric.data import HeteroData
import math
from torch import Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import numpy as np

class Stage1Model(torch.nn.Module):
    """阶段1模型：预测中药-症状初步关联 - 增强版"""
    def __init__(self, data: HeteroData, hidden_channels, out_channels, num_relations):
        super().__init__()
        self.data = data
        self.hidden_channels = hidden_channels
        
        # 为每个节点类型创建可学习的嵌入
        self.embeddings = nn.ModuleDict({
            node_type: nn.Embedding(data[node_type].num_nodes, hidden_channels)
            for node_type in data.node_types
        })
        
        # 使用更强大的图卷积层
        self.conv1 = HeteroConv({
            rel: GATConv((-1, -1), hidden_channels, heads=2, concat=False, add_self_loops=False)
            for rel in num_relations
        })
        
        # 第二层卷积的输入维度需要与第一层的输出维度一致
        self.conv2 = HeteroConv({
            rel: GATConv(hidden_channels, out_channels, heads=1, concat=False, add_self_loops=False)
            for rel in num_relations
        })
        
        # 增强注意力机制
        self.attention = nn.Sequential(
            nn.Linear(out_channels * 2, out_channels * 4),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(0.3),
            nn.Linear(out_channels * 4, 1)
        )
        
        # 初始化参数
        self.reset_parameters()

    def reset_parameters(self):
        for emb in self.embeddings.values():
            nn.init.xavier_uniform_(emb.weight)
        for conv in self.conv1.convs.values():
            conv.reset_parameters()
        for conv in self.conv2.convs.values():
            conv.reset_parameters()
        for layer in self.attention:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

    def forward(self, x_dict, edge_index_dict):
        # 如果没有提供x_dict，则使用嵌入
        if x_dict is None:
            x_dict = {
                node_type: self.embeddings[node_type](
                    torch.arange(self.data[node_type].num_nodes, device=next(self.parameters()).device)
                )
                for node_type in self.data.node_types
            }
        
        # 第一层卷积 + 激活
        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict = {key: F.leaky_relu(x, negative_slope=0.2) for key, x in x_dict.items()}
        
        # 第二层卷积 + 激活
        x_dict = self.conv2(x_dict, edge_index_dict)
        x_dict = {key: F.leaky_relu(x, negative_slope=0.2) for key, x in x_dict.items()}
        
        return x_dict
    
    def predict_herb_symptom(self, x_dict):
        """改进的预测方法：使用注意力机制"""
        herb_emb = x_dict['herb']
        sympt_emb = x_dict['symptom']
        
        # 创建所有可能的组合
        num_herbs = herb_emb.size(0)
        num_symptoms = sympt_emb.size(0)
        
        # 扩展维度以便配对
        herb_expanded = herb_emb.unsqueeze(1).expand(-1, num_symptoms, -1)
        sympt_expanded = sympt_emb.unsqueeze(0).expand(num_herbs, -1, -1)
        
        # 拼接特征
        combined = torch.cat([herb_expanded, sympt_expanded], dim=-1)
        
        # 应用注意力机制
        attention_scores = self.attention(combined).squeeze(-1)
        scores = torch.sigmoid(attention_scores)
        
        return scores

class PathTrustworthiness(nn.Module):
    """路径可信度评估器"""
    def __init__(self, hidden_channels):
        super().__init__()
        self.trust_net = nn.Sequential(
            nn.Linear(hidden_channels * 5, hidden_channels * 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_channels * 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, path_emb):
        # path_emb: [batch_size, num_nodes, hidden_dim]
        batch_size, num_nodes, hidden_dim = path_emb.shape
        flattened = path_emb.view(batch_size, -1)
        return self.trust_net(flattened)

class PathTransformer(nn.Module):
    """路径评分Transformer模型 - 增强版：加入节点类型嵌入和路径注意力机制"""
    def __init__(self, d_model: int = 128, nhead: int = 8, num_layers: int = 4, dim_feedforward: int = 512, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        
        # 节点类型嵌入：5种节点类型（中药、靶点、通路、疾病、症状）
        self.node_type_embed = nn.Embedding(5, d_model)
        
        # 路径注意力机制
        self.path_attention = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        
        self.encoder = nn.Linear(d_model, d_model)
        
        # 位置编码
        self.positional_encoding = PositionalEncoding(d_model, dropout)
        
        # Transformer编码器
        encoder_layers = TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, batch_first=True
        )
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers)
        
        # 注意力池化
        self.attention_pool = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Tanh(),
            nn.Linear(d_model, 1)
        )
        
        # 评分头部
        self.scorer = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1),
            nn.Sigmoid()
        )
    
    def forward(self, path_emb: Tensor, node_types: Tensor) -> Tensor:
        """
        输入: 
          path_emb [batch_size, num_nodes, d_model]
          node_types [batch_size, num_nodes] 节点类型索引（整数）
        输出: score [batch_size, 1]
        """
        # 添加节点类型嵌入
        type_emb = self.node_type_embed(node_types)
        path_emb = path_emb + type_emb
        
        # 路径注意力机制
        attn_output, _ = self.path_attention(path_emb, path_emb, path_emb)
        path_emb = path_emb + attn_output
        
        # 线性编码
        path_emb = self.encoder(path_emb)
        
        # 添加位置编码
        path_emb = self.positional_encoding(path_emb)
        
        # Transformer编码
        encoded = self.transformer_encoder(path_emb)
        
        # 注意力池化
        attn_weights = F.softmax(self.attention_pool(encoded), dim=1)
        pooled = torch.sum(attn_weights * encoded, dim=1)
        
        # 评分
        score = self.scorer(pooled)
        return score

class PositionalEncoding(nn.Module):
    """位置编码模块"""
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 10):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class Stage2Model(torch.nn.Module):
    """阶段2模型：预测解释链及其分数 - 增强版"""
    def __init__(self, hidden_channels):
        super().__init__()
        self.hidden_channels = hidden_channels
        
        # 路径评分Transformer
        self.path_scorer = PathTransformer(
            d_model=hidden_channels,
            nhead=8,
            num_layers=4,
            dim_feedforward=512,
            dropout=0.1
        )
        
        # 路径可信度评估
        self.trust_evaluator = PathTrustworthiness(hidden_channels)
        
        # 路径多样性评估
        self.diversity_scorer = nn.Sequential(
            nn.Linear(hidden_channels * 5, hidden_channels * 2),
            nn.ReLU(),
            nn.Linear(hidden_channels * 2, 1),
            nn.Sigmoid()
        )
        
        # 初始化参数
        self.reset_parameters()

    def reset_parameters(self):
        for name, param in self.path_scorer.named_parameters():
            if 'weight' in name:
                if param.dim() > 1:
                    nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)
        
        for layer in self.trust_evaluator.trust_net:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        
        for layer in self.diversity_scorer:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

    def score_path(self, path_emb, node_types):
        """评分路径: [batch_size, num_nodes, hidden], node_types: [batch_size, num_nodes]"""
        # 原始路径评分
        path_score = self.path_scorer(path_emb, node_types)
        
        # 路径可信度评估
        trust_score = self.trust_evaluator(path_emb)
        
        # 路径多样性评估
        diversity_score = self.diversity_scorer(path_emb.view(path_emb.size(0), -1))
        
        # 综合评分
        return 0.5 * path_score + 0.3 * trust_score + 0.2 * diversity_score