import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
import math
import spacy


class TypeAttentiveGCN(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=768, num_dep_types=100):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # 边类型嵌入
        self.type_embedding = nn.Embedding(num_dep_types, input_dim)

        # 注意力参数
        self.W_i = nn.Linear(input_dim, hidden_dim)
        self.W_j = nn.Linear(input_dim, hidden_dim)
        self.W_lp = nn.Linear(input_dim, hidden_dim)
        self.W_g = nn.Linear(hidden_dim, 1)

        # 特征增强
        self.W_enh = nn.Linear(input_dim + input_dim, hidden_dim)  # 拼接输入为 [h_j; q_lp]

        # 聚合前映射
        self.W_k = nn.Linear(hidden_dim, hidden_dim)

        # 残差连接
        self.residual = nn.Linear(input_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, h, adj, dep_types):
        n = h.size(0)
        device = h.device

        # 依存边类型嵌入
        q_lp = self.type_embedding(dep_types)  # [n, n, input_dim]

        # 注意力计算
        h_i = self.W_i(h).unsqueeze(1).expand(-1, n, -1)  # [n, n, hidden_dim]
        h_j = self.W_j(h).unsqueeze(0).expand(n, -1, -1)  # [n, n, hidden_dim]
        q_trans = self.W_lp(q_lp)  # [n, n, hidden_dim]
        y = self.W_g(torch.tanh(h_i + h_j + q_trans)).squeeze(-1)  # [n, n]

        mask = (adj > 0).float()  # 仅在有边处计算
        p = F.softmax(y * mask, dim=-1)  # [n, n]

        # 增强特征（拼接 h_j 与 type embedding）
        h_j_expand = h.unsqueeze(0).expand(n, -1, -1)  # [n, n, input_dim]
        h_aug = torch.cat([h_j_expand, q_lp], dim=-1)  # [n, n, 2*input_dim]
        h_tilde = torch.tanh(self.W_enh(h_aug))  # [n, n, hidden_dim]

        # 加权聚合邻居表示
        p_expand = p.unsqueeze(-1)  # [n, n, 1]
        h_weighted = torch.sum(p_expand * self.W_k(h_tilde), dim=1)  # [n, hidden_dim]

        # 残差 & 归一化
        h_out = self.norm(h_weighted + self.residual(h))
        return F.relu(h_out)


class DependencyProcessor:
    def __init__(self, max_types=100):
        self.nlp = spacy.load("en_core_web_md")
        self.dep_type2id = defaultdict(lambda: len(self.dep_type2id))
        self.max_types = max_types

    def process_sentence(self, sentence):
        doc = self.nlp(sentence)
        n = len(doc)
        adj = torch.zeros((n, n))
        dep_types = torch.zeros((n, n), dtype=torch.long)

        for token in doc:
            i, j = token.i, token.head.i
            if i >= n or j >= n:
                continue
            type_out = token.dep_ + '_out'
            type_in = token.dep_ + '_in'
            idx_out = min(self.dep_type2id[type_out], self.max_types - 1)
            idx_in = min(self.dep_type2id[type_in], self.max_types - 1)
            adj[i, j] = 1
            adj[j, i] = 1
            dep_types[i, j] = idx_out
            dep_types[j, i] = idx_in

        return adj, dep_types
