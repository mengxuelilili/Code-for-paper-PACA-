import torch
import torch.nn as nn
import os

# -----------------------------
# 序列嵌入
# -----------------------------
class AntiEmbeddings(nn.Module):
    def __init__(self, vocab_size=22, hidden_size=1024, layer_norm_eps=1e-12):
        super().__init__()
        self.residue_embedding = nn.Embedding(vocab_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(0.1)

    def forward(self, seq):
        emb = self.residue_embedding(seq)
        emb = self.LayerNorm(emb)
        emb = self.dropout(emb)
        return emb

# -----------------------------
# 双向交叉注意力
# -----------------------------
class BidirectionalCrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=1, dropout=0.1, res=False):
        super().__init__()
        self.antibody_to_antigen_attention = nn.MultiheadAttention(embed_dim, num_heads, dropout)
        self.antigen_to_antibody_attention = nn.MultiheadAttention(embed_dim, num_heads, dropout)
        self.antibody_norm = nn.LayerNorm(embed_dim)
        self.antigen_norm = nn.LayerNorm(embed_dim)
        self.res = res

    def forward(self, antibody, antigen):
        ab_q = antibody.permute(1,0,2)
        ag_kv = antigen.permute(1,0,2)
        ab_out,_ = self.antibody_to_antigen_attention(ab_q, ag_kv, ag_kv)
        ab_out = ab_out.permute(1,0,2)

        ag_q = antigen.permute(1,0,2)
        ab_kv = antibody.permute(1,0,2)
        ag_out,_ = self.antigen_to_antibody_attention(ag_q, ab_kv, ab_kv)
        ag_out = ag_out.permute(1,0,2)

        if self.res:
            ab_out = self.antibody_norm(ab_out + antibody)
            ag_out = self.antigen_norm(ag_out + antigen)
        return ab_out, ag_out

# -----------------------------
# 动态池化
# -----------------------------
class Pool(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim
        self.linear = None  # 延迟初始化

    def forward(self, x):
        batch_size = x.size(0)
        x_flat = x.view(batch_size, -1)
        if self.linear is None:
            in_dim = x_flat.size(1)
            out_dim = self.latent_dim * self.latent_dim
            self.linear = nn.Sequential(nn.Linear(in_dim, out_dim), nn.ReLU()).to(x.device)
        out = self.linear(x_flat)
        return out.view(batch_size, self.latent_dim, self.latent_dim)

# -----------------------------
# 主模型
# -----------------------------
class AntiBinder(nn.Module):
    def __init__(self, vocab_size=22, hidden_dim=1024, latent_dim=32, res=False):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        self.seq_emb = AntiEmbeddings(vocab_size, hidden_dim)
        self.bidirectional_crossatt = BidirectionalCrossAttention(hidden_dim, res=res)
        self.change_dim = nn.Sequential(nn.Linear(hidden_dim, latent_dim), nn.ReLU())
        self.pool_ab = Pool(latent_dim)
        self.pool_ag = Pool(latent_dim)
        self.alpha = nn.Parameter(torch.tensor([1.0]))
        self.cls = None  # 动态初始化

    def forward(self, antibody, antigen):
        # antibody/antigen: [seq_tensor] -> embedding
        ab_emb = self.seq_emb(antibody)
        ag_emb = self.seq_emb(antigen)

        ab_emb, ag_emb = self.bidirectional_crossatt(ab_emb, ag_emb)

        ab_emb = self.change_dim(ab_emb)
        ag_emb = self.change_dim(ag_emb)

        ab_pool = self.pool_ab(ab_emb)
        ag_pool = self.pool_ag(ag_emb)

        ab_flat = ab_pool.flatten(1)
        ag_flat = ag_pool.flatten(1)
        concat = torch.cat([ab_flat, self.alpha*ag_flat], dim=1)

        if self.cls is None:
            self.cls = nn.Sequential(
                nn.Linear(concat.size(1), 1024),
                nn.ReLU(),
                nn.Linear(1024,1),
                nn.Sigmoid()
            ).to(concat.device)

        return self.cls(concat)