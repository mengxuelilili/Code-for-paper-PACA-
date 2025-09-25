#
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------
# 位置编码
# -----------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=2000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

# -----------------------
# Roformer 注意力
# -----------------------
class RoformerAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=4, dropout_rate=0.1, init_cdr_weight=1.0):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.embed_dim = embed_dim

        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.out = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout_rate)
        self.norm = nn.LayerNorm(embed_dim)

        inv_freq = 1.0 / (10000 ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
        self.register_buffer("inv_freq", inv_freq)

        self.cdr_weight = nn.Parameter(torch.tensor(init_cdr_weight), requires_grad=True)

    def rotate_half(self, x):
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        return torch.stack((-x2, x1), dim=-1).reshape_as(x)

    def apply_rotary_pos_emb(self, x, sin_pos, cos_pos):
        return (x * cos_pos) + (self.rotate_half(x) * sin_pos)

    def forward(self, x, cdr_mask=None):
        batch_size, seq_len, _ = x.size()
        device = x.device

        pos = torch.arange(seq_len, device=device).type_as(self.inv_freq)
        sinusoid = torch.einsum("i,j->ij", pos, self.inv_freq)
        sin_pos = torch.cat([sinusoid.sin(), sinusoid.sin()], dim=-1)
        cos_pos = torch.cat([sinusoid.cos(), sinusoid.cos()], dim=-1)

        sin_pos = sin_pos.unsqueeze(0).unsqueeze(0)
        cos_pos = cos_pos.unsqueeze(0).unsqueeze(0)

        Q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        Q = self.apply_rotary_pos_emb(Q, sin_pos, cos_pos)
        K = self.apply_rotary_pos_emb(K, sin_pos, cos_pos)

        attn_scores = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(self.head_dim)

        if cdr_mask is not None:
            attn_scores = attn_scores + cdr_mask * self.cdr_weight

        attn_probs = torch.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(self.dropout(attn_probs), V)

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        out = self.dropout(self.out(attn_output))

        return self.norm(x + out)

# -----------------------
# Global Attention
# -----------------------
class GlobalAttention(nn.Module):
    def __init__(self, antigen_embed_dim, hidden_dim=500, num_heads=2):
        super().__init__()
        self.fc_antigen = nn.Linear(antigen_embed_dim, hidden_dim)
        self.position_enc = PositionalEncoding(hidden_dim)
        self.self_attn = nn.MultiheadAttention(hidden_dim, num_heads)
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, antigen_features):
        x = self.fc_antigen(antigen_features)
        x = self.position_enc(x)
        x = x.permute(1, 0, 2)
        attn_output, _ = self.self_attn(x, x, x)
        attn_output = attn_output.permute(1, 0, 2)
        attn_output = self.layer_norm(attn_output + x.permute(1, 0, 2))
        return attn_output

# -----------------------
# Cross Attention
# -----------------------
class CrossAttentionWithCheck(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads)

    def forward(self, query, key, value):
        q = query.permute(1, 0, 2)
        k = key.permute(1, 0, 2)
        v = value.permute(1, 0, 2)
        attn_output, _ = self.attention(q, k, v)
        return attn_output.permute(1, 0, 2)

# -----------------------
# Fusion
# -----------------------
class MultimodalFusionLayer(nn.Module):
    def __init__(self, input_dim_light, input_dim_heavy, input_dim_antigen,
                 hidden_dim=512, num_heads=8):
        super().__init__()
        self.fc_light = nn.Linear(input_dim_light, hidden_dim)
        self.fc_heavy = nn.Linear(input_dim_heavy, hidden_dim)
        self.fc_antigen = nn.Linear(input_dim_antigen, hidden_dim)

        self.cross_attention_light = CrossAttentionWithCheck(hidden_dim, num_heads)
        self.cross_attention_heavy = CrossAttentionWithCheck(hidden_dim, num_heads)

        self.gate = nn.Linear(hidden_dim * 4, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, light_chain, heavy_chain, antigen):
        light_proj = self.fc_light(light_chain)
        heavy_proj = self.fc_heavy(heavy_chain)
        antigen_proj = self.fc_antigen(antigen)

        light_attn = self.cross_attention_light(light_proj, antigen_proj, antigen_proj)
        heavy_attn = self.cross_attention_heavy(heavy_proj, antigen_proj, antigen_proj)

        light_feat = torch.cat([light_attn.mean(dim=1), light_attn.max(dim=1)[0]], dim=-1)
        heavy_feat = torch.cat([heavy_attn.mean(dim=1), heavy_attn.max(dim=1)[0]], dim=-1)

        fused = torch.cat([light_feat, heavy_feat], dim=-1)
        fused = self.gate(fused)
        fused = self.norm(F.gelu(fused))
        return fused

# -----------------------
# Attention Pooling
# -----------------------
class AttentionPooling(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.attn_vector = nn.Linear(embed_dim, 1)

    def forward(self, x):
        attn_scores = self.attn_vector(x)        # [batch, seq_len, 1]
        attn_weights = torch.softmax(attn_scores, dim=1)
        pooled = torch.sum(attn_weights * x, dim=1)
        return pooled

# -----------------------
# 新版 Affinity Predictor
# -----------------------
class AffinityPredictorWithTransformer(nn.Module):
    def __init__(self, input_dim=512, num_heads=4, hidden_dim=256,
                 dropout_rate=0.1, num_layers=2):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout_rate,
            batch_first=True,
            activation="gelu"
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.attn_pool = AttentionPooling(input_dim)

        self.fc = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, 1)
        )

    def forward(self, fused_features):
        x = fused_features.unsqueeze(1)
        x = self.transformer(x)
        x = self.attn_pool(x)
        return self.fc(x).squeeze(-1)

# -----------------------
# Combined Model
# -----------------------
class CombinedModel(nn.Module):
    def __init__(self, cdr_boundaries_light, cdr_boundaries_heavy,
                 embed_dim=532, antigen_embed_dim=500,
                 fusion_hidden_dim=256, num_heads=2,
                 affinity_num_heads=4, dropout_rate=0.1):
        super().__init__()
        self.cdr_boundaries_light = cdr_boundaries_light
        self.cdr_boundaries_heavy = cdr_boundaries_heavy

        self.roformer_light = RoformerAttention(embed_dim, num_heads)
        self.roformer_heavy = RoformerAttention(embed_dim, num_heads)

        self.global_attention = GlobalAttention(antigen_embed_dim)

        self.fusion = MultimodalFusionLayer(
            input_dim_light=embed_dim,
            input_dim_heavy=embed_dim,
            input_dim_antigen=antigen_embed_dim,
            hidden_dim=fusion_hidden_dim,
            num_heads=num_heads
        )

        #  AffinityPredictorWithTransformer
        self.affinity_predictor = AffinityPredictorWithTransformer(
            input_dim=fusion_hidden_dim,
            num_heads=affinity_num_heads,
            hidden_dim=fusion_hidden_dim * 2,
            dropout_rate=dropout_rate,
            num_layers=2
        )

    def create_cdr_mask(self, batch_size, num_heads, seq_length, cdr_boundaries, device):
        mask = torch.zeros(seq_length, seq_length, dtype=torch.float32, device=device)
        for cdr in cdr_boundaries:
            for pos in cdr:
                if pos.isdigit():
                    mask[int(pos), int(pos)] = 1
        mask = mask.unsqueeze(0).unsqueeze(0).expand(batch_size, num_heads, seq_length, seq_length)
        return mask

    def forward(self, antibody_light, antibody_heavy, antigen):
        if antibody_light.dim() == 4: antibody_light = antibody_light.squeeze(1)
        if antibody_heavy.dim() == 4: antibody_heavy = antibody_heavy.squeeze(1)
        if antigen.dim() == 4: antigen = antigen.squeeze(1)

        batch_size = antibody_light.size(0)
        seq_len_light = antibody_light.size(1)
        seq_len_heavy = antibody_heavy.size(1)

        cdr_mask_light = self.create_cdr_mask(batch_size, self.roformer_light.num_heads,
                                              seq_len_light, self.cdr_boundaries_light,
                                              antibody_light.device)
        cdr_mask_heavy = self.create_cdr_mask(batch_size, self.roformer_heavy.num_heads,
                                              seq_len_heavy, self.cdr_boundaries_heavy,
                                              antibody_heavy.device)

        light_cdr = self.roformer_light(antibody_light, cdr_mask_light)
        heavy_cdr = self.roformer_heavy(antibody_heavy, cdr_mask_heavy)
        antigen_global = self.global_attention(antigen)

        fused_features = self.fusion(light_cdr, heavy_cdr, antigen_global)
        affinity_value = self.affinity_predictor(fused_features)
        return affinity_value
