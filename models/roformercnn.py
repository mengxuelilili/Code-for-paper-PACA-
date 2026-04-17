# 准确使用的模型架构保存的模型/tmp/AbAgCDR/model/best_modelxin.pth
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
# Roformer 注意力（已修复 CDR 输入）
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

    def forward(self, x, cdr_bias=None):
        batch_size, seq_len, _ = x.size()
        device = x.device

        pos = torch.arange(seq_len, device=device).type_as(self.inv_freq)
        sinusoid = torch.einsum("i,j->ij", pos, self.inv_freq)
        sin_part = torch.sin(sinusoid)
        cos_part = torch.cos(sinusoid)
        sin_pos = torch.cat([sin_part, sin_part], dim=-1).unsqueeze(0).unsqueeze(0)
        cos_pos = torch.cat([cos_part, cos_part], dim=-1).unsqueeze(0).unsqueeze(0)

        Q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        Q = self.apply_rotary_pos_emb(Q, sin_pos, cos_pos)
        K = self.apply_rotary_pos_emb(K, sin_pos, cos_pos)

        attn_scores = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(self.head_dim)

        if cdr_bias is not None:
            # cdr_bias: [batch_size, seq_len]
            cdr_bias = cdr_bias.unsqueeze(1).unsqueeze(1)  # [B, 1, 1, L]
            attn_scores = attn_scores + cdr_bias * self.cdr_weight

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
        residual = x.permute(1, 0, 2)
        attn_output = self.layer_norm(attn_output + residual)
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
        attn_scores = self.attn_vector(x)
        attn_weights = torch.softmax(attn_scores, dim=1)
        pooled = torch.sum(attn_weights * x, dim=1)
        return pooled

# -----------------------
# Affinity Predictor
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
# Combined Model（已修复 CDR bias 逻辑）
# -----------------------
class CombinedModel(nn.Module):
    def __init__(self, cdr_boundaries_light, cdr_boundaries_heavy,
                 embed_dim=532, antigen_embed_dim=500,
                 fusion_hidden_dim=256, num_heads=2,
                 affinity_num_heads=4, dropout_rate=0.1):
        super().__init__()
        self.cdr_boundaries_light = cdr_boundaries_light
        self.cdr_boundaries_heavy = cdr_boundaries_heavy

        self.roformer_light = RoformerAttention(embed_dim, num_heads, dropout_rate)
        self.roformer_heavy = RoformerAttention(embed_dim, num_heads, dropout_rate)

        self.global_attention = GlobalAttention(antigen_embed_dim, hidden_dim=antigen_embed_dim)

        self.fusion = MultimodalFusionLayer(
            input_dim_light=embed_dim,
            input_dim_heavy=embed_dim,
            input_dim_antigen=antigen_embed_dim,
            hidden_dim=fusion_hidden_dim,
            num_heads=num_heads
        )

        self.affinity_predictor = AffinityPredictorWithTransformer(
            input_dim=fusion_hidden_dim,
            num_heads=affinity_num_heads,
            hidden_dim=fusion_hidden_dim * 2,
            dropout_rate=dropout_rate,
            num_layers=2
        )

    def _get_cdr_bias(self, seq_len, cdr_boundaries, device):
        """生成 CDR 位置偏置向量 [seq_len]"""
        bias = torch.zeros(seq_len, device=device)
        for cdr in cdr_boundaries:
            for pos_str in cdr:
                if pos_str.isdigit():
                    idx = int(pos_str)
                    if idx < seq_len:
                        bias[idx] = 1.0
        return bias

    def forward(self, antibody_light, antibody_heavy, antigen):
        # 去除多余的维度 (如 [B, 1, L, D] -> [B, L, D])
        if antibody_light.dim() == 4:
            antibody_light = antibody_light.squeeze(1)
        if antibody_heavy.dim() == 4:
            antibody_heavy = antibody_heavy.squeeze(1)
        if antigen.dim() == 4:
            antigen = antigen.squeeze(1)

        batch_size = antibody_light.size(0)
        seq_len_light = antibody_light.size(1)
        seq_len_heavy = antibody_heavy.size(1)

        # 生成 CDR bias 向量
        cdr_bias_light = self._get_cdr_bias(seq_len_light, self.cdr_boundaries_light, antibody_light.device)
        cdr_bias_heavy = self._get_cdr_bias(seq_len_heavy, self.cdr_boundaries_heavy, antibody_heavy.device)

        # 扩展为 batch 维度
        cdr_bias_light = cdr_bias_light.unsqueeze(0).expand(batch_size, -1)  # [B, L]
        cdr_bias_heavy = cdr_bias_heavy.unsqueeze(0).expand(batch_size, -1)  # [B, L]

        # 通过 RoFormer
        light_cdr = self.roformer_light(antibody_light, cdr_bias_light)
        heavy_cdr = self.roformer_heavy(antibody_heavy, cdr_bias_heavy)

        # 抗原全局建模
        antigen_global = self.global_attention(antigen)

        # 多模态融合
        fused_features = self.fusion(light_cdr, heavy_cdr, antigen_global)

        # 亲和力预测
        affinity_value = self.affinity_predictor(fused_features)
        return affinity_value

# # 消融ROPE
# import math
# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# # -----------------------
# # 位置编码 (保持不变，用于标准注意力对比)
# # -----------------------
# class PositionalEncoding(nn.Module):
#     def __init__(self, d_model, max_len=2000):
#         super().__init__()
#         pe = torch.zeros(max_len, d_model)
#         position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
#         div_term = torch.exp(
#             torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)
#         self.register_buffer('pe', pe.unsqueeze(0))

#     def forward(self, x):
#         # x: [Batch, Seq, Dim]
#         return x + self.pe[:, :x.size(1), :]

# # -----------------------
# # ✅ 替代方案：标准自注意力 (Standard Self-Attention)
# # 用于消融实验：移除 RoPE 和 CDR Bias
# # -----------------------
# class StandardSelfAttention(nn.Module):
#     def __init__(self, embed_dim, num_heads=4, dropout_rate=0.1):
#         super().__init__()
#         assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
#         self.num_heads = num_heads
#         self.head_dim = embed_dim // num_heads
#         self.embed_dim = embed_dim

#         # 使用 PyTorch 原生 MultiheadAttention
#         self.self_attn = nn.MultiheadAttention(
#             embed_dim=embed_dim,
#             num_heads=num_heads,
#             dropout=dropout_rate,
#             batch_first=True  # 输入格式为 [Batch, Seq, Dim]
#         )

#         self.norm1 = nn.LayerNorm(embed_dim)
#         self.norm2 = nn.LayerNorm(embed_dim)
        
#         # 前馈网络 (FFN)
#         self.ffn = nn.Sequential(
#             nn.Linear(embed_dim, embed_dim * 4),
#             nn.GELU(),
#             nn.Dropout(dropout_rate),
#             nn.Linear(embed_dim * 4, embed_dim)
#         )
        
#         self.dropout = nn.Dropout(dropout_rate)

#     def forward(self, x, cdr_bias=None):
#         """
#         注意：cdr_bias 参数保留以兼容接口，但在本类中被忽略 (Unused)。
#         这确保了消融实验除了注意力机制外，其他调用逻辑一致。
#         """
#         # 1. 添加位置编码 (使用正弦位置编码，而非 RoPE)
#         # 为了公平对比，我们在进入注意力之前加一次位置编码
#         # 或者可以在模型外部加，这里为了模块独立性，内部不加，由外部传入带位置编码的x
#         # 但通常 Standard Transformer 会在 Embedding 后加 PosEnc。
#         # 为了模拟原 Roformer 的输入流程，我们假设输入 x 已经包含了某种位置信息，
#         # 或者我们在这里简单加一个正弦编码以确保有位置信息。
#         # 这里选择：不内部加，依赖外部输入或假设序列顺序本身包含信息。
#         # 修正：为了公平，原 Roformer 内部计算 RoPE，这里我们也应该内部处理位置信息。
#         # 但 MultiheadAttention 不自带位置编码。
#         # 策略：在 CombinedModel 中统一加 PositionalEncoding，或者在这里加。
#         # 让我们在这里加一个简单的正弦位置编码，以替代 RoPE 的位置感知能力。
        
#         batch_size, seq_len, _ = x.size()
#         device = x.device
        
#         # 简单的正弦位置编码注入 (一次性生成缓存可优化，此处简化演示)
#         pe = torch.zeros(seq_len, self.embed_dim, device=device)
#         position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, self.embed_dim, 2).float() * (-math.log(10000.0) / self.embed_dim))
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)
#         pe = pe.unsqueeze(0) # [1, Seq, Dim]
        
#         x = x + pe

#         # 2. 多头自注意力
#         attn_output, _ = self.self_attn(x, x, x)
#         x = self.norm1(x + self.dropout(attn_output))

#         # 3. 前馈网络
#         ffn_output = self.ffn(x)
#         x = self.norm2(x + self.dropout(ffn_output))

#         return x

# # -----------------------
# # Global Attention (保持不变)
# # -----------------------
# class GlobalAttention(nn.Module):
#     def __init__(self, antigen_embed_dim, hidden_dim=500, num_heads=2):
#         super().__init__()
#         self.fc_antigen = nn.Linear(antigen_embed_dim, hidden_dim)
#         self.position_enc = PositionalEncoding(hidden_dim)
#         self.self_attn = nn.MultiheadAttention(hidden_dim, num_heads)
#         self.layer_norm = nn.LayerNorm(hidden_dim)

#     def forward(self, antigen_features):
#         x = self.fc_antigen(antigen_features)
#         x = self.position_enc(x)
#         x = x.permute(1, 0, 2)
#         attn_output, _ = self.self_attn(x, x, x)
#         attn_output = attn_output.permute(1, 0, 2)
#         residual = x.permute(1, 0, 2)
#         attn_output = self.layer_norm(attn_output + residual)
#         return attn_output

# # -----------------------
# # Cross Attention (保持不变)
# # -----------------------
# class CrossAttentionWithCheck(nn.Module):
#     def __init__(self, embed_dim, num_heads):
#         super().__init__()
#         self.attention = nn.MultiheadAttention(embed_dim, num_heads)

#     def forward(self, query, key, value):
#         q = query.permute(1, 0, 2)
#         k = key.permute(1, 0, 2)
#         v = value.permute(1, 0, 2)
#         attn_output, _ = self.attention(q, k, v)
#         return attn_output.permute(1, 0, 2)

# # -----------------------
# # Fusion (保持不变)
# # -----------------------
# class MultimodalFusionLayer(nn.Module):
#     def __init__(self, input_dim_light, input_dim_heavy, input_dim_antigen,
#                  hidden_dim=512, num_heads=8):
#         super().__init__()
#         self.fc_light = nn.Linear(input_dim_light, hidden_dim)
#         self.fc_heavy = nn.Linear(input_dim_heavy, hidden_dim)
#         self.fc_antigen = nn.Linear(input_dim_antigen, hidden_dim)

#         self.cross_attention_light = CrossAttentionWithCheck(hidden_dim, num_heads)
#         self.cross_attention_heavy = CrossAttentionWithCheck(hidden_dim, num_heads)

#         self.gate = nn.Linear(hidden_dim * 4, hidden_dim)
#         self.norm = nn.LayerNorm(hidden_dim)

#     def forward(self, light_chain, heavy_chain, antigen):
#         light_proj = self.fc_light(light_chain)
#         heavy_proj = self.fc_heavy(heavy_chain)
#         antigen_proj = self.fc_antigen(antigen)

#         light_attn = self.cross_attention_light(light_proj, antigen_proj, antigen_proj)
#         heavy_attn = self.cross_attention_heavy(heavy_proj, antigen_proj, antigen_proj)

#         light_feat = torch.cat([light_attn.mean(dim=1), light_attn.max(dim=1)[0]], dim=-1)
#         heavy_feat = torch.cat([heavy_attn.mean(dim=1), heavy_attn.max(dim=1)[0]], dim=-1)

#         fused = torch.cat([light_feat, heavy_feat], dim=-1)
#         fused = self.gate(fused)
#         fused = self.norm(F.gelu(fused))
#         return fused

# # -----------------------
# # Attention Pooling (保持不变)
# # -----------------------
# class AttentionPooling(nn.Module):
#     def __init__(self, embed_dim):
#         super().__init__()
#         self.attn_vector = nn.Linear(embed_dim, 1)

#     def forward(self, x):
#         attn_scores = self.attn_vector(x)
#         attn_weights = torch.softmax(attn_scores, dim=1)
#         pooled = torch.sum(attn_weights * x, dim=1)
#         return pooled

# # -----------------------
# # Affinity Predictor (保持不变)
# # -----------------------
# class AffinityPredictorWithTransformer(nn.Module):
#     def __init__(self, input_dim=512, num_heads=4, hidden_dim=256,
#                  dropout_rate=0.1, num_layers=2):
#         super().__init__()
#         encoder_layer = nn.TransformerEncoderLayer(
#             d_model=input_dim,
#             nhead=num_heads,
#             dim_feedforward=hidden_dim * 2,
#             dropout=dropout_rate,
#             batch_first=True,
#             activation="gelu"
#         )
#         self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
#         self.attn_pool = AttentionPooling(input_dim)

#         self.fc = nn.Sequential(
#             nn.Linear(input_dim, 512),
#             nn.LayerNorm(512),
#             nn.GELU(),
#             nn.Dropout(dropout_rate),
#             nn.Linear(512, 256),
#             nn.LayerNorm(256),
#             nn.GELU(),
#             nn.Dropout(dropout_rate),
#             nn.Linear(256, 128),
#             nn.LayerNorm(128),
#             nn.GELU(),
#             nn.Linear(128, 1)
#         )

#     def forward(self, fused_features):
#         x = fused_features.unsqueeze(1)
#         x = self.transformer(x)
#         x = self.attn_pool(x)
#         return self.fc(x).squeeze(-1)

# # -----------------------
# # ✅ Combined Model (已消融 RoFormer)
# # -----------------------
# class CombinedModel(nn.Module):
#     def __init__(self, cdr_boundaries_light, cdr_boundaries_heavy,
#                  embed_dim=532, antigen_embed_dim=500,
#                  fusion_hidden_dim=256, num_heads=2,
#                  affinity_num_heads=4, dropout_rate=0.1):
#         super().__init__()
#         # 保存边界定义（虽然消融版不用了，但为了构造函数签名一致，保留）
#         self.cdr_boundaries_light = cdr_boundaries_light
#         self.cdr_boundaries_heavy = cdr_boundaries_heavy

#         # 🔴 修改点：使用标准自注意力替代 RoFormer
#         self.light_encoder = StandardSelfAttention(embed_dim, num_heads, dropout_rate)
#         self.heavy_encoder = StandardSelfAttention(embed_dim, num_heads, dropout_rate)

#         self.global_attention = GlobalAttention(antigen_embed_dim, hidden_dim=antigen_embed_dim)

#         self.fusion = MultimodalFusionLayer(
#             input_dim_light=embed_dim,
#             input_dim_heavy=embed_dim,
#             input_dim_antigen=antigen_embed_dim,
#             hidden_dim=fusion_hidden_dim,
#             num_heads=num_heads
#         )

#         self.affinity_predictor = AffinityPredictorWithTransformer(
#             input_dim=fusion_hidden_dim,
#             num_heads=affinity_num_heads,
#             hidden_dim=fusion_hidden_dim * 2,
#             dropout_rate=dropout_rate,
#             num_layers=2
#         )

#     def forward(self, antibody_light, antibody_heavy, antigen):
#         # 去除多余的维度 (如 [B, 1, L, D] -> [B, L, D])
#         if antibody_light.dim() == 4:
#             antibody_light = antibody_light.squeeze(1)
#         if antibody_heavy.dim() == 4:
#             antibody_heavy = antibody_heavy.squeeze(1)
#         if antigen.dim() == 4:
#             antigen = antigen.squeeze(1)

#         # 🔴 修改点：移除 CDR Bias 生成逻辑
#         # 原代码会计算 cdr_bias 并传给注意力层，现在直接跳过
#         # 标准注意力层内部会忽略传入的 bias 参数，或者我们根本不调用 bias 相关逻辑
        
#         # 通过标准编码器 (内部自动处理正弦位置编码)
#         # 注意：StandardSelfAttention 的 forward 签名保留了 cdr_bias 参数以兼容，但传 None 即可
#         light_encoded = self.light_encoder(antibody_light, cdr_bias=None)
#         heavy_encoded = self.heavy_encoder(antibody_heavy, cdr_bias=None)

#         # 抗原全局建模
#         antigen_global = self.global_attention(antigen)

#         # 多模态融合
#         fused_features = self.fusion(light_encoded, heavy_encoded, antigen_global)

#         # 亲和力预测
#         affinity_value = self.affinity_predictor(fused_features)
#         return affinity_value

# # 消融CA
# import math
# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# # -----------------------
# # 位置编码 (保持不变)
# # -----------------------
# class PositionalEncoding(nn.Module):
#     def __init__(self, d_model, max_len=2000):
#         super().__init__()
#         pe = torch.zeros(max_len, d_model)
#         position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
#         div_term = torch.exp(
#             torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)
#         self.register_buffer('pe', pe.unsqueeze(0))

#     def forward(self, x):
#         return x + self.pe[:, :x.size(1), :]

# # -----------------------
# # Roformer 注意力 (保持不变)
# # -----------------------
# class RoformerAttention(nn.Module):
#     def __init__(self, embed_dim, num_heads=4, dropout_rate=0.1, init_cdr_weight=1.0):
#         super().__init__()
#         assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
#         self.num_heads = num_heads
#         self.head_dim = embed_dim // num_heads
#         self.embed_dim = embed_dim

#         self.query = nn.Linear(embed_dim, embed_dim)
#         self.key = nn.Linear(embed_dim, embed_dim)
#         self.value = nn.Linear(embed_dim, embed_dim)
#         self.out = nn.Linear(embed_dim, embed_dim)

#         self.dropout = nn.Dropout(dropout_rate)
#         self.norm = nn.LayerNorm(embed_dim)

#         inv_freq = 1.0 / (10000 ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
#         self.register_buffer("inv_freq", inv_freq)

#         self.cdr_weight = nn.Parameter(torch.tensor(init_cdr_weight), requires_grad=True)

#     def rotate_half(self, x):
#         x1 = x[..., ::2]
#         x2 = x[..., 1::2]
#         return torch.stack((-x2, x1), dim=-1).reshape_as(x)

#     def apply_rotary_pos_emb(self, x, sin_pos, cos_pos):
#         return (x * cos_pos) + (self.rotate_half(x) * sin_pos)

#     def forward(self, x, cdr_bias=None):
#         batch_size, seq_len, _ = x.size()
#         device = x.device

#         pos = torch.arange(seq_len, device=device).type_as(self.inv_freq)
#         sinusoid = torch.einsum("i,j->ij", pos, self.inv_freq)
#         sin_part = torch.sin(sinusoid)
#         cos_part = torch.cos(sinusoid)
#         sin_pos = torch.cat([sin_part, sin_part], dim=-1).unsqueeze(0).unsqueeze(0)
#         cos_pos = torch.cat([cos_part, cos_part], dim=-1).unsqueeze(0).unsqueeze(0)

#         Q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
#         K = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
#         V = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

#         Q = self.apply_rotary_pos_emb(Q, sin_pos, cos_pos)
#         K = self.apply_rotary_pos_emb(K, sin_pos, cos_pos)

#         attn_scores = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(self.head_dim)

#         if cdr_bias is not None:
#             cdr_bias = cdr_bias.unsqueeze(1).unsqueeze(1)
#             attn_scores = attn_scores + cdr_bias * self.cdr_weight

#         attn_probs = torch.softmax(attn_scores, dim=-1)
#         attn_output = torch.matmul(self.dropout(attn_probs), V)

#         attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
#         out = self.dropout(self.out(attn_output))

#         return self.norm(x + out)

# # -----------------------
# # Global Attention (保持不变)
# # -----------------------
# class GlobalAttention(nn.Module):
#     def __init__(self, antigen_embed_dim, hidden_dim=500, num_heads=2):
#         super().__init__()
#         self.fc_antigen = nn.Linear(antigen_embed_dim, hidden_dim)
#         self.position_enc = PositionalEncoding(hidden_dim)
#         self.self_attn = nn.MultiheadAttention(hidden_dim, num_heads)
#         self.layer_norm = nn.LayerNorm(hidden_dim)

#     def forward(self, antigen_features):
#         x = self.fc_antigen(antigen_features)
#         x = self.position_enc(x)
#         x = x.permute(1, 0, 2)
#         attn_output, _ = self.self_attn(x, x, x)
#         attn_output = attn_output.permute(1, 0, 2)
#         residual = x.permute(1, 0, 2)
#         attn_output = self.layer_norm(attn_output + residual)
#         return attn_output

# # -----------------------
# # ❌ 已移除：CrossAttentionWithCheck (不再需要)
# # -----------------------

# # -----------------------
# # ✅ 修改版：MultimodalFusionLayer (无 Cross Attention)
# # -----------------------
# class MultimodalFusionLayer(nn.Module):
#     def __init__(self, input_dim_light, input_dim_heavy, input_dim_antigen,
#                  hidden_dim=512, num_heads=8):
#         super().__init__()
#         # 投影层保持不变，用于统一维度
#         self.fc_light = nn.Linear(input_dim_light, hidden_dim)
#         self.fc_heavy = nn.Linear(input_dim_heavy, hidden_dim)
#         self.fc_antigen = nn.Linear(input_dim_antigen, hidden_dim)

#         # 🔴 移除 Cross Attention 层
#         # self.cross_attention_light = ...
#         # self.cross_attention_heavy = ...

#         # ✅ 新增：简单的融合门控机制，替代复杂的交叉注意力
#         # 输入将是：[Light_Pooled, Heavy_Pooled, Antigen_Pooled] 的拼接
#         # 每个部分 pooling 后是 hidden_dim * 2 (mean + max)，总共 3 个部分
#         # 但为了简化对比，我们这里采用：分别 Pooling 后拼接，再过一个大的 MLP
        
#         # 定义 Pooling 后的维度：Mean + Max = 2 * hidden_dim
#         pooled_dim = hidden_dim * 2 
        
#         # 总输入维度：Light(2H) + Heavy(2H) + Antigen(2H) = 6 * hidden_dim
#         total_input_dim = pooled_dim * 3 
        
#         self.fusion_mlp = nn.Sequential(
#             nn.Linear(total_input_dim, hidden_dim * 2),
#             nn.LayerNorm(hidden_dim * 2),
#             nn.GELU(),
#             nn.Dropout(0.1),
#             nn.Linear(hidden_dim * 2, hidden_dim),
#             nn.LayerNorm(hidden_dim),
#             nn.GELU()
#         )

#     def forward(self, light_chain, heavy_chain, antigen):
#         # 1. 投影
#         light_proj = self.fc_light(light_chain)      # [B, L, H]
#         heavy_proj = self.fc_heavy(heavy_chain)      # [B, L, H]
#         antigen_proj = self.fc_antigen(antigen)      # [B, L, H]

#         # 2. 🔴 移除 Cross Attention 交互
#         # 原逻辑：light_attn = cross_attn(light, antigen, antigen)
#         # 新逻辑：直接对各自序列进行 Pooling，假设交互信息在后续 MLP 中隐式学习，或完全缺失
        
#         # 定义一个简单的 Pooling 函数 (Mean + Max)
#         def pool_features(x):
#             mean_pool = x.mean(dim=1)       # [B, H]
#             max_pool = x.max(dim=1)[0]      # [B, H]
#             return torch.cat([mean_pool, max_pool], dim=-1) # [B, 2H]

#         light_feat = pool_features(light_proj)
#         heavy_feat = pool_features(heavy_proj)
#         antigen_feat = pool_features(antigen_proj)

#         # 3. 拼接所有特征
#         fused = torch.cat([light_feat, heavy_feat, antigen_feat], dim=-1) # [B, 6H]

#         # 4. 通过 MLP 融合
#         fused = self.fusion_mlp(fused)
        
#         return fused

# # -----------------------
# # Attention Pooling (保持不变，虽然 Fusion 层内部用了简单 pooling，但这个类留给 Predictor 用)
# # -----------------------
# class AttentionPooling(nn.Module):
#     def __init__(self, embed_dim):
#         super().__init__()
#         self.attn_vector = nn.Linear(embed_dim, 1)

#     def forward(self, x):
#         attn_scores = self.attn_vector(x)
#         attn_weights = torch.softmax(attn_scores, dim=1)
#         pooled = torch.sum(attn_weights * x, dim=1)
#         return pooled

# # -----------------------
# # Affinity Predictor (保持不变)
# # -----------------------
# class AffinityPredictorWithTransformer(nn.Module):
#     def __init__(self, input_dim=512, num_heads=4, hidden_dim=256,
#                  dropout_rate=0.1, num_layers=2):
#         super().__init__()
#         encoder_layer = nn.TransformerEncoderLayer(
#             d_model=input_dim,
#             nhead=num_heads,
#             dim_feedforward=hidden_dim * 2,
#             dropout=dropout_rate,
#             batch_first=True,
#             activation="gelu"
#         )
#         self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
#         self.attn_pool = AttentionPooling(input_dim)

#         self.fc = nn.Sequential(
#             nn.Linear(input_dim, 512),
#             nn.LayerNorm(512),
#             nn.GELU(),
#             nn.Dropout(dropout_rate),
#             nn.Linear(512, 256),
#             nn.LayerNorm(256),
#             nn.GELU(),
#             nn.Dropout(dropout_rate),
#             nn.Linear(256, 128),
#             nn.LayerNorm(128),
#             nn.GELU(),
#             nn.Linear(128, 1)
#         )

#     def forward(self, fused_features):
#         x = fused_features.unsqueeze(1)
#         x = self.transformer(x)
#         x = self.attn_pool(x)
#         return self.fc(x).squeeze(-1)

# # -----------------------
# # Combined Model (已消融 Cross Attention)
# # -----------------------
# class CombinedModel(nn.Module):
#     def __init__(self, cdr_boundaries_light, cdr_boundaries_heavy,
#                  embed_dim=532, antigen_embed_dim=500,
#                  fusion_hidden_dim=256, num_heads=2,
#                  affinity_num_heads=4, dropout_rate=0.1):
#         super().__init__()
#         self.cdr_boundaries_light = cdr_boundaries_light
#         self.cdr_boundaries_heavy = cdr_boundaries_heavy

#         # RoFormer 保持不变 (这是我们要保留的部分)
#         self.roformer_light = RoformerAttention(embed_dim, num_heads, dropout_rate)
#         self.roformer_heavy = RoformerAttention(embed_dim, num_heads, dropout_rate)

#         self.global_attention = GlobalAttention(antigen_embed_dim, hidden_dim=antigen_embed_dim)

#         # 🔴 使用修改后的 Fusion 层 (无 Cross Attention)
#         self.fusion = MultimodalFusionLayer(
#             input_dim_light=embed_dim,
#             input_dim_heavy=embed_dim,
#             input_dim_antigen=antigen_embed_dim,
#             hidden_dim=fusion_hidden_dim,
#             num_heads=num_heads # 此参数在消融版中未被使用，但保留以兼容构造函数
#         )

#         self.affinity_predictor = AffinityPredictorWithTransformer(
#             input_dim=fusion_hidden_dim,
#             num_heads=affinity_num_heads,
#             hidden_dim=fusion_hidden_dim * 2,
#             dropout_rate=dropout_rate,
#             num_layers=2
#         )

#     def _get_cdr_bias(self, seq_len, cdr_boundaries, device):
#         bias = torch.zeros(seq_len, device=device)
#         for cdr in cdr_boundaries:
#             for pos_str in cdr:
#                 if pos_str.isdigit():
#                     idx = int(pos_str)
#                     if idx < seq_len:
#                         bias[idx] = 1.0
#         return bias

#     def forward(self, antibody_light, antibody_heavy, antigen):
#         if antibody_light.dim() == 4:
#             antibody_light = antibody_light.squeeze(1)
#         if antibody_heavy.dim() == 4:
#             antibody_heavy = antibody_heavy.squeeze(1)
#         if antigen.dim() == 4:
#             antigen = antigen.squeeze(1)

#         batch_size = antibody_light.size(0)
#         seq_len_light = antibody_light.size(1)
#         seq_len_heavy = antibody_heavy.size(1)

#         # CDR Bias 生成 (保持不变，因为 RoFormer 还在)
#         cdr_bias_light = self._get_cdr_bias(seq_len_light, self.cdr_boundaries_light, antibody_light.device)
#         cdr_bias_heavy = self._get_cdr_bias(seq_len_heavy, self.cdr_boundaries_heavy, antibody_heavy.device)

#         cdr_bias_light = cdr_bias_light.unsqueeze(0).expand(batch_size, -1)
#         cdr_bias_heavy = cdr_bias_heavy.unsqueeze(0).expand(batch_size, -1)

#         # RoFormer 编码 (保持不变)
#         light_cdr = self.roformer_light(antibody_light, cdr_bias_light)
#         heavy_cdr = self.roformer_heavy(antibody_heavy, cdr_bias_heavy)

#         # 抗原全局建模 (保持不变)
#         antigen_global = self.global_attention(antigen)

#         # 🔴 多模态融合 (现在内部不包含 Cross Attention)
#         fused_features = self.fusion(light_cdr, heavy_cdr, antigen_global)

#         # 亲和力预测
#         affinity_value = self.affinity_predictor(fused_features)
#         return affinity_value

# # Transformer-style 相对位置编码（Relative Positional Encoding）PWAA+RPE
# import math
# import torch
# import torch.nn as nn
# import torch.nn.functional as F


# class PositionalEncoding(nn.Module):
#     def __init__(self, d_model, max_len=2000):
#         super().__init__()
#         pe = torch.zeros(max_len, d_model)
#         position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
#         div_term = torch.exp(
#             torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)
#         self.register_buffer('pe', pe.unsqueeze(0))

#     def forward(self, x):
#         return x + self.pe[:, :x.size(1), :]


# class RelativePositionalAttention(nn.Module):
#     """
#     使用相对位置编码（RPE）的多头注意力
#     """
#     def __init__(self, embed_dim, num_heads=4, dropout_rate=0.1, max_relative_position=64):
#         super().__init__()
#         assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
#         self.embed_dim = embed_dim
#         self.num_heads = num_heads
#         self.head_dim = embed_dim // num_heads
#         self.dropout = nn.Dropout(dropout_rate)

#         self.q_proj = nn.Linear(embed_dim, embed_dim)
#         self.k_proj = nn.Linear(embed_dim, embed_dim)
#         self.v_proj = nn.Linear(embed_dim, embed_dim)
#         self.out_proj = nn.Linear(embed_dim, embed_dim)

#         self.max_relative_position = max_relative_position
#         self.relative_positions = nn.Embedding(2 * max_relative_position + 1, self.head_dim)

#     def forward(self, x, mask=None):
#         """
#         x: [batch, seq_len, embed_dim]
#         mask: [batch, num_heads, seq_len, seq_len] 可选，用于CDR加权
#         """
#         batch_size, seq_len, _ = x.size()

#         # 线性变换 + 多头拆分
#         Q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
#         K = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
#         V = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

#         # 相对位置编码
#         range_vec = torch.arange(seq_len, device=x.device)
#         distance_mat = range_vec[None, :] - range_vec[:, None]  # [seq_len, seq_len]
#         distance_mat_clipped = torch.clamp(distance_mat, -self.max_relative_position, self.max_relative_position)
#         final_mat = distance_mat_clipped + self.max_relative_position  # shift to 0 ~ 2*max
#         rel_pos_emb = self.relative_positions(final_mat)  # [seq_len, seq_len, head_dim]

#         # 计算注意力分数
#         attn_scores = torch.einsum('bhid,bhjd->bhij', Q, K)  # [batch, heads, seq, seq]
#         rel_scores = torch.einsum('bhid,ijd->bhij', Q, rel_pos_emb)  # 相对位置加权
#         attn_scores = (attn_scores + rel_scores) / math.sqrt(self.head_dim)

#         if mask is not None:
#             attn_scores = attn_scores * (1 + mask)

#         attn_weights = F.softmax(attn_scores, dim=-1)
#         attn_output = torch.einsum('bhij,bhjd->bhid', self.dropout(attn_weights), V)
#         attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
#         return self.out_proj(attn_output)

# class GlobalAttention(nn.Module):
#     def __init__(self, antigen_embed_dim, hidden_dim=500, num_heads=2):
#         super().__init__()
#         self.fc_antigen = nn.Linear(antigen_embed_dim, hidden_dim)
#         self.position_enc = PositionalEncoding(hidden_dim)
#         self.self_attn = nn.MultiheadAttention(hidden_dim, num_heads)
#         self.layer_norm = nn.LayerNorm(hidden_dim)

#     def forward(self, antigen_features):
#         x = self.fc_antigen(antigen_features)  # [batch, seq_len, hidden_dim]
#         x = self.position_enc(x)
#         x = x.permute(1, 0, 2)
#         attn_output, _ = self.self_attn(x, x, x)
#         attn_output = attn_output.permute(1, 0, 2)
#         return self.layer_norm(attn_output + x.permute(1, 0, 2))


# class CrossAttentionWithCheck(nn.Module):
#     def __init__(self, embed_dim, num_heads):
#         super().__init__()
#         self.attention = nn.MultiheadAttention(embed_dim, num_heads)

#     def forward(self, query, key, value):
#         q = query.permute(1, 0, 2)
#         k = key.permute(1, 0, 2)
#         v = value.permute(1, 0, 2)
#         attn_output, _ = self.attention(q, k, v)
#         return attn_output.permute(1, 0, 2)


# class MultimodalFusionLayer(nn.Module):
#     def __init__(self, input_dim_light, input_dim_heavy, input_dim_antigen,
#                  hidden_dim=512, num_heads=8):
#         super().__init__()
#         self.fc_light = nn.Linear(input_dim_light, hidden_dim)
#         self.fc_heavy = nn.Linear(input_dim_heavy, hidden_dim)
#         self.fc_antigen = nn.Linear(input_dim_antigen, hidden_dim)

#         self.cross_attention_light = CrossAttentionWithCheck(hidden_dim, num_heads)
#         self.cross_attention_heavy = CrossAttentionWithCheck(hidden_dim, num_heads)
#         self.cross_attention_antigen = CrossAttentionWithCheck(hidden_dim, num_heads)

#         self.pool = nn.Sequential(
#             nn.AdaptiveAvgPool1d(1),
#             nn.Flatten(start_dim=1)
#         )
#         self.attention_weights = nn.Parameter(torch.ones(4))
#         self.fc_fusion = nn.Sequential(
#             nn.Linear(hidden_dim * 4, hidden_dim),
#             nn.LayerNorm(hidden_dim),
#             nn.GELU(),
#             nn.Dropout(0.1)
#         )

#     def _pool_and_weight(self, x, weight):
#         x_pooled = self.pool(x.permute(0, 2, 1))
#         return x_pooled * torch.sigmoid(weight)

#     def forward(self, light_chain, heavy_chain, antigen):
#         light_proj = self.fc_light(light_chain)
#         heavy_proj = self.fc_heavy(heavy_chain)
#         antigen_proj = self.fc_antigen(antigen)

#         light_antigen = self.cross_attention_light(light_proj, antigen_proj, antigen_proj)
#         heavy_antigen = self.cross_attention_heavy(heavy_proj, antigen_proj, antigen_proj)
#         antigen_light = self.cross_attention_antigen(antigen_proj, light_proj, light_proj)
#         antigen_heavy = self.cross_attention_antigen(antigen_proj, heavy_proj, heavy_proj)

#         pooled_features = [
#             self._pool_and_weight(light_antigen, self.attention_weights[0]),
#             self._pool_and_weight(heavy_antigen, self.attention_weights[1]),
#             self._pool_and_weight(antigen_light, self.attention_weights[2]),
#             self._pool_and_weight(antigen_heavy, self.attention_weights[3])
#         ]
#         return self.fc_fusion(torch.cat(pooled_features, dim=1))


# class DepthwiseSeparableConv1D(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size):
#         super().__init__()
#         self.depthwise = nn.Conv1d(in_channels, in_channels, kernel_size,
#                                    padding='same', groups=in_channels)
#         self.pointwise = nn.Conv1d(in_channels, out_channels, 1)

#     def forward(self, x):
#         return self.pointwise(self.depthwise(x))


# class ResidualConvBlock(nn.Module):
#     def __init__(self, channels, dilation=1, dropout_rate=0.1):
#         super().__init__()
#         self.conv = nn.Sequential(
#             nn.Conv1d(channels, channels, 3, padding='same', dilation=dilation),
#             nn.BatchNorm1d(channels),
#             nn.LeakyReLU(),
#             nn.Dropout(dropout_rate)
#         )

#     def forward(self, x):
#         return x + self.conv(x)


# class LinformerAttention(nn.Module):
#     def __init__(self, dim, seq_len, heads=8, k=64):
#         super().__init__()
#         self.heads = heads
#         self.dim_head = dim // heads
#         self.k = k

#         self.E = nn.Parameter(torch.randn(seq_len, k))
#         self.F = nn.Parameter(torch.randn(seq_len, k))
#         self.to_qkv = nn.Linear(dim, dim * 3)
#         self.to_out = nn.Linear(dim, dim)

#     def forward(self, x):
#         n, b, _ = x.shape
#         h = self.heads
#         qkv = self.to_qkv(x).chunk(3, dim=-1)
#         q, k, v = map(lambda t: t.view(n, b, h, -1).permute(1, 2, 0, 3), qkv)
#         k = torch.einsum('b h n d, n k -> b h k d', k, self.E)
#         v = torch.einsum('b h n d, n k -> b h k d', v, self.F)
#         scale = (self.dim_head ** -0.5)
#         dots = torch.einsum('b h n d, b h k d -> b h n k', q, k) * scale
#         attn = dots.softmax(dim=-1)
#         out = torch.einsum('b h n k, b h k d -> b h n d', attn, v)
#         out = out.permute(2, 0, 1, 3).reshape(n, b, -1)
#         return self.to_out(out)


# class AffinityPredictorCNNWithAttention(nn.Module):
#     def __init__(self, input_dim=256, num_filters=16, num_heads=2, dropout_rate=0.1):
#         super().__init__()
#         self.conv_block = nn.Sequential(
#             DepthwiseSeparableConv1D(1, num_filters, 3),
#             nn.BatchNorm1d(num_filters),
#             nn.LeakyReLU(),
#             nn.Dropout(dropout_rate),
#             ResidualConvBlock(num_filters, 2, dropout_rate),
#             ResidualConvBlock(num_filters, 4, dropout_rate),
#             DepthwiseSeparableConv1D(num_filters, num_filters * 2, 3),
#             nn.BatchNorm1d(num_filters * 2),
#             nn.LeakyReLU(),
#             nn.Dropout(dropout_rate),
#             ResidualConvBlock(num_filters * 2, 8, dropout_rate),
#             ResidualConvBlock(num_filters * 2, 16, dropout_rate)
#         )
#         self.attention = LinformerAttention(num_filters * 2, input_dim, num_heads)
#         self.fc_reg = nn.Sequential(
#             nn.Linear(num_filters * 2, 256),
#             nn.LayerNorm(256),
#             nn.LeakyReLU(),
#             nn.Dropout(dropout_rate),
#             nn.Linear(256, 128),
#             nn.LayerNorm(128),
#             nn.LeakyReLU(),
#             nn.Dropout(dropout_rate),
#             nn.Linear(128, 1)
#         )

#     def forward(self, fused_features):
#         if fused_features.dim() == 2:
#             x = fused_features.unsqueeze(1)
#         else:
#             if fused_features.size(1) != 1:
#                 x = fused_features.unsqueeze(1)
#             else:
#                 x = fused_features
#         x = self.conv_block(x)
#         x_attn = x.permute(2, 0, 1)
#         attn_output = self.attention(x_attn)
#         x = x + attn_output.permute(1, 2, 0)
#         x = F.adaptive_avg_pool1d(x, 1).squeeze(-1)
#         return self.fc_reg(x).squeeze(-1)

# class CombinedModel(nn.Module):
#     def __init__(self, cdr_boundaries_light, cdr_boundaries_heavy, num_heads=2,
#                  embed_dim=532, antigen_embed_dim=500, hidden_dim=256):
#         super().__init__()
#         self.cdr_boundaries_light = cdr_boundaries_light
#         self.cdr_boundaries_heavy = cdr_boundaries_heavy
#         self.roformer_light = RelativePositionalAttention(embed_dim, num_heads)
#         self.roformer_heavy = RelativePositionalAttention(embed_dim, num_heads)
#         self.global_attention = GlobalAttention(antigen_embed_dim)

#         self.fusion = MultimodalFusionLayer(
#             input_dim_light=embed_dim,
#             input_dim_heavy=embed_dim,
#             input_dim_antigen=antigen_embed_dim,
#             hidden_dim=hidden_dim,
#             num_heads=num_heads
#         )
#         self.affinity_predictor = AffinityPredictorCNNWithAttention(
#             input_dim=hidden_dim,
#             num_filters=16,
#             num_heads=2,
#             dropout_rate=0.1
#         )

#     def create_cdr_mask(self, batch_size, num_heads, seq_length, cdr_boundaries, device):
#         mask = torch.zeros(seq_length, seq_length, dtype=torch.float32, device=device)
#         for cdr in cdr_boundaries:
#             for pos in cdr:
#                 if pos.isdigit():
#                     pos_index = int(pos)
#                     mask[pos_index, pos_index] = 1
#         mask = mask.unsqueeze(0).unsqueeze(0)
#         mask = mask.expand(batch_size, num_heads, seq_length, seq_length)
#         return mask

#     def forward(self, antibody_light, antibody_heavy, antigen):
#         if antibody_light.dim() == 4:
#             antibody_light = antibody_light.squeeze(1)
#         if antibody_heavy.dim() == 4:
#             antibody_heavy = antibody_heavy.squeeze(1)
#         if antigen.dim() == 4:
#             antigen = antigen.squeeze(1)

#         batch_size = antibody_heavy.size(0)
#         seq_length_heavy = antibody_heavy.size(1)
#         seq_length_light = antibody_light.size(1)

#         cdr_mask_heavy = self.create_cdr_mask(batch_size, self.roformer_heavy.num_heads, seq_length_heavy,
#                                               self.cdr_boundaries_heavy, antibody_heavy.device)
#         cdr_mask_light = self.create_cdr_mask(batch_size, self.roformer_light.num_heads, seq_length_light,
#                                               self.cdr_boundaries_light, antibody_light.device)

#         light_cdr = self.roformer_light(antibody_light, cdr_mask_light)
#         heavy_cdr = self.roformer_heavy(antibody_heavy, cdr_mask_heavy)
#         antigen_global = self.global_attention(antigen)

#         fused_features = self.fusion(light_cdr, heavy_cdr, antigen_global)
#         affinity_value = self.affinity_predictor(fused_features)
#         return affinity_value

# #RoformerAttention 换成 LearnedPositionalAttention  PWAA+LPE
# import math
# import torch
# import torch.nn as nn
# import torch.nn.functional as F


# class PositionalEncoding(nn.Module):
#     def __init__(self, d_model, max_len=2000):
#         super().__init__()
#         pe = torch.zeros(max_len, d_model)
#         position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
#         div_term = torch.exp(
#             torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)
#         self.register_buffer('pe', pe.unsqueeze(0))

#     def forward(self, x):
#         return x + self.pe[:, :x.size(1), :]


# class LearnedPositionalAttention(nn.Module):
#     """BERT风格：learned absolute positional encoding + Multihead Attention"""
#     def __init__(self, embed_dim, num_heads=4, dropout_rate=0.1,
#                  max_len=2000, init_cdr_weight=1.0):
#         super().__init__()
#         assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
#         self.num_heads = num_heads
#         self.head_dim = embed_dim // num_heads
#         self.embed_dim = embed_dim

#         self.query = nn.Linear(embed_dim, embed_dim)
#         self.key = nn.Linear(embed_dim, embed_dim)
#         self.value = nn.Linear(embed_dim, embed_dim)
#         self.out = nn.Linear(embed_dim, embed_dim)
#         self.dropout = nn.Dropout(dropout_rate)

#         # learned positional embedding (BERT风格)
#         self.pos_embedding = nn.Embedding(max_len, embed_dim)

#         # CDR 权重
#         self.cdr_weight = nn.Parameter(torch.tensor(init_cdr_weight), requires_grad=True)

#     def forward(self, x, cdr_mask=None):
#         batch_size, seq_len, _ = x.size()
#         device = x.device

#         # 添加learned positional embedding
#         pos_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, seq_len)
#         pos_emb = self.pos_embedding(pos_ids)   # [batch, seq_len, embed_dim]
#         x = x + pos_emb

#         # Q, K, V
#         Q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
#         K = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
#         V = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

#         # Scaled dot-product attention
#         attn_scores = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(self.head_dim)
#         if cdr_mask is not None:
#             attn_scores = attn_scores * (1 + cdr_mask * self.cdr_weight)

#         attn_weights = torch.softmax(attn_scores, dim=-1)
#         attn_output = torch.matmul(self.dropout(attn_weights), V)

#         # 合并多头
#         attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
#         return self.out(attn_output)


# class GlobalAttention(nn.Module):
#     def __init__(self, antigen_embed_dim, hidden_dim=500, num_heads=2):
#         super().__init__()
#         self.fc_antigen = nn.Linear(antigen_embed_dim, hidden_dim)
#         self.position_enc = PositionalEncoding(hidden_dim)
#         self.self_attn = nn.MultiheadAttention(hidden_dim, num_heads)
#         self.layer_norm = nn.LayerNorm(hidden_dim)

#     def forward(self, antigen_features):
#         x = self.fc_antigen(antigen_features)  # [batch, seq_len, hidden_dim]
#         x = self.position_enc(x)
#         x = x.permute(1, 0, 2)
#         attn_output, _ = self.self_attn(x, x, x)
#         attn_output = attn_output.permute(1, 0, 2)
#         return self.layer_norm(attn_output + x.permute(1, 0, 2))


# class CrossAttentionWithCheck(nn.Module):
#     def __init__(self, embed_dim, num_heads):
#         super().__init__()
#         self.attention = nn.MultiheadAttention(embed_dim, num_heads)

#     def forward(self, query, key, value):
#         q = query.permute(1, 0, 2)
#         k = key.permute(1, 0, 2)
#         v = value.permute(1, 0, 2)
#         attn_output, _ = self.attention(q, k, v)
#         return attn_output.permute(1, 0, 2)


# class MultimodalFusionLayer(nn.Module):
#     def __init__(self, input_dim_light, input_dim_heavy, input_dim_antigen,
#                  hidden_dim=512, num_heads=8):
#         super().__init__()
#         self.fc_light = nn.Linear(input_dim_light, hidden_dim)
#         self.fc_heavy = nn.Linear(input_dim_heavy, hidden_dim)
#         self.fc_antigen = nn.Linear(input_dim_antigen, hidden_dim)

#         self.cross_attention_light = CrossAttentionWithCheck(hidden_dim, num_heads)
#         self.cross_attention_heavy = CrossAttentionWithCheck(hidden_dim, num_heads)
#         self.cross_attention_antigen = CrossAttentionWithCheck(hidden_dim, num_heads)

#         self.pool = nn.Sequential(
#             nn.AdaptiveAvgPool1d(1),
#             nn.Flatten(start_dim=1)
#         )
#         self.attention_weights = nn.Parameter(torch.ones(4))
#         self.fc_fusion = nn.Sequential(
#             nn.Linear(hidden_dim * 4, hidden_dim),
#             nn.LayerNorm(hidden_dim),
#             nn.GELU(),
#             nn.Dropout(0.1)
#         )

#     def _pool_and_weight(self, x, weight):
#         x_pooled = self.pool(x.permute(0, 2, 1))
#         return x_pooled * torch.sigmoid(weight)

#     def forward(self, light_chain, heavy_chain, antigen):
#         light_proj = self.fc_light(light_chain)
#         heavy_proj = self.fc_heavy(heavy_chain)
#         antigen_proj = self.fc_antigen(antigen)

#         light_antigen = self.cross_attention_light(light_proj, antigen_proj, antigen_proj)
#         heavy_antigen = self.cross_attention_heavy(heavy_proj, antigen_proj, antigen_proj)
#         antigen_light = self.cross_attention_antigen(antigen_proj, light_proj, light_proj)
#         antigen_heavy = self.cross_attention_antigen(antigen_proj, heavy_proj, heavy_proj)

#         pooled_features = [
#             self._pool_and_weight(light_antigen, self.attention_weights[0]),
#             self._pool_and_weight(heavy_antigen, self.attention_weights[1]),
#             self._pool_and_weight(antigen_light, self.attention_weights[2]),
#             self._pool_and_weight(antigen_heavy, self.attention_weights[3])
#         ]
#         return self.fc_fusion(torch.cat(pooled_features, dim=1))


# class DepthwiseSeparableConv1D(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size):
#         super().__init__()
#         self.depthwise = nn.Conv1d(in_channels, in_channels, kernel_size,
#                                    padding='same', groups=in_channels)
#         self.pointwise = nn.Conv1d(in_channels, out_channels, 1)

#     def forward(self, x):
#         return self.pointwise(self.depthwise(x))


# class ResidualConvBlock(nn.Module):
#     def __init__(self, channels, dilation=1, dropout_rate=0.1):
#         super().__init__()
#         self.conv = nn.Sequential(
#             nn.Conv1d(channels, channels, 3, padding='same', dilation=dilation),
#             nn.BatchNorm1d(channels),
#             nn.LeakyReLU(),
#             nn.Dropout(dropout_rate)
#         )

#     def forward(self, x):
#         return x + self.conv(x)


# class LinformerAttention(nn.Module):
#     def __init__(self, dim, seq_len, heads=8, k=64):
#         super().__init__()
#         self.heads = heads
#         self.dim_head = dim // heads
#         self.k = k

#         self.E = nn.Parameter(torch.randn(seq_len, k))
#         self.F = nn.Parameter(torch.randn(seq_len, k))
#         self.to_qkv = nn.Linear(dim, dim * 3)
#         self.to_out = nn.Linear(dim, dim)

#     def forward(self, x):
#         n, b, _ = x.shape
#         h = self.heads
#         qkv = self.to_qkv(x).chunk(3, dim=-1)
#         q, k, v = map(lambda t: t.view(n, b, h, -1).permute(1, 2, 0, 3), qkv)
#         k = torch.einsum('b h n d, n k -> b h k d', k, self.E)
#         v = torch.einsum('b h n d, n k -> b h k d', v, self.F)
#         scale = (self.dim_head ** -0.5)
#         dots = torch.einsum('b h n d, b h k d -> b h n k', q, k) * scale
#         attn = dots.softmax(dim=-1)
#         out = torch.einsum('b h n k, b h k d -> b h n d', attn, v)
#         out = out.permute(2, 0, 1, 3).reshape(n, b, -1)
#         return self.to_out(out)


# class AffinityPredictorCNNWithAttention(nn.Module):
#     def __init__(self, input_dim=256, num_filters=16, num_heads=2, dropout_rate=0.1):
#         super().__init__()
#         self.conv_block = nn.Sequential(
#             DepthwiseSeparableConv1D(1, num_filters, 3),
#             nn.BatchNorm1d(num_filters),
#             nn.LeakyReLU(),
#             nn.Dropout(dropout_rate),
#             ResidualConvBlock(num_filters, 2, dropout_rate),
#             ResidualConvBlock(num_filters, 4, dropout_rate),
#             DepthwiseSeparableConv1D(num_filters, num_filters * 2, 3),
#             nn.BatchNorm1d(num_filters * 2),
#             nn.LeakyReLU(),
#             nn.Dropout(dropout_rate),
#             ResidualConvBlock(num_filters * 2, 8, dropout_rate),
#             ResidualConvBlock(num_filters * 2, 16, dropout_rate)
#         )
#         self.attention = LinformerAttention(num_filters * 2, input_dim, num_heads)
#         self.fc_reg = nn.Sequential(
#             nn.Linear(num_filters * 2, 256),
#             nn.LayerNorm(256),
#             nn.LeakyReLU(),
#             nn.Dropout(dropout_rate),
#             nn.Linear(256, 128),
#             nn.LayerNorm(128),
#             nn.LeakyReLU(),
#             nn.Dropout(dropout_rate),
#             nn.Linear(128, 1)
#         )

#     def forward(self, fused_features):
#         if fused_features.dim() == 2:
#             x = fused_features.unsqueeze(1)
#         else:
#             if fused_features.size(1) != 1:
#                 x = fused_features.unsqueeze(1)
#             else:
#                 x = fused_features
#         x = self.conv_block(x)
#         x_attn = x.permute(2, 0, 1)
#         attn_output = self.attention(x_attn)
#         x = x + attn_output.permute(1, 2, 0)
#         x = F.adaptive_avg_pool1d(x, 1).squeeze(-1)
#         return self.fc_reg(x).squeeze(-1)


# class CombinedModel(nn.Module):
#     def __init__(self, cdr_boundaries_light, cdr_boundaries_heavy, num_heads=2,
#                  embed_dim=512, antigen_embed_dim=480, hidden_dim=256):
#         super().__init__()
#         self.cdr_boundaries_light = cdr_boundaries_light
#         self.cdr_boundaries_heavy = cdr_boundaries_heavy

#         # ✅ 使用 LearnedPositionalAttention 替代 RoformerAttention
#         self.roformer_light = LearnedPositionalAttention(embed_dim, num_heads)
#         self.roformer_heavy = LearnedPositionalAttention(embed_dim, num_heads)

#         self.global_attention = GlobalAttention(antigen_embed_dim)
#         self.fusion = MultimodalFusionLayer(
#             input_dim_light=embed_dim,
#             input_dim_heavy=embed_dim,
#             input_dim_antigen=antigen_embed_dim,
#             hidden_dim=hidden_dim,
#             num_heads=num_heads
#         )
#         self.affinity_predictor = AffinityPredictorCNNWithAttention(
#             input_dim=hidden_dim,
#             num_filters=16,
#             num_heads=2,
#             dropout_rate=0.1
#         )

#     def create_cdr_mask(self, batch_size, num_heads, seq_length, cdr_boundaries, device):
#         mask = torch.zeros(seq_length, seq_length, dtype=torch.float32, device=device)
#         for cdr in cdr_boundaries:
#             for pos in cdr:
#                 if pos.isdigit():
#                     pos_index = int(pos)
#                     mask[pos_index, pos_index] = 1
#         mask = mask.unsqueeze(0).unsqueeze(0)
#         return mask.expand(batch_size, num_heads, seq_length, seq_length)

#     def forward(self, antibody_light, antibody_heavy, antigen):
#         if antibody_light.dim() == 4:
#             antibody_light = antibody_light.squeeze(1)
#         if antibody_heavy.dim() == 4:
#             antibody_heavy = antibody_heavy.squeeze(1)
#         if antigen.dim() == 4:
#             antigen = antigen.squeeze(1)

#         batch_size = antibody_heavy.size(0)
#         seq_length_heavy = antibody_heavy.size(1)
#         seq_length_light = antibody_light.size(1)

#         cdr_mask_heavy = self.create_cdr_mask(batch_size, self.roformer_heavy.num_heads,
#                                               seq_length_heavy, self.cdr_boundaries_heavy,
#                                               antibody_heavy.device)
#         cdr_mask_light = self.create_cdr_mask(batch_size, self.roformer_light.num_heads,
#                                               seq_length_light, self.cdr_boundaries_light,
#                                               antibody_light.device)

#         light_cdr = self.roformer_light(antibody_light, cdr_mask_light)
#         heavy_cdr = self.roformer_heavy(antibody_heavy, cdr_mask_heavy)
#         antigen_global = self.global_attention(antigen)

#         fused_features = self.fusion(light_cdr, heavy_cdr, antigen_global)
#         return self.affinity_predictor(fused_features)

