#
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
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

class RoformerAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=4, dropout_rate=0.1, init_cdr_weight=1.0):  # 增加注意力头数量
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
        # 初始化旋转参数
        self.register_buffer(
            "inv_freq",
            1.0 / (10000 ** (torch.arange(0, self.head_dim, 1).float() / self.head_dim)))
        self.cdr_weight = nn.Parameter(torch.tensor(init_cdr_weight), requires_grad=True)

    def rotate_half(self, x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    def apply_rotary_pos_emb(self, x, sin_pos, cos_pos):
        return (x * cos_pos) + (self.rotate_half(x) * sin_pos)

    def forward(self, x, cdr_mask=None):
        batch_size, seq_len, _ = x.size()
        device = x.device
        # 生成旋转位置编码
        pos = torch.arange(seq_len, device=device).type_as(self.inv_freq)
        sinusoid = torch.einsum('i,j->ij', pos, self.inv_freq)
        sin_pos, cos_pos = sinusoid.sin(), sinusoid.cos()
        sin_pos = sin_pos.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, dim]
        cos_pos = cos_pos.unsqueeze(0).unsqueeze(0)

        # 线性变换并分头
        Q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # 应用旋转位置编码
        Q_rot = self.apply_rotary_pos_emb(Q, sin_pos, cos_pos)
        K_rot = self.apply_rotary_pos_emb(K, sin_pos, cos_pos)

        # 计算注意力
        attn_scores = torch.matmul(Q_rot, K_rot.transpose(-1, -2)) / math.sqrt(self.head_dim)
        if cdr_mask is not None:
            # attn_scores += cdr_mask * self.cdr_weight
            # 修正为乘法掩码
            attn_scores = attn_scores * (1 + cdr_mask * self.cdr_weight)
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(self.dropout(attn_weights), V)
        # 合并多头输出
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.embed_dim)
        return self.out(attn_output)

class GlobalAttention(nn.Module):
    def __init__(self, antigen_embed_dim, hidden_dim=500, num_heads=2):
        super().__init__()
        self.fc_antigen = nn.Linear(antigen_embed_dim, hidden_dim)
        self.position_enc = PositionalEncoding(hidden_dim)
        self.self_attn = nn.MultiheadAttention(hidden_dim, num_heads)
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, antigen_features):
        x = self.fc_antigen(antigen_features)  # [batch, seq_len, hidden_dim]
        x = self.position_enc(x)  # 添加位置编码
        x = x.permute(1, 0, 2)  # 调整维度顺序为 [seq_len, batch, hidden_dim]
        attn_output, _ = self.self_attn(x, x, x)  # 自注意力机制
        attn_output = attn_output.permute(1, 0, 2)  # 恢复维度顺序为 [batch, seq_len, hidden_dim]
        # 使用 LayerNorm 进行归一化
        attn_output = self.layer_norm(attn_output + x.permute(1, 0, 2))
        return attn_output

class MultimodalFusionLayer(nn.Module):
    def __init__(self, input_dim_light, input_dim_heavy, input_dim_antigen,
                 hidden_dim=512, num_heads=8):
        super().__init__()
        # 维度匹配的全连接层（保持设备一致性）
        self.fc_light = nn.Linear(input_dim_light, hidden_dim)
        self.fc_heavy = nn.Linear(input_dim_heavy, hidden_dim)
        self.fc_antigen = nn.Linear(input_dim_antigen, hidden_dim)

        # 带维度验证的交叉注意力层
        self.cross_attention_light = CrossAttentionWithCheck(hidden_dim, num_heads)
        self.cross_attention_heavy = CrossAttentionWithCheck(hidden_dim, num_heads)
        self.cross_attention_antigen = CrossAttentionWithCheck(hidden_dim, num_heads)

        # 改进的池化层
        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(start_dim=1)
        )

        # 动态权重初始化（使用与模型相同的设备）
        self.attention_weights = nn.Parameter(torch.ones(4))
        self.fc_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )

    def forward(self, light_chain, heavy_chain, antigen):
        # ================= 输入维度验证 =================
        self._validate_input_dims(light_chain, heavy_chain, antigen)  # 关键：确保方法存在

        # ================= 特征投影+残差连接（确保设备一致性）=================
        light_proj = self._project_with_residual(light_chain, self.fc_light)
        heavy_proj = self._project_with_residual(heavy_chain, self.fc_heavy)
        antigen_proj = self._project_with_residual(antigen, self.fc_antigen)

        # ================= 交叉注意力处理 =================
        light_antigen = self._safe_cross_attention(
            self.cross_attention_light, light_proj, antigen_proj
        )
        heavy_antigen = self._safe_cross_attention(
            self.cross_attention_heavy, heavy_proj, antigen_proj
        )
        antigen_light = self._safe_cross_attention(
            self.cross_attention_antigen, antigen_proj, light_proj
        )
        antigen_heavy = self._safe_cross_attention(
            self.cross_attention_antigen, antigen_proj, heavy_proj
        )

        # ================= 特征池化与融合 =================
        pooled_features = [
            self._pool_and_weight(light_antigen, self.attention_weights[0]),
            self._pool_and_weight(heavy_antigen, self.attention_weights[1]),
            self._pool_and_weight(antigen_light, self.attention_weights[2]),
            self._pool_and_weight(antigen_heavy, self.attention_weights[3])
        ]

        fused = self.fc_fusion(torch.cat(pooled_features, dim=1))
        return fused


    def _validate_input_dims(self, *inputs):
        """严格的输入维度验证（必须存在于类中）"""
        assert inputs[0].dim() == 3, f"抗体轻链应为3D tensor，实际维度: {inputs[0].shape}"
        assert inputs[1].dim() == 3, f"抗体重链应为3D tensor，实际维度: {inputs[1].shape}"
        assert inputs[2].dim() == 3, f"抗原应为3D tensor，实际维度: {inputs[2].shape}"

        # print(f"\n[维度验证] 输入形状:")
        # print(f"轻链: {inputs[0].shape} (应: [batch, seq_len, {self.fc_light.in_features}])")
        # print(f"重链: {inputs[1].shape} (应: [batch, seq_len, {self.fc_heavy.in_features}])")
        # print(f"抗原: {inputs[2].shape} (应: [batch, seq_len, {self.fc_antigen.in_features}])")
    def _project_with_residual(self, x, fc_layer):
        """带残差连接的特征投影（修复设备不匹配问题）"""
        projected = fc_layer(x)
        # 残差连接需要维度匹配（确保线性层在相同设备上）
        if x.size(-1) != projected.size(-1):
            # 关键修改：添加device参数确保在相同设备
            residual = nn.Linear(x.size(-1), projected.size(-1), device=x.device)(x)
        else:
            residual = x
        return projected + residual

    def _safe_cross_attention(self, attention_layer, query, key_value):
        """安全的交叉注意力处理"""
        try:
            return attention_layer(query, key_value, key_value)
        except RuntimeError as e:
            raise ValueError(
                f"注意力维度不匹配: Q{query.shape} K{key_value.shape} V{key_value.shape}"
            ) from e

    def _pool_and_weight(self, x, weight):
        """带加权的池化处理"""
        # 输入形状: [batch, seq_len, hidden_dim]
        x_pooled = self.pool(x.permute(0, 2, 1))  # 转换为 [batch, hidden_dim, seq_len]
        return x_pooled * torch.sigmoid(weight)  # 可学习的加权系数

class CrossAttentionWithCheck(nn.Module):
    """带维度检查的交叉注意力层"""
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads)

    def forward(self, query, key, value):
        # 转换为 [seq_len, batch, embed_dim]
        q = query.permute(1, 0, 2)
        k = key.permute(1, 0, 2)
        v = value.permute(1, 0, 2)

        # 执行注意力
        attn_output, _ = self.attention(q, k, v)

        # 恢复原始维度 [batch, seq_len, embed_dim]
        return attn_output.permute(1, 0, 2)

# ================= 先定义基础模块 =================测试下面的调优模块所以先注释
class DepthwiseSeparableConv1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.depthwise = nn.Conv1d(
            in_channels, in_channels, kernel_size,
            padding='same', groups=in_channels
        )
        self.pointwise = nn.Conv1d(in_channels, out_channels, 1)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))

class ResidualConvBlock(nn.Module):
    def __init__(self, channels, dilation=1, dropout_rate=0.1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(channels, channels, 3,
                      padding='same', dilation=dilation),
            nn.BatchNorm1d(channels),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate)
        )

    def forward(self, x):
        return x + self.conv(x)

class LinformerAttention(nn.Module):
    def __init__(self, dim, seq_len, heads=8, k=64):
        super().__init__()
        self.heads = heads
        self.dim_head = dim // heads
        self.k = k

        self.E = nn.Parameter(torch.randn(seq_len, k))
        self.F = nn.Parameter(torch.randn(seq_len, k))
        self.to_qkv = nn.Linear(dim, dim * 3)
        self.to_out = nn.Linear(dim, dim)

    def forward(self, x):
        n, b, _ = x.shape  # [seq_len, batch, dim]
        h = self.heads

        # 生成QKV并分割多头
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.view(n, b, h, -1).permute(1, 2, 0, 3), qkv)

        # 键/值投影
        k = torch.einsum('b h n d, n k -> b h k d', k, self.E)
        v = torch.einsum('b h n d, n k -> b h k d', v, self.F)

        # 注意力计算
        scale = (self.dim_head ** -0.5)
        dots = torch.einsum('b h n d, b h k d -> b h n k', q, k) * scale
        attn = dots.softmax(dim=-1)
        out = torch.einsum('b h n k, b h k d -> b h n d', attn, v)

        # 合并多头
        out = out.permute(2, 0, 1, 3).reshape(n, b, -1)
        return self.to_out(out)

class AffinityPredictorCNNWithAttention(nn.Module):
    def __init__(self, input_dim=256, num_filters=16, num_heads=2, dropout_rate=0.1):
        super().__init__()
        self.conv_block = nn.Sequential(
            DepthwiseSeparableConv1D(1, num_filters, 3),
            nn.BatchNorm1d(num_filters),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate),
            ResidualConvBlock(num_filters, 2, dropout_rate),
            ResidualConvBlock(num_filters, 4, dropout_rate),
            DepthwiseSeparableConv1D(num_filters, num_filters*2, 3),
            nn.BatchNorm1d(num_filters*2),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate),
            # 新增更多残差块
            ResidualConvBlock(num_filters*2, 8, dropout_rate),
            ResidualConvBlock(num_filters*2, 16, dropout_rate)
        )
        self.attention = LinformerAttention(num_filters*2, input_dim, num_heads)
        self.fc_reg = nn.Sequential(
            nn.Linear(num_filters*2, 256),  # 增加全连接层宽度
            nn.LayerNorm(256),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 1)
        )

    def forward(self, fused_features):
        if fused_features.dim() == 2:
            x = fused_features.unsqueeze(1)
        else:
            x = F.adaptive_max_pool1d(fused_features.permute(0, 2, 1), 1).permute(0, 2, 1)
        x = self.conv_block(x)
        x_attn = x.permute(2, 0, 1)
        attn_output = self.attention(x_attn)
        x = x + attn_output.permute(1, 2, 0)
        x = F.adaptive_avg_pool1d(x, 1).squeeze(-1)
        return self.fc_reg(x).squeeze(-1)

class CombinedModel(nn.Module):
    def __init__(self, cdr_boundaries_light, cdr_boundaries_heavy, num_heads=2, embed_dim=532, antigen_embed_dim=500, hidden_dim=256):
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
            hidden_dim=hidden_dim,
            num_heads=num_heads
        )
        self.affinity_predictor = AffinityPredictorCNNWithAttention(input_dim=256, num_filters=16, num_heads=2, dropout_rate=0.1)
    def create_cdr_mask(self, batch_size, num_heads, seq_length, cdr_boundaries, device):
        mask = torch.zeros(seq_length, seq_length, dtype=torch.float32, device=device)
        for cdr in cdr_boundaries:
            for pos in cdr:
                if pos.isdigit():
                    pos_index = int(pos)
                    mask[pos_index, pos_index] = 1
        mask = mask.unsqueeze(0).unsqueeze(0)
        mask = mask.expand(batch_size, num_heads, seq_length, seq_length)
        return mask

    def forward(self, antibody_light, antibody_heavy, antigen):
        if antibody_light.dim() == 4:
            antibody_light = antibody_light.squeeze(1)
        if antibody_heavy.dim() == 4:
            antibody_heavy = antibody_heavy.squeeze(1)
        if antigen.dim() == 4:
            antigen = antigen.squeeze(1)

        batch_size = antibody_heavy.size(0)
        seq_length_heavy = antibody_heavy.size(1)
        seq_length_light = antibody_light.size(1)

        cdr_mask_heavy = self.create_cdr_mask(batch_size, self.roformer_heavy.num_heads, seq_length_heavy,
                                              self.cdr_boundaries_heavy, antibody_heavy.device)
        cdr_mask_light = self.create_cdr_mask(batch_size, self.roformer_light.num_heads, seq_length_light,
                                              self.cdr_boundaries_light, antibody_light.device)

        light_cdr = self.roformer_light(antibody_light, cdr_mask_light)
        heavy_cdr = self.roformer_heavy(antibody_heavy, cdr_mask_heavy)
        antigen_global = self.global_attention(antigen)
        fused_features = self.fusion(light_cdr, heavy_cdr, antigen_global)
        # print('fused_features shape:', fused_features.shape)
        affinity_value = self.affinity_predictor(fused_features)
        return affinity_value
    