import torch
import torch.nn as nn
import math
import os
import matplotlib.pyplot as plt
import seaborn as sns

# 定义位置编码类
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

# 定义全局注意力类
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
        attn_output, attn_output_weights = self.self_attn(x, x, x, need_weights=True)  # 自注意力机制
        attn_output = attn_output.permute(1, 0, 2)  # 恢复维度顺序为 [batch, seq_len, hidden_dim]
        # 使用 LayerNorm 进行归一化
        attn_output = self.layer_norm(attn_output + x.permute(1, 0, 2))
        context = attn_output.mean(dim=1).unsqueeze(1)  # 获取序列的上下文表示
        return context, attn_output_weights

# 加载数据
train_data_path = "/tmp/AbAgCDR/data/benchmark_processedL5.pt"
if not os.path.exists(train_data_path):
    raise FileNotFoundError(f"The file {train_data_path} does not exist. Please check the path.")

train_data = torch.load(train_data_path)  # 直接加载数据
if 'antigen' not in train_data:
    raise KeyError("The key 'antigen' does not exist in the loaded data.")

antigen_embed = train_data['antigen']

# 检查抗原嵌入的维度
if antigen_embed.dim() != 3:
    raise ValueError("Antigen embeddings should be a 3D tensor (batch_size, seq_len, embed_dim).")

# 初始化模型
model = GlobalAttention(antigen_embed_dim=antigen_embed.size(-1))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
antigen_embed = antigen_embed.to(device)

# 前向传播获取注意力权重
with torch.no_grad():
    # 确保切片不会超出原始序列长度
    max_seq_len = min(10, antigen_embed.size(1))
    context, attention_weights = model(antigen_embed[:20, :max_seq_len])  # 使用前两个样本，且序列长度不超过2

# 注意力权重形状为 (batch_size, num_heads, seq_len, seq_len)
attention_weights = attention_weights.squeeze(0)  # 移除batch维度，只关注第一个样本
head_attention_weights = attention_weights[0, :, :].detach().cpu().numpy()  # 选择第一个头进行演示

# 绘制热力图
plt.figure(figsize=(10, 8))
sns.heatmap(head_attention_weights, annot=True, cmap='viridis')
plt.title('Attention Weights Heatmap', fontsize=14)
plt.xlabel('Sequence Position', fontsize=11)
plt.ylabel('Sequence Position', fontsize=11)
plt.show()