import torch
import torch.nn as nn
import torch.nn.functional as F

# 确定设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CDRAttention(nn.Module):
    def __init__(self, input_dim, num_heads, cdr_boundaries=None):
        super(CDRAttention, self).__init__()
        self.multihead_attention = nn.MultiheadAttention(embed_dim=input_dim, num_heads=num_heads, batch_first=True)
        self.cdr_boundaries = cdr_boundaries

    def create_cdr_mask(self, seq_length):
        # 初始化掩码为全1 (加权，而非屏蔽非CDR区域)
        mask = torch.ones(seq_length, dtype=torch.float32)
        # 将CDR区间加权为更高的值，增强注意力
        for start, end in self.cdr_boundaries:
            mask[start:end + 1] = 2  # 权重加倍
        return mask.unsqueeze(0)  # 扩展维度 [1, seq_length]

    def forward(self, x, cdr_mask=None):
        # 将cdr_mask转化为注意力权重，应用加权
        if cdr_mask is not None:
            cdr_mask = cdr_mask.unsqueeze(-1)  # 扩展维度为 [1, seq_length, 1]
            x = x * cdr_mask  # 权重加到输入上

        # 使用multihead_attention
        attn_output, _ = self.multihead_attention(x, x, x)
        return attn_output


class MultiLSTMAttentionCDRModel(nn.Module):
    def __init__(self, input_dim_heavy, input_dim_light, input_dim_antigen, lstm_hidden_dim, num_classes,
                 cdr_boundaries_heavy, cdr_boundaries_light):
        super(MultiLSTMAttentionCDRModel, self).__init__()

        # 使用双向 LSTM (BiLSTM) 捕获双向依赖信息
        self.bilstm_heavy = nn.LSTM(input_size=input_dim_heavy, hidden_size=lstm_hidden_dim, batch_first=True,
                                    bidirectional=True)
        self.bilstm_light = nn.LSTM(input_size=input_dim_light, hidden_size=lstm_hidden_dim, batch_first=True,
                                    bidirectional=True)
        self.bilstm_antigen = nn.LSTM(input_size=input_dim_antigen, hidden_size=lstm_hidden_dim, batch_first=True,
                                      bidirectional=True)

        # CDR 注意力机制
        self.cdr_attention_heavy = CDRAttention(input_dim=2 * lstm_hidden_dim, num_heads=4,
                                                cdr_boundaries=cdr_boundaries_heavy)
        self.cdr_attention_light = CDRAttention(input_dim=2 * lstm_hidden_dim, num_heads=4,
                                                cdr_boundaries=cdr_boundaries_light)

        # 卷积层，进一步提取局部特征
        self.conv_heavy = nn.Conv1d(in_channels=2 * lstm_hidden_dim, out_channels=128, kernel_size=3, padding=1) #128
        self.conv_light = nn.Conv1d(in_channels=2 * lstm_hidden_dim, out_channels=128, kernel_size=3, padding=1) #128
        self.conv_antigen = nn.Conv1d(in_channels=2 * lstm_hidden_dim, out_channels=128, kernel_size=3, padding=1) #128

        # 多头注意力机制
        self.multihead_attention = nn.MultiheadAttention(embed_dim=128, num_heads=4, batch_first=True) #128

        # Dropout层
        self.dropout = nn.Dropout(0.5)

        # 层归一化
        self.layer_norm_heavy = nn.LayerNorm(128) #128
        self.layer_norm_light = nn.LayerNorm(128) #128

        # 定义全连接层
        self.fc = nn.Sequential(
            nn.Linear(128, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x_heavy, x_light, x_antigen):
        # 处理重链序列，使用双向 LSTM
        bilstm_output_heavy, _ = self.bilstm_heavy(x_heavy)
        bilstm_output_light, _ = self.bilstm_light(x_light)
        bilstm_output_antigen, _ = self.bilstm_antigen(x_antigen)

        # CDR 掩码创建
        cdr_mask_heavy = self.cdr_attention_heavy.create_cdr_mask(bilstm_output_heavy.size(1)).to(device)
        cdr_mask_light = self.cdr_attention_light.create_cdr_mask(bilstm_output_light.size(1)).to(device)

        # CDR 注意力加权
        cdr_output_heavy = self.cdr_attention_heavy(bilstm_output_heavy, cdr_mask=cdr_mask_heavy)
        cdr_output_light = self.cdr_attention_light(bilstm_output_light, cdr_mask=cdr_mask_light)

        # 卷积层提取局部特征
        conv_output_heavy = self.conv_heavy(cdr_output_heavy.permute(0, 2, 1))  # 需要转换维度
        conv_output_light = self.conv_light(cdr_output_light.permute(0, 2, 1))
        conv_output_antigen = self.conv_antigen(bilstm_output_antigen.permute(0, 2, 1))

        # 重新转换维度回来
        conv_output_heavy = conv_output_heavy.permute(0, 2, 1)
        conv_output_light = conv_output_light.permute(0, 2, 1)
        conv_output_antigen = conv_output_antigen.permute(0, 2, 1)

        # 层归一化
        conv_output_heavy = self.layer_norm_heavy(conv_output_heavy)
        conv_output_light = self.layer_norm_light(conv_output_light)

        # Dropout
        conv_output_heavy = self.dropout(conv_output_heavy)
        conv_output_light = self.dropout(conv_output_light)

        # 合并卷积后的输出
        combined_output = torch.cat((conv_output_heavy, conv_output_light, conv_output_antigen), dim=1)

        # 多头注意力机制
        attn_output, _ = self.multihead_attention(combined_output, combined_output, combined_output)

        # 平均池化后通过全连接层进行分类
        output = self.fc(attn_output.mean(dim=1))

        return output
