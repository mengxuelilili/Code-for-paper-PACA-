import matplotlib.pyplot as plt
import seaborn as sns
import torch

def plot_attention_with_cdr(cdr_boundaries_heavy, cdr_boundaries_light, len_heavy, len_light):
    # 分别生成轻链和重链的随机注意力矩阵
    attention_scores_light = torch.rand(len_light, len_light)
    attention_scores_heavy = torch.rand(len_heavy, len_heavy)

    # 初始化轻链和重链的 CDR 掩码
    cdr_mask_light = torch.zeros(len_light, len_light)
    cdr_mask_heavy = torch.zeros(len_heavy, len_heavy)

    # 根据轻链和重链的 CDR 边界设置掩码
    for start, end in cdr_boundaries_light:
        cdr_mask_light[start:end, start:end] = 1  # 轻链 CDR 区域
    for start, end in cdr_boundaries_heavy:
        cdr_mask_heavy[start:end, start:end] = 1  # 重链 CDR 区域

    # 增强 Attention Matrix 的 CDR 区域
    cdr_weight = 2.0  # 可调权重
    enhanced_attention_light = attention_scores_light + cdr_mask_light * cdr_weight
    enhanced_attention_heavy = attention_scores_heavy + cdr_mask_heavy * cdr_weight

    # 可视化轻链的 Attention Matrix
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    sns.heatmap(enhanced_attention_light.numpy(), cmap="Blues", annot=False, cbar=True)
    plt.title("Light Chain Attention Matrix with CDR Enhancement", fontsize=14)
    plt.xlabel("Key Position", fontsize=12)
    plt.ylabel("Query Position", fontsize=12)

    # 可视化重链的 Attention Matrix
    plt.subplot(1, 2, 2)
    sns.heatmap(enhanced_attention_heavy.numpy(), cmap="Blues", annot=False, cbar=True)
    plt.title("Heavy Chain Attention Matrix with CDR Enhancement", fontsize=14)
    plt.xlabel("Key Position", fontsize=12)
    plt.ylabel("Query Position", fontsize=12)

    plt.tight_layout()
    plt.show()

# 定义重链和轻链的 CDR 边界，以及各自的长度
cdr_boundaries_heavy = [(30, 35), (50, 65), (95, 102)]
cdr_boundaries_light = [(24, 34), (50, 56), (89, 97)]
len_heavy = 260  # 重链长度
len_light = 263  # 轻链长度

# 调用函数绘制
plot_attention_with_cdr(cdr_boundaries_heavy, cdr_boundaries_light, len_heavy, len_light)
