# import matplotlib.pyplot as plt
# import numpy as np

# # ==========================================
# # 1. 实验数据 (保持不变)
# # ==========================================
# datasets = ['Paddle2021', 'SAbDab', 'AB-Bind', 'SKEMPI2.0']

# L_values = [3, 4, 5, 6, 7]
# L_scores_list = [
#     [0.5934, 0.6071, 0.6398, 0.5880, 0.6145],
#     [0.6527, 0.6120, 0.7100, 0.6131, 0.6799],
#     [0.7802, 0.7567, 0.8042, 0.7734, 0.8008],
#     [0.7382, 0.6352, 0.7428, 0.7606, 0.6746]
# ]
# L_best_val = 5

# Layer_values = [1, 2, 3]
# Layer_scores_list = [
#     [0.6346, 0.6398, 0.5833],
#     [0.5188, 0.7100, 0.5779],
#     [0.6250, 0.8042, 0.7782],
#     [0.6450, 0.7428, 0.7089]
# ]
# Layer_best_val = 2

# Hidden_dim_values = [64, 128, 256]
# Hidden_dim_scores_list = [
#     [0.6144, 0.6047, 0.6398],
#     [0.6285, 0.6300, 0.7100],
#     [0.7873, 0.7404, 0.8042],
#     [0.7291, 0.6499, 0.7428]
# ]
# Hidden_dim_best_val = 256

# # ==========================================
# # 2. 全局设置
# # ==========================================
# plt.rcParams['font.family'] = 'Times New Roman'
# plt.rcParams['mathtext.fontset'] = 'stix'
# plt.rcParams['axes.unicode_minus'] = False

# fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
# colors = ['#2E86AB', '#A23B72', '#E74C3C', '#27AE60']

# # ==========================================
# # 图 A
# # ==========================================
# for i, (dataset, scores) in enumerate(zip(datasets, L_scores_list)):
#     ax1.plot(L_values, scores, marker='o', linestyle='-', color=colors[i], linewidth=2, markersize=8)
#     ax1.scatter([L_best_val], [scores[L_values.index(L_best_val)]], color='red', s=150, zorder=5, marker='*')
#     ax1.text(L_values[-1] + 0.1, scores[-1], dataset, color=colors[i], fontsize=11, fontweight='bold', va='center')

# ax1.text(-0.1, 1.08, 'A', transform=ax1.transAxes, fontsize=18, fontweight='bold', va='top', ha='right')
# ax1.set_title('Sensitivity to Half-window Size (L)', fontsize=16, fontweight='normal', pad=10)
# ax1.set_xlabel('Half-window Size (L)', fontsize=14)
# ax1.set_ylabel('Performance (PCC)', fontsize=14)
# ax1.set_xticks(L_values)
# ax1.grid(True, linestyle='--', alpha=0.6)
# ax1.set_xlim(L_values[0] - 0.2, L_values[-1] + 1.5)

# # ==========================================
# # 图 B (修改点：微调了重叠的标签位置)
# # ==========================================
# for i, (dataset, scores) in enumerate(zip(datasets, Layer_scores_list)):
#     ax2.plot(Layer_values, scores, marker='s', linestyle='-', color=colors[i], linewidth=2, markersize=8)
#     ax2.scatter([Layer_best_val], [scores[Layer_values.index(Layer_best_val)]], color='red', s=150, zorder=5, marker='*')
    
#     # 手动微调：将 Paddle2021 和 SAbDab 的 Y 坐标稍微挪开，防止重叠
#     if dataset == 'Paddle2021':
#         ax2.text(Layer_values[-1] + 0.05, scores[-1] - 0.015, dataset, color=colors[i], fontsize=11, fontweight='bold', va='center')
#     elif dataset == 'SAbDab':
#         ax2.text(Layer_values[-1] + 0.05, scores[-1] + 0.015, dataset, color=colors[i], fontsize=11, fontweight='bold', va='center')
#     else:
#         ax2.text(Layer_values[-1] + 0.05, scores[-1], dataset, color=colors[i], fontsize=11, fontweight='bold', va='center')

# ax2.text(-0.1, 1.08, 'B', transform=ax2.transAxes, fontsize=18, fontweight='bold', va='top', ha='right')
# ax2.set_title('Sensitivity to Number of Layers', fontsize=16, fontweight='normal', pad=10)
# ax2.set_xlabel('Number of Layers', fontsize=14)
# ax2.set_ylabel('Performance (PCC)', fontsize=14)
# ax2.set_xticks(Layer_values)
# ax2.grid(True, linestyle='--', alpha=0.6)
# ax2.set_xlim(Layer_values[0] - 0.2, Layer_values[-1] + 0.5)

# # ==========================================
# # 图 C (修改点：微调了重叠标签，并修正了X轴贴紧度)
# # ==========================================
# for i, (dataset, scores) in enumerate(zip(datasets, Hidden_dim_scores_list)):
#     ax3.plot(Hidden_dim_values, scores, marker='^', linestyle='-', color=colors[i], linewidth=2, markersize=8)
#     ax3.scatter([Hidden_dim_best_val], [scores[Hidden_dim_values.index(Hidden_dim_best_val)]], color='red', s=150, zorder=5, marker='*')
    
#     if dataset == 'Paddle2021':
#         ax3.text(Hidden_dim_values[-1] + 10, scores[-1] - 0.015, dataset, color=colors[i], fontsize=11, fontweight='bold', va='center')
#     elif dataset == 'SAbDab':
#         ax3.text(Hidden_dim_values[-1] + 10, scores[-1] + 0.015, dataset, color=colors[i], fontsize=11, fontweight='bold', va='center')
#     else:
#         ax3.text(Hidden_dim_values[-1] + 10, scores[-1], dataset, color=colors[i], fontsize=11, fontweight='bold', va='center')

# ax3.text(-0.1, 1.08, 'C', transform=ax3.transAxes, fontsize=18, fontweight='bold', va='top', ha='right')
# ax3.set_title('Sensitivity to Fusion Hidden Dim', fontsize=16, fontweight='normal', pad=10)
# ax3.set_xlabel('Fusion Hidden Dimension', fontsize=14)
# ax3.set_ylabel('Performance (PCC)', fontsize=14)
# ax3.set_xticks(Hidden_dim_values)
# ax3.set_xticklabels(Hidden_dim_values, fontsize=12) # 显式设置字体大小
# ax3.tick_params(axis='x', pad=5) # 让X轴标签离轴近一点
# ax3.grid(True, linestyle='--', alpha=0.6)
# ax3.set_xlim(Hidden_dim_values[0] - 15, Hidden_dim_values[-1] + 80)

# plt.tight_layout()
# plt.savefig('/root/autodl-tmp/AbAgCDR/fig9/Hyperparameter_Sensitivity_3in1_Fixed.png', dpi=330, bbox_inches='tight')
# print("✅ 完美修正！审稿人挑不出毛病了。")
# plt.show()


import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# 1. 实验数据
# ==========================================
datasets = ['Paddle2021', 'SAbDab', 'AB-Bind', 'SKEMPI2.0']

L_values = [3, 4, 5, 6, 7]
L_scores_list = [
    [0.5934, 0.6071, 0.6398, 0.5880, 0.6145],
    [0.6527, 0.6120, 0.7100, 0.6131, 0.6799],
    [0.7802, 0.7567, 0.8042, 0.7734, 0.8008],
    [0.7382, 0.6352, 0.7428, 0.7606, 0.6746]
]
L_best_val = 5

Layer_values = [1, 2, 3]
Layer_scores_list = [
    [0.6346, 0.6398, 0.5833],
    [0.5188, 0.7100, 0.5779],
    [0.6250, 0.8042, 0.7782],
    [0.6450, 0.7428, 0.7089]
]
Layer_best_val = 2

Hidden_dim_values = [64, 128, 256]
Hidden_dim_scores_list = [
    [0.6144, 0.6047, 0.6398],
    [0.6285, 0.6300, 0.7100],
    [0.7873, 0.7404, 0.8042],
    [0.7291, 0.6499, 0.7428]
]
Hidden_dim_best_val = 256

# ==========================================
# 2. 全局设置
# ==========================================
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['axes.unicode_minus'] = False

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
colors = ['#2E86AB', '#A23B72', '#E74C3C', '#27AE60']

# ==========================================
# 图 A (保持原样)
# ==========================================
for i, (dataset, scores) in enumerate(zip(datasets, L_scores_list)):
    ax1.plot(L_values, scores, marker='o', linestyle='-', color=colors[i], linewidth=2, markersize=8)
    ax1.scatter([L_best_val], [scores[L_values.index(L_best_val)]], color='red', s=150, zorder=5, marker='*')
    
    if dataset == 'SAbDab':
        ax1.text(L_values[-1] + 0.1, scores[-1] + 0.015, dataset, color=colors[i], fontsize=11, fontweight='bold', va='center')
    else:
        ax1.text(L_values[-1] + 0.1, scores[-1], dataset, color=colors[i], fontsize=11, fontweight='bold', va='center')

ax1.text(-0.1, 1.08, 'A', transform=ax1.transAxes, fontsize=18, fontweight='bold', va='top', ha='right')
ax1.set_title('Sensitivity to Half-window Size (L)', fontsize=16, fontweight='normal', pad=10)
ax1.set_xlabel('Half-window Size (L)', fontsize=14)
ax1.set_ylabel('Performance (PCC)', fontsize=14)
ax1.set_xticks(L_values)
ax1.grid(True, linestyle='--', alpha=0.6)
ax1.set_xlim(L_values[0] - 0.2, L_values[-1] + 1.5) # 保留足够空间放文字

# ==========================================
# 图 B (【核心修正区域】)
# ==========================================
for i, (dataset, scores) in enumerate(zip(datasets, Layer_scores_list)):
    ax2.plot(Layer_values, scores, marker='s', linestyle='-', color=colors[i], linewidth=2, markersize=8)
    ax2.scatter([Layer_best_val], [scores[Layer_values.index(Layer_best_val)]], color='red', s=150, zorder=5, marker='*')
    
    # 微调文字位置
    if dataset == 'Paddle2021':
        # 稍微往下偏移
        ax2.text(Layer_values[-1] + 0.1, scores[-1] - 0.015, dataset, color=colors[i], fontsize=11, fontweight='bold', va='center')
    elif dataset == 'SAbDab':
        # 稍微往上偏移
        ax2.text(Layer_values[-1] + 0.1, scores[-1] + 0.015, dataset, color=colors[i], fontsize=11, fontweight='bold', va='center')
    else:
        ax2.text(Layer_values[-1] + 0.1, scores[-1], dataset, color=colors[i], fontsize=11, fontweight='bold', va='center')

ax2.text(-0.1, 1.08, 'B', transform=ax2.transAxes, fontsize=18, fontweight='bold', va='top', ha='right')
ax2.set_title('Sensitivity to Number of Layers', fontsize=16, fontweight='normal', pad=10)
ax2.set_xlabel('Number of Layers', fontsize=14)
ax2.set_ylabel('Performance (PCC)', fontsize=14)
ax2.set_xticks(Layer_values)
ax2.grid(True, linestyle='--', alpha=0.6)

# 【核心修改】：将 X 轴最大边界从原来的 3.5 拓宽到 4.0 或 4.2，留出足够空间给最右侧的标签！
ax2.set_xlim(Layer_values[0] - 0.2, Layer_values[-1] + 1.2) 

# ==========================================
# 图 C (保持原样)
# ==========================================
for i, (dataset, scores) in enumerate(zip(datasets, Hidden_dim_scores_list)):
    ax3.plot(Hidden_dim_values, scores, marker='^', linestyle='-', color=colors[i], linewidth=2, markersize=8)
    ax3.scatter([Hidden_dim_best_val], [scores[Hidden_dim_values.index(Hidden_dim_best_val)]], color='red', s=150, zorder=5, marker='*')
    
    if dataset == 'Paddle2021':
        ax3.text(Hidden_dim_values[-1] + 10, scores[-1] - 0.015, dataset, color=colors[i], fontsize=11, fontweight='bold', va='center')
    elif dataset == 'SAbDab':
        ax3.text(Hidden_dim_values[-1] + 10, scores[-1] + 0.015, dataset, color=colors[i], fontsize=11, fontweight='bold', va='center')
    else:
        ax3.text(Hidden_dim_values[-1] + 10, scores[-1], dataset, color=colors[i], fontsize=11, fontweight='bold', va='center')

ax3.text(-0.1, 1.08, 'C', transform=ax3.transAxes, fontsize=18, fontweight='bold', va='top', ha='right')
ax3.set_title('Sensitivity to Fusion Hidden Dim', fontsize=16, fontweight='normal', pad=10)
ax3.set_xlabel('Fusion Hidden Dimension', fontsize=14)
ax3.set_ylabel('Performance (PCC)', fontsize=14)
ax3.set_xticks(Hidden_dim_values)
ax3.set_xticklabels(Hidden_dim_values, fontsize=12)
ax3.tick_params(axis='x', pad=5)
ax3.grid(True, linestyle='--', alpha=0.6)
ax3.set_xlim(Hidden_dim_values[0] - 15, Hidden_dim_values[-1] + 80)

plt.tight_layout()
plt.savefig('/root/autodl-tmp/AbAgCDR/fig9/Hyperparameter_Sensitivity_3in1_Final.png', dpi=330, bbox_inches='tight')
print("✅ 修正完成！图B的文字已完全收纳在图框内。")
plt.show()

