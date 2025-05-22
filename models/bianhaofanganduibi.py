# import pandas as pd
# import matplotlib.pyplot as plt
# import numpy as np
#
# # 准备数据（重新组织数据结构）
# data = {
#     'SAbDab': {
#         'AbM': {'mse': 0.8679, 'rmse': 0.9316, 'mae': 0.6516, 'R2': 0.1743, 'pearson': 0.5581},
#         'Imgt': {'mse': 0.8624, 'rmse': 0.9286, 'mae': 0.6495, 'R2': 0.1796, 'pearson': 0.5601},
#         'Kabat': {'mse': 0.8758, 'rmse': 0.9359, 'mae': 0.6555, 'R2': 0.1668, 'pearson': 0.5537},
#         'chothia': {'mse': 0.8668, 'rmse': 0.9310, 'mae': 0.6513, 'R2': 0.1754, 'pearson': 0.5585}
#     },
#     'AB_Bind': {
#         'AbM': {'mse': 0.8056, 'rmse': 0.8975, 'mae': 0.6588, 'R2': 0.4721, 'pearson': 0.6997},
#         'Imgt': {'mse': 0.8045, 'rmse': 0.8969, 'mae': 0.6582, 'R2': 0.4728, 'pearson': 0.7005},
#         'Kabat': {'mse': 0.8042, 'rmse': 0.8968, 'mae': 0.6593, 'R2': 0.4730, 'pearson': 0.6998},
#         'chothia': {'mse': 0.8048, 'rmse': 0.8971, 'mae': 0.6584, 'R2': 0.4726, 'pearson': 0.7003}
#     },
#     'Benchmark': {
#         'AbM': {'mse': 0.5211, 'rmse': 0.7219, 'mae': 0.5006, 'R2': 0.4509, 'pearson': 0.7539},
#         'Imgt': {'mse': 0.5282, 'rmse': 0.7268, 'mae': 0.5045, 'R2': 0.4434, 'pearson': 0.7525},
#         'Kabat': {'mse': 0.5481, 'rmse': 0.7403, 'mae': 0.5119, 'R2': 0.4225, 'pearson': 0.7428},
#         'chothia': {'mse': 0.5257, 'rmse': 0.7250, 'mae': 0.5017, 'R2': 0.4461, 'pearson': 0.7537}
#     },
#     'SKEMPI 2.0': {
#         'AbM': {'mse': 0.5921, 'rmse': 0.7695, 'mae': 0.5657, 'R2': 0.3094, 'pearson': 0.5692},
#         'Imgt': {'mse': 0.6001, 'rmse': 0.7746, 'mae': 0.5565, 'R2': 0.3002, 'pearson': 0.5681},
#         'Kabat': {'mse': 0.5893, 'rmse': 0.7677, 'mae': 0.5661, 'R2': 0.3127, 'pearson': 0.5722},
#         'chothia': {'mse': 0.5953, 'rmse': 0.7715, 'mae': 0.5597, 'R2': 0.3058, 'pearson': 0.5689}
#     }
# }
#
# # 转换为DataFrame
# df = pd.DataFrame.from_dict({(i, j): data[i][j]
#                              for i in data.keys()
#                              for j in data[i].keys()},
#                             orient='index')
#
# # 重置索引并重命名
# df = df.reset_index()
# df = df.rename(columns={'level_0': 'Dataset', 'level_1': 'Scheme'})
#
# # 设置绘图风格
# plt.style.use('seaborn')
# colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # 四种颜色对应四种编号方案
# schemes = ['AbM', 'Imgt', 'Kabat', 'chothia']
# metrics = ['mse', 'rmse', 'mae', 'R2', 'pearson']
# datasets = ['SAbDab', 'AB_Bind', 'Benchmark', 'SKEMPI 2.0']
#
# # 创建图表
# fig, axes = plt.subplots(2, 3, figsize=(18, 12))
# axes = axes.flatten()
#
# for i, metric in enumerate(metrics):
#     ax = axes[i]
#
#     # 设置每个柱子的位置
#     x = np.arange(len(datasets))
#     width = 0.2
#
#     # 绘制每种编号方案的柱子
#     for j, scheme in enumerate(schemes):
#         values = df[df['Scheme'] == scheme].set_index('Dataset')[metric]
#         ax.bar(x + j * width, values.loc[datasets], width, label=scheme, color=colors[j])
#
#     # 添加标签和标题
#     ax.set_title(f'{metric.upper()} Comparison', fontsize=12)
#     ax.set_xticks(x + width * 1.5)
#     ax.set_xticklabels(datasets)
#     ax.grid(True, linestyle='--', alpha=0.6)
#
#     # 特殊处理R2和pearson的y轴范围
#     if metric == 'R2':
#         ax.set_ylim(0, 0.6)
#     elif metric == 'pearson':
#         ax.set_ylim(0.4, 0.8)
#
# # 移除多余的子图
# for j in range(i + 1, len(axes)):
#     fig.delaxes(axes[j])
#
# # 添加图例
# handles, labels = ax.get_legend_handles_labels()
# fig.legend(handles, labels, loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.05))
#
# plt.tight_layout()
# plt.savefig('all_metrics_comparison.png', bbox_inches='tight', dpi=300)
# plt.show()

# import pandas as pd
# import matplotlib.pyplot as plt
# import numpy as np
#
# # 准备数据
# data = {
#     'SAbDab': {
#         'AbM': {'mse': 0.8679, 'rmse': 0.9316, 'mae': 0.6516, 'R2': 0.1743, 'pearson': 0.5581},
#         'Imgt': {'mse': 0.8624, 'rmse': 0.9286, 'mae': 0.6495, 'R2': 0.1796, 'pearson': 0.5601},
#         'Kabat': {'mse': 0.8758, 'rmse': 0.9359, 'mae': 0.6555, 'R2': 0.1668, 'pearson': 0.5537},
#         'chothia': {'mse': 0.8668, 'rmse': 0.9310, 'mae': 0.6513, 'R2': 0.1754, 'pearson': 0.5585}
#     },
#     'AB_Bind': {
#         'AbM': {'mse': 0.8056, 'rmse': 0.8975, 'mae': 0.6588, 'R2': 0.4721, 'pearson': 0.6997},
#         'Imgt': {'mse': 0.8045, 'rmse': 0.8969, 'mae': 0.6582, 'R2': 0.4728, 'pearson': 0.7005},
#         'Kabat': {'mse': 0.8042, 'rmse': 0.8968, 'mae': 0.6593, 'R2': 0.4730, 'pearson': 0.6998},
#         'chothia': {'mse': 0.8048, 'rmse': 0.8971, 'mae': 0.6584, 'R2': 0.4726, 'pearson': 0.7003}
#     },
#     'Benchmark': {
#         'AbM': {'mse': 0.5211, 'rmse': 0.7219, 'mae': 0.5006, 'R2': 0.4509, 'pearson': 0.7539},
#         'Imgt': {'mse': 0.5282, 'rmse': 0.7268, 'mae': 0.5045, 'R2': 0.4434, 'pearson': 0.7525},
#         'Kabat': {'mse': 0.5481, 'rmse': 0.7403, 'mae': 0.5119, 'R2': 0.4225, 'pearson': 0.7428},
#         'chothia': {'mse': 0.5257, 'rmse': 0.7250, 'mae': 0.5017, 'R2': 0.4461, 'pearson': 0.7537}
#     },
#     'SKEMPI 2.0': {
#         'AbM': {'mse': 0.5921, 'rmse': 0.7695, 'mae': 0.5657, 'R2': 0.3094, 'pearson': 0.5692},
#         'Imgt': {'mse': 0.6001, 'rmse': 0.7746, 'mae': 0.5565, 'R2': 0.3002, 'pearson': 0.5681},
#         'Kabat': {'mse': 0.5893, 'rmse': 0.7677, 'mae': 0.5661, 'R2': 0.3127, 'pearson': 0.5722},
#         'chothia': {'mse': 0.5953, 'rmse': 0.7715, 'mae': 0.5597, 'R2': 0.3058, 'pearson': 0.5689}
#     }
# }
#
# # 转换为DataFrame
# df = pd.DataFrame.from_dict({(i, j): data[i][j]
#                              for i in data.keys()
#                              for j in data[i].keys()},
#                             orient='index').reset_index()
# df.columns = ['Dataset', 'Scheme', 'mse', 'rmse', 'mae', 'R2', 'pearson']
#
# # 设置绘图参数
# plt.style.use('seaborn')
# colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # 四种颜色对应四种编号方案
# schemes = ['AbM', 'Imgt', 'Kabat', 'chothia']
# metrics = ['mse', 'rmse', 'mae', 'R2', 'pearson']
# datasets = ['SAbDab', 'AB_Bind', 'Benchmark', 'SKEMPI 2.0']
# width = 0.2  # 柱子的宽度
#
# # 为每个指标单独绘制图表
# for metric in metrics:
#     plt.figure(figsize=(10, 6))
#
#     # 设置每个柱子的位置
#     x = np.arange(len(datasets))
#
#     # 绘制每种编号方案的柱子
#     for j, scheme in enumerate(schemes):
#         values = df[df['Scheme'] == scheme].set_index('Dataset')[metric]
#         plt.bar(x + j * width, values.loc[datasets], width, label=scheme, color=colors[j])
#
#     # 添加图表元素
#     plt.title(f'{metric.upper()} Comparison Across Datasets', fontsize=14)
#     plt.xlabel('Dataset', fontsize=12)
#     plt.ylabel(metric.upper(), fontsize=12)
#     plt.xticks(x + width * 1.5, datasets)
#     plt.grid(True, linestyle='--', alpha=0.6)
#     plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
#
#     # 调整y轴范围
#     if metric == 'R2':
#         plt.ylim(0, 0.6)
#     elif metric == 'pearson':
#         plt.ylim(0.4, 0.8)
#
#     # 调整布局并保存
#     plt.tight_layout()
#     plt.savefig(f'{metric}_comparison.png', bbox_inches='tight', dpi=300)
#     plt.show()

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 准备数据
data = {
    'SAbDab': {
        'AbM': {'MSE': 0.8679, 'RMSE': 0.9316, 'MAE': 0.6516, 'R2': 0.1743, 'PCC': 0.5581},
        'Imgt': {'MSE': 0.8624, 'RMSE': 0.9286, 'MAE': 0.6495, 'R2': 0.1796, 'PCC': 0.5601},
        'Kabat': {'MSE': 0.8758, 'RMSE': 0.9359, 'MAE': 0.6555, 'R2': 0.1668, 'PCC': 0.5537},
        'chothia': {'MSE': 0.8668, 'RMSE': 0.9310, 'MAE': 0.6513, 'R2': 0.1754, 'PCC': 0.5585}
    },
    'AB_Bind': {
        'AbM': {'MSE': 0.8056, 'RMSE': 0.8975, 'MAE': 0.6588, 'R2': 0.4721, 'PCC': 0.6997},
        'Imgt': {'MSE': 0.8045, 'RMSE': 0.8969, 'MAE': 0.6582, 'R2': 0.4728, 'PCC': 0.7005},
        'Kabat': {'MSE': 0.8042, 'RMSE': 0.8968, 'MAE': 0.6593, 'R2': 0.4730, 'PCC': 0.6998},
        'chothia': {'MSE': 0.8048, 'RMSE': 0.8971, 'MAE': 0.6584, 'R2': 0.4726, 'PCC': 0.7003}
    },
    'Benchmark': {
        'AbM': {'MSE': 0.5211, 'RMSE': 0.7219, 'MAE': 0.5006, 'R2': 0.4509, 'PCC': 0.7539},
        'Imgt': {'MSE': 0.5282, 'RMSE': 0.7268, 'MAE': 0.5045, 'R2': 0.4434, 'PCC': 0.7525},
        'Kabat': {'MSE': 0.5481, 'RMSE': 0.7403, 'MAE': 0.5119, 'R2': 0.4225, 'PCC': 0.7428},
        'chothia': {'MSE': 0.5257, 'RMSE': 0.7250, 'MAE': 0.5017, 'R2': 0.4461, 'PCC': 0.7537}
    },
    'SKEMPI 2.0': {
        'AbM': {'MSE': 0.5921, 'RMSE': 0.7695, 'MAE': 0.5657, 'R2': 0.3094, 'PCC': 0.5692},
        'Imgt': {'MSE': 0.6001, 'RMSE': 0.7746, 'MAE': 0.5565, 'R2': 0.3002, 'PCC': 0.5681},
        'Kabat': {'MSE': 0.5893, 'RMSE': 0.7677, 'MAE': 0.5661, 'R2': 0.3127, 'PCC': 0.5722},
        'chothia': {'MSE': 0.5953, 'RMSE': 0.7715, 'MAE': 0.5597, 'R2': 0.3058, 'PCC': 0.5689}
    }
}

# 转换为DataFrame
df = pd.DataFrame.from_dict({(i, j): data[i][j]
                             for i in data.keys()
                             for j in data[i].keys()},
                            orient='index').reset_index()
df.columns = ['Dataset', 'Scheme', 'MSE', 'RMSE', 'MAE', 'R2', 'PCC']

# 设置绘图参数
plt.style.use('seaborn')
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # 四种颜色对应四种编号方案
schemes = ['AbM', 'Imgt', 'Kabat', 'chothia']
metrics = ['MSE', 'RMSE', 'MAE', 'R2', 'PCC']
datasets = ['SAbDab', 'AB_Bind', 'Benchmark', 'SKEMPI 2.0']
width = 0.2  # 柱子的宽度

# 为每个指标单独绘制图表
for metric in metrics:
    plt.figure(figsize=(10, 6))
    # 设置每个柱子的位置
    x = np.arange(len(datasets))
    # 绘制每种编号方案的柱子
    for j, scheme in enumerate(schemes):
        values = df[df['Scheme'] == scheme].set_index('Dataset')[metric]
        plt.bar(x + j * width, values.loc[datasets], width, label=scheme, color=colors[j])

    # 添加图表元素
    plt.title(f'{metric.upper()} Comparison Across Datasets', fontsize=16, fontweight='bold')  # 增大标题字体并加粗
    plt.xlabel('Dataset', fontsize=14, fontweight='bold')  # 增大x轴标签字体并加粗
    plt.ylabel(metric.upper(), fontsize=14, fontweight='bold')  # 增大y轴标签字体并加粗
    plt.xticks(x + width * 1.5, datasets, fontsize=12, fontweight='bold')  # 增大x轴刻度字体并加粗
    plt.yticks(fontsize=12, fontweight='bold')  # 增大y轴刻度字体并加粗
    plt.grid(True, linestyle='--', alpha=0.6)
    legend = plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=14, frameon=True, edgecolor='black', facecolor='white')
    for text in legend.get_texts():
        text.set_weight('bold')  # 设置图例字体加粗

    # 调整y轴范围
    if metric == 'R2':
        plt.ylim(0, 0.6)
    elif metric == 'PCC':
        plt.ylim(0.4, 0.8)

    # 调整布局并保存
    plt.tight_layout()
    plt.savefig(f'{metric}_comparison.png', bbox_inches='tight', dpi=330)
    plt.show()