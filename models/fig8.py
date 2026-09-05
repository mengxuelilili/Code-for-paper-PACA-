# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
# from scipy.stats import pearsonr
# import os

# # ======================================================
# # 设置matplotlib字体（论文格式，全部字体≥10）
# # ======================================================
# plt.rcParams['font.family'] = 'Arial'
# plt.rcParams['font.size'] = 10
# plt.rcParams['axes.labelsize'] = 11
# plt.rcParams['axes.titlesize'] = 13
# plt.rcParams['xtick.labelsize'] = 10
# plt.rcParams['ytick.labelsize'] = 10
# plt.rcParams['legend.fontsize'] = 10

# # 固定两套颜色：全部数据集共用
# color_true = '#2E86AB'    # A图蓝色：散点 + True小提琴
# color_pred = '#F7C948'    # A图黄色：Predicted小提琴

# # 数据集简写 -> 完整英文标题
# dataset_name_map = {
#     "paddle":   "A Paddle2021: Regression and Violin Plot",
#     "sabdab":   "B SAbDab: Regression and Violin Plot",
#     "abbind":   "C AB-Bind: Regression and Violin Plot",
#     "skempi":   "D SKEMPI2.0: Regression and Violin Plot"
# }
# # 读取文件用简写
# datasets = ['paddle', 'sabdab', 'abbind', 'skempi']

# # ============ 【核心】4个标题独立绝对画布Y坐标，0=画布底部，1=画布顶部，各行互不干扰 ============
# # 你可以单独修改其中任意数字，只改动对应那一行标题，别的行完全不受影响
# title_abs_y = [
#     0.95,   # (A) Paddle2021 标题y
#     0.705,   # (B) SAbDab 标题y
#     0.46,   # (C) AB‑Bind 标题y
#     0.215    # (D) SKEMPI 2.0 标题y
# ]

# # ======================================================
# # 创建4行 × 2列的子图
# # ======================================================
# fig, axes = plt.subplots(4, 2, figsize=(14, 16))

# for idx, dataset in enumerate(datasets):
#     file_path = f"/root/autodl-tmp/AbAgCDR/resultsxin2/{dataset.lower()}_predictions_seed_42.csv"

#     try:
#         df = pd.read_csv(file_path)
#     except FileNotFoundError:
#         print(f"⚠️ 警告: 未找到文件 {file_path}，跳过该数据集。")
#         continue

#     y_true = df['true_ddg'].values
#     y_pred = df['pred_ddg'].values

#     # ========== 回归散点图 (左列) ==========
#     ax_scatter = axes[idx, 0]
#     ax_scatter.scatter(y_true, y_pred, alpha=0.6, s=30,
#                        color=color_true, edgecolors='black', linewidth=0.5)

#     min_val = min(y_true.min(), y_pred.min()) - 0.5
#     max_val = max(y_true.max(), y_pred.max()) + 0.5
#     ax_scatter.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='y = x')

#     r2 = np.corrcoef(y_true, y_pred)[0, 1]**2
#     pcc, _ = pearsonr(y_true, y_pred)

#     ax_scatter.set_xlabel('True ΔG(kcal/mol)', fontsize=14)
#     ax_scatter.set_ylabel('Predicted ΔG(kcal/mol)', fontsize=14)

#     # R² PCC文本框
#     ax_scatter.text(0.03, 0.86, f'R² = {r2:.4f}\nPCC = {pcc:.4f}',
#                     transform=ax_scatter.transAxes, fontsize=11,
#                     verticalalignment='top',
#                     bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

#     ax_scatter.legend(loc='lower right', fontsize=11)
#     ax_scatter.grid(True, alpha=0.3)
#     ax_scatter.set_xlim(min_val, max_val)
#     ax_scatter.set_ylim(min_val, max_val)

#     # ========== 小提琴图 (右列) ==========
#     ax_violin = axes[idx, 1]
#     data_to_plot = [y_true, y_pred]

#     parts = ax_violin.violinplot(data_to_plot, positions=[1, 2], showmeans=False, showmedians=True)

#     if 'bodies' in parts:
#         parts['bodies'][0].set_facecolor(color_true)
#         parts['bodies'][0].set_alpha(0.6)
#         parts['bodies'][1].set_facecolor(color_pred)
#         parts['bodies'][1].set_alpha(0.6)

#     keys_to_fix = ['cbars', 'cmins', 'cmaxes', 'cmedians']
#     for key in keys_to_fix:
#         if key in parts:
#             collection = parts[key]
#             collection.set_color('black')
#             if key == 'cmedians':
#                 collection.set_linewidth(1.5)

#     ax_violin.set_xticks([1, 2])
#     ax_violin.set_xticklabels(['True', 'Predicted'], fontsize=14)
#     ax_violin.set_ylabel('ΔG(kcal/mol)', fontsize=14)
#     ax_violin.grid(True, alpha=0.3, axis='y')

# # ---------------------- 行标题：使用独立绝对Y坐标，每行独立可调 ----------------------
# for idx, dataset in enumerate(datasets):
#     ax_left = axes[idx,0]
#     ax_right = axes[idx,1]
#     posL = ax_left.get_position()
#     posR = ax_right.get_position()
#     row_mid_x = (posL.x0 + posR.x1) / 2

#     # 直接读取数组里该行专属y值，修改数组中的数字只影响这一行
#     fig.text(row_mid_x, title_abs_y[idx], dataset_name_map[dataset],
#              ha="center", va="bottom", fontsize=16, weight='bold')

# # ===================== 布局：增大hspace，子图行之间拉开距离 =====================
# plt.subplots_adjust(left=0.07, right=0.97, top=0.94, bottom=0.04,
#                     hspace=0.50,   # 增大 hspace，子图行之间空隙放大
#                     wspace=0.15)

# # ======================================================
# # 保存图片
# # ======================================================
# output_dir = '/root/autodl-tmp/AbAgCDR/fig8/'
# if not os.path.exists(output_dir):
#     os.makedirs(output_dir)
# output_path = os.path.join(output_dir, 'fig8_improved.png')

# plt.savefig(output_path, dpi=330)
# print(f"✅ 图片已保存: {output_path}")
# plt.close()


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import os

# ======================================================
# 设置matplotlib字体（论文格式）
# ======================================================
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10

# 固定两套颜色
color_true = '#2E86AB'    
color_pred = '#F7C948'    

# 分别设置左右两个图的标题
dataset_left_title_map = {
    "paddle":   "Paddle2021: Regression plot",
    "sabdab":   "SAbDab: Regression plot",
    "abbind":   "AB-Bind: Regression plot",
    "skempi":   "SKEMPI2.0: Regression plot"
}

dataset_right_title_map = {
    "paddle":   "Paddle2021: Violin Plot",
    "sabdab":   "SAbDab: Violin Plot",
    "abbind":   "AB-Bind: Violin Plot",
    "skempi":   "SKEMPI2.0: Violin Plot"
}

# 面板标签 (放在大图外侧)
panel_labels = ['A', 'B', 'C', 'D']

# 读取文件
datasets = ['paddle', 'sabdab', 'abbind', 'skempi']

# ======================================================
# 创建4行 × 2列的子图
# ======================================================
fig, axes = plt.subplots(4, 2, figsize=(14, 16))

for idx, dataset in enumerate(datasets):
    file_path = f"/root/autodl-tmp/AbAgCDR/resultsxin2/{dataset.lower()}_predictions_seed_42.csv"

    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"⚠️ 警告: 未找到文件 {file_path}，跳过该数据集。")
        continue

    y_true = df['true_ddg'].values
    y_pred = df['pred_ddg'].values

    # ========== 回归散点图 (左列) ==========
    ax_scatter = axes[idx, 0]
    ax_scatter.scatter(y_true, y_pred, alpha=0.6, s=30,
                       color=color_true, edgecolors='black', linewidth=0.5)

    min_val = min(y_true.min(), y_pred.min()) - 0.5
    max_val = max(y_true.max(), y_pred.max()) + 0.5
    ax_scatter.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='y = x')

    r2 = np.corrcoef(y_true, y_pred)[0, 1]**2
    pcc, _ = pearsonr(y_true, y_pred)

    ax_scatter.set_xlabel('True ΔG(kcal/mol)', fontsize=14)
    ax_scatter.set_ylabel('Predicted ΔG(kcal/mol)', fontsize=14)

    # R² PCC文本框
    ax_scatter.text(0.03, 0.86, f'R² = {r2:.4f}\nPCC = {pcc:.4f}',
                    transform=ax_scatter.transAxes, fontsize=11,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    ax_scatter.legend(loc='lower right', fontsize=11)
    ax_scatter.grid(True, alpha=0.3)
    ax_scatter.set_xlim(min_val, max_val)
    ax_scatter.set_ylim(min_val, max_val)

    # 加上左图的标题
    ax_scatter.set_title(dataset_left_title_map[dataset], fontsize=15, fontweight='normal', pad=15)

    # ========== 小提琴图 (右列) ==========
    ax_violin = axes[idx, 1]
    data_to_plot = [y_true, y_pred]

    parts = ax_violin.violinplot(data_to_plot, positions=[1, 2], showmeans=False, showmedians=True)

    if 'bodies' in parts:
        parts['bodies'][0].set_facecolor(color_true)
        parts['bodies'][0].set_alpha(0.6)
        parts['bodies'][1].set_facecolor(color_pred)
        parts['bodies'][1].set_alpha(0.6)

    keys_to_fix = ['cbars', 'cmins', 'cmaxes', 'cmedians']
    for key in keys_to_fix:
        if key in parts:
            collection = parts[key]
            collection.set_color('black')
            if key == 'cmedians':
                collection.set_linewidth(1.5)

    ax_violin.set_xticks([1, 2])
    ax_violin.set_xticklabels(['True', 'Predicted'], fontsize=14)
    ax_violin.set_ylabel('ΔG(kcal/mol)', fontsize=14)
    ax_violin.grid(True, alpha=0.3, axis='y')

    # 加上右图的标题
    ax_violin.set_title(dataset_right_title_map[dataset], fontsize=15, fontweight='normal', pad=15)

    # ========== 添加外部面板标签 A/B/C/D ==========
    # 放在整个大行（左图）的最左上角外侧，和两个子图的标题保持同一水平线
    ax_scatter.text(-0.12, 1.06, panel_labels[idx], 
                    transform=ax_scatter.transAxes, 
                    fontsize=16, fontweight='bold', va='bottom', ha='right')

# ===================== 布局 =====================
# 这里的 hspace 稍微调小一点点(0.4)，因为现在左图有标题，右图也有标题，需要留有足够的空间
plt.subplots_adjust(left=0.07, right=0.97, top=0.94, bottom=0.04,
                    hspace=0.45,   
                    wspace=0.15)

# ======================================================
# 保存图片
# ======================================================
output_dir = '/root/autodl-tmp/AbAgCDR/fig8/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
output_path = os.path.join(output_dir, 'fig8_improved.png')

plt.savefig(output_path, dpi=330, bbox_inches='tight')
print(f"✅ 图片已保存: {output_path}")
plt.close()