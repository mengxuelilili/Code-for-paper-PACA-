

# # 这个是真实值画图7不过为了和文章中对比实验中的指标保持一致我换成了归一化的图7
# import matplotlib.pyplot as plt
# import numpy as np
# # ===================== 数据（来自控制台截图） =====================
# datasets = ['Paddle2021', 'AB-Bind', 'SAbDab', 'SKEMPI2.0']
# # -------- MAE [Strong, Medium, Weak] --------
# paca_mae = {
#     'Paddle2021': [1.4578, 0.9404, 2.1311],
#     'AB-Bind':    [0.4913, 0.9211, 1.5628],
#     'SAbDab':     [1.9131, 1.0498, 1.0574],
#     'SKEMPI2.0':  [1.0650, 0.9129, 1.5202],
# }
# baseline_mae = {
#     'Paddle2021': [1.4976, 0.9871, 2.3092],
#     'AB-Bind':    [0.6758, 1.0244, 1.6408],
#     'SAbDab':     [1.9572, 1.1759, 1.4414],
#     'SKEMPI2.0':  [1.4917, 0.8849, 1.3874],
# }
# # -------- RMSE [Strong, Medium, Weak] --------
# paca_rmse = {
#     'Paddle2021': [1.8792, 1.2685, 2.7027],
#     'AB-Bind':    [0.7562, 1.1792, 2.0995],
#     'SAbDab':     [2.2904, 1.3117, 1.2963],
#     'SKEMPI2.0':  [1.4296, 1.1365, 2.1050],
# }
# baseline_rmse = {
#     'Paddle2021': [2.1100, 1.3053, 3.0208],
#     'AB-Bind':    [0.9300, 1.3355, 2.2391],
#     'SAbDab':     [2.4425, 1.4406, 1.9783],
#     'SKEMPI2.0':  [1.9427, 1.0365, 1.8687],
# }
# sample_sizes = {
#     'Paddle2021': [87, 169, 86],
#     'AB-Bind':    [44, 86, 44],
#     'SAbDab':     [14, 27, 14],
#     'SKEMPI2.0':  [14, 26, 14],
# }

# color_paca = '#1A4778'
# color_rpe = '#993A44'
# group_labels = ['Strong', 'Medium', 'Weak']
# width = 0.40

# fig = plt.figure(figsize=(16, 9.2))
# gs_main = fig.add_gridspec(2, 2, wspace=0.10, hspace=0.32)
# handles_global = None
# labels_global = None

# for main_idx, dataset in enumerate(datasets):
#     row = main_idx // 2
#     col = main_idx % 2
#     gs_sub = gs_main[row, col].subgridspec(1, 2, wspace=0.25)
#     ax_rmse = fig.add_subplot(gs_sub[0, 0])
#     ax_mae = fig.add_subplot(gs_sub[0, 1])

#     def draw_single_axis(ax, paca_arr, base_arr, ylab):
#         x = np.arange(len(group_labels))
#         b1 = ax.bar(x - width/2, paca_arr, width,
#                     color=color_paca, edgecolor='black', linewidth=1.05,
#                     label="PACA-Affinity")
#         b2 = ax.bar(x + width/2, base_arr, width,
#                     color=color_rpe, edgecolor='black', linewidth=1.05,
#                     label="PACA+RPE")
#         # 柱子数值：只保留两位小数，减少拥挤
#         for bar in b1:
#             h = bar.get_height()
#             ax.text(bar.get_x() + bar.get_width()/2, h + 0.12,
#                     f"{h:.2f}", ha="center", va="bottom",
#                     fontsize=7, weight="bold")
#         for bar in b2:
#             h = bar.get_height()
#             ax.text(bar.get_x() + bar.get_width()/2, h + 0.12,
#                     f"{h:.2f}", ha="center", va="bottom",
#                     fontsize=7, weight="bold")
#         # improvement箭头标注：全部居中；统一抬高y避免重叠
#         for i in range(3):
#             pv = paca_arr[i]
#             bv = base_arr[i]
#             imp = (1 - pv / bv) * 100
#             if imp > 0:
#                 c = "#008800"
#                 arrow = "↑"
#             elif imp < 0:
#                 c = "#dd0000"
#                 arrow = "↓"
#             else:
#                 c = "black"
#                 arrow = "="
#             yp = max(pv, bv) + 0.5
#             ax.text(i, yp, f"{arrow}{abs(imp):.1f}%",
#                     ha="center", va="bottom", fontsize=7.5,
#                     color=c, weight="bold")
#         # 底部n样本数，字体调小防拥挤
#         for i, n in enumerate(sample_sizes[dataset]):
#             ax.text(i, -0.15, f"n={n}", ha="center", va="top",
#                     fontsize=10, transform=ax.get_xaxis_transform())
#         ax.set_xticks(x)
#         ax.set_xticklabels(group_labels, fontsize=10)
#         ax.set_ylabel(ylab, fontsize=10)
#         ax.grid(axis="y", alpha=0.22, linestyle="--")
#         ymax = max(max(paca_arr), max(base_arr)) * 1.45
#         ax.set_ylim(0, ymax)
#         return b1, b2

#     b1_rmse, b2_rmse = draw_single_axis(ax_rmse, paca_rmse[dataset], baseline_rmse[dataset], ylab="RMSE")
#     draw_single_axis(ax_mae, paca_mae[dataset], baseline_mae[dataset], ylab="MAE")

#     # 只在第一个循环抓取图例对象
#     if main_idx == 0:
#         handles_global = [b1_rmse, b2_rmse]
#         labels_global = ["PACA-Affinity", "PACA+RPE"]

#     label_char = chr(ord('A') + main_idx)
#     ax_rmse.set_title("")
#     ax_mae.set_title("")
#     pos_rmse = ax_rmse.get_position()
#     pos_mae = ax_mae.get_position()
#     center_x = (pos_rmse.x1 + pos_mae.x0) / 2
#     top_y = pos_rmse.y1 + 0.015
#     fig.text(
#         center_x,
#         top_y,
#         f"({label_char}) {dataset}",
#         ha="center", va="bottom",
#         fontsize=12, weight="bold"
#     )

# # 添加全局右上角统一图例
# fig.legend(handles_global, labels_global,
#            loc="upper right",
#            bbox_to_anchor=(0.97, 0.97),
#            frameon=True, edgecolor="black", fontsize=9)

# out_path = "/root/autodl-tmp/AbAgCDR/fig7/fig7_rmse_mae_combined.png"
# plt.savefig(out_path, dpi=330, bbox_inches="tight")
# plt.close()
# print(f"✅ 已输出修改图：{out_path}")

import matplotlib.pyplot as plt
import numpy as np

# ===================== 数据（来自控制台截图 - 归一化空间） =====================
datasets = ['Paddle2021', 'AB-Bind', 'SAbDab', 'SKEMPI2.0']

# -------- MAE [Strong, Medium, Weak] --------
paca_mae = {
    'Paddle2021': [0.6414, 0.4138, 0.9377],
    'AB-Bind':    [0.2162, 0.4053, 0.6877],
    'SAbDab':     [0.8418, 0.4619, 0.4653],
    'SKEMPI2.0':  [0.4686, 0.4017, 0.6689],
}

baseline_mae = {
    'Paddle2021': [0.6590, 0.4343, 1.0161],
    'AB-Bind':    [0.2974, 0.4508, 0.7220],
    'SAbDab':     [0.8612, 0.5174, 0.6342],
    'SKEMPI2.0':  [0.6564, 0.3894, 0.6105],
}

# -------- RMSE [Strong, Medium, Weak] --------
paca_rmse = {
    'Paddle2021': [0.8268, 0.5582, 1.1892],
    'AB-Bind':    [0.3328, 0.5188, 0.9238],
    'SAbDab':     [1.0078, 0.5771, 0.5704],
    'SKEMPI2.0':  [0.6290, 0.5001, 0.9262],
}

baseline_rmse = {
    'Paddle2021': [0.9284, 0.5743, 1.3292],
    'AB-Bind':    [0.4092, 0.5876, 0.9852],
    'SAbDab':     [1.0747, 0.6339, 0.8705],
    'SKEMPI2.0':  [0.8548, 0.4561, 0.8222],
}

sample_sizes = {
    'Paddle2021': [87, 169, 86],
    'AB-Bind':    [44, 86, 44],
    'SAbDab':     [14, 27, 14],
    'SKEMPI2.0':  [14, 26, 14],
}

color_paca = '#1A4778'
color_rpe = '#993A44'
group_labels = ['Strong', 'Medium', 'Weak']
width = 0.40

fig = plt.figure(figsize=(16, 9.2))
gs_main = fig.add_gridspec(2, 2, wspace=0.10, hspace=0.32)
handles_global = None
labels_global = None

for main_idx, dataset in enumerate(datasets):
    row = main_idx // 2
    col = main_idx % 2
    gs_sub = gs_main[row, col].subgridspec(1, 2, wspace=0.25)
    ax_rmse = fig.add_subplot(gs_sub[0, 0])
    ax_mae = fig.add_subplot(gs_sub[0, 1])

    def draw_single_axis(ax, paca_arr, base_arr, ylab):
        x = np.arange(len(group_labels))
        b1 = ax.bar(x - width/2, paca_arr, width,
                    color=color_paca, edgecolor='black', linewidth=1.05,
                    label="PACA-Affinity")
        b2 = ax.bar(x + width/2, base_arr, width,
                    color=color_rpe, edgecolor='black', linewidth=1.05,
                    label="PACA+RPE")
        
        # 柱子数值：保留3位小数，使用相对偏移量避免在小数值时文字飞太高
        for bar in b1:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + h * 0.08,
                    f"{h:.3f}", ha="center", va="bottom",
                    fontsize=7, weight="bold")
        for bar in b2:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + h * 0.08,
                    f"{h:.3f}", ha="center", va="bottom",
                    fontsize=7, weight="bold")
        
        # improvement箭头标注
        for i in range(3):
            pv = paca_arr[i]
            bv = base_arr[i]
            imp = (1 - pv / bv) * 100
            
            if imp > 0:
                c = "#008800"
                arrow = "↑"
            elif imp < 0:
                c = "#dd0000"
                arrow = "↓"
            else:
                c = "black"
                arrow = "="
            
            # 动态计算Y轴位置：取最大值并增加一定余量
            max_h = max(pv, bv)
            yp = max_h + max(max_h * 0.2, 0.05) 
            
            ax.text(i, yp, f"{arrow}{abs(imp):.1f}%",
                    ha="center", va="bottom", fontsize=7.5,
                    color=c, weight="bold")
        
        # 底部n样本数
        for i, n in enumerate(sample_sizes[dataset]):
            ax.text(i, -0.15, f"n={n}", ha="center", va="top",
                    fontsize=10, transform=ax.get_xaxis_transform())
        
        ax.set_xticks(x)
        ax.set_xticklabels(group_labels, fontsize=10)
        ax.set_ylabel(ylab, fontsize=10)
        ax.grid(axis="y", alpha=0.22, linestyle="--")
        
        # 动态设置Y轴上限，确保所有文字都能显示
        ymax = max(max(paca_arr), max(base_arr)) * 1.6
        ax.set_ylim(0, ymax)
        return b1, b2

    b1_rmse, b2_rmse = draw_single_axis(ax_rmse, paca_rmse[dataset], baseline_rmse[dataset], ylab="RMSE")
    draw_single_axis(ax_mae, paca_mae[dataset], baseline_mae[dataset], ylab="MAE")

    # 只在第一个循环抓取图例对象
    if main_idx == 0:
        handles_global = [b1_rmse, b2_rmse]
        labels_global = ["PACA-Affinity", "PACA+RPE"]

    label_char = chr(ord('A') + main_idx)
    ax_rmse.set_title("")
    ax_mae.set_title("")
    
    # 添加子图标题 (A) Paddle2021 等
    pos_rmse = ax_rmse.get_position()
    pos_mae = ax_mae.get_position()
    center_x = (pos_rmse.x1 + pos_mae.x0) / 2
    top_y = pos_rmse.y1 + 0.015
    fig.text(
        center_x,
        top_y,
        f"({label_char}) {dataset}",
        ha="center", va="bottom",
        fontsize=12, weight="bold"
    )

# 添加全局右上角统一图例
fig.legend(handles_global, labels_global,
           loc="upper right",
           bbox_to_anchor=(0.97, 0.97),
           frameon=True, edgecolor="black", fontsize=9)

out_path = "/root/autodl-tmp/AbAgCDR/fig7/fig7_rmse_mae_combined_normalized.png"
plt.savefig(out_path, dpi=330, bbox_inches="tight")
plt.close()
print(f"✅ 已输出修改图（归一化数据）：{out_path}")