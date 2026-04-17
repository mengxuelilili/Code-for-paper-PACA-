# import pandas as pd
# import matplotlib.pyplot as plt
# import numpy as np
# import os
# from matplotlib import rcParams

# # =================配置区域=================
# # 输入：预测结果 CSV 文件路径
# INPUT_CSV = '/tmp/AbAgCDR/resultsxin/prediction_7BZ5_final.csv'

# # 输出：保存图片的路径
# OUTPUT_IMAGE = '/tmp/AbAgCDR/resultsxin/prediction_7BZ5_distribution.png'

# # 绘图参数
# DPI = 330               # 要求的高分辨率
# FIG_SIZE = (10, 6)      # 图片尺寸 (宽, 高) 英寸
# BIN_COUNT = 30          # 柱状图的柱子数量
# COLOR = '#4C72B0'       # 柱子颜色 (科学蓝)
# EDGE_COLOR = 'black'    # 柱子边缘颜色
# GRID_ALPHA = 0.3        # 网格透明度
# # =========================================

# def plot_histogram():
#     # 1. 检查文件是否存在
#     if not os.path.exists(INPUT_CSV):
#         raise FileNotFoundError(f"❌ 找不到预测结果文件: {INPUT_CSV}\n请先运行预测脚本生成该文件。")

#     print(f"📂 正在加载数据: {INPUT_CSV}")
#     df = pd.read_csv(INPUT_CSV)

#     # 确认列名
#     if 'predicted_delta_g_kcal_mol' not in df.columns:
#         raise ValueError("❌ CSV 文件中缺少 'predicted_delta_g_kcal_mol' 列。请检查文件内容。")

#     data = df['predicted_delta_g_kcal_mol'].dropna()
    
#     if len(data) == 0:
#         raise ValueError("❌ 没有有效的数据可绘制。")

#     print(f"📊 有效数据样本数: {len(data)}")
#     print(f"📈 数据范围: [{data.min():.2f}, {data.max():.2f}] kcal/mol")

#     # 2. 设置绘图风格
#     # 使用 seaborn 风格或者默认风格微调
#     rcParams['font.family'] = 'sans-serif'
#     rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'SimHei'] # 支持中文以防万一
#     rcParams['axes.linewidth'] = 1.2
    
#     fig, ax = plt.subplots(figsize=FIG_SIZE, dpi=DPI)

#     # 3. 绘制直方图
#     n, bins, patches = ax.hist(
#         data, 
#         bins=BIN_COUNT, 
#         color=COLOR, 
#         edgecolor=EDGE_COLOR, 
#         linewidth=1.2, 
#         alpha=0.85,
#         label='Predicted $\Delta G$ Distribution'
#     )

#     # 4. 美化图表
#     ax.set_title('Distribution of Predicted Binding Affinity ($\Delta G$)\n(7BZ5 Unlabeled Dataset)', 
#                  fontsize=14, fontweight='bold', pad=15)
#     ax.set_xlabel('Predicted $\Delta G$ (kcal/mol)', fontsize=12, fontweight='bold')
#     ax.set_ylabel('Frequency (Count)', fontsize=12, fontweight='bold')
    
#     # 设置网格
#     ax.grid(axis='y', linestyle='--', alpha=GRID_ALPHA, linewidth=1.0)
#     ax.set_axisbelow(True) # 网格在柱子后面

#     # 5. 添加统计信息文本框
#     stats_text = (
#         f"Count: {len(data)}\n"
#         f"Mean: {data.mean():.2f}\n"
#         f"Std Dev: {data.std():.2f}\n"
#         f"Min: {data.min():.2f}\n"
#         f"Max: {data.max():.2f}"
#     )
    
#     # 将文本框放在右上角或空白处
#     props = dict(boxstyle='round', facecolor='wheat', alpha=0.5, edgecolor='gray')
#     ax.text(0.98, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
#             verticalalignment='top', horizontalalignment='right', bbox=props)

#     # 6. 调整布局并保存
#     plt.tight_layout()
    
#     # 确保输出目录存在
#     os.makedirs(os.path.dirname(OUTPUT_IMAGE), exist_ok=True)
    
#     print(f"💾 正在保存高清图片 ({DPI} DPI) 到: {OUTPUT_IMAGE}")
#     plt.savefig(
#         OUTPUT_IMAGE, 
#         dpi=DPI, 
#         bbox_inches='tight', 
#         format='png',
#         facecolor='white',
#         edgecolor='none'
#     )
    
#     plt.close(fig) # 释放内存
#     print("✅ 图片保存成功！")

# if __name__ == "__main__":
#     try:
#         plot_histogram()
#     except Exception as e:
#         print(f"❌ 发生错误: {e}")

# import pandas as pd
# import matplotlib.pyplot as plt
# import numpy as np
# import os
# from matplotlib import rcParams

# # =================配置区域=================
# # 输入：预测结果 CSV 文件路径
# INPUT_CSV = '/tmp/AbAgCDR/resultsxin/prediction_7BZ5_final.csv'

# # 输出：保存图片的路径
# OUTPUT_IMAGE = '/tmp/AbAgCDR/resultsxin/prediction_7BZ5_barplot.png'

# # 绘图参数
# DPI = 330               # 高分辨率
# FIG_SIZE = (12, 6)      # 图片更宽一些，以便容纳 30 个柱子
# BAR_COLOR = '#4C72B0'   # 默认柱子颜色 (科学蓝)
# HIGHLIGHT_COLOR = '#D62728' # 高亮柱子颜色 (红色)
# EDGE_COLOR = 'black'    # 柱子边缘颜色
# GRID_ALPHA = 0.3        # 网格透明度

# # 【关键配置】是否需要高亮某个特定的样本？
# # 如果不需要高亮，请设置为 None
# # 如果需要高亮第 5 个样本 (Index=5)，请设置为 5
# # 注意：这里的数字对应 CSV 中的 'sample_id' 或者行号 (从 1 开始计数)
# HIGHLIGHT_INDEX = 25     # <--- 修改这里，设为 None 则不高亮

# # =========================================

# def plot_bar_chart():
#     # 1. 检查文件是否存在
#     if not os.path.exists(INPUT_CSV):
#         raise FileNotFoundError(f"❌ 找不到预测结果文件: {INPUT_CSV}")

#     print(f"📂 正在加载数据: {INPUT_CSV}")
    
#     try:
#         df = pd.read_csv(INPUT_CSV)
#     except Exception as e:
#         raise ValueError(f"❌ 无法读取 CSV 文件: {e}")

#     target_col = 'predicted_delta_g_kcal_mol'
#     if target_col not in df.columns:
#         raise ValueError(f"❌ CSV 文件中缺少 '{target_col}' 列。")

#     # 提取数据
#     # 确保按顺序排列，如果有 sample_id 列则按 ID 排序，否则按行顺序
#     if 'sample_id' in df.columns:
#         df = df.sort_values('sample_id')
    
#     raw_data = df[target_col]
#     data = pd.to_numeric(raw_data, errors='coerce').dropna()
    
#     if len(data) == 0:
#         raise ValueError("❌ 没有有效的数值数据。")

#     # 生成 X 轴索引 (从 1 开始)
#     indices = np.arange(1, len(data) + 1)
#     values = data.values

#     print(f"✅ 成功加载 {len(data)} 个样本。")
#     print(f"📈 数据范围: [{values.min():.2f}, {values.max():.2f}] kcal/mol")
    
#     if HIGHLIGHT_INDEX is not None:
#         print(f"🔴 将高亮显示样本 Index: {HIGHLIGHT_INDEX}")

#     # 2. 设置绘图风格
#     rcParams['font.family'] = 'sans-serif'
#     rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans'] 
#     rcParams['axes.linewidth'] = 1.2
    
#     fig, ax = plt.subplots(figsize=FIG_SIZE, dpi=DPI)

#     # 3. 绘制柱状图
#     # 创建颜色列表：默认全是蓝色，如果匹配高亮索引则变为红色
#     colors = [BAR_COLOR] * len(data)
#     if HIGHLIGHT_INDEX is not None:
#         # 注意：indices 是从 1 开始的，列表索引是从 0 开始的
#         if 1 <= HIGHLIGHT_INDEX <= len(data):
#             colors[HIGHLIGHT_INDEX - 1] = HIGHLIGHT_COLOR
#         else:
#             print(f"⚠️ 警告：高亮索引 {HIGHLIGHT_INDEX} 超出数据范围 (1-{len(data)})，已忽略。")

#     bars = ax.bar(
#         indices, 
#         values, 
#         color=colors, 
#         edgecolor=EDGE_COLOR, 
#         linewidth=0.8, 
#         width=0.8,       # 柱子宽度 (0-1 之间，1 表示无间隙)
#         label='Predicted $\Delta G$'
#     )

#     # 4. 美化图表
#     ax.set_title(f'Predicted dG for All Samples\n(Red = Index {HIGHLIGHT_INDEX})' if HIGHLIGHT_INDEX else 'Predicted dG for All Samples', 
#                  fontsize=14, fontweight='bold', pad=15)
    
#     ax.set_xlabel('Sample Index (1-based)', fontsize=14, fontweight='bold')
#     ax.set_ylabel('Predicted dG (kcal/mol)', fontsize=14, fontweight='bold')
    
#     # 设置 X 轴刻度，避免太密
#     step = max(1, len(data) // 10)  # 大约显示 10 个刻度
#     ax.set_xticks(np.arange(1, len(data) + 1, step=step))
    
#     # 设置 Y 轴范围，稍微留点空隙
#     y_min, y_max = values.min(), values.max()
#     y_range = y_max - y_min
#     ax.set_ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)

#     # 设置网格 (只在 Y 轴)
#     ax.grid(axis='y', linestyle='--', alpha=GRID_ALPHA, linewidth=1.0)
#     ax.set_axisbelow(True)

#     # 5. 添加统计信息文本框 (可选)
#     stats_text = (
#         f"Count: {len(data)}\n"
#         f"Mean: {np.mean(values):.2f}\n"
#         f"Min: {np.min(values):.2f}\n"
#         f"Max: {np.max(values):.2f}"
#     )
#     props = dict(boxstyle='round', facecolor='wheat', alpha=0.6, edgecolor='gray')
#     ax.text(0.98, 0.95, stats_text, transform=ax.transAxes, fontsize=12,
#             verticalalignment='top', horizontalalignment='right', bbox=props)

#     # 6. 保存
#     plt.tight_layout()
#     os.makedirs(os.path.dirname(OUTPUT_IMAGE), exist_ok=True)
    
#     print(f"💾 正在保存图片到: {OUTPUT_IMAGE}")
#     plt.savefig(
#         OUTPUT_IMAGE, 
#         dpi=DPI, 
#         bbox_inches='tight', 
#         format='png',
#         facecolor='white'
#     )
    
#     plt.close(fig)
#     print("✅ 图片保存成功！")

# if __name__ == "__main__":
#     try:
#         plot_bar_chart()
#     except Exception as e:
#         print(f"❌ 发生错误: {e}")
#         import traceback
#         traceback.print_exc()


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib import rcParams

# =================配置区域=================
# 输入：预测结果 CSV 文件路径
INPUT_CSV = '/tmp/AbAgCDR/resultsxin/prediction_1BQL_final.csv'

# 输出：保存图片的路径
OUTPUT_IMAGE = '/tmp/AbAgCDR/resultsxin/prediction_1BQL_barplot.png'

# 绘图参数
DPI = 330               # 高分辨率
FIG_SIZE = (12, 6)      # 图片更宽一些，以便容纳 30 个柱子
BAR_COLOR = '#4C72B0'   # 默认柱子颜色 (科学蓝)
HIGHLIGHT_COLOR = '#D62728' # 高亮柱子颜色 (红色)
EDGE_COLOR = 'black'    # 柱子边缘颜色
GRID_ALPHA = 0.3        # 网格透明度

# 【关键配置】是否需要高亮某个特定的样本？
HIGHLIGHT_INDEX = 5     # <--- 修改这里，设为 None 则不高亮

# =========================================

def plot_bar_chart():
    # 1. 检查文件是否存在
    if not os.path.exists(INPUT_CSV):
        raise FileNotFoundError(f"❌ 找不到预测结果文件: {INPUT_CSV}")

    print(f"📂 正在加载数据: {INPUT_CSV}")
    
    try:
        df = pd.read_csv(INPUT_CSV)
    except Exception as e:
        raise ValueError(f"❌ 无法读取 CSV 文件: {e}")

    target_col = 'predicted_delta_g_kcal_mol'
    if target_col not in df.columns:
        raise ValueError(f"❌ CSV 文件中缺少 '{target_col}' 列。")

    # 提取数据
    if 'sample_id' in df.columns:
        df = df.sort_values('sample_id')
    
    raw_data = df[target_col]
    data = pd.to_numeric(raw_data, errors='coerce').dropna()
    
    if len(data) == 0:
        raise ValueError("❌ 没有有效的数值数据。")

    # 生成 X 轴索引 (从 1 开始)
    indices = np.arange(1, len(data) + 1)
    values = data.values

    print(f"✅ 成功加载 {len(data)} 个样本。")
    print(f"📈 数据范围: [{values.min():.2f}, {values.max():.2f}] kcal/mol")
    
    if HIGHLIGHT_INDEX is not None:
        print(f"🔴 将高亮显示样本 Index: {HIGHLIGHT_INDEX}")

    # 2. 设置绘图风格
    rcParams['font.family'] = 'sans-serif'
    rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans'] 
    rcParams['axes.linewidth'] = 1.2
    
    fig, ax = plt.subplots(figsize=FIG_SIZE, dpi=DPI)

    # 3. 绘制柱状图
    colors = [BAR_COLOR] * len(data)
    if HIGHLIGHT_INDEX is not None:
        if 1 <= HIGHLIGHT_INDEX <= len(data):
            colors[HIGHLIGHT_INDEX - 1] = HIGHLIGHT_COLOR
        else:
            print(f"⚠️ 警告：高亮索引 {HIGHLIGHT_INDEX} 超出数据范围 (1-{len(data)})，已忽略。")

    bars = ax.bar(
        indices, 
        values, 
        color=colors, 
        edgecolor=EDGE_COLOR, 
        linewidth=0.8, 
        width=0.8,
        label='Predicted $\Delta G$'
    )

    # 4. 美化图表
    ax.set_title(f'Predicted dG for All Samples\n(Red = Index {HIGHLIGHT_INDEX})' if HIGHLIGHT_INDEX else 'Predicted dG for All Samples', 
                 fontsize=14, fontweight='bold', pad=15)
    
    ax.set_xlabel('Sample Index (1-based)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Predicted dG (kcal/mol)', fontsize=14, fontweight='bold')
    
    step = max(1, len(data) // 10)
    ax.set_xticks(np.arange(1, len(data) + 1, step=step))
    
    y_min, y_max = values.min(), values.max()
    y_range = y_max - y_min
    ax.set_ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)

    ax.grid(axis='y', linestyle='--', alpha=GRID_ALPHA, linewidth=1.0)
    ax.set_axisbelow(True)

    # 5. 调整文本框位置，让它离图更近
    stats_text = (
        f"Count: {len(data)}\n"
        f"Mean: {np.mean(values):.2f}\n"
        f"Min: {np.min(values):.2f}\n"
        f"Max: {np.max(values):.2f}"
    )
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.6, edgecolor='gray')
    
    # 位置微调：x 从 0.92 → 0.83，离图更近
    fig.text(0.90, 0.88, stats_text, fontsize=12,
             verticalalignment='top', horizontalalignment='left', bbox=props)

    # 6. 布局调整：让绘图区域更靠右，减少空白
    plt.tight_layout()
    # 从 0.85 → 0.90，绘图区域更靠右，文本框离图更近
    plt.subplots_adjust(right=0.90)
    
    os.makedirs(os.path.dirname(OUTPUT_IMAGE), exist_ok=True)
    
    print(f"💾 正在保存图片到: {OUTPUT_IMAGE}")
    plt.savefig(
        OUTPUT_IMAGE, 
        dpi=DPI, 
        bbox_inches='tight', 
        format='png',
        facecolor='white'
    )
    
    plt.close(fig)
    print("✅ 图片保存成功！")

if __name__ == "__main__":
    try:
        plot_bar_chart()
    except Exception as e:
        print(f"❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()