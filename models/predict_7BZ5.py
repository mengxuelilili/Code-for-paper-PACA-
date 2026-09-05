# import torch
# import torch.nn as nn
# import pandas as pd
# import numpy as np
# import os
# import math
# from torch.utils.data import Dataset, DataLoader
# import matplotlib.pyplot as plt
# from matplotlib import rcParams
# from roformercnn import CombinedModel
# from xiaorong import getCDRPos


# # ======================================================
# # 1. 配置区域
# # ======================================================
# MODEL_PATH = '/tmp/AbAgCDR/model/best_model_seed_42.pth'
# DATA_PATH = '/tmp/AbAgCDR/data/6MQR_data.pt'
# OUTPUT_CSV = '/root/autodl-tmp/AbAgCDR/fig11/prediction_6MQR_final.csv'
# OUTPUT_IMAGE = '/root/autodl-tmp/AbAgCDR/fig11/prediction_6MQR_distribution.png'

# DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# BATCH_SIZE = 32
# PLOT_DPI = 330

# # ======================================================
# # 2. 数据处理函数
# # ======================================================
# def collate_fn(batch):
#     X_a_list = [torch.tensor(item[0], dtype=torch.float32) if not isinstance(item[0], torch.Tensor) else item[0] for item in batch]
#     X_b_list = [torch.tensor(item[1], dtype=torch.float32) if not isinstance(item[1], torch.Tensor) else item[1] for item in batch]
#     ag_list = [torch.tensor(item[2], dtype=torch.float32) if not isinstance(item[2], torch.Tensor) else item[2] for item in batch]
    
#     has_y = len(batch[0]) > 3
#     if has_y:
#         y_list = [torch.tensor(item[3], dtype=torch.float32) if not isinstance(item[3], torch.Tensor) else item[3] for item in batch]
#         y_tensor = torch.stack(y_list)
#     else:
#         y_tensor = torch.zeros(len(batch))

#     if not X_a_list:
#         return torch.empty(0), torch.empty(0), torch.empty(0), torch.empty(0)

#     max_len = max(
#         max(x.shape[0] for x in X_a_list),
#         max(x.shape[0] for x in X_b_list),
#         max(x.shape[0] for x in ag_list)
#     )

#     def pad_to_len(x, L):
#         if x.shape[0] < L:
#             pad = torch.zeros(L - x.shape[0], x.shape[1], dtype=x.dtype)
#             return torch.cat([x, pad], dim=0)
#         return x[:L]

#     X_a_padded = torch.stack([pad_to_len(x, max_len) for x in X_a_list])
#     X_b_padded = torch.stack([pad_to_len(x, max_len) for x in X_b_list])
#     ag_padded = torch.stack([pad_to_len(x, max_len) for x in ag_list])

#     return X_a_padded, X_b_padded, ag_padded, y_tensor

# class UnlabeledDataset(Dataset):
#     def __init__(self, data_dict):
#         self.X_a = data_dict['X_a']
#         self.X_b = data_dict['X_b']
#         self.X_g = data_dict['antigen']
#         self.len = self.X_a.shape[0]

#     def __len__(self):
#         return self.len

#     def __getitem__(self, idx):
#         return (self.X_a[idx], self.X_b[idx], self.X_g[idx], 0.0)

# # ======================================================
# # 3. 绘图函数
# # ======================================================
# def save_distribution_plot(data_values, save_path, dpi=330):
#     print(f"\n🎨 正在绘制分布图 (DPI={dpi})...")
#     rcParams['font.family'] = 'sans-serif'
#     rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'SimHei'] 
#     rcParams['axes.linewidth'] = 1.2
    
#     fig, ax = plt.subplots(figsize=(10, 6), dpi=dpi)
#     n, bins, patches = ax.hist(data_values, bins=30, color='#4C72B0', edgecolor='black', linewidth=1.2, alpha=0.85)
    
#     ax.set_title('Distribution of Predicted Binding Affinity ($\Delta G$)\n(7BZ5 Unlabeled Dataset)', fontsize=14, fontweight='bold', pad=15)
#     ax.set_xlabel('Predicted $\Delta G$ (kcal/mol)', fontsize=12, fontweight='bold')
#     ax.set_ylabel('Frequency (Count)', fontsize=12, fontweight='bold')
    
#     ax.grid(axis='y', linestyle='--', alpha=0.3, linewidth=1.0)
#     ax.set_axisbelow(True)
    
#     stats_text = (f"Count: {len(data_values):,}\nMean: {np.mean(data_values):.2f}\nStd Dev: {np.std(data_values):.2f}\nMin: {np.min(data_values):.2f}\nMax: {np.max(data_values):.2f}")
#     props = dict(boxstyle='round', facecolor='wheat', alpha=0.6, edgecolor='gray')
#     ax.text(0.98, 0.95, stats_text, transform=ax.transAxes, fontsize=10, verticalalignment='top', horizontalalignment='right', bbox=props)
    
#     plt.tight_layout()
#     os.makedirs(os.path.dirname(save_path), exist_ok=True)
#     plt.savefig(save_path, dpi=dpi, bbox_inches='tight', format='png', facecolor='white')
#     plt.close(fig)
#     print(f"✅ 图片已保存至: {save_path}")

# # ======================================================
# # 4. 主预测流程
# # ======================================================
# def main():
#     print(f"🚀 开始预测任务... 设备: {DEVICE}")

#     if not os.path.exists(MODEL_PATH):
#         raise FileNotFoundError(f"❌ 模型文件不存在: {MODEL_PATH}")
#     if not os.path.exists(DATA_PATH):
#         raise FileNotFoundError(f"❌ 数据文件不存在: {DATA_PATH}")

#     print(f"📂 加载无标签数据: {DATA_PATH}")
#     data_dict = torch.load(DATA_PATH, map_location='cpu')
    
#     required_keys = ['X_a', 'X_b', 'antigen']
#     for key in required_keys:
#         if key not in data_dict:
#             raise ValueError(f"❌ 数据文件中缺少键: {key}")
            
#     dataset = UnlabeledDataset(data_dict)
#     loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
#     print(f"📊 数据样本数: {len(dataset)}")

#     # --- 初始化模型 ---
#     print("⚙️ 初始化模型架构 (已严格匹配权重)...")
    
#     model = CombinedModel(
#         [getCDRPos("H1"), getCDRPos("H2"), getCDRPos("H3")],
#         [getCDRPos("L1"), getCDRPos("L2"), getCDRPos("L3")],
#         num_heads=2,
#         embed_dim=532,
#         antigen_embed_dim=500
#     )
#     model = model.to(DEVICE)

#     # --- 加载权重 ---
#     print(f"⚖️ 加载模型权重: {MODEL_PATH}")
#     checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    
#     state_dict = checkpoint.get('model_state_dict', checkpoint)
    
#     try:
#         # strict=True 确保所有权重都精确匹配
#         model.load_state_dict(state_dict, strict=True)
#         print("✅ 模型权重加载成功！(架构完全匹配)")
#     except RuntimeError as e:
#         print(f"❌ 加载失败：{e}")
#         print("💡 尝试非严格加载以查看具体缺失/多余的层...")
#         try:
#             model.load_state_dict(state_dict, strict=False)
#             print("⚠️ 非严格加载成功，部分权重可能未加载。")
#         except Exception as e2:
#             print(f"❌ 彻底失败: {e2}")
#             return

#     model.eval()

#     label_scaler = checkpoint.get('label_scaler', None)
#     if label_scaler is None:
#         print("⚠️ 警告: 未找到 label_scaler，输出为标准化值。")
#     else:
#         print("✅ 找到 label_scaler，将执行反归一化。")

#     # --- 推理 ---
#     print("🔮 正在预测...")
#     all_preds_scaled = []
    
#     with torch.no_grad():
#         for X_a, X_b, X_g, _ in loader:
#             if X_a.shape[0] == 0: continue
            
#             X_a = X_a.to(DEVICE)
#             X_b = X_b.to(DEVICE)
#             X_g = X_g.to(DEVICE)
            
#             # 关键顺序：Heavy (X_b), Light (X_a), Antigen (X_g)
#             try:
#                 pred = model(X_b, X_a, X_g).view(-1)
#                 all_preds_scaled.extend(pred.cpu().numpy())
#             except Exception as e:
#                 print(f"❌ 前向传播出错: {e}")
#                 import traceback
#                 traceback.print_exc()
#                 return

#     all_preds_scaled = np.array(all_preds_scaled).reshape(-1, 1)

#     if label_scaler is not None:
#         all_preds_real = label_scaler.inverse_transform(all_preds_scaled)
#     else:
#         all_preds_real = all_preds_scaled

#     preds_flat = all_preds_real.flatten()

#     # --- 保存结果 ---
#     os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
#     results_df = pd.DataFrame({
#         'sample_id': range(len(preds_flat)),
#         'predicted_delta_g_kcal_mol': preds_flat,
#         'predicted_delta_g_scaled': all_preds_scaled.flatten()
#     })
#     results_df.to_csv(OUTPUT_CSV, index=False)
#     print(f"\n💾 CSV 已保存至: {OUTPUT_CSV}")

#     # --- 绘图 ---
#     save_distribution_plot(preds_flat, OUTPUT_IMAGE, dpi=PLOT_DPI)

#     print(f"\n📈 统计摘要:")
#     print(f"   Min: {preds_flat.min():.4f}, Max: {preds_flat.max():.4f}, Mean: {preds_flat.mean():.4f}")
#     print("\n✅ 所有任务完成！")

# if __name__ == "__main__":
#     main()

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# 导入你本地的模型和工具
from roformercnn import CombinedModel
from xiaorong import getCDRPos

# ================= 1. 配置区域 =================
# 修改：添加三个PDB数据集的路径
DATA_PATHS = {
    '1BQL': '/tmp/AbAgCDR/data/1BQL_data.pt', 
    '6MQR': '/tmp/AbAgCDR/data/6MQR_data.pt',
    '7BZ5': '/tmp/AbAgCDR/data/7BZ5_data.pt'
}
MODEL_PATH = '/tmp/AbAgCDR/model/best_model_seed_42.pth'
OUTPUT_IMAGE = '/root/autodl-tmp/AbAgCDR/fig11/Fig11_Case_Study_Ranking.png'

# 核心修改：指定每个PDB中Native（红色柱子）在序列中的具体位置（基于1的索引，运行时会减去1转为Python的0基索引）
NATIVE_INDICES = {
    '1BQL': 5,  
    '6MQR': 15, 
    '7BZ5': 25  
}

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 32

# ================= 2. 数据处理函数 (保持不变) =================
def collate_fn(batch):
    X_a_list = [torch.tensor(item[0], dtype=torch.float32) if not isinstance(item[0], torch.Tensor) else item[0] for item in batch]
    X_b_list = [torch.tensor(item[1], dtype=torch.float32) if not isinstance(item[1], torch.Tensor) else item[1] for item in batch]
    ag_list = [torch.tensor(item[2], dtype=torch.float32) if not isinstance(item[2], torch.Tensor) else item[2] for item in batch]
    
    has_y = len(batch[0]) > 3
    if has_y:
        y_list = [torch.tensor(item[3], dtype=torch.float32) if not isinstance(item[3], torch.Tensor) else item[3] for item in batch]
        y_tensor = torch.stack(y_list)
    else:
        y_tensor = torch.zeros(len(batch))

    if not X_a_list:
        return torch.empty(0), torch.empty(0), torch.empty(0), torch.empty(0)

    max_len = max(
        max(x.shape[0] for x in X_a_list),
        max(x.shape[0] for x in X_b_list),
        max(x.shape[0] for x in ag_list)
    )

    def pad_to_len(x, L):
        if x.shape[0] < L:
            pad = torch.zeros(L - x.shape[0], x.shape[1], dtype=x.dtype)
            return torch.cat([x, pad], dim=0)
        return x[:L]

    X_a_padded = torch.stack([pad_to_len(x, max_len) for x in X_a_list])
    X_b_padded = torch.stack([pad_to_len(x, max_len) for x in X_b_list])
    ag_padded = torch.stack([pad_to_len(x, max_len) for x in ag_list])

    return X_a_padded, X_b_padded, ag_padded, y_tensor

class UnlabeledDataset(Dataset):
    def __init__(self, data_dict):
        self.X_a = data_dict['X_a']
        self.X_b = data_dict['X_b']
        self.X_g = data_dict['antigen']
        self.len = self.X_a.shape[0]

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return (self.X_a[idx], self.X_b[idx], self.X_g[idx], 0.0)

# ================= 3. 核心绘图函数：三联柱状图 =================
def save_case_study_plot(all_predictions, save_path, dpi=300):
    print(f"\n🎨 正在绘制图11...")
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['mathtext.fontset'] = 'stix'
    
    # 创建 1行3列 的子图
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=dpi)
    
    pdb_ids = ['1BQL', '6MQR', '7BZ5']
    
    # 遍历三个数据集进行绘图
    for i, pdb_id in enumerate(pdb_ids):
        ax = axes[i]
        preds = all_predictions[pdb_id]
        n_candidates = len(preds)
        
        # 核心修改：获取指定位置的索引
        native_idx_1_based = NATIVE_INDICES.get(pdb_id, 1) 
        native_idx = native_idx_1_based - 1  # 转换为0基索引
        
        # 设置颜色（原生抗体用红色，其他用蓝色）
        colors = ['#D9534F' if j == native_idx else '#4C72B0' for j in range(n_candidates)]
        
        # 绘制柱状图
        ax.bar(range(n_candidates), preds, color=colors, edgecolor='black', linewidth=0.8, alpha=0.85)
        
        # 设置标题和坐标轴 (严格使用 ΔG，匹配你原图格式)
        ax.set_title(f'Predicted $\\Delta G$ for All Samples\n(Red = Index {native_idx_1_based})', fontsize=13, fontweight='bold')
        ax.set_xlabel('Sample Index (1-based)', fontsize=11)
        ax.set_ylabel('Predicted $\\Delta G$ (kcal/mol)', fontsize=11)
        
        # 设置 X 轴刻度
        ax.set_xticks(range(0, n_candidates, 5))  # 每5个显示一个刻度，避免拥挤
        ax.set_xticklabels(range(1, n_candidates + 1, 5), fontsize=10)
        
        # 添加网格线
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        ax.set_axisbelow(True)
        
        # 添加统计信息框 (原图中有，这里还原)
        stats_text = (f"Count: {len(preds):,}\nMean: {np.mean(preds):.2f}\nStd: {np.std(preds):.2f}\nMin: {np.min(preds):.2f}\nMax: {np.max(preds):.2f}")
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.6, edgecolor='gray')
        ax.text(0.98, 0.95, stats_text, transform=ax.transAxes, fontsize=9, 
                verticalalignment='top', horizontalalignment='right', bbox=props)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight', format='png', facecolor='white')
    plt.close(fig)
    print(f"✅ 图片已保存至: {save_path}")

# ================= 4. 主流程：预测三个数据集 =================
def main():
    print(f"🚀 开始图11案例研究预测任务... 设备: {DEVICE}")

    # 加载模型和 Scaler
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"❌ 模型文件不存在: {MODEL_PATH}")
        
    print("⚙️ 初始化模型...")
    model = CombinedModel(
        [getCDRPos("H1"), getCDRPos("H2"), getCDRPos("H3")],
        [getCDRPos("L1"), getCDRPos("L2"), getCDRPos("L3")],
        num_heads=2,
        embed_dim=532,
        antigen_embed_dim=500
    ).to(DEVICE)

    print(f"⚖️ 加载模型权重: {MODEL_PATH}")
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    try:
        model.load_state_dict(state_dict, strict=True)
        print("✅ 模型权重加载成功！")
    except RuntimeError as e:
        print(f"❌ 加载失败，尝试非严格加载... {e}")
        model.load_state_dict(state_dict, strict=False)
        print("⚠️ 非严格加载成功！")
    
    model.eval()
    label_scaler = checkpoint.get('label_scaler', None)

    all_predictions = {}
    all_preds_scaled = {}

    # 循环处理 1BQL, 6MQR, 7BZ5
    for pdb_id, data_path in DATA_PATHS.items():
        print(f"\n📂 处理数据: {pdb_id} ({data_path})")
        if not os.path.exists(data_path):
            print(f"⚠️ 文件不存在，跳过: {data_path}")
            continue

        data_dict = torch.load(data_path, map_location='cpu')
        dataset = UnlabeledDataset(data_dict)
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
        print(f"📊 候选抗体数量: {len(dataset)}")
        
        preds_scaled = []
        with torch.no_grad():
            for X_a, X_b, X_g, _ in loader:
                if X_a.shape[0] == 0: continue
                X_a, X_b, X_g = X_a.to(DEVICE), X_b.to(DEVICE), X_g.to(DEVICE)
                try:
                    pred = model(X_b, X_a, X_g).view(-1)
                    preds_scaled.extend(pred.cpu().numpy())
                except Exception as e:
                    print(f"❌ 预测出错: {e}")
                    return

        # 反归一化
        preds_scaled = np.array(preds_scaled).reshape(-1, 1)
        if label_scaler is not None:
            preds_real = label_scaler.inverse_transform(preds_scaled).flatten()
        else:
            preds_real = preds_scaled.flatten()

        all_predictions[pdb_id] = preds_real
        all_preds_scaled[pdb_id] = preds_scaled.flatten()

        # 保存每个PDB的CSV
        csv_path = f'/root/autodl-tmp/AbAgCDR/fig11/prediction_{pdb_id}.csv'
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        pd.DataFrame({
            'candidate_index': range(1, len(preds_real) + 1), # 保存1-based索引
            'predicted_delta_G_kcal_mol': preds_real
        }).to_csv(csv_path, index=False)
        print(f"💾 结果已保存至: {csv_path}")

    # 画图
    if len(all_predictions) > 0:
        save_case_study_plot(all_predictions, OUTPUT_IMAGE, dpi=300)
        print("\n📈 预测统计摘要:")
        for pdb, preds in all_predictions.items():
            native_idx = NATIVE_INDICES[pdb] - 1
            print(f"  {pdb}: Native(Index {NATIVE_INDICES[pdb]}) ΔG = {preds[native_idx]:.2f}, Mean ΔG = {np.mean(preds):.2f}")
        print("\n✅ 图11生成完毕！")
    else:
        print("❌ 没有成功处理任何数据文件，请检查路径。")

if __name__ == "__main__":
    main()