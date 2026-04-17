# """
# IMMSAB 模型预测脚本 - 简化版
# 只预测结合亲和力值 (ΔG)
# ✅ 每个数据集单独保存为一个文件
# """

# import os
# import numpy as np
# import pandas as pd
# import torch
# import torch.nn as nn
# from torch.utils.data import DataLoader, Dataset
# from tqdm import tqdm

# # === 设置 ===
# MODEL_PATH = "/tmp/AbAgCDR/model/best_modelIMMSAB.pth"
# DATA_DIR = "/tmp/AbAgCDR/data"
# OUTPUT_DIR = "/tmp/AbAgCDR/results"

# # === 模型定义 ===
# class AntibodyAffinityModel(nn.Module):
#     def __init__(self, ab_embed_dim=532, ag_embed_dim=500, cnn_out_channels=64, dropout=0.3):
#         super().__init__()

#         def make_cnn(in_dim, out_channels=cnn_out_channels):
#             return nn.Sequential(
#                 nn.Conv1d(in_dim, out_channels, kernel_size=5, padding=2),
#                 nn.ReLU(),
#                 nn.AdaptiveMaxPool1d(1)
#             )

#         self.cnn_heavy = make_cnn(ab_embed_dim)
#         self.cnn_light = make_cnn(ab_embed_dim)
#         self.cnn_antigen = make_cnn(ag_embed_dim)

#         self.fc = nn.Sequential(
#             nn.Linear(cnn_out_channels * 3, 512),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(512, 1)
#         )

#     def forward(self, heavy, light, antigen):
#         h = self.cnn_heavy(heavy.permute(0, 2, 1))
#         l = self.cnn_light(light.permute(0, 2, 1))
#         g = self.cnn_antigen(antigen.permute(0, 2, 1))
#         out = torch.cat([h.squeeze(-1), l.squeeze(-1), g.squeeze(-1)], dim=1)
#         return self.fc(out).squeeze(-1)


# # === 数据集 ===
# class PredictDataset(Dataset):
#     def __init__(self, samples):
#         self.samples = samples
    
#     def __len__(self):
#         return len(self.samples)
    
#     def __getitem__(self, idx):
#         heavy, light, antigen, delta_g = self.samples[idx]
#         return {
#             'heavy': torch.tensor(heavy, dtype=torch.float32),
#             'light': torch.tensor(light, dtype=torch.float32),
#             'antigen': torch.tensor(antigen, dtype=torch.float32),
#             'delta_g': torch.tensor(delta_g, dtype=torch.float32)
#         }


# def collate_fn(batch):
#     heavy = torch.stack([item['heavy'] for item in batch])
#     light = torch.stack([item['light'] for item in batch])
#     antigen = torch.stack([item['antigen'] for item in batch])
#     delta_g = torch.stack([item['delta_g'] for item in batch])
#     return heavy, light, antigen, delta_g


# # === 主程序 ===
# def main():
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"使用设备：{device}")
    
#     # 1. 加载模型
#     print(f"\n加载模型：{MODEL_PATH}")
#     checkpoint = torch.load(MODEL_PATH, map_location=device)
    
#     model = AntibodyAffinityModel()
#     model.load_state_dict(checkpoint['model_state_dict'])
#     model = model.to(device)
#     model.eval()
    
#     label_scaler = checkpoint.get('label_scaler', None)
    
#     if label_scaler is not None:
#         print(f"✅ 模型加载完成")
#         print(f"   标准化器均值：{label_scaler.mean_[0]:.4f}")
#         print(f"   标准化器标准差：{np.sqrt(label_scaler.var_[0]):.4f}")
#     else:
#         print(f"⚠️ 未找到标签标准化器，将使用原始值")
    
#     # 2. 定义要预测的数据集
#     datasets = {
#         'Paddle': f"{DATA_DIR}/final_dataset_train.pt",
#         'AbBind': f"{DATA_DIR}/abbind_data.pt",
#         'SAbDab': f"{DATA_DIR}/sabdab_data.pt",
#         'SKEMPI': f"{DATA_DIR}/skempi_data.pt",
#         'Benchmark': f"{DATA_DIR}/benchmark_data.pt",
#     }
    
#     os.makedirs(OUTPUT_DIR, exist_ok=True)
    
#     # 3. 预测每个数据集
#     for name, path in datasets.items():
#         if not os.path.exists(path):
#             print(f"\n⚠️  跳过 {name}: 文件不存在")
#             continue
        
#         print(f"\n{'='*60}")
#         print(f"🔮 预测 {name} 数据集")
#         print(f"{'='*60}")
#         data = torch.load(path, map_location='cpu')
        
#         X_a = data.get("X_a", [])
#         X_b = data.get("X_b", [])
#         antigen = data.get("antigen", [])
#         y = data.get("y", [])
        
#         samples = list(zip(X_a, X_b, antigen, y))
#         dataset = PredictDataset(samples)
#         loader = DataLoader(dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
        
#         # 预测
#         preds_scaled = []
#         labels_scaled = []
#         with torch.no_grad():
#             for heavy, light, antigen, delta_g in loader:
#                 heavy, light, antigen = heavy.to(device), light.to(device), antigen.to(device)
#                 pred = model(heavy, light, antigen)
#                 preds_scaled.extend(pred.cpu().numpy())
#                 labels_scaled.extend(delta_g.numpy())
        
#         preds_scaled = np.array(preds_scaled).flatten()
#         labels_scaled = np.array(labels_scaled).flatten()
        
#         # 反归一化
#         if label_scaler is not None:
#             preds = label_scaler.inverse_transform(preds_scaled.reshape(-1, 1)).flatten()
#             labels = label_scaler.inverse_transform(labels_scaled.reshape(-1, 1)).flatten()
#             print(f"   标准化值范围：[{labels_scaled.min():.4f}, {labels_scaled.max():.4f}]")
#             print(f"   反归一化后范围：[{labels.min():.4f}, {labels.max():.4f}]")
#         else:
#             preds = preds_scaled
#             labels = labels_scaled
        
#         # 👈 保存当前数据集的结果
#         output_file = f"{OUTPUT_DIR}/IMMSAB_{name.lower()}_predicted_affinity.tsv"
#         df = pd.DataFrame({
#             'Index': range(1, len(preds) + 1),
#             'True_delta_g': labels,
#             'Predicted_delta_g': preds
#         })
#         df.to_csv(output_file, sep='\t', index=False)
        
#         print(f"\n📊 {name} 结果:")
#         print(f"   样本数：{len(preds)}")
#         print(f"   真实值范围：[{labels.min():.4f}, {labels.max():.4f}]")
#         print(f"   预测值范围：[{preds.min():.4f}, {preds.max():.4f}]")
#         print(f"   真实值均值：{labels.mean():.4f}")
#         print(f"   预测值均值：{preds.mean():.4f}")
#         print(f"   结果保存至：{output_file}")
#         print(f"\n前 10 条预览:")
#         print(df.head(10).to_string(index=False))
    
#     # 4. 打印汇总
#     print("\n" + "="*60)
#     print("✅ 所有数据集预测完成！")
#     print("="*60)
#     print("\n输出文件列表:")
#     for name in datasets.keys():
#         output_file = f"{OUTPUT_DIR}/IMMSAB_{name.lower()}_predicted_affinity.tsv"
#         if os.path.exists(output_file):
#             size = os.path.getsize(output_file)
#             print(f"   📄 {output_file} ({size:,} bytes)")


# if __name__ == "__main__":
#     main()

# """
# IMMSAB 模型预测脚本 - 简化版
# 只预测结合亲和力值 (ΔG)
# ✅ 支持 TSV 和 PT 两种数据格式
# ✅ 每个数据集单独保存为一个文件
# """


# """
# IMMSAB 模型预测脚本
# ✅ 自动查找 PT 和 TSV 数据文件
# ✅ 每个数据集单独保存为一个文件
# """

# import os
# import numpy as np
# import pandas as pd
# import torch
# import torch.nn as nn
# from torch.utils.data import DataLoader, Dataset
# from tqdm import tqdm

# # === 设置 ===
# MODEL_PATH = "/tmp/AbAgCDR/model/best_modelIMMSAB.pth"
# DATA_DIR = "/tmp/AbAgCDR/data"
# OUTPUT_DIR = "/tmp/AbAgCDR/results"
# EMBED_DIR = "/tmp/AbAgCDR/embeddings"

# # === 模型定义 ===
# class AntibodyAffinityModel(nn.Module):
#     def __init__(self, ab_embed_dim=532, ag_embed_dim=500, cnn_out_channels=64, dropout=0.3):
#         super().__init__()
#         def make_cnn(in_dim, out_channels=cnn_out_channels):
#             return nn.Sequential(
#                 nn.Conv1d(in_dim, out_channels, kernel_size=5, padding=2),
#                 nn.ReLU(),
#                 nn.AdaptiveMaxPool1d(1)
#             )
#         self.cnn_heavy = make_cnn(ab_embed_dim)
#         self.cnn_light = make_cnn(ab_embed_dim)
#         self.cnn_antigen = make_cnn(ag_embed_dim)
#         self.fc = nn.Sequential(
#             nn.Linear(cnn_out_channels * 3, 512),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(512, 1)
#         )

#     def forward(self, heavy, light, antigen):
#         h = self.cnn_heavy(heavy.permute(0, 2, 1))
#         l = self.cnn_light(light.permute(0, 2, 1))
#         g = self.cnn_antigen(antigen.permute(0, 2, 1))
#         out = torch.cat([h.squeeze(-1), l.squeeze(-1), g.squeeze(-1)], dim=1)
#         return self.fc(out).squeeze(-1)


# class PredictDataset(Dataset):
#     def __init__(self, samples):
#         self.samples = samples
#     def __len__(self):
#         return len(self.samples)
#     def __getitem__(self, idx):
#         heavy, light, antigen, delta_g = self.samples[idx]
#         return {
#             'heavy': torch.tensor(heavy, dtype=torch.float32),
#             'light': torch.tensor(light, dtype=torch.float32),
#             'antigen': torch.tensor(antigen, dtype=torch.float32),
#             'delta_g': torch.tensor(delta_g, dtype=torch.float32)
#         }

# def collate_fn(batch):
#     heavy = torch.stack([item['heavy'] for item in batch])
#     light = torch.stack([item['light'] for item in batch])
#     antigen = torch.stack([item['antigen'] for item in batch])
#     delta_g = torch.stack([item['delta_g'] for item in batch])
#     return heavy, light, antigen, delta_g


# # === 加载 PT 格式数据 ===
# def load_pt_data(path):
#     """加载 .pt 格式的预处理特征数据"""
#     print(f"   📥 加载 PT 文件：{path}")
#     data = torch.load(path, map_location='cpu')
#     X_a = data.get("X_a", data.get("heavy", []))
#     X_b = data.get("X_b", data.get("light", []))
#     antigen = data.get("antigen", [])
#     y = data.get("y", data.get("labels", []))
#     return X_a, X_b, antigen, y


# # === 从 embeddings 目录加载 TSV 对应的嵌入 ===
# def load_tsv_with_embeddings(tsv_path, embed_dir):
#     """从 TSV 文件加载标签，从 embed_dir 加载预计算嵌入"""
#     print(f"   📖 读取 TSV 文件：{tsv_path}")
#     df = pd.read_csv(tsv_path, sep='\t')
    
#     # 检查必需的列
#     required_cols = ['antibody_seq_a', 'antibody_seq_b', 'antigen_seq', 'delta_g']
#     for col in required_cols:
#         if col not in df.columns:
#             raise ValueError(f"缺少必要列：{col}")
    
#     y = df['delta_g'].values
    
#     # 尝试从 embeddings 目录加载
#     if embed_dir and os.path.exists(embed_dir):
#         print(f"   🔄 从 {embed_dir} 加载预计算嵌入...")
        
#         # 列出 embeddings 目录中的所有 npy 文件
#         npy_files = [f for f in os.listdir(embed_dir) if f.endswith('.npy')]
#         print(f"   找到 {len(npy_files)} 个嵌入文件：{npy_files[:5]}...")
        
#         # 尝试匹配文件名
#         base_name = os.path.basename(tsv_path).replace('.tsv', '')
#         print(f"   尝试匹配：{base_name}")
        
#         # 尝试多种命名方式
#         possible_names = [
#             base_name,
#             base_name.replace('pairs_seq_', ''),
#             base_name.replace('_abbind2', '_abbind'),
#         ]
        
#         for name in possible_names:
#             # 尝试加载 hchain/lchain/ag 或 heavy/light/antigen
#             for pattern in [('hchain.npy', 'lchain.npy', 'ag.npy'),
#                             ('heavy.npy', 'light.npy', 'antigen.npy'),
#                             ('hchain_adj.npy', 'lchain_adj.npy', 'ag_adj.npy')]:
#                 h_path = f"{embed_dir}/{pattern[0]}"
#                 l_path = f"{embed_dir}/{pattern[1]}"
#                 ag_path = f"{embed_dir}/{pattern[2]}"
                
#                 if os.path.exists(h_path) and os.path.exists(l_path) and os.path.exists(ag_path):
#                     print(f"   ✅ 找到嵌入文件：{pattern}")
#                     h_embs = np.load(h_path)
#                     l_embs = np.load(l_path)
#                     ag_embs = np.load(ag_path)
                    
#                     # 尝试从 sources.npy 匹配当前数据集
#                     sources_path = f"{embed_dir}/sources.npy"
#                     if os.path.exists(sources_path):
#                         sources = np.load(sources_path, allow_pickle=True)
#                         print(f"   数据源：{np.unique(sources)}")
                        
#                         # 找到匹配的索引
#                         source_name = base_name
#                         matching_idx = np.where(sources == source_name)[0]
                        
#                         if len(matching_idx) == 0:
#                             # 尝试模糊匹配
#                             for src in np.unique(sources):
#                                 if base_name.split('_')[-1] in str(src).lower():
#                                     matching_idx = np.where(sources == src)[0]
#                                     print(f"   模糊匹配到：{src}")
#                                     break
                        
#                         if len(matching_idx) > 0:
#                             matching_idx = matching_idx[:len(y)]
#                             return h_embs[matching_idx], l_embs[matching_idx], ag_embs[matching_idx], y
#                     else:
#                         # 没有 sources，直接返回全部
#                         return h_embs[:len(y)], l_embs[:len(y)], ag_embs[:len(y)], y
        
#         print(f"   ⚠️  未找到匹配的嵌入文件")
    
#     raise ValueError(f"无法加载嵌入特征，请检查 {embed_dir} 目录")


# # === 自动查找数据文件 ===
# def find_available_datasets(data_dir, embed_dir):
#     """自动查找所有可用的数据集"""
#     print(f"\n🔍 搜索数据文件...")
#     print(f"   数据目录：{data_dir}")
#     print(f"   嵌入目录：{embed_dir}")
    
#     # 数据集名称和可能的文件名
#     dataset_configs = {
#         'Paddle': {
#             'pt_files': ['train_data.pt', 'final_dataset_train.pt', 'paddle_data.pt'],
#             'tsv_files': ['final_dataset_train.tsv', 'train.tsv'],
#         },
#         'AbBind': {
#             'pt_files': ['abbind_data.pt', 'pairs_seq_abbind2.pt'],
#             'tsv_files': ['pairs_seq_abbind2.tsv', 'abbind.tsv'],
#         },
#         'SAbDab': {
#             'pt_files': ['sabdab_data.pt', 'pairs_seq_sabdab.pt'],
#             'tsv_files': ['pairs_seq_sabdab.tsv', 'sabdab.tsv'],
#         },
#         'SKEMPI': {
#             'pt_files': ['skempi_data.pt', 'pairs_seq_skempi.pt'],
#             'tsv_files': ['pairs_seq_skempi.tsv', 'skempi.tsv'],
#         },
#         'Benchmark': {
#             'pt_files': ['benchmark_data.pt', 'pairs_seq_benchmark1.pt'],
#             'tsv_files': ['pairs_seq_benchmark1.tsv', 'benchmark.tsv'],
#         },
#     }
    
#     available_datasets = {}
    
#     for name, config in dataset_configs.items():
#         file_path = None
#         file_type = None
        
#         # 优先查找 .pt 文件
#         for filename in config['pt_files']:
#             path = f"{data_dir}/{filename}"
#             if os.path.exists(path):
#                 file_path = path
#                 file_type = 'pt'
#                 break
        
#         # 如果没有 .pt，查找 .tsv
#         if file_path is None:
#             for filename in config['tsv_files']:
#                 path = f"{data_dir}/{filename}"
#                 if os.path.exists(path):
#                     file_path = path
#                     file_type = 'tsv'
#                     break
        
#         if file_path:
#             available_datasets[name] = {'path': file_path, 'type': file_type}
#             print(f"   ✅ {name}: {file_path} ({file_type})")
#         else:
#             print(f"   ❌ {name}: 未找到数据文件")
    
#     # 显示目录中所有相关文件
#     print(f"\n📂 数据目录内容:")
#     for f in sorted(os.listdir(data_dir)):
#         if f.endswith('.pt') or f.endswith('.tsv'):
#             size = os.path.getsize(f"{data_dir}/{f}")
#             print(f"      {f} ({size:,} bytes)")
    
#     if os.path.exists(embed_dir):
#         print(f"\n📂 嵌入目录内容:")
#         for f in sorted(os.listdir(embed_dir))[:10]:
#             size = os.path.getsize(f"{embed_dir}/{f}")
#             print(f"      {f} ({size:,} bytes)")
    
#     return available_datasets


# # === 主程序 ===
# def main():
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"使用设备：{device}")
    
#     # 1. 加载模型
#     print(f"\n加载模型：{MODEL_PATH}")
#     checkpoint = torch.load(MODEL_PATH, map_location=device)
    
#     model = AntibodyAffinityModel()
#     model.load_state_dict(checkpoint['model_state_dict'])
#     model = model.to(device)
#     model.eval()
    
#     label_scaler = checkpoint.get('label_scaler', None)
#     if label_scaler is not None:
#         print(f"✅ 模型加载完成")
#         print(f"   标准化器均值：{label_scaler.mean_[0]:.4f}")
#         print(f"   标准化器标准差：{np.sqrt(label_scaler.var_[0]):.4f}")
    
#     # 2. 自动查找可用数据集
#     datasets = find_available_datasets(DATA_DIR, EMBED_DIR)
    
#     if not datasets:
#         print("\n❌ 错误：未找到任何可用的数据文件！")
#         return
    
#     os.makedirs(OUTPUT_DIR, exist_ok=True)
    
#     # 3. 预测每个数据集
#     results_summary = {}
    
#     for name, info in datasets.items():
#         path = info['path']
#         file_type = info['type']
        
#         print(f"\n{'='*60}")
#         print(f"🔮 预测 {name} 数据集")
#         print(f"{'='*60}")
#         print(f"   数据文件：{path}")
#         print(f"   文件类型：{file_type}")
        
#         try:
#             # 加载数据
#             if file_type == 'pt':
#                 X_a, X_b, antigen, y = load_pt_data(path)
#             else:
#                 X_a, X_b, antigen, y = load_tsv_with_embeddings(path, EMBED_DIR)
            
#             if len(X_a) == 0:
#                 print(f"   ❌ 数据为空，跳过")
#                 continue
            
#             print(f"   样本数量：{len(X_a)}")
#             print(f"   heavy 维度：{X_a[0].shape if len(X_a) > 0 else 'N/A'}")
#             print(f"   light 维度：{X_b[0].shape if len(X_b) > 0 else 'N/A'}")
#             print(f"   antigen 维度：{antigen[0].shape if len(antigen) > 0 else 'N/A'}")
            
#             samples = list(zip(X_a, X_b, antigen, y))
#             dataset = PredictDataset(samples)
#             loader = DataLoader(dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
            
#             # 预测
#             preds_scaled = []
#             labels_scaled = []
#             with torch.no_grad():
#                 for heavy, light, antigen, delta_g in loader:
#                     heavy, light, antigen = heavy.to(device), light.to(device), antigen.to(device)
#                     pred = model(heavy, light, antigen)
#                     preds_scaled.extend(pred.cpu().numpy())
#                     labels_scaled.extend(delta_g.numpy())
            
#             preds_scaled = np.array(preds_scaled).flatten()
#             labels_scaled = np.array(labels_scaled).flatten()
            
#             # 反归一化
#             if label_scaler is not None:
#                 preds = label_scaler.inverse_transform(preds_scaled.reshape(-1, 1)).flatten()
#                 labels = label_scaler.inverse_transform(labels_scaled.reshape(-1, 1)).flatten()
#                 print(f"   标准化值范围：[{labels_scaled.min():.4f}, {labels_scaled.max():.4f}]")
#                 print(f"   反归一化后范围：[{labels.min():.4f}, {labels.max():.4f}]")
#             else:
#                 preds = preds_scaled
#                 labels = labels_scaled
            
#             # 保存结果
#             output_file = f"{OUTPUT_DIR}/IMMSAB_{name.lower()}_predicted_affinity.tsv"
#             df = pd.DataFrame({
#                 'True_delta_g': labels,
#                 'Predicted_delta_g': preds
#             })
#             df.to_csv(output_file, sep='\t', index=False)
            
#             # 计算指标
#             pcc = np.corrcoef(labels, preds)[0, 1] if len(labels) > 1 else 0
#             rmse = np.sqrt(np.mean((labels - preds) ** 2))
#             mae = np.mean(np.abs(labels - preds))
#             r2 = 1 - np.sum((labels - preds) ** 2) / (np.sum((labels - np.mean(labels)) ** 2) + 1e-8)
            
#             results_summary[name] = {'N': len(preds), 'PCC': pcc, 'R2': r2, 'RMSE': rmse, 'MAE': mae}
            
#             print(f"\n📊 {name} 结果:")
#             print(f"   样本数：{len(preds)}")
#             print(f"   PCC: {pcc:.4f}")
#             print(f"   R²: {r2:.4f}")
#             print(f"   RMSE: {rmse:.4f}")
#             print(f"   MAE: {mae:.4f}")
#             print(f"   真实值范围：[{labels.min():.4f}, {labels.max():.4f}]")
#             print(f"   预测值范围：[{preds.min():.4f}, {preds.max():.4f}]")
#             print(f"   结果保存至：{output_file}")
#             print(f"\n前 10 条预览:")
#             print(df.head(10).to_string(index=False))
            
#         except Exception as e:
#             print(f"   ❌ 处理失败：{e}")
#             import traceback
#             traceback.print_exc()
    
#     # 4. 打印汇总
#     print("\n" + "="*60)
#     print("✅ 预测完成！")
#     print("="*60)
    
#     if results_summary:
#         print("\n📊 结果汇总:")
#         print(f"{'Dataset':<12} | {'N':>6} | {'PCC':>8} | {'R²':>8} | {'RMSE':>8} | {'MAE':>8}")
#         print("-"*60)
#         for name, metrics in results_summary.items():
#             print(f"{name:<12} | {metrics['N']:>6} | {metrics['PCC']:>8.4f} | {metrics['R2']:>8.4f} | {metrics['RMSE']:>8.4f} | {metrics['MAE']:>8.4f}")
    
#     print("\n📁 输出文件列表:")
#     for name in datasets.keys():
#         output_file = f"{OUTPUT_DIR}/IMMSAB_{name.lower()}_predicted_affinity.tsv"
#         if os.path.exists(output_file):
#             size = os.path.getsize(output_file)
#             print(f"   ✅ {output_file} ({size:,} bytes)")
#         else:
#             print(f"   ❌ {output_file} (未生成)")


# if __name__ == "__main__":
#     main()

# """
# IMMSAB 模型预测脚本
# ✅ 自动查找 PT 和 TSV 数据文件
# ✅ 每个数据集单独保存为一个文件
# ✅ 生成真实值 vs 预测值回归图
# """

# import os
# import numpy as np
# import pandas as pd
# import torch
# import torch.nn as nn
# from torch.utils.data import DataLoader, Dataset
# from tqdm import tqdm
# import matplotlib.pyplot as plt
# from scipy import stats
# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# # === 设置 ===
# MODEL_PATH = "/tmp/AbAgCDR/model/best_modelIMMSAB.pth"
# DATA_DIR = "/tmp/AbAgCDR/data"
# OUTPUT_DIR = "/tmp/AbAgCDR/results"
# PLOT_DIR = "/tmp/AbAgCDR/results/plots"  # 回归图保存目录
# EMBED_DIR = "/tmp/AbAgCDR/embeddings"

# # === 模型定义 ===
# class AntibodyAffinityModel(nn.Module):
#     def __init__(self, ab_embed_dim=532, ag_embed_dim=500, cnn_out_channels=64, dropout=0.3):
#         super().__init__()
#         def make_cnn(in_dim, out_channels=cnn_out_channels):
#             return nn.Sequential(
#                 nn.Conv1d(in_dim, out_channels, kernel_size=5, padding=2),
#                 nn.ReLU(),
#                 nn.AdaptiveMaxPool1d(1)
#             )
#         self.cnn_heavy = make_cnn(ab_embed_dim)
#         self.cnn_light = make_cnn(ab_embed_dim)
#         self.cnn_antigen = make_cnn(ag_embed_dim)
#         self.fc = nn.Sequential(
#             nn.Linear(cnn_out_channels * 3, 512),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(512, 1)
#         )

#     def forward(self, heavy, light, antigen):
#         h = self.cnn_heavy(heavy.permute(0, 2, 1))
#         l = self.cnn_light(light.permute(0, 2, 1))
#         g = self.cnn_antigen(antigen.permute(0, 2, 1))
#         out = torch.cat([h.squeeze(-1), l.squeeze(-1), g.squeeze(-1)], dim=1)
#         return self.fc(out).squeeze(-1)


# class PredictDataset(Dataset):
#     def __init__(self, samples):
#         self.samples = samples
#     def __len__(self):
#         return len(self.samples)
#     def __getitem__(self, idx):
#         heavy, light, antigen, delta_g = self.samples[idx]
#         return {
#             'heavy': torch.tensor(heavy, dtype=torch.float32),
#             'light': torch.tensor(light, dtype=torch.float32),
#             'antigen': torch.tensor(antigen, dtype=torch.float32),
#             'delta_g': torch.tensor(delta_g, dtype=torch.float32)
#         }

# def collate_fn(batch):
#     heavy = torch.stack([item['heavy'] for item in batch])
#     light = torch.stack([item['light'] for item in batch])
#     antigen = torch.stack([item['antigen'] for item in batch])
#     delta_g = torch.stack([item['delta_g'] for item in batch])
#     return heavy, light, antigen, delta_g


# # === 加载 PT 格式数据 ===
# def load_pt_data(path):
#     """加载 .pt 格式的预处理特征数据"""
#     print(f"   📥 加载 PT 文件：{path}")
#     data = torch.load(path, map_location='cpu')
#     X_a = data.get("X_a", data.get("heavy", []))
#     X_b = data.get("X_b", data.get("light", []))
#     antigen = data.get("antigen", [])
#     y = data.get("y", data.get("labels", []))
#     return X_a, X_b, antigen, y


# # === 从 embeddings 目录加载 TSV 对应的嵌入 ===
# def load_tsv_with_embeddings(tsv_path, embed_dir):
#     """从 TSV 文件加载标签，从 embed_dir 加载预计算嵌入"""
#     print(f"   📖 读取 TSV 文件：{tsv_path}")
#     df = pd.read_csv(tsv_path, sep='\t')
    
#     required_cols = ['antibody_seq_a', 'antibody_seq_b', 'antigen_seq', 'delta_g']
#     for col in required_cols:
#         if col not in df.columns:
#             raise ValueError(f"缺少必要列：{col}")
    
#     y = df['delta_g'].values
    
#     if embed_dir and os.path.exists(embed_dir):
#         print(f"   🔄 从 {embed_dir} 加载预计算嵌入...")
        
#         npy_files = [f for f in os.listdir(embed_dir) if f.endswith('.npy')]
#         print(f"   找到 {len(npy_files)} 个嵌入文件")
        
#         base_name = os.path.basename(tsv_path).replace('.tsv', '')
#         print(f"   尝试匹配：{base_name}")
        
#         possible_names = [
#             base_name,
#             base_name.replace('pairs_seq_', ''),
#             base_name.replace('_abbind2', '_abbind'),
#         ]
        
#         for name in possible_names:
#             for pattern in [('hchain.npy', 'lchain.npy', 'ag.npy'),
#                             ('heavy.npy', 'light.npy', 'antigen.npy'),
#                             ('hchain_adj.npy', 'lchain_adj.npy', 'ag_adj.npy')]:
#                 h_path = f"{embed_dir}/{pattern[0]}"
#                 l_path = f"{embed_dir}/{pattern[1]}"
#                 ag_path = f"{embed_dir}/{pattern[2]}"
                
#                 if os.path.exists(h_path) and os.path.exists(l_path) and os.path.exists(ag_path):
#                     print(f"   ✅ 找到嵌入文件：{pattern}")
#                     h_embs = np.load(h_path)
#                     l_embs = np.load(l_path)
#                     ag_embs = np.load(ag_path)
                    
#                     sources_path = f"{embed_dir}/sources.npy"
#                     if os.path.exists(sources_path):
#                         sources = np.load(sources_path, allow_pickle=True)
#                         print(f"   数据源：{np.unique(sources)}")
                        
#                         source_name = base_name
#                         matching_idx = np.where(sources == source_name)[0]
                        
#                         if len(matching_idx) == 0:
#                             for src in np.unique(sources):
#                                 if base_name.split('_')[-1] in str(src).lower():
#                                     matching_idx = np.where(sources == src)[0]
#                                     print(f"   模糊匹配到：{src}")
#                                     break
                        
#                         if len(matching_idx) > 0:
#                             matching_idx = matching_idx[:len(y)]
#                             return h_embs[matching_idx], l_embs[matching_idx], ag_embs[matching_idx], y
#                     else:
#                         return h_embs[:len(y)], l_embs[:len(y)], ag_embs[:len(y)], y
        
#         print(f"   ⚠️  未找到匹配的嵌入文件")
    
#     raise ValueError(f"无法加载嵌入特征，请检查 {embed_dir} 目录")


# # === 自动查找数据文件 ===
# def find_available_datasets(data_dir, embed_dir):
#     """自动查找所有可用的数据集"""
#     print(f"\n🔍 搜索数据文件...")
    
#     dataset_configs = {
#         'Paddle': {
#             'pt_files': ['train_data.pt', 'final_dataset_train.pt', 'paddle_data.pt'],
#             'tsv_files': ['final_dataset_train.tsv', 'train.tsv'],
#         },
#         'AbBind': {
#             'pt_files': ['abbind_data.pt', 'pairs_seq_abbind2.pt'],
#             'tsv_files': ['pairs_seq_abbind2.tsv', 'abbind.tsv'],
#         },
#         'SAbDab': {
#             'pt_files': ['sabdab_data.pt', 'pairs_seq_sabdab.pt'],
#             'tsv_files': ['pairs_seq_sabdab.tsv', 'sabdab.tsv'],
#         },
#         'SKEMPI': {
#             'pt_files': ['skempi_data.pt', 'pairs_seq_skempi.pt'],
#             'tsv_files': ['pairs_seq_skempi.tsv', 'skempi.tsv'],
#         },
#         'Benchmark': {
#             'pt_files': ['benchmark_data.pt', 'pairs_seq_benchmark1.pt'],
#             'tsv_files': ['pairs_seq_benchmark1.tsv', 'benchmark.tsv'],
#         },
#     }
    
#     available_datasets = {}
    
#     for name, config in dataset_configs.items():
#         file_path = None
#         file_type = None
        
#         for filename in config['pt_files']:
#             path = f"{data_dir}/{filename}"
#             if os.path.exists(path):
#                 file_path = path
#                 file_type = 'pt'
#                 break
        
#         if file_path is None:
#             for filename in config['tsv_files']:
#                 path = f"{data_dir}/{filename}"
#                 if os.path.exists(path):
#                     file_path = path
#                     file_type = 'tsv'
#                     break
        
#         if file_path:
#             available_datasets[name] = {'path': file_path, 'type': file_type}
#             print(f"   ✅ {name}: {file_path} ({file_type})")
#         else:
#             print(f"   ❌ {name}: 未找到数据文件")
    
#     return available_datasets


# # === 绘制回归图 ===
# def plot_regression(y_true, y_pred, output_file, dataset_name="Dataset"):
#     """绘制真实值 vs 预测值回归图"""
#     fig, ax = plt.subplots(figsize=(10, 8))
    
#     # 散点图
#     ax.scatter(y_true, y_pred, alpha=0.6, s=50, color='steelblue', 
#                edgecolors='black', linewidth=0.5, label='Predictions')
    
#     # 理想拟合线 (y=x)
#     min_val = min(y_true.min(), y_pred.min())
#     max_val = max(y_true.max(), y_pred.max())
#     margin = (max_val - min_val) * 0.1
#     min_val -= margin
#     max_val += margin
    
#     ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, 
#             label='Ideal fit (y=x)')
    
#     # 线性回归线
#     slope, intercept, r_value, p_value, std_err = stats.linregress(y_true, y_pred)
#     regression_line = slope * np.array([min_val, max_val]) + intercept
#     ax.plot([min_val, max_val], regression_line, 'g-', linewidth=2, 
#             label=f'Linear fit (R²={r_value**2:.4f})')
    
#     # 计算指标
#     pcc = np.corrcoef(y_true, y_pred)[0, 1] if len(y_true) > 1 else 0
#     rmse = np.sqrt(mean_squared_error(y_true, y_pred))
#     mae = mean_absolute_error(y_true, y_pred)
#     r2 = r2_score(y_true, y_pred)
    
#     # 添加文本框
#     textstr = (f'Dataset: {dataset_name}\n'
#                f'N = {len(y_true)}\n'
#                f'PCC = {pcc:.4f}\n'
#                f'R² = {r2:.4f}\n'
#                f'RMSE = {rmse:.4f}\n'
#                f'MAE = {mae:.4f}')
#     props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
#     ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=12,
#             verticalalignment='top', bbox=props)
    
#     # 标签和标题
#     ax.set_xlabel('True ΔG (kcal/mol)', fontsize=14)
#     ax.set_ylabel('Predicted ΔG (kcal/mol)', fontsize=14)
#     ax.set_title(f'Antibody-Antigen Binding Affinity Prediction\n{dataset_name}', 
#                  fontsize=16, fontweight='bold')
#     ax.legend(loc='lower right', fontsize=12)
#     ax.grid(True, alpha=0.3, linestyle='--')
    
#     plt.tight_layout()
#     plt.savefig(output_file, dpi=300, bbox_inches='tight')
#     plt.close()
    
#     print(f"   📊 回归图已保存：{output_file}")
    
#     return {'PCC': pcc, 'R2': r2, 'RMSE': rmse, 'MAE': mae}


# # === 主程序 ===
# def main():
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"使用设备：{device}")
    
#     # 创建输出目录
#     os.makedirs(OUTPUT_DIR, exist_ok=True)
#     os.makedirs(PLOT_DIR, exist_ok=True)
    
#     # 1. 加载模型
#     print(f"\n加载模型：{MODEL_PATH}")
#     checkpoint = torch.load(MODEL_PATH, map_location=device)
    
#     model = AntibodyAffinityModel()
#     model.load_state_dict(checkpoint['model_state_dict'])
#     model = model.to(device)
#     model.eval()
    
#     label_scaler = checkpoint.get('label_scaler', None)
#     if label_scaler is not None:
#         print(f"✅ 模型加载完成")
#         print(f"   标准化器均值：{label_scaler.mean_[0]:.4f}")
#         print(f"   标准化器标准差：{np.sqrt(label_scaler.var_[0]):.4f}")
    
#     # 2. 自动查找可用数据集
#     datasets = find_available_datasets(DATA_DIR, EMBED_DIR)
    
#     if not datasets:
#         print("\n❌ 错误：未找到任何可用的数据文件！")
#         return
    
#     # 3. 预测每个数据集
#     results_summary = {}
#     all_y_true = []
#     all_y_pred = []
#     all_datasets = []
    
#     for name, info in datasets.items():
#         path = info['path']
#         file_type = info['type']
        
#         print(f"\n{'='*60}")
#         print(f"🔮 预测 {name} 数据集")
#         print(f"{'='*60}")
        
#         try:
#             # 加载数据
#             if file_type == 'pt':
#                 X_a, X_b, antigen, y = load_pt_data(path)
#             else:
#                 X_a, X_b, antigen, y = load_tsv_with_embeddings(path, EMBED_DIR)
            
#             if len(X_a) == 0:
#                 print(f"   ❌ 数据为空，跳过")
#                 continue
            
#             print(f"   样本数量：{len(X_a)}")
            
#             samples = list(zip(X_a, X_b, antigen, y))
#             dataset = PredictDataset(samples)
#             loader = DataLoader(dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
            
#             # 预测
#             preds_scaled = []
#             labels_scaled = []
#             with torch.no_grad():
#                 for heavy, light, antigen, delta_g in loader:
#                     heavy, light, antigen = heavy.to(device), light.to(device), antigen.to(device)
#                     pred = model(heavy, light, antigen)
#                     preds_scaled.extend(pred.cpu().numpy())
#                     labels_scaled.extend(delta_g.numpy())
            
#             preds_scaled = np.array(preds_scaled).flatten()
#             labels_scaled = np.array(labels_scaled).flatten()
            
#             # 反归一化
#             if label_scaler is not None:
#                 preds = label_scaler.inverse_transform(preds_scaled.reshape(-1, 1)).flatten()
#                 labels = label_scaler.inverse_transform(labels_scaled.reshape(-1, 1)).flatten()
#             else:
#                 preds = preds_scaled
#                 labels = labels_scaled
            
#             # 保存预测结果
#             output_file = f"{OUTPUT_DIR}/IMMSAB_{name.lower()}_predicted_affinity.tsv"
#             df = pd.DataFrame({
#                 'Index': range(1, len(preds) + 1),
#                 'True_delta_g': labels,
#                 'Predicted_delta_g': preds
#             })
#             df.to_csv(output_file, sep='\t', index=False)
            
#             # 绘制回归图
#             plot_file = f"{PLOT_DIR}/{name.lower()}_regression.png"
#             metrics = plot_regression(labels, preds, plot_file, dataset_name=name)
            
#             # 保存汇总数据用于总体图
#             all_y_true.extend(labels)
#             all_y_pred.extend(preds)
#             all_datasets.extend([name] * len(labels))
            
#             results_summary[name] = {
#                 'N': len(preds), 
#                 'PCC': metrics['PCC'], 
#                 'R2': metrics['R2'], 
#                 'RMSE': metrics['RMSE'], 
#                 'MAE': metrics['MAE']
#             }
            
#             print(f"\n📊 {name} 结果:")
#             print(f"   样本数：{len(preds)}")
#             print(f"   PCC: {metrics['PCC']:.4f}")
#             print(f"   R²: {metrics['R2']:.4f}")
#             print(f"   RMSE: {metrics['RMSE']:.4f}")
#             print(f"   MAE: {metrics['MAE']:.4f}")
#             print(f"   结果保存至：{output_file}")
            
#         except Exception as e:
#             print(f"   ❌ 处理失败：{e}")
#             import traceback
#             traceback.print_exc()
    
#     # 4. 绘制总体回归图
#     if len(all_y_true) > 0:
#         overall_plot_file = f"{PLOT_DIR}/overall_regression.png"
#         plot_regression(np.array(all_y_true), np.array(all_y_pred), 
#                        overall_plot_file, dataset_name="All Datasets Combined")
    
#     # 5. 打印汇总
#     print("\n" + "="*70)
#     print("✅ 预测完成！")
#     print("="*70)
    
#     if results_summary:
#         print("\n📊 结果汇总:")
#         print(f"{'Dataset':<12} | {'N':>6} | {'PCC':>8} | {'R²':>8} | {'RMSE':>8} | {'MAE':>8}")
#         print("-"*70)
#         for name, metrics in results_summary.items():
#             print(f"{name:<12} | {metrics['N']:>6} | {metrics['PCC']:>8.4f} | {metrics['R2']:>8.4f} | {metrics['RMSE']:>8.4f} | {metrics['MAE']:>8.4f}")
#         print("-"*70)
        
#         # 总体指标
#         overall_pcc = np.corrcoef(all_y_true, all_y_pred)[0, 1]
#         overall_r2 = r2_score(all_y_true, all_y_pred)
#         overall_rmse = np.sqrt(mean_squared_error(all_y_true, all_y_pred))
#         overall_mae = mean_absolute_error(all_y_true, all_y_pred)
#         print(f"{'OVERALL':<12} | {len(all_y_true):>6} | {overall_pcc:>8.4f} | {overall_r2:>8.4f} | {overall_rmse:>8.4f} | {overall_mae:>8.4f}")
    
#     print("\n📁 输出文件列表:")
#     print("\n预测结果文件:")
#     for name in datasets.keys():
#         output_file = f"{OUTPUT_DIR}/IMMSAB_{name.lower()}_predicted_affinity.tsv"
#         if os.path.exists(output_file):
#             size = os.path.getsize(output_file)
#             print(f"   ✅ {output_file} ({size:,} bytes)")
    
#     print("\n回归图文件:")
#     for name in datasets.keys():
#         plot_file = f"{PLOT_DIR}/{name.lower()}_regression.png"
#         if os.path.exists(plot_file):
#             size = os.path.getsize(plot_file)
#             print(f"   ✅ {plot_file} ({size:,} bytes)")
    
#     if os.path.exists(f"{PLOT_DIR}/overall_regression.png"):
#         size = os.path.getsize(f"{PLOT_DIR}/overall_regression.png")
#         print(f"   ✅ {PLOT_DIR}/overall_regression.png ({size:,} bytes)")


# if __name__ == "__main__":
#     main()

"""
IMMSAB 模型预测脚本
✅ 直接读取 TSV 文件 (氨基酸序列)
✅ 自动使用 ESM2 生成嵌入特征
✅ 每个数据集单独保存为一个文件
✅ 生成真实值 vs 预测值回归图
✅ 真实值直接使用 TSV 中的 delta_g（不反归一化）
✅ 仅预测值需要反归一化
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import esm

# === 设置 ===
MODEL_PATH = "/tmp/AbAgCDR/model/best_modelIMMSAB2.pth"
DATA_DIR = "/tmp/AbAgCDR/data"
OUTPUT_DIR = "/tmp/AbAgCDR/results"
PLOT_DIR = "/tmp/AbAgCDR/results/plots"

# TSV 输入文件路径
INPUT_FILES = {
    'Paddle': f"{DATA_DIR}/final_dataset_train.tsv",
    'AbBind': f"{DATA_DIR}/pairs_seq_abbind2.tsv",
    'SAbDab': f"{DATA_DIR}/pairs_seq_sabdab.tsv",
    'SKEMPI': f"{DATA_DIR}/pairs_seq_skempi.tsv",
    'Benchmark': f"{DATA_DIR}/pairs_seq_benchmark1.tsv",
}

# === 模型定义 ===
class AntibodyAffinityModel(nn.Module):
    def __init__(self, ab_embed_dim=532, ag_embed_dim=500, cnn_out_channels=64, dropout=0.3):
        super().__init__()
        def make_cnn(in_dim, out_channels=cnn_out_channels):
            return nn.Sequential(
                nn.Conv1d(in_dim, out_channels, kernel_size=5, padding=2),
                nn.ReLU(),
                nn.AdaptiveMaxPool1d(1)
            )
        self.cnn_heavy = make_cnn(ab_embed_dim)
        self.cnn_light = make_cnn(ab_embed_dim)
        self.cnn_antigen = make_cnn(ag_embed_dim)
        self.fc = nn.Sequential(
            nn.Linear(cnn_out_channels * 3, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 1)
        )

    def forward(self, heavy, light, antigen):
        h = self.cnn_heavy(heavy.permute(0, 2, 1))
        l = self.cnn_light(light.permute(0, 2, 1))
        g = self.cnn_antigen(antigen.permute(0, 2, 1))
        out = torch.cat([h.squeeze(-1), l.squeeze(-1), g.squeeze(-1)], dim=1)
        return self.fc(out).squeeze(-1)


# === ESM2 嵌入提取器 ===
class ESM2Embedder:
    def __init__(self, model_name="esm2_t12_35M_UR50D"):
        print(f"🧠 加载 ESM2 模型 ({model_name})...")
        self.model, self.alphabet = esm.pretrained.esm2_t12_35M_UR50D()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        self.model.eval()
        self.batch_converter = self.alphabet.get_batch_converter()
        print(f"✅ ESM2 加载完成 (设备：{self.device})")
    
    def get_embedding(self, seq):
        seq = str(seq).upper().strip()
        valid_aas = "ARNDCQEGHILKMFPSTWYV"
        seq = ''.join([aa for aa in seq if aa in valid_aas])
        if not seq:
            seq = "A"
        
        data = [("protein", seq)]
        _, _, batch_tokens = self.batch_converter(data)
        batch_tokens = batch_tokens.to(self.device)
        
        with torch.no_grad():
            results = self.model(batch_tokens, repr_layers=[12])
            token_repr = results["representations"][12]
        
        emb = token_repr[0, 1:-1].mean(dim=0).cpu().numpy()
        return emb
    
    def embed_batch(self, sequences, target_dim):
        embeddings = []
        for seq in tqdm(sequences, desc="Extracting embeddings", leave=False):
            emb = self.get_embedding(seq)
            embeddings.append(emb)
        
        embeddings = np.array(embeddings)
        
        if embeddings.shape[1] > target_dim:
            embeddings = embeddings[:, :target_dim]
        elif embeddings.shape[1] < target_dim:
            pad = np.zeros((embeddings.shape[0], target_dim - embeddings.shape[1]))
            embeddings = np.concatenate([embeddings, pad], axis=1)
        
        return embeddings[:, np.newaxis, :].astype(np.float32)


# === 数据集类 ===
class PredictDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        heavy, light, antigen, delta_g = self.samples[idx]
        return {
            'heavy': torch.tensor(heavy, dtype=torch.float32),
            'light': torch.tensor(light, dtype=torch.float32),
            'antigen': torch.tensor(antigen, dtype=torch.float32),
            'delta_g': torch.tensor(delta_g, dtype=torch.float32)
        }

def collate_fn(batch):
    heavy = torch.stack([item['heavy'] for item in batch])
    light = torch.stack([item['light'] for item in batch])
    antigen = torch.stack([item['antigen'] for item in batch])
    delta_g = torch.stack([item['delta_g'] for item in batch])
    return heavy, light, antigen, delta_g


# === 加载 TSV 数据并生成嵌入 ===
def load_tsv_and_embed(path, embedder):
    print(f"   📖 读取 TSV 文件：{path}")
    df = pd.read_csv(path, sep='\t')
    
    required_cols = ['antibody_seq_a', 'antibody_seq_b', 'antigen_seq', 'delta_g']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"缺少必要列：{col}")
    
    seq_h = df['antibody_seq_a'].tolist()
    seq_l = df['antibody_seq_b'].tolist()
    seq_a = df['antigen_seq'].tolist()
    y = df['delta_g'].values.astype(np.float32)
    
    print(f"   样本数：{len(df)}")
    print(f"   原始 delta_g 范围：[{y.min():.4f}, {y.max():.4f}]")
    print(f"   🔄 生成 ESM2 嵌入...")
    
    X_h = embedder.embed_batch(seq_h, target_dim=532)
    X_l = embedder.embed_batch(seq_l, target_dim=532)
    X_a = embedder.embed_batch(seq_a, target_dim=500)
    
    print(f"   嵌入形状：H={X_h.shape}, L={X_l.shape}, A={X_a.shape}")
    
    return X_h, X_l, X_a, y


# === 绘制回归图 ===
def plot_regression(y_true, y_pred, output_file, dataset_name="Dataset"):
    fig, ax = plt.subplots(figsize=(10, 8))
    
    ax.scatter(y_true, y_pred, alpha=0.6, s=50, color='steelblue', 
               edgecolors='black', linewidth=0.5, label='Predictions')
    
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    margin = (max_val - min_val) * 0.1
    min_val -= margin
    max_val += margin
    
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, 
            label='Ideal fit (y=x)')
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(y_true, y_pred)
    regression_line = slope * np.array([min_val, max_val]) + intercept
    ax.plot([min_val, max_val], regression_line, 'g-', linewidth=2, 
            label=f'Linear fit (R²={r_value**2:.4f})')
    
    pcc = np.corrcoef(y_true, y_pred)[0, 1] if len(y_true) > 1 else 0
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    textstr = (f'Dataset: {dataset_name}\n'
               f'N = {len(y_true)}\n'
               f'PCC = {pcc:.4f}\n'
               f'R² = {r2:.4f}\n'
               f'RMSE = {rmse:.4f}\n'
               f'MAE = {mae:.4f}')
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=props)
    
    ax.set_xlabel('True ΔG (kcal/mol)', fontsize=14)
    ax.set_ylabel('Predicted ΔG (kcal/mol)', fontsize=14)
    ax.set_title(f'Antibody-Antigen Binding Affinity Prediction\n{dataset_name}', 
                 fontsize=16, fontweight='bold')
    ax.legend(loc='lower right', fontsize=12)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   📊 回归图已保存：{output_file}")
    
    return {'PCC': pcc, 'R2': r2, 'RMSE': rmse, 'MAE': mae}


# === 主程序 ===
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  使用设备：{device}")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(PLOT_DIR, exist_ok=True)
    
    # 1. 加载模型
    print(f"\n📦 加载模型：{MODEL_PATH}")
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    
    model = AntibodyAffinityModel()
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    label_scaler = checkpoint.get('label_scaler', None)
    if label_scaler is not None:
        print(f"✅ 模型加载完成")
        print(f"   标准化器均值：{label_scaler.mean_[0]:.4f}")
        print(f"   标准化器标准差：{np.sqrt(label_scaler.var_[0]):.4f}")
    else:
        print(f"⚠️ 未找到标签标准化器")
    
    # 2. 初始化 ESM2
    print("\n" + "="*60)
    embedder = ESM2Embedder()
    print("="*60)
    
    # 3. 检查输入文件
    print("\n🔍 检查输入文件...")
    available_files = {}
    for name, path in INPUT_FILES.items():
        if os.path.exists(path):
            available_files[name] = path
            print(f"   ✅ {name}: {path}")
        else:
            print(f"   ❌ {name}: 文件不存在 ({path})")
    
    if not available_files:
        print("\n❌ 错误：未找到任何 TSV 文件！")
        return
    
    # 4. 预测每个数据集
    results_summary = {}
    all_y_true = []
    all_y_pred = []
    
    for name, path in available_files.items():
        print(f"\n{'='*60}")
        print(f"🔮 预测 {name} 数据集")
        print(f"{'='*60}")
        
        try:
            # 加载数据并生成嵌入
            X_h, X_l, X_a, y_true = load_tsv_and_embed(path, embedder)
            
            if len(X_h) == 0:
                print(f"   ❌ 数据为空，跳过")
                continue
            
            # 构建 DataLoader
            samples = list(zip(X_h, X_l, X_a, y_true))
            dataset = PredictDataset(samples)
            loader = DataLoader(dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
            
            # 预测
            print(f"   🔮 开始预测...")
            preds_scaled = []
            with torch.no_grad():
                for heavy, light, antigen, delta_g in loader:
                    heavy, light, antigen = heavy.to(device), light.to(device), antigen.to(device)
                    pred = model(heavy, light, antigen)
                    preds_scaled.extend(pred.cpu().numpy())
            
            preds_scaled = np.array(preds_scaled).flatten()
            
            # 👈 关键修复：仅预测值需要反归一化，真实值直接使用 TSV 中的原始值
            if label_scaler is not None:
                y_pred = label_scaler.inverse_transform(preds_scaled.reshape(-1, 1)).flatten()
                print(f"   预测值标准化范围：[{preds_scaled.min():.4f}, {preds_scaled.max():.4f}]")
                print(f"   预测值反归一化范围：[{y_pred.min():.4f}, {y_pred.max():.4f}]")
            else:
                y_pred = preds_scaled
            
            # 真实值保持不变（TSV 中的原始 delta_g）
            print(f"   真实值范围 (TSV 原始值)：[{y_true.min():.4f}, {y_true.max():.4f}]")
            
            # 保存预测结果
            output_file = f"{OUTPUT_DIR}/IMMSAB_{name.lower()}_predicted_affinity.tsv"
            df = pd.DataFrame({
                'Index': range(1, len(y_pred) + 1),
                'True_delta_g': y_true,  # 👈 直接使用 TSV 中的原始值
                'Predicted_delta_g': y_pred
            })
            df.to_csv(output_file, sep='\t', index=False)
            print(f"   💾 结果保存至：{output_file}")
            
            # 绘制回归图
            plot_file = f"{PLOT_DIR}/{name.lower()}_regression.png"
            metrics = plot_regression(y_true, y_pred, plot_file, dataset_name=name)
            
            # 保存汇总数据
            all_y_true.extend(y_true)
            all_y_pred.extend(y_pred)
            
            results_summary[name] = {
                'N': len(y_pred), 
                'PCC': metrics['PCC'], 
                'R2': metrics['R2'], 
                'RMSE': metrics['RMSE'], 
                'MAE': metrics['MAE']
            }
            
            print(f"\n📊 {name} 结果:")
            print(f"   样本数：{len(y_pred)}")
            print(f"   PCC: {metrics['PCC']:.4f}")
            print(f"   R²: {metrics['R2']:.4f}")
            print(f"   RMSE: {metrics['RMSE']:.4f}")
            print(f"   MAE: {metrics['MAE']:.4f}")
            
        except Exception as e:
            print(f"   ❌ 处理失败：{e}")
            import traceback
            traceback.print_exc()
    
    # 5. 绘制总体回归图
    if len(all_y_true) > 0:
        overall_plot_file = f"{PLOT_DIR}/overall_regression.png"
        plot_regression(np.array(all_y_true), np.array(all_y_pred), 
                       overall_plot_file, dataset_name="All Datasets Combined")
    
    # 6. 打印汇总
    print("\n" + "="*70)
    print("✅ 预测完成！")
    print("="*70)
    
    if results_summary:
        print("\n📊 结果汇总:")
        print(f"{'Dataset':<12} | {'N':>6} | {'PCC':>8} | {'R²':>8} | {'RMSE':>8} | {'MAE':>8}")
        print("-"*70)
        for name, metrics in results_summary.items():
            print(f"{name:<12} | {metrics['N']:>6} | {metrics['PCC']:>8.4f} | {metrics['R2']:>8.4f} | {metrics['RMSE']:>8.4f} | {metrics['MAE']:>8.4f}")
        print("-"*70)
        
        overall_pcc = np.corrcoef(all_y_true, all_y_pred)[0, 1]
        overall_r2 = r2_score(all_y_true, all_y_pred)
        overall_rmse = np.sqrt(mean_squared_error(all_y_true, all_y_pred))
        overall_mae = mean_absolute_error(all_y_true, all_y_pred)
        print(f"{'OVERALL':<12} | {len(all_y_true):>6} | {overall_pcc:>8.4f} | {overall_r2:>8.4f} | {overall_rmse:>8.4f} | {overall_mae:>8.4f}")
    
    print("\n📁 输出文件列表:")
    print("\n预测结果文件:")
    for name in available_files.keys():
        output_file = f"{OUTPUT_DIR}/IMMSAB_{name.lower()}_predicted_affinity.tsv"
        if os.path.exists(output_file):
            size = os.path.getsize(output_file)
            print(f"   ✅ {output_file} ({size:,} bytes)")
    
    print("\n回归图文件:")
    for name in available_files.keys():
        plot_file = f"{PLOT_DIR}/{name.lower()}_regression.png"
        if os.path.exists(plot_file):
            size = os.path.getsize(plot_file)
            print(f"   ✅ {plot_file} ({size:,} bytes)")
    
    if os.path.exists(f"{PLOT_DIR}/overall_regression.png"):
        size = os.path.getsize(f"{PLOT_DIR}/overall_regression.png")
        print(f"   ✅ {PLOT_DIR}/overall_regression.png ({size:,} bytes)")


if __name__ == "__main__":
    main()