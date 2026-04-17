# import os
# import torch
# import torch.nn as nn
# from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler
# import pandas as pd
# import numpy as np
# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# from scipy.stats import pearsonr
# from pathlib import Path

# # -----------------------------
# # 配置（关键修改：BATCH_SIZE = 32）
# # -----------------------------
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# BATCH_SIZE = 32  # ✅ 从 8 增大到 32（根据 GPU 显存可尝试 64）
# EPOCHS = 100
# LR = 1e-4

# train_tsvs = {
#     "/tmp/AbAgCDR/data/final_dataset_train.tsv",
#     "/tmp/AbAgCDR/data/pairs_seq_skempi_clean.tsv",
#     "/tmp/AbAgCDR/data/pairs_seq_sabdab_clean.tsv",
#     "/tmp/AbAgCDR/data/pairs_seq_abbind2_clean.tsv",
# }
# # benchmark_tsv = "/tmp/AbAgCDR/data_annotated/pairs_seq_benchmark1.tsv"

# SEED = 42
# torch.manual_seed(SEED)
# np.random.seed(SEED)

# # -----------------------------
# # 氨基酸词表
# # -----------------------------
# AMINO_ACID_VOCAB = {aa: i + 1 for i, aa in enumerate("ACDEFGHIKLMNPQRSTVWY")}
# AMINO_ACID_VOCAB['X'] = 0
# AMINO_ACID_VOCAB['<pad>'] = 0

# # -----------------------------
# # Dataset (纯序列 + at_type)
# # -----------------------------
# class AntibodyAntigenDataset(Dataset):
#     def __init__(self, df: pd.DataFrame, max_heavy_len=300, max_light_len=120, max_antigen_len=1024):
#         cols = ['H-FR1','H-CDR1','H-FR2','H-CDR2','H-FR3','H-CDR3','H-FR4', 'antibody_seq_b', 'antigen_seq', 'delta_g']
#         df_clean = df.copy()

#         seq_cols = ['H-FR1','H-CDR1','H-FR2','H-CDR2','H-FR3','H-CDR3','H-FR4', 'antibody_seq_b', 'antigen_seq']
#         for col in seq_cols:
#             if col in df_clean.columns:
#                 df_clean[col] = df_clean[col].fillna("").astype(str).str.strip()

#         if 'delta_g' in df_clean.columns:
#             df_clean['delta_g'] = pd.to_numeric(df_clean['delta_g'], errors='coerce')

#         self.df = df_clean.dropna(subset=cols).reset_index(drop=True)
#         self.max_h = max_heavy_len
#         self.max_l = max_light_len
#         self.max_ag = max_antigen_len
#         print(f"✅ Kept {len(self.df)} / {len(df)} samples with complete CDR + antigen + label.")

#     def _build_ab(self, row):
#         # 构建重链序列和类型
#         seq = row['H-FR1'] + row['H-CDR1'] + row['H-FR2'] + row['H-CDR2'] + row['H-FR3'] + row['H-CDR3'] + row['H-FR4']
#         types = (
#             [1]*len(row['H-FR1']) +
#             [3]*len(row['H-CDR1']) +
#             [1]*len(row['H-FR2']) +
#             [4]*len(row['H-CDR2']) +
#             [1]*len(row['H-FR3']) +
#             [5]*len(row['H-CDR3']) +
#             [1]*len(row['H-FR4'])
#         )
#         return seq, types

#     def _build_light(self, seq_b):
#         # 将整个轻链序列视为一个“超长 CDR”
#         v_region = seq_b[:110]  # 轻链可变区，假设前110个氨基酸为有效区域
#         seq = v_region
#         types = [6] * len(v_region)  # 新类型ID（避免与重链冲突）
#         return seq, types

#     def _pad(self, x, max_len, pad_val=0):
#         return (x + [pad_val] * max_len)[:max_len]

#     def __len__(self):
#         return len(self.df)

#     def __getitem__(self, idx):
#         row = self.df.iloc[idx]
#         h_seq, h_type = self._build_ab(row)
#         l_seq, l_type = self._build_light(row['antibody_seq_b'])
#         ag_seq = row['antigen_seq']

#         # 合并重链和轻链序列及其类型
#         ab_seq = h_seq + l_seq
#         ab_type = h_type + l_type

#         ab_seq_ids = torch.tensor(self._pad([AMINO_ACID_VOCAB.get(a, 0) for a in ab_seq], self.max_h + self.max_l), dtype=torch.long)
#         ab_type_ids = torch.tensor(self._pad(ab_type, self.max_h + self.max_l, pad_val=0), dtype=torch.long)
#         ag_seq_ids = torch.tensor(self._pad([AMINO_ACID_VOCAB.get(a, 0) for a in ag_seq], self.max_ag), dtype=torch.long)
#         label = torch.tensor(row['delta_g'], dtype=torch.float32)

#         antibody_input = [ab_seq_ids, ab_type_ids]
#         antigen_input = [ag_seq_ids]
#         return antibody_input, antigen_input, label

# # -----------------------------
# # Model (关键修改：移除 Sigmoid)
# # -----------------------------
# class AntiEmbeddings(nn.Module):
#     def __init__(self, vocab_size=22, type_vocab_size=7, hidden_size=1024, eps=1e-12):  # ⚠️ type_vocab_size 改为 7（0~6）
#         super().__init__()
#         self.seq_emb = nn.Embedding(vocab_size, hidden_size)
#         self.type_emb = nn.Embedding(type_vocab_size, hidden_size)
#         self.norm = nn.LayerNorm(hidden_size, eps=eps)
#         self.dropout = nn.Dropout(0.1)

#     def forward(self, seq, type_ids=None):
#         emb = self.seq_emb(seq)
#         if type_ids is not None:
#             emb = emb + self.type_emb(type_ids)
#         return self.dropout(self.norm(emb))

# class BidirectionalCrossAttention(nn.Module):
#     def __init__(self, embed_dim, num_heads=1, dropout=0.1, res=False):
#         super().__init__()
#         self.ab2ag = nn.MultiheadAttention(embed_dim, num_heads, dropout)
#         self.ag2ab = nn.MultiheadAttention(embed_dim, num_heads, dropout)
#         self.res = res
#         if res:
#             self.norm_ab = nn.LayerNorm(embed_dim)
#             self.norm_ag = nn.LayerNorm(embed_dim)

#     def forward(self, ab, ag):
#         ab_q = ab.permute(1, 0, 2)
#         ag_kv = ag.permute(1, 0, 2)
#         ab_out, _ = self.ab2ag(ab_q, ag_kv, ag_kv)
#         ab_out = ab_out.permute(1, 0, 2)

#         ag_q = ag.permute(1, 0, 2)
#         ab_kv = ab.permute(1, 0, 2)
#         ag_out, _ = self.ag2ab(ag_q, ab_kv, ab_kv)
#         ag_out = ag_out.permute(1, 0, 2)

#         if self.res:
#             ab_out = self.norm_ab(ab_out + ab)
#             ag_out = self.norm_ag(ag_out + ag)
#         return ab_out, ag_out

# class DynamicPool(nn.Module):
#     def __init__(self, latent_dim):
#         super().__init__()
#         self.latent_dim = latent_dim
#         self.proj = None

#     def forward(self, x):
#         b = x.size(0)
#         x_flat = x.view(b, -1)
#         if self.proj is None:
#             self.proj = nn.Sequential(
#                 nn.Linear(x_flat.size(1), self.latent_dim * self.latent_dim),
#                 nn.ReLU()
#             ).to(x.device)
#             self.add_module("proj", self.proj)
#         out = self.proj(x_flat)
#         return out.view(b, self.latent_dim, self.latent_dim)

# class AntiBinder(nn.Module):
#     def __init__(self, hidden_dim=1024, latent_dim=32, res=False):
#         super().__init__()
#         self.embed = AntiEmbeddings(hidden_size=hidden_dim)
#         self.cross_attn = BidirectionalCrossAttention(hidden_dim, res=res)
#         self.dim_reduce = nn.Sequential(nn.Linear(hidden_dim, latent_dim), nn.ReLU())
#         self.pool_ab = DynamicPool(latent_dim)
#         self.pool_ag = DynamicPool(latent_dim)
#         self.alpha = nn.Parameter(torch.tensor(1.0))
#         self.head = None

#     def forward(self, antibody, antigen):
#         ab_emb = self.embed(antibody[0], antibody[1])
#         ag_emb = self.embed(antigen[0])
#         ab_emb, ag_emb = self.cross_attn(ab_emb, ag_emb)
#         ab_emb = self.dim_reduce(ab_emb)
#         ag_emb = self.dim_reduce(ag_emb)
#         ab_pooled = self.pool_ab(ab_emb)
#         ag_pooled = self.pool_ag(ag_emb)
#         concat = torch.cat([ab_pooled.flatten(1), self.alpha * ag_pooled.flatten(1)], dim=1)
#         if self.head is None:
#             self.head = nn.Sequential(
#                 nn.Linear(concat.size(1), 1024),
#                 nn.ReLU(),
#                 nn.Linear(1024, 1)
#                 # ✅ NO SIGMOID
#             ).to(concat.device)
#         return self.head(concat)

# # -----------------------------
# # 加载并合并所有训练数据
# # -----------------------------
# all_dfs = []
# source_ranges = {}
# start_idx = 0

# for tsv_path, weight in train_tsvs.items():
#     name = Path(tsv_path).stem
#     print(f"Loading {tsv_path} as '{name}'...")
#     df = pd.read_csv(tsv_path, sep='\t', engine='python', on_bad_lines='skip')
#     end_idx = start_idx + len(df)
#     source_ranges[name] = (start_idx, end_idx)
#     df['source'] = name
#     all_dfs.append(df)
#     start_idx = end_idx

# full_df = pd.concat(all_dfs, ignore_index=True)
# print(f"Total raw samples loaded: {len(full_df)}")

# # -----------------------------
# # ✅ 直接使用原始 delta_g（不再标准化）
# # -----------------------------
# full_df['delta_g'] = pd.to_numeric(full_df['delta_g'], errors='coerce')
# full_df = full_df.dropna(subset=['delta_g']).reset_index(drop=True)

# # -----------------------------
# # 创建 Dataset ✅ 修正此处！
# # -----------------------------
# dataset = AntibodyAntigenDataset(
#     full_df,
#     max_heavy_len=300,
#     max_light_len=120,
#     max_antigen_len=1024
# )

# if len(dataset) == 0:
#     raise RuntimeError("❌ No valid samples after filtering!")

# # -----------------------------
# # 划分数据集
# # -----------------------------
# total_len = len(dataset)
# if total_len < 3:
#     raise ValueError(f"Dataset too small ({total_len} samples)!")

# train_len = int(total_len * 0.7)
# val_len = int(total_len * 0.15)
# test_len = total_len - train_len - val_len

# if train_len == 0: train_len = 1
# if val_len == 0: val_len = 1
# test_len = total_len - train_len - val_len
# if test_len <= 0:
#     test_len = 1
#     val_len = total_len - train_len - test_len

# generator = torch.Generator().manual_seed(SEED)
# train_ds, val_ds, test_ds = random_split(dataset, [train_len, val_len, test_len], generator=generator)

# def get_sources_from_indices(indices):
#     sources = []
#     for idx in indices:
#         for name, (start, end) in source_ranges.items():
#             if start <= idx < end:
#                 sources.append(name)
#                 break
#         else:
#             sources.append("unknown")
#     return sources

# test_sources = get_sources_from_indices(test_ds.indices)

# # -----------------------------
# # 加权采样
# # -----------------------------
# train_weights = []
# for idx in train_ds.indices:
#     for name, (start, end) in source_ranges.items():
#         if start <= idx < end:
#             weight = train_tsvs[f"/tmp/AbAgCDR/data_annotated/{name}.tsv"]
#             train_weights.append(weight)
#             break
#     else:
#         train_weights.append(1.0)

# sampler = WeightedRandomSampler(
#     weights=train_weights,
#     num_samples=len(train_weights),
#     replacement=True,
#     generator=generator
# )

# train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler)
# val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
# test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

# # -----------------------------
# # 模型、优化器（关键修改：HuberLoss）
# # -----------------------------
# model = AntiBinder().to(DEVICE)
# criterion = nn.HuberLoss(delta=1.0)  # ✅ 替代 MSELoss
# optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# # -----------------------------
# # Evaluate 函数（不变）
# # -----------------------------
# def evaluate(loader):
#     model.eval()
#     all_preds, all_labels = [], []
#     with torch.no_grad():
#         for ab, ag, label in loader:
#             ab = [t.to(DEVICE) for t in ab]
#             ag = [t.to(DEVICE) for t in ag]
#             label = label.to(DEVICE)
#             out = model(ab, ag).squeeze(1)
#             all_preds.append(out.cpu())
#             all_labels.append(label.cpu())

#     y_pred = torch.cat(all_preds).numpy().flatten()
#     y_true = torch.cat(all_labels).numpy().flatten()

#     mse = mean_squared_error(y_true, y_pred)
#     rmse = np.sqrt(mse)
#     mae = mean_absolute_error(y_true, y_pred)
#     r2 = r2_score(y_true, y_pred)
#     pcc = pearsonr(y_true, y_pred)[0] if len(set(y_true)) > 1 else 0.0

#     return mse, rmse, mae, r2, pcc, y_pred, y_true

# # -----------------------------
# # 训练循环（不变）
# # -----------------------------
# best_val_loss = float('inf')
# for epoch in range(1, EPOCHS + 1):
#     model.train()
#     total_loss = 0
#     for ab, ag, label in train_loader:
#         ab = [t.to(DEVICE) for t in ab]
#         ag = [t.to(DEVICE) for t in ag]
#         label = label.to(DEVICE)
#         optimizer.zero_grad()
#         out = model(ab, ag).squeeze(1)
#         loss = criterion(out, label)
#         loss.backward()
#         optimizer.step()
#         total_loss += loss.item() * label.size(0)
#     train_loss = total_loss / len(train_loader.dataset)

#     val_mse, val_rmse, val_mae, val_r2, val_pcc, _, _ = evaluate(val_loader)
#     train_mse, train_rmse, train_mae, train_r2, train_pcc, _, _ = evaluate(train_loader)

#     print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Val MSE={val_mse:.4f}, RMSE={val_rmse:.4f}, "
#           f"MAE={val_mae:.4f}, R²={val_r2:.4f}, Val PCC={val_pcc:.4f}")

#     if val_mse < best_val_loss:
#         best_val_loss = val_mse
#         torch.save(model.state_dict(), "best_modelAntibert.pth")

# # -----------------------------
# # 测试：合并 test
# # -----------------------------
# test_mse, test_rmse, test_mae, test_r2, test_pcc, y_pred_test, y_true_test = evaluate(test_loader)
# print("\n🏆 Merged Test Metrics (original scale):")
# print(f"MSE={test_mse:.4f}, RMSE={test_rmse:.4f}, MAE={test_mae:.4f}, R²={test_r2:.4f}, PCC={test_pcc:.4f}")

# # -----------------------------
# # 按来源评估 test
# # -----------------------------
# test_labels_orig = [dataset[i][2].item() for i in test_ds.indices]
# y_true_orig = np.array(test_labels_orig)
# y_pred_orig = y_pred_test
# test_sources_arr = np.array(test_sources)

# print("\n🔍 Per-Dataset Test Results:")
# for src in np.unique(test_sources_arr):
#     mask = (test_sources_arr == src)
#     if mask.sum() == 0:
#         continue
#     y_t = y_true_orig[mask]
#     y_p = y_pred_orig[mask]
#     mse = mean_squared_error(y_t, y_p)
#     rmse = np.sqrt(mse)
#     mae = mean_absolute_error(y_t, y_p)
#     r2 = r2_score(y_t, y_p)
#     pcc = pearsonr(y_t, y_p)[0] if len(set(y_t)) > 1 else 0.0
#     print(f"\n{src.upper()} TEST → R²: {r2:.4f}, MSE: {mse:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}, PCC: {pcc:.4f}")

# # -----------------------------
# # Benchmark ✅ 修正此处！
# # -----------------------------
# print("\n🧪 Loading benchmark dataset...")
# if os.path.exists(benchmark_tsv):
#     bench_df = pd.read_csv(benchmark_tsv, sep='\t', engine='python', on_bad_lines='skip')
#     if 'delta_g' in bench_df.columns:
#         bench_df['delta_g'] = pd.to_numeric(bench_df['delta_g'], errors='coerce')
#         bench_df = bench_df.dropna(subset=['delta_g']).reset_index(drop=True)
#         if len(bench_df) > 0:
#             bench_dataset = AntibodyAntigenDataset(
#                 bench_df,
#                 max_heavy_len=300,
#                 max_light_len=120,
#                 max_antigen_len=1024
#             )
#             if len(bench_dataset) > 0:
#                 bench_loader = DataLoader(bench_dataset, batch_size=BATCH_SIZE, shuffle=False)
#                 bench_mse, bench_rmse, bench_mae, bench_r2, bench_pcc, _, _ = evaluate(bench_loader)
#                 print(f"\n🎯 BENCHMARK TEST → R²: {bench_r2:.4f}, MSE: {bench_mse:.4f}, "
#                       f"RMSE: {bench_rmse:.4f}, MAE: {bench_mae:.4f}, PCC: {bench_pcc:.4f}")
#             else:
#                 print("⚠️ Benchmark dataset has no valid samples after filtering.")
#         else:
#             print("⚠️ Benchmark dataset has no valid delta_g values.")
#     else:
#         print("⚠️ Benchmark TSV missing 'delta_g' column.")
# else:
#     print("⚠️ Benchmark TSV not found:", benchmark_tsv)

# import os
# import torch
# import torch.nn as nn
# from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler
# import pandas as pd
# import numpy as np
# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# from scipy.stats import pearsonr
# from pathlib import Path

# # -----------------------------
# # 配置
# # -----------------------------
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# BATCH_SIZE = 32  
# EPOCHS = 100
# LR = 1e-4
# SEED = 42

# torch.manual_seed(SEED)
# np.random.seed(SEED)

# # 定义训练数据源及其相对权重 (用于加权采样)
# # 格式: {文件路径: 权重}
# # 权重越高，该数据源在训练中被采样的概率越大
# TRAIN_DATA_CONFIG = {
#     "/tmp/AbAgCDR/data/final_dataset_train.tsv": 1.0,
#     "/tmp/AbAgCDR/data/pairs_seq_skempi_clean.tsv": 1.0,
#     "/tmp/AbAgCDR/data/pairs_seq_sabdab_clean.tsv": 1.0,
#     "/tmp/AbAgCDR/data/pairs_seq_abbind2_clean.tsv": 1.0,
# }

# # -----------------------------
# # 氨基酸词表
# # -----------------------------
# AMINO_ACID_VOCAB = {aa: i + 1 for i, aa in enumerate("ACDEFGHIKLMNPQRSTVWY")}
# AMINO_ACID_VOCAB['X'] = 0
# AMINO_ACID_VOCAB['<pad>'] = 0

# # -----------------------------
# # Dataset (纯序列 + at_type)
# # -----------------------------
# class AntibodyAntigenDataset(Dataset):
#     def __init__(self, df: pd.DataFrame, max_heavy_len=300, max_light_len=120, max_antigen_len=1024):
#         # 确保包含所有必要的列
#         required_cols = ['H-FR1','H-CDR1','H-FR2','H-CDR2','H-FR3','H-CDR3','H-FR4', 'antibody_seq_b', 'antigen_seq', 'delta_g']
        
#         df_clean = df.copy()

#         # 清理序列列
#         seq_cols = ['H-FR1','H-CDR1','H-FR2','H-CDR2','H-FR3','H-CDR3','H-FR4', 'antibody_seq_b', 'antigen_seq']
#         for col in seq_cols:
#             if col in df_clean.columns:
#                 df_clean[col] = df_clean[col].fillna("").astype(str).str.strip()

#         # 清理标签列
#         if 'delta_g' in df_clean.columns:
#             df_clean['delta_g'] = pd.to_numeric(df_clean['delta_g'], errors='coerce')

#         # 丢弃任何必要列为空的行
#         self.df = df_clean.dropna(subset=required_cols).reset_index(drop=True)
        
#         self.max_h = max_heavy_len
#         self.max_l = max_light_len
#         self.max_ag = max_antigen_len
        
#         print(f"✅ Kept {len(self.df)} / {len(df)} samples with complete CDR + antigen + label.")

#     def _build_ab(self, row):
#         # 构建重链序列和类型
#         # 类型定义: 1=FR, 3=CDR1, 4=CDR2, 5=CDR3
#         h_parts = ['H-FR1', 'H-CDR1', 'H-FR2', 'H-CDR2', 'H-FR3', 'H-CDR3', 'H-FR4']
#         h_types_map = {
#             'H-FR1': 1, 'H-CDR1': 3, 'H-FR2': 1, 
#             'H-CDR2': 4, 'H-FR3': 1, 'H-CDR3': 5, 'H-FR4': 1
#         }
        
#         seq = ""
#         types = []
#         for part in h_parts:
#             s = str(row[part])
#             seq += s
#             types.extend([h_types_map[part]] * len(s))
            
#         return seq, types

#     def _build_light(self, seq_b):
#         # 将整个轻链序列视为一个“超长 CDR”
#         # 假设前110个氨基酸为有效可变区，可根据实际情况调整
#         v_region = str(seq_b)[:110]  
#         seq = v_region
#         types = [6] * len(v_region)  # 新类型ID 6 (Light Chain)
#         return seq, types

#     def _pad(self, x, max_len, pad_val=0):
#         if len(x) >= max_len:
#             return x[:max_len]
#         return x + [pad_val] * (max_len - len(x))

#     def __len__(self):
#         return len(self.df)

#     def __getitem__(self, idx):
#         row = self.df.iloc[idx]
#         h_seq, h_type = self._build_ab(row)
#         l_seq, l_type = self._build_light(row['antibody_seq_b'])
#         ag_seq = str(row['antigen_seq'])

#         # 合并重链和轻链
#         ab_seq = h_seq + l_seq
#         ab_type = h_type + l_type

#         # 转换为 ID 并 Padding
#         ab_seq_ids = torch.tensor(self._pad([AMINO_ACID_VOCAB.get(a, 0) for a in ab_seq], self.max_h + self.max_l), dtype=torch.long)
#         ab_type_ids = torch.tensor(self._pad(ab_type, self.max_h + self.max_l, pad_val=0), dtype=torch.long)
#         ag_seq_ids = torch.tensor(self._pad([AMINO_ACID_VOCAB.get(a, 0) for a in ag_seq], self.max_ag), dtype=torch.long)
        
#         label = torch.tensor(row['delta_g'], dtype=torch.float32)

#         antibody_input = [ab_seq_ids, ab_type_ids]
#         antigen_input = [ag_seq_ids]
        
#         return antibody_input, antigen_input, label

# # -----------------------------
# # Model Definition
# # -----------------------------
# class AntiEmbeddings(nn.Module):
#     def __init__(self, vocab_size=22, type_vocab_size=7, hidden_size=1024, eps=1e-12):
#         super().__init__()
#         self.seq_emb = nn.Embedding(vocab_size, hidden_size)
#         self.type_emb = nn.Embedding(type_vocab_size, hidden_size)
#         self.norm = nn.LayerNorm(hidden_size, eps=eps)
#         self.dropout = nn.Dropout(0.1)

#     def forward(self, seq, type_ids=None):
#         emb = self.seq_emb(seq)
#         if type_ids is not None:
#             emb = emb + self.type_emb(type_ids)
#         return self.dropout(self.norm(emb))

# class BidirectionalCrossAttention(nn.Module):
#     def __init__(self, embed_dim, num_heads=1, dropout=0.1, res=False):
#         super().__init__()
#         self.ab2ag = nn.MultiheadAttention(embed_dim, num_heads, dropout, batch_first=True) # 优化：使用 batch_first
#         self.ag2ab = nn.MultiheadAttention(embed_dim, num_heads, dropout, batch_first=True)
#         self.res = res
#         if res:
#             self.norm_ab = nn.LayerNorm(embed_dim)
#             self.norm_ag = nn.LayerNorm(embed_dim)

#     def forward(self, ab, ag):
#         # ab, ag shape: [B, L, D]
#         # MultiheadAttention with batch_first=True expects [B, L, D]
        
#         ab_out, _ = self.ab2ag(ab, ag, ag)
#         ag_out, _ = self.ag2ab(ag, ab, ab)

#         if self.res:
#             ab_out = self.norm_ab(ab_out + ab)
#             ag_out = self.norm_ag(ag_out + ag)
#         return ab_out, ag_out

# class DynamicPool(nn.Module):
#     def __init__(self, latent_dim):
#         super().__init__()
#         self.latent_dim = latent_dim
#         self.proj = None

#     def forward(self, x):
#         b, l, d = x.size()
#         x_flat = x.reshape(b, -1) # Flatten [B, L, D] -> [B, L*D]
        
#         if self.proj is None:
#             # 初始化投影层
#             input_dim = x_flat.size(1)
#             self.proj = nn.Sequential(
#                 nn.Linear(input_dim, self.latent_dim * self.latent_dim),
#                 nn.ReLU()
#             ).to(x.device)
#             self.add_module("proj_dynamic", self.proj)
            
#         out = self.proj(x_flat)
#         return out.view(b, self.latent_dim, self.latent_dim)

# class AntiBinder(nn.Module):
#     def __init__(self, hidden_dim=1024, latent_dim=32, res=False):
#         super().__init__()
#         self.embed = AntiEmbeddings(hidden_size=hidden_dim)
#         # 增加头数可能有助于捕捉更多特征，这里保持原样或可微调
#         self.cross_attn = BidirectionalCrossAttention(hidden_dim, num_heads=4, res=res) 
#         self.dim_reduce = nn.Sequential(nn.Linear(hidden_dim, latent_dim), nn.ReLU())
#         self.pool_ab = DynamicPool(latent_dim)
#         self.pool_ag = DynamicPool(latent_dim)
#         self.alpha = nn.Parameter(torch.tensor(1.0))
#         self.head = None

#     def forward(self, antibody, antigen):
#         # antibody: [seq_ids, type_ids], antigen: [seq_ids]
#         ab_emb = self.embed(antibody[0], antibody[1])
#         ag_emb = self.embed(antigen[0])
        
#         ab_emb, ag_emb = self.cross_attn(ab_emb, ag_emb)
        
#         ab_emb = self.dim_reduce(ab_emb)
#         ag_emb = self.dim_reduce(ag_emb)
        
#         ab_pooled = self.pool_ab(ab_emb)
#         ag_pooled = self.pool_ag(ag_emb)
        
#         concat = torch.cat([ab_pooled.flatten(1), self.alpha * ag_pooled.flatten(1)], dim=1)
        
#         if self.head is None:
#             input_dim = concat.size(1)
#             self.head = nn.Sequential(
#                 nn.Linear(input_dim, 1024),
#                 nn.ReLU(),
#                 nn.Dropout(0.1),
#                 nn.Linear(1024, 1)
#             ).to(concat.device)
            
#         return self.head(concat)

# # -----------------------------
# # 加载并合并所有训练数据
# # -----------------------------
# all_dfs = []
# source_ranges = {}
# source_weights_map = {} # 映射索引范围到权重
# start_idx = 0

# print("Loading training datasets...")
# for tsv_path, weight in TRAIN_DATA_CONFIG.items():
#     if not os.path.exists(tsv_path):
#         print(f"⚠️ Warning: File not found, skipping: {tsv_path}")
#         continue
        
#     name = Path(tsv_path).stem
#     print(f"Loading {tsv_path} as '{name}' (Weight: {weight})...")
    
#     try:
#         df = pd.read_csv(tsv_path, sep='\t', engine='python', on_bad_lines='skip')
#         end_idx = start_idx + len(df)
        
#         source_ranges[name] = (start_idx, end_idx)
#         source_weights_map[name] = weight
        
#         df['source'] = name
#         all_dfs.append(df)
#         start_idx = end_idx
#     except Exception as e:
#         print(f"❌ Error loading {tsv_path}: {e}")

# if not all_dfs:
#     raise RuntimeError("❌ No valid datasets loaded. Check file paths.")

# full_df = pd.concat(all_dfs, ignore_index=True)
# print(f"Total raw samples loaded: {len(full_df)}")

# # 清理标签
# full_df['delta_g'] = pd.to_numeric(full_df['delta_g'], errors='coerce')
# full_df = full_df.dropna(subset=['delta_g']).reset_index(drop=True)

# # 创建 Dataset
# dataset = AntibodyAntigenDataset(
#     full_df,
#     max_heavy_len=300,
#     max_light_len=120,
#     max_antigen_len=1024
# )

# if len(dataset) == 0:
#     raise RuntimeError("❌ No valid samples after filtering!")

# # -----------------------------
# # 划分数据集
# # -----------------------------
# total_len = len(dataset)
# if total_len < 10:
#     raise ValueError(f"Dataset too small ({total_len} samples)!")

# train_ratio = 0.7
# val_ratio = 0.15
# test_ratio = 0.15

# train_len = int(total_len * train_ratio)
# val_len = int(total_len * val_ratio)
# test_len = total_len - train_len - val_len

# # 确保每个集合至少有一个样本
# if train_len == 0: train_len = 1
# if val_len == 0: val_len = 1
# if test_len <= 0: test_len = 1

# # 重新计算 val_len 以适应总和
# val_len = total_len - train_len - test_len

# generator = torch.Generator().manual_seed(SEED)
# train_ds, val_ds, test_ds = random_split(dataset, [train_len, val_len, test_len], generator=generator)

# # 获取测试集来源分布
# def get_sources_from_indices(indices):
#     sources = []
#     for idx in indices:
#         found = False
#         for name, (start, end) in source_ranges.items():
#             if start <= idx < end:
#                 sources.append(name)
#                 found = True
#                 break
#         if not found:
#             sources.append("unknown")
#     return sources

# test_sources = get_sources_from_indices(test_ds.indices)

# # -----------------------------
# # 加权采样器 (Weighted Random Sampler)
# # -----------------------------
# print("Setting up Weighted Random Sampler...")
# train_weights = []
# for idx in train_ds.indices:
#     assigned_weight = 1.0
#     for name, (start, end) in source_ranges.items():
#         if start <= idx < end:
#             assigned_weight = source_weights_map.get(name, 1.0)
#             break
#     train_weights.append(assigned_weight)

# sampler = WeightedRandomSampler(
#     weights=train_weights,
#     num_samples=len(train_weights),
#     replacement=True,
#     generator=generator
# )

# train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler)
# val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
# test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

# # -----------------------------
# # 模型、优化器、损失函数
# # -----------------------------
# model = AntiBinder(hidden_dim=1024, latent_dim=32, res=True).to(DEVICE)
# criterion = nn.HuberLoss(delta=1.0) 
# optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-6)

# # -----------------------------
# # Evaluate 函数
# # -----------------------------
# def evaluate(loader):
#     model.eval()
#     all_preds, all_labels = [], []
#     with torch.no_grad():
#         for ab, ag, label in loader:
#             ab = [t.to(DEVICE) for t in ab]
#             ag = [t.to(DEVICE) for t in ag]
#             label = label.to(DEVICE)
            
#             out = model(ab, ag).squeeze(1)
#             all_preds.append(out.cpu())
#             all_labels.append(label.cpu())

#     if not all_preds:
#         return 0, 0, 0, 0, 0, [], []

#     y_pred = torch.cat(all_preds).numpy().flatten()
#     y_true = torch.cat(all_labels).numpy().flatten()

#     mse = mean_squared_error(y_true, y_pred)
#     rmse = np.sqrt(mse)
#     mae = mean_absolute_error(y_true, y_pred)
#     r2 = r2_score(y_true, y_pred)
    
#     # 处理 PCC 计算中的常数标签情况
#     if len(set(y_true)) > 1 and len(set(y_pred)) > 1:
#         pcc = pearsonr(y_true, y_pred)[0]
#     else:
#         pcc = 0.0

#     return mse, rmse, mae, r2, pcc, y_pred, y_true

# # -----------------------------
# # 训练循环
# # -----------------------------
# print(f"Starting training on {DEVICE}...")
# best_val_loss = float('inf')
# patience_counter = 0
# max_patience = 15

# for epoch in range(1, EPOCHS + 1):
#     model.train()
#     total_loss = 0
#     count = 0
    
#     for ab, ag, label in train_loader:
#         ab = [t.to(DEVICE) for t in ab]
#         ag = [t.to(DEVICE) for t in ag]
#         label = label.to(DEVICE)
        
#         optimizer.zero_grad()
#         out = model(ab, ag).squeeze(1)
#         loss = criterion(out, label)
#         loss.backward()
        
#         # 梯度裁剪防止爆炸
#         torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
#         optimizer.step()
        
#         total_loss += loss.item() * label.size(0)
#         count += label.size(0)
        
#     avg_train_loss = total_loss / count if count > 0 else 0

#     # 验证
#     val_mse, val_rmse, val_mae, val_r2, val_pcc, _, _ = evaluate(val_loader)
    
#     # 学习率调度
#     scheduler.step(val_mse)
#     current_lr = optimizer.param_groups[0]['lr']

#     print(f"Epoch {epoch:03d}: Loss={avg_train_loss:.4f}, Val MSE={val_mse:.4f}, R²={val_r2:.4f}, PCC={val_pcc:.4f}, LR={current_lr:.2e}")

#     # Early Stopping & Save Best
#     if val_mse < best_val_loss:
#         best_val_loss = val_mse
#         torch.save(model.state_dict(), "best_modelAntibert.pth")
#         patience_counter = 0
#         # print(f"  -> New best model saved!")
#     else:
#         patience_counter += 1
#         if patience_counter >= max_patience:
#             print(f"Early stopping triggered at epoch {epoch}.")
#             break

# # -----------------------------
# # 最终测试评估
# # -----------------------------
# print("\n" + "="*50)
# print("FINAL EVALUATION")
# print("="*50)

# # 加载最佳模型
# model.load_state_dict(torch.load("best_modelAntibert.pth"))
# print("Loaded best model weights.")

# # 1. 整体测试集
# test_mse, test_rmse, test_mae, test_r2, test_pcc, y_pred_test, y_true_test = evaluate(test_loader)
# print(f"\n🏆 Merged Test Metrics:")
# print(f"MSE={test_mse:.4f}, RMSE={test_rmse:.4f}, MAE={test_mae:.4f}, R²={test_r2:.4f}, PCC={test_pcc:.4f}")

# # 2. 按数据源细分测试集
# test_labels_orig = [dataset[i][2].item() for i in test_ds.indices]
# y_true_orig = np.array(test_labels_orig)
# y_pred_orig = y_pred_test
# test_sources_arr = np.array(test_sources)

# print("\n🔍 Per-Dataset Test Results:")
# unique_sources = np.unique(test_sources_arr)
# if len(unique_sources) == 0:
#     print("No source information available.")
# else:
#     for src in unique_sources:
#         mask = (test_sources_arr == src)
#         if mask.sum() == 0:
#             continue
#         y_t = y_true_orig[mask]
#         y_p = y_pred_orig[mask]
        
#         mse = mean_squared_error(y_t, y_p)
#         rmse = np.sqrt(mse)
#         mae = mean_absolute_error(y_t, y_p)
#         r2 = r2_score(y_t, y_p)
#         pcc = pearsonr(y_t, y_p)[0] if len(set(y_t)) > 1 else 0.0
        
#         print(f"\n{src.upper()} (n={mask.sum()}) → R²: {r2:.4f}, MSE: {mse:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}, PCC: {pcc:.4f}")

# print("\n✅ Training and Evaluation Complete.")


import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import warnings
import json
from collections import defaultdict

warnings.filterwarnings('ignore')

# -----------------------------
# 0. 配置
# -----------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32  
EPOCHS = 100
LR = 1e-4
SEED = 42
MAX_PATIENCE = 15

torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

TRAIN_DATA_CONFIG = {
    "/tmp/AbAgCDR/data/final_dataset_train.tsv": 1.0,
    "/tmp/AbAgCDR/data/pairs_seq_skempi_clean.tsv": 1.0,
    "/tmp/AbAgCDR/data/pairs_seq_sabdab_clean.tsv": 1.0,
    "/tmp/AbAgCDR/data/pairs_seq_abbind2_clean.tsv": 1.0,
}

# -----------------------------
# 1. 氨基酸词表
# -----------------------------
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
AMINO_ACID_VOCAB = {aa: i + 1 for i, aa in enumerate(AMINO_ACIDS)}
AMINO_ACID_VOCAB['X'] = 0
AMINO_ACID_VOCAB['<pad>'] = 0
AMINO_ACID_VOCAB['-'] = 0
AMINO_ACID_VOCAB['U'] = 0
AMINO_ACID_VOCAB['O'] = 0

VOCAB_SIZE_ACTUAL = max(AMINO_ACID_VOCAB.values()) + 1
print(f"🧬 词表大小确定为：{VOCAB_SIZE_ACTUAL}")

def clean_sequence(seq):
    if not isinstance(seq, str):
        return ""
    seq = seq.strip().upper()
    return ''.join([aa for aa in seq if aa in AMINO_ACID_VOCAB])

# -----------------------------
# 2. Dataset
# -----------------------------
class AntibodyAntigenDataset(Dataset):
    def __init__(self, df: pd.DataFrame, max_heavy_len=300, max_light_len=120, max_antigen_len=1024):
        self.max_h = max_heavy_len
        self.max_l = max_light_len
        self.max_ag = max_antigen_len
        self.max_ab = max_heavy_len + max_light_len
        
        df_clean = df.copy()
        heavy_candidates = ['heavy', 'h_seq', 'antibody_seq_a', 'vh_seq', 'heavy_chain', 'h-chain', 'vh']
        light_candidates = ['light', 'l_seq', 'antibody_seq_b', 'vl_seq', 'light_chain', 'l-chain', 'vl']
        antigen_candidates = ['antigen', 'antigen_seq', 'ag_seq', 'target_seq', 'protein_seq']
        label_candidates = ['delta_g', 'dg', 'affinity', 'label', 'ant_binding', 'kd', 'ka']

        def find_col(candidates, dataframe):
            cols_lower = [str(c).lower().strip() for c in dataframe.columns]
            for cand in candidates:
                if cand.lower() in cols_lower:
                    return dataframe.columns[cols_lower.index(cand.lower())]
            return None

        col_h = find_col(heavy_candidates, df_clean)
        col_l = find_col(light_candidates, df_clean)
        col_ag = find_col(antigen_candidates, df_clean)
        col_label = find_col(label_candidates, df_clean)

        if not all([col_h, col_l, col_ag, col_label]):
            raise ValueError("缺少必要列")

        df_clean = df_clean.rename(columns={col_h: 'heavy', col_l: 'light', col_ag: 'antigen', col_label: 'delta_g'})

        for col in ['heavy', 'light', 'antigen']:
            df_clean[col] = df_clean[col].apply(clean_sequence)
        
        df_clean['delta_g'] = pd.to_numeric(df_clean['delta_g'], errors='coerce')
        self.df = df_clean.dropna(subset=['heavy', 'light', 'antigen', 'delta_g']).reset_index(drop=True)
        
        self.df = self.df[
            (self.df['heavy'].str.len() > 0) & 
            (self.df['light'].str.len() > 0) & 
            (self.df['antigen'].str.len() > 0)
        ].reset_index(drop=True)

        print(f"✅ 数据加载完成：{len(self.df)} 样本")

    def _pad(self, x, max_len, pad_val=0):
        if len(x) >= max_len:
            return x[:max_len]
        return x + [pad_val] * (max_len - len(x))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        h_seq, l_seq, ag_seq = row['heavy'], row['light'], row['antigen']
        
        h_ids = [AMINO_ACID_VOCAB.get(a, 0) for a in h_seq]
        l_ids = [AMINO_ACID_VOCAB.get(a, 0) for a in l_seq]
        ag_ids = [AMINO_ACID_VOCAB.get(a, 0) for a in ag_seq]
        
        if not h_ids: h_ids = [0]
        if not l_ids: l_ids = [0]
        if not ag_ids: ag_ids = [0]
        
        limit = VOCAB_SIZE_ACTUAL - 1
        h_ids = [x if 0 <= x <= limit else 0 for x in h_ids]
        l_ids = [x if 0 <= x <= limit else 0 for x in l_ids]
        ag_ids = [x if 0 <= x <= limit else 0 for x in ag_ids]

        h_types = [1] * len(h_ids)
        l_types = [2] * len(l_ids)
        
        ab_ids = h_ids + l_ids
        ab_types = h_types + l_types
        
        ab_seq_ids = torch.tensor(self._pad(ab_ids, self.max_ab), dtype=torch.long)
        ab_type_ids = torch.tensor(self._pad(ab_types, self.max_ab, pad_val=0), dtype=torch.long)
        ag_seq_ids = torch.tensor(self._pad(ag_ids, self.max_ag), dtype=torch.long)
        
        if ab_seq_ids.max() >= VOCAB_SIZE_ACTUAL or ab_seq_ids.min() < 0:
            raise ValueError(f"样本 {idx} 包含非法索引！Max: {ab_seq_ids.max()}, Vocab: {VOCAB_SIZE_ACTUAL}")
        if ag_seq_ids.max() >= VOCAB_SIZE_ACTUAL or ag_seq_ids.min() < 0:
            raise ValueError(f"样本 {idx} 包含非法索引！Max: {ag_seq_ids.max()}, Vocab: {VOCAB_SIZE_ACTUAL}")

        label = torch.tensor(row['delta_g'], dtype=torch.float32)
        
        return [ab_seq_ids, ab_type_ids], [ag_seq_ids], label, idx

# -----------------------------
# 3. 模型定义
# -----------------------------
class AntiEmbeddings(nn.Module):
    def __init__(self, vocab_size, type_vocab_size, hidden_size, eps=1e-12):
        super().__init__()
        self.seq_emb = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        self.type_emb = nn.Embedding(type_vocab_size, hidden_size, padding_idx=0)
        self.norm = nn.LayerNorm(hidden_size, eps=eps)
        self.dropout = nn.Dropout(0.1)

    def forward(self, seq, type_ids=None):
        emb = self.seq_emb(seq)
        if type_ids is not None:
            emb = emb + self.type_emb(type_ids)
        return self.dropout(self.norm(emb))

class BidirectionalCrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=4, dropout=0.1, res=True):
        super().__init__()
        self.ab2ag = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.ag2ab = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.res = res
        if res:
            self.norm_ab = nn.LayerNorm(embed_dim)
            self.norm_ag = nn.LayerNorm(embed_dim)

    def forward(self, ab, ag):
        ab_out, _ = self.ab2ag(ab, ag, ag)
        ag_out, _ = self.ag2ab(ag, ab, ab)
        if self.res:
            ab_out = self.norm_ab(ab_out + ab)
            ag_out = self.norm_ag(ag_out + ag)
        return ab_out, ag_out

class StablePool(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, latent_dim),
            nn.ReLU(),
            nn.LayerNorm(latent_dim)
        )
        self.global_pool = lambda x: torch.mean(x, dim=1)

    def forward(self, x):
        projected = self.proj(x)
        pooled = self.global_pool(projected)
        return pooled

class AntiBinder(nn.Module):
    def __init__(self, hidden_dim=1024, latent_dim=32, res=True, vocab_size=22):
        super().__init__()
        self.embed = AntiEmbeddings(vocab_size=vocab_size, type_vocab_size=7, hidden_size=hidden_dim)
        self.cross_attn = BidirectionalCrossAttention(hidden_dim, num_heads=4, res=res) 
        self.dim_reduce = nn.Sequential(nn.Linear(hidden_dim, latent_dim), nn.ReLU(), nn.LayerNorm(latent_dim))
        self.pool_ab = StablePool(latent_dim, latent_dim)
        self.pool_ag = StablePool(latent_dim, latent_dim)
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.head = nn.Sequential(
            nn.Linear(latent_dim * 2, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 1)
        )

    def forward(self, antibody, antigen):
        ab_emb = self.embed(antibody[0], antibody[1])
        ag_emb = self.embed(antigen[0])
        ab_emb, ag_emb = self.cross_attn(ab_emb, ag_emb)
        ab_emb = self.dim_reduce(ab_emb)
        ag_emb = self.dim_reduce(ag_emb)
        ab_pooled = self.pool_ab(ab_emb)
        ag_pooled = self.pool_ag(ag_emb)
        concat = torch.cat([ab_pooled, self.alpha * ag_pooled], dim=1)
        return self.head(concat)

# -----------------------------
# 4. 主程序
# -----------------------------
def main():
    all_dfs = []
    source_ranges = {}
    source_weights_map = {}
    start_idx = 0

    print("📂 正在加载数据...")
    for tsv_path, weight in TRAIN_DATA_CONFIG.items():
        if not os.path.exists(tsv_path):
            print(f"⚠️ 跳过：{tsv_path}")
            continue
        name = Path(tsv_path).stem
        try:
            df = pd.read_csv(tsv_path, sep='\t', engine='python', on_bad_lines='skip')
            if df.empty: continue
            end_idx = start_idx + len(df)
            source_ranges[name] = (start_idx, end_idx)
            source_weights_map[name] = weight
            df['source'] = name
            all_dfs.append(df)
            start_idx = end_idx
        except Exception as e:
            print(f"❌ 读取失败 {tsv_path}: {e}")

    if not all_dfs:
        raise RuntimeError("❌ 未加载到任何数据。")

    full_df = pd.concat(all_dfs, ignore_index=True)
    dataset = AntibodyAntigenDataset(full_df)
    total_len = len(dataset)

    print(f"\n📐 开始分层划分 (6:2:2)...")
    idx_to_source = {}
    for name, (start, end) in source_ranges.items():
        for i in range(start, end):
            if i < total_len: idx_to_source[i] = name

    source_indices = defaultdict(list)
    for idx in range(total_len):
        source_indices[idx_to_source.get(idx, "unknown")].append(idx)

    train_indices_all, val_indices_all, test_indices_all = [], [], []
    test_indices_by_source = {}

    for src, indices in source_indices.items():
        n = len(indices)
        if n == 0: continue
        np.random.shuffle(indices)
        n_train, n_val = int(n * 0.6), int(n * 0.2)
        t, v, te = indices[:n_train], indices[n_train:n_train+n_val], indices[n_train+n_val:]
        train_indices_all.extend(t)
        val_indices_all.extend(v)
        test_indices_all.extend(te)
        test_indices_by_source[src] = te
        print(f"   [{src}]: Train={len(t)}, Val={len(v)}, Test={len(te)}")

    train_labels_raw = np.array([dataset[idx][2].item() for idx in train_indices_all])
    scaler = StandardScaler()
    scaler.fit(train_labels_raw.reshape(-1, 1))
    print(f"📈 Scaler: Mean={scaler.mean_[0]:.4f}, Std={scaler.scale_[0]:.4f}")

    def get_label_map(indices):
        raw = np.array([dataset[i][2].item() for i in indices])
        scaled = scaler.transform(raw.reshape(-1, 1)).flatten()
        return {idx: val for idx, val in zip(indices, scaled)}

    train_map = get_label_map(train_indices_all)
    val_map = get_label_map(val_indices_all)
    test_map = get_label_map(test_indices_all)

    class ScaledSubsetDataset(Dataset):
        def __init__(self, indices, label_map, orig_ds):
            self.indices = indices
            self.label_map = label_map
            self.orig_ds = orig_ds
        def __len__(self): return len(self.indices)
        def __getitem__(self, i):
            gid = self.indices[i]
            ab, ag, _, _ = self.orig_ds[gid]
            return ab, ag, torch.tensor(self.label_map[gid], dtype=torch.float32)

    train_ds = ScaledSubsetDataset(train_indices_all, train_map, dataset)
    val_ds = ScaledSubsetDataset(val_indices_all, val_map, dataset)
    test_ds = ScaledSubsetDataset(test_indices_all, test_map, dataset)

    train_weights = []
    for gid in train_indices_all:
        w = 1.0
        for name, (s, e) in source_ranges.items():
            if s <= gid < e: w = source_weights_map.get(name, 1.0); break
        train_weights.append(w)

    generator = torch.Generator().manual_seed(SEED)
    sampler = WeightedRandomSampler(weights=train_weights, num_samples=len(train_weights), replacement=True, generator=generator)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = AntiBinder(hidden_dim=1024, latent_dim=32, res=True, vocab_size=VOCAB_SIZE_ACTUAL).to(DEVICE)
    criterion = nn.HuberLoss(delta=1.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-6)

    def evaluate(loader, inverse=False):
        model.eval()
        preds, labels = [], []
        with torch.no_grad():
            for ab, ag, lbl in loader:
                ab = [t.to(DEVICE) for t in ab]
                ag = [t.to(DEVICE) for t in ag]
                lbl = lbl.to(DEVICE)
                out = model(ab, ag).squeeze(1)
                preds.append(out.cpu())
                labels.append(lbl.cpu())
        y_pred = torch.cat(preds).numpy().flatten()
        y_true = torch.cat(labels).numpy().flatten()
        if inverse:
            y_pred = scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
            y_true = scaler.inverse_transform(y_true.reshape(-1, 1)).flatten()
            return y_pred, y_true
        else:
            mse_val = mean_squared_error(y_true, y_pred)
            # 【关键修复】强制转换为 float
            return float(mse_val), None

    print(f"\n🚀 开始训练 (Device: {DEVICE}, Workers: 0)...")
    best_loss = float('inf')
    patience = 0
    save_path = "best_model_stable.pth"
    
    for epoch in range(1, EPOCHS + 1):
        model.train()
        tot_loss, cnt = 0, 0
        for ab, ag, lbl in train_loader:
            ab = [t.to(DEVICE) for t in ab]
            ag = [t.to(DEVICE) for t in ag]
            lbl = lbl.to(DEVICE)
            
            optimizer.zero_grad()
            loss = criterion(model(ab, ag).squeeze(1), lbl)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            tot_loss += loss.item() * lbl.size(0)
            cnt += lbl.size(0)
        
        avg_loss = tot_loss / cnt
        
        # 获取验证集 MSE (现在是 float 了)
        val_mse, _ = evaluate(val_loader, inverse=False)
        
        scheduler.step(val_mse)
        
        log = ""
        if epoch % 5 == 0 or epoch == 1:
            yp, yt = evaluate(val_loader, inverse=True)
            log = f" | Real R²={r2_score(yt, yp):.3f}"
        
        if val_mse < best_loss:
            best_loss = val_mse
            torch.save({'model_state_dict': model.state_dict(), 'scaler': {'mean': scaler.mean_[0], 'scale': scaler.scale_[0]}}, save_path)
            patience = 0
            status = "✨ Best"
        else:
            patience += 1
            status = ""
            if patience >= MAX_PATIENCE:
                print(f"Early Stopping at Epoch {epoch}")
                break
        
        print(f"Epoch {epoch:03d} | Loss: {avg_loss:.4f} | Val MSE: {val_mse:.4f}{log} {status}")

    # 评估部分
    print("\n" + "="*60)
    print("🏆 FINAL EVALUATION (By Source)")
    print("="*60)
    
    ckpt = torch.load(save_path)
    model.load_state_dict(ckpt['model_state_dict'])
    
    final_results = {"summary": {}, "by_source": {}}
    
    yp, yt = evaluate(test_loader, inverse=True)
    overall = {
        "RMSE": float(np.sqrt(mean_squared_error(yt, yp))),
        "MAE": float(mean_absolute_error(yt, yp)),
        "R2": float(r2_score(yt, yp)),
        "PCC": float(pearsonr(yt, yp)[0]) if len(set(yt))>1 else 0.0,
        "Count": len(yt)
    }
    final_results["summary"]["Overall"] = overall
    print(f"\n[Overall] N={overall['Count']} | RMSE: {overall['RMSE']:.4f} | R²: {overall['R2']:.4f}")

    print(f"\n{'Source':<30} | {'N':<5} | {'RMSE':<8} | {'MAE':<8} | {'R²':<8} | {'PCC':<8}")
    print("-" * 75)
    
    for src, indices in test_indices_by_source.items():
        if not indices: continue
        src_map = get_label_map(indices)
        src_ds = ScaledSubsetDataset(indices, src_map, dataset)
        src_loader = DataLoader(src_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
        
        yp, yt = evaluate(src_loader, inverse=True)
        if len(yt) == 0: continue
        
        m = {
            "RMSE": float(np.sqrt(mean_squared_error(yt, yp))),
            "MAE": float(mean_absolute_error(yt, yp)),
            "R2": float(r2_score(yt, yp)),
            "PCC": float(pearsonr(yt, yp)[0]) if len(set(yt))>1 else 0.0,
            "Count": len(yt)
        }
        final_results["by_source"][src] = m
        print(f"{src:<30} | {m['Count']:<5} | {m['RMSE']:<8.4f} | {m['MAE']:<8.4f} | {m['R2']:<8.4f} | {m['PCC']:<8.4f}")

    with open("test_results_final.json", "w") as f:
        json.dump(final_results, f, indent=4)
    print("\n💾 结果已保存至 test_results_final.json")
    print("✅ 完成。")

if __name__ == "__main__":
    main()