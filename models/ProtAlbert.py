# import os
# import warnings
# import pickle
# warnings.filterwarnings('ignore')

# import numpy as np
# import pandas as pd
# import torch
# import torch.nn as nn
# from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
# from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
# from scipy.stats import pearsonr
# from sklearn.model_selection import train_test_split
# import copy
# import itertools
# from datetime import datetime
# from pathlib import Path

# # ==============================
# # 本地加载 ProtAlbert（无网络）
# # ==============================
# from transformers import AlbertModel, AlbertTokenizer

# def setup_protalbert(model_dir="models/ProtAlbert"):
#     model_dir = Path(model_dir)
#     required_files = ["pytorch_model.bin", "config.json", "spm_model.model"]
#     for f in required_files:
#         if not (model_dir / f).exists():
#             raise FileNotFoundError(
#                 f"Required file '{f}' not found in {model_dir}. "
#                 "Please manually place all three files in this directory."
#             )
#     print(f"🧠 Loading ProtAlbert from: {model_dir}")
#     tokenizer = AlbertTokenizer(
#         vocab_file=str(model_dir / "spm_model.model"),
#         do_lower_case=False,
#         keep_accents=True
#     )
#     model = AlbertModel.from_pretrained(str(model_dir))
#     return model, tokenizer

# # ==============================
# # 模型定义（回归任务）
# # ==============================
# class TriProtDTA(nn.Module):
#     def __init__(self, emb_dim):
#         super().__init__()
#         self.emb_dim = emb_dim
#         # 分别处理每条链（可选）
#         self.proj_l = nn.Linear(emb_dim, 512)
#         self.proj_h = nn.Linear(emb_dim, 512)
#         self.proj_ag = nn.Linear(emb_dim, 512)
        
#         # 显式建模两两交互
#         self.interaction = nn.Sequential(
#             nn.Linear(512 * 3 + 512 * 3, 512),  # 原始 + 两两点积/拼接
#             nn.ReLU(),
#             nn.Dropout(0.3)
#         )
        
#         self.head = nn.Sequential(
#             nn.Linear(512, 256),
#             nn.ReLU(),
#             nn.Dropout(0.5),
#             nn.Linear(256, 1)
#         )

#     def forward(self, l_emb, h_emb, ag_emb):
#         l = self.proj_l(l_emb)
#         h = self.proj_h(h_emb)
#         ag = self.proj_ag(ag_emb)
        
#         # 简单交互：拼接 + 两两点积（模拟内积相似度）
#         pairwise = torch.cat([
#             l * h,      # L-H interaction
#             l * ag,     # L-Ag
#             h * ag      # H-Ag
#         ], dim=1)
        
#         x = torch.cat([l, h, ag, pairwise], dim=1)
#         x = self.interaction(x)
#         return self.head(x)

# # ==============================
# # 数据集：使用预计算 embedding
# # ==============================
# class CachedEmbeddingDataset(Dataset):
#     def __init__(self, samples, emb_cache):
#         self.samples = samples
#         self.emb_cache = emb_cache

#     def __len__(self):
#         return len(self.samples)

#     def __getitem__(self, idx):
#         l_seq, h_seq, ag_seq, label = self.samples[idx]
#         l_emb = torch.tensor(self.emb_cache[l_seq], dtype=torch.float32)
#         h_emb = torch.tensor(self.emb_cache[h_seq], dtype=torch.float32)
#         ag_emb = torch.tensor(self.emb_cache[ag_seq], dtype=torch.float32)
#         label = torch.tensor([label], dtype=torch.float32)
#         return {
#             'l_emb': l_emb,
#             'h_emb': h_emb,
#             'ag_emb': ag_emb,
#             'label': label
#         }

# def collate_fn(batch):
#     return {
#         'l_emb': torch.stack([item['l_emb'] for item in batch]),
#         'h_emb': torch.stack([item['h_emb'] for item in batch]),
#         'ag_emb': torch.stack([item['ag_emb'] for item in batch]),
#         'label': torch.stack([item['label'] for item in batch])
#     }

# # ==============================
# # 工具函数
# # ==============================
# def evaluate(model, dataloader, device, y_mean, y_std, is_normalized_label=True):
#     model.eval()
#     all_preds, all_labels = [], []
#     with torch.no_grad():
#         for batch in dataloader:
#             l = batch['l_emb'].to(device, non_blocking=True)
#             h = batch['h_emb'].to(device, non_blocking=True)
#             ag = batch['ag_emb'].to(device, non_blocking=True)
#             labels = batch['label'].to(device, non_blocking=True).cpu().numpy().flatten()
#             preds = model(l, h, ag).cpu().numpy().flatten()

#             # 反归一化预测值
#             preds = preds * y_std + y_mean

#             # 如果标签是归一化的，也要反归一化；否则保持原始值
#             if is_normalized_label:
#                 labels = labels * y_std + y_mean

#             all_preds.append(preds)
#             all_labels.append(labels)

#     all_preds = np.concatenate(all_preds)
#     all_labels = np.concatenate(all_labels)

#     if len(all_labels) == 0:
#         return {'MSE': np.nan, 'RMSE': np.nan, 'MAE': np.nan, 'R2': np.nan, 'PCC': 0.0}

#     # --- 手动计算 Pearson 相关系数（更稳定） ---
#     x = np.array(all_labels)
#     y = np.array(all_preds)
#     if x.size == 0 or y.size == 0:
#         pcc = 0.0
#     else:
#         x_mean = np.mean(x)
#         y_mean = np.mean(y)
#         xm = x - x_mean
#         ym = y - y_mean
#         norm_x = np.linalg.norm(xm)
#         norm_y = np.linalg.norm(ym)
#         if norm_x == 0 or norm_y == 0:
#             pcc = 0.0  # 无变异 → 无相关性
#         else:
#             pcc = np.dot(xm, ym) / (norm_x * norm_y)
#             pcc = np.clip(pcc, -1.0, 1.0)  # 防止浮点误差

#     # --- 计算其他指标 ---
#     try:
#         r2 = r2_score(all_labels, all_preds)
#     except:
#         r2 = -np.inf

#     mse = mean_squared_error(all_labels, all_preds)
#     rmse = np.sqrt(mse)
#     mae = mean_absolute_error(all_labels, all_preds)

#     # 确保 PCC 是有效 float
#     pcc = float(pcc) if np.isfinite(pcc) else 0.0
#     r2 = float(r2) if np.isfinite(r2) else -float('inf')

#     return {'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'R2': r2, 'PCC': pcc}


# def split_samples(samples, test_size=0.2, val_size=0.2, random_state=42):
#     train_val, test = train_test_split(samples, test_size=test_size, random_state=random_state)
#     train, val = train_test_split(train_val, test_size=val_size/(1-test_size), random_state=random_state)
#     return train, val, test

# def clean_seq(seq):
#     if pd.isna(seq) or seq is None:
#         return "A"
#     valid_aas = set("ARNDCQEGHILKMFPSTWYV")
#     cleaned = ''.join([aa for aa in str(seq).upper().strip() if aa in valid_aas])
#     return cleaned if cleaned else "A"

# def load_and_encode_tsv(tsv_path):
#     df = pd.read_csv(tsv_path, sep='\t', on_bad_lines='skip')
#     required_cols = ['antibody_seq_a', 'antibody_seq_b', 'antigen_seq', 'delta_g']
#     for col in required_cols:
#         if col not in df.columns:
#             raise ValueError(f"Missing column '{col}' in {tsv_path}")
#     df = df[(df['delta_g'] >= -20) & (df['delta_g'] <= 5)]
#     samples = []
#     for _, row in df.iterrows():
#         a = clean_seq(row['antibody_seq_a'])
#         b = clean_seq(row['antibody_seq_b'])
#         ag = clean_seq(row['antigen_seq'])
#         dg = float(row['delta_g'])
#         if a and b and ag and not pd.isna(dg):
#             samples.append((a, b, ag, dg))
#     return samples

# class TrainerWithScheduler:
#     def __init__(self, model, train_loader, val_loader, params, device, y_mean, y_std):
#         self.model = model.to(device)
#         self.train_loader = train_loader
#         self.val_loader = val_loader
#         self.params = params
#         self.device = device
#         self.y_mean = y_mean
#         self.y_std = y_std
#         self.optimizer = torch.optim.Adam(
#             model.parameters(),
#             lr=params['lr'],
#             weight_decay=params['weight_decay']
#         )
#         self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
#             self.optimizer,
#             mode='max',
#             factor=params['lr_factor'],
#             patience=params['scheduler_patience'],
#             min_lr=params['min_lr']
#         )
#         self.criterion = nn.MSELoss()
#         self.patience = params['patience']

#     def train(self):
#         best_score = -np.inf
#         patience_counter = 0
#         for epoch in range(self.params['epochs']):
#             self.model.train()
#             for batch in self.train_loader:
#                 l = batch['l_emb'].to(self.device, non_blocking=True)
#                 h = batch['h_emb'].to(self.device, non_blocking=True)
#                 ag = batch['ag_emb'].to(self.device, non_blocking=True)
#                 labels = batch['label'].to(self.device, non_blocking=True)
#                 self.optimizer.zero_grad()
#                 preds = self.model(l, h, ag)
#                 loss = self.criterion(preds, labels)
#                 loss.backward()
#                 self.optimizer.step()

#             val_metrics = evaluate(self.model, self.val_loader, self.device, self.y_mean, self.y_std, is_normalized_label=True)
#             current_pcc = val_metrics['PCC'] if not np.isnan(val_metrics['PCC']) else -1.0
#             self.scheduler.step(current_pcc)

#             if current_pcc > best_score:
#                 best_score = current_pcc
#                 patience_counter = 0
#                 best_model = copy.deepcopy(self.model.state_dict())
#             else:
#                 patience_counter += 1
#                 if patience_counter >= self.patience:
#                     break

#         self.model.load_state_dict(best_model)
#         return self.model

# # ==============================
# # 格式化输出函数
# # ==============================
# def format_metric(val, fmt="{:.4f}"):
#     if np.isnan(val) or not np.isfinite(val):
#         return "N/A"
#     return fmt.format(val)

# # ==============================
# # Main Function
# # ==============================
# def main():
#     train_tsvs = {
#         "/tmp/AbAgCDR/data/final_dataset_train.tsv": 1.0,
#         "/tmp/AbAgCDR/data/pairs_seq_skempi_clean.tsv": 1.0,
#         "/tmp/AbAgCDR/data/pairs_seq_sabdab_clean.tsv": 1.0,
#         "/tmp/AbAgCDR/data/pairs_seq_abbind2_clean.tsv": 1.0,
#     }
#     # benchmark_tsv = "/tmp/AbAgCDR/data/pairs_seq_benchmark1.tsv"
#     CONFIG = {
#         "test_size": 0.2,
#         "val_size": 0.2,
#         "seed": 42,
#         "run_dir": "../runs"
#     }

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"Using device: {device}")

#     prot_albert_model, _ = setup_protalbert("models/ProtAlbert")
#     emb_dim = prot_albert_model.config.hidden_size
#     print(f"✅ ProtAlbert embedding dimension: {emb_dim}")

#     cache_path = "/tmp/AbAgCDR/data/protalbert_embeddings.pkl"
#     with open(cache_path, 'rb') as f:
#         emb_cache = pickle.load(f)
#     print(f"✅ Loaded {len(emb_cache)} embeddings from cache")

#     # Load raw samples (labels are original ΔG)
#     all_train_samples_raw = []
#     all_val_samples_raw = []
#     all_test_samples_raw = []
#     sample_weights = []
#     dataset_test_splits_raw = {}

#     for tsv_path, weight in train_tsvs.items():
#         print(f"Loading {tsv_path}...")
#         samples = load_and_encode_tsv(tsv_path)
#         train, val, test = split_samples(
#             samples,
#             test_size=CONFIG["test_size"],
#             val_size=CONFIG["val_size"],
#             random_state=CONFIG["seed"]
#         )
#         all_train_samples_raw.extend(train)
#         all_val_samples_raw.extend(val)
#         all_test_samples_raw.extend(test)
#         sample_weights.extend([weight] * len(train))
#         dataset_name = Path(tsv_path).stem
#         dataset_test_splits_raw[dataset_name] = test

#     print(f"Total Train: {len(all_train_samples_raw)}, Val: {len(all_val_samples_raw)}, Test: {len(all_test_samples_raw)}")

#     # Normalize only train/val labels
#     y_train = np.array([s[3] for s in all_train_samples_raw])
#     y_mean, y_std = y_train.mean(), y_train.std() + 1e-8

#     def normalize_label(samples, mean, std):
#         return [(s[0], s[1], s[2], (s[3] - mean) / std) for s in samples]

#     all_train_samples = normalize_label(all_train_samples_raw, y_mean, y_std)
#     all_val_samples = normalize_label(all_val_samples_raw, y_mean, y_std)

#     # Keep test sets in original scale!
#     all_test_samples = all_test_samples_raw
#     dataset_test_splits_norm = {name: samples for name, samples in dataset_test_splits_raw.items()}

#     sampler = WeightedRandomSampler(
#         weights=sample_weights,
#         num_samples=len(sample_weights),
#         replacement=True
#     )

#     param_grid = {
#         'lr': [1e-4],
#         'batch_size': [32],
#         'epochs': [50],
#         'patience': [10],
#         'weight_decay': [1e-5],
#         'scheduler_patience': [3],
#         'lr_factor': [0.5],
#         'min_lr': [1e-6]
#     }

#     keys, values = zip(*param_grid.items())
#     param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

#     best_score = -np.inf
#     best_params = None
#     best_model_state = None
#     save_path = Path(CONFIG["run_dir"]) / f"best_model_{datetime.now().strftime('%Y%m%d_%H%M')}.pth"
#     save_path.parent.mkdir(parents=True, exist_ok=True)

#     for trial_idx, params in enumerate(param_combinations):
#         print(f"\n{'='*50}\nTrial {trial_idx+1}/{len(param_combinations)} | Params: {params}\n{'='*50}")

#         train_loader = DataLoader(
#             CachedEmbeddingDataset(all_train_samples, emb_cache),
#             batch_size=params['batch_size'],
#             sampler=sampler,
#             collate_fn=collate_fn,
#             shuffle=False,
#             num_workers=4,
#             pin_memory=True
#         )
#         val_loader = DataLoader(
#             CachedEmbeddingDataset(all_val_samples, emb_cache),
#             batch_size=params['batch_size'],
#             shuffle=False,
#             collate_fn=collate_fn,
#             num_workers=2,
#             pin_memory=True
#         )

#         model = TriProtDTA(emb_dim=emb_dim)
#         trainer = TrainerWithScheduler(model, train_loader, val_loader, params, device, y_mean, y_std)
#         trained_model = trainer.train()

#         val_metrics = evaluate(trained_model, val_loader, device, y_mean, y_std, is_normalized_label=True)
#         score = val_metrics['PCC'] if not np.isnan(val_metrics['PCC']) else -1.0
#         print(f"✅ Val PCC: {format_metric(val_metrics['PCC'])}")

#         if score > best_score:
#             best_score = score
#             best_params = copy.deepcopy(params)
#             best_model_state = copy.deepcopy(trained_model.state_dict())
#             torch.save({
#                 'model_state_dict': best_model_state,
#                 'params': best_params,
#                 'y_mean': y_mean,
#                 'y_std': y_std,
#             }, save_path)
#             print(f"🎉 Best model saved at {save_path}")

#     # ==============================
#     # Final Evaluation
#     # ==============================
#     final_model = TriProtDTA(emb_dim=emb_dim)
#     final_model.load_state_dict(best_model_state)
#     final_model.to(device)

#     print("\n" + "="*80)
#     print("🔍 FINAL EVALUATION ON ALL TEST SETS")
#     print("="*80)

#     # 1. Per-dataset test sets
#     for name, test_samples in dataset_test_splits_norm.items():
#         if not test_samples:
#             print(f"\n⚠️  {name.upper()}: No test samples. Skipped.")
#             continue
#         test_loader = DataLoader(
#             CachedEmbeddingDataset(test_samples, emb_cache),
#             batch_size=64,
#             shuffle=False,
#             collate_fn=collate_fn,
#             num_workers=2,
#             pin_memory=True
#         )
#         metrics = evaluate(final_model, test_loader, device, y_mean, y_std, is_normalized_label=False)
#         print(f"\n{name.upper():<22} → R²: {format_metric(metrics['R2'])} | RMSE: {format_metric(metrics['RMSE'])} | PCC: {format_metric(metrics['PCC'])} | MAE: {format_metric(metrics['MAE'])}")

#     # 2. Merged internal test
#     if all_test_samples:
#         test_loader_merged = DataLoader(
#             CachedEmbeddingDataset(all_test_samples, emb_cache),
#             batch_size=64,
#             shuffle=False,
#             collate_fn=collate_fn,
#             num_workers=2,
#             pin_memory=True
#         )
#         test_metrics_merged = evaluate(final_model, test_loader_merged, device, y_mean, y_std, is_normalized_label=False)
#         print(f"\n{'MERGED INTERNAL TEST':<22} → R²: {format_metric(test_metrics_merged['R2'])} | MSE: {format_metric(test_metrics_merged['MSE'])} | RMSE: {format_metric(test_metrics_merged['RMSE'])} | PCC: {format_metric(test_metrics_merged['PCC'])} | MAE: {format_metric(test_metrics_merged['MAE'])}")
#     else:
#         print("\n⚠️  No merged test samples.")

#     # # 3. Benchmark
#     # try:
#     #     print("\n🧪 Loading benchmark dataset...")
#     #     bench_samples_raw = load_and_encode_tsv(benchmark_tsv)
#     #     if bench_samples_raw:
#     #         bench_loader = DataLoader(
#     #             CachedEmbeddingDataset(bench_samples_raw, emb_cache),
#     #             batch_size=64,
#     #             shuffle=False,
#     #             collate_fn=collate_fn,
#     #             num_workers=2,
#     #             pin_memory=True
#     #         )
#     #         bench_metrics = evaluate(final_model, bench_loader, device, y_mean, y_std, is_normalized_label=False)
#     #         print(f"\n{'BENCHMARK TEST':<22} → R²: {format_metric(bench_metrics['R2'])} | MSE: {format_metric(bench_metrics['MSE'])} | RMSE: {format_metric(bench_metrics['RMSE'])} | PCC: {format_metric(bench_metrics['PCC'])} | MAE: {format_metric(bench_metrics['MAE'])}")
#     #     else:
#     #         print("⚠️  Benchmark dataset is empty.")
#     # except Exception as e:
#     #     print(f"❌ Failed to evaluate on benchmark: {e}")

#     print("\n" + "="*80)
#     print(f"🎉 Training completed! Best Val PCC: {format_metric(best_score)}")
#     print(f"💾 Model saved at: {save_path}")
#     print("="*80)

# if __name__ == "__main__":
#     main()


# import os
# import warnings
# import sys
# import math
# import copy
# import itertools
# from datetime import datetime
# from pathlib import Path

# warnings.filterwarnings('ignore')

# import numpy as np
# import pandas as pd
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.utils.data import Dataset, DataLoader
# from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
# from sklearn.model_selection import GroupShuffleSplit

# # ==============================
# # 0. 依赖检查与模型导入
# # ==============================
# try:
#     import sentencepiece
#     from transformers import AlbertModel, AlbertTokenizer
# except ImportError:
#     print("❌ 错误：缺少依赖库。请运行：pip install sentencepiece transformers")
#     sys.exit(1)

# # ==============================
# # 1. 全局模型加载 (单例模式 + 严格校验)
# # ==============================
# GLOBAL_ALBERT_MODEL = None
# GLOBAL_ALBERT_TOKENIZER = None
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# def setup_protalbert(model_dir="models/ProtAlbert"):
#     global GLOBAL_ALBERT_MODEL, GLOBAL_ALBERT_TOKENIZER
    
#     if GLOBAL_ALBERT_MODEL is not None:
#         return GLOBAL_ALBERT_MODEL, GLOBAL_ALBERT_TOKENIZER

#     model_dir = Path(model_dir)
    
#     # 严格检查文件存在性
#     config_exists = (model_dir / "config.json").exists()
#     spm_exists = (model_dir / "spiece.model").exists() # ALBERT 必须包含此文件
#     weight_exists = (model_dir / "pytorch_model.bin").exists() or (model_dir / "model.safetensors").exists()

#     if not (config_exists and spm_exists and weight_exists):
#         print(f"❌ 错误：本地模型文件不完整或缺失 'spiece.model'。")
#         print(f"   检测到文件: config={config_exists}, spiece={spm_exists}, weights={weight_exists}")
#         print(f"   这通常意味着你上传了 BERT 模型而不是 ALBERT，或者文件损坏。")
#         print(f"   正在尝试从 HuggingFace 重新下载正确的 'albert-base-v2' ...")
        
#         try:
#             model_name = "albert-base-v2"
#             tokenizer = AlbertTokenizer.from_pretrained(model_name)
#             model = AlbertModel.from_pretrained(model_name)
            
#             os.makedirs(model_dir, exist_ok=True)
#             tokenizer.save_pretrained(model_dir)
#             model.save_pretrained(model_dir)
#             print(f"✅ 正确模型已下载并保存至：{model_dir}")
#         except Exception as e:
#             print(f"❌ 下载失败：{e}")
#             raise RuntimeError("无法加载或下载正确的 ALBERT 模型。请手动检查网络或上传文件。")
#     else:
#         print(f"🧠 检测到本地模型文件，正在验证并加载...")
#         try:
#             # 尝试加载
#             tokenizer = AlbertTokenizer.from_pretrained(str(model_dir))
#             model = AlbertModel.from_pretrained(str(model_dir))
            
#             # 验证维度
#             emb_dim = model.config.hidden_size
#             if emb_dim != 768:
#                 raise ValueError(f"维度不匹配：检测到 {emb_dim}，但我们需要 albert-base-v2 (768)。")
                
#             print("✅ 本地模型验证通过并加载成功。")
#         except Exception as e:
#             print(f"⚠️ 本地模型验证失败：{e}")
#             print("   可能原因：模型文件是 BERT 或其他版本的 ALBERT。正在重新下载...")
#             # 递归调用自己以触发下载逻辑 (简单处理：直接执行下载代码)
#             model_name = "albert-base-v2"
#             tokenizer = AlbertTokenizer.from_pretrained(model_name)
#             model = AlbertModel.from_pretrained(model_name)
#             os.makedirs(model_dir, exist_ok=True)
#             tokenizer.save_pretrained(model_dir)
#             model.save_pretrained(model_dir)
#             print(f"✅ 已重新下载正确的模型至：{model_dir}")

#     model = model.to(DEVICE)
#     model.eval() 
    
#     GLOBAL_ALBERT_MODEL = model
#     GLOBAL_ALBERT_TOKENIZER = tokenizer
    
#     emb_dim = model.config.hidden_size
#     print(f"✅ ALBERT 就绪。设备：{DEVICE}, 维度：{emb_dim} (预期 768)")
    
#     if emb_dim != 768:
#         print(f"❌ 致命错误：模型维度为 {emb_dim}，程序终止。请确保使用 'albert-base-v2'。")
#         sys.exit(1)
        
#     return model, tokenizer

# # ==============================
# # 2. 模型定义 (TriProtDTA+)
# # ==============================
# class TriProtDTAPlus(nn.Module):
#     def __init__(self, emb_dim, hidden_dim=512, dropout=0.3):
#         super().__init__()
#         self.emb_dim = emb_dim
        
#         def create_proj():
#             return nn.Sequential(
#                 nn.Linear(emb_dim, hidden_dim),
#                 nn.LayerNorm(hidden_dim),
#                 nn.GELU(),
#                 nn.Dropout(dropout)
#             )
        
#         self.proj_l = create_proj()
#         self.proj_h = create_proj()
#         self.proj_ag = create_proj()
        
#         input_dim = hidden_dim * 6
        
#         self.interaction = nn.Sequential(
#             nn.Linear(input_dim, hidden_dim),
#             nn.LayerNorm(hidden_dim),
#             nn.GELU(),
#             nn.Dropout(dropout),
#             nn.Linear(hidden_dim, hidden_dim // 2),
#             nn.LayerNorm(hidden_dim // 2),
#             nn.GELU(),
#             nn.Dropout(dropout)
#         )
        
#         self.head = nn.Sequential(
#             nn.Linear(hidden_dim // 2, 64),
#             nn.LayerNorm(64),
#             nn.GELU(),
#             nn.Dropout(dropout * 1.5),
#             nn.Linear(64, 1)
#         )

#     def forward(self, l_emb, h_emb, ag_emb):
#         l = self.proj_l(l_emb)
#         h = self.proj_h(h_emb)
#         ag = self.proj_ag(ag_emb)
        
#         pairwise_lh = l * h
#         pairwise_lag = l * ag
#         pairwise_hag = h * ag
        
#         x = torch.cat([l, h, ag, pairwise_lh, pairwise_lag, pairwise_hag], dim=1)
#         x = self.interaction(x)
#         return self.head(x)

# # ==============================
# # 3. 数据集类 (实时计算 Embedding - 修复版)
# # ==============================
# class LiveEmbeddingDataset(Dataset):
#     def __init__(self, samples, tokenizer, max_len=512):
#         self.samples = samples
#         self.tokenizer = tokenizer
#         self.max_len = max_len

#     def __len__(self):
#         return len(self.samples)

#     def __getitem__(self, idx):
#         l_seq, h_seq, ag_seq, label = self.samples[idx]
#         return {
#             'l_seq': l_seq,
#             'h_seq': h_seq,
#             'ag_seq': ag_seq,
#             'label': label
#         }

# def collate_fn_with_albert(batch, tokenizer, model, device, max_len=512, noise_std=0.0):
#     """
#     自定义 collate_fn:
#     1. Tokenize (CPU)
#     2. Model Forward (GPU)
#     3. 【关键】将结果移回 CPU (Dense Tensor) 以支持 pin_memory
#     """
#     l_seqs = [item['l_seq'] for item in batch]
#     h_seqs = [item['h_seq'] for item in batch]
#     ag_seqs = [item['ag_seq'] for item in batch]
#     labels = torch.tensor([item['label'] for item in batch], dtype=torch.float32).unsqueeze(1)
    
#     def get_batch_emb(seqs):
#         # Tokenize (CPU)
#         inputs = tokenizer(
#             seqs, 
#             return_tensors="pt", 
#             padding=True, 
#             truncation=True, 
#             max_length=max_len
#         )
#         # 移至 GPU 进行计算
#         inputs = inputs.to(device)
        
#         with torch.no_grad():
#             outputs = model(**inputs)
#             embs = outputs.last_hidden_state[:, 0, :] # GPU Tensor
        
#         # 【关键修复】：移回 CPU，变为 dense tensor，以便 DataLoader pin_memory
#         embs = embs.cpu()
        
#         if noise_std > 0.0:
#             noise = torch.randn_like(embs) * noise_std
#             embs = embs + noise
            
#         return embs

#     l_emb = get_batch_emb(l_seqs)
#     h_emb = get_batch_emb(h_seqs)
#     ag_emb = get_batch_emb(ag_seqs)
    
#     return {
#         'l_emb': l_emb,
#         'h_emb': h_emb,
#         'ag_emb': ag_emb,
#         'label': labels
#     }

# # ==============================
# # 4. 评估函数
# # ==============================
# def evaluate(model, dataloader, device, y_mean, y_std):
#     model.eval()
#     all_preds, all_labels = [], []
    
#     with torch.no_grad():
#         for batch in dataloader:
#             # batch 中的 tensor 此时可能在 pinned memory (CPU) 中，需移至 device
#             l = batch['l_emb'].to(device, non_blocking=True)
#             h = batch['h_emb'].to(device, non_blocking=True)
#             ag = batch['ag_emb'].to(device, non_blocking=True)
#             labels = batch['label'].to(device, non_blocking=True).cpu().numpy().flatten()
            
#             preds = model(l, h, ag).cpu().numpy().flatten()

#             preds = preds * y_std + y_mean
#             labels = labels * y_std + y_mean

#             all_preds.append(preds)
#             all_labels.append(labels)

#     if len(all_labels) == 0:
#         return {'MSE': np.nan, 'RMSE': np.nan, 'MAE': np.nan, 'R2': np.nan, 'PCC': 0.0}

#     all_preds = np.concatenate(all_preds)
#     all_labels = np.concatenate(all_labels)

#     if len(all_labels) < 2:
#         pcc = 0.0
#     else:
#         pcc = np.corrcoef(all_labels, all_preds)[0, 1]
#         if np.isnan(pcc): pcc = 0.0

#     try:
#         r2 = r2_score(all_labels, all_preds)
#     except:
#         r2 = -np.inf

#     return {
#         'MSE': mean_squared_error(all_labels, all_preds),
#         'RMSE': np.sqrt(mean_squared_error(all_labels, all_preds)),
#         'MAE': mean_absolute_error(all_labels, all_preds),
#         'R2': r2,
#         'PCC': float(pcc)
#     }

# # ==============================
# # 5. 数据划分与清洗
# # ==============================
# def split_samples_strict(samples, test_size=0.2, val_size=0.2, random_state=42):
#     df = pd.DataFrame(samples, columns=['heavy', 'light', 'antigen', 'label'])
#     df['group_id'] = df['heavy'].astype(str) + "|" + df['light'].astype(str) + "|" + df['antigen'].astype(str)
    
#     indices = np.arange(len(df))
#     groups = df['group_id'].values
    
#     if len(groups) == 0:
#         return [], [], []

#     gss_test = GroupShuffleSplit(test_size=test_size, n_splits=1, random_state=random_state)
#     try:
#         train_val_idx, test_idx = next(gss_test.split(indices, groups=groups))
#     except ValueError:
#         return samples, [], []
    
#     df_train_val = df.iloc[train_val_idx].reset_index(drop=True)
#     df_test = df.iloc[test_idx].reset_index(drop=True)
    
#     if len(df_train_val) < 5:
#         df_train = df_train_val
#         df_val = pd.DataFrame(columns=df_train_val.columns)
#     else:
#         tv_indices = np.arange(len(df_train_val))
#         tv_groups = df_train_val['group_id'].values
#         adjusted_val_size = val_size / (1.0 - test_size)
#         gss_val = GroupShuffleSplit(test_size=adjusted_val_size, n_splits=1, random_state=random_state+1)
#         train_idx, val_idx = next(gss_val.split(tv_indices, groups=tv_groups))
#         df_train = df_train_val.iloc[train_idx]
#         df_val = df_train_val.iloc[val_idx]
    
#     to_list = lambda df_part: list(df_part[['heavy', 'light', 'antigen', 'label']].itertuples(index=False, name=None))
    
#     train_list = to_list(df_train)
#     val_list = to_list(df_val)
#     test_list = to_list(df_test)
    
#     train_groups = set(df_train['group_id']) if not df_train.empty else set()
#     test_groups = set(df_test['group_id']) if not df_test.empty else set()
#     val_groups = set(df_val['group_id']) if not df_val.empty else set()
    
#     if train_groups.intersection(test_groups) or train_groups.intersection(val_groups):
#         raise RuntimeError("❌ 严重错误：检测到数据泄露！")
    
#     print(f"   ✅ 严格划分完成：Train={len(train_list)}, Val={len(val_list)}, Test={len(test_list)}")
#     return train_list, val_list, test_list

# def clean_seq(seq):
#     if pd.isna(seq) or seq is None:
#         return "A"
#     valid_aas = set("ARNDCQEGHILKMFPSTWYV")
#     cleaned = ''.join([aa for aa in str(seq).upper().strip() if aa in valid_aas])
#     return cleaned if cleaned else "A"

# def load_and_encode_tsv(tsv_path):
#     if not os.path.exists(tsv_path):
#         return []
#     try:
#         df = pd.read_csv(tsv_path, sep='\t', on_bad_lines='skip')
#     except Exception as e:
#         return []

#     cols = [str(c).lower() for c in df.columns]
#     mapping = {}
#     light_keys = ['antibody_seq_a', 'light', 'vl', 'l_chain', 'light_chain']
#     heavy_keys = ['antibody_seq_b', 'heavy', 'vh', 'h_chain', 'heavy_chain']
#     antigen_keys = ['antigen_seq', 'antigen', 'ag', 'target_seq']
#     label_keys = ['delta_g', 'dg', 'affinity']
    
#     def find_col(keys, cols_lower, original_cols):
#         for k in keys:
#             if k in cols_lower:
#                 return original_cols[cols_lower.index(k)]
#         return None

#     orig_cols = list(df.columns)
#     mapping['h'] = find_col(heavy_keys, cols, orig_cols)
#     mapping['l'] = find_col(light_keys, cols, orig_cols)
#     mapping['ag'] = find_col(antigen_keys, cols, orig_cols)
#     mapping['dg'] = find_col(label_keys, cols, orig_cols)
    
#     if not all(mapping.values()):
#         return []

#     try:
#         df = df[(df[mapping['dg']] >= -20) & (df[mapping['dg']] <= 5)]
#     except:
#         pass
        
#     samples = []
#     for _, row in df.iterrows():
#         a = clean_seq(row[mapping['l']])
#         b = clean_seq(row[mapping['h']])
#         ag = clean_seq(row[mapping['ag']])
#         try:
#             dg = float(row[mapping['dg']])
#         except:
#             continue
#         if a and b and ag:
#             samples.append((a, b, ag, dg))
            
#     print(f"   📥 加载有效样本：{len(samples)}/{len(df)}")
#     return samples

# # ==============================
# # 6. 高级训练器
# # ==============================
# class AdvancedTrainer:
#     def __init__(self, model, train_loader, val_loader, params, device, y_mean, y_std):
#         self.model = model.to(device)
#         self.train_loader = train_loader
#         self.val_loader = val_loader
#         self.params = params
#         self.device = device
#         self.y_mean = y_mean
#         self.y_std = y_std
        
#         self.optimizer = torch.optim.AdamW(
#             model.parameters(), 
#             lr=params['lr'], 
#             weight_decay=params['weight_decay'],
#             betas=(0.9, 0.98)
#         )
        
#         self.criterion = nn.HuberLoss(delta=1.0)
#         self.patience = params['patience']
#         self.total_steps = len(train_loader) * params['epochs']
#         self.warmup_steps = int(self.total_steps * 0.1)

#     def get_lr(self, step):
#         if step < self.warmup_steps:
#             return self.params['lr'] * (step / max(1, self.warmup_steps))
#         else:
#             progress = (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
#             return self.params['min_lr'] + (self.params['lr'] - self.params['min_lr']) * 0.5 * (1.0 + math.cos(math.pi * progress))

#     def train(self):
#         best_score = -np.inf
#         patience_counter = 0
#         best_model_state = None
#         global_step = 0
        
#         for epoch in range(self.params['epochs']):
#             self.model.train()
#             total_loss = 0
            
#             for batch in self.train_loader:
#                 l = batch['l_emb'].to(self.device, non_blocking=True)
#                 h = batch['h_emb'].to(self.device, non_blocking=True)
#                 ag = batch['ag_emb'].to(self.device, non_blocking=True)
#                 labels = batch['label'].to(self.device, non_blocking=True)
                
#                 self.optimizer.zero_grad()
#                 preds = self.model(l, h, ag)
#                 loss = self.criterion(preds, labels)
#                 loss.backward()
                
#                 torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
#                 self.optimizer.step()
                
#                 global_step += 1
#                 current_lr = self.get_lr(global_step)
#                 for param_group in self.optimizer.param_groups:
#                     param_group['lr'] = current_lr
                    
#                 total_loss += loss.item()

#             val_metrics = evaluate(self.model, self.val_loader, self.device, self.y_mean, self.y_std)
#             current_pcc = val_metrics['PCC'] if not np.isnan(val_metrics['PCC']) else -1.0

#             if current_pcc > best_score:
#                 best_score = current_pcc
#                 patience_counter = 0
#                 best_model_state = copy.deepcopy(self.model.state_dict())
#             else:
#                 patience_counter += 1
#                 if patience_counter >= self.patience:
#                     print(f"   ⏹️ Early Stopping at epoch {epoch+1} (Best PCC: {best_score:.4f})")
#                     break
            
#             if (epoch + 1) % 5 == 0 or epoch == 0:
#                 avg_lr = self.optimizer.param_groups[0]['lr']
#                 print(f"   Epoch {epoch+1:02d} | Loss: {total_loss/len(self.train_loader):.4f} | Val PCC: {current_pcc:.4f} | LR: {avg_lr:.2e}")

#         if best_model_state is not None:
#             self.model.load_state_dict(best_model_state)
#         return self.model

# def format_metric(val, fmt="{:.4f}"):
#     if val is None or not np.isfinite(val):
#         return "N/A"
#     return fmt.format(val)

# # ==============================
# # 7. 主程序
# # ==============================
# def main():
#     train_tsvs = {
#         "/tmp/AbAgCDR/data/final_dataset_train.tsv": 1.0,
#         "/tmp/AbAgCDR/data/pairs_seq_skempi_clean.tsv": 1.0,
#         "/tmp/AbAgCDR/data/pairs_seq_sabdab_clean.tsv": 1.0,
#         "/tmp/AbAgCDR/data/pairs_seq_abbind2_clean.tsv": 1.0,
#     }
    
#     CONFIG = {
#         "test_size": 0.2,
#         "val_size": 0.2,
#         "seed": 42,
#         "run_dir": "../runs",
#         "max_len": 512
#     }

#     print(f"🚀 Using device: {DEVICE}")

#     # 1. 加载 ALBERT (带严格校验)
#     albert_model, tokenizer = setup_protalbert("/tmp/AbAgCDR/models/ProtAlbert")
#     emb_dim = albert_model.config.hidden_size
    
#     # 2. 加载并划分数据
#     all_train_samples_raw = []
#     all_val_samples_raw = []
#     all_test_samples_raw = []
#     dataset_test_splits_raw = {}

#     for tsv_path, weight in train_tsvs.items():
#         print(f"\n📂 处理数据集：{Path(tsv_path).name}")
#         samples = load_and_encode_tsv(tsv_path)
#         if not samples:
#             continue
            
#         train, val, test = split_samples_strict(
#             samples,
#             test_size=CONFIG["test_size"],
#             val_size=CONFIG["val_size"],
#             random_state=CONFIG["seed"]
#         )
#         all_train_samples_raw.extend(train)
#         all_val_samples_raw.extend(val)
#         all_test_samples_raw.extend(test)
        
#         dataset_name = Path(tsv_path).stem
#         dataset_test_splits_raw[dataset_name] = test

#     print(f"\n📊 数据汇总：Train={len(all_train_samples_raw)}, Val={len(all_val_samples_raw)}, Test={len(all_test_samples_raw)}")
    
#     if len(all_train_samples_raw) == 0:
#         print("❌ 没有训练数据，退出。")
#         return

#     # 3. 标签标准化
#     y_train = np.array([s[3] for s in all_train_samples_raw])
#     y_mean, y_std = y_train.mean(), y_train.std() + 1e-8
#     print(f"📈 标签统计：Mean={y_mean:.4f}, Std={y_std:.4f}")

#     def normalize_label(samples, mean, std):
#         return [(s[0], s[1], s[2], (s[3] - mean) / std) for s in samples]

#     all_train_samples = normalize_label(all_train_samples_raw, y_mean, y_std)
#     all_val_samples = normalize_label(all_val_samples_raw, y_mean, y_std)
#     all_test_samples = all_test_samples_raw 
#     dataset_test_splits_norm = dataset_test_splits_raw 

#     # 4. 超参数配置
#     param_grid = {
#         'lr': [2e-4, 1e-4, 1e-3],
#         'batch_size': [16, 32], 
#         'epochs': [60],
#         'patience': [10],
#         'weight_decay': [1e-4, 1e-5],
#         'min_lr': [1e-6]
#     }

#     keys, values = zip(*param_grid.items())
#     param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

#     best_score = -np.inf
#     best_model_state = None
#     save_path = Path(CONFIG["run_dir"]) / f"best_model_e2e_{datetime.now().strftime('%Y%m%d_%H%M')}.pth"
#     save_path.parent.mkdir(parents=True, exist_ok=True)

#     # 5. 训练循环
#     for trial_idx, params in enumerate(param_combinations):
#         print(f"\n{'='*60}")
#         print(f"🚀 Trial {trial_idx+1}/{len(param_combinations)}")
#         print(f"Params: LR={params['lr']}, Batch={params['batch_size']}, WD={params['weight_decay']}")
#         print(f"{'='*60}")

#         train_dataset = LiveEmbeddingDataset(all_train_samples, tokenizer, max_len=CONFIG["max_len"])
#         val_dataset = LiveEmbeddingDataset(all_val_samples, tokenizer, max_len=CONFIG["max_len"])

#         # 注意：num_workers=0 是因为我们在主进程中操作 GPU 模型
#         train_loader = DataLoader(
#             train_dataset,
#             batch_size=params['batch_size'],
#             shuffle=True,
#             collate_fn=lambda batch: collate_fn_with_albert(batch, tokenizer, albert_model, DEVICE, CONFIG["max_len"], noise_std=0.01 * y_std),
#             num_workers=0, 
#             pin_memory=True 
#         )
        
#         val_loader = DataLoader(
#             val_dataset,
#             batch_size=params['batch_size'],
#             shuffle=False,
#             collate_fn=lambda batch: collate_fn_with_albert(batch, tokenizer, albert_model, DEVICE, CONFIG["max_len"], noise_std=0.0),
#             num_workers=0,
#             pin_memory=True
#         )

#         model = TriProtDTAPlus(emb_dim=emb_dim, hidden_dim=512, dropout=0.3)
#         trainer = AdvancedTrainer(model, train_loader, val_loader, params, DEVICE, y_mean, y_std)
#         trained_model = trainer.train()

#         val_metrics = evaluate(trained_model, val_loader, DEVICE, y_mean, y_std)
#         score = val_metrics['PCC']
#         print(f"✅ Best Val PCC: {format_metric(score)}")

#         if score > best_score:
#             best_score = score
#             best_model_state = copy.deepcopy(trained_model.state_dict())
#             torch.save({
#                 'model_state_dict': best_model_state,
#                 'params': params,
#                 'y_mean': y_mean,
#                 'y_std': y_std,
#                 'emb_dim': emb_dim
#             }, save_path)
#             print(f"💾 新最佳模型已保存：{save_path}")

#     # 6. 最终评估
#     if best_model_state is None:
#         print("❌ 未训练出有效模型。")
#         return

#     final_model = TriProtDTAPlus(emb_dim=emb_dim)
#     final_model.load_state_dict(best_model_state)
#     final_model.to(DEVICE)

#     print("\n" + "="*80)
#     print("🔍 FINAL EVALUATION (Strict No-Leakage, End-to-End)")
#     print("="*80)

#     for name, test_samples in dataset_test_splits_norm.items():
#         if not test_samples:
#             continue
#         test_dataset = LiveEmbeddingDataset(test_samples, tokenizer, max_len=CONFIG["max_len"])
#         test_loader = DataLoader(
#             test_dataset,
#             batch_size=32,
#             shuffle=False,
#             collate_fn=lambda batch: collate_fn_with_albert(batch, tokenizer, albert_model, DEVICE, CONFIG["max_len"], noise_std=0.0),
#             num_workers=0,
#             pin_memory=True
#         )
#         metrics = evaluate(final_model, test_loader, DEVICE, y_mean, y_std)
#         line = f"{name.upper():<25} | R²: {format_metric(metrics['R2'])} | RMSE: {format_metric(metrics['RMSE'])} | PCC: {format_metric(metrics['PCC'])} | MAE: {format_metric(metrics['MAE'])}"
#         print(line)

#     if all_test_samples:
#         test_dataset_merged = LiveEmbeddingDataset(all_test_samples, tokenizer, max_len=CONFIG["max_len"])
#         test_loader_merged = DataLoader(
#             test_dataset_merged,
#             batch_size=32,
#             shuffle=False,
#             collate_fn=lambda batch: collate_fn_with_albert(batch, tokenizer, albert_model, DEVICE, CONFIG["max_len"], noise_std=0.0),
#             num_workers=0,
#             pin_memory=True
#         )
#         metrics = evaluate(final_model, test_loader_merged, DEVICE, y_mean, y_std)
#         line = f"{'MERGED TEST SET':<25} | R²: {format_metric(metrics['R2'])} | RMSE: {format_metric(metrics['RMSE'])} | PCC: {format_metric(metrics['PCC'])} | MAE: {format_metric(metrics['MAE'])}"
#         print(line)

#     print("\n" + "="*80)
#     print(f"🎉 训练完成！最佳验证集 PCC: {format_metric(best_score)}")
#     print(f"💾 模型路径：{save_path}")
#     print("="*80)

# if __name__ == "__main__":
#     main()

import os
import warnings
import sys
import math
import copy
import itertools
from datetime import datetime
from pathlib import Path
from collections import defaultdict

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import GroupShuffleSplit

# ==============================
# 0. 依赖检查
# ==============================
try:
    import sentencepiece
    from transformers import AlbertModel, AlbertTokenizer
except ImportError:
    print("❌ 错误：缺少依赖库。请运行：pip install sentencepiece transformers")
    sys.exit(1)

# ==============================
# 1. 全局模型加载
# ==============================
GLOBAL_ALBERT_MODEL = None
GLOBAL_ALBERT_TOKENIZER = None
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def setup_protalbert(model_dir="models/ProtAlbert"):
    global GLOBAL_ALBERT_MODEL, GLOBAL_ALBERT_TOKENIZER
    
    if GLOBAL_ALBERT_MODEL is not None:
        return GLOBAL_ALBERT_MODEL, GLOBAL_ALBERT_TOKENIZER

    model_dir = Path(model_dir)
    
    config_exists = (model_dir / "config.json").exists()
    spm_exists = (model_dir / "spiece.model").exists()
    weight_exists = (model_dir / "pytorch_model.bin").exists() or (model_dir / "model.safetensors").exists()

    if not (config_exists and spm_exists and weight_exists):
        print(f"❌ 错误：本地模型文件不完整。正在尝试下载 'albert-base-v2' ...")
        try:
            model_name = "albert-base-v2"
            tokenizer = AlbertTokenizer.from_pretrained(model_name)
            model = AlbertModel.from_pretrained(model_name)
            
            os.makedirs(model_dir, exist_ok=True)
            tokenizer.save_pretrained(model_dir)
            model.save_pretrained(model_dir)
            print(f"✅ 模型已下载并保存至：{model_dir}")
        except Exception as e:
            print(f"❌ 下载失败：{e}")
            print("💡 提示：请运行 'export HF_ENDPOINT=https://hf-mirror.com'")
            raise RuntimeError("无法加载模型。")
    else:
        print(f"🧠 加载本地模型...")
        try:
            tokenizer = AlbertTokenizer.from_pretrained(str(model_dir))
            model = AlbertModel.from_pretrained(str(model_dir))
            if model.config.hidden_size != 768:
                raise ValueError("维度不匹配")
            print("✅ 本地模型验证通过。")
        except Exception as e:
            print(f"⚠️ 本地模型验证失败：{e}，尝试重新下载...")
            model_name = "albert-base-v2"
            tokenizer = AlbertTokenizer.from_pretrained(model_name)
            model = AlbertModel.from_pretrained(model_name)
            os.makedirs(model_dir, exist_ok=True)
            tokenizer.save_pretrained(model_dir)
            model.save_pretrained(model_dir)

    model = model.to(DEVICE)
    model.eval() 
    
    GLOBAL_ALBERT_MODEL = model
    GLOBAL_ALBERT_TOKENIZER = tokenizer
    print(f"✅ ALBERT 就绪。设备：{DEVICE}, 维度：768")
        
    return model, tokenizer

# ==============================
# 2. 模型定义 (关键：Bias 初始化为负值)
# ==============================
class TriProtDTAPlus(nn.Module):
    def __init__(self, emb_dim, hidden_dim=512, dropout=0.3):
        super().__init__()
        self.emb_dim = emb_dim
        
        def create_proj():
            return nn.Sequential(
                nn.Linear(emb_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            )
        
        self.proj_l = create_proj()
        self.proj_h = create_proj()
        self.proj_ag = create_proj()
        
        input_dim = hidden_dim * 6
        
        self.interaction = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        self.head = nn.Sequential(
            nn.Linear(hidden_dim // 2, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout * 1.5),
            nn.Linear(64, 1)
        )
        
        # 【关键修复】Delta G 范围是 [-15.5, -2]，均值约 -8.5
        # 强制将输出层的 Bias 初始化为 -8.0，让模型一开始就预测负值
        # 这能防止模型在训练初期输出正数，导致反归一化后误差爆炸
        torch.nn.init.constant_(self.head[-1].bias, -8.0)
        print("💡 已初始化输出层 Bias 为 -8.0，以适配 Delta G 的负值特性 (-15.5 ~ -2)。")

    def forward(self, l_emb, h_emb, ag_emb):
        l = self.proj_l(l_emb)
        h = self.proj_h(h_emb)
        ag = self.proj_ag(ag_emb)
        
        pairwise_lh = l * h
        pairwise_lag = l * ag
        pairwise_hag = h * ag
        
        x = torch.cat([l, h, ag, pairwise_lh, pairwise_lag, pairwise_hag], dim=1)
        x = self.interaction(x)
        return self.head(x)

# ==============================
# 3. 数据集类 (支持 Dataset-ID)
# ==============================
class LiveEmbeddingDataset(Dataset):
    def __init__(self, samples, tokenizer, max_len=512):
        """
        samples 格式: [(l_seq, h_seq, ag_seq, label, dataset_id), ...]
        """
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        l_seq, h_seq, ag_seq, label, ds_id = self.samples[idx]
        return {
            'l_seq': l_seq,
            'h_seq': h_seq,
            'ag_seq': ag_seq,
            'label': label,
            'ds_id': ds_id
        }

def collate_fn_with_albert_dataset_norm(batch, tokenizer, model, device, stats_dict, max_len=512, noise_std=0.0):
    """
    支持分数据集标准化的 Collate Function
    stats_dict: {dataset_id: {'mean': float, 'std': float}}
    """
    l_seqs = [item['l_seq'] for item in batch]
    h_seqs = [item['h_seq'] for item in batch]
    ag_seqs = [item['ag_seq'] for item in batch]
    
    labels_raw = torch.tensor([item['label'] for item in batch], dtype=torch.float32).unsqueeze(1)
    ds_ids = [item['ds_id'] for item in batch]
    
    # 【核心逻辑】逐样本进行归一化：(y - mean) / std
    labels_norm = torch.zeros_like(labels_raw)
    for i, ds_id in enumerate(ds_ids):
        if ds_id not in stats_dict:
            # 保护机制：如果遇到未知 ID，使用 0 均值 1 方差（相当于不归一化）
            m, s = 0.0, 1.0 
        else:
            m = stats_dict[ds_id]['mean']
            s = stats_dict[ds_id]['std']
        labels_norm[i, 0] = (labels_raw[i, 0] - m) / (s + 1e-8)
    
    def get_batch_emb(seqs):
        inputs = tokenizer(seqs, return_tensors="pt", padding=True, truncation=True, max_length=max_len)
        inputs = inputs.to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            embs = outputs.last_hidden_state[:, 0, :]
        embs = embs.cpu()
        if noise_std > 0.0:
            embs = embs + torch.randn_like(embs) * noise_std
        return embs

    l_emb = get_batch_emb(l_seqs)
    h_emb = get_batch_emb(h_seqs)
    ag_emb = get_batch_emb(ag_seqs)
    
    return {
        'l_emb': l_emb,
        'h_emb': h_emb,
        'ag_emb': ag_emb,
        'label': labels_norm,       # 归一化后的标签 (用于 Loss 计算)
        'label_raw': labels_raw,    # 原始标签 (用于调试)
        'ds_ids': ds_ids            # 数据集 ID (用于反归一化)
    }

# ==============================
# 4. 评估函数 (分数据集反归一化)
# ==============================
def evaluate_dataset_aware(model, dataloader, device, stats_dict):
    model.eval()
    all_preds_real, all_labels_real = [], []
    
    with torch.no_grad():
        for batch in dataloader:
            l = batch['l_emb'].to(device, non_blocking=True)
            h = batch['h_emb'].to(device, non_blocking=True)
            ag = batch['ag_emb'].to(device, non_blocking=True)
            
            labels_norm = batch['label'].to(device, non_blocking=True).cpu().numpy().flatten()
            ds_ids = batch['ds_ids']
            
            preds_norm = model(l, h, ag).cpu().numpy().flatten()

            # 【核心逻辑】逐样本反归一化：y_norm * std + mean
            preds_real = np.zeros_like(preds_norm)
            labels_real = np.zeros_like(labels_norm)
            
            for i, ds_id in enumerate(ds_ids):
                if ds_id in stats_dict:
                    m = stats_dict[ds_id]['mean']
                    s = stats_dict[ds_id]['std']
                    preds_real[i] = preds_norm[i] * s + m
                    labels_real[i] = labels_norm[i] * s + m
                else:
                    # 未知数据集，无法还原绝对值
                    preds_real[i] = preds_norm[i]
                    labels_real[i] = labels_norm[i]

            all_preds_real.extend(preds_real.tolist())
            all_labels_real.extend(labels_real.tolist())

    if len(all_labels_real) == 0:
        return {'MSE': np.nan, 'RMSE': np.nan, 'MAE': np.nan, 'R2': np.nan, 'PCC': 0.0, 'Mean_Shift': 0.0}

    preds_arr = np.array(all_preds_real)
    labels_arr = np.array(all_labels_real)

    mse = mean_squared_error(labels_arr, preds_arr)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(labels_arr, preds_arr)
    
    if len(labels_arr) < 2:
        pcc = 0.0
        r2_raw = -np.inf
        r2_corrected = -np.inf
    else:
        pcc = np.corrcoef(labels_arr, preds_arr)[0, 1]
        if np.isnan(pcc): pcc = 0.0
        try:
            r2_raw = r2_score(labels_arr, preds_arr)
        except:
            r2_raw = -np.inf

        # 计算校正 R2 (诊断用)
        mean_shift = np.mean(preds_arr) - np.mean(labels_arr)
        preds_corrected = preds_arr - mean_shift
        try:
            r2_corrected = r2_score(labels_arr, preds_corrected)
        except:
            r2_corrected = -np.inf
            
        mean_shift = np.mean(preds_arr) - np.mean(labels_arr)

    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2_raw,
        'R2_Corrected': r2_corrected,
        'PCC': float(pcc),
        'Mean_Shift': mean_shift,
        'Pred_Mean': np.mean(preds_arr),
        'True_Mean': np.mean(labels_arr)
    }

# ==============================
# 5. 数据划分与清洗
# ==============================
def split_samples_strict(samples, test_size=0.2, val_size=0.2, random_state=42):
    df = pd.DataFrame(samples, columns=['heavy', 'light', 'antigen', 'label', 'ds_id'])
    df['group_id'] = df['heavy'].astype(str) + "|" + df['light'].astype(str) + "|" + df['antigen'].astype(str)
    
    indices = np.arange(len(df))
    groups = df['group_id'].values
    
    if len(groups) == 0:
        return [], [], []

    gss_test = GroupShuffleSplit(test_size=test_size, n_splits=1, random_state=random_state)
    try:
        train_val_idx, test_idx = next(gss_test.split(indices, groups=groups))
    except ValueError:
        return samples, [], []
    
    df_train_val = df.iloc[train_val_idx].reset_index(drop=True)
    df_test = df.iloc[test_idx].reset_index(drop=True)
    
    if len(df_train_val) < 5:
        df_train = df_train_val
        df_val = pd.DataFrame(columns=df_train_val.columns)
    else:
        tv_indices = np.arange(len(df_train_val))
        tv_groups = df_train_val['group_id'].values
        adjusted_val_size = val_size / (1.0 - test_size)
        gss_val = GroupShuffleSplit(test_size=adjusted_val_size, n_splits=1, random_state=random_state+1)
        train_idx, val_idx = next(gss_val.split(tv_indices, groups=tv_groups))
        df_train = df_train_val.iloc[train_idx]
        df_val = df_train_val.iloc[val_idx]
    
    to_list = lambda df_part: list(df_part[['heavy', 'light', 'antigen', 'label', 'ds_id']].itertuples(index=False, name=None))
    
    train_list = to_list(df_train)
    val_list = to_list(df_val)
    test_list = to_list(df_test)
    
    train_groups = set(df_train['group_id']) if not df_train.empty else set()
    test_groups = set(df_test['group_id']) if not df_test.empty else set()
    val_groups = set(df_val['group_id']) if not df_val.empty else set()
    
    if train_groups.intersection(test_groups) or train_groups.intersection(val_groups):
        raise RuntimeError("❌ 严重错误：检测到数据泄露！")
    
    print(f"   ✅ 严格划分完成：Train={len(train_list)}, Val={len(val_list)}, Test={len(test_list)}")
    return train_list, val_list, test_list

def clean_seq(seq):
    if pd.isna(seq) or seq is None:
        return "A"
    valid_aas = set("ARNDCQEGHILKMFPSTWYV")
    cleaned = ''.join([aa for aa in str(seq).upper().strip() if aa in valid_aas])
    return cleaned if cleaned else "A"

def load_and_encode_tsv(tsv_path, dataset_id):
    if not os.path.exists(tsv_path):
        return []
    try:
        df = pd.read_csv(tsv_path, sep='\t', on_bad_lines='skip')
    except Exception as e:
        return []

    cols = [str(c).lower() for c in df.columns]
    mapping = {}
    light_keys = ['antibody_seq_a', 'light', 'vl', 'l_chain', 'light_chain']
    heavy_keys = ['antibody_seq_b', 'heavy', 'vh', 'h_chain', 'heavy_chain']
    antigen_keys = ['antigen_seq', 'antigen', 'ag', 'target_seq']
    label_keys = ['delta_g', 'dg', 'affinity']
    
    def find_col(keys, cols_lower, original_cols):
        for k in keys:
            if k in cols_lower:
                return original_cols[cols_lower.index(k)]
        return None

    orig_cols = list(df.columns)
    mapping['h'] = find_col(heavy_keys, cols, orig_cols)
    mapping['l'] = find_col(light_keys, cols, orig_cols)
    mapping['ag'] = find_col(antigen_keys, cols, orig_cols)
    mapping['dg'] = find_col(label_keys, cols, orig_cols)
    
    if not all(mapping.values()):
        return []

    # 【重要】再次确认标签范围，过滤异常值
    # 你的标签应该在 -15.5 到 -2 之间
    try:
        df = df[(df[mapping['dg']] >= -20.0) & (df[mapping['dg']] <= 0.0)]
    except:
        pass
        
    samples = []
    for _, row in df.iterrows():
        a = clean_seq(row[mapping['l']])
        b = clean_seq(row[mapping['h']])
        ag = clean_seq(row[mapping['ag']])
        try:
            dg = float(row[mapping['dg']])
        except:
            continue
        if a and b and ag:
            samples.append((a, b, ag, dg, dataset_id))
            
    print(f"   📥 加载有效样本：{len(samples)}/{len(df)} (ID: {dataset_id})")
    return samples

# ==============================
# 6. 高级训练器
# ==============================
class AdvancedTrainer:
    def __init__(self, model, train_loader, val_loader, params, device, stats_dict):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.params = params
        self.device = device
        self.stats_dict = stats_dict
        
        self.optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=params['lr'], 
            weight_decay=params['weight_decay'],
            betas=(0.9, 0.98)
        )
        
        self.criterion = nn.HuberLoss(delta=1.0)
        self.patience = params['patience']
        self.total_steps = len(train_loader) * params['epochs']
        self.warmup_steps = int(self.total_steps * 0.1)

    def get_lr(self, step):
        if step < self.warmup_steps:
            return self.params['lr'] * (step / max(1, self.warmup_steps))
        else:
            progress = (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
            return self.params['min_lr'] + (self.params['lr'] - self.params['min_lr']) * 0.5 * (1.0 + math.cos(math.pi * progress))

    def train(self):
        best_score = -np.inf
        patience_counter = 0
        best_model_state = None
        global_step = 0
        
        for epoch in range(self.params['epochs']):
            self.model.train()
            total_loss = 0
            
            for batch in self.train_loader:
                l = batch['l_emb'].to(self.device, non_blocking=True)
                h = batch['h_emb'].to(self.device, non_blocking=True)
                ag = batch['ag_emb'].to(self.device, non_blocking=True)
                labels = batch['label'].to(self.device, non_blocking=True)
                
                self.optimizer.zero_grad()
                preds = self.model(l, h, ag)
                loss = self.criterion(preds, labels)
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                
                global_step += 1
                current_lr = self.get_lr(global_step)
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = current_lr
                    
                total_loss += loss.item()

            val_metrics = evaluate_dataset_aware(self.model, self.val_loader, self.device, self.stats_dict)
            current_pcc = val_metrics['PCC'] if not np.isnan(val_metrics['PCC']) else -1.0

            if current_pcc > best_score:
                best_score = current_pcc
                patience_counter = 0
                best_model_state = copy.deepcopy(self.model.state_dict())
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    print(f"   ⏹️ Early Stopping at epoch {epoch+1} (Best PCC: {best_score:.4f})")
                    break
            
            if (epoch + 1) % 5 == 0 or epoch == 0:
                avg_lr = self.optimizer.param_groups[0]['lr']
                print(f"   Epoch {epoch+1:02d} | Loss: {total_loss/len(self.train_loader):.4f} | Val PCC: {current_pcc:.4f} | LR: {avg_lr:.2e}")

        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
        return self.model

def format_metric(val, fmt="{:.4f}"):
    if val is None or not np.isfinite(val):
        return "N/A"
    return fmt.format(val)

# ==============================
# 7. 主程序
# ==============================
def main():
    train_tsvs = {
        "FINAL_DATASET_TRAIN": "/tmp/AbAgCDR/data/final_dataset_train.tsv",
        "SKEMPI": "/tmp/AbAgCDR/data/pairs_seq_skempi_clean.tsv",
        "SABDAB": "/tmp/AbAgCDR/data/pairs_seq_sabdab_clean.tsv",
        "ABBIND2": "/tmp/AbAgCDR/data/pairs_seq_abbind2_clean.tsv",
    }
    
    CONFIG = {
        "test_size": 0.2,
        "val_size": 0.2,
        "seed": 42,
        "run_dir": "../runs",
        "max_len": 512
    }

    print(f"🚀 Using device: {DEVICE}")
    albert_model, tokenizer = setup_protalbert("/tmp/AbAgCDR/models/ProtAlbert")
    emb_dim = albert_model.config.hidden_size
    
    all_train_samples_raw = []
    all_val_samples_raw = []
    all_test_samples_raw = []
    dataset_test_splits_raw = {}

    # 1. 加载数据
    for ds_id, tsv_path in train_tsvs.items():
        print(f"\n📂 处理数据集：{ds_id}")
        samples = load_and_encode_tsv(tsv_path, dataset_id=ds_id)
        if not samples:
            continue
            
        train, val, test = split_samples_strict(
            samples,
            test_size=CONFIG["test_size"],
            val_size=CONFIG["val_size"],
            random_state=CONFIG["seed"]
        )
        all_train_samples_raw.extend(train)
        all_val_samples_raw.extend(val)
        all_test_samples_raw.extend(test)
        
        dataset_test_splits_raw[ds_id] = test

    print(f"\n📊 数据汇总：Train={len(all_train_samples_raw)}, Val={len(all_val_samples_raw)}, Test={len(all_test_samples_raw)}")
    
    if len(all_train_samples_raw) == 0:
        print("❌ 没有训练数据，退出。")
        return

    # 2. 【核心】计算每个数据集的 Mean 和 Std
    stats_dict = {}
    print("\n📈 计算各数据集标准化统计量 (Dataset-wise Normalization):")
    print("   注意：标签范围应为 [-15.5, -2]，均值应为负数。")
    
    train_by_ds = defaultdict(list)
    for s in all_train_samples_raw:
        train_by_ds[s[4]].append(s[3])
        
    for ds_id, labels in train_by_ds.items():
        arr = np.array(labels)
        m = arr.mean()
        s = arr.std() + 1e-8
        stats_dict[ds_id] = {'mean': m, 'std': s}
        print(f"   [{ds_id:<10}] Count: {len(arr):<5} | Mean: {m:>7.2f} | Std: {s:>6.2f} | Min: {arr.min():.2f} | Max: {arr.max():.2f}")

    # 3. 超参数配置
    param_grid = {
        'lr': [2e-4, 1e-4],
        'batch_size': [16, 32], 
        'epochs': [60],
        'patience': [10],
        'weight_decay': [1e-4, 1e-5],
        'min_lr': [1e-6]
    }

    keys, values = zip(*param_grid.items())
    param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    best_score = -np.inf
    best_model_state = None
    save_path = Path(CONFIG["run_dir"]) / f"best_model_ds_norm_{datetime.now().strftime('%Y%m%d_%H%M')}.pth"
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # 4. 训练循环
    for trial_idx, params in enumerate(param_combinations):
        print(f"\n{'='*60}")
        print(f"🚀 Trial {trial_idx+1}/{len(param_combinations)}")
        print(f"Params: LR={params['lr']}, Batch={params['batch_size']}")
        print(f"{'='*60}")

        train_dataset = LiveEmbeddingDataset(all_train_samples_raw, tokenizer, max_len=CONFIG["max_len"])
        val_dataset = LiveEmbeddingDataset(all_val_samples_raw, tokenizer, max_len=CONFIG["max_len"])

        train_collate = lambda batch: collate_fn_with_albert_dataset_norm(
            batch, tokenizer, albert_model, DEVICE, stats_dict, CONFIG["max_len"], noise_std=0.01
        )
        val_collate = lambda batch: collate_fn_with_albert_dataset_norm(
            batch, tokenizer, albert_model, DEVICE, stats_dict, CONFIG["max_len"], noise_std=0.0
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=params['batch_size'],
            shuffle=True,
            collate_fn=train_collate,
            num_workers=0, 
            pin_memory=True 
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=params['batch_size'],
            shuffle=False,
            collate_fn=val_collate,
            num_workers=0,
            pin_memory=True
        )

        model = TriProtDTAPlus(emb_dim=emb_dim, hidden_dim=512, dropout=0.3)
        trainer = AdvancedTrainer(model, train_loader, val_loader, params, DEVICE, stats_dict)
        trained_model = trainer.train()

        val_metrics = evaluate_dataset_aware(trained_model, val_loader, DEVICE, stats_dict)
        score = val_metrics['PCC']
        print(f"✅ Best Val PCC: {format_metric(score)} (R²: {format_metric(val_metrics['R2'])})")

        if score > best_score:
            best_score = score
            best_model_state = copy.deepcopy(trained_model.state_dict())
            torch.save({
                'model_state_dict': best_model_state,
                'params': params,
                'stats_dict': stats_dict,
                'emb_dim': emb_dim
            }, save_path)
            print(f"💾 新最佳模型已保存：{save_path}")

    # 5. 最终评估
    if best_model_state is None:
        print("❌ 未训练出有效模型。")
        return

    final_model = TriProtDTAPlus(emb_dim=emb_dim)
    final_model.load_state_dict(best_model_state)
    final_model.to(DEVICE)

    print("\n" + "="*90)
    print("🔍 FINAL EVALUATION (Dataset-wise Normalization + Negative Bias Init)")
    print("="*90)
    print(f"{'Dataset':<25} | {'R²':<8} | {'R²(cor)':<8} | {'RMSE':<8} | {'MAE':<8} | {'PCC':<8} | {'Shift':<8}")
    print("-" * 90)

    for name, test_samples in dataset_test_splits_raw.items():
        if not test_samples:
            continue
        test_dataset = LiveEmbeddingDataset(test_samples, tokenizer, max_len=CONFIG["max_len"])
        test_loader = DataLoader(
            test_dataset,
            batch_size=32,
            shuffle=False,
            collate_fn=lambda batch: collate_fn_with_albert_dataset_norm(
                batch, tokenizer, albert_model, DEVICE, stats_dict, CONFIG["max_len"], noise_std=0.0
            ),
            num_workers=0,
            pin_memory=True
        )
        metrics = evaluate_dataset_aware(final_model, test_loader, DEVICE, stats_dict)
        
        line = (f"{name.upper():<25} | "
                f"{format_metric(metrics['R2']):<8} | "
                f"{format_metric(metrics.get('R2_Corrected', np.nan)):<8} | "
                f"{format_metric(metrics['RMSE']):<8} | "
                f"{format_metric(metrics['MAE']):<8} | "
                f"{format_metric(metrics['PCC']):<8} | "
                f"{format_metric(metrics['Mean_Shift']):<8}")
        print(line)

    if all_test_samples_raw:
        test_dataset_merged = LiveEmbeddingDataset(all_test_samples_raw, tokenizer, max_len=CONFIG["max_len"])
        test_loader_merged = DataLoader(
            test_dataset_merged,
            batch_size=32,
            shuffle=False,
            collate_fn=lambda batch: collate_fn_with_albert_dataset_norm(
                batch, tokenizer, albert_model, DEVICE, stats_dict, CONFIG["max_len"], noise_std=0.0
            ),
            num_workers=0,
            pin_memory=True
        )
        metrics = evaluate_dataset_aware(final_model, test_loader_merged, DEVICE, stats_dict)
        line = (f"{'MERGED TEST SET':<25} | "
                f"{format_metric(metrics['R2']):<8} | "
                f"{format_metric(metrics.get('R2_Corrected', np.nan)):<8} | "
                f"{format_metric(metrics['RMSE']):<8} | "
                f"{format_metric(metrics['MAE']):<8} | "
                f"{format_metric(metrics['PCC']):<8} | "
                f"{format_metric(metrics['Mean_Shift']):<8}")
        print(line)

    print("="*90)
    print("💡 预期结果:")
    print("   1. R² 应从负数变为正数 (例如 0.2 ~ 0.5)。")
    print("   2. RMSE 应从 ~27 降低到合理范围 (例如 2.0 ~ 5.0)。")
    print("   3. Shift 应接近 0。")
    print(f"🎉 训练完成！最佳验证集 PCC: {format_metric(best_score)}")
    print(f"💾 模型路径：{save_path}")
    print("="*90)

if __name__ == "__main__":
    main()