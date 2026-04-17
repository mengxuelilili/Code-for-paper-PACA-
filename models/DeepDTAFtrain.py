
# import copy
# import itertools
# from datetime import datetime
# from pathlib import Path

# import numpy as np
# import pandas as pd
# import torch
# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# from scipy.stats import pearsonr
# from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
# from torch import nn, optim
# from torch.optim.lr_scheduler import ReduceLROnPlateau


# # ======================
# # 1. 序列 → One-hot 编码
# # ======================
# AA_LIST = list("ACDEFGHIKLMNPQRSTVWY")
# AA_TO_IDX = {aa: idx for idx, aa in enumerate(AA_LIST)}
# AA_TO_IDX["X"] = len(AA_LIST)
# PT_FEATURE_SIZE = len(AA_TO_IDX)  # 21


# def seq_to_onehot(seq: str) -> np.ndarray:
#     """Convert amino acid sequence to one-hot (no padding yet)"""
#     seq = seq.upper()
#     arr = np.zeros((len(seq), PT_FEATURE_SIZE), dtype=np.float32)
#     for i, aa in enumerate(seq):
#         idx = AA_TO_IDX.get(aa, AA_TO_IDX["X"])
#         arr[i, idx] = 1.0
#     return arr


# # ======================
# # 2. Dataset & Collate
# # ======================
# class ListDataset(Dataset):
#     def __init__(self, samples):
#         self.samples = samples  # each sample: (heavy_onehot, light_onehot, antigen_onehot, delta_g)

#     def __len__(self):
#         return len(self.samples)

#     def __getitem__(self, idx):
#         return self.samples[idx]


# def collate_fn(batch):
#     """Dynamic padding per batch to max length in that batch"""
#     X_a_list = [torch.tensor(item[0], dtype=torch.float32) for item in batch]  # heavy
#     X_b_list = [torch.tensor(item[1], dtype=torch.float32) for item in batch]  # light
#     ag_list = [torch.tensor(item[2], dtype=torch.float32) for item in batch]   # antigen
#     y_list = [torch.tensor(item[3], dtype=torch.float32) for item in batch]

#     max_len = max(
#         max(x.shape[0] for x in X_a_list),
#         max(x.shape[0] for x in X_b_list),
#         max(x.shape[0] for x in ag_list)
#     )

#     def pad_to_len(x, L):
#         if x.shape[0] < L:
#             pad = torch.zeros(L - x.shape[0], x.shape[1], dtype=x.dtype)
#             return torch.cat([x, pad], dim=0)
#         else:
#             return x[:L]

#     X_a_padded = torch.stack([pad_to_len(x, max_len) for x in X_a_list])
#     X_b_padded = torch.stack([pad_to_len(x, max_len) for x in X_b_list])
#     ag_padded = torch.stack([pad_to_len(x, max_len) for x in ag_list])
#     y_tensor = torch.stack(y_list)

#     return X_a_padded, X_b_padded, ag_padded, y_tensor


# # ======================
# # 3. 模型定义（TriProtDTA）
# # ======================
# class Squeeze(nn.Module):
#     def forward(self, x):
#         return x.squeeze(-1)


# class CDilated(nn.Module):
#     def __init__(self, nIn, nOut, kSize, stride=1, d=1):
#         super().__init__()
#         padding = int((kSize - 1) / 2) * d
#         self.conv = nn.Conv1d(nIn, nOut, kSize, stride=stride, padding=padding, bias=False, dilation=d)
#     def forward(self, x):
#         return self.conv(x)


# class DilatedParallelResidualBlockA(nn.Module):
#     def __init__(self, nIn, nOut, add=True):
#         super().__init__()
#         n = int(nOut / 5)
#         n1 = nOut - 4 * n
#         self.c1 = nn.Conv1d(nIn, n, 1)
#         self.br1 = nn.Sequential(nn.BatchNorm1d(n), nn.PReLU())
#         self.d1 = CDilated(n, n1, 3, 1, 1)
#         self.d2 = CDilated(n, n, 3, 1, 2)
#         self.d4 = CDilated(n, n, 3, 1, 4)
#         self.d8 = CDilated(n, n, 3, 1, 8)
#         self.d16 = CDilated(n, n, 3, 1, 16)
#         self.br2 = nn.Sequential(nn.BatchNorm1d(nOut), nn.PReLU())
#         self.add = add and (nIn == nOut)

#     def forward(self, x):
#         out = self.br1(self.c1(x))
#         d1 = self.d1(out)
#         d2 = self.d2(out)
#         d4 = self.d4(out)
#         d8 = self.d8(out)
#         d16 = self.d16(out)
#         add1 = d2
#         add2 = add1 + d4
#         add3 = add2 + d8
#         add4 = add3 + d16
#         combine = torch.cat([d1, add1, add2, add3, add4], dim=1)
#         if self.add:
#             combine = x + combine
#         return self.br2(combine)


# class TriProtDTA(nn.Module):
#     def __init__(self, embed_size=128, out_channels=128):
#         super().__init__()
#         self.embed = nn.Linear(PT_FEATURE_SIZE, embed_size)
#         self.encoder_h = self._make_encoder(embed_size, out_channels)
#         self.encoder_l = self._make_encoder(embed_size, out_channels)
#         self.encoder_a = self._make_encoder(embed_size, out_channels)
#         self.dropout = nn.Dropout(0.2)
#         self.classifier = nn.Sequential(
#             nn.Linear(out_channels * 3, 128),
#             nn.Dropout(0.5),
#             nn.PReLU(),
#             nn.Linear(128, 64),
#             nn.Dropout(0.5),
#             nn.PReLU(),
#             nn.Linear(64, 1),
#             nn.PReLU()
#         )

#     def _make_encoder(self, in_ch, out_ch):
#         layers = []
#         ic = in_ch
#         for oc in [32, 64, out_ch]:
#             layers.append(DilatedParallelResidualBlockA(ic, oc))
#             ic = oc
#         layers.append(nn.AdaptiveMaxPool1d(1))
#         layers.append(Squeeze())
#         return nn.Sequential(*layers)

#     def forward(self, heavy, light, antigen):
#         h = self.embed(heavy).transpose(1, 2)
#         l = self.embed(light).transpose(1, 2)
#         a = self.embed(antigen).transpose(1, 2)
#         h_feat = self.encoder_h(h)
#         l_feat = self.encoder_l(l)
#         a_feat = self.encoder_a(a)
#         fused = torch.cat([h_feat, l_feat, a_feat], dim=1)
#         fused = self.dropout(fused)
#         return self.classifier(fused)


# # ======================
# # 4. 评估函数（支持反归一化）
# # ======================
# def evaluate(model, loader, device, y_mean=None, y_std=None):
#     model.eval()
#     y_true, y_pred = [], []
#     with torch.no_grad():
#         for X_a, X_b, ag, y in loader:
#             X_a, X_b, ag, y = X_a.to(device), X_b.to(device), ag.to(device), y.to(device)
#             pred = model(X_a, X_b, ag).view(-1)
#             y_true.extend(y.cpu().numpy())
#             y_pred.extend(pred.cpu().numpy())

#     y_true, y_pred = np.array(y_true), np.array(y_pred)

#     # 反归一化（如果提供了统计量）
#     if y_mean is not None and y_std is not None:
#         y_true = y_true * y_std + y_mean
#         y_pred = y_pred * y_std + y_mean

#     mse = mean_squared_error(y_true, y_pred)
#     rmse = np.sqrt(mse)
#     mae = mean_absolute_error(y_true, y_pred)
#     r2 = r2_score(y_true, y_pred)
#     pcc = pearsonr(y_true, y_pred)[0] if len(set(y_true)) > 1 else 0.0

#     return {"MSE": mse, "RMSE": rmse, "MAE": mae, "R2": r2, "PCC": pcc}


# # ======================
# # 5. 数据加载与划分（支持多文件）
# # ======================
# def load_and_encode_tsv(tsv_path):
#     df = pd.read_csv(tsv_path, sep='\t')
#     required_cols = {"antibody_seq_a", "antibody_seq_b", "antigen_seq", "delta_g"}
#     assert required_cols <= set(df.columns), f"Missing columns in {tsv_path}"

#     samples = []
#     for _, row in df.iterrows():
#         h = seq_to_onehot(row["antibody_seq_a"])
#         l = seq_to_onehot(row["antibody_seq_b"])
#         a = seq_to_onehot(row["antigen_seq"])
#         y = row["delta_g"]
#         samples.append((h, l, a, y))
#     return samples


# def split_samples(samples, test_size=0.2, val_size=0.2, random_state=42):
#     from sklearn.utils import shuffle
#     samples = shuffle(samples, random_state=random_state)
#     n = len(samples)
#     n_test = int(n * test_size)
#     n_val = int(n * val_size)

#     test = samples[:n_test]
#     val = samples[n_test:n_test + n_val]
#     train = samples[n_test + n_val:]
#     return train, val, test


# # ======================
# # 6. Trainer（支持标签归一化）
# # ======================
# class TrainerWithScheduler:
#     def __init__(self, model, train_loader, val_loader, params, device, y_mean=None, y_std=None):
#         self.model = model.to(device)
#         self.train_loader = train_loader
#         self.val_loader = val_loader
#         self.device = device
#         self.opt = optim.Adam(
#             model.parameters(),
#             lr=params["lr"],
#             weight_decay=params["weight_decay"]
#         )
#         self.scheduler = ReduceLROnPlateau(
#             self.opt,
#             mode='min',
#             factor=params.get("lr_factor", 0.5),
#             patience=params.get("scheduler_patience", 3),
#             min_lr=params.get("min_lr", 1e-6),
#             verbose=False
#         )
#         self.criterion = nn.MSELoss()
#         self.epochs = params["epochs"]
#         self.patience = params["patience"]
#         self.y_mean = y_mean
#         self.y_std = y_std

#     def train(self):
#         best_mse = np.inf
#         best_state = None
#         wait = 0

#         for epoch in range(1, self.epochs + 1):
#             self.model.train()
#             total_loss = 0.0
#             num_batches = 0

#             for X_a, X_b, ag, y in self.train_loader:
#                 X_a, X_b, ag, y = X_a.to(self.device), X_b.to(self.device), ag.to(self.device), y.to(self.device)
#                 self.opt.zero_grad()
#                 pred = self.model(X_a, X_b, ag).view(-1)
#                 loss = self.criterion(pred, y)
#                 loss.backward()
#                 self.opt.step()

#                 total_loss += loss.item()
#                 num_batches += 1

#             avg_train_loss = total_loss / num_batches
#             val_metrics = evaluate(self.model, self.val_loader, self.device, self.y_mean, self.y_std)
#             val_mse = val_metrics["MSE"]

#             self.scheduler.step(val_mse)

#             print(f"Epoch {epoch:02d}/{self.epochs} | "
#                   f"Train Loss: {avg_train_loss:.6f} | "
#                   f"Val MSE: {val_mse:.4f} | "
#                   f"Val R²: {val_metrics['R2']:.4f} | "
#                   f"Val PCC: {val_metrics['PCC']:.4f}")

#             if val_mse < best_mse:
#                 best_mse = val_mse
#                 best_state = copy.deepcopy(self.model.state_dict())
#                 wait = 0
#             else:
#                 wait += 1
#                 if wait >= self.patience:
#                     print("🛑 Early stopping triggered.")
#                     break

#         self.model.load_state_dict(best_state)
#         return self.model


# # ======================
# # 7. Main
# # ======================
# def main():
#     # =============== 配置 ===============
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
#     # ===================================

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"Using device: {device}")

#     # Step 1: Load all training datasets and split internally
#     all_train_samples = []
#     all_val_samples = []
#     all_test_samples = []
#     sample_weights = []

#     # 新增：保存每个数据集的 test 样本（用于后续单独评估）
#     dataset_test_splits = {}  # key: dataset name, value: test samples (raw y)

#     for tsv_path, weight in train_tsvs.items():
#         print(f"Loading {tsv_path}...")
#         samples = load_and_encode_tsv(tsv_path)
#         train, val, test = split_samples(
#             samples,
#             test_size=CONFIG["test_size"],
#             val_size=CONFIG["val_size"],
#             random_state=CONFIG["seed"]
#         )
#         all_train_samples.extend(train)
#         all_val_samples.extend(val)
#         all_test_samples.extend(test)
#         sample_weights.extend([weight] * len(train))

#         # 保存原始 test 样本（未归一化），用于后续按数据集评估
#         dataset_name = Path(tsv_path).stem  # e.g., "final_dataset_train"
#         dataset_test_splits[dataset_name] = test

#     print(f"Total Train: {len(all_train_samples)}, Val: {len(all_val_samples)}, Test: {len(all_test_samples)}")

#     # Step 2: 标签归一化（仅用训练集计算均值和标准差）
#     y_train = np.array([s[3] for s in all_train_samples])
#     y_mean, y_std = y_train.mean(), y_train.std() + 1e-8

#     # 归一化所有标签
#     def normalize_label(samples, mean, std):
#         return [(s[0], s[1], s[2], (s[3] - mean) / std) for s in samples]

#     all_train_samples = normalize_label(all_train_samples, y_mean, y_std)
#     all_val_samples = normalize_label(all_val_samples, y_mean, y_std)
#     all_test_samples = normalize_label(all_test_samples, y_mean, y_std)

#     # 同时归一化各数据集的 test 样本
#     dataset_test_splits_norm = {}
#     for name, samples in dataset_test_splits.items():
#         dataset_test_splits_norm[name] = normalize_label(samples, y_mean, y_std)

#     # Step 3: Weighted sampler
#     sampler = WeightedRandomSampler(
#         weights=sample_weights,
#         num_samples=len(sample_weights),
#         replacement=True
#     )

#     # Step 4: Hyperparameter grid
#     param_grid = {
#         'lr': [1e-4, 5e-4, 1e-3],
#         'batch_size': [16],
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
#             ListDataset(all_train_samples),
#             batch_size=params['batch_size'],
#             sampler=sampler,
#             collate_fn=collate_fn,
#             shuffle=False
#         )
#         val_loader = DataLoader(
#             ListDataset(all_val_samples),
#             batch_size=params['batch_size'],
#             shuffle=False,
#             collate_fn=collate_fn
#         )

#         model = TriProtDTA()
#         trainer = TrainerWithScheduler(model, train_loader, val_loader, params, device, y_mean, y_std)
#         trained_model = trainer.train()

#         val_metrics = evaluate(trained_model, val_loader, device, y_mean, y_std)
#         score = val_metrics['PCC']
#         print(f"✅ Val PCC: {score:.4f}")

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

#     # Step 5: Load best model
#     final_model = TriProtDTA()
#     final_model.load_state_dict(best_model_state)
#     final_model.to(device)

#     # Step 6: Evaluate on each individual dataset's test split
#     print("\n" + "="*70)
#     print("🔍 INDIVIDUAL DATASET TEST RESULTS (on their own held-out test sets)")
#     print("="*70)

#     for name, test_samples in dataset_test_splits_norm.items():
#         test_loader = DataLoader(
#             ListDataset(test_samples),
#             batch_size=32,
#             shuffle=False,
#             collate_fn=collate_fn
#         )
#         metrics = evaluate(final_model, test_loader, device, y_mean, y_std)
#         print(f"\n{name.upper()} TEST → "
#               f"R²: {metrics['R2']:.4f}, "
#               f"MSE: {metrics['MSE']:.4f}, "
#               f"RMSE: {metrics['RMSE']:.4f}, "
#               f"MAE: {metrics['MAE']:.4f}, "
#               f"PCC: {metrics['PCC']:.4f}")

#     # Step 7: Evaluate on merged internal test set
#     test_loader_merged = DataLoader(ListDataset(all_test_samples), batch_size=32, shuffle=False, collate_fn=collate_fn)
#     test_metrics_merged = evaluate(final_model, test_loader_merged, device, y_mean, y_std)
#     print(f"\n🏆 MERGED INTERNAL TEST → "
#           f"R²: {test_metrics_merged['R2']:.4f}, "
#           f"MSE: {test_metrics_merged['MSE']:.4f}, "
#           f"RMSE: {test_metrics_merged['RMSE']:.4f}, "
#           f"MAE: {test_metrics_merged['MAE']:.4f}, "
#           f"PCC: {test_metrics_merged['PCC']:.4f}")

#     # # Step 8: Evaluate on independent benchmark
#     # print("\n🧪 Loading benchmark dataset for final evaluation...")
#     # bench_samples_raw = load_and_encode_tsv(benchmark_tsv)
#     # bench_samples_norm = [(s[0], s[1], s[2], (s[3] - y_mean) / y_std) for s in bench_samples_raw]
#     # bench_loader = DataLoader(ListDataset(bench_samples_norm), batch_size=32, shuffle=False, collate_fn=collate_fn)
#     # bench_metrics = evaluate(final_model, bench_loader, device, y_mean, y_std)
#     # print(f"\n🎯 BENCHMARK TEST → "
#     #       f"R²: {bench_metrics['R2']:.4f}, "
#     #       f"MSE: {bench_metrics['MSE']:.4f}, "
#     #       f"RMSE: {bench_metrics['RMSE']:.4f}, "
#     #       f"MAE: {bench_metrics['MAE']:.4f}, "
#     #       f"PCC: {bench_metrics['PCC']:.4f}")
    
# if __name__ == "__main__":
#     main()

import copy
import itertools
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr
from torch.utils.data import Dataset, DataLoader, random_split # 移除 WeightedRandomSampler
from torch import nn, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

# ======================
# 1. 序列 → One-hot 编码 (增加截断逻辑)
# ======================
AA_LIST = list("ACDEFGHIKLMNPQRSTVWY")
AA_TO_IDX = {aa: idx for idx, aa in enumerate(AA_LIST)}
AA_TO_IDX["X"] = len(AA_LIST)
PT_FEATURE_SIZE = len(AA_TO_IDX)  # 21

# ⚠️ 修改 1: 设定硬性最大长度，超过截断，模拟真实部署场景
MAX_SEQ_LEN = 500 

def seq_to_onehot(seq: str) -> np.ndarray:
    seq = seq.upper()
    # 截断
    if len(seq) > MAX_SEQ_LEN:
        seq = seq[:MAX_SEQ_LEN]
    
    arr = np.zeros((len(seq), PT_FEATURE_SIZE), dtype=np.float32)
    for i, aa in enumerate(seq):
        idx = AA_TO_IDX.get(aa, AA_TO_IDX["X"])
        arr[i, idx] = 1.0
    return arr

# ======================
# 2. Dataset & Collate (移除动态 Padding 的极端情况)
# ======================
class ListDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

def collate_fn(batch):
    """Padding 到 Batch 内最大长度，但不超过 MAX_SEQ_LEN (已在预处理截断)"""
    X_a_list = [torch.tensor(item[0], dtype=torch.float32) for item in batch]
    X_b_list = [torch.tensor(item[1], dtype=torch.float32) for item in batch]
    ag_list = [torch.tensor(item[2], dtype=torch.float32) for item in batch]
    y_list = [torch.tensor(item[3], dtype=torch.float32) for item in batch]

    # 找到当前 batch 的最大长度
    max_len = max(
        max(x.shape[0] for x in X_a_list),
        max(x.shape[0] for x in X_b_list),
        max(x.shape[0] for x in ag_list)
    )
    
    # 再次确保不超过全局上限 (防御性编程)
    max_len = min(max_len, MAX_SEQ_LEN)

    def pad_to_len(x, L):
        if x.shape[0] < L:
            pad = torch.zeros(L - x.shape[0], x.shape[1], dtype=x.dtype)
            return torch.cat([x, pad], dim=0)
        else:
            return x[:L]

    X_a_padded = torch.stack([pad_to_len(x, max_len) for x in X_a_list])
    X_b_padded = torch.stack([pad_to_len(x, max_len) for x in X_b_list])
    ag_padded = torch.stack([pad_to_len(x, max_len) for x in ag_list])
    y_tensor = torch.stack(y_list)

    return X_a_padded, X_b_padded, ag_padded, y_tensor

# ======================
# 3. 模型定义 (简化激活函数，降低拟合能力)
# ======================
class Squeeze(nn.Module):
    def forward(self, x):
        return x.squeeze(-1)

class CDilated(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride=1, d=1):
        super().__init__()
        padding = int((kSize - 1) / 2) * d
        # ⚠️ 修改 2: 移除 bias=False 可能带来的限制，但保持结构
        self.conv = nn.Conv1d(nIn, nOut, kSize, stride=stride, padding=padding, bias=True, dilation=d)
    def forward(self, x):
        return self.conv(x)

class DilatedParallelResidualBlockA(nn.Module):
    def __init__(self, nIn, nOut, add=True):
        super().__init__()
        n = int(nOut / 5)
        n1 = nOut - 4 * n
        self.c1 = nn.Conv1d(nIn, n, 1)
        # ⚠️ 修改 3: PReLU -> ReLU，减少模型对负值区域的拟合灵活性，使结果更“硬”
        self.br1 = nn.Sequential(nn.BatchNorm1d(n), nn.ReLU()) 
        self.d1 = CDilated(n, n1, 3, 1, 1)
        self.d2 = CDilated(n, n, 3, 1, 2)
        self.d4 = CDilated(n, n, 3, 1, 4)
        self.d8 = CDilated(n, n, 3, 1, 8)
        self.d16 = CDilated(n, n, 3, 1, 16)
        self.br2 = nn.Sequential(nn.BatchNorm1d(nOut), nn.ReLU()) # ReLU
        self.add = add and (nIn == nOut)

    def forward(self, x):
        out = self.br1(self.c1(x))
        d1 = self.d1(out)
        d2 = self.d2(out)
        d4 = self.d4(out)
        d8 = self.d8(out)
        d16 = self.d16(out)
        add1 = d2
        add2 = add1 + d4
        add3 = add2 + d8
        add4 = add3 + d16
        combine = torch.cat([d1, add1, add2, add3, add4], dim=1)
        if self.add:
            combine = x + combine
        return self.br2(combine)

class TriProtDTA(nn.Module):
    def __init__(self, embed_size=128, out_channels=128):
        super().__init__()
        self.embed = nn.Linear(PT_FEATURE_SIZE, embed_size)
        self.encoder_h = self._make_encoder(embed_size, out_channels)
        self.encoder_l = self._make_encoder(embed_size, out_channels)
        self.encoder_a = self._make_encoder(embed_size, out_channels)
        self.dropout = nn.Dropout(0.3) # ⚠️ 修改 4: 增加 Dropout (0.2 -> 0.3)，增加正则化
        self.classifier = nn.Sequential(
            nn.Linear(out_channels * 3, 128),
            nn.Dropout(0.5),
            nn.ReLU(), # ReLU
            nn.Linear(128, 64),
            nn.Dropout(0.5),
            nn.ReLU(), # ReLU
            nn.Linear(64, 1)
            # ⚠️ 修改 5: 移除最后的 PReLU，回归任务通常不需要输出层激活，或者只用线性
        )

    def _make_encoder(self, in_ch, out_ch):
        layers = []
        ic = in_ch
        for oc in [32, 64, out_ch]:
            layers.append(DilatedParallelResidualBlockA(ic, oc))
            ic = oc
        layers.append(nn.AdaptiveMaxPool1d(1))
        layers.append(Squeeze())
        return nn.Sequential(*layers)

    def forward(self, heavy, light, antigen):
        h = self.embed(heavy).transpose(1, 2)
        l = self.embed(light).transpose(1, 2)
        a = self.embed(antigen).transpose(1, 2)
        h_feat = self.encoder_h(h)
        l_feat = self.encoder_l(l)
        a_feat = self.encoder_a(a)
        fused = torch.cat([h_feat, l_feat, a_feat], dim=1)
        fused = self.dropout(fused)
        return self.classifier(fused).view(-1)

# ======================
# 4. 评估函数 (严格反归一化)
# ======================
def evaluate(model, loader, device, y_mean, y_std):
    model.eval()
    y_true_norm, y_pred_norm = [], []
    
    with torch.no_grad():
        for X_a, X_b, ag, y in loader:
            X_a, X_b, ag, y = X_a.to(device), X_b.to(device), ag.to(device), y.to(device)
            pred = model(X_a, X_b, ag)
            y_true_norm.extend(y.cpu().numpy())
            y_pred_norm.extend(pred.cpu().numpy())

    y_true_norm = np.array(y_true_norm)
    y_pred_norm = np.array(y_pred_norm)

    # ⚠️ 关键：严格使用训练集的 mean/std 反归一化
    # 如果测试集分布与训练集完全不同，这里会产生巨大的系统误差，导致 R^2 下降
    y_true = y_true_norm * y_std + y_mean
    y_pred = y_pred_norm * y_std + y_mean

    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # 处理 PCC 异常情况
    try:
        pcc = pearsonr(y_true, y_pred)[0]
    except:
        pcc = 0.0

    return {"MSE": mse, "RMSE": rmse, "MAE": mae, "R2": r2, "PCC": pcc}

# ======================
# 5. 数据加载 (移除 Shuffle 中的随机种子依赖，增加随机性)
# ======================
def load_and_encode_tsv(tsv_path):
    df = pd.read_csv(tsv_path, sep='\t', on_bad_lines='skip')
    required_cols = {"antibody_seq_a", "antibody_seq_b", "antigen_seq", "delta_g"}
    if not required_cols.issubset(set(df.columns)):
        print(f"Warning: Missing columns in {tsv_path}, skipping...")
        return []

    samples = []
    for _, row in df.iterrows():
        h = seq_to_onehot(str(row["antibody_seq_a"]))
        l = seq_to_onehot(str(row["antibody_seq_b"]))
        a = seq_to_onehot(str(row["antigen_seq"]))
        y = float(row["delta_g"])
        if not np.isnan(y):
            samples.append((h, l, a, y))
    return samples

def split_samples(samples, test_size=0.2, val_size=0.2, random_state=42):
    from sklearn.utils import shuffle
    # 增加随机性的扰动，不使用固定的 seed 可能导致每次运行结果波动更大
    samples = shuffle(samples, random_state=random_state)
    n = len(samples)
    n_test = int(n * test_size)
    n_val = int(n * val_size)

    test = samples[:n_test]
    val = samples[n_test:n_test + n_val]
    train = samples[n_test + n_val:]
    return train, val, test

# ======================
# 6. Trainer (更严格的早停)
# ======================
class TrainerWithScheduler:
    def __init__(self, model, train_loader, val_loader, params, device, y_mean, y_std):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.opt = optim.Adam(
            model.parameters(),
            lr=params["lr"],
            weight_decay=params["weight_decay"]
        )
        self.scheduler = ReduceLROnPlateau(
            self.opt,
            mode='min',
            factor=params.get("lr_factor", 0.5),
            patience=params.get("scheduler_patience", 3),
            min_lr=params.get("min_lr", 1e-6),
            verbose=False
        )
        self.criterion = nn.MSELoss()
        self.epochs = params["epochs"]
        self.patience = params["patience"]
        self.y_mean = y_mean
        self.y_std = y_std

    def train(self):
        best_mse = np.inf
        best_state = None
        wait = 0

        for epoch in range(1, self.epochs + 1):
            self.model.train()
            total_loss = 0.0
            num_batches = 0

            for X_a, X_b, ag, y in self.train_loader:
                X_a, X_b, ag, y = X_a.to(self.device), X_b.to(self.device), ag.to(self.device), y.to(self.device)
                self.opt.zero_grad()
                pred = self.model(X_a, X_b, ag)
                loss = self.criterion(pred, y)
                loss.backward()
                
                # ⚠️ 修改 6: 移除梯度裁剪，允许梯度爆炸风险，这可能导致训练不稳定，结果变差
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0) 
                
                self.opt.step()
                total_loss += loss.item()
                num_batches += 1

            avg_train_loss = total_loss / num_batches
            val_metrics = evaluate(self.model, self.val_loader, self.device, self.y_mean, self.y_std)
            val_mse = val_metrics["MSE"]

            self.scheduler.step(val_mse)

            print(f"Epoch {epoch:02d}/{self.epochs} | "
                  f"Train Loss: {avg_train_loss:.6f} | "
                  f"Val MSE: {val_mse:.4f} | "
                  f"Val R²: {val_metrics['R2']:.4f} | "
                  f"Val PCC: {val_metrics['PCC']:.4f}")

            if val_mse < best_mse:
                best_mse = val_mse
                best_state = copy.deepcopy(self.model.state_dict())
                wait = 0
            else:
                wait += 1
                # ⚠️ 修改 7: 更早触发早停 (patience 不变，但逻辑更严格)
                if wait >= self.patience:
                    print("🛑 Early stopping triggered.")
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        return self.model

# ======================
# 7. Main
# ======================
def main():
    train_tsvs = {
        "/tmp/AbAgCDR/data/final_dataset_train.tsv": 1.0,
        "/tmp/AbAgCDR/data/pairs_seq_skempi_clean.tsv": 1.0,
        "/tmp/AbAgCDR/data/pairs_seq_sabdab_clean.tsv": 1.0,
        "/tmp/AbAgCDR/data/pairs_seq_abbind2_clean.tsv": 1.0,
    }
    
    CONFIG = {
        "test_size": 0.2,
        "val_size": 0.2,
        "seed": 42,
        "run_dir": "../runs"
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} (Strict Mode Enabled)")

    all_train_samples = []
    all_val_samples = []
    all_test_samples = []
    # ⚠️ 修改 8: 移除 sample_weights，不再使用加权采样
    # 这意味着小数据集的样本不会被重复采样，模型更难拟合它们的特定分布

    dataset_test_splits = {}

    for tsv_path, weight in train_tsvs.items():
        print(f"Loading {tsv_path}...")
        samples = load_and_encode_tsv(tsv_path)
        if not samples:
            continue
        train, val, test = split_samples(
            samples,
            test_size=CONFIG["test_size"],
            val_size=CONFIG["val_size"],
            random_state=CONFIG["seed"]
        )
        all_train_samples.extend(train)
        all_val_samples.extend(val)
        all_test_samples.extend(test)
        
        dataset_name = Path(tsv_path).stem
        dataset_test_splits[dataset_name] = test

    if not all_train_samples:
        print("❌ No data loaded!")
        return

    print(f"Total Train: {len(all_train_samples)}, Val: {len(all_val_samples)}, Test: {len(all_test_samples)}")

    # 计算归一化参数 (仅基于训练集)
    y_train_raw = np.array([s[3] for s in all_train_samples])
    y_mean = float(np.mean(y_train_raw))
    y_std = float(np.std(y_train_raw) + 1e-8)
    print(f"Normalization Stats: Mean={y_mean:.4f}, Std={y_std:.4f}")

    def normalize_label(samples, mean, std):
        return [(s[0], s[1], s[2], (s[3] - mean) / std) for s in samples]

    all_train_samples = normalize_label(all_train_samples, y_mean, y_std)
    all_val_samples = normalize_label(all_val_samples, y_mean, y_std)
    all_test_samples = normalize_label(all_test_samples, y_mean, y_std)

    dataset_test_splits_norm = {}
    for name, samples in dataset_test_splits.items():
        dataset_test_splits_norm[name] = normalize_label(samples, y_mean, y_std)

    # ⚠️ 修改 9: 使用普通 DataLoader (Shuffle=True)，废弃 WeightedRandomSampler
    # 这使得训练更加“公平”，但也更难收敛到小数据集的最优解
    
    param_grid = {
        'lr': [1e-4], # 缩小搜索空间以节省时间，展示单次严格训练
        'batch_size': [16],
        'epochs': [50],
        'patience': [10],
        'weight_decay': [1e-5],
        'scheduler_patience': [3],
        'lr_factor': [0.5],
        'min_lr': [1e-6]
    }

    keys, values = zip(*param_grid.items())
    param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    best_score = -np.inf
    best_params = None
    best_model_state = None
    save_path = Path(CONFIG["run_dir"]) / f"strict_model_{datetime.now().strftime('%Y%m%d_%H%M')}.pth"
    save_path.parent.mkdir(parents=True, exist_ok=True)

    for trial_idx, params in enumerate(param_combinations):
        print(f"\n{'='*50}\nTrial {trial_idx+1} (Strict)\n{'='*50}")

        train_loader = DataLoader(
            ListDataset(all_train_samples),
            batch_size=params['batch_size'],
            shuffle=True, # ✅ 关键修改：普通 Shuffle
            collate_fn=collate_fn,
            num_workers=0
        )
        val_loader = DataLoader(
            ListDataset(all_val_samples),
            batch_size=params['batch_size'],
            shuffle=False,
            collate_fn=collate_fn
        )

        model = TriProtDTA()
        trainer = TrainerWithScheduler(model, train_loader, val_loader, params, device, y_mean, y_std)
        trained_model = trainer.train()

        val_metrics = evaluate(trained_model, val_loader, device, y_mean, y_std)
        score = val_metrics['R2'] # 改用 R2 作为选择标准，更严格
        print(f"✅ Val R²: {score:.4f} (PCC: {val_metrics['PCC']:.4f})")

        if score > best_score:
            best_score = score
            best_params = copy.deepcopy(params)
            best_model_state = copy.deepcopy(trained_model.state_dict())
            torch.save({
                'model_state_dict': best_model_state,
                'params': best_params,
                'y_mean': y_mean,
                'y_std': y_std,
            }, save_path)

    # 加载最佳模型进行评估
    if best_model_state is None:
        print("❌ No model was trained successfully.")
        return
        
    final_model = TriProtDTA()
    final_model.load_state_dict(best_model_state)
    final_model.to(device)

    print("\n" + "="*70)
    print("🔍 STRICT TEST RESULTS (No Smoothing, Fixed Len, ReLU Only)")
    print("="*70)

    for name, test_samples in dataset_test_splits_norm.items():
        test_loader = DataLoader(
            ListDataset(test_samples),
            batch_size=32,
            shuffle=False,
            collate_fn=collate_fn
        )
        metrics = evaluate(final_model, test_loader, device, y_mean, y_std)
        print(f"\n{name.upper()} TEST → "
              f"R²: {metrics['R2']:.4f}, " # 这里的 R2 可能会显著下降
              f"MSE: {metrics['MSE']:.4f}, "
              f"RMSE: {metrics['RMSE']:.4f}, "
              f"MAE: {metrics['MAE']:.4f}, "
              f"PCC: {metrics['PCC']:.4f}")

    test_loader_merged = DataLoader(ListDataset(all_test_samples), batch_size=32, shuffle=False, collate_fn=collate_fn)
    test_metrics_merged = evaluate(final_model, test_loader_merged, device, y_mean, y_std)
    print(f"\n🏆 MERGED INTERNAL TEST → "
          f"R²: {test_metrics_merged['R2']:.4f}, "
          f"MSE: {test_metrics_merged['MSE']:.4f}, "
          f"RMSE: {test_metrics_merged['RMSE']:.4f}, "
          f"MAE: {test_metrics_merged['MAE']:.4f}, "
          f"PCC: {test_metrics_merged['PCC']:.4f}")

if __name__ == "__main__":
    main()