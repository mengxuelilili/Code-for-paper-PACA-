import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split
import copy
import itertools
from datetime import datetime
from pathlib import Path

# ==============================
# Simplified Model (ESM only)
# ==============================
class TriProtDTA(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(480 * 3, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.activation = nn.Sigmoid()

    def forward(self, l_emb, h_emb, ag_emb):
        x = torch.cat([l_emb, h_emb, ag_emb], dim=1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return self.activation(x)

# ==============================
# Dataset using cached embeddings
# ==============================
class ListDataset(Dataset):
    def __init__(self, indices, l_embs, h_embs, ag_embs, labels):
        self.indices = indices
        self.l_embs = l_embs
        self.h_embs = h_embs
        self.ag_embs = ag_embs
        self.labels = labels

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        return {
            'l_emb': torch.tensor(self.l_embs[real_idx], dtype=torch.float32),
            'h_emb': torch.tensor(self.h_embs[real_idx], dtype=torch.float32),
            'ag_emb': torch.tensor(self.ag_embs[real_idx], dtype=torch.float32),
            'label': torch.tensor([self.labels[real_idx]], dtype=torch.float32)
        }

def collate_fn(batch):
    return {
        'l_emb': torch.stack([item['l_emb'] for item in batch]),
        'h_emb': torch.stack([item['h_emb'] for item in batch]),
        'ag_emb': torch.stack([item['ag_emb'] for item in batch]),
        'label': torch.stack([item['label'] for item in batch])
    }

def evaluate(model, dataloader, device, y_mean, y_std):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in dataloader:
            l = batch['l_emb'].to(device)
            h = batch['h_emb'].to(device)
            ag = batch['ag_emb'].to(device)
            labels = batch['label'].to(device)
            preds = model(l, h, ag)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_preds = np.concatenate(all_preds).flatten()
    all_labels = np.concatenate(all_labels).flatten()
    all_preds = all_preds * y_std + y_mean
    all_labels = all_labels * y_std + y_mean

    mse = mean_squared_error(all_labels, all_preds)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(all_labels, all_preds)
    r2 = r2_score(all_labels, all_preds)
    pcc = pearsonr(all_labels, all_preds)[0]
    return {'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'R2': r2, 'PCC': pcc}

def split_samples(samples, test_size=0.2, val_size=0.2, random_state=42):
    """Split into train/val/test"""
    train_val, test = train_test_split(samples, test_size=test_size, random_state=random_state)
    train, val = train_test_split(train_val, test_size=val_size/(1-test_size), random_state=random_state)
    return train, val, test

def load_and_encode_tsv(tsv_path):
    df = pd.read_csv(tsv_path, sep='\t', on_bad_lines='skip')
    required_cols = ['antibody_seq_a', 'antibody_seq_b', 'antigen_seq', 'delta_g']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing column '{col}' in {tsv_path}")
    df = df[(df['delta_g'] >= -20) & (df['delta_g'] <= 5)]
    samples = []
    for _, row in df.iterrows():
        a = str(row['antibody_seq_a'])
        b = str(row['antibody_seq_b'])
        ag = str(row['antigen_seq'])
        dg = float(row['delta_g'])
        if a and b and ag and not pd.isna(dg):
            samples.append((a, b, ag, dg))
    return samples

class TrainerWithScheduler:
    def __init__(self, model, train_loader, val_loader, params, device, y_mean, y_std):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.params = params
        self.device = device
        self.y_mean = y_mean
        self.y_std = y_std
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=params['lr'],
            weight_decay=params['weight_decay']
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=params['lr_factor'],
            patience=params['scheduler_patience'],
            min_lr=params['min_lr']
        )
        self.criterion = nn.MSELoss()
        self.patience = params['patience']

    def train(self):
        best_score = -np.inf
        patience_counter = 0
        for epoch in range(self.params['epochs']):
            self.model.train()
            total_loss = 0
            for batch in self.train_loader:
                l = batch['l_emb'].to(self.device)
                h = batch['h_emb'].to(self.device)
                ag = batch['ag_emb'].to(self.device)
                labels = batch['label'].to(self.device)
                self.optimizer.zero_grad()
                preds = self.model(l, h, ag)
                loss = self.criterion(preds, labels)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

            val_metrics = evaluate(self.model, self.val_loader, self.device, self.y_mean, self.y_std)
            self.scheduler.step(val_metrics['PCC'])

            if val_metrics['PCC'] > best_score:
                best_score = val_metrics['PCC']
                patience_counter = 0
                best_model = copy.deepcopy(self.model.state_dict())
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    break

        self.model.load_state_dict(best_model)
        return self.model

# ✅ 新增：为 benchmark 实时计算 ESM 嵌入（仅用于最终测试）
def get_esm_embedding(seq):
    import esm
    # 静态加载模型（只加载一次）
    if not hasattr(get_esm_embedding, "model"):
        print("🧠 Loading ESM-2 for benchmark evaluation...")
        get_esm_embedding.model, get_esm_embedding.alphabet = esm.pretrained.esm2_t12_35M_UR50D()
        get_esm_embedding.batch_converter = get_esm_embedding.alphabet.get_batch_converter()
        get_esm_embedding.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        get_esm_embedding.model = get_esm_embedding.model.to(get_esm_embedding.device)
        get_esm_embedding.model.eval()
    
    seq = str(seq).upper().strip()
    valid_aas = "ARNDCQEGHILKMFPSTWYV"
    seq = ''.join([aa for aa in seq if aa in valid_aas])
    if len(seq) == 0:
        seq = "A"
    data = [("protein", seq)]
    _, _, batch_tokens = get_esm_embedding.batch_converter(data)
    batch_tokens = batch_tokens.to(get_esm_embedding.device)
    with torch.no_grad():
        results = get_esm_embedding.model(batch_tokens, repr_layers=[12])
        token_repr = results["representations"][12]
    emb = token_repr[0, 1:-1].mean(dim=0).cpu().numpy()
    return emb

def main():
    # =============== 配置 ===============
    train_tsvs = {
        "/tmp/AbAgCDR/data/final_dataset_train.tsv": 1.0,
        "/tmp/AbAgCDR/data/pairs_seq_skempi_clean.tsv": 1.0,
        "/tmp/AbAgCDR/data/pairs_seq_sabdab_clean.tsv": 1.0,
        "/tmp/AbAgCDR/data/pairs_seq_abbind2_clean.tsv": 1.0,
    }
    # benchmark_tsv = "/tmp/AbAgCDR/data/pairs_seq_benchmark1.tsv"
    CONFIG = {
        "test_size": 0.2,
        "val_size": 0.2,
        "seed": 42,
        "run_dir": "/tmp/AbAgCDR/runs"
    }
    # ===================================

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Step 0: Load cached embeddings
    CACHE_DIR = Path("/tmp/AbAgCDR/embeddings")
    l_embs = np.load(CACHE_DIR / "lchain.npy")
    h_embs = np.load(CACHE_DIR / "hchain.npy")
    ag_embs = np.load(CACHE_DIR / "ag.npy")
    labels = np.load(CACHE_DIR / "labels.npy")
    sources = np.load(CACHE_DIR / "sources.npy")

    # Map source name to indices
    source_to_indices = {}
    for i, src in enumerate(sources):
        if src not in source_to_indices:
            source_to_indices[src] = []
        source_to_indices[src].append(i)

    # Step 1: Split each dataset internally
    all_train_indices = []
    all_val_indices = []
    all_test_indices = []
    sample_weights = []
    dataset_test_splits = {}

    for tsv_path, weight in train_tsvs.items():
        src_name = Path(tsv_path).stem
        if src_name not in source_to_indices:
            continue
        indices = source_to_indices[src_name]
        train_idx, val_idx, test_idx = split_samples(
            indices,
            test_size=CONFIG["test_size"],
            val_size=CONFIG["val_size"],
            random_state=CONFIG["seed"]
        )
        all_train_indices.extend(train_idx)
        all_val_indices.extend(val_idx)
        all_test_indices.extend(test_idx)
        sample_weights.extend([weight] * len(train_idx))
        dataset_test_splits[src_name] = test_idx

    print(f"Total Train: {len(all_train_indices)}, Val: {len(all_val_indices)}, Test: {len(all_test_indices)}")

    # Step 2: Normalize labels
    y_train = labels[all_train_indices]
    y_mean, y_std = y_train.mean(), y_train.std() + 1e-8

    # Step 3: Weighted sampler
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    # Step 4: Hyperparameter grid
    param_grid = {
        'lr': [1e-4],
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
    save_path = Path(CONFIG["run_dir"]) / f"best_model_{datetime.now().strftime('%Y%m%d_%H%M')}.pth"
    save_path.parent.mkdir(parents=True, exist_ok=True)

    for trial_idx, params in enumerate(param_combinations):
        print(f"\n{'='*50}\nTrial {trial_idx+1}/{len(param_combinations)} | Params: {params}\n{'='*50}")

        train_loader = DataLoader(
            ListDataset(all_train_indices, l_embs, h_embs, ag_embs, labels),
            batch_size=params['batch_size'],
            sampler=sampler,
            collate_fn=collate_fn,
            shuffle=False
        )
        val_loader = DataLoader(
            ListDataset(all_val_indices, l_embs, h_embs, ag_embs, labels),
            batch_size=params['batch_size'],
            shuffle=False,
            collate_fn=collate_fn
        )

        model = TriProtDTA()
        trainer = TrainerWithScheduler(model, train_loader, val_loader, params, device, y_mean, y_std)
        trained_model = trainer.train()

        val_metrics = evaluate(trained_model, val_loader, device, y_mean, y_std)
        score = val_metrics['PCC']
        print(f"✅ Val PCC: {score:.4f}")

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
            print(f"🎉 Best model saved at {save_path}")

    # Step 5: Load best model
    final_model = TriProtDTA()
    final_model.load_state_dict(best_model_state)
    final_model.to(device)

    # Step 6: Evaluate on each individual dataset's test split
    print("\n" + "="*70)
    print("🔍 INDIVIDUAL DATASET TEST RESULTS (on their own held-out test sets)")
    print("="*70)

    for name, test_indices in dataset_test_splits.items():
        test_loader = DataLoader(
            ListDataset(test_indices, l_embs, h_embs, ag_embs, labels),
            batch_size=32,
            shuffle=False,
            collate_fn=collate_fn
        )
        metrics = evaluate(final_model, test_loader, device, y_mean, y_std)
        print(f"\n{name.upper()} TEST → "
              f"R²: {metrics['R2']:.4f}, "
              f"MSE: {metrics['MSE']:.4f}, "
              f"RMSE: {metrics['RMSE']:.4f}, "
              f"MAE: {metrics['MAE']:.4f}, "
              f"PCC: {metrics['PCC']:.4f}")

    # Step 7: Evaluate on merged internal test set
    test_loader_merged = DataLoader(
        ListDataset(all_test_indices, l_embs, h_embs, ag_embs, labels),
        batch_size=32,
        shuffle=False,
        collate_fn=collate_fn
    )
    test_metrics_merged = evaluate(final_model, test_loader_merged, device, y_mean, y_std)
    print(f"\n🏆 MERGED INTERNAL TEST → "
          f"R²: {test_metrics_merged['R2']:.4f}, "
          f"MSE: {test_metrics_merged['MSE']:.4f}, "
          f"RMSE: {test_metrics_merged['RMSE']:.4f}, "
          f"MAE: {test_metrics_merged['MAE']:.4f}, "
          f"PCC: {test_metrics_merged['PCC']:.4f}")

    # # Step 8: Evaluate on independent benchmark
    # print("\n🧪 Loading benchmark dataset for final evaluation...")
    # bench_samples_raw = load_and_encode_tsv(benchmark_tsv)
    # bench_l, bench_h, bench_ag, bench_y = [], [], [], []
    # for a, b, ag, dg in bench_samples_raw:
    #     try:
    #         l_emb = get_esm_embedding(a)
    #         h_emb = get_esm_embedding(b)
    #         ag_emb = get_esm_embedding(ag)
    #         bench_l.append(l_emb)
    #         bench_h.append(h_emb)
    #         bench_ag.append(ag_emb)
    #         bench_y.append(dg)
    #     except Exception as e:
    #         print(f"⚠️ Skipping invalid sequence: {e}")
    #         continue

    # if bench_l:
    #     bench_l = np.array(bench_l)
    #     bench_h = np.array(bench_h)
    #     bench_ag = np.array(bench_ag)
    #     bench_y_norm = (np.array(bench_y) - y_mean) / y_std
    #     bench_loader = DataLoader(
    #         ListDataset(range(len(bench_y_norm)), bench_l, bench_h, bench_ag, bench_y_norm),
    #         batch_size=32,
    #         shuffle=False,
    #         collate_fn=collate_fn
    #     )
    #     bench_metrics = evaluate(final_model, bench_loader, device, y_mean, y_std)
    #     print(f"\n🎯 BENCHMARK TEST → "
    #           f"R²: {bench_metrics['R2']:.4f}, "
    #           f"MSE: {bench_metrics['MSE']:.4f}, "
    #           f"RMSE: {bench_metrics['RMSE']:.4f}, "
    #           f"MAE: {bench_metrics['MAE']:.4f}, "
    #           f"PCC: {bench_metrics['PCC']:.4f}")
    # else:
    #     print("⚠️ No valid benchmark samples.")

if __name__ == "__main__":
    main()