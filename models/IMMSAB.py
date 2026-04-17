# train.py
# train.py
# import os
# import random
# import numpy as np
# import torch
# import torch.nn as nn
# from torch.utils.data import DataLoader, WeightedRandomSampler
# import itertools
# import copy
# from IMMSABmodels import AntibodyAffinityModel  # ← 使用我提供的 ESM2 + CNN 模型
# from utils import load_dataset, split_dataset, ListDataset, collate_fn  # ← 上面定义的


# def set_seed(seed=22):
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     os.environ['PYTHONHASHSEED'] = str(seed)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False


# def pearson_corr(x, y):
#     x_mean = torch.mean(x)
#     y_mean = torch.mean(y)
#     numerator = torch.sum((x - x_mean) * (y - y_mean))
#     denominator = torch.sqrt(torch.sum((x - x_mean)**2)) * torch.sqrt(torch.sum((y - y_mean)**2))
#     return numerator / (denominator + 1e-8)


# def evaluate(model, loader, device):
#     model.eval()
#     preds, labels = [], []
#     with torch.no_grad():
#         for heavy, light, antigen, y in loader:
#             heavy, light, antigen, y = heavy.to(device), light.to(device), antigen.to(device), y.to(device)
#             pred = model(heavy, light, antigen)
#             preds.append(pred)
#             labels.append(y)
#     preds = torch.cat(preds)
#     labels = torch.cat(labels)
#     pcc = pearson_corr(preds, labels).item()
#     mse = torch.mean((preds - labels) ** 2).item()
#     rmse = np.sqrt(mse)
#     mae = torch.mean(torch.abs(preds - labels)).item()
#     ss_res = torch.sum((labels - preds) ** 2)
#     ss_tot = torch.sum((labels - torch.mean(labels)) ** 2)
#     r2 = (1 - ss_res / (ss_tot + 1e-8)).item()
#     return {"PCC": pcc, "MSE": mse, "RMSE": rmse, "MAE": mae, "R2": r2}


# def main():
#     set_seed(22)
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"Using device: {device}")

#     paths = {
#         "train": "/tmp/AbAgCDR/data/train_data.pt",
#         "abbind": "/tmp/AbAgCDR/data/abbind_data.pt",
#         "sabdab": "/tmp/AbAgCDR/data/sabdab_data.pt",
#         "skempi": "/tmp/AbAgCDR/data/skempi_data.pt"
#     }
#     benchmark_path = "/tmp/AbAgCDR/data/benchmark_data.pt"

#     # Load and split each dataset (6:2:2)
#     all_splits = {}
#     for name, path in paths.items():
#         data = load_dataset(path)
#         all_splits[name] = split_dataset(data)

#     # Load benchmark (no split)
#     benchmark_data = load_dataset(benchmark_path)
#     benchmark_samples = list(zip(
#         benchmark_data["X_a"],
#         benchmark_data["X_b"],
#         benchmark_data["antigen"],
#         benchmark_data["y"]
#     ))
#     benchmark_loader = DataLoader(
#         ListDataset(benchmark_samples),
#         batch_size=32,
#         shuffle=False,
#         collate_fn=collate_fn
#     )

#     # Collect training samples with weights
#     all_train_samples = []
#     sample_weights = []
#     dataset_weights = {'train': 0.7, 'abbind': 0.1, 'sabdab': 0.1, 'skempi': 0.1}

#     for name in paths.keys():
#         tr = all_splits[name]["train"]
#         w = dataset_weights[name]
#         for i in range(len(tr[3])):
#             all_train_samples.append((tr[0][i], tr[1][i], tr[2][i], tr[3][i]))
#             sample_weights.append(w)

#     # Validation: combine all val sets (no weighting)
#     val_samples = []
#     for split in all_splits.values():
#         va = split["val"]
#         for i in range(len(va[3])):
#             val_samples.append((va[0][i], va[1][i], va[2][i], va[3][i]))

#     # Hyperparameter grid
#     param_grid = {
#         'lr': [1e-4, 5e-4],
#         'batch_size': [16, 32],
#         'epochs': [50],
#         'weight_decay': [1e-5]
#     }
#     keys, values = zip(*param_grid.items())
#     param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

#     # Get label scaler from train_data
#     train_data_for_scaler = load_dataset(paths["train"])
#     label_scaler = train_data_for_scaler.get("label_scaler", None)

#     best_score = -np.inf
#     best_params = None
#     best_model_state = None
#     save_path = "/tmp/AbAgCDR/model/best_modelIMMSAB.pth"

#     for trial_idx, params in enumerate(param_combinations):
#         print(f"\n{'='*50}\nTrial {trial_idx+1}/{len(param_combinations)} | Params: {params}\n{'='*50}")

#         # Create loaders
#         train_loader = DataLoader(
#             ListDataset(all_train_samples),
#             batch_size=params['batch_size'],
#             sampler=WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True),
#             collate_fn=collate_fn,
#             shuffle=False
#         )
#         val_loader = DataLoader(
#             ListDataset(val_samples),
#             batch_size=params['batch_size'],
#             shuffle=False,
#             collate_fn=collate_fn
#         )

#         # Model (ESM2 frozen by default)
#         model = AntibodyAffinityModel().to(device)
#         optimizer = torch.optim.AdamW(model.fc.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
#         scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5, factor=0.5, min_lr=1e-6)

#         # Training loop
#         best_val_pcc = -np.inf
#         patience_counter = 0
#         for epoch in range(params['epochs']):
#             model.train()
#             for heavy, light, antigen, y in train_loader:
#                 heavy, light, antigen, y = heavy.to(device), light.to(device), antigen.to(device), y.to(device)
#                 pred = model(heavy, light, antigen)
#                 loss = nn.HuberLoss(delta=1.0)(pred, y)
#                 optimizer.zero_grad()
#                 loss.backward()
#                 optimizer.step()

#             # Validate
#             val_metrics = evaluate(model, val_loader, device)
#             scheduler.step(val_metrics['PCC'])

#             if val_metrics['PCC'] > best_val_pcc:
#                 best_val_pcc = val_metrics['PCC']
#                 patience_counter = 0
#             else:
#                 patience_counter += 1
#                 if patience_counter >= 15:
#                     break

#         print(f"✅ Final Val PCC: {best_val_pcc:.4f}")

#         if best_val_pcc > best_score:
#             best_score = best_val_pcc
#             best_params = copy.deepcopy(params)
#             best_model_state = copy.deepcopy(model.state_dict())
#             torch.save({
#                 'model_state_dict': best_model_state,
#                 'params': best_params,
#                 'label_scaler': label_scaler
#             }, save_path)
#             print(f"🎉 New best model saved at {save_path}")

#     # Final evaluation on all test sets + benchmark
#     print(f"\n🏆 Best hyperparameters: {best_params} | Best Val PCC: {best_score:.4f}")
#     final_model = AntibodyAffinityModel().to(device)
#     final_model.load_state_dict(best_model_state)

#     print("\n" + "="*60)
#     print("🧪 FINAL TEST RESULTS")
#     print("="*60)

#     for name, split in all_splits.items():
#         test_samples = list(zip(split["test"][0], split["test"][1], split["test"][2], split["test"][3]))
#         test_loader = DataLoader(ListDataset(test_samples), batch_size=32, shuffle=False, collate_fn=collate_fn)
#         metrics = evaluate(final_model, test_loader, device)
#         print(f"\n{name.upper()} TEST → R²: {metrics['R2']:.4f},MSE: {metrics['MSE']:.4f}, RMSE: {metrics['RMSE']:.4f}, MAE: {metrics['MAE']:.4f}, PCC: {metrics['PCC']:.4f}")

#     bench_metrics = evaluate(final_model, benchmark_loader, device)
#     print(f"\n🎯 BENCHMARK TEST → R²: {bench_metrics['R2']:.4f}, MSE: {bench_metrics['MSE']:.4f}, RMSE: {bench_metrics['RMSE']:.4f}, MAE: {bench_metrics['MAE']:.4f}, PCC: {bench_metrics['PCC']:.4f}")


# if __name__ == "__main__":
#     main()

"""
IMMSAB 训练脚本 - TSV 版本
✅ 直接读取 TSV 文件 (氨基酸序列)
✅ 自动使用 ESM2 生成嵌入特征
✅ 标签标准化 (StandardScaler)
✅ 保存标准化器供预测使用
✅ 修复：嵌入数据形状 (Batch, 1, 532)
"""

import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import itertools
import copy
from tqdm import tqdm

# === 导入 ESM2 ===
import esm

# === 导入模型 ===
from IMMSABmodels import AntibodyAffinityModel

# === 配置 ===
DATA_DIR = "/tmp/AbAgCDR/data"
MODEL_SAVE_PATH = "/tmp/AbAgCDR/model/best_modelIMMSAB2.pth"
SCALER_SAVE_PATH = "/tmp/AbAgCDR/model/label_scaler2.pkl"

# TSV 文件路径
TSV_FILES = {
    'train': f"{DATA_DIR}/final_dataset_train.tsv",
    'abbind': f"{DATA_DIR}/pairs_seq_abbind2.tsv",
    'sabdab': f"{DATA_DIR}/pairs_seq_sabdab.tsv",
    'skempi': f"{DATA_DIR}/pairs_seq_skempi.tsv",
}
BENCHMARK_TSV = f"{DATA_DIR}/pairs_seq_benchmark1.tsv"

# === 随机种子 ===
def set_seed(seed=22):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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
        """获取单条序列的嵌入 (Mean Pooling, 1280 维)"""
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
        
        # Mean pooling (去掉 BOS 和 EOS)
        emb = token_repr[0, 1:-1].mean(dim=0).cpu().numpy()
        return emb
    
    def embed_batch(self, sequences, target_dim):
        """
        批量提取嵌入并调整到目标维度
        
        ⚠️ 关键修复：
        模型期望输入形状：(Batch, SeqLen, FeatureDim)
        其中 FeatureDim = 532 (heavy/light) 或 500 (antigen)
        SeqLen = 1 (因为我们对整个序列做了 mean pooling)
        
        模型 forward 中会做 permute(0, 2, 1):
        (B, 1, 532) → (B, 532, 1) → Conv1d(532→64) → (B, 64, 1)
        """
        embeddings = []
        for seq in tqdm(sequences, desc="Extracting embeddings", leave=False):
            emb = self.get_embedding(seq)
            embeddings.append(emb)
        
        embeddings = np.array(embeddings)  # (N, 1280)
        
        # 维度调整：1280 → target_dim (532 或 500)
        if embeddings.shape[1] > target_dim:
            embeddings = embeddings[:, :target_dim]
        elif embeddings.shape[1] < target_dim:
            pad = np.zeros((embeddings.shape[0], target_dim - embeddings.shape[1]))
            embeddings = np.concatenate([embeddings, pad], axis=1)
        
        # 👈 关键修复：形状改为 (N, 1, target_dim)
        # 这样 permute(0, 2, 1) 后变成 (N, target_dim, 1)，符合 Conv1d 输入要求
        return embeddings[:, np.newaxis, :].astype(np.float32)


# === 加载 TSV 数据 ===
def load_tsv_data(path):
    """加载 TSV 文件并提取序列和标签"""
    print(f"📥 加载：{path}")
    df = pd.read_csv(path, sep='\t')
    
    required = ['antibody_seq_a', 'antibody_seq_b', 'antigen_seq', 'delta_g']
    for col in required:
        if col not in df.columns:
            raise ValueError(f"缺少列：{col}")
    
    seq_h = df['antibody_seq_a'].tolist()
    seq_l = df['antibody_seq_b'].tolist()
    seq_a = df['antigen_seq'].tolist()
    y = df['delta_g'].values.astype(np.float32)
    
    print(f"   样本数：{len(df)}")
    return seq_h, seq_l, seq_a, y


# === 数据划分 (6:2:2) ===
def split_data(X_h, X_l, X_a, y, test_size=0.2, val_size=0.25, random_state=42):
    """划分 train/val/test (6:2:2)"""
    idx = np.arange(len(y))
    train_val_idx, test_idx = train_test_split(idx, test_size=test_size, random_state=random_state)
    train_idx, val_idx = train_test_split(train_val_idx, test_size=val_size, random_state=random_state)
    
    def get_subset(idx):
        return [X_h[i] for i in idx], [X_l[i] for i in idx], [X_a[i] for i in idx], y[idx]
    
    return {
        'train': get_subset(train_idx),
        'val': get_subset(val_idx),
        'test': get_subset(test_idx)
    }


# === 数据集类 ===
class ListDataset(torch.utils.data.Dataset):
    def __init__(self, samples):
        self.samples = samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        h, l, a, y = self.samples[idx]
        return {
            'heavy': torch.tensor(h, dtype=torch.float32),
            'light': torch.tensor(l, dtype=torch.float32),
            'antigen': torch.tensor(a, dtype=torch.float32),
            'delta_g': torch.tensor(y, dtype=torch.float32)
        }


def collate_fn(batch):
    h = torch.stack([x['heavy'] for x in batch])
    l = torch.stack([x['light'] for x in batch])
    a = torch.stack([x['antigen'] for x in batch])
    y = torch.stack([x['delta_g'] for x in batch])
    return h, l, a, y


# === 评估函数 ===
def pearson_corr(x, y):
    x_mean = torch.mean(x)
    y_mean = torch.mean(y)
    numerator = torch.sum((x - x_mean) * (y - y_mean))
    denominator = torch.sqrt(torch.sum((x - x_mean)**2)) * torch.sqrt(torch.sum((y - y_mean)**2))
    return numerator / (denominator + 1e-8)


def evaluate(model, loader, device):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for heavy, light, antigen, y in loader:
            heavy, light, antigen, y = heavy.to(device), light.to(device), antigen.to(device), y.to(device)
            pred = model(heavy, light, antigen)
            preds.append(pred)
            labels.append(y)
    
    preds = torch.cat(preds).flatten()
    labels = torch.cat(labels).flatten()
    
    pcc = pearson_corr(preds, labels).item()
    mse = torch.mean((preds - labels) ** 2).item()
    rmse = np.sqrt(mse)
    mae = torch.mean(torch.abs(preds - labels)).item()
    ss_res = torch.sum((labels - preds) ** 2)
    ss_tot = torch.sum((labels - torch.mean(labels)) ** 2)
    r2 = (1 - ss_res / (ss_tot + 1e-8)).item()
    
    return {"PCC": pcc, "MSE": mse, "RMSE": rmse, "MAE": mae, "R2": r2}


# === 主程序 ===
def main():
    set_seed(22)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  使用设备：{device}")
    
    # 1. 初始化 ESM2 嵌入器
    embedder = ESM2Embedder()
    
    # 2. 加载并处理所有数据集
    print("\n" + "="*60)
    print("📊 加载并处理数据")
    print("="*60)
    
    all_data = {}
    all_splits = {}
    
    for name, path in TSV_FILES.items():
        if not os.path.exists(path):
            print(f"⚠️  跳过 {name}: 文件不存在")
            continue
        
        print(f"\n处理 {name}...")
        seq_h, seq_l, seq_a, y = load_tsv_data(path)
        
        # 生成嵌入
        print(f"   生成 ESM2 嵌入...")
        X_h = embedder.embed_batch(seq_h, target_dim=532)
        X_l = embedder.embed_batch(seq_l, target_dim=532)
        X_a = embedder.embed_batch(seq_a, target_dim=500)
        
        print(f"   嵌入形状：H={X_h.shape}, L={X_l.shape}, A={X_a.shape}")
        
        all_data[name] = {'X_h': X_h, 'X_l': X_l, 'X_a': X_a, 'y': y}
        
        # 划分数据集
        splits = split_data(X_h, X_l, X_a, y)
        all_splits[name] = splits
        
        print(f"   划分：train={len(splits['train'][0])}, val={len(splits['val'][0])}, test={len(splits['test'][0])}")
    
    # 3. 处理 Benchmark (不划分)
    print(f"\n处理 Benchmark...")
    if os.path.exists(BENCHMARK_TSV):
        seq_h, seq_l, seq_a, y = load_tsv_data(BENCHMARK_TSV)
        X_h = embedder.embed_batch(seq_h, target_dim=532)
        X_l = embedder.embed_batch(seq_l, target_dim=532)
        X_a = embedder.embed_batch(seq_a, target_dim=500)
        benchmark_data = {'X_h': X_h, 'X_l': X_l, 'X_a': X_a, 'y': y}
    else:
        benchmark_data = None
    
    # 4. 标签标准化 (使用训练集拟合)
    print("\n" + "="*60)
    print("📈 标签标准化")
    print("="*60)
    
    all_train_y = []
    for name, splits in all_splits.items():
        all_train_y.extend(splits['train'][3])
    all_train_y = np.array(all_train_y).reshape(-1, 1)
    
    label_scaler = StandardScaler()
    label_scaler.fit(all_train_y)
    
    print(f"   标准化器均值：{label_scaler.mean_[0]:.4f}")
    print(f"   标准化器标准差：{np.sqrt(label_scaler.var_[0]):.4f}")
    print(f"   原始标签范围：[{all_train_y.min():.4f}, {all_train_y.max():.4f}]")
    
    # 保存标准化器
    import pickle
    os.makedirs(os.path.dirname(SCALER_SAVE_PATH), exist_ok=True)
    with open(SCALER_SAVE_PATH, 'wb') as f:
        pickle.dump(label_scaler, f)
    print(f"   ✅ 标准化器已保存：{SCALER_SAVE_PATH}")
    
    # 5. 准备训练数据 (带权重)
    print("\n" + "="*60)
    print("📦 准备训练数据")
    print("="*60)
    
    all_train_samples = []
    sample_weights = []
    dataset_weights = {'train': 0.7, 'abbind': 0.1, 'sabdab': 0.1, 'skempi': 0.1}
    
    for name, splits in all_splits.items():
        tr = splits['train']
        w = dataset_weights.get(name, 0.1)
        
        # 标准化标签
        y_scaled = label_scaler.transform(tr[3].reshape(-1, 1)).flatten()
        
        for i in range(len(tr[0])):
            all_train_samples.append((tr[0][i], tr[1][i], tr[2][i], y_scaled[i]))
            sample_weights.append(w)
    
    print(f"   总训练样本：{len(all_train_samples)}")
    
    # 验证集
    val_samples = []
    for name, splits in all_splits.items():
        va = splits['val']
        y_scaled = label_scaler.transform(va[3].reshape(-1, 1)).flatten()
        for i in range(len(va[0])):
            val_samples.append((va[0][i], va[1][i], va[2][i], y_scaled[i]))
    
    print(f"   总验证样本：{len(val_samples)}")
    
    # 6. 超参数搜索
    param_grid = {
        'lr': [1e-4, 5e-4],
        'batch_size': [16, 32],
        'epochs': [50],
        'weight_decay': [1e-5]
    }
    keys, values = zip(*param_grid.items())
    param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    print(f"\n   超参数组合：{len(param_combinations)} 种")
    
    # 7. 训练循环
    print("\n" + "="*60)
    print("🏃 开始训练")
    print("="*60)
    
    best_score = -np.inf
    best_params = None
    best_model_state = None
    
    for trial_idx, params in enumerate(param_combinations):
        print(f"\n{'='*50}")
        print(f"Trial {trial_idx+1}/{len(param_combinations)} | Params: {params}")
        print(f"{'='*50}")
        
        # DataLoader
        train_loader = DataLoader(
            ListDataset(all_train_samples),
            batch_size=params['batch_size'],
            sampler=WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True),
            collate_fn=collate_fn,
            shuffle=False
        )
        val_loader = DataLoader(
            ListDataset(val_samples),
            batch_size=params['batch_size'],
            shuffle=False,
            collate_fn=collate_fn
        )
        
        # 模型
        model = AntibodyAffinityModel().to(device)
        optimizer = torch.optim.AdamW(model.fc.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5, factor=0.5, min_lr=1e-6)
        
        # 训练
        best_val_pcc = -np.inf
        patience_counter = 0
        
        for epoch in range(params['epochs']):
            model.train()
            epoch_loss = 0
            for heavy, light, antigen, y in train_loader:
                heavy, light, antigen, y = heavy.to(device), light.to(device), antigen.to(device), y.to(device)
                pred = model(heavy, light, antigen)
                loss = nn.HuberLoss(delta=1.0)(pred, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            # 验证
            val_metrics = evaluate(model, val_loader, device)
            scheduler.step(val_metrics['PCC'])
            
            if (epoch + 1) % 10 == 0:
                print(f"   Epoch {epoch+1}/{params['epochs']} | Loss: {epoch_loss/len(train_loader):.4f} | Val PCC: {val_metrics['PCC']:.4f}")
            
            if val_metrics['PCC'] > best_val_pcc:
                best_val_pcc = val_metrics['PCC']
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= 15:
                    print(f"   ⏹️  早停 (patience={patience_counter})")
                    break
        
        print(f"   ✅ Final Val PCC: {best_val_pcc:.4f}")
        
        # 保存最佳模型
        if best_val_pcc > best_score:
            best_score = best_val_pcc
            best_params = copy.deepcopy(params)
            best_model_state = copy.deepcopy(model.state_dict())
            
            os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
            torch.save({
                'model_state_dict': best_model_state,
                'params': best_params,
                'label_scaler': label_scaler
            }, MODEL_SAVE_PATH)
            print(f"   🎉 新最佳模型已保存：{MODEL_SAVE_PATH}")
    
    # 8. 最终测试
    print(f"\n{'='*60}")
    print(f"🏆 最佳超参数：{best_params} | Best Val PCC: {best_score:.4f}")
    print(f"{'='*60}")
    
    final_model = AntibodyAffinityModel().to(device)
    final_model.load_state_dict(best_model_state)
    
    print("\n" + "="*60)
    print("🧪 FINAL TEST RESULTS")
    print("="*60)
    
    for name, splits in all_splits.items():
        test = splits['test']
        y_scaled = label_scaler.transform(test[3].reshape(-1, 1)).flatten()
        test_samples = [(test[0][i], test[1][i], test[2][i], y_scaled[i]) for i in range(len(test[0]))]
        test_loader = DataLoader(ListDataset(test_samples), batch_size=32, shuffle=False, collate_fn=collate_fn)
        
        metrics = evaluate(final_model, test_loader, device)
        print(f"\n{name.upper()} TEST → PCC: {metrics['PCC']:.4f}, R²: {metrics['R2']:.4f}, RMSE: {metrics['RMSE']:.4f}, MSE: {metrics['MSE']:.4f}, MAE: {metrics['MAE']:.4f}")
    
    if benchmark_data:
        y_scaled = label_scaler.transform(benchmark_data['y'].reshape(-1, 1)).flatten()
        bench_samples = [(benchmark_data['X_h'][i], benchmark_data['X_l'][i], benchmark_data['X_a'][i], y_scaled[i]) 
                         for i in range(len(benchmark_data['y']))]
        bench_loader = DataLoader(ListDataset(bench_samples), batch_size=32, shuffle=False, collate_fn=collate_fn)
        bench_metrics = evaluate(final_model, bench_loader, device)
        print(f"\n🎯 BENCHMARK → PCC: {bench_metrics['PCC']:.4f}, R²: {bench_metrics['R2']:.4f}, RMSE: {bench_metrics['RMSE']:.4f}, MSE: {bench_metrics['MSE']:.4f}, MAE: {bench_metrics['MAE']:.4f}")
    
    print("\n" + "="*60)
    print("✅ 训练完成！")
    print("="*60)
    print(f"\n输出文件:")
    print(f"   📦 模型：{MODEL_SAVE_PATH}")
    print(f"   📦 标准化器：{SCALER_SAVE_PATH}")


if __name__ == "__main__":
    main()