"""
ESM-2 embedding + LightGBM baseline

用法:
  python esm2_lgbm_baseline.py /path/to/dataset.tsv

数据格式:
  tsv 文件，至少包含列: antibody_seq_a, antibody_seq_b, antigen_seq, delta_g
"""

import warnings
warnings.filterwarnings("ignore")

import os
import time
import joblib
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr, spearmanr
import logging
import lightgbm as lgb



import esm

# =====================
# 配置
# =====================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EMB_CACHE = "esm2_embeddings.pkl"
N_SPLITS = 5   # KFold CV

# =====================
# 1. 加载 ESM-2 模型
# =====================
print("🔹 Loading ESM-2 model...")
esm_model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()  # 小模型，快
esm_model = esm_model.to(DEVICE)
batch_converter = alphabet.get_batch_converter()
esm_model.eval()


# =====================
# 2. 序列转 ESM embedding
# =====================
def compute_sequence_embeddings(seqs, batch_size=16):
    embeddings = []
    with torch.no_grad():
        for idx in tqdm(range(0, len(seqs), batch_size), desc="ESM embedding"):
            b = min(batch_size, len(seqs) - idx)
            batch_seqs = [(str(idx + j), seqs[idx + j]) for j in range(b)]
            batch_labels, batch_strs, batch_tokens = batch_converter(batch_seqs)
            batch_tokens = batch_tokens.to(DEVICE)

            out = esm_model(batch_tokens, repr_layers=[6], return_contacts=False)
            token_reps = out["representations"][6]  # (B, L, D)

            # 取每条序列的均值池化 (去掉 cls/eos)
            for i in range(b):
                rep = token_reps[i, 1:len(batch_strs[i]) + 1].mean(0).cpu().numpy()
                embeddings.append(rep)
    return np.array(embeddings)


# =====================
# 3. 读取 & 构造 embedding
# =====================
def build_or_load_embeddings(data_path, cache_path=EMB_CACHE, force_recompute=False):
    if os.path.exists(cache_path) and not force_recompute:
        print(f"✅ Loading cached embeddings from {cache_path}")
        data = joblib.load(cache_path)
        return data["emb_a"], data["emb_b"], data["emb_ag"], data["y"], data["df"]

    print("🔹 Computing embeddings from scratch...")
    df = pd.read_csv(data_path, sep="\t")
    seq_a = df["antibody_seq_a"].values
    seq_b = df["antibody_seq_b"].values
    seq_ag = df["antigen_seq"].values
    y = df["delta_g"].values

    emb_a = compute_sequence_embeddings(seq_a)
    emb_b = compute_sequence_embeddings(seq_b)
    emb_ag = compute_sequence_embeddings(seq_ag)

    joblib.dump({"emb_a": emb_a, "emb_b": emb_b, "emb_ag": emb_ag, "y": y, "df": df}, cache_path)
    print(f"✅ Saved embeddings to {cache_path}")
    return emb_a, emb_b, emb_ag, y, df


# =====================
# 4. LightGBM 训练 + CV
# =====================
def train_evaluate_lgb(emb_a, emb_b, emb_ag, y, n_splits=5):
    X = np.concatenate([emb_a, emb_b, emb_ag], axis=1)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    y_preds, y_trues = [], []
    models = []![](AB_Bind散点图.png)

    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        print(f"🔹 Fold {fold+1}/{n_splits}")
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = lgb.LGBMRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=7,
            min_data_in_leaf=2,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], eval_metric="l2")  # 去掉 verbose

        preds = model.predict(X_val)
        y_preds.extend(preds)
        y_trues.extend(y_val)
        models.append(model)

    # 评估
    mse = mean_squared_error(y_trues, y_preds)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_trues, y_preds)
    r2 = r2_score(y_trues, y_preds)
    pearson, _ = pearsonr(y_trues, y_preds)
    spearman, _ = spearmanr(y_trues, y_preds)

    metrics = {
        "MSE": mse, "RMSE": rmse, "MAE": mae,
        "R²": r2, "Pearson": pearson, "Spearman": spearman
    }
    return models, scaler, (y_trues, y_preds), metrics



# =====================
# 5. 主流程
# =====================
if __name__ == "__main__":
    tsv_path = "/tmp/AbAgCDR/data/final_dataset_train.tsv"
    start = time.time()

    emb_a, emb_b, emb_ag, y, df = build_or_load_embeddings(tsv_path, cache_path=EMB_CACHE, force_recompute=False)
    models, scaler, (y_trues, y_preds), metrics = train_evaluate_lgb(emb_a, emb_b, emb_ag, y, n_splits=N_SPLITS)

    # 保存 pipeline
    joblib.dump({'models': models, 'scaler': scaler, 'emb_cache': EMB_CACHE}, "esm2_lgbm_pipeline.pkl")
    print("✅ Saved pipeline -> esm2_lgbm_pipeline.pkl")

    print("回归性能指标:")
    for k, v in metrics.items():
        print(f"{k} = {v:.4f}")

    # 可视化
    plt.figure(figsize=(6, 6))
    plt.scatter(y_trues, y_preds, alpha=0.6, edgecolors="k")
    plt.plot([min(y_trues), max(y_trues)], [min(y_trues), max(y_trues)], "r--")
    plt.xlabel("True ΔG")
    plt.ylabel("Predicted ΔG")
    plt.title("ESM-2 + LightGBM Baseline")
    plt.tight_layout()
    plt.show()

    print(f"⏱ Total time = {time.time()-start:.1f}s")

