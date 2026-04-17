import warnings

warnings.filterwarnings("ignore")

import os
import time
import joblib
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr, spearmanr
import lightgbm as lgb
import esm

# =====================
# 配置
# =====================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EMB_CACHE = "esm2_embeddings_cdr.pkl"
N_SPLITS = 5  # KFold
ESM_MODEL_NAME = "esm2_t6_8M_UR50D"  # 可以换成 t33_650M_UR50D

# =====================
# 加载 ESM-2 模型
# =====================
print(f"🔹 Loading ESM-2 model: {ESM_MODEL_NAME} ...")
esm_model, alphabet = getattr(esm.pretrained, ESM_MODEL_NAME)()
esm_model = esm_model.to(DEVICE)
esm_model.eval()
batch_converter = alphabet.get_batch_converter()

# 获取模型最大层数
MAX_LAYER = esm_model.num_layers
print(f"ESM model has {MAX_LAYER} layers")
# 使用最后几层
ESM_LAYERS = [max(1, MAX_LAYER - 2), MAX_LAYER]  # 比如倒数两层


# =====================
# 辅助函数: 计算 embedding
# =====================
def compute_sequence_embeddings(seqs, batch_size=16, layers=ESM_LAYERS):
    embeddings = []
    with torch.no_grad():
        for idx in tqdm(range(0, len(seqs), batch_size), desc="ESM embedding"):
            b = min(batch_size, len(seqs) - idx)
            batch_seqs = [(str(idx + j), seqs[idx + j]) for j in range(b)]
            _, batch_strs, batch_tokens = batch_converter(batch_seqs)
            batch_tokens = batch_tokens.to(DEVICE)
            out = esm_model(batch_tokens, repr_layers=layers, return_contacts=False)

            # 均值池化多层
            for i in range(b):
                reps = []
                for l in layers:
                    reps.append(out["representations"][l][i, 1:len(batch_strs[i]) + 1].mean(0))
                embeddings.append(torch.cat(reps).cpu().numpy())
    return np.array(embeddings)


# =====================
# 读取 & 构造 embedding (包含 CDR 区域交互)
# =====================
def build_or_load_embeddings(tsv_path, cache_path=EMB_CACHE, force_recompute=False):
    if os.path.exists(cache_path) and not force_recompute:
        print(f"✅ Loading cached embeddings from {cache_path}")
        data = joblib.load(cache_path)
        return data
    print("🔹 Computing embeddings from scratch...")
    df = pd.read_csv(tsv_path, sep="\t")
    seq_a = df["antibody_seq_a"].values
    seq_b = df["antibody_seq_b"].values
    seq_ag = df["antigen_seq"].values
    y = df["delta_g"].values

    emb_a = compute_sequence_embeddings(seq_a)
    emb_b = compute_sequence_embeddings(seq_b)
    emb_ag = compute_sequence_embeddings(seq_ag)

    # 生成简单 CDR-抗原交互特征 (elementwise product)
    # 假设 df 里有 CDR 索引列：cdr_start/cdr_end，可自行调整
    if {"cdr_start_a", "cdr_end_a", "cdr_start_b", "cdr_end_b"}.issubset(df.columns):
        cdr_feat = []
        for i in range(len(df)):
            cdr_a = emb_a[i, df.loc[i, "cdr_start_a"]:df.loc[i, "cdr_end_a"]].mean(axis=0)
            cdr_b = emb_b[i, df.loc[i, "cdr_start_b"]:df.loc[i, "cdr_end_b"]].mean(axis=0)
            # 与抗原 embedding 做简单交互
            feat = np.concatenate([cdr_a * emb_ag[i], cdr_b * emb_ag[i]])
            cdr_feat.append(feat)
        cdr_feat = np.array(cdr_feat)
    else:
        cdr_feat = np.zeros((len(df), 0))  # 没有 CDR 信息则空

    joblib.dump({"emb_a": emb_a, "emb_b": emb_b, "emb_ag": emb_ag,
                 "cdr_feat": cdr_feat, "y": y, "df": df}, cache_path)
    print(f"✅ Saved embeddings to {cache_path}")
    return {"emb_a": emb_a, "emb_b": emb_b, "emb_ag": emb_ag,
            "cdr_feat": cdr_feat, "y": y, "df": df}


# =====================
# LightGBM 训练
# =====================
def train_lgb(data_dict, n_splits=N_SPLITS):
    X = np.concatenate([data_dict["emb_a"], data_dict["emb_b"], data_dict["emb_ag"], data_dict["cdr_feat"]], axis=1)
    y = data_dict["y"]
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    y_preds, y_trues = [], []
    models = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        print(f"🔹 Fold {fold + 1}/{n_splits}")
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
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], eval_metric="l2")
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
    metrics = {"MSE": mse, "RMSE": rmse, "MAE": mae,
               "R2": r2, "Pearson": pearson, "Spearman": spearman}
    return models, scaler, metrics


# =====================
# 主流程
# =====================
if __name__ == "__main__":
    tsv_path = "/tmp/AbAgCDR/data/final_dataset_train.tsv"
    start = time.time()
    data_dict = build_or_load_embeddings(tsv_path)
    models, scaler, metrics = train_lgb(data_dict)

    joblib.dump({"models": models, "scaler": scaler, "EMB_CACHE": EMB_CACHE}, "NewESM2_LightGBM_pipeline.pkl")
    print("✅ Saved pipeline -> NewESM2_LightGBM_pipeline.pkl")

    print("回归性能指标:")
    for k, v in metrics.items():
        print(f"{k} = {v:.4f}")
    print(f"⏱ Total time = {time.time() - start:.1f}s")
