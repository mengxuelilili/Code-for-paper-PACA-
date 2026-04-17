# import warnings
#
# warnings.filterwarnings("ignore")
#
# import os
# import joblib
# import numpy as np
# import pandas as pd
# import torch
# from tqdm import tqdm
# import esm
# from sklearn.preprocessing import StandardScaler
#
# # =====================
# # 配置
# # =====================
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# PIPELINE_PATH = "NewESM2_LightGBM_pipeline.pkl"
# TEST_FILES = [
#     "/tmp/AbAgCDR/data/pairs_seq_sabdab.tsv",
#     "/tmp/AbAgCDR/data/pairs_seq_abbind.tsv",
#     "/tmp/AbAgCDR/data/pairs_seq_skempi.tsv",
#     "/tmp/AbAgCDR/data/pairs_seq_benchmark1.tsv",
# ]
#
# # =====================
# # 加载 pipeline
# # =====================
# pipeline = joblib.load(PIPELINE_PATH)
# models = pipeline["models"]
# scaler = pipeline["scaler"]
# EMB_CACHE = pipeline["EMB_CACHE"]
#
# # =====================
# # 加载 ESM 模型
# # =====================
# print("🔹 Loading ESM-2 model...")
# esm_model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()  # 与训练一致
# esm_model = esm_model.to(DEVICE)
# esm_model.eval()
# batch_converter = alphabet.get_batch_converter()
# MAX_LAYER = esm_model.num_layers
# ESM_LAYERS = [max(1, MAX_LAYER - 2), MAX_LAYER]
#
#
# # =====================
# # 计算 embedding
# # =====================
# def compute_sequence_embeddings(seqs, batch_size=16, layers=ESM_LAYERS):
#     embeddings = []
#     with torch.no_grad():
#         for idx in tqdm(range(0, len(seqs), batch_size), desc="ESM embedding"):
#             b = min(batch_size, len(seqs) - idx)
#             batch_seqs = [(str(idx + j), seqs[idx + j]) for j in range(b)]
#             _, batch_strs, batch_tokens = batch_converter(batch_seqs)
#             batch_tokens = batch_tokens.to(DEVICE)
#             out = esm_model(batch_tokens, repr_layers=layers, return_contacts=False)
#             for i in range(b):
#                 reps = []
#                 for l in layers:
#                     reps.append(out["representations"][l][i, 1:len(batch_strs[i]) + 1].mean(0))
#                 embeddings.append(torch.cat(reps).cpu().numpy())
#     return np.array(embeddings)
#
#
# # =====================
# # 测试函数
# # =====================
# def test_dataset(file):
#     df = pd.read_csv(file, sep="\t")
#     seq_a = df["antibody_seq_a"].values
#     seq_b = df["antibody_seq_b"].values
#     seq_ag = df["antigen_seq"].values
#     y_true = df["delta_g"].values
#
#     emb_a = compute_sequence_embeddings(seq_a)
#     emb_b = compute_sequence_embeddings(seq_b)
#     emb_ag = compute_sequence_embeddings(seq_ag)
#
#     # CDR 交互特征
#     if {"cdr_start_a", "cdr_end_a", "cdr_start_b", "cdr_end_b"}.issubset(df.columns):
#         cdr_feat = []
#         for i in range(len(df)):
#             cdr_a = emb_a[i, df.loc[i, "cdr_start_a"]:df.loc[i, "cdr_end_a"]].mean(axis=0)
#             cdr_b = emb_b[i, df.loc[i, "cdr_start_b"]:df.loc[i, "cdr_end_b"]].mean(axis=0)
#             feat = np.concatenate([cdr_a * emb_ag[i], cdr_b * emb_ag[i]])
#             cdr_feat.append(feat)
#         cdr_feat = np.array(cdr_feat)
#     else:
#         cdr_feat = np.zeros((len(df), 0))
#
#     X = np.concatenate([emb_a, emb_b, emb_ag, cdr_feat], axis=1)
#     X = scaler.transform(X)
#
#     # ensemble 预测 (KFold 模型取平均)
#     y_pred = np.zeros(len(df))
#     for model in models:
#         y_pred += model.predict(X)
#     y_pred /= len(models)
#
#     df_res = df.copy()
#     df_res["predicted_delta_g"] = y_pred
#     return df_res
#
#
# # =====================
# # 测试所有文件并保存
# # =====================
# all_results = []
# for f in TEST_FILES:
#     print(f"--- Testing {f} ---")
#     df_res = test_dataset(f)
#     all_results.append(df_res)
# all_df = pd.concat(all_results, ignore_index=True)
# all_df.to_csv("NewESM2_LightGBM_test_results.csv", index=False)
# print("✅ All test results saved to NewESM2_LightGBM_test_results.csv")

import warnings
warnings.filterwarnings("ignore")

import os
import joblib
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr, spearmanr
import esm

# =====================
# 配置
# =====================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PIPELINE_PATH = "NewESM2_LightGBM_pipeline.pkl"

TEST_FILES = [
    "/tmp/AbAgCDR/data/pairs_seq_sabdab.tsv",
    "/tmp/AbAgCDR/data/pairs_seq_abbind.tsv",
    "/tmp/AbAgCDR/data/pairs_seq_skempi.tsv",
    "/tmp/AbAgCDR/data/pairs_seq_benchmark1.tsv",
]

# =====================
# 加载 pipeline
# =====================
pipeline = joblib.load(PIPELINE_PATH)
models = pipeline["models"]
scaler = pipeline["scaler"]

# =====================
# 加载 ESM 模型
# =====================
print("🔹 Loading ESM-2 model...")
esm_model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
esm_model = esm_model.to(DEVICE)
esm_model.eval()
batch_converter = alphabet.get_batch_converter()
MAX_LAYER = esm_model.num_layers
ESM_LAYERS = [max(1, MAX_LAYER-2), MAX_LAYER]

# =====================
# 计算 embedding
# =====================
def compute_sequence_embeddings(seqs, batch_size=16, layers=ESM_LAYERS):
    embeddings = []
    with torch.no_grad():
        for idx in tqdm(range(0, len(seqs), batch_size), desc="ESM embedding"):
            b = min(batch_size, len(seqs) - idx)
            batch_seqs = [(str(idx+j), seqs[idx+j]) for j in range(b)]
            _, batch_strs, batch_tokens = batch_converter(batch_seqs)
            batch_tokens = batch_tokens.to(DEVICE)
            out = esm_model(batch_tokens, repr_layers=layers, return_contacts=False)
            for i in range(b):
                reps = []
                for l in layers:
                    reps.append(out["representations"][l][i, 1:len(batch_strs[i])+1].mean(0))
                embeddings.append(torch.cat(reps).cpu().numpy())
    return np.array(embeddings)

# =====================
# 测试单个数据集并计算指标
# =====================
def test_dataset(file):
    df = pd.read_csv(file, sep="\t")
    seq_a = df["antibody_seq_a"].values
    seq_b = df["antibody_seq_b"].values
    seq_ag = df["antigen_seq"].values

    y_true = df["delta_g"].values

    emb_a = compute_sequence_embeddings(seq_a)
    emb_b = compute_sequence_embeddings(seq_b)
    emb_ag = compute_sequence_embeddings(seq_ag)

    # CDR 交互特征
    if {"cdr_start_a","cdr_end_a","cdr_start_b","cdr_end_b"}.issubset(df.columns):
        cdr_feat = []
        for i in range(len(df)):
            cdr_a = emb_a[i, df.loc[i,"cdr_start_a"]:df.loc[i,"cdr_end_a"]].mean(axis=0)
            cdr_b = emb_b[i, df.loc[i,"cdr_start_b"]:df.loc[i,"cdr_end_b"]].mean(axis=0)
            feat = np.concatenate([cdr_a*emb_ag[i], cdr_b*emb_ag[i]])
            cdr_feat.append(feat)
        cdr_feat = np.array(cdr_feat)
    else:
        cdr_feat = np.zeros((len(df),0))

    X = np.concatenate([emb_a, emb_b, emb_ag, cdr_feat], axis=1)
    X = scaler.transform(X)

    # ensemble 预测
    y_pred = np.zeros(len(df))
    for model in models:
        y_pred += model.predict(X)
    y_pred /= len(models)

    # 计算指标
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    pearson, _ = pearsonr(y_true, y_pred)
    spearman, _ = spearmanr(y_true, y_pred)
    metrics = {
        "MSE": mse, "RMSE": rmse, "MAE": mae,
        "R2": r2, "Pearson": pearson, "Spearman": spearman,
        "n_samples": len(df)
    }

    df_res = df.copy()
    df_res["predicted_delta_g"] = y_pred
    return df_res, metrics

# =====================
# 测试所有文件
# =====================
all_results = []
all_metrics = []

for f in TEST_FILES:
    print(f"\n--- Testing {f} ---")
    df_res, metrics = test_dataset(f)
    all_results.append(df_res)
    all_metrics.append({"file": os.path.basename(f), **metrics})
    print("Metrics:", metrics)

# 汇总保存
all_df = pd.concat(all_results, ignore_index=True)
all_df.to_csv("NewESM2_LightGBM_test_results.csv", index=False)
metrics_df = pd.DataFrame(all_metrics)
metrics_df.to_csv("NewESM2_LightGBM_test_metrics.csv", index=False)
print("\n✅ All test results saved to NewESM2_LightGBM_test_results.csv")
print("✅ Summary metrics saved to NewESM2_LightGBM_test_metrics.csv")
