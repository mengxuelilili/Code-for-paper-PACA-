"""
升级版 ESM2 + CDR Cross Attention + LightGBM 测试脚本
直接加载训练好的 pipeline 进行新数据预测
"""

# import os
# import joblib
# import numpy as np
# import pandas as pd
# import torch
# from tqdm import tqdm
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# from scipy.stats import pearsonr, spearmanr
# import esm
#
# # ----------------------------
# # 配置
# # ----------------------------
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# CROSS_FEAT_LEN = 10
#
# # 测试数据集路径列表
# test_files = [
#     "/tmp/AbAgCDR/data/pairs_seq_sabdab.tsv",
#     "/tmp/AbAgCDR/data/pairs_seq_abbind.tsv",
#     "/tmp/AbAgCDR/data/pairs_seq_skempi.tsv",
#     "/tmp/AbAgCDR/data/pairs_seq_benchmark1.tsv",
# ]
#
# # ----------------------------
# # 1. 加载 ESM2 模型
# # ----------------------------
# print("🔹 Loading ESM-2 model...")
# esm_model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
# esm_model = esm_model.to(DEVICE)
# batch_converter = alphabet.get_batch_converter()
# esm_model.eval()
#
#
# # ----------------------------
# # 2. ESM embedding
# # ----------------------------
# def compute_sequence_embeddings(seqs, repr_layer=6, batch_size=16):
#     embeddings, token_reps_all = [], []
#     with torch.no_grad():
#         for idx in tqdm(range(0, len(seqs), batch_size), desc="ESM embedding"):
#             b = min(batch_size, len(seqs) - idx)
#             batch_seqs = [(str(idx + j), seqs[idx + j]) for j in range(b)]
#             labels, strs, tokens = batch_converter(batch_seqs)
#             tokens = tokens.to(DEVICE)
#             out = esm_model(tokens, repr_layers=[repr_layer], return_contacts=False)
#             reps = out["representations"][repr_layer]
#
#             for i in range(b):
#                 seq_len = len(strs[i])
#                 mean_emb = reps[i, 1:seq_len + 1].mean(0).cpu().numpy()
#                 embeddings.append(mean_emb)
#                 token_reps_all.append(reps[i, 1:seq_len + 1].cpu().numpy())
#     return np.array(embeddings), token_reps_all
#
#
# # ----------------------------
# # 3. 固定长度池化
# # ----------------------------
# def fixed_length_pool(vec, target_len=CROSS_FEAT_LEN):
#     vec_len = len(vec)
#     if vec_len >= target_len:
#         return vec[:target_len]
#     else:
#         return np.pad(vec, (0, target_len - vec_len), 'constant')
#
#
# # ----------------------------
# # 4. CDR编号映射
# # ----------------------------
# def getCDRPos(_loop):
#     CDRS = {
#         'L1': ['24','25','26','27','28','29','30','30A','30B','30C','30D','30E','30F','30G','30H','30I','31','32','33','34'],
#         'L2': ['50','51','51A','52','52A','52B','52C','52D','53','54','55','56'],
#         'L3': ['89','90','91','92','93','94','95','95A','95B','95C','95D','95E','95F','95G','95H','95I','95J','96','97'],
#         'H1': ['26','27','28','29','30','31','31A','31B','31C','31D','31E','31F','31G','31H','31I','31J','32'],
#         'H2': ['52','52A','52B','52C','52D','52E','52F','52G','52H','52I','52J','52K','52L','52M','52N','52O','53','54','55','56'],
#         'H3': ['95','96','97','98','99','100','100A','100B','100C','100D','100E','100F','100G','100H','100I','100J','100K','100L','100M','100N','100O','100P','100Q','100R','100S','100T','100U','100V','100W','100X','100Y','100Z','101','102']
#     }
#     return CDRS[_loop]
#
#
# # ----------------------------
# # 5. Cross Attention 特征
# # ----------------------------
# def extract_cdr_attention_features(df, tokens_a, tokens_b, tokens_ag):
#     features = []
#
#     for i in tqdm(range(len(df)), desc="Extracting CDR Attention Features"):
#         ta = tokens_a[i]
#         tb = tokens_b[i]
#         tag = tokens_ag[i]
#
#         att_h2a = np.matmul(ta.mean(axis=0, keepdims=True), tag.T).flatten()
#         att_l2a = np.matmul(tb.mean(axis=0, keepdims=True), tag.T).flatten()
#         att_a2h = np.matmul(tag.mean(axis=0, keepdims=True), ta.T).flatten()
#         att_a2l = np.matmul(tag.mean(axis=0, keepdims=True), tb.T).flatten()
#
#         feat = np.concatenate([
#             fixed_length_pool(att_h2a),
#             fixed_length_pool(att_l2a),
#             fixed_length_pool(att_a2h),
#             fixed_length_pool(att_a2l)
#         ])
#         features.append(feat)
#
#     return np.array(features)
#
#
# # ----------------------------
# # 6. 构建特征
# # ----------------------------
# def build_features(df):
#     seq_a = df["antibody_seq_a"].values
#     seq_b = df["antibody_seq_b"].values
#     seq_ag = df["antigen_seq"].values
#
#     emb_a, tokens_a = compute_sequence_embeddings(seq_a)
#     emb_b, tokens_b = compute_sequence_embeddings(seq_b)
#     emb_ag, tokens_ag = compute_sequence_embeddings(seq_ag)
#
#     features_cross = extract_cdr_attention_features(df, tokens_a, tokens_b, tokens_ag)
#
#     # 合并 ESM embedding + Cross Attention
#     features = np.concatenate([emb_a, emb_b, emb_ag, features_cross], axis=1)
#     return features
#
#
# # ----------------------------
# # 7. 加载 pipeline & 预测
# # ----------------------------
# pipeline = joblib.load("esm2_cdr_attention_lgbm_pipeline.pkl")
# models = pipeline['models']
# scaler = pipeline['scaler']
#
# for tsv_path in test_files:
#     print(f"\n--- Testing {tsv_path} ---")
#     df = pd.read_csv(tsv_path, sep="\t")
#     y_true = df["delta_g"].values
#
#     features = build_features(df)
#     X_scaled = scaler.transform(features)
#
#     # 多模型平均预测
#     preds = np.mean([m.predict(X_scaled) for m in models], axis=0)
#
#     # 输出指标
#     mse = mean_squared_error(y_true, preds)
#     rmse = np.sqrt(mse)
#     mae = mean_absolute_error(y_true, preds)
#     r2 = r2_score(y_true, preds)
#     pearson = pearsonr(y_true, preds)[0]
#     spearman = spearmanr(y_true, preds)[0]
#
#     print(f"Samples: {len(y_true)}")
#     print(f"MSE = {mse:.4f}, RMSE = {rmse:.4f}, MAE = {mae:.4f}, R² = {r2:.4f}")
#     print(f"Pearson = {pearson:.4f}, Spearman = {spearman:.4f}")

# import os
# import joblib
# import numpy as np
# import pandas as pd
# import torch
# from tqdm import tqdm
# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# from scipy.stats import pearsonr, spearmanr
# import esm
#
# # ----------------------------
# # 配置
# # ----------------------------
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# CROSS_FEAT_LEN = 10
#
# # 测试数据集路径列表
# test_files = [
#     "/tmp/AbAgCDR/data/pairs_seq_sabdab.tsv",
#     "/tmp/AbAgCDR/data/pairs_seq_abbind.tsv",
#     "/tmp/AbAgCDR/data/pairs_seq_skempi.tsv",
#     "/tmp/AbAgCDR/data/pairs_seq_benchmark1.tsv",
# ]
#
# # ----------------------------
# # 1. 加载 ESM2 模型
# # ----------------------------
# print("🔹 Loading ESM-2 model...")
# esm_model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
# esm_model = esm_model.to(DEVICE)
# batch_converter = alphabet.get_batch_converter()
# esm_model.eval()
#
# # ----------------------------
# # 2. ESM embedding
# # ----------------------------
# def compute_sequence_embeddings(seqs, repr_layer=6, batch_size=16):
#     embeddings, token_reps_all = [], []
#     with torch.no_grad():
#         for idx in tqdm(range(0, len(seqs), batch_size), desc="ESM embedding"):
#             b = min(batch_size, len(seqs) - idx)
#             batch_seqs = [(str(idx + j), seqs[idx + j]) for j in range(b)]
#             labels, strs, tokens = batch_converter(batch_seqs)
#             tokens = tokens.to(DEVICE)
#             out = esm_model(tokens, repr_layers=[repr_layer], return_contacts=False)
#             reps = out["representations"][repr_layer]
#
#             for i in range(b):
#                 seq_len = len(strs[i])
#                 mean_emb = reps[i, 1:seq_len + 1].mean(0).cpu().numpy()
#                 embeddings.append(mean_emb)
#                 token_reps_all.append(reps[i, 1:seq_len + 1].cpu().numpy())
#     return np.array(embeddings), token_reps_all
#
# # ----------------------------
# # 3. 固定长度池化
# # ----------------------------
# def fixed_length_pool(vec, target_len=CROSS_FEAT_LEN):
#     vec_len = len(vec)
#     if vec_len >= target_len:
#         return vec[:target_len]
#     else:
#         return np.pad(vec, (0, target_len - vec_len), 'constant')
#
# # ----------------------------
# # 4. CDR编号映射
# # ----------------------------
# def getCDRPos(_loop):
#     CDRS = {
#         'L1': ['24','25','26','27','28','29','30','30A','30B','30C','30D','30E','30F','30G','30H','30I','31','32','33','34'],
#         'L2': ['50','51','51A','52','52A','52B','52C','52D','53','54','55','56'],
#         'L3': ['89','90','91','92','93','94','95','95A','95B','95C','95D','95E','95F','95G','95H','95I','95J','96','97'],
#         'H1': ['26','27','28','29','30','31','31A','31B','31C','31D','31E','31F','31G','31H','31I','31J','32'],
#         'H2': ['52','52A','52B','52C','52D','52E','52F','52G','52H','52I','52J','52K','52L','52M','52N','52O','53','54','55','56'],
#         'H3': ['95','96','97','98','99','100','100A','100B','100C','100D','100E','100F','100G','100H','100I','100J','100K','100L','100M','100N','100O','100P','100Q','100R','100S','100T','100U','100V','100W','100X','100Y','100Z','101','102']
#     }
#     return CDRS[_loop]
#
# # ----------------------------
# # 5. Cross Attention 特征
# # ----------------------------
# def extract_cdr_attention_features(df, tokens_a, tokens_b, tokens_ag):
#     features = []
#
#     for i in tqdm(range(len(df)), desc="Extracting CDR Attention Features"):
#         ta = tokens_a[i]
#         tb = tokens_b[i]
#         tag = tokens_ag[i]
#
#         att_h2a = np.matmul(ta.mean(axis=0, keepdims=True), tag.T).flatten()
#         att_l2a = np.matmul(tb.mean(axis=0, keepdims=True), tag.T).flatten()
#         att_a2h = np.matmul(tag.mean(axis=0, keepdims=True), ta.T).flatten()
#         att_a2l = np.matmul(tag.mean(axis=0, keepdims=True), tb.T).flatten()
#
#         feat = np.concatenate([
#             fixed_length_pool(att_h2a),
#             fixed_length_pool(att_l2a),
#             fixed_length_pool(att_a2h),
#             fixed_length_pool(att_a2l)
#         ])
#         features.append(feat)
#
#     return np.array(features)
#
# # ----------------------------
# # 6. 构建特征
# # ----------------------------
# def build_features(df):
#     seq_a = df["antibody_seq_a"].values
#     seq_b = df["antibody_seq_b"].values
#     seq_ag = df["antigen_seq"].values
#
#     emb_a, tokens_a = compute_sequence_embeddings(seq_a)
#     emb_b, tokens_b = compute_sequence_embeddings(seq_b)
#     emb_ag, tokens_ag = compute_sequence_embeddings(seq_ag)
#
#     features_cross = extract_cdr_attention_features(df, tokens_a, tokens_b, tokens_ag)
#
#     # 合并 ESM embedding + Cross Attention
#     features = np.concatenate([emb_a, emb_b, emb_ag, features_cross], axis=1)
#     return features
#
# # ----------------------------
# # 7. 加载 pipeline & 预测
# # ----------------------------
# pipeline = joblib.load("esm2_cdr_attention_lgbm_pipeline.pkl")
# model = pipeline['model']          # 单模型
# feat_scaler = pipeline['feat_scaler']
# label_scaler = pipeline['label_scaler']
#
# for tsv_path in test_files:
#     print(f"\n--- Testing {tsv_path} ---")
#     df = pd.read_csv(tsv_path, sep="\t")
#     y_true = df["delta_g"].values.reshape(-1,1)
#
#     features = build_features(df)
#     X_scaled = feat_scaler.transform(features)
#
#     # 单模型预测
#     y_pred_scaled = model.predict(X_scaled).reshape(-1,1)
#
#     # 反归一化
#     y_pred = label_scaler.inverse_transform(y_pred_scaled)
#
#     # 输出指标
#     mse = mean_squared_error(y_true, y_pred)
#     rmse = np.sqrt(mse)
#     mae = mean_absolute_error(y_true, y_pred)
#     r2 = r2_score(y_true, y_pred)
#     pearson = pearsonr(y_true.flatten(), y_pred.flatten())[0]
#     spearman = spearmanr(y_true.flatten(), y_pred.flatten())[0]
#
#     print(f"Samples: {len(y_true)}")
#     print(f"MSE = {mse:.4f}, RMSE = {rmse:.4f}, MAE = {mae:.4f}, R² = {r2:.4f}")
#     print(f"Pearson = {pearson:.4f}, Spearman = {spearman:.4f}")
# import os
# import joblib
# import numpy as np
# import pandas as pd
# import torch
# from tqdm import tqdm
# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# from scipy.stats import pearsonr, spearmanr
# import esm
#
# # ----------------------------
# # 配置
# # ----------------------------
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# CROSS_FEAT_LEN = 10
#
# # 测试数据集路径列表
# test_files = [
#     "/tmp/AbAgCDR/data/pairs_seq_sabdab.tsv",
#     "/tmp/AbAgCDR/data/pairs_seq_abbind.tsv",
#     "/tmp/AbAgCDR/data/pairs_seq_skempi.tsv",
#     "/tmp/AbAgCDR/data/pairs_seq_benchmark1.tsv",
# ]
#
# # ----------------------------
# # 1. 加载 ESM2 模型
# # ----------------------------
# print("🔹 Loading ESM-2 model...")
# esm_model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
# esm_model = esm_model.to(DEVICE)
# batch_converter = alphabet.get_batch_converter()
# esm_model.eval()
#
# # ----------------------------
# # 2. ESM embedding
# # ----------------------------
# def compute_sequence_embeddings(seqs, repr_layer=6, batch_size=16):
#     embeddings, token_reps_all = [], []
#     with torch.no_grad():
#         for idx in tqdm(range(0, len(seqs), batch_size), desc="ESM embedding"):
#             b = min(batch_size, len(seqs) - idx)
#             batch_seqs = [(str(idx + j), seqs[idx + j]) for j in range(b)]
#             labels, strs, tokens = batch_converter(batch_seqs)
#             tokens = tokens.to(DEVICE)
#             out = esm_model(tokens, repr_layers=[repr_layer], return_contacts=False)
#             reps = out["representations"][repr_layer]
#
#             for i in range(b):
#                 seq_len = len(strs[i])
#                 mean_emb = reps[i, 1:seq_len + 1].mean(0).cpu().numpy()
#                 embeddings.append(mean_emb)
#                 token_reps_all.append(reps[i, 1:seq_len + 1].cpu().numpy())
#     return np.array(embeddings), token_reps_all
#
# # ----------------------------
# # 3. 固定长度池化
# # ----------------------------
# def fixed_length_pool(vec, target_len=CROSS_FEAT_LEN):
#     vec_len = len(vec)
#     return vec[:target_len] if vec_len >= target_len else np.pad(vec, (0, target_len - vec_len), 'constant')
#
# # ----------------------------
# # 4. CDR编号映射
# # ----------------------------
# def getCDRPos(_loop):
#     CDRS = {
#         'L1': ['24','25','26','27','28','29','30','30A','30B','30C','30D','30E','30F','30G','30H','30I','31','32','33','34'],
#         'L2': ['50','51','51A','52','52A','52B','52C','52D','53','54','55','56'],
#         'L3': ['89','90','91','92','93','94','95','95A','95B','95C','95D','95E','95F','95G','95H','95I','95J','96','97'],
#         'H1': ['26','27','28','29','30','31','31A','31B','31C','31D','31E','31F','31G','31H','31I','31J','32'],
#         'H2': ['52','52A','52B','52C','52D','52E','52F','52G','52H','52I','52J','52K','52L','52M','52N','52O','53','54','55','56'],
#         'H3': ['95','96','97','98','99','100','100A','100B','100C','100D','100E','100F','100G','100H','100I','100J','100K','100L','100M','100N','100O','100P','100Q','100R','100S','100T','100U','100V','100W','100X','100Y','100Z','101','102']
#     }
#     return CDRS[_loop]
#
# # ----------------------------
# # 5. Cross Attention 特征
# # ----------------------------
# def extract_cdr_attention_features(df, tokens_a, tokens_b, tokens_ag):
#     features = []
#     for i in tqdm(range(len(df)), desc="Extracting CDR Attention Features"):
#         ta = tokens_a[i]
#         tb = tokens_b[i]
#         tag = tokens_ag[i]
#
#         att_h2a = np.matmul(ta.mean(axis=0, keepdims=True), tag.T).flatten()
#         att_l2a = np.matmul(tb.mean(axis=0, keepdims=True), tag.T).flatten()
#         att_a2h = np.matmul(tag.mean(axis=0, keepdims=True), ta.T).flatten()
#         att_a2l = np.matmul(tag.mean(axis=0, keepdims=True), tb.T).flatten()
#
#         feat = np.concatenate([
#             fixed_length_pool(att_h2a),
#             fixed_length_pool(att_l2a),
#             fixed_length_pool(att_a2h),
#             fixed_length_pool(att_a2l)
#         ])
#         features.append(feat)
#
#     return np.array(features)
#
# # ----------------------------
# # 6. 构建特征
# # ----------------------------
# def build_features(df):
#     seq_a = df["antibody_seq_a"].values
#     seq_b = df["antibody_seq_b"].values
#     seq_ag = df["antigen_seq"].values
#
#     emb_a, tokens_a = compute_sequence_embeddings(seq_a)
#     emb_b, tokens_b = compute_sequence_embeddings(seq_b)
#     emb_ag, tokens_ag = compute_sequence_embeddings(seq_ag)
#
#     features_cross = extract_cdr_attention_features(df, tokens_a, tokens_b, tokens_ag)
#     features = np.concatenate([emb_a, emb_b, emb_ag, features_cross], axis=1)
#     return features
#
# # ----------------------------
# # 7. 加载 pipeline & 预测 (多模型+PCA)
# # ----------------------------
# pipeline = joblib.load("esm2_cdr_attention_lgbm_pipeline.pkl")
# models = pipeline['models']          # 多模型
# feat_scaler = pipeline['feat_scaler']
# label_scaler = pipeline['label_scaler']
# pca = pipeline['pca']
#
# for tsv_path in test_files:
#     print(f"\n--- Testing {tsv_path} ---")
#     df = pd.read_csv(tsv_path, sep="\t")
#     y_true = df["delta_g"].values.reshape(-1,1)
#
#     features = build_features(df)
#     X_scaled = feat_scaler.transform(features)
#     X_pca = pca.transform(X_scaled)
#
#     # 多模型平均预测
#     y_pred_scaled = np.mean([m.predict(X_pca) for m in models], axis=0).reshape(-1,1)
#
#     # 反归一化
#     y_pred = label_scaler.inverse_transform(y_pred_scaled)
#
#     # 输出指标
#     mse = mean_squared_error(y_true, y_pred)
#     rmse = np.sqrt(mse)
#     mae = mean_absolute_error(y_true, y_pred)
#     r2 = r2_score(y_true, y_pred)
#     pearson = pearsonr(y_true.flatten(), y_pred.flatten())[0]
#     spearman = spearmanr(y_true.flatten(), y_pred.flatten())[0]
#
#     print(f"Samples: {len(y_true)}")
#     print(f"MSE = {mse:.4f}, RMSE = {rmse:.4f}, MAE = {mae:.4f}, R² = {r2:.4f}")
#     print(f"Pearson = {pearson:.4f}, Spearman = {spearman:.4f}")
import os
import joblib
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr, spearmanr
import esm

# ----------------------------
# 配置
# ----------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CROSS_FEAT_LEN = 10

# 测试数据集路径列表
test_files = [
    "/tmp/AbAgCDR/data/pairs_seq_sabdab.tsv",
    "/tmp/AbAgCDR/data/pairs_seq_abbind.tsv",
    "/tmp/AbAgCDR/data/pairs_seq_skempi.tsv",
    "/tmp/AbAgCDR/data/pairs_seq_benchmark1.tsv",
]

# ----------------------------
# 1. 加载 ESM2 模型
# ----------------------------
print("🔹 Loading ESM-2 model...")
esm_model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
esm_model = esm_model.to(DEVICE)
batch_converter = alphabet.get_batch_converter()
esm_model.eval()

# ----------------------------
# 2. ESM embedding
# ----------------------------
def compute_sequence_embeddings(seqs, repr_layer=6, batch_size=16):
    embeddings, token_reps_all = [], []
    with torch.no_grad():
        for idx in tqdm(range(0, len(seqs), batch_size), desc="ESM embedding"):
            b = min(batch_size, len(seqs) - idx)
            batch_seqs = [(str(idx + j), seqs[idx + j]) for j in range(b)]
            labels, strs, tokens = batch_converter(batch_seqs)
            tokens = tokens.to(DEVICE)
            out = esm_model(tokens, repr_layers=[repr_layer], return_contacts=False)
            reps = out["representations"][repr_layer]

            for i in range(b):
                seq_len = len(strs[i])
                mean_emb = reps[i, 1:seq_len + 1].mean(0).cpu().numpy()
                embeddings.append(mean_emb)
                token_reps_all.append(reps[i, 1:seq_len + 1].cpu().numpy())
    return np.array(embeddings), token_reps_all

# ----------------------------
# 3. 固定长度池化
# ----------------------------
def fixed_length_pool(vec, target_len=CROSS_FEAT_LEN):
    vec_len = len(vec)
    return vec[:target_len] if vec_len >= target_len else np.pad(vec, (0, target_len - vec_len), 'constant')

# ----------------------------
# 4. CDR编号映射
# ----------------------------
def getCDRPos(_loop):
    CDRS = {
        'L1': ['24','25','26','27','28','29','30','30A','30B','30C','30D','30E','30F','30G','30H','30I','31','32','33','34'],
        'L2': ['50','51','51A','52','52A','52B','52C','52D','53','54','55','56'],
        'L3': ['89','90','91','92','93','94','95','95A','95B','95C','95D','95E','95F','95G','95H','95I','95J','96','97'],
        'H1': ['26','27','28','29','30','31','31A','31B','31C','31D','31E','31F','31G','31H','31I','31J','32'],
        'H2': ['52','52A','52B','52C','52D','52E','52F','52G','52H','52I','52J','52K','52L','52M','52N','52O','53','54','55','56'],
        'H3': ['95','96','97','98','99','100','100A','100B','100C','100D','100E','100F','100G','100H','100I','100J','100K','100L','100M','100N','100O','100P','100Q','100R','100S','100T','100U','100V','100W','100X','100Y','100Z','101','102']
    }
    return CDRS[_loop]

# ----------------------------
# 5. Cross Attention 特征
# ----------------------------
def extract_cdr_attention_features(df, tokens_a, tokens_b, tokens_ag):
    features = []
    for i in tqdm(range(len(df)), desc="Extracting CDR Attention Features"):
        ta = tokens_a[i]
        tb = tokens_b[i]
        tag = tokens_ag[i]

        att_h2a = np.matmul(ta.mean(axis=0, keepdims=True), tag.T).flatten()
        att_l2a = np.matmul(tb.mean(axis=0, keepdims=True), tag.T).flatten()
        att_a2h = np.matmul(tag.mean(axis=0, keepdims=True), ta.T).flatten()
        att_a2l = np.matmul(tag.mean(axis=0, keepdims=True), tb.T).flatten()

        feat = np.concatenate([
            fixed_length_pool(att_h2a),
            fixed_length_pool(att_l2a),
            fixed_length_pool(att_a2h),
            fixed_length_pool(att_a2l)
        ])
        features.append(feat)

    return np.array(features)

# ----------------------------
# 6. 构建特征
# ----------------------------
def build_features(df):
    seq_a = df["antibody_seq_a"].values
    seq_b = df["antibody_seq_b"].values
    seq_ag = df["antigen_seq"].values

    emb_a, tokens_a = compute_sequence_embeddings(seq_a)
    emb_b, tokens_b = compute_sequence_embeddings(seq_b)
    emb_ag, tokens_ag = compute_sequence_embeddings(seq_ag)

    features_cross = extract_cdr_attention_features(df, tokens_a, tokens_b, tokens_ag)
    features = np.concatenate([emb_a, emb_b, emb_ag, features_cross], axis=1)
    return features

# ----------------------------
# 7. 加载 pipeline & 预测 (多模型+PCA)
# ----------------------------
pipeline = joblib.load("esm2_cdr_attention_lgbm_pipeline.pkl")
models = pipeline['models']          # 多模型
feat_scaler = pipeline['feat_scaler']
label_scaler = pipeline['label_scaler']
pca = pipeline['pca']

for tsv_path in test_files:
    print(f"\n--- Testing {tsv_path} ---")
    df = pd.read_csv(tsv_path, sep="\t")
    y_true_scaled = label_scaler.transform(df["delta_g"].values.reshape(-1,1)).ravel()

    features = build_features(df)
    X_scaled = feat_scaler.transform(features)
    X_pca = pca.transform(X_scaled)

    # 多模型平均预测
    y_pred_scaled = np.mean([m.predict(X_pca) for m in models], axis=0)

    # 输出指标（标准化 ΔG）
    mse = mean_squared_error(y_true_scaled, y_pred_scaled)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true_scaled, y_pred_scaled)
    r2 = r2_score(y_true_scaled, y_pred_scaled)
    pearson = pearsonr(y_true_scaled, y_pred_scaled)[0]
    spearman = spearmanr(y_true_scaled, y_pred_scaled)[0]

    print(f"Samples: {len(y_true_scaled)}")
    print(f"MSE = {mse:.4f}, RMSE = {rmse:.4f}, MAE = {mae:.4f}, R² = {r2:.4f}")
    print(f"Pearson = {pearson:.4f}, Spearman = {spearman:.4f}")
