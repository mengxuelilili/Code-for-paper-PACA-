"""
升级版 ESM2 + LightGBM Pipeline
支持 CDR-H/L → Antigen 与 Antigen → CDR-H/L Cross Attention 特征
"""

# import os
# import time
# import warnings
# warnings.filterwarnings("ignore")
#
# import joblib
# import numpy as np
# import pandas as pd
# import torch
# from tqdm import tqdm
# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import KFold
# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# from scipy.stats import pearsonr, spearmanr
# import lightgbm as lgb
# import esm
#
# # ----------------------------
# # 配置
# # ----------------------------
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# EMB_CACHE = "esm2_cdr_attention_embeddings.pkl"
# N_SPLITS = 5
# CROSS_FEAT_LEN = 10  # Cross Attention 输出固定长度
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
# # 2. 序列转 ESM embedding
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
# # 3. 固定长度池化函数
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
# # 5. Cross Attention 特征提取
# # ----------------------------
# def extract_cdr_attention_features(df, tokens_a, tokens_b, tokens_ag):
#     features = []
#
#     # CDR序列索引
#     cdr_h = np.concatenate([getCDRPos('H1'), getCDRPos('H2'), getCDRPos('H3')])
#     cdr_l = np.concatenate([getCDRPos('L1'), getCDRPos('L2'), getCDRPos('L3')])
#
#     for i in tqdm(range(len(df)), desc="Extracting CDR Attention Features"):
#         ta = tokens_a[i]  # (La, D)
#         tb = tokens_b[i]  # (Lb, D)
#         tag = tokens_ag[i]  # (Lag, D)
#
#         # 注意：这里用简单点积 attention
#         # H -> A
#         att_h2a = np.matmul(ta.mean(axis=0, keepdims=True), tag.T).flatten()
#         feat_h2a = fixed_length_pool(att_h2a)
#
#         # L -> A
#         att_l2a = np.matmul(tb.mean(axis=0, keepdims=True), tag.T).flatten()
#         feat_l2a = fixed_length_pool(att_l2a)
#
#         # A -> H
#         att_a2h = np.matmul(tag.mean(axis=0, keepdims=True), ta.T).flatten()
#         feat_a2h = fixed_length_pool(att_a2h)
#
#         # A -> L
#         att_a2l = np.matmul(tag.mean(axis=0, keepdims=True), tb.T).flatten()
#         feat_a2l = fixed_length_pool(att_a2l)
#
#         # 合并成单条特征向量
#         feat = np.concatenate([feat_h2a, feat_l2a, feat_a2h, feat_a2l])
#         features.append(feat)
#
#     return np.array(features)
#
#
# # ----------------------------
# # 6. 训练脚本主流程
# # ----------------------------
# def build_or_load_embeddings(tsv_path, force_recompute=False):
#     if os.path.exists(EMB_CACHE) and not force_recompute:
#         print(f"✅ Loading cached embeddings from {EMB_CACHE}")
#         data = joblib.load(EMB_CACHE)
#         return data['features'], data['y'], data['df']
#
#     print("🔹 Computing embeddings from scratch...")
#     df = pd.read_csv(tsv_path, sep="\t")
#     seq_a = df["antibody_seq_a"].values
#     seq_b = df["antibody_seq_b"].values
#     seq_ag = df["antigen_seq"].values
#     y = df["delta_g"].values
#
#     emb_a, tokens_a = compute_sequence_embeddings(seq_a)
#     emb_b, tokens_b = compute_sequence_embeddings(seq_b)
#     emb_ag, tokens_ag = compute_sequence_embeddings(seq_ag)
#
#     # Cross Attention 特征
#     features_cross = extract_cdr_attention_features(df, tokens_a, tokens_b, tokens_ag)
#
#     # 合并 ESM embedding + Cross Attention 特征
#     features = np.concatenate([emb_a, emb_b, emb_ag, features_cross], axis=1)
#
#     joblib.dump({'features': features, 'y': y, 'df': df}, EMB_CACHE)
#     print(f"✅ Saved embeddings & features to {EMB_CACHE}")
#     return features, y, df
#
#
# def train_lightgbm(features, y):
#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(features)
#
#     kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
#     models, y_trues, y_preds = [], [], []
#
#     for fold, (train_idx, val_idx) in enumerate(kf.split(X_scaled)):
#         print(f"🔹 Fold {fold+1}/{N_SPLITS}")
#         X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
#         y_train, y_val = y[train_idx], y[val_idx]
#
#         model = lgb.LGBMRegressor(
#             n_estimators=500,
#             learning_rate=0.05,
#             max_depth=7,
#             min_data_in_leaf=2,
#             subsample=0.8,
#             colsample_bytree=0.8,
#             random_state=42
#         )
#         model.fit(X_train, y_train, eval_set=[(X_val, y_val)], eval_metric='l2')
#
#         y_preds.extend(model.predict(X_val))
#         y_trues.extend(y_val)
#         models.append(model)
#
#     metrics = {
#         "MSE": mean_squared_error(y_trues, y_preds),
#         "RMSE": np.sqrt(mean_squared_error(y_trues, y_preds)),
#         "MAE": mean_absolute_error(y_trues, y_preds),
#         "R2": r2_score(y_trues, y_preds),
#         "Pearson": pearsonr(y_trues, y_preds)[0],
#         "Spearman": spearmanr(y_trues, y_preds)[0]
#     }
#     return models, scaler, metrics
#
#
# if __name__ == "__main__":
#     tsv_path = "/tmp/AbAgCDR/data/final_dataset_train.tsv"
#     start = time.time()
#
#     features, y, df = build_or_load_embeddings(tsv_path)
#     models, scaler, metrics = train_lightgbm(features, y)
#
#     # 保存 pipeline
#     joblib.dump({'models': models, 'scaler': scaler, 'emb_cache': EMB_CACHE}, "esm2_cdr_attention_lgbm_pipeline.pkl")
#     print("✅ Saved pipeline -> esm2_cdr_attention_lgbm_pipeline.pkl")
#
#     # 输出指标
#     print("回归性能指标:")
#     for k, v in metrics.items():
#         print(f"{k}: {v:.4f}")
#
#     print(f"⏱ Total time = {time.time() - start:.1f}s")

# import os
# import time
# import warnings
# warnings.filterwarnings("ignore")
#
# import joblib
# import numpy as np
# import pandas as pd
# import torch
# from tqdm import tqdm
# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import train_test_split, KFold
# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# from scipy.stats import pearsonr, spearmanr
# import lightgbm as lgb
# import esm
#
# # ----------------------------
# # 配置
# # ----------------------------
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# EMB_CACHE = "esm2_cdr_attention_embeddings.pkl"
# N_SPLITS = 5
# CROSS_FEAT_LEN = 10  # Cross Attention 输出固定长度
# TEST_SIZE = 0.15
# VAL_SIZE = 0.15
# RANDOM_STATE = 42
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
# # 2. 序列转 ESM embedding
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
# # 3. 固定长度池化函数
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
# # 5. Cross Attention 特征提取
# # ----------------------------
# def extract_cdr_attention_features(df, tokens_a, tokens_b, tokens_ag):
#     features = []
#     cdr_h = np.concatenate([getCDRPos('H1'), getCDRPos('H2'), getCDRPos('H3')])
#     cdr_l = np.concatenate([getCDRPos('L1'), getCDRPos('L2'), getCDRPos('L3')])
#
#     for i in tqdm(range(len(df)), desc="Extracting CDR Attention Features"):
#         ta = tokens_a[i]
#         tb = tokens_b[i]
#         tag = tokens_ag[i]
#
#         # 简单点积 attention
#         att_h2a = np.matmul(ta.mean(axis=0, keepdims=True), tag.T).flatten()
#         feat_h2a = fixed_length_pool(att_h2a)
#
#         att_l2a = np.matmul(tb.mean(axis=0, keepdims=True), tag.T).flatten()
#         feat_l2a = fixed_length_pool(att_l2a)
#
#         att_a2h = np.matmul(tag.mean(axis=0, keepdims=True), ta.T).flatten()
#         feat_a2h = fixed_length_pool(att_a2h)
#
#         att_a2l = np.matmul(tag.mean(axis=0, keepdims=True), tb.T).flatten()
#         feat_a2l = fixed_length_pool(att_a2l)
#
#         feat = np.concatenate([feat_h2a, feat_l2a, feat_a2h, feat_a2l])
#         features.append(feat)
#
#     return np.array(features)
#
# # ----------------------------
# # 6. 构建或加载 Embedding
# # ----------------------------
# def build_or_load_embeddings(tsv_path, force_recompute=False):
#     if os.path.exists(EMB_CACHE) and not force_recompute:
#         print(f"✅ Loading cached embeddings from {EMB_CACHE}")
#         data = joblib.load(EMB_CACHE)
#         return data['features'], data['y'], data['df']
#
#     print("🔹 Computing embeddings from scratch...")
#     df = pd.read_csv(tsv_path, sep="\t")
#     seq_a = df["antibody_seq_a"].values
#     seq_b = df["antibody_seq_b"].values
#     seq_ag = df["antigen_seq"].values
#     y = df["delta_g"].values
#
#     emb_a, tokens_a = compute_sequence_embeddings(seq_a)
#     emb_b, tokens_b = compute_sequence_embeddings(seq_b)
#     emb_ag, tokens_ag = compute_sequence_embeddings(seq_ag)
#
#     features_cross = extract_cdr_attention_features(df, tokens_a, tokens_b, tokens_ag)
#     features = np.concatenate([emb_a, emb_b, emb_ag, features_cross], axis=1)
#
#     joblib.dump({'features': features, 'y': y, 'df': df}, EMB_CACHE)
#     print(f"✅ Saved embeddings & features to {EMB_CACHE}")
#     return features, y, df
#
# # ----------------------------
# # LightGBM训练
# # ----------------------------
# def train_lightgbm(features, y):
#     # 划分训练/验证/测试
#     X_train_val, X_test, y_train_val, y_test = train_test_split(
#         features, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
#     )
#     X_train, X_val, y_train, y_val = train_test_split(
#         X_train_val, y_train_val, test_size=VAL_SIZE/(1-TEST_SIZE), random_state=RANDOM_STATE
#     )
#
#     # 特征归一化
#     feat_scaler = StandardScaler()
#     X_train_scaled = feat_scaler.fit_transform(X_train)
#     X_val_scaled = feat_scaler.transform(X_val)
#     X_test_scaled = feat_scaler.transform(X_test)
#
#     # 标签归一化
#     label_scaler = StandardScaler()
#     y_train_scaled = label_scaler.fit_transform(y_train.reshape(-1,1)).ravel()
#     y_val_scaled = label_scaler.transform(y_val.reshape(-1,1)).ravel()
#     y_test_scaled = label_scaler.transform(y_test.reshape(-1,1)).ravel()
#
#     # LightGBM模型
#     model = lgb.LGBMRegressor(
#         n_estimators=500,
#         learning_rate=0.05,
#         max_depth=7,
#         min_data_in_leaf=2,
#         subsample=0.8,
#         colsample_bytree=0.8,
#         random_state=RANDOM_STATE,
#         verbose=50  # ⚠️ sklearn API 在这里设置
#     )
#
#     model.fit(
#         X_train_scaled, y_train_scaled,
#         eval_set=[(X_val_scaled, y_val_scaled)],
#         eval_metric='l2'  # ✅ 不加 verbose_eval
#     )
#
#     # PCC 趋势检查
#     y_train_pred_scaled = model.predict(X_train_scaled)
#     y_val_pred_scaled = model.predict(X_val_scaled)
#     y_train_pred_orig = label_scaler.inverse_transform(y_train_pred_scaled.reshape(-1,1))
#     y_val_pred_orig = label_scaler.inverse_transform(y_val_pred_scaled.reshape(-1,1))
#
#     pcc_train = pearsonr(y_train, y_train_pred_orig.flatten())[0]
#     pcc_val = pearsonr(y_val, y_val_pred_orig.flatten())[0]
#
#     if pcc_train < 0:
#         print(f"⚠️ 训练集 PCC 为负 ({pcc_train:.4f})")
#     if pcc_val < 0:
#         print(f"⚠️ 验证集 PCC 为负 ({pcc_val:.4f})")
#
#     # 测试集指标
#     y_test_pred_scaled = model.predict(X_test_scaled)
#     y_test_pred_orig = label_scaler.inverse_transform(y_test_pred_scaled.reshape(-1,1))
#
#     metrics = {
#         "MSE": mean_squared_error(y_test, y_test_pred_orig),
#         "RMSE": np.sqrt(mean_squared_error(y_test, y_test_pred_orig)),
#         "MAE": mean_absolute_error(y_test, y_test_pred_orig),
#         "R2": r2_score(y_test, y_test_pred_orig),
#         "Pearson": pearsonr(y_test, y_test_pred_orig.flatten())[0],
#         "Spearman": spearmanr(y_test, y_test_pred_orig.flatten())[0]
#     }
#
#     return model, feat_scaler, label_scaler, metrics, X_test_scaled, y_test
#
#
# # ----------------------------
# # 8. 主流程
# # ----------------------------
# if __name__ == "__main__":
#     tsv_path = "/tmp/AbAgCDR/data/final_dataset_train.tsv"
#     start = time.time()
#
#     features, y, df = build_or_load_embeddings(tsv_path)
#     model, feat_scaler, label_scaler, metrics, X_test_scaled, y_test = train_lightgbm(features, y)
#
#     # 保存 pipeline
#     joblib.dump({
#         'model': model,
#         'feat_scaler': feat_scaler,
#         'label_scaler': label_scaler,
#         'emb_cache': EMB_CACHE
#     }, "esm2_cdr_attention_lgbm_pipeline.pkl")
#     print("✅ Saved pipeline -> esm2_cdr_attention_lgbm_pipeline.pkl")
#
#     # 输出测试指标
#     print("\n测试集指标（反标准化）：")
#     for k, v in metrics.items():
#         print(f"{k}: {v:.4f}")
#
#     print(f"\n⏱ Total time = {time.time() - start:.1f}s")

# import os
# import time
# import warnings
# warnings.filterwarnings("ignore")
#
# import joblib
# import numpy as np
# import pandas as pd
# import torch
# from tqdm import tqdm
# from sklearn.preprocessing import StandardScaler
# from sklearn.decomposition import PCA
# from sklearn.model_selection import KFold
# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# from scipy.stats import pearsonr, spearmanr
# import lightgbm as lgb
# import esm
#
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# EMB_CACHE = "esm2_cdr_attention_embeddings.pkl"
# N_SPLITS = 5
# CROSS_FEAT_LEN = 10
# RANDOM_STATE = 42
# PCA_DIM = 256
#
# # ----------------------------
# # 1. ESM2模型
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
# def fixed_length_pool(vec, target_len=CROSS_FEAT_LEN):
#     vec_len = len(vec)
#     return vec[:target_len] if vec_len >= target_len else np.pad(vec, (0, target_len - vec_len), 'constant')
#
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
# def extract_cdr_attention_features(df, tokens_a, tokens_b, tokens_ag):
#     features = []
#     for i in tqdm(range(len(df)), desc="Extracting CDR Attention Features"):
#         ta = tokens_a[i]
#         tb = tokens_b[i]
#         tag = tokens_ag[i]
#         att_h2a = np.matmul(ta.mean(axis=0, keepdims=True), tag.T).flatten()
#         att_l2a = np.matmul(tb.mean(axis=0, keepdims=True), tag.T).flatten()
#         att_a2h = np.matmul(tag.mean(axis=0, keepdims=True), ta.T).flatten()
#         att_a2l = np.matmul(tag.mean(axis=0, keepdims=True), tb.T).flatten()
#         feat = np.concatenate([fixed_length_pool(att_h2a),
#                                fixed_length_pool(att_l2a),
#                                fixed_length_pool(att_a2h),
#                                fixed_length_pool(att_a2l)])
#         features.append(feat)
#     return np.array(features)
#
# def build_or_load_embeddings(tsv_path, force_recompute=False):
#     if os.path.exists(EMB_CACHE) and not force_recompute:
#         print(f"✅ Loading cached embeddings from {EMB_CACHE}")
#         data = joblib.load(EMB_CACHE)
#         return data['features'], data['y'], data['df']
#
#     print("🔹 Computing embeddings from scratch...")
#     df = pd.read_csv(tsv_path, sep="\t")
#     seq_a, seq_b, seq_ag = df["antibody_seq_a"].values, df["antibody_seq_b"].values, df["antigen_seq"].values
#     y = df["delta_g"].values
#
#     emb_a, tokens_a = compute_sequence_embeddings(seq_a)
#     emb_b, tokens_b = compute_sequence_embeddings(seq_b)
#     emb_ag, tokens_ag = compute_sequence_embeddings(seq_ag)
#     features_cross = extract_cdr_attention_features(df, tokens_a, tokens_b, tokens_ag)
#     features = np.concatenate([emb_a, emb_b, emb_ag, features_cross], axis=1)
#
#     joblib.dump({'features': features, 'y': y, 'df': df}, EMB_CACHE)
#     print(f"✅ Saved embeddings & features to {EMB_CACHE}")
#     return features, y, df
#
# # ----------------------------
# # 训练
# # ----------------------------
# def train_lightgbm(features, y):
#     # 标签归一化
#     label_scaler = StandardScaler()
#     y_scaled = label_scaler.fit_transform(y.reshape(-1,1)).ravel()
#
#     # 特征归一化 + PCA
#     feat_scaler = StandardScaler()
#     features_scaled = feat_scaler.fit_transform(features)
#     pca = PCA(n_components=PCA_DIM, random_state=RANDOM_STATE)
#     features_pca = pca.fit_transform(features_scaled)
#
#     kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
#     oof_preds = np.zeros_like(y_scaled)
#     models = []
#
#     for fold, (train_idx, val_idx) in enumerate(kf.split(features_pca)):
#         print(f"🔹 Fold {fold+1}/{N_SPLITS}")
#         X_train, X_val = features_pca[train_idx], features_pca[val_idx]
#         y_train, y_val = y_scaled[train_idx], y_scaled[val_idx]
#
#         model = lgb.LGBMRegressor(
#             n_estimators=5000,
#             learning_rate=0.01,
#             max_depth=6,
#             min_data_in_leaf=5,
#             subsample=0.9,
#             colsample_bytree=0.8,
#             reg_alpha=0.1,
#             reg_lambda=0.1,
#             random_state=RANDOM_STATE
#         )
#
#         # 使用 sklearn API 的 early stopping 回调
#         model.fit(
#             X_train, y_train,
#             eval_set=[(X_val, y_val)],
#             eval_metric='l2',
#             callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
#         )
#
#         oof_preds[val_idx] = model.predict(X_val)
#         models.append(model)
#
#     y_pred_orig = label_scaler.inverse_transform(oof_preds.reshape(-1,1))
#     metrics = {
#         "MSE": mean_squared_error(y, y_pred_orig),
#         "RMSE": np.sqrt(mean_squared_error(y, y_pred_orig)),
#         "MAE": mean_absolute_error(y, y_pred_orig),
#         "R2": r2_score(y, y_pred_orig),
#         "Pearson": pearsonr(y, y_pred_orig.flatten())[0],
#         "Spearman": spearmanr(y, y_pred_orig.flatten())[0]
#     }
#
#     return models, feat_scaler, label_scaler, pca, metrics
#
# # ----------------------------
# # 主流程
# # ----------------------------
# if __name__ == "__main__":
#     tsv_path = "/tmp/AbAgCDR/data/final_dataset_train.tsv"
#     start = time.time()
#
#     features, y, df = build_or_load_embeddings(tsv_path)
#     models, feat_scaler, label_scaler, pca, metrics = train_lightgbm(features, y)
#
#     joblib.dump({
#         'models': models,
#         'feat_scaler': feat_scaler,
#         'label_scaler': label_scaler,
#         'pca': pca,
#         'emb_cache': EMB_CACHE
#     }, "esm2_cdr_attention_lgbm_pipeline.pkl")
#     print("✅ Saved pipeline -> esm2_cdr_attention_lgbm_pipeline.pkl")
#
#     print("\n测试指标（反标准化）：")
#     for k, v in metrics.items():
#         print(f"{k}: {v:.4f}")
#     print(f"\n⏱ Total time = {time.time() - start:.1f}s")
#
#

# import os
# import time
# import warnings
# warnings.filterwarnings("ignore")
#
# import joblib
# import numpy as np
# import pandas as pd
# import torch
# from tqdm import tqdm
# from sklearn.preprocessing import StandardScaler
# from sklearn.decomposition import PCA
# from sklearn.model_selection import KFold
# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# from scipy.stats import pearsonr, spearmanr
# import lightgbm as lgb
# import esm
#
# # ----------------------------
# # 配置
# # ----------------------------
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# EMB_CACHE = "esm2_cdr_attention_embeddings.pkl"
# N_SPLITS = 5
# CROSS_FEAT_LEN = 10
# RANDOM_STATE = 42
# PCA_DIM = 256
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
# # 2. ESM embedding 函数
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
# def fixed_length_pool(vec, target_len=CROSS_FEAT_LEN):
#     vec_len = len(vec)
#     return vec[:target_len] if vec_len >= target_len else np.pad(vec, (0, target_len - vec_len), 'constant')
#
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
# def extract_cdr_attention_features(df, tokens_a, tokens_b, tokens_ag):
#     features = []
#     for i in tqdm(range(len(df)), desc="Extracting CDR Attention Features"):
#         ta, tb, tag = tokens_a[i], tokens_b[i], tokens_ag[i]
#         att_h2a = np.matmul(ta.mean(axis=0, keepdims=True), tag.T).flatten()
#         att_l2a = np.matmul(tb.mean(axis=0, keepdims=True), tag.T).flatten()
#         att_a2h = np.matmul(tag.mean(axis=0, keepdims=True), ta.T).flatten()
#         att_a2l = np.matmul(tag.mean(axis=0, keepdims=True), tb.T).flatten()
#         feat = np.concatenate([fixed_length_pool(att_h2a),
#                                fixed_length_pool(att_l2a),
#                                fixed_length_pool(att_a2h),
#                                fixed_length_pool(att_a2l)])
#         features.append(feat)
#     return np.array(features)
#
# def build_or_load_embeddings(tsv_path, force_recompute=False):
#     if os.path.exists(EMB_CACHE) and not force_recompute:
#         print(f"✅ Loading cached embeddings from {EMB_CACHE}")
#         data = joblib.load(EMB_CACHE)
#         return data['features'], data['y'], data['df']
#
#     print("🔹 Computing embeddings from scratch...")
#     df = pd.read_csv(tsv_path, sep="\t")
#     seq_a, seq_b, seq_ag = df["antibody_seq_a"].values, df["antibody_seq_b"].values, df["antigen_seq"].values
#     y = df["delta_g"].values
#
#     emb_a, tokens_a = compute_sequence_embeddings(seq_a)
#     emb_b, tokens_b = compute_sequence_embeddings(seq_b)
#     emb_ag, tokens_ag = compute_sequence_embeddings(seq_ag)
#     features_cross = extract_cdr_attention_features(df, tokens_a, tokens_b, tokens_ag)
#     features = np.concatenate([emb_a, emb_b, emb_ag, features_cross], axis=1)
#
#     joblib.dump({'features': features, 'y': y, 'df': df}, EMB_CACHE)
#     print(f"✅ Saved embeddings & features to {EMB_CACHE}")
#     return features, y, df
#
# # ----------------------------
# # LightGBM 训练
# # ----------------------------
# def train_lightgbm(features, y):
#     # 标签归一化
#     label_scaler = StandardScaler()
#     y_scaled = label_scaler.fit_transform(y.reshape(-1,1)).ravel()
#
#     # 特征归一化 + PCA
#     feat_scaler = StandardScaler()
#     features_scaled = feat_scaler.fit_transform(features)
#     pca = PCA(n_components=PCA_DIM, random_state=RANDOM_STATE)
#     features_pca = pca.fit_transform(features_scaled)
#
#     kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
#     oof_preds = np.zeros_like(y_scaled)
#     models = []
#
#     for fold, (train_idx, val_idx) in enumerate(kf.split(features_pca)):
#         print(f"🔹 Fold {fold+1}/{N_SPLITS}")
#         X_train, X_val = features_pca[train_idx], features_pca[val_idx]
#         y_train, y_val = y_scaled[train_idx], y_scaled[val_idx]
#
#         model = lgb.LGBMRegressor(
#             n_estimators=5000,
#             learning_rate=0.01,
#             max_depth=6,
#             min_data_in_leaf=5,
#             subsample=0.9,
#             colsample_bytree=0.8,
#             reg_alpha=0.1,
#             reg_lambda=0.1,
#             random_state=RANDOM_STATE
#         )
#
#         model.fit(
#             X_train, y_train,
#             eval_set=[(X_val, y_val)],
#             eval_metric='l2',
#             callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
#         )
#
#         oof_preds[val_idx] = model.predict(X_val)
#         models.append(model)
#
#     # 反标准化
#     y_pred_orig = label_scaler.inverse_transform(oof_preds.reshape(-1,1))
#     metrics = {
#         "MSE": mean_squared_error(y, y_pred_orig),
#         "RMSE": np.sqrt(mean_squared_error(y, y_pred_orig)),
#         "MAE": mean_absolute_error(y, y_pred_orig),
#         "R2": r2_score(y, y_pred_orig),
#         "Pearson": pearsonr(y, y_pred_orig.flatten())[0],
#         "Spearman": spearmanr(y, y_pred_orig.flatten())[0]
#     }
#
#     return models, feat_scaler, label_scaler, pca, metrics
#
# # ----------------------------
# # 主流程
# # ----------------------------
# if __name__ == "__main__":
#     tsv_path = "/tmp/AbAgCDR/data/final_dataset_train.tsv"
#     start = time.time()
#
#     features, y, df = build_or_load_embeddings(tsv_path)
#     models, feat_scaler, label_scaler, pca, metrics = train_lightgbm(features, y)
#
#     # 保存 pipeline
#     joblib.dump({
#         'models': models,
#         'feat_scaler': feat_scaler,
#         'label_scaler': label_scaler,
#         'pca': pca,
#         'emb_cache': EMB_CACHE
#     }, "esm2_cdr_attention_lgbm_pipeline.pkl")
#     print("✅ Saved pipeline -> esm2_cdr_attention_lgbm_pipeline.pkl")
#
#     # 打印训练指标
#     print("\n训练指标（反标准化 ΔG）：")
#     for k, v in metrics.items():
#         print(f"{k}: {v:.4f}")
#     print(f"\n⏱ Total time = {time.time() - start:.1f}s")
#
#
# # ----------------------------
# # 测试函数（任意 TSV）
# # ----------------------------
# def test_pipeline(tsv_path, pipeline_path="esm2_cdr_attention_lgbm_pipeline.pkl"):
#     print(f"\n--- Testing {tsv_path} ---")
#     pipeline = joblib.load(pipeline_path)
#     models = pipeline['models']
#     feat_scaler = pipeline['feat_scaler']
#     label_scaler = pipeline['label_scaler']
#     pca = pipeline['pca']
#     emb_cache = pipeline['emb_cache']
#
#     df = pd.read_csv(tsv_path, sep="\t")
#     y_true = df["delta_g"].values.reshape(-1,1)
#
#     # ESM embedding
#     seq_a, seq_b, seq_ag = df["antibody_seq_a"].values, df["antibody_seq_b"].values, df["antigen_seq"].values
#     emb_a, tokens_a = compute_sequence_embeddings(seq_a)
#     emb_b, tokens_b = compute_sequence_embeddings(seq_b)
#     emb_ag, tokens_ag = compute_sequence_embeddings(seq_ag)
#     features_cross = extract_cdr_attention_features(df, tokens_a, tokens_b, tokens_ag)
#     features = np.concatenate([emb_a, emb_b, emb_ag, features_cross], axis=1)
#
#     # 特征归一化 + PCA
#     features_scaled = feat_scaler.transform(features)
#     features_pca = pca.transform(features_scaled)
#
#     # 多模型预测平均
#     preds_scaled = np.mean([model.predict(features_pca) for model in models], axis=0)
#     y_pred = label_scaler.inverse_transform(preds_scaled.reshape(-1,1))
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
import time
import warnings
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr, spearmanr
import lightgbm as lgb
import esm

# ----------------------------
# 配置
# ----------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EMB_CACHE = "esm2_cdr_attention_embeddings.pkl"
N_SPLITS = 5
CROSS_FEAT_LEN = 10
RANDOM_STATE = 42
PCA_DIM = 256

# ----------------------------
# 1. 加载 ESM2 模型
# ----------------------------
print("🔹 Loading ESM-2 model...")
esm_model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
esm_model = esm_model.to(DEVICE)
batch_converter = alphabet.get_batch_converter()
esm_model.eval()

# ----------------------------
# 2. ESM embedding 函数
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

def fixed_length_pool(vec, target_len=CROSS_FEAT_LEN):
    vec_len = len(vec)
    return vec[:target_len] if vec_len >= target_len else np.pad(vec, (0, target_len - vec_len), 'constant')

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

def extract_cdr_attention_features(df, tokens_a, tokens_b, tokens_ag):
    features = []
    for i in tqdm(range(len(df)), desc="Extracting CDR Attention Features"):
        ta, tb, tag = tokens_a[i], tokens_b[i], tokens_ag[i]
        att_h2a = np.matmul(ta.mean(axis=0, keepdims=True), tag.T).flatten()
        att_l2a = np.matmul(tb.mean(axis=0, keepdims=True), tag.T).flatten()
        att_a2h = np.matmul(tag.mean(axis=0, keepdims=True), ta.T).flatten()
        att_a2l = np.matmul(tag.mean(axis=0, keepdims=True), tb.T).flatten()
        feat = np.concatenate([fixed_length_pool(att_h2a),
                               fixed_length_pool(att_l2a),
                               fixed_length_pool(att_a2h),
                               fixed_length_pool(att_a2l)])
        features.append(feat)
    return np.array(features)

def build_or_load_embeddings(tsv_path, force_recompute=False):
    if os.path.exists(EMB_CACHE) and not force_recompute:
        print(f"✅ Loading cached embeddings from {EMB_CACHE}")
        data = joblib.load(EMB_CACHE)
        return data['features'], data['y'], data['df']

    print("🔹 Computing embeddings from scratch...")
    df = pd.read_csv(tsv_path, sep="\t")
    seq_a, seq_b, seq_ag = df["antibody_seq_a"].values, df["antibody_seq_b"].values, df["antigen_seq"].values
    y = df["delta_g"].values

    emb_a, tokens_a = compute_sequence_embeddings(seq_a)
    emb_b, tokens_b = compute_sequence_embeddings(seq_b)
    emb_ag, tokens_ag = compute_sequence_embeddings(seq_ag)
    features_cross = extract_cdr_attention_features(df, tokens_a, tokens_b, tokens_ag)
    features = np.concatenate([emb_a, emb_b, emb_ag, features_cross], axis=1)

    joblib.dump({'features': features, 'y': y, 'df': df}, EMB_CACHE)
    print(f"✅ Saved embeddings & features to {EMB_CACHE}")
    return features, y, df

# ----------------------------
# LightGBM 训练（标准化 ΔG）
# ----------------------------
def train_lightgbm(features, y):
    # 标签归一化
    label_scaler = StandardScaler()
    y_scaled = label_scaler.fit_transform(y.reshape(-1,1)).ravel()

    # 特征归一化 + PCA
    feat_scaler = StandardScaler()
    features_scaled = feat_scaler.fit_transform(features)
    pca = PCA(n_components=PCA_DIM, random_state=RANDOM_STATE)
    features_pca = pca.fit_transform(features_scaled)

    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    oof_preds = np.zeros_like(y_scaled)
    models = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(features_pca)):
        print(f"🔹 Fold {fold+1}/{N_SPLITS}")
        X_train, X_val = features_pca[train_idx], features_pca[val_idx]
        y_train, y_val = y_scaled[train_idx], y_scaled[val_idx]

        model = lgb.LGBMRegressor(
            n_estimators=5000,
            learning_rate=0.01,
            max_depth=6,
            min_data_in_leaf=5,
            subsample=0.9,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=RANDOM_STATE
        )

        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric='l2',
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
        )

        oof_preds[val_idx] = model.predict(X_val)
        models.append(model)

    # 不反标准化，直接输出标准化指标
    metrics = {
        "MSE": mean_squared_error(y_scaled, oof_preds),
        "RMSE": np.sqrt(mean_squared_error(y_scaled, oof_preds)),
        "MAE": mean_absolute_error(y_scaled, oof_preds),
        "R2": r2_score(y_scaled, oof_preds),
        "Pearson": pearsonr(y_scaled, oof_preds)[0],
        "Spearman": spearmanr(y_scaled, oof_preds)[0]
    }

    return models, feat_scaler, label_scaler, pca, metrics

# ----------------------------
# 测试函数（任意 TSV，标准化 ΔG）
# ----------------------------
def test_pipeline(tsv_path, pipeline_path="esm2_cdr_attention_lgbm_pipeline.pkl"):
    print(f"\n--- Testing {tsv_path} ---")
    pipeline = joblib.load(pipeline_path)
    models = pipeline['models']
    feat_scaler = pipeline['feat_scaler']
    label_scaler = pipeline['label_scaler']
    pca = pipeline['pca']

    df = pd.read_csv(tsv_path, sep="\t")
    y_true_scaled = label_scaler.transform(df["delta_g"].values.reshape(-1,1)).ravel()

    # ESM embedding
    seq_a, seq_b, seq_ag = df["antibody_seq_a"].values, df["antibody_seq_b"].values, df["antigen_seq"].values
    emb_a, tokens_a = compute_sequence_embeddings(seq_a)
    emb_b, tokens_b = compute_sequence_embeddings(seq_b)
    emb_ag, tokens_ag = compute_sequence_embeddings(seq_ag)
    features_cross = extract_cdr_attention_features(df, tokens_a, tokens_b, tokens_ag)
    features = np.concatenate([emb_a, emb_b, emb_ag, features_cross], axis=1)

    # 特征归一化 + PCA
    features_scaled = feat_scaler.transform(features)
    features_pca = pca.transform(features_scaled)

    # 多模型预测平均
    preds_scaled = np.mean([model.predict(features_pca) for model in models], axis=0)

    # 输出指标（标准化 ΔG）
    mse = mean_squared_error(y_true_scaled, preds_scaled)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true_scaled, preds_scaled)
    r2 = r2_score(y_true_scaled, preds_scaled)
    pearson = pearsonr(y_true_scaled, preds_scaled)[0]
    spearman = spearmanr(y_true_scaled, preds_scaled)[0]

    print(f"Samples: {len(y_true_scaled)}")
    print(f"MSE = {mse:.4f}, RMSE = {rmse:.4f}, MAE = {mae:.4f}, R² = {r2:.4f}")
    print(f"Pearson = {pearson:.4f}, Spearman = {spearman:.4f}")

# ----------------------------
# 主流程
# ----------------------------
if __name__ == "__main__":
    tsv_path = "/tmp/AbAgCDR/data/final_dataset_train.tsv"
    start = time.time()

    features, y, df = build_or_load_embeddings(tsv_path)
    models, feat_scaler, label_scaler, pca, metrics = train_lightgbm(features, y)

    # 保存 pipeline
    joblib.dump({
        'models': models,
        'feat_scaler': feat_scaler,
        'label_scaler': label_scaler,
        'pca': pca,
        'emb_cache': EMB_CACHE
    }, "esm2_cdr_attention_lgbm_pipeline.pkl")
    print("✅ Saved pipeline -> esm2_cdr_attention_lgbm_pipeline.pkl")

    # 打印训练指标（标准化 ΔG）
    print("\n训练指标（标准化 ΔG）：")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
    print(f"\n⏱ Total time = {time.time() - start:.1f}s")


