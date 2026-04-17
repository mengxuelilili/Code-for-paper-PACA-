# import warnings
# warnings.filterwarnings('ignore')

# import numpy as np
# import pandas as pd
# import joblib
# import itertools
# import copy
# from pathlib import Path
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# from scipy.stats import pearsonr
# import lightgbm as lgb

# # ===================== #
# # 1. 特征提取器（BAA + 理化 + 全局统计）
# # ===================== #
# AA_LIST = list("ACDEFGHIKLMNPQRSTVWY")

# HYDROPATHY = {
#     'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5,
#     'Q': -3.5, 'E': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5,
#     'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6,
#     'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2
# }

# def seq_to_baa(seq):
#     counts = np.zeros(20)
#     for aa in seq:
#         if aa in AA_LIST:
#             idx = AA_LIST.index(aa)
#             counts[idx] += 1
#     return counts / len(seq) if len(seq) > 0 else counts

# def seq_to_physchem(seq):
#     hydro_vals = [HYDROPATHY.get(aa, 0.0) for aa in seq]
#     charge_vals = [1.0 if aa in 'RK' else -1.0 if aa in 'DE' else 0.0 for aa in seq]
#     mean_hydro = np.mean(hydro_vals) if hydro_vals else 0.0
#     mean_charge = np.mean(charge_vals) if charge_vals else 0.0
#     return np.array([mean_hydro, mean_charge])

# def seq_to_global(seq):
#     total_hydro = sum(HYDROPATHY.get(aa, 0.0) for aa in seq)
#     net_charge = sum(1 for aa in seq if aa in 'RK') - sum(1 for aa in seq if aa in 'DE')
#     length = len(seq)
#     return np.array([total_hydro, net_charge, length])

# def extract_features(light_seq, heavy_seq, antigen_seq):
#     baa = np.concatenate([seq_to_baa(s) for s in [light_seq, heavy_seq, antigen_seq]])
#     phys = np.concatenate([seq_to_physchem(s) for s in [light_seq, heavy_seq, antigen_seq]])
#     glob = np.concatenate([seq_to_global(s) for s in [light_seq, heavy_seq, antigen_seq]])
#     return np.concatenate([baa, phys, glob])  # 60 + 6 + 9 = 75 维

# def load_and_encode_tsv(tsv_path):
#     df = pd.read_csv(tsv_path, sep='\t', on_bad_lines='skip')
#     required_cols = ['antibody_seq_a', 'antibody_seq_b', 'antigen_seq', 'delta_g']
#     for col in required_cols:
#         if col not in df.columns:
#             raise ValueError(f"Missing column '{col}' in {tsv_path}")
    
#     # 过滤 ΔG 异常值（生物合理范围）
#     df = df[(df['delta_g'] >= -20) & (df['delta_g'] <= 5)]
    
#     samples = []
#     for _, row in df.iterrows():
#         a = str(row['antibody_seq_a'])
#         b = str(row['antibody_seq_b'])
#         ag = str(row['antigen_seq'])
#         dg = float(row['delta_g'])
#         if a and b and ag and not pd.isna(dg):
#             samples.append((a, b, ag, dg))
#     return samples

# def split_samples(samples, test_size=0.2, val_size=0.2, random_state=42):
#     train_val, test = train_test_split(samples, test_size=test_size, random_state=random_state)
#     train, val = train_test_split(train_val, test_size=val_size/(1-test_size), random_state=random_state)
#     return train, val, test

# def compute_metrics(y_true, y_pred):
#     mse = mean_squared_error(y_true, y_pred)
#     rmse = np.sqrt(mse)
#     mae = mean_absolute_error(y_true, y_pred)
#     r2 = r2_score(y_true, y_pred)
#     try:
#         pcc = pearsonr(y_true, y_pred)[0]
#     except:
#         pcc = np.nan
#     return {'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'R2': r2, 'PCC': pcc}

# def evaluate_lgb(model, X, y, y_low, y_high):
#     """反归一化到原始 ΔG 空间"""
#     y_pred_norm = model.predict(X)
#     y_true_orig = y * (y_high - y_low) + y_low
#     y_pred_orig = y_pred_norm * (y_high - y_low) + y_low
#     return compute_metrics(y_true_orig, y_pred_orig)

# # ===================== #
# # 2. Main Function
# # ===================== #
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

#     print("🚀 Starting FINAL OPTIMIZED LightGBM pipeline...")

#     # Step 1: Load data
#     all_train_samples = []
#     all_val_samples = []
#     all_test_samples = []
#     dataset_test_splits = {}

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
#         dataset_name = Path(tsv_path).stem
#         dataset_test_splits[dataset_name] = test

#     print(f"Total Train: {len(all_train_samples)}, Val: {len(all_val_samples)}, Test: {len(all_test_samples)}")

#     # Step 2: Min-Max 归一化（带截断）
#     y_train_raw = np.array([s[3] for s in all_train_samples])
#     y_low, y_high = np.percentile(y_train_raw, [1, 99])  # 1% 和 99% 分位数
#     print(f"ΔG range (1%-99%): [{y_low:.2f}, {y_high:.2f}]")

#     def normalize_label(samples):
#         out = []
#         for a, b, ag, dg in samples:
#             dg_clip = np.clip(dg, y_low, y_high)
#             dg_norm = (dg_clip - y_low) / (y_high - y_low)  # 映射到 [0, 1]
#             out.append((a, b, ag, dg_norm))
#         return out

#     all_train_norm = normalize_label(all_train_samples)
#     all_val_norm = normalize_label(all_val_samples)
#     all_test_norm = normalize_label(all_test_samples)
#     dataset_test_splits_norm = {k: normalize_label(v) for k, v in dataset_test_splits.items()}

#     # Step 3: Extract features
#     def samples_to_xy(samples):
#         X, y = [], []
#         for a, b, ag, dg in samples:
#             feat = extract_features(a, b, ag)
#             X.append(feat)
#             y.append(dg)
#         return np.array(X), np.array(y)

#     X_train, y_train = samples_to_xy(all_train_norm)
#     X_val, y_val = samples_to_xy(all_val_norm)

#     print(f"Feature dimension: {X_train.shape[1]}")  # 应为 75

#     # Step 4: Optimized hyperparameters (for MAE + robustness)
#     param_grid = {
#         'learning_rate': [0.001, 0.005, 0.0001],
#         'max_depth': [5],
#         'num_leaves': [31],
#         'min_data_in_leaf': [20],
#         'feature_fraction': [0.8],
#         'bagging_fraction': [0.8],
#         'bagging_freq': [3],
#         'lambda_l1': [0.0],
#         'lambda_l2': [0.1],
#         'num_iterations': [50],
#         'early_stopping_rounds': [5]
#     }

#     keys, values = zip(*param_grid.items())
#     param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

#     best_score = np.inf  # 注意：现在用 MAE，越小越好！
#     best_params = None
#     best_model = None
#     save_path = Path(CONFIG["run_dir"]) / f"2best_lightgbm_final_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.pkl"
#     save_path.parent.mkdir(parents=True, exist_ok=True)

#     for trial_idx, params in enumerate(param_combinations):
#         print(f"\n{'='*50}\nTrial {trial_idx+1}/{len(param_combinations)} | Params: {params}\n{'='*50}")

#         train_data = lgb.Dataset(X_train, label=y_train)
#         val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

#         model = lgb.train(
#             params={
#                 'objective': 'regression_l1',   # ← 关键：MAE loss
#                 'metric': 'mae',
#                 'verbose': -1,
#                 **{k: v for k, v in params.items() if k not in ['num_iterations', 'early_stopping_rounds']}
#             },
#             train_set=train_data,
#             num_boost_round=params['num_iterations'],
#             valid_sets=[val_data],
#             callbacks=[
#                 lgb.early_stopping(stopping_rounds=params['early_stopping_rounds'], verbose=False),
#                 lgb.log_evaluation(0)
#             ]
#         )

#         print(f"✅ Actual boosting rounds: {model.best_iteration}")

#         val_metrics = evaluate_lgb(model, X_val, y_val, y_low, y_high)
#         score = val_metrics['MAE']  # ← 以 MAE 为选择标准
#         print(f"✅ Val MAE: {score:.4f}, PCC: {val_metrics['PCC']:.4f}")

#         if score < best_score:
#             best_score = score
#             best_params = copy.deepcopy(params)
#             best_model = model
#             joblib.dump({
#                 'model': best_model,
#                 'y_low': y_low,
#                 'y_high': y_high,
#                 'params': best_params
#             }, save_path)
#             print(f"🎉 Best model saved at {save_path}")

#     # Step 5: Final evaluations
#     print("\n" + "="*70)
#     print("🔍 FINAL TEST RESULTS (Absolute Error Focused)")
#     print("="*70)

#     for name, test_samples in dataset_test_splits_norm.items():
#         X_test, y_test = samples_to_xy(test_samples)
#         metrics = evaluate_lgb(best_model, X_test, y_test, y_low, y_high)
#         print(f"\n{name.upper()} TEST → "
#               f"R²: {metrics['R2']:.4f}, "
#               f"MSE: {metrics['MSE']:.4f}, "
#               f"RMSE: {metrics['RMSE']:.4f}, "
#               f"MAE: {metrics['MAE']:.4f}, "
#               f"PCC: {metrics['PCC']:.4f}")

#     X_test_merged, y_test_merged = samples_to_xy(all_test_norm)
#     merged_metrics = evaluate_lgb(best_model, X_test_merged, y_test_merged, y_low, y_high)
#     print(f"\n🏆 MERGED INTERNAL TEST → "
#           f"R²: {merged_metrics['R2']:.4f}, "
#           f"MSE: {merged_metrics['MSE']:.4f}, "
#           f"RMSE: {merged_metrics['RMSE']:.4f}, "
#           f"MAE: {merged_metrics['MAE']:.4f}, "
#           f"PCC: {merged_metrics['PCC']:.4f}")

#     # print("\n🧪 Loading benchmark dataset...")
#     # bench_samples_raw = load_and_encode_tsv(benchmark_tsv)
#     # bench_samples_norm = normalize_label(bench_samples_raw)
#     # X_bench, y_bench = samples_to_xy(bench_samples_norm)
#     # bench_metrics = evaluate_lgb(best_model, X_bench, y_bench, y_low, y_high)
#     # print(f"\n🎯 BENCHMARK TEST → "
#     #       f"R²: {bench_metrics['R2']:.4f}, "
#     #       f"MSE: {bench_metrics['MSE']:.4f}, "
#     #       f"RMSE: {bench_metrics['RMSE']:.4f}, "
#     #       f"MAE: {bench_metrics['MAE']:.4f}, "
#     #       f"PCC: {bench_metrics['PCC']:.4f}")

# if __name__ == "__main__":
#     main()

# import warnings
# warnings.filterwarnings('ignore')

# import numpy as np
# import pandas as pd
# import joblib
# import itertools
# import copy
# from pathlib import Path
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# from scipy.stats import pearsonr
# import lightgbm as lgb
# from collections import Counter

# # ===================== #
# # 1. 特征提取器 (保持不变)
# # ===================== #
# AA_LIST = list("ACDEFGHIKLMNPQRSTVWY")
# AA_TO_IDX = {aa: i for i, aa in enumerate(AA_LIST)}

# HYDROPATHY = {
#     'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5,
#     'Q': -3.5, 'E': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5,
#     'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6,
#     'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2
# }

# def seq_to_baa(seq):
#     counts = np.zeros(20)
#     valid_len = 0
#     for aa in seq:
#         if aa in AA_LIST:
#             counts[AA_TO_IDX[aa]] += 1
#             valid_len += 1
#     return counts / valid_len if valid_len > 0 else counts

# def seq_to_physchem_detailed(seq):
#     if not seq:
#         return np.zeros(6)
#     hydro_vals = [HYDROPATHY.get(aa, 0.0) for aa in seq]
#     charge_vals = [1.0 if aa in 'RK' else -1.0 if aa in 'DE' else 0.0 for aa in seq]
#     feats = [
#         np.mean(hydro_vals), np.std(hydro_vals),
#         np.mean(charge_vals), np.sum(charge_vals),
#         np.max(hydro_vals), np.min(hydro_vals)
#     ]
#     return np.array(feats)

# def extract_features_v2(light_seq, heavy_seq, antigen_seq):
#     seqs = [light_seq, heavy_seq, antigen_seq]
#     baa = np.concatenate([seq_to_baa(s) for s in seqs])
#     phys = np.concatenate([seq_to_physchem_detailed(s) for s in seqs])
    
#     glob_feats = []
#     for s in seqs:
#         total_hydro = sum(HYDROPATHY.get(aa, 0.0) for aa in s)
#         net_charge = sum(1 for aa in s if aa in 'RK') - sum(1 for aa in s if aa in 'DE')
#         length = len(s)
#         glob_feats.extend([total_hydro, net_charge, length])
#     glob = np.array(glob_feats)
    
#     return np.concatenate([baa, phys, glob])

# def load_and_encode_tsv(tsv_path):
#     df = pd.read_csv(tsv_path, sep='\t', on_bad_lines='skip')
    
#     # 列名兼容处理
#     if 'antibody_seq_b' not in df.columns and 'heavy_seq' in df.columns:
#         df['antibody_seq_b'] = df['heavy_seq']
#     if 'antibody_seq_a' not in df.columns and 'light_seq' in df.columns:
#         df['antibody_seq_a'] = df['light_seq']
        
#     required_cols = ['antibody_seq_a', 'antibody_seq_b', 'antigen_seq', 'delta_g']
#     for col in required_cols:
#         if col not in df.columns:
#             raise ValueError(f"Missing column '{col}' in {tsv_path}")
    
#     # ⚠️ 注意：这里不要过度裁剪，让归一化步骤去处理边界
#     df['delta_g'] = pd.to_numeric(df['delta_g'], errors='coerce')
#     df = df.dropna(subset=['delta_g'])
    
#     samples = []
#     for _, row in df.iterrows():
#         a = str(row.get('antibody_seq_a', ''))
#         b = str(row.get('antibody_seq_b', ''))
#         ag = str(row.get('antigen_seq', ''))
#         dg = float(row['delta_g'])
        
#         if a and b and ag and len(a) > 5 and len(ag) > 5:
#             samples.append((a, b, ag, dg))
            
#     return samples

# def split_samples(samples, test_size=0.2, val_size=0.2, random_state=42):
#     train_val, test = train_test_split(samples, test_size=test_size, random_state=random_state)
#     train, val = train_test_split(train_val, test_size=val_size/(1-test_size), random_state=random_state)
#     return train, val, test

# def compute_metrics(y_true, y_pred):
#     mse = mean_squared_error(y_true, y_pred)
#     rmse = np.sqrt(mse)
#     mae = mean_absolute_error(y_true, y_pred)
#     r2 = r2_score(y_true, y_pred)
#     try:
#         pcc, _ = pearsonr(y_true, y_pred)
#     except:
#         pcc = 0.0
#     return {'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'R2': r2, 'PCC': pcc}

# # ===================== #
# # 2. 核心修改：固定归一化范围
# # ===================== #
# Y_LOW_FIXED = -15.5
# Y_HIGH_FIXED = -2.0
# Y_RANGE = Y_HIGH_FIXED - Y_LOW_FIXED

# print(f"🔒 Using Fixed Normalization Range: [{Y_LOW_FIXED}, {Y_HIGH_FIXED}]")

# def normalize_label_fixed(dg_value):
#     """将原始值裁剪并映射到 [0, 1]"""
#     dg_clip = np.clip(dg_value, Y_LOW_FIXED, Y_HIGH_FIXED)
#     return (dg_clip - Y_LOW_FIXED) / Y_RANGE

# def denormalize_label_fixed(dg_norm):
#     """将 [0, 1] 映射回原始值"""
#     return dg_norm * Y_RANGE + Y_LOW_FIXED

# def evaluate_lgb_fixed(model, X, y_true_norm):
#     """预测并反归一化，然后计算指标"""
#     y_pred_norm = model.predict(X)
    
#     # 反归一化
#     y_true_orig = denormalize_label_fixed(y_true_norm)
#     y_pred_orig = denormalize_label_fixed(y_pred_norm)
    
#     return compute_metrics(y_true_orig, y_pred_orig)

# # def main():
# #     train_tsvs = {
# #         "/tmp/AbAgCDR/data/final_dataset_train.tsv": 1.0,
# #         "/tmp/AbAgCDR/data/pairs_seq_skempi_clean.tsv": 1.0,
# #         "/tmp/AbAgCDR/data/pairs_seq_sabdab_clean.tsv": 1.0,
# #         "/tmp/AbAgCDR/data/pairs_seq_abbind2_clean.tsv": 1.0,
# #     }

# #     CONFIG = {"test_size": 0.2, "val_size": 0.2, "seed": 42, "run_dir": "../runs"}
# #     print("🚀 Starting LightGBM with Fixed Normalization [-14, -2]...")

# #     all_train_samples = []
# #     all_val_samples = []
# #     all_test_samples = []
# #     dataset_test_splits = {}

# #     for tsv_path, weight in train_tsvs.items():
# #         if not Path(tsv_path).exists():
# #             print(f"Skipping missing file: {tsv_path}")
# #             continue
# #         print(f"Loading {tsv_path}...")
# #         samples = load_and_encode_tsv(tsv_path)
# #         if not samples:
# #             continue
        
# #         # 检查数据分布是否严重超出设定范围
# #         dgs = [s[3] for s in samples]
# #         out_of_range = sum(1 for x in dgs if x < Y_LOW_FIXED or x > Y_HIGH_FIXED)
# #         if out_of_range > 0:
# #             print(f"  ⚠️ Warning: {out_of_range}/{len(samples)} samples are OUTSIDE [{Y_LOW_FIXED}, {Y_HIGH_FIXED}]. They will be clipped!")

# #         train, val, test = split_samples(samples, CONFIG["test_size"], CONFIG["val_size"], CONFIG["seed"])
# #         all_train_samples.extend(train)
# #         all_val_samples.extend(val)
# #         all_test_samples.extend(test)
# #         dataset_name = Path(tsv_path).stem
# #         dataset_test_splits[dataset_name] = test

# #     if not all_train_samples:
# #         print("❌ No data loaded!")
# #         return

# #     print(f"Total Train: {len(all_train_samples)}, Val: {len(all_val_samples)}, Test: {len(all_test_samples)}")

# #     # 应用固定归一化
# #     def normalize_samples(samples):
# #         out = []
# #         for a, b, ag, dg in samples:
# #             dg_norm = normalize_label_fixed(dg)
# #             out.append((a, b, ag, dg_norm))
# #         return out

# #     all_train_norm = normalize_samples(all_train_samples)
# #     all_val_norm = normalize_samples(all_val_samples)
# #     all_test_norm = normalize_samples(all_test_samples)
# #     dataset_test_splits_norm = {k: normalize_samples(v) for k, v in dataset_test_splits.items()}

# #     def samples_to_xy(samples):
# #         X, y = [], []
# #         for a, b, ag, dg_norm in samples:
# #             feat = extract_features_v2(a, b, ag)
# #             X.append(feat)
# #             y.append(dg_norm)
# #         return np.array(X), np.array(y)

# #     print("Extracting features...")
# #     X_train, y_train = samples_to_xy(all_train_norm)
# #     X_val, y_val = samples_to_xy(all_val_norm)
# #     print(f"Feature shape: {X_train.shape}")

# #     # 超参数 (针对归一化后的 [0,1] 数据)
# #     param_grid = {
# #         'learning_rate': [0.01, 0.05, 0.001, 0.005, 0.0001],
# #         'max_depth': [8, 10],
# #         'num_leaves': [63, 127],
# #         'min_data_in_leaf': [10, 20],
# #         'feature_fraction': [0.8, 0.9],
# #         'bagging_fraction': [0.8],
# #         'bagging_freq': [5],
# #         'lambda_l2': [0.1, 1.0],
# #         'num_iterations': [1000],
# #         'early_stopping_rounds': [50]
# #     }

# #     keys, values = zip(*param_grid.items())
# #     import random
# #     # 随机采样 5 组以节省时间
# #     param_combinations = [dict(zip(keys, v)) for v in random.sample(list(itertools.product(*values)), 5)]

# #     best_score = np.inf
# #     best_model = None
# #     best_params = None

# #     for trial_idx, params in enumerate(param_combinations):
# #         print(f"\nTrial {trial_idx+1}: {params}")
        
# #         train_data = lgb.Dataset(X_train, label=y_train)
# #         val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

# #         model = lgb.train(
# #             params={
# #                 'objective': 'regression_l1', 
# #                 'metric': 'mae',
# #                 'verbose': -1,
# #                 'seed': CONFIG['seed'],
# #                 **{k: v for k, v in params.items() if k not in ['num_iterations', 'early_stopping_rounds']}
# #             },
# #             train_set=train_data,
# #             num_boost_round=params['num_iterations'],
# #             valid_sets=[val_data],
# #             callbacks=[
# #                 lgb.early_stopping(stopping_rounds=params['early_stopping_rounds'], verbose=False),
# #                 lgb.log_evaluation(0)
# #             ]
# #         )

# #         val_metrics = evaluate_lgb_fixed(model, X_val, y_val)
# #         score = val_metrics['MAE']
# #         print(f"Val MAE: {score:.4f}, R2: {val_metrics['R2']:.4f}, PCC: {val_metrics['PCC']:.4f}")

# #         if score < best_score:
# #             best_score = score
# #             best_model = copy.deepcopy(model)
# #             best_params = params

# #     print("\n" + "="*70)
# #     print(f"🔍 FINAL TEST RESULTS (Fixed Range [{Y_LOW_FIXED}, {Y_HIGH_FIXED}])")
# #     print("="*70)

# #     for name, test_samples_norm in dataset_test_splits_norm.items():
# #         X_test, y_test_norm = samples_to_xy(test_samples_norm)
# #         metrics = evaluate_lgb_fixed(best_model, X_test, y_test_norm)
# #         print(f"\n{name.upper()} TEST → "
# #               f"R²: {metrics['R2']:.4f}, "
# #               f"MSE: {metrics['MSE']:.4f}, "
# #               f"RMSE: {metrics['RMSE']:.4f}, "
# #               f"MAE: {metrics['MAE']:.4f}, "
# #               f"PCC: {metrics['PCC']:.4f}")

# # if __name__ == "__main__":
# #     main()
# def main():
#     train_tsvs = {
#         "/tmp/AbAgCDR/data/final_dataset_train.tsv": 1.0,
#         "/tmp/AbAgCDR/data/pairs_seq_skempi_clean.tsv": 1.0,
#         "/tmp/AbAgCDR/data/pairs_seq_sabdab_clean.tsv": 1.0,
#         "/tmp/AbAgCDR/data/pairs_seq_abbind2_clean.tsv": 1.0,
#     }

#     CONFIG = {"test_size": 0.2, "val_size": 0.2, "seed": 42, "run_dir": "../runs"}
#     print("🚀 Starting LightGBM with Fixed Normalization [-15.5, -2.0]...")

#     all_train_samples = []
#     all_val_samples = []
#     all_test_samples = []
#     dataset_test_splits = {}

#     for tsv_path, weight in train_tsvs.items():
#         if not Path(tsv_path).exists():
#             print(f"Skipping missing file: {tsv_path}")
#             continue
#         print(f"Loading {tsv_path}...")
#         samples = load_and_encode_tsv(tsv_path)
#         if not samples:
#             continue
        
#         dgs = [s[3] for s in samples]
#         out_of_range = sum(1 for x in dgs if x < Y_LOW_FIXED or x > Y_HIGH_FIXED)
#         if out_of_range > 0:
#             print(f"  ⚠️ Warning: {out_of_range}/{len(samples)} samples are OUTSIDE [{Y_LOW_FIXED}, {Y_HIGH_FIXED}]. They will be clipped!")

#         train, val, test = split_samples(samples, CONFIG["test_size"], CONFIG["val_size"], CONFIG["seed"])
#         all_train_samples.extend(train)
#         all_val_samples.extend(val)
#         all_test_samples.extend(test)
#         dataset_name = Path(tsv_path).stem
#         dataset_test_splits[dataset_name] = test

#     if not all_train_samples:
#         print("❌ No data loaded!")
#         return

#     print(f"Total Train: {len(all_train_samples)}, Val: {len(all_val_samples)}, Test: {len(all_test_samples)}")

#     def normalize_samples(samples):
#         out = []
#         for a, b, ag, dg in samples:
#             dg_norm = normalize_label_fixed(dg)
#             out.append((a, b, ag, dg_norm))
#         return out

#     all_train_norm = normalize_samples(all_train_samples)
#     all_val_norm = normalize_samples(all_val_samples)
#     all_test_norm = normalize_samples(all_test_samples)
#     dataset_test_splits_norm = {k: normalize_samples(v) for k, v in dataset_test_splits.items()}

#     def samples_to_xy(samples):
#         X, y = [], []
#         for a, b, ag, dg_norm in samples:
#             feat = extract_features_v2(a, b, ag)
#             X.append(feat)
#             y.append(dg_norm)
#         return np.array(X), np.array(y)

#     print("Extracting features...")
#     X_train, y_train = samples_to_xy(all_train_norm)
#     X_val, y_val = samples_to_xy(all_val_norm)
#     print(f"Feature shape: {X_train.shape}")

#     param_grid = {
#         'learning_rate': [0.01, 0.05, 0.001, 0.005, 0.0001],
#         'max_depth': [8, 10],
#         'num_leaves': [63, 127],
#         'min_data_in_leaf': [10, 20],
#         'feature_fraction': [0.8, 0.9],
#         'bagging_fraction': [0.8],
#         'bagging_freq': [5],
#         'lambda_l2': [0.1, 1.0],
#         'num_iterations': [1000],
#         'early_stopping_rounds': [50]
#     }

#     keys, values = zip(*param_grid.items())
#     import random
#     param_combinations = [dict(zip(keys, v)) for v in random.sample(list(itertools.product(*values)), 5)]

#     best_score = np.inf
#     best_model = None
#     best_params = None

#     for trial_idx, params in enumerate(param_combinations):
#         print(f"\nTrial {trial_idx+1}: {params}")
        
#         train_data = lgb.Dataset(X_train, label=y_train)
#         val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

#         model = lgb.train(
#             params={
#                 'objective': 'regression_l1', 
#                 'metric': 'mae',
#                 'verbose': -1,
#                 'seed': CONFIG['seed'],
#                 **{k: v for k, v in params.items() if k not in ['num_iterations', 'early_stopping_rounds']}
#             },
#             train_set=train_data,
#             num_boost_round=params['num_iterations'],
#             valid_sets=[val_data],
#             callbacks=[
#                 lgb.early_stopping(stopping_rounds=params['early_stopping_rounds'], verbose=False),
#                 lgb.log_evaluation(0)
#             ]
#         )

#         val_metrics = evaluate_lgb_fixed(model, X_val, y_val)
#         score = val_metrics['MAE']
#         print(f"Val MAE: {score:.4f}, R2: {val_metrics['R2']:.4f}, PCC: {val_metrics['PCC']:.4f}")

#         if score < best_score:
#             best_score = score
#             best_model = copy.deepcopy(model)
#             best_params = params

#     print("\n" + "="*70)
#     print(f"🔍 FINAL TEST RESULTS (Fixed Range [{Y_LOW_FIXED}, {Y_HIGH_FIXED}])")
#     print("="*70)

#     for name, test_samples_norm in dataset_test_splits_norm.items():
#         X_test, y_test_norm = samples_to_xy(test_samples_norm)
#         metrics = evaluate_lgb_fixed(best_model, X_test, y_test_norm)
#         print(f"\n{name.upper()} TEST → "
#               f"R²: {metrics['R2']:.4f}, "
#               f"MSE: {metrics['MSE']:.4f}, "
#               f"RMSE: {metrics['RMSE']:.4f}, "
#               f"MAE: {metrics['MAE']:.4f}, "
#               f"PCC: {metrics['PCC']:.4f}")

#     # ===================== #
#     # ✅ 保存最佳模型
#     # ===================== #
#     model_save_dir = Path("/tmp/AbAgCDR/model")
#     model_save_dir.mkdir(parents=True, exist_ok=True)
    
#     # 保存为LightGBM原生格式（推荐，跨平台兼容性好）
#     model_path_lgb = model_save_dir / "lightgbm_best_model.txt"
#     best_model.save_model(str(model_path_lgb))
#     print(f"\n💾 Model saved (LightGBM format): {model_path_lgb}")
    
#     # 保存为joblib格式（方便Python直接加载）
#     model_path_joblib = model_save_dir / "lightgbm_best_model.pkl"
#     joblib.dump(best_model, str(model_path_joblib))
#     print(f"💾 Model saved (joblib format): {model_path_joblib}")
    
#     # 保存最佳参数
#     params_path = model_save_dir / "best_params.txt"
#     with open(params_path, 'w') as f:
#         f.write(f"Best validation MAE: {best_score:.6f}\n")
#         f.write(f"Best parameters:\n")
#         for k, v in best_params.items():
#             f.write(f"  {k}: {v}\n")
#     print(f"💾 Best params saved: {params_path}")
    
#     print("\n✅ Training completed and model saved successfully!")

# if __name__ == "__main__":
#     main()

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import joblib
import copy
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr, spearmanr
import lightgbm as lgb

# ===================== #
# 特征（不变）
# ===================== #
AA_LIST = list("ACDEFGHIKLMNPQRSTVWY")
AA_TO_IDX = {aa: i for i, aa in enumerate(AA_LIST)}

HYDROPATHY = {
    'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5,
    'Q': -3.5, 'E': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5,
    'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6,
    'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2
}

def seq_to_baa(seq):
    counts = np.zeros(20)
    valid_len = 0
    for aa in seq:
        if aa in AA_LIST:
            counts[AA_TO_IDX[aa]] += 1
            valid_len += 1
    return counts / valid_len if valid_len > 0 else counts

def seq_to_physchem_detailed(seq):
    hydro_vals = [HYDROPATHY.get(aa, 0.0) for aa in seq]
    charge_vals = [1.0 if aa in 'RK' else -1.0 if aa in 'DE' else 0.0 for aa in seq]
    return np.array([
        np.mean(hydro_vals), np.std(hydro_vals),
        np.mean(charge_vals), np.sum(charge_vals),
        np.max(hydro_vals), np.min(hydro_vals)
    ])

def extract_features_v2(a, b, ag):
    seqs = [a, b, ag]
    baa = np.concatenate([seq_to_baa(s) for s in seqs])
    phys = np.concatenate([seq_to_physchem_detailed(s) for s in seqs])
    glob = []
    for s in seqs:
        total_hydro = sum(HYDROPATHY.get(aa, 0.0) for aa in s)
        net_charge = sum(1 for aa in s if aa in 'RK') - sum(1 for aa in s if aa in 'DE')
        glob.extend([total_hydro, net_charge, len(s)])
    return np.concatenate([baa, phys, np.array(glob)])

# ===================== #
# 数据加载
# ===================== #
def load_data(path):
    df = pd.read_csv(path, sep='\t', on_bad_lines='skip')

    if 'antibody_seq_b' not in df.columns and 'heavy_seq' in df.columns:
        df['antibody_seq_b'] = df['heavy_seq']
    if 'antibody_seq_a' not in df.columns and 'light_seq' in df.columns:
        df['antibody_seq_a'] = df['light_seq']

    df['delta_g'] = pd.to_numeric(df['delta_g'], errors='coerce')
    df = df.dropna(subset=['delta_g'])

    samples = []
    for _, row in df.iterrows():
        a, b, ag, dg = str(row['antibody_seq_a']), str(row['antibody_seq_b']), str(row['antigen_seq']), float(row['delta_g'])
        if len(a) > 5 and len(ag) > 5:
            samples.append((a, b, ag, dg))
    return samples

# ===================== #
# 指标
# ===================== #
def metrics(y_true, y_pred):
    return {
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAE": mean_absolute_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
        "PCC": pearsonr(y_true, y_pred)[0],
        "Spearman": spearmanr(y_true, y_pred)[0]
    }

# ===================== #
# 主函数（LODO + Calibration）
# ===================== #
def main():
    datasets = {
        "FINAL": "/tmp/AbAgCDR/data/final_dataset_train.tsv",
        "SKEMPI": "/tmp/AbAgCDR/data/pairs_seq_skempi_clean.tsv",
        "SABDAB": "/tmp/AbAgCDR/data/pairs_seq_sabdab_clean.tsv",
        "ABBIND": "/tmp/AbAgCDR/data/pairs_seq_abbind2_clean.tsv",
    }

    all_data = {name: load_data(path) for name, path in datasets.items()}

    print("🚀 LODO Evaluation + Calibration")

    for test_name in datasets.keys():
        print("\n" + "="*60)
        print(f"🧪 TEST ON: {test_name}")
        print("="*60)

        # ===== 构建训练集 =====
        train_samples = []
        for name, data in all_data.items():
            if name != test_name:
                train_samples.extend(data)

        test_samples = all_data[test_name]

        train, val = train_test_split(train_samples, test_size=0.2, random_state=42)

        # ===== 标准化 =====
        y_train_raw = np.array([s[3] for s in train])
        mean, std = y_train_raw.mean(), y_train_raw.std() + 1e-8

        def norm(y): return (y - mean) / std
        def denorm(y): return y * std + mean

        def to_xy(samples):
            X, y = [], []
            for a, b, ag, dg in samples:
                X.append(extract_features_v2(a, b, ag))
                y.append(norm(dg))
            return np.array(X), np.array(y)

        X_train, y_train = to_xy(train)
        X_val, y_val = to_xy(val)
        X_test, y_test = to_xy(test_samples)

        model = lgb.train(
            {
                'objective': 'regression',
                'metric': 'rmse',
                'learning_rate': 0.01,
                'num_leaves': 63,
                'max_depth': 8,
                'seed': 42,
                'verbose': -1
            },
            lgb.Dataset(X_train, label=y_train),
            num_boost_round=1000,
            valid_sets=[lgb.Dataset(X_val, label=y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False)]
        )

        # ===================== #
        # ⭐ NEW: 校准（只用val）
        # ===================== #
        y_val_pred = denorm(model.predict(X_val))
        y_val_true = denorm(y_val)

        # 拟合线性关系
        a, b = np.polyfit(y_val_pred, y_val_true, 1)

        # ===================== #
        # 测试
        # ===================== #
        y_pred = denorm(model.predict(X_test))

        # ⭐ 应用校准
        y_pred_calibrated = a * y_pred + b

        y_true = denorm(y_test)

        m = metrics(y_true, y_pred_calibrated)

        print(f"{test_name} → "
              f"R2={m['R2']:.4f}, "
              f"RMSE={m['RMSE']:.4f}, "
              f"MAE={m['MAE']:.4f}, "
              f"PCC={m['PCC']:.4f}, "
              f"Spearman={m['Spearman']:.4f}")

    print("\n✅ Calibration applied (no leakage).")

if __name__ == "__main__":
    main()