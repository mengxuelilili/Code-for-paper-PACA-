# import os
# import warnings
# warnings.filterwarnings('ignore')

# import numpy as np
# import pandas as pd
# import lightgbm as lgb
# from pathlib import Path
# import matplotlib.pyplot as plt
# from scipy.stats import pearsonr
# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# # ===================== #
# # 1. 特征提取器 (必须与训练脚本完全一致)
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

# # ===================== #
# # 2. 归一化参数 (必须与训练脚本完全一致)
# # ===================== #
# Y_LOW_FIXED = -15.5
# Y_HIGH_FIXED = -2.0
# Y_RANGE = Y_HIGH_FIXED - Y_LOW_FIXED

# def denormalize_label_fixed(dg_norm):
#     return dg_norm * Y_RANGE + Y_LOW_FIXED

# # ===================== #
# # 3. 数据加载函数
# # ===================== #
# def load_tsv_for_prediction(tsv_path):
#     df = pd.read_csv(tsv_path, sep='\t', on_bad_lines='skip')
    
#     # 列名兼容处理
#     if 'antibody_seq_b' not in df.columns and 'heavy_seq' in df.columns:
#         df['antibody_seq_b'] = df['heavy_seq']
#     if 'antibody_seq_a' not in df.columns and 'light_seq' in df.columns:
#         df['antibody_seq_a'] = df['light_seq']
    
#     required_cols = ['antibody_seq_a', 'antibody_seq_b', 'antigen_seq']
#     missing_cols = [col for col in required_cols if col not in df.columns]
#     if missing_cols:
#         raise ValueError(f"Missing columns: {missing_cols} in {tsv_path}")
    
#     samples = []
#     valid_indices = []
    
#     for idx, row in df.iterrows():
#         a = str(row.get('antibody_seq_a', ''))
#         b = str(row.get('antibody_seq_b', ''))
#         ag = str(row.get('antigen_seq', ''))
        
#         if a and b and ag and len(a) > 5 and len(ag) > 5:
#             samples.append((a, b, ag))
#             valid_indices.append(idx)
    
#     print(f"✅ Loaded {len(samples)} valid samples from {tsv_path}")
#     return samples, valid_indices, df

# def samples_to_X(samples):
#     X = []
#     for a, b, ag in samples:
#         feat = extract_features_v2(a, b, ag)
#         X.append(feat)
#     return np.array(X)

# # ===================== #
# # 4. 评估函数
# # ===================== #
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
# # 5. 绘图函数
# # ===================== #
# def plot_regression(y_true, y_pred, title, save_path):
#     """绘制回归散点图"""
#     plt.figure(figsize=(8, 8))
    
#     # 散点图
#     plt.scatter(y_true, y_pred, alpha=0.6, edgecolors='w', linewidth=0.5, color='blue')
    
#     # 对角线（完美预测）
#     min_val = min(min(y_true), min(y_pred))
#     max_val = max(max(y_true), max(y_pred))
#     plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    
#     # 计算指标
#     metrics = compute_metrics(y_true, y_pred)
    
#     # 添加指标文本
#     textstr = f'RMSE: {metrics["RMSE"]:.4f}\nMAE: {metrics["MAE"]:.4f}\nR²: {metrics["R2"]:.4f}\nPCC: {metrics["PCC"]:.4f}'
#     plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes,
#              fontsize=12, verticalalignment='top',
#              bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
#     plt.xlabel('True ΔG', fontsize=12)
#     plt.ylabel('Predicted ΔG', fontsize=12)
#     plt.title(title, fontsize=14)
#     plt.legend()
#     plt.grid(True, alpha=0.3)
#     plt.tight_layout()
#     plt.savefig(save_path, dpi=300, bbox_inches='tight')
#     plt.close()
#     print(f"📊 Plot saved to: {save_path}")

# # ===================== #
# # 6. 主预测函数
# # ===================== #
# def main():
#     # --- 路径配置 --- 
#     model_path = "/tmp/AbAgCDR/model_improved/lightgbm_improved_model.txt"
#     tsv_path = "/tmp/AbAgCDR/data/final_dataset_train.tsv"
#     output_dir = "/tmp/AbAgCDR/resultsxin"
#     os.makedirs(output_dir, exist_ok=True)
#     output_csv = os.path.join(output_dir, "lightgbm_train_predictions.csv")
#     plot_path = os.path.join(output_dir, "lightgbm_train_regression_plot.png")
    
#     print("="*70)
#     print("🚀 LightGBM Prediction Script")
#     print("="*70)
#     print(f"📦 Model: {model_path}")
#     print(f"📂 Input: {tsv_path}")
#     print(f"📁 Output CSV: {output_csv}")
#     print(f"📊 Output Plot: {plot_path}")
#     print("="*70)
    
#     # --- 1. 检查文件 ---
#     if not os.path.exists(model_path):
#         raise FileNotFoundError(f"❌ Model not found: {model_path}")
#     if not os.path.exists(tsv_path):
#         raise FileNotFoundError(f"❌ Input file not found: {tsv_path}")
    
#     # --- 2. 加载模型 ---
#     print("\n📦 Loading model...")
#     model = lgb.Booster(model_file=model_path)
#     print(f"✅ Model loaded successfully")
#     print(f"   Number of trees: {model.num_trees()}")
    
#     # --- 3. 加载数据并提取特征 ---
#     print("\n📊 Loading data and extracting features...")
#     samples, valid_indices, original_df = load_tsv_for_prediction(tsv_path)
    
#     if not samples:
#         print("❌ No valid samples found!")
#         return
    
#     X = samples_to_X(samples)
#     print(f"   Feature matrix shape: {X.shape}")
    
#     # --- 4. 预测 ---
#     print("\n🔮 Making predictions...")
#     y_pred_norm = model.predict(X)
#     y_pred = denormalize_label_fixed(y_pred_norm)
    
#     print(f"   Predictions range: [{y_pred.min():.4f}, {y_pred.max():.4f}]")
#     print(f"   Number of predictions: {len(y_pred)}")
    
#     # --- 5. 检查是否有真实标签 ---
#     has_true_labels = 'delta_g' in original_df.columns
    
#     # --- 6. 保存预测结果（包含Index, true_ddg, pred_ddg）---
#     print("\n💾 Saving predictions...")
    
#     if has_true_labels:
#         # 提取对应索引的真实值
#         y_true = original_df.loc[valid_indices, 'delta_g'].values
        
#         # 创建包含三列的DataFrame
#         results_df = pd.DataFrame({
#             'Index': valid_indices,
#             'true_ddg': y_true,
#             'pred_ddg': y_pred
#         })
#         print(f"   Including true_ddg column from input file")
#     else:
#         # 如果没有真实标签，只保存Index和pred_ddg
#         results_df = pd.DataFrame({
#             'Index': valid_indices,
#             'pred_ddg': y_pred
#         })
#         print(f"   No 'delta_g' column found - saving only Index and pred_ddg")
    
#     # 保存为CSV
#     results_df.to_csv(output_csv, index=False)
#     print(f"✅ Predictions saved to: {output_csv}")
#     print(f"   Format: {', '.join(results_df.columns)}")
#     print(f"   Total rows: {len(results_df)}")
    
#     # --- 7. 如果有真实标签，绘制散点回归图并显示指标 ---
#     if has_true_labels:
#         print("\n📈 Generating regression plot...")
        
#         if len(y_true) == len(y_pred):
#             # 计算并显示指标
#             metrics = compute_metrics(y_true, y_pred)
#             print("\n" + "="*70)
#             print("📊 VALIDATION METRICS")
#             print("="*70)
#             print(f"Samples: {len(y_true)}")
#             print(f"RMSE: {metrics['RMSE']:.4f}")
#             print(f"MAE:  {metrics['MAE']:.4f}")
#             print(f"R²:   {metrics['R2']:.4f}")
#             print(f"PCC:  {metrics['PCC']:.4f}")
#             print("="*70)
            
#             # 绘制散点回归图
#             plot_regression(
#                 y_true, y_pred,
#                 title=f'LightGBM: True vs Predicted ΔG\n({os.path.basename(tsv_path)})',
#                 save_path=plot_path
#             )
#         else:
#             print(f"⚠️ Length mismatch: y_true={len(y_true)}, y_pred={len(y_pred)}")
#     else:
#         print("\n⚠️ No 'delta_g' column found - skipping regression plot and metrics")
    
#     print(f"\n✅ Prediction completed successfully!")
#     print(f"📁 Results saved to: {output_dir}")

# if __name__ == "__main__":
#     main()

import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import lightgbm as lgb
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ===================== #
# 1. 特征提取器 (必须与训练脚本完全一致 - V3版本)
# ===================== #
AA_LIST = list("ACDEFGHIKLMNPQRSTVWY")
AA_TO_IDX = {aa: i for i, aa in enumerate(AA_LIST)}

HYDROPATHY = {
    'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5,
    'Q': -3.5, 'E': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5,
    'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6,
    'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2
}

# 分子量
MOL_WEIGHT = {
    'A': 89.1, 'R': 174.2, 'N': 132.1, 'D': 133.1, 'C': 121.2,
    'Q': 146.2, 'E': 147.1, 'G': 75.1, 'H': 155.2, 'I': 131.2,
    'L': 131.2, 'K': 146.2, 'M': 149.2, 'F': 165.2, 'P': 115.1,
    'S': 105.1, 'T': 119.1, 'W': 204.2, 'Y': 181.2, 'V': 117.1
}

# 等电点
ISOELECTRIC = {
    'A': 6.0, 'R': 10.8, 'N': 5.4, 'D': 2.8, 'C': 5.0,
    'Q': 5.7, 'E': 3.2, 'G': 6.0, 'H': 7.6, 'I': 6.0,
    'L': 6.0, 'K': 9.7, 'M': 5.7, 'F': 5.5, 'P': 6.3,
    'S': 5.7, 'T': 5.6, 'W': 5.9, 'Y': 5.7, 'V': 6.0
}

def seq_to_baa(seq):
    """氨基酸组成频率"""
    counts = np.zeros(20)
    valid_len = 0
    for aa in seq:
        if aa in AA_LIST:
            counts[AA_TO_IDX[aa]] += 1
            valid_len += 1
    return counts / valid_len if valid_len > 0 else counts

def seq_to_physchem_detailed_v3(seq):
    """详细的理化性质（V3版本）"""
    if not seq:
        return np.zeros(10)
    
    hydro_vals = [HYDROPATHY.get(aa, 0.0) for aa in seq]
    charge_vals = [1.0 if aa in 'RK' else -1.0 if aa in 'DE' else 0.0 for aa in seq]
    mw_vals = [MOL_WEIGHT.get(aa, 0.0) for aa in seq]
    ie_vals = [ISOELECTRIC.get(aa, 0.0) for aa in seq]
    
    feats = [
        np.mean(hydro_vals), np.std(hydro_vals), np.max(hydro_vals), np.min(hydro_vals),
        np.mean(charge_vals), np.sum(charge_vals), np.std(charge_vals),
        np.mean(mw_vals), np.mean(ie_vals),
        len(seq)  # 序列长度本身也是特征
    ]
    return np.array(feats)

def extract_cross_features(light_seq, heavy_seq, antigen_seq):
    """提取抗体-抗原交互特征"""
    ab_seq = light_seq + heavy_seq
    cross_feats = []
    
    # 1. 电荷互补性
    ab_charge = sum(1 for aa in ab_seq if aa in 'RK') - sum(1 for aa in ab_seq if aa in 'DE')
    ag_charge = sum(1 for aa in antigen_seq if aa in 'RK') - sum(1 for aa in antigen_seq if aa in 'DE')
    cross_feats.append(ab_charge * ag_charge)  # 电荷乘积
    cross_feats.append(abs(ab_charge - ag_charge))  # 电荷差
    
    # 2. 疏水性互补
    ab_hydro = np.mean([HYDROPATHY.get(aa, 0.0) for aa in ab_seq])
    ag_hydro = np.mean([HYDROPATHY.get(aa, 0.0) for aa in antigen_seq])
    cross_feats.append(ab_hydro * ag_hydro)
    cross_feats.append(abs(ab_hydro - ag_hydro))
    
    # 3. 长度比例
    len_ratio = len(ab_seq) / len(antigen_seq) if len(antigen_seq) > 0 else 0
    cross_feats.append(len_ratio)
    
    # 4. 分子量比例
    ab_mw = np.mean([MOL_WEIGHT.get(aa, 0.0) for aa in ab_seq])
    ag_mw = np.mean([MOL_WEIGHT.get(aa, 0.0) for aa in antigen_seq])
    cross_feats.append(ab_mw / ag_mw if ag_mw > 0 else 0)
    
    # 5. 简单的序列相似性（氨基酸类型分布）
    ab_aa_counts = np.zeros(20)
    ag_aa_counts = np.zeros(20)
    for aa in ab_seq:
        if aa in AA_LIST:
            ab_aa_counts[AA_TO_IDX[aa]] += 1
    for aa in antigen_seq:
        if aa in AA_LIST:
            ag_aa_counts[AA_TO_IDX[aa]] += 1
    
    # 余弦相似度
    ab_norm = np.linalg.norm(ab_aa_counts)
    ag_norm = np.linalg.norm(ag_aa_counts)
    if ab_norm > 0 and ag_norm > 0:
        cos_sim = np.dot(ab_aa_counts, ag_aa_counts) / (ab_norm * ag_norm)
    else:
        cos_sim = 0
    cross_feats.append(cos_sim)
    
    return np.array(cross_feats)

def extract_features_v3(light_seq, heavy_seq, antigen_seq):
    """改进的特征提取（V3版本）- 与训练脚本完全一致"""
    seqs = [light_seq, heavy_seq, antigen_seq]
    
    # 1. 基础特征（氨基酸频率）- 60维
    baa = np.concatenate([seq_to_baa(s) for s in seqs])
    
    # 2. 理化性质特征（增强版）- 30维 (10*3)
    phys = np.concatenate([seq_to_physchem_detailed_v3(s) for s in seqs])
    
    # 3. 全局特征 - 18维 (6*3)
    glob_feats = []
    for s in seqs:
        total_hydro = sum(HYDROPATHY.get(aa, 0.0) for aa in s)
        net_charge = sum(1 for aa in s if aa in 'RK') - sum(1 for aa in s if aa in 'DE')
        length = len(s)
        aromatics = sum(1 for aa in s if aa in 'FWY')  # 芳香族氨基酸
        aliphatic = sum(1 for aa in s if aa in 'ILV')  # 脂肪族氨基酸
        small = sum(1 for aa in s if aa in 'GAS')      # 小氨基酸
        
        glob_feats.extend([total_hydro, net_charge, length, aromatics, aliphatic, small])
    glob = np.array(glob_feats)
    
    # 4. 交互特征 - 7维
    cross = extract_cross_features(light_seq, heavy_seq, antigen_seq)
    
    # 总维度: 60 + 30 + 18 + 7 = 115
    return np.concatenate([baa, phys, glob, cross])

# ===================== #
# 2. 归一化参数 (必须与训练脚本完全一致)
# ===================== #
Y_LOW_FIXED = -15.5
Y_HIGH_FIXED = -2.0
Y_RANGE = Y_HIGH_FIXED - Y_LOW_FIXED

def denormalize_label_fixed(dg_norm):
    return dg_norm * Y_RANGE + Y_LOW_FIXED

# ===================== #
# 3. 数据加载函数
# ===================== #
def load_tsv_for_prediction(tsv_path):
    df = pd.read_csv(tsv_path, sep='\t', on_bad_lines='skip')
    
    # 列名兼容处理
    if 'antibody_seq_b' not in df.columns and 'heavy_seq' in df.columns:
        df['antibody_seq_b'] = df['heavy_seq']
    if 'antibody_seq_a' not in df.columns and 'light_seq' in df.columns:
        df['antibody_seq_a'] = df['light_seq']
    
    required_cols = ['antibody_seq_a', 'antibody_seq_b', 'antigen_seq']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns: {missing_cols} in {tsv_path}")
    
    samples = []
    valid_indices = []
    
    for idx, row in df.iterrows():
        a = str(row.get('antibody_seq_a', ''))
        b = str(row.get('antibody_seq_b', ''))
        ag = str(row.get('antigen_seq', ''))
        
        if a and b and ag and len(a) > 5 and len(ag) > 5:
            samples.append((a, b, ag))
            valid_indices.append(idx)
    
    print(f"✅ Loaded {len(samples)} valid samples from {tsv_path}")
    return samples, valid_indices, df

def samples_to_X_v3(samples):
    """使用V3特征提取"""
    X = []
    for a, b, ag in samples:
        feat = extract_features_v3(a, b, ag)
        X.append(feat)
    return np.array(X)

# ===================== #
# 4. 评估函数
# ===================== #
def compute_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    try:
        pcc, _ = pearsonr(y_true, y_pred)
    except:
        pcc = 0.0
    return {'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'R2': r2, 'PCC': pcc}

# ===================== #
# 5. 绘图函数
# ===================== #
def plot_regression(y_true, y_pred, title, save_path):
    """绘制回归散点图"""
    plt.figure(figsize=(8, 8))
    
    # 散点图
    plt.scatter(y_true, y_pred, alpha=0.6, edgecolors='w', linewidth=0.5, color='blue')
    
    # 对角线（完美预测）
    min_val = min(min(y_true), min(y_pred))
    max_val = max(max(y_true), max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    
    # 计算指标
    metrics = compute_metrics(y_true, y_pred)
    
    # 添加指标文本
    textstr = f'RMSE: {metrics["RMSE"]:.4f}\nMAE: {metrics["MAE"]:.4f}\nR²: {metrics["R2"]:.4f}\nPCC: {metrics["PCC"]:.4f}'
    plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes,
             fontsize=12, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.xlabel('True ΔG', fontsize=12)
    plt.ylabel('Predicted ΔG', fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"📊 Plot saved to: {save_path}")

# ===================== #
# 6. 主预测函数
# ===================== #
def main():
    # --- 路径配置 --- 
    model_path = "/tmp/AbAgCDR/model_improved/lightgbm_improved_model.txt"
    tsv_path = "/tmp/AbAgCDR/data/pairs_seq_skempi_clean.tsv"
    output_dir = "/tmp/AbAgCDR/resultsxin"
    os.makedirs(output_dir, exist_ok=True)
    output_csv = os.path.join(output_dir, "lightgbm_skempi_predictions.csv")
    plot_path = os.path.join(output_dir, "lightgbm_skempi_regression_plot.png")
    
    print("="*70)
    print("🚀 LightGBM Prediction Script (V3 Features)")
    print("="*70)
    print(f"📦 Model: {model_path}")
    print(f"📂 Input: {tsv_path}")
    print(f"📁 Output CSV: {output_csv}")
    print(f"📊 Output Plot: {plot_path}")
    print("="*70)
    
    # --- 1. 检查文件 ---
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"❌ Model not found: {model_path}")
    if not os.path.exists(tsv_path):
        raise FileNotFoundError(f"❌ Input file not found: {tsv_path}")
    
    # --- 2. 加载模型 ---
    print("\n📦 Loading model...")
    model = lgb.Booster(model_file=model_path)
    print(f"✅ Model loaded successfully")
    print(f"   Number of trees: {model.num_trees()}")
    
    # --- 3. 加载数据并提取特征 (使用V3) ---
    print("\n📊 Loading data and extracting V3 features...")
    samples, valid_indices, original_df = load_tsv_for_prediction(tsv_path)
    
    if not samples:
        print("❌ No valid samples found!")
        return
    
    X = samples_to_X_v3(samples)
    print(f"   Feature matrix shape: {X.shape}")
    print(f"   Feature dimension: {X.shape[1]} (should be 115)")
    
    # 验证特征维度
    if X.shape[1] != 115:
        print(f"⚠️ Warning: Feature dimension mismatch! Expected 115, got {X.shape[1]}")
    
    # --- 4. 预测 ---
    print("\n🔮 Making predictions...")
    y_pred_norm = model.predict(X)
    y_pred = denormalize_label_fixed(y_pred_norm)
    
    print(f"   Predictions range: [{y_pred.min():.4f}, {y_pred.max():.4f}]")
    print(f"   Number of predictions: {len(y_pred)}")
    
    # --- 5. 检查是否有真实标签 ---
    has_true_labels = 'delta_g' in original_df.columns
    
    # --- 6. 保存预测结果（包含Index, true_ddg, pred_ddg）---
    print("\n💾 Saving predictions...")
    
    if has_true_labels:
        # 提取对应索引的真实值
        y_true = original_df.loc[valid_indices, 'delta_g'].values
        
        # 创建包含三列的DataFrame
        results_df = pd.DataFrame({
            'Index': valid_indices,
            'true_ddg': y_true,
            'pred_ddg': y_pred
        })
        print(f"   Including true_ddg column from input file")
    else:
        # 如果没有真实标签，只保存Index和pred_ddg
        results_df = pd.DataFrame({
            'Index': valid_indices,
            'pred_ddg': y_pred
        })
        print(f"   No 'delta_g' column found - saving only Index and pred_ddg")
    
    # 保存为CSV
    results_df.to_csv(output_csv, index=False)
    print(f"✅ Predictions saved to: {output_csv}")
    print(f"   Format: {', '.join(results_df.columns)}")
    print(f"   Total rows: {len(results_df)}")
    
    # --- 7. 如果有真实标签，绘制散点回归图并显示指标 ---
    if has_true_labels:
        print("\n📈 Generating regression plot...")
        
        if len(y_true) == len(y_pred):
            # 计算并显示指标
            metrics = compute_metrics(y_true, y_pred)
            print("\n" + "="*70)
            print("📊 VALIDATION METRICS")
            print("="*70)
            print(f"Samples: {len(y_true)}")
            print(f"RMSE: {metrics['RMSE']:.4f}")
            print(f"MAE:  {metrics['MAE']:.4f}")
            print(f"R²:   {metrics['R2']:.4f}")
            print(f"PCC:  {metrics['PCC']:.4f}")
            print("="*70)
            
            # 绘制散点回归图
            plot_regression(
                y_true, y_pred,
                title=f'LightGBM (V3): True vs Predicted ΔG\n({os.path.basename(tsv_path)})',
                save_path=plot_path
            )
        else:
            print(f"⚠️ Length mismatch: y_true={len(y_true)}, y_pred={len(y_pred)}")
    else:
        print("\n⚠️ No 'delta_g' column found - skipping regression plot and metrics")
    
    print(f"\n✅ Prediction completed successfully!")
    print(f"📁 Results saved to: {output_dir}")

if __name__ == "__main__":
    main()