# import pandas as pd
# import numpy as np
# from scipy import stats

# # 读取数据
# ours = pd.read_csv("/tmp/AbAgCDR/results/train_predictions.csv")
# baseline = pd.read_csv("/tmp/AbAgCDR/results/IMMSAB_paddle_predicted_affinity.tsv", sep='\t')

# # 自动检测列名
# def find_column(df, possible_names):
#     for name in possible_names:
#         if name in df.columns:
#             return name
#     return None

# # 真实值列名
# true_col_ours = find_column(ours, ['True_ddG', 'True_delta_g', 'true', 'y_true'])
# true_col_base = find_column(baseline, ['True_delta_g', 'True_ddG', 'true', 'y_true'])

# # 预测值列名
# pred_col_ours = find_column(ours, ['Predicted_ddG', 'Predicted_delta_g', 'pred', 'y_pred'])
# pred_col_base = find_column(baseline, ['Predicted_delta_g', 'Predicted_ddG', 'pred', 'y_pred'])

# print(f"Ours - 真实值列：{true_col_ours}, 预测值列：{pred_col_ours}")
# print(f"Baseline - 真实值列：{true_col_base}, 预测值列：{pred_col_base}")

# # 提取数据
# y_true = ours[true_col_ours].values
# y_pred_ours = ours[pred_col_ours].values
# y_pred_base = baseline[pred_col_base].values

# # 确保长度一致
# min_len = min(len(y_true), len(y_pred_ours), len(y_pred_base))
# y_true = y_true[:min_len]
# y_pred_ours = y_pred_ours[:min_len]
# y_pred_base = y_pred_base[:min_len]

# # 计算误差
# se_ours = (y_true - y_pred_ours) ** 2
# se_base = (y_true - y_pred_base) ** 2

# # Paired t-test
# t_stat, p_value = stats.ttest_rel(se_base, se_ours)

# print(f"\nt-statistic: {t_stat:.4f}")
# print(f"p-value: {p_value:.6f}")
# print(f"Mean SE (Ours): {np.mean(se_ours):.4f}")
# print(f"Mean SE (Baseline): {np.mean(se_base):.4f}")

# # -*- coding: utf-8 -*-
# """
# 配对 t 检验脚本 - 比较两个模型的预测性能
# 支持多种 CSV/TSV 格式
# """

# import pandas as pd
# import numpy as np
# from scipy import stats
# import os

# # ============================================================================
# # 配置
# # ============================================================================

# OURS_FILE = "/tmp/AbAgCDR/resultsxin/skempi_predictions.csv"
# BASELINE_FILE = "/tmp/AbAgCDR/resultsxin/lightgbm_skempi_predictions.csv"
# OUTPUT_FILE = "/tmp/AbAgCDR/resultsxin/pairedskempi_ttest_results.txt"

# # ============================================================================
# # 列名检测
# # ============================================================================

# def find_column(df, possible_names):
#     """自动检测列名"""
#     cols_lower = {c.lower(): c for c in df.columns}
#     for name in possible_names:
#         if name.lower() in cols_lower:
#             return cols_lower[name.lower()]
#     return None

# def load_predictions(file_path):
#     """加载预测文件，自动识别格式"""
#     print(f"📥 加载：{file_path}")
    
#     # 自动识别分隔符
#     if file_path.endswith('.tsv'):
#         df = pd.read_csv(file_path, sep='\t')
#     elif file_path.endswith('.csv'):
#         df = pd.read_csv(file_path, sep=',')
#     else:
#         # 尝试自动检测
#         with open(file_path, 'r') as f:
#             first_line = f.readline()
#             sep = '\t' if '\t' in first_line else ','
#         df = pd.read_csv(file_path, sep=sep)
    
#     print(f"   列名：{list(df.columns)}")
#     print(f"   行数：{len(df)}")
    
#     return df

# def extract_data(df, file_name):
#     """从 DataFrame 中提取真实值和预测值"""
    
#     # 真实值列名候选
#     true_candidates = [
#         'delta_g', 'ddg', 'true', 'y_true', 'true_ddg', 'true_delta_g',
#         'Delta_G', 'DDG', 'True', 'Y_True', 'label', 'target'
#     ]
    
#     # 预测值列名候选
#     pred_candidates = [
#         'predicted_delta_g', 'pred', 'y_pred', 'predicted_ddg', 'pred_ddg',
#         'pred_delta_g', 'Predicted_delta_g', 'Predicted_ddG', 'prediction',
#         'output', 'score', 'affinity'
#     ]
    
#     # 检测列名
#     true_col = find_column(df, true_candidates)
#     pred_col = find_column(df, pred_candidates)
    
#     print(f"\n{file_name}:")
#     print(f"   真实值列：{true_col}")
#     print(f"   预测值列：{pred_col}")
    
#     if true_col is None or pred_col is None:
#         print(f"   ⚠️  警告：未找到匹配的列名，请手动指定")
#         print(f"   可用列：{list(df.columns)}")
#         return None, None, None, None
    
#     # 提取数据
#     y_true = df[true_col].values.astype(float)
#     y_pred = df[pred_col].values.astype(float)
    
#     # 检查是否有 Index 列
#     index_col = find_column(df, ['index', 'idx', 'id', 'no', 'num'])
#     if index_col:
#         indices = df[index_col].values
#     else:
#         indices = np.arange(len(df))
    
#     return y_true, y_pred, indices, {
#         'true_col': true_col,
#         'pred_col': pred_col,
#         'index_col': index_col
#     }

# # ============================================================================
# # 主函数
# # ============================================================================

# def main():
#     print("="*70)
#     print("🔬 配对 t 检验 - 模型性能比较")
#     print("="*70)
    
#     # 1. 加载数据
#     print("\n" + "="*70)
#     print("📊 加载数据")
#     print("="*70)
    
#     df_ours = load_predictions(OURS_FILE)
#     df_baseline = load_predictions(BASELINE_FILE)
    
#     # 2. 提取数据
#     print("\n" + "="*70)
#     print("📋 提取数据")
#     print("="*70)
    
#     y_true_ours, y_pred_ours, idx_ours, info_ours = extract_data(df_ours, "Ours")
#     y_true_base, y_pred_base, idx_base, info_base = extract_data(df_baseline, "Baseline")
    
#     if y_true_ours is None or y_pred_ours is None:
#         print("\n❌ 无法从 Ours 文件提取数据，退出")
#         return
    
#     if y_true_base is None or y_pred_base is None:
#         print("\n❌ 无法从 Baseline 文件提取数据，退出")
#         return
    
#     # 3. 对齐数据
#     print("\n" + "="*70)
#     print("🔗 对齐数据")
#     print("="*70)
    
#     # 方案 1：如果都有 Index 列，按 Index 对齐
#     if info_ours['index_col'] and info_base['index_col']:
#         print("   按 Index 列对齐...")
#         df_ours_temp = pd.DataFrame({
#             'index': idx_ours,
#             'y_true': y_true_ours,
#             'y_pred': y_pred_ours
#         })
#         df_base_temp = pd.DataFrame({
#             'index': idx_base,
#             'y_pred': y_pred_base
#         })
        
#         merged = pd.merge(df_ours_temp, df_base_temp, on='index', how='inner')
#         y_true = merged['y_true'].values
#         y_pred_ours = merged['y_pred_x'].values
#         y_pred_base = merged['y_pred_y'].values
        
#         print(f"   对齐后样本数：{len(y_true)}")
    
#     # 方案 2：按真实值对齐（假设真实值相同）
#     elif len(y_true_ours) == len(y_true_base):
#         print("   按位置对齐（长度相同）...")
#         # 检查真实值是否一致
#         if np.allclose(y_true_ours, y_true_base, rtol=1e-5):
#             print("   ✅ 真实值一致")
#             y_true = y_true_ours
#         else:
#             print("   ⚠️  真实值不完全一致，使用 Ours 的真实值")
#             y_true = y_true_ours
#         y_pred_ours = y_pred_ours
#         y_pred_base = y_pred_base
        
#     # 方案 3：截断到最小长度
#     else:
#         print("   ⚠️  长度不同，截断到最小长度...")
#         min_len = min(len(y_true_ours), len(y_true_base), 
#                      len(y_pred_ours), len(y_pred_base))
#         y_true = y_true_ours[:min_len]
#         y_pred_ours = y_pred_ours[:min_len]
#         y_pred_base = y_pred_base[:min_len]
#         print(f"   对齐后样本数：{min_len}")
    
#     print(f"\n   最终样本数：{len(y_true)}")
    
#     # 4. 计算误差
#     print("\n" + "="*70)
#     print("📐 计算误差")
#     print("="*70)
    
#     # 平方误差 (SE)
#     se_ours = (y_true - y_pred_ours) ** 2
#     se_base = (y_true - y_pred_base) ** 2
    
#     # 绝对误差 (AE)
#     ae_ours = np.abs(y_true - y_pred_ours)
#     ae_base = np.abs(y_true - y_pred_base)
    
#     print(f"\n   Ours:")
#     print(f"   - Mean SE:  {np.mean(se_ours):.6f}")
#     print(f"   - Mean AE:  {np.mean(ae_ours):.6f}")
#     print(f"   - Std SE:   {np.std(se_ours):.6f}")
    
#     print(f"\n   Baseline:")
#     print(f"   - Mean SE:  {np.mean(se_base):.6f}")
#     print(f"   - Mean AE:  {np.mean(ae_base):.6f}")
#     print(f"   - Std SE:   {np.std(se_base):.6f}")
    
#     # 5. 配对 t 检验
#     print("\n" + "="*70)
#     print("🧪 配对 t 检验")
#     print("="*70)
    
#     # 基于平方误差 (SE)
#     t_stat_se, p_value_se = stats.ttest_rel(se_base, se_ours)
    
#     # 基于绝对误差 (AE)
#     t_stat_ae, p_value_ae = stats.ttest_rel(ae_base, ae_ours)
    
#     print(f"\n   基于平方误差 (SE):")
#     print(f"   - t-statistic: {t_stat_se:.6f}")
#     print(f"   - p-value:     {p_value_se:.10f}")
    
#     print(f"\n   基于绝对误差 (AE):")
#     print(f"   - t-statistic: {t_stat_ae:.6f}")
#     print(f"   - p-value:     {p_value_ae:.10f}")
    
#     # 6. 效应量 (Cohen's d)
#     print("\n" + "="*70)
#     print("📊 效应量 (Cohen's d)")
#     print("="*70)
    
#     mean_diff_se = np.mean(se_base) - np.mean(se_ours)
#     std_diff_se = np.std(se_base - se_ours)
#     cohens_d_se = mean_diff_se / std_diff_se if std_diff_se > 0 else 0
    
#     mean_diff_ae = np.mean(ae_base) - np.mean(ae_ours)
#     std_diff_ae = np.std(ae_base - ae_ours)
#     cohens_d_ae = mean_diff_ae / std_diff_ae if std_diff_ae > 0 else 0
    
#     print(f"\n   基于 SE: Cohen's d = {cohens_d_se:.4f}")
#     print(f"   基于 AE: Cohen's d = {cohens_d_ae:.4f}")
    
#     # 效应量解释
#     def interpret_cohens_d(d):
#         d = abs(d)
#         if d < 0.2:
#             return "可忽略"
#         elif d < 0.5:
#             return "小"
#         elif d < 0.8:
#             return "中"
#         else:
#             return "大"
    
#     print(f"\n   效应量解释：{interpret_cohens_d(cohens_d_se)}")
    
#     # 7. 其他指标比较
#     print("\n" + "="*70)
#     print("📈 其他指标比较")
#     print("="*70)
    
#     from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
#     from scipy.stats import pearsonr
    
#     # Ours
#     rmse_ours = np.sqrt(mean_squared_error(y_true, y_pred_ours))
#     mae_ours = mean_absolute_error(y_true, y_pred_ours)
#     r2_ours = r2_score(y_true, y_pred_ours)
#     pcc_ours, _ = pearsonr(y_true, y_pred_ours)
    
#     # Baseline
#     rmse_base = np.sqrt(mean_squared_error(y_true, y_pred_base))
#     mae_base = mean_absolute_error(y_true, y_pred_base)
#     r2_base = r2_score(y_true, y_pred_base)
#     pcc_base, _ = pearsonr(y_true, y_pred_base)
    
#     print(f"\n   {'指标':<15} {'Ours':<15} {'Baseline':<15} {'提升':<15}")
#     print(f"   {'-'*60}")
#     print(f"   {'RMSE':<15} {rmse_ours:<15.6f} {rmse_base:<15.6f} {((rmse_base-rmse_ours)/rmse_base*100):>6.2f}%")
#     print(f"   {'MAE':<15} {mae_ours:<15.6f} {mae_base:<15.6f} {((mae_base-mae_ours)/mae_base*100):>6.2f}%")
#     print(f"   {'R²':<15} {r2_ours:<15.6f} {r2_base:<15.6f} {((r2_ours-r2_base)/abs(r2_base)*100):>6.2f}%")
#     print(f"   {'PCC':<15} {pcc_ours:<15.6f} {pcc_base:<15.6f} {((pcc_ours-pcc_base)/abs(pcc_base)*100):>6.2f}%")
    
#     # 8. 显著性判断
#     print("\n" + "="*70)
#     print("✅ 结论")
#     print("="*70)
    
#     alpha = 0.05
#     if p_value_se < alpha:
#         if np.mean(se_ours) < np.mean(se_base):
#             print(f"\n   🎉 Ours 显著优于 Baseline (p < {alpha})")
#         else:
#             print(f"\n   ⚠️  Baseline 显著优于 Ours (p < {alpha})")
#     else:
#         print(f"\n   😐 无显著差异 (p = {p_value_se:.6f} > {alpha})")
    
#     print(f"\n   置信水平：{(1-alpha)*100:.0f}%")
#     print(f"   p-value: {p_value_se:.10f}")
    
#     # 9. 保存结果
#     print("\n" + "="*70)
#     print("💾 保存结果")
#     print("="*70)
    
#     os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    
#     with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
#         f.write("="*70 + "\n")
#         f.write("配对 t 检验结果 - 模型性能比较\n")
#         f.write("="*70 + "\n\n")
        
#         f.write("数据文件:\n")
#         f.write(f"  Ours:     {OURS_FILE}\n")
#         f.write(f"  Baseline: {BASELINE_FILE}\n")
#         f.write(f"  样本数：  {len(y_true)}\n\n")
        
#         f.write("-"*70 + "\n")
#         f.write("误差统计\n")
#         f.write("-"*70 + "\n")
#         f.write(f"  Ours - Mean SE: {np.mean(se_ours):.6f}\n")
#         f.write(f"  Ours - Mean AE: {np.mean(ae_ours):.6f}\n")
#         f.write(f"  Baseline - Mean SE: {np.mean(se_base):.6f}\n")
#         f.write(f"  Baseline - Mean AE: {np.mean(ae_base):.6f}\n\n")
        
#         f.write("-"*70 + "\n")
#         f.write("配对 t 检验\n")
#         f.write("-"*70 + "\n")
#         f.write(f"  基于 SE:\n")
#         f.write(f"    t-statistic: {t_stat_se:.6f}\n")
#         f.write(f"    p-value:     {p_value_se:.10f}\n")
#         f.write(f"    Cohen's d:   {cohens_d_se:.4f} ({interpret_cohens_d(cohens_d_se)})\n\n")
#         f.write(f"  基于 AE:\n")
#         f.write(f"    t-statistic: {t_stat_ae:.6f}\n")
#         f.write(f"    p-value:     {p_value_ae:.10f}\n")
#         f.write(f"    Cohen's d:   {cohens_d_ae:.4f} ({interpret_cohens_d(cohens_d_ae)})\n\n")
        
#         f.write("-"*70 + "\n")
#         f.write("指标比较\n")
#         f.write("-"*70 + "\n")
#         f.write(f"  {'指标':<10} {'Ours':<15} {'Baseline':<15} {'提升':<10}\n")
#         f.write(f"  {'RMSE':<10} {rmse_ours:<15.6f} {rmse_base:<15.6f} {((rmse_base-rmse_ours)/rmse_base*100):>6.2f}%\n")
#         f.write(f"  {'MAE':<10} {mae_ours:<15.6f} {mae_base:<15.6f} {((mae_base-mae_ours)/mae_base*100):>6.2f}%\n")
#         f.write(f"  {'R²':<10} {r2_ours:<15.6f} {r2_base:<15.6f} {((r2_ours-r2_base)/abs(r2_base)*100):>6.2f}%\n")
#         f.write(f"  {'PCC':<10} {pcc_ours:<15.6f} {pcc_base:<15.6f} {((pcc_ours-pcc_base)/abs(pcc_base)*100):>6.2f}%\n\n")
        
#         f.write("-"*70 + "\n")
#         f.write("结论\n")
#         f.write("-"*70 + "\n")
#         if p_value_se < alpha:
#             if np.mean(se_ours) < np.mean(se_base):
#                 f.write(f"  🎉 Ours 显著优于 Baseline (p < {alpha})\n")
#             else:
#                 f.write(f"  ⚠️  Baseline 显著优于 Ours (p < {alpha})\n")
#         else:
#             f.write(f"  😐 无显著差异 (p = {p_value_se:.6f} > {alpha})\n")
    
#     print(f"✅ 结果已保存：{OUTPUT_FILE}")
    
#     print("\n" + "="*70)
#     print("✅ 分析完成！")
#     print("="*70)
    
#     # 返回结果字典
#     return {
#         't_stat_se': t_stat_se,
#         'p_value_se': p_value_se,
#         't_stat_ae': t_stat_ae,
#         'p_value_ae': p_value_ae,
#         'cohens_d_se': cohens_d_se,
#         'cohens_d_ae': cohens_d_ae,
#         'rmse_ours': rmse_ours,
#         'rmse_base': rmse_base,
#         'mae_ours': mae_ours,
#         'mae_base': mae_base,
#         'r2_ours': r2_ours,
#         'r2_base': r2_base,
#         'pcc_ours': pcc_ours,
#         'pcc_base': pcc_base,
#         'n_samples': len(y_true)
#     }


# if __name__ == "__main__":
#     main()

# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
# from scipy import stats
# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# import os

# # ===================== #
# # 1. 残差分析函数
# # ===================== #
# def plot_residual_analysis(y_true, y_pred_ours, y_pred_base, model_names, save_path):
#     """绘制残差分析图"""
    
#     fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
#     # 计算残差
#     resid_ours = y_pred_ours - y_true
#     resid_base = y_pred_base - y_true
    
#     # 计算基本统计量
#     print(f"\n📊 {model_names[0]} 残差统计:")
#     print(f"   Mean: {np.mean(resid_ours):.4f}, Std: {np.std(resid_ours):.4f}")
#     print(f"   Min: {np.min(resid_ours):.4f}, Max: {np.max(resid_ours):.4f}")
    
#     print(f"\n📊 {model_names[1]} 残差统计:")
#     print(f"   Mean: {np.mean(resid_base):.4f}, Std: {np.std(resid_base):.4f}")
#     print(f"   Min: {np.min(resid_base):.4f}, Max: {np.max(resid_base):.4f}")
    
#     # 1. 残差散点图（最重要的图）
#     ax = axes[0, 0]
#     ax.scatter(y_true, resid_ours, alpha=0.6, label=model_names[0], s=30, c='blue', edgecolors='w', linewidth=0.5)
#     ax.scatter(y_true, resid_base, alpha=0.6, label=model_names[1], s=30, c='red', edgecolors='w', linewidth=0.5)
#     ax.axhline(y=0, color='black', linestyle='--', linewidth=2, label='Zero Error')
#     ax.set_xlabel('True ΔG', fontsize=12)
#     ax.set_ylabel('Residuals (Pred - True)', fontsize=12)
#     ax.legend(fontsize=10)
#     ax.set_title('Residual Plot', fontsize=14, fontweight='bold')
#     ax.grid(True, alpha=0.3)
    
#     # 2. 残差分布直方图
#     ax = axes[0, 1]
#     ax.hist(resid_ours, bins=30, alpha=0.5, label=model_names[0], density=True, color='blue', edgecolor='black')
#     ax.hist(resid_base, bins=30, alpha=0.5, label=model_names[1], density=True, color='red', edgecolor='black')
#     ax.axvline(x=0, color='black', linestyle='--', linewidth=2)
#     ax.axvline(x=np.mean(resid_ours), color='blue', linestyle='-', linewidth=1, alpha=0.5)
#     ax.axvline(x=np.mean(resid_base), color='red', linestyle='-', linewidth=1, alpha=0.5)
#     ax.set_xlabel('Residuals', fontsize=12)
#     ax.set_ylabel('Density', fontsize=12)
#     ax.legend(fontsize=10)
#     ax.set_title('Residual Distribution', fontsize=14, fontweight='bold')
#     ax.grid(True, alpha=0.3)
    
#     # 3. Q-Q图（检查正态性）- 两个模型并排显示
#     ax = axes[0, 2]
    
#     # 计算分位数
#     osm_ours, osr_ours = stats.probplot(resid_ours, dist="norm", fit=False)
#     osm_base, osr_base = stats.probplot(resid_base, dist="norm", fit=False)
    
#     ax.scatter(osm_ours, osr_ours, alpha=0.6, label=model_names[0], s=30, c='blue', edgecolors='w')
#     ax.scatter(osm_base, osr_base, alpha=0.6, label=model_names[1], s=30, c='red', edgecolors='w')
    
#     # 添加对角线
#     min_val = min(osm_ours.min(), osm_base.min())
#     max_val = max(osm_ours.max(), osm_base.max())
#     ax.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2, label='Normal Line')
    
#     ax.set_xlabel('Theoretical Quantiles', fontsize=12)
#     ax.set_ylabel('Sample Quantiles', fontsize=12)
#     ax.legend(fontsize=10)
#     ax.set_title('Q-Q Plot', fontsize=14, fontweight='bold')
#     ax.grid(True, alpha=0.3)
    
#     # 4. 残差 vs 预测值
#     ax = axes[1, 0]
#     ax.scatter(y_pred_ours, resid_ours, alpha=0.6, label=model_names[0], s=30, c='blue', edgecolors='w')
#     ax.scatter(y_pred_base, resid_base, alpha=0.6, label=model_names[1], s=30, c='red', edgecolors='w')
#     ax.axhline(y=0, color='black', linestyle='--', linewidth=2)
#     ax.set_xlabel('Predicted ΔG', fontsize=12)
#     ax.set_ylabel('Residuals', fontsize=12)
#     ax.legend(fontsize=10)
#     ax.set_title('Residuals vs Predicted', fontsize=14, fontweight='bold')
#     ax.grid(True, alpha=0.3)
    
#     # 5. 绝对残差 vs 真实值
#     ax = axes[1, 1]
#     ax.scatter(y_true, np.abs(resid_ours), alpha=0.6, label=model_names[0], s=30, c='blue', edgecolors='w')
#     ax.scatter(y_true, np.abs(resid_base), alpha=0.6, label=model_names[1], s=30, c='red', edgecolors='w')
#     ax.set_xlabel('True ΔG', fontsize=12)
#     ax.set_ylabel('|Residuals|', fontsize=12)
#     ax.legend(fontsize=10)
#     ax.set_title('Absolute Residuals vs True Values', fontsize=14, fontweight='bold')
#     ax.grid(True, alpha=0.3)
    
#     # 6. 累计误差分布
#     ax = axes[1, 2]
#     sorted_resid_ours = np.sort(np.abs(resid_ours))
#     sorted_resid_base = np.sort(np.abs(resid_base))
#     y_vals = np.arange(1, len(sorted_resid_ours)+1) / len(sorted_resid_ours)
    
#     ax.plot(sorted_resid_ours, y_vals, label=model_names[0], linewidth=2, color='blue')
#     ax.plot(sorted_resid_base, y_vals, label=model_names[1], linewidth=2, color='red')
    
#     # 标记中位数
#     median_ours = np.median(np.abs(resid_ours))
#     median_base = np.median(np.abs(resid_base))
#     ax.axvline(x=median_ours, color='blue', linestyle='--', alpha=0.5, label=f'{model_names[0]} Median: {median_ours:.3f}')
#     ax.axvline(x=median_base, color='red', linestyle='--', alpha=0.5, label=f'{model_names[1]} Median: {median_base:.3f}')
    
#     ax.set_xlabel('Absolute Residual', fontsize=12)
#     ax.set_ylabel('Cumulative Probability', fontsize=12)
#     ax.legend(fontsize=8)
#     ax.set_title('Cumulative Error Distribution', fontsize=14, fontweight='bold')
#     ax.grid(True, alpha=0.3)
    
#     plt.suptitle('Residual Analysis: Model Comparison', fontsize=16, fontweight='bold', y=1.02)
#     plt.tight_layout()
#     plt.savefig(save_path, dpi=300, bbox_inches='tight')
#     plt.show()
#     plt.close()
    
#     print(f"\n✅ Residual analysis plot saved to: {save_path}")

# # ===================== #
# # 2. 数据加载函数
# # ===================== #
# def load_and_merge_predictions(ours_file, baseline_file):
#     """加载并合并两个模型的预测结果"""
    
#     print("="*70)
#     print("📂 加载预测结果")
#     print("="*70)
    
#     # 加载数据
#     df_ours = pd.read_csv(ours_file)
#     df_base = pd.read_csv(baseline_file)
    
#     print(f"\n📊 Ours 文件: {ours_file}")
#     print(f"   列名: {list(df_ours.columns)}")
#     print(f"   行数: {len(df_ours)}")
    
#     print(f"\n📊 Baseline 文件: {baseline_file}")
#     print(f"   列名: {list(df_base.columns)}")
#     print(f"   行数: {len(df_base)}")
    
#     # 检查是否有Index列
#     if 'Index' in df_ours.columns and 'Index' in df_base.columns:
#         print("\n🔗 按 Index 列合并...")
#         merged = pd.merge(df_ours, df_base, on='Index', suffixes=('_ours', '_base'))
#     else:
#         print("\n⚠️ 没有找到 Index 列，假设行顺序对应...")
#         # 假设行顺序一致
#         df_base_renamed = df_base.copy()
#         if 'pred_ddg' in df_base.columns:
#             df_base_renamed = df_base_renamed.rename(columns={'pred_ddg': 'pred_ddg_base'})
#         if 'true_ddg' in df_base.columns:
#             df_base_renamed = df_base_renamed.rename(columns={'true_ddg': 'true_ddg_base'})
        
#         merged = pd.concat([df_ours, df_base_renamed], axis=1)
    
#     print(f"\n✅ 合并后数据行数: {len(merged)}")
    
#     # 确定列名
#     true_col = None
#     pred_ours_col = None
#     pred_base_col = None
    
#     for col in merged.columns:
#         if 'true' in col.lower() or 'ddg' in col.lower() and 'pred' not in col.lower():
#             if '_ours' in col or 'ours' not in col.lower():
#                 true_col = col
#         if 'pred' in col.lower() and 'ours' in col.lower():
#             pred_ours_col = col
#         if 'pred' in col.lower() and 'base' in col.lower():
#             pred_base_col = col
    
#     # 如果没有找到带后缀的，尝试通用列名
#     if true_col is None:
#         if 'true_ddg' in merged.columns:
#             true_col = 'true_ddg'
    
#     if pred_ours_col is None:
#         if 'pred_ddg_ours' in merged.columns:
#             pred_ours_col = 'pred_ddg_ours'
#         elif 'pred_ddg' in merged.columns and 'pred_ddg_base' not in merged.columns:
#             pred_ours_col = 'pred_ddg'
    
#     if pred_base_col is None:
#         if 'pred_ddg_base' in merged.columns:
#             pred_base_col = 'pred_ddg_base'
#         elif 'pred_ddg' in merged.columns and pred_ours_col != 'pred_ddg':
#             pred_base_col = 'pred_ddg'
    
#     print(f"\n📌 识别的列名:")
#     print(f"   真实值列: {true_col}")
#     print(f"   Ours预测列: {pred_ours_col}")
#     print(f"   Baseline预测列: {pred_base_col}")
    
#     if true_col is None or pred_ours_col is None or pred_base_col is None:
#         print("\n❌ 错误: 无法识别必要的列名!")
#         print(f"   可用列: {list(merged.columns)}")
#         return None, None, None, None
    
#     # 提取数据并去除NaN
#     valid_mask = merged[true_col].notna() & merged[pred_ours_col].notna() & merged[pred_base_col].notna()
#     merged_clean = merged[valid_mask]
    
#     y_true = merged_clean[true_col].values.astype(float)
#     y_pred_ours = merged_clean[pred_ours_col].values.astype(float)
#     y_pred_base = merged_clean[pred_base_col].values.astype(float)
    
#     print(f"\n✅ 有效样本数: {len(y_true)}")
    
#     return y_true, y_pred_ours, y_pred_base, merged_clean

# # ===================== #
# # 3. 主函数
# # ===================== #
# def main():
#     # 配置路径
#     ours_file = "/tmp/AbAgCDR/resultsxin/skempi_predictions.csv"
#     baseline_file = "/tmp/AbAgCDR/resultsxin/lightgbm_skempi_predictions.csv"
#     output_dir = "/tmp/AbAgCDR/resultsxin"
#     output_plot = os.path.join(output_dir, "residual_analysis_complete.png")
    
#     # 创建输出目录
#     os.makedirs(output_dir, exist_ok=True)
    
#     # 检查文件是否存在
#     if not os.path.exists(ours_file):
#         print(f"❌ 文件不存在: {ours_file}")
#         return
#     if not os.path.exists(baseline_file):
#         print(f"❌ 文件不存在: {baseline_file}")
#         return
    
#     # 加载数据
#     y_true, y_pred_ours, y_pred_base, merged_df = load_and_merge_predictions(ours_file, baseline_file)
    
#     if y_true is None:
#         return
    
#     # 计算基本指标
#     print("\n" + "="*70)
#     print("📈 基本指标比较")
#     print("="*70)
    
#     from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
#     from scipy.stats import pearsonr
    
#     rmse_ours = np.sqrt(mean_squared_error(y_true, y_pred_ours))
#     rmse_base = np.sqrt(mean_squared_error(y_true, y_pred_base))
#     mae_ours = mean_absolute_error(y_true, y_pred_ours)
#     mae_base = mean_absolute_error(y_true, y_pred_base)
#     r2_ours = r2_score(y_true, y_pred_ours)
#     r2_base = r2_score(y_true, y_pred_base)
#     pcc_ours, _ = pearsonr(y_true, y_pred_ours)
#     pcc_base, _ = pearsonr(y_true, y_pred_base)
    
#     print(f"\n{'Metric':<15} {'Ours':<15} {'Baseline':<15} {'Difference':<15}")
#     print(f"{'-'*60}")
#     print(f"{'RMSE':<15} {rmse_ours:<15.4f} {rmse_base:<15.4f} {((rmse_base-rmse_ours)/rmse_base*100):>+6.2f}%")
#     print(f"{'MAE':<15} {mae_ours:<15.4f} {mae_base:<15.4f} {((mae_base-mae_ours)/mae_base*100):>+6.2f}%")
#     print(f"{'R²':<15} {r2_ours:<15.4f} {r2_base:<15.4f} {((r2_ours-r2_base)/abs(r2_base)*100):>+6.2f}%")
#     print(f"{'PCC':<15} {pcc_ours:<15.4f} {pcc_base:<15.4f} {((pcc_ours-pcc_base)/abs(pcc_base)*100):>+6.2f}%")
    
#     # 绘制残差分析图
#     print("\n" + "="*70)
#     print("🎨 绘制残差分析图...")
#     print("="*70)
    
#     model_names = ['PACA', 'LightGBM']
    
#     plot_residual_analysis(
#         y_true, y_pred_ours, y_pred_base,
#         model_names,
#         output_plot
#     )
    
#     print(f"\n✅ 分析完成！")
#     print(f"📊 残差分析图: {output_plot}")
    
#     # 保存合并后的数据（可选）
#     output_csv = os.path.join(output_dir, "merged_predictions_for_analysis.csv")
#     merged_df.to_csv(output_csv, index=False)
#     print(f"📁 合并数据保存至: {output_csv}")

# if __name__ == "__main__":
#     main()

# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
# from scipy import stats
# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# import os

# # 设置中文字体支持（可选）
# plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica', 'sans-serif']
# plt.rcParams['axes.unicode_minus'] = False

# # ===================== #
# # 1. 残差分析函数（改进版）
# # ===================== #
# def plot_residual_analysis_improved(y_true, y_pred_ours, y_pred_base, model_names, save_path):
#     """绘制改进的残差分析图（更清晰的布局）"""
    
#     # 创建图形，调整布局
#     fig = plt.figure(figsize=(18, 12))
    
#     # 使用GridSpec自定义布局
#     gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.25)
#     axes = gs.subplots()
    
#     # 计算残差
#     resid_ours = y_pred_ours - y_true
#     resid_base = y_pred_base - y_true
    
#     # 计算统计量
#     metrics = {}
#     for name, resid in zip(model_names, [resid_ours, resid_base]):
#         metrics[name] = {
#             'mean': np.mean(resid),
#             'std': np.std(resid),
#             'min': np.min(resid),
#             'max': np.max(resid),
#             'median': np.median(resid)
#         }
    
#     # 打印统计信息
#     print("\n" + "="*70)
#     print("📊 残差统计")
#     print("="*70)
#     for name in model_names:
#         print(f"\n{name}:")
#         print(f"  Mean: {metrics[name]['mean']:.4f}")
#         print(f"  Std:  {metrics[name]['std']:.4f}")
#         print(f"  Median: {metrics[name]['median']:.4f}")
#         print(f"  Range: [{metrics[name]['min']:.4f}, {metrics[name]['max']:.4f}]")
    
#     # 1. 残差散点图
#     ax = axes[0, 0]
#     ax.scatter(y_true, resid_ours, alpha=0.6, label=model_names[0], 
#                s=30, c='#1f77b4', edgecolors='white', linewidth=0.5)
#     ax.scatter(y_true, resid_base, alpha=0.6, label=model_names[1], 
#                s=30, c='#d62728', edgecolors='white', linewidth=0.5)
#     ax.axhline(y=0, color='black', linestyle='--', linewidth=2, alpha=0.7)
    
#     # 添加均值线
#     ax.axhline(y=metrics[model_names[0]]['mean'], color='#1f77b4', 
#                linestyle=':', linewidth=1.5, alpha=0.5)
#     ax.axhline(y=metrics[model_names[1]]['mean'], color='#d62728', 
#                linestyle=':', linewidth=1.5, alpha=0.5)
    
#     ax.set_xlabel('True ΔG', fontsize=12)
#     ax.set_ylabel('Residuals (Pred - True)', fontsize=12)
#     ax.legend(loc='best', fontsize=10)
#     ax.set_title('(a) Residual Plot', fontsize=14, fontweight='bold', loc='left')
#     ax.grid(True, alpha=0.3, linestyle='--')
    
#     # 添加统计信息到图
#     textstr = f'{model_names[0]} mean: {metrics[model_names[0]]["mean"]:.3f}\n{model_names[1]} mean: {metrics[model_names[1]]["mean"]:.3f}'
#     ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=9,
#             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
#     # 2. 残差分布直方图
#     ax = axes[0, 1]
    
#     # 计算合适的bins范围
#     all_resid = np.concatenate([resid_ours, resid_base])
#     bins = np.linspace(all_resid.min(), all_resid.max(), 40)
    
#     ax.hist(resid_ours, bins=bins, alpha=0.6, label=model_names[0], 
#             density=True, color='#1f77b4', edgecolor='black', linewidth=0.5)
#     ax.hist(resid_base, bins=bins, alpha=0.6, label=model_names[1], 
#             density=True, color='#d62728', edgecolor='black', linewidth=0.5)
    
#     ax.axvline(x=0, color='black', linestyle='--', linewidth=2)
#     ax.axvline(x=metrics[model_names[0]]['mean'], color='#1f77b4', 
#                linestyle=':', linewidth=2, label=f'{model_names[0]} mean')
#     ax.axvline(x=metrics[model_names[1]]['mean'], color='#d62728', 
#                linestyle=':', linewidth=2, label=f'{model_names[1]} mean')
    
#     ax.set_xlabel('Residuals', fontsize=12)
#     ax.set_ylabel('Density', fontsize=12)
#     ax.legend(loc='best', fontsize=9)
#     ax.set_title('(b) Residual Distribution', fontsize=14, fontweight='bold', loc='left')
#     ax.grid(True, alpha=0.3, linestyle='--')
    
#     # 3. Q-Q图
#     ax = axes[0, 2]
    
#     # 计算分位数
#     (osm_ours, osr_ours), (slope_ours, intercept_ours, r_ours) = stats.probplot(resid_ours, dist="norm")
#     (osm_base, osr_base), (slope_base, intercept_base, r_base) = stats.probplot(resid_base, dist="norm")
    
#     ax.scatter(osm_ours, osr_ours, alpha=0.6, label=model_names[0], 
#                s=30, c='#1f77b4', edgecolors='white', linewidth=0.5)
#     ax.scatter(osm_base, osr_base, alpha=0.6, label=model_names[1], 
#                s=30, c='#d62728', edgecolors='white', linewidth=0.5)
    
#     # 添加拟合线
#     x_fit = np.array([osm_ours.min(), osm_ours.max()])
#     ax.plot(x_fit, intercept_ours + slope_ours * x_fit, 'b--', alpha=0.5, linewidth=1.5)
#     x_fit = np.array([osm_base.min(), osm_base.max()])
#     ax.plot(x_fit, intercept_base + slope_base * x_fit, 'r--', alpha=0.5, linewidth=1.5)
    
#     ax.set_xlabel('Theoretical Quantiles', fontsize=12)
#     ax.set_ylabel('Sample Quantiles', fontsize=12)
#     ax.legend(loc='best', fontsize=9)
#     ax.set_title('(c) Q-Q Plot', fontsize=14, fontweight='bold', loc='left')
#     ax.grid(True, alpha=0.3, linestyle='--')
    
#     # 添加R²值
#     textstr = f'{model_names[0]} R²: {r_ours**2:.3f}\n{model_names[1]} R²: {r_base**2:.3f}'
#     ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=9,
#             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
#     # 4. 残差 vs 预测值
#     ax = axes[1, 0]
#     ax.scatter(y_pred_ours, resid_ours, alpha=0.6, label=model_names[0], 
#                s=30, c='#1f77b4', edgecolors='white', linewidth=0.5)
#     ax.scatter(y_pred_base, resid_base, alpha=0.6, label=model_names[1], 
#                s=30, c='#d62728', edgecolors='white', linewidth=0.5)
#     ax.axhline(y=0, color='black', linestyle='--', linewidth=2)
#     ax.set_xlabel('Predicted ΔG', fontsize=12)
#     ax.set_ylabel('Residuals', fontsize=12)
#     ax.legend(loc='best', fontsize=9)
#     ax.set_title('(d) Residuals vs Predicted', fontsize=14, fontweight='bold', loc='left')
#     ax.grid(True, alpha=0.3, linestyle='--')
    
#     # 5. 绝对残差 vs 真实值
#     ax = axes[1, 1]
#     abs_resid_ours = np.abs(resid_ours)
#     abs_resid_base = np.abs(resid_base)
    
#     ax.scatter(y_true, abs_resid_ours, alpha=0.6, label=model_names[0], 
#                s=30, c='#1f77b4', edgecolors='white', linewidth=0.5)
#     ax.scatter(y_true, abs_resid_base, alpha=0.6, label=model_names[1], 
#                s=30, c='#d62728', edgecolors='white', linewidth=0.5)
    
#     # 添加趋势线
#     z_ours = np.polyfit(y_true, abs_resid_ours, 1)
#     p_ours = np.poly1d(z_ours)
#     z_base = np.polyfit(y_true, abs_resid_base, 1)
#     p_base = np.poly1d(z_base)
    
#     x_trend = np.linspace(y_true.min(), y_true.max(), 50)
#     ax.plot(x_trend, p_ours(x_trend), 'b--', alpha=0.5, linewidth=2, label=f'{model_names[0]} trend')
#     ax.plot(x_trend, p_base(x_trend), 'r--', alpha=0.5, linewidth=2, label=f'{model_names[1]} trend')
    
#     ax.set_xlabel('True ΔG', fontsize=12)
#     ax.set_ylabel('|Residuals|', fontsize=12)
#     ax.legend(loc='best', fontsize=8)
#     ax.set_title('(e) Absolute Residuals vs True Values', fontsize=14, fontweight='bold', loc='left')
#     ax.grid(True, alpha=0.3, linestyle='--')
    
#     # 6. 累计误差分布
#     ax = axes[1, 2]
    
#     sorted_ours = np.sort(abs_resid_ours)
#     sorted_base = np.sort(abs_resid_base)
#     y_vals = np.arange(1, len(sorted_ours)+1) / len(sorted_ours)
    
#     ax.plot(sorted_ours, y_vals, label=model_names[0], linewidth=2.5, color='#1f77b4')
#     ax.plot(sorted_base, y_vals, label=model_names[1], linewidth=2.5, color='#d62728')
    
#     # 标记关键分位数
#     for q in [0.25, 0.5, 0.75]:
#         q_ours = np.percentile(abs_resid_ours, q*100)
#         q_base = np.percentile(abs_resid_base, q*100)
#         ax.axhline(y=q, color='gray', linestyle=':', alpha=0.3)
#         ax.plot(q_ours, q, 'bo', markersize=6)
#         ax.plot(q_base, q, 'ro', markersize=6)
    
#     ax.set_xlabel('Absolute Residual', fontsize=12)
#     ax.set_ylabel('Cumulative Probability', fontsize=12)
#     ax.legend(loc='lower right', fontsize=9)
#     ax.set_title('(f) Cumulative Error Distribution', fontsize=14, fontweight='bold', loc='left')
#     ax.grid(True, alpha=0.3, linestyle='--')
    
#     # 添加整体标题
#     plt.suptitle('Residual Analysis: PACA vs LightGBM on SKEMPI Dataset', 
#                  fontsize=16, fontweight='bold', y=0.98)
    
#     plt.tight_layout()
#     plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
#     plt.show()
#     plt.close()
    
#     print(f"\n✅ 改进的残差分析图已保存至: {save_path}")

# # ===================== #
# # 2. 数据加载函数
# # ===================== #
# def load_and_merge_predictions(ours_file, baseline_file):
#     """加载并合并两个模型的预测结果"""
    
#     print("="*70)
#     print("📂 加载预测结果")
#     print("="*70)
    
#     # 加载数据
#     df_ours = pd.read_csv(ours_file)
#     df_base = pd.read_csv(baseline_file)
    
#     print(f"\n📊 Ours 文件: {ours_file}")
#     print(f"   列名: {list(df_ours.columns)}")
#     print(f"   行数: {len(df_ours)}")
    
#     print(f"\n📊 Baseline 文件: {baseline_file}")
#     print(f"   列名: {list(df_base.columns)}")
#     print(f"   行数: {len(df_base)}")
    
#     # 按Index合并
#     if 'Index' in df_ours.columns and 'Index' in df_base.columns:
#         print("\n🔗 按 Index 列合并...")
#         merged = pd.merge(df_ours, df_base, on='Index', suffixes=('_ours', '_base'))
#     else:
#         print("\n⚠️ 没有找到 Index 列，假设行顺序对应...")
#         merged = pd.concat([df_ours, df_base], axis=1)
#         # 重命名列以避免重复
#         if 'pred_ddg' in merged.columns:
#             if merged['pred_ddg'].iloc[0] == df_ours['pred_ddg'].iloc[0]:
#                 merged = merged.rename(columns={'pred_ddg': 'pred_ddg_ours'})
    
#     print(f"\n✅ 合并后数据行数: {len(merged)}")
    
#     # 识别列名
#     true_col = None
#     pred_ours_col = None
#     pred_base_col = None
    
#     # 查找真实值列
#     for col in merged.columns:
#         if 'true' in col.lower() or 'ddg' in col.lower() and 'pred' not in col.lower():
#             true_col = col
#             break
    
#     # 查找预测列
#     for col in merged.columns:
#         if 'pred' in col.lower() and 'ours' in col.lower():
#             pred_ours_col = col
#         elif 'pred' in col.lower() and 'base' in col.lower():
#             pred_base_col = col
    
#     # 如果没找到带后缀的，尝试通用列名
#     if pred_ours_col is None and 'pred_ddg' in merged.columns:
#         # 假设第一个pred_ddg是ours
#         pred_cols = [col for col in merged.columns if 'pred' in col.lower()]
#         if len(pred_cols) >= 2:
#             pred_ours_col = pred_cols[0]
#             pred_base_col = pred_cols[1]
#         elif len(pred_cols) == 1:
#             pred_ours_col = pred_cols[0]
#             # 尝试从baseline文件找
#             if 'pred_ddg' in df_base.columns:
#                 pred_base_col = 'pred_ddg'
    
#     print(f"\n📌 识别的列名:")
#     print(f"   真实值列: {true_col}")
#     print(f"   Ours预测列: {pred_ours_col}")
#     print(f"   Baseline预测列: {pred_base_col}")
    
#     if true_col is None or pred_ours_col is None or pred_base_col is None:
#         print("\n❌ 错误: 无法识别必要的列名!")
#         print(f"   可用列: {list(merged.columns)}")
#         return None, None, None, None
    
#     # 提取数据
#     y_true = merged[true_col].values.astype(float)
#     y_pred_ours = merged[pred_ours_col].values.astype(float)
#     y_pred_base = merged[pred_base_col].values.astype(float)
    
#     # 去除NaN
#     valid_mask = ~(np.isnan(y_true) | np.isnan(y_pred_ours) | np.isnan(y_pred_base))
#     y_true = y_true[valid_mask]
#     y_pred_ours = y_pred_ours[valid_mask]
#     y_pred_base = y_pred_base[valid_mask]
#     merged_clean = merged[valid_mask]
    
#     print(f"\n✅ 有效样本数: {len(y_true)}")
    
#     return y_true, y_pred_ours, y_pred_base, merged_clean

# # ===================== #
# # 3. 主函数
# # ===================== #
# def main():
#     # 配置路径
#     ours_file = "/tmp/AbAgCDR/resultsxin/skempi_predictions.csv"
#     baseline_file = "/tmp/AbAgCDR/resultsxin/lightgbm_skempi_predictions.csv"
#     output_dir = "/tmp/AbAgCDR/resultsxin"
#     output_plot = os.path.join(output_dir, "residual_analysis_improved.png")
    
#     # 创建输出目录
#     os.makedirs(output_dir, exist_ok=True)
    
#     # 检查文件
#     if not os.path.exists(ours_file):
#         print(f"❌ 文件不存在: {ours_file}")
#         return
#     if not os.path.exists(baseline_file):
#         print(f"❌ 文件不存在: {baseline_file}")
#         return
    
#     # 加载数据
#     y_true, y_pred_ours, y_pred_base, merged_df = load_and_merge_predictions(ours_file, baseline_file)
    
#     if y_true is None:
#         return
    
#     # 计算基本指标
#     print("\n" + "="*70)
#     print("📈 基本指标比较")
#     print("="*70)
    
#     rmse_ours = np.sqrt(mean_squared_error(y_true, y_pred_ours))
#     rmse_base = np.sqrt(mean_squared_error(y_true, y_pred_base))
#     mae_ours = mean_absolute_error(y_true, y_pred_ours)
#     mae_base = mean_absolute_error(y_true, y_pred_base)
#     r2_ours = r2_score(y_true, y_pred_ours)
#     r2_base = r2_score(y_true, y_pred_base)
#     pcc_ours, _ = stats.pearsonr(y_true, y_pred_ours)
#     pcc_base, _ = stats.pearsonr(y_true, y_pred_base)
    
#     print(f"\n{'Metric':<15} {'PACA':<15} {'LightGBM':<15} {'Improvement':<15}")
#     print(f"{'-'*60}")
#     print(f"{'RMSE':<15} {rmse_ours:<15.4f} {rmse_base:<15.4f} {((rmse_base-rmse_ours)/rmse_base*100):>+6.2f}%")
#     print(f"{'MAE':<15} {mae_ours:<15.4f} {mae_base:<15.4f} {((mae_base-mae_ours)/mae_base*100):>+6.2f}%")
#     print(f"{'R²':<15} {r2_ours:<15.4f} {r2_base:<15.4f} {((r2_ours-r2_base)/abs(r2_base)*100):>+6.2f}%")
#     print(f"{'PCC':<15} {pcc_ours:<15.4f} {pcc_base:<15.4f} {((pcc_ours-pcc_base)/abs(pcc_base)*100):>+6.2f}%")
    
#     # 绘制改进的残差分析图
#     print("\n" + "="*70)
#     print("🎨 绘制改进的残差分析图...")
#     print("="*70)
    
#     plot_residual_analysis_improved(
#         y_true, y_pred_ours, y_pred_base,
#         ['PACA', 'LightGBM'],
#         output_plot
#     )
    
#     print(f"\n✅ 分析完成！")
#     print(f"📊 改进的残差分析图: {output_plot}")
    
#     # 保存合并数据
#     output_csv = os.path.join(output_dir, "merged_data_for_analysis.csv")
#     merged_df.to_csv(output_csv, index=False)
#     print(f"📁 合并数据保存至: {output_csv}")

# if __name__ == "__main__":
#     main()

# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
# from scipy import stats
# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# import os

# def plot_residual_analysis(y_true, y_pred_ours, y_pred_base, model_names, save_path):
#     """绘制残差分析图"""
    
#     fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
#     # 计算残差
#     resid_ours = y_pred_ours - y_true
#     resid_base = y_pred_base - y_true
    
#     # 打印统计信息（用于验证）
#     print("\n" + "="*70)
#     print("📊 残差统计（验证）")
#     print("="*70)
#     print(f"\n{model_names[0]} 前10个残差:")
#     for i in range(min(10, len(resid_ours))):
#         print(f"  {i}: true={y_true[i]:.2f}, pred={y_pred_ours[i]:.2f}, resid={resid_ours[i]:.2f}")
    
#     # 1. 残差散点图
#     ax = axes[0, 0]
#     ax.scatter(y_true, resid_ours, alpha=0.6, label=model_names[0], s=30, c='#1f77b4')
#     ax.scatter(y_true, resid_base, alpha=0.6, label=model_names[1], s=30, c='#d62728')
#     ax.axhline(y=0, color='black', linestyle='--', linewidth=2)
#     ax.set_xlabel('True ΔG')
#     ax.set_ylabel('Residuals (Pred - True)')
#     ax.legend()
#     ax.set_title('(a) Residual Plot')
#     ax.grid(True, alpha=0.3)
    
#     # 2. 残差分布直方图
#     ax = axes[0, 1]
#     ax.hist(resid_ours, bins=30, alpha=0.6, label=model_names[0], density=True, color='#1f77b4')
#     ax.hist(resid_base, bins=30, alpha=0.6, label=model_names[1], density=True, color='#d62728')
#     ax.axvline(x=0, color='black', linestyle='--')
#     ax.set_xlabel('Residuals')
#     ax.set_ylabel('Density')
#     ax.legend()
#     ax.set_title('(b) Residual Distribution')
#     ax.grid(True, alpha=0.3)
    
#     # 3. Q-Q图
#     ax = axes[0, 2]
#     stats.probplot(resid_ours, dist="norm", plot=ax)
#     ax.set_title(f'(c) Q-Q Plot ({model_names[0]})')
#     ax.grid(True, alpha=0.3)
    
#     # 4. 残差 vs 预测值
#     ax = axes[1, 0]
#     ax.scatter(y_pred_ours, resid_ours, alpha=0.6, label=model_names[0], s=30, c='#1f77b4')
#     ax.scatter(y_pred_base, resid_base, alpha=0.6, label=model_names[1], s=30, c='#d62728')
#     ax.axhline(y=0, color='black', linestyle='--')
#     ax.set_xlabel('Predicted ΔG')
#     ax.set_ylabel('Residuals')
#     ax.legend()
#     ax.set_title('(d) Residuals vs Predicted')
#     ax.grid(True, alpha=0.3)
    
#     # 5. 绝对残差 vs 真实值
#     ax = axes[1, 1]
#     ax.scatter(y_true, np.abs(resid_ours), alpha=0.6, label=model_names[0], s=30, c='#1f77b4')
#     ax.scatter(y_true, np.abs(resid_base), alpha=0.6, label=model_names[1], s=30, c='#d62728')
#     ax.set_xlabel('True ΔG')
#     ax.set_ylabel('|Residuals|')
#     ax.legend()
#     ax.set_title('(e) Absolute Residuals vs True Values')
#     ax.grid(True, alpha=0.3)
    
#     # 6. 累计误差分布
#     ax = axes[1, 2]
#     sorted_ours = np.sort(np.abs(resid_ours))
#     sorted_base = np.sort(np.abs(resid_base))
#     y_vals = np.arange(1, len(sorted_ours)+1) / len(sorted_ours)
    
#     ax.plot(sorted_ours, y_vals, label=model_names[0], linewidth=2, color='#1f77b4')
#     ax.plot(sorted_base, y_vals, label=model_names[1], linewidth=2, color='#d62728')
#     ax.set_xlabel('Absolute Residual')
#     ax.set_ylabel('Cumulative Probability')
#     ax.legend()
#     ax.set_title('(f) Cumulative Error Distribution')
#     ax.grid(True, alpha=0.3)
    
#     plt.suptitle(f'Residual Analysis: {model_names[0]} vs {model_names[1]}', fontsize=16)
#     plt.tight_layout()
#     plt.savefig(save_path, dpi=300, bbox_inches='tight')
#     plt.show()
#     plt.close()
    
#     print(f"\n✅ 残差分析图已保存至: {save_path}")

# def main():
#     # 文件路径
#     ours_file = "/tmp/AbAgCDR/resultsxin/skempi_predictions.csv"
#     base_file = "/tmp/AbAgCDR/resultsxin/PWAARPEskempi_predictions.csv"
#     output_dir = "/tmp/AbAgCDR/resultsxin"
#     output_plot = os.path.join(output_dir, "residual_skempi_analysis_final.png")
    
#     print("="*70)
#     print("🚀 残差分析 - PACA vs PACARPE")
#     print("="*70)
    
#     # 直接加载数据，不做复杂合并
#     df_ours = pd.read_csv(ours_file)
#     df_base = pd.read_csv(base_file)
    
#     # 最简单的合并方式
#     y_true = df_ours['true_ddg'].values
#     y_pred_ours = df_ours['pred_ddg'].values
#     y_pred_base = df_base['pred_ddg'].values
    
#     print(f"\n✅ 数据加载成功:")
#     print(f"   样本数: {len(y_true)}")
#     print(f"   PACA预测范围: [{y_pred_ours.min():.2f}, {y_pred_ours.max():.2f}]")
#     print(f"   PACARPE预测范围: [{y_pred_base.min():.2f}, {y_pred_base.max():.2f}]")
#     print(f"   真实值范围: [{y_true.min():.2f}, {y_true.max():.2f}]")
    
#     # 验证数据（打印前5行）
#     print("\n📊 前5行数据验证:")
#     print("Index | true_ddg | PACA_pred | PACARPE_pred")
#     print("-" * 45)
#     for i in range(5):
#         print(f"{i:5d} | {y_true[i]:8.2f} | {y_pred_ours[i]:9.2f} | {y_pred_base[i]:12.2f}")
    
#     # 检查PACA是否有残差为0的情况
#     diff = np.abs(y_true - y_pred_ours)
#     zero_diff = np.sum(diff < 1e-10)
#     if zero_diff > 0:
#         print(f"\n⚠️ 警告: 发现 {zero_diff} 个样本PACA预测值完全等于真实值!")
    
#     # 运行残差分析
#     plot_residual_analysis(
#         y_true, y_pred_ours, y_pred_base,
#         ['PACA', 'PACARPE'],
#         output_plot
#     )

# if __name__ == "__main__":
#     main()

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import os

def plot_residual_analysis(y_true, y_pred_ours, y_pred_base, model_names, save_path):
    """绘制精简版残差分析图（论文友好版）"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 计算残差
    resid_ours = y_pred_ours - y_true
    resid_base = y_pred_base - y_true
    
    print("\n" + "="*70)
    print("📊 残差统计（验证）")
    print("="*70)

    print(f"{model_names[0]} Mean Residual: {np.mean(resid_ours):.4f}")
    print(f"{model_names[1]} Mean Residual: {np.mean(resid_base):.4f}")

    # =========================
    # (a) 残差分布
    # =========================
    ax = axes[0, 0]
    ax.hist(resid_ours, bins=30, alpha=0.6, label=model_names[0], density=True)
    ax.hist(resid_base, bins=30, alpha=0.6, label=model_names[1], density=True)
    ax.axvline(x=0, linestyle='--')
    ax.set_xlabel('Residuals')
    ax.set_ylabel('Density')
    ax.legend()
    ax.set_title('(a) Residual Distribution')
    ax.grid(True, alpha=0.3)


    # =========================
    # (b) Q-Q Plot (Comparative) - 修改版
    # =========================
    ax = axes[0, 1]
    
    # 1. 绘制 PACA (蓝色)
    # probplot 返回 (osm, osr), slope, intercept, r
    # osm: 理论分位数, osr: 实际排序后的数据
    osm_ours, osr_ours = stats.probplot(resid_ours, dist="norm", fit=False)
    ax.scatter(osm_ours, osr_ours, color='blue', alpha=0.6, s=30, label=model_names[0])
    
    # 2. 绘制 PACARPE (橙色)
    osm_base, osr_base = stats.probplot(resid_base, dist="norm", fit=False)
    ax.scatter(osm_base, osr_base, color='orange', alpha=0.6, s=30, label=model_names[1])
    
    # 3. 绘制参考红线 (y=x)
    # 获取坐标轴范围以绘制对角线
    min_val = min(osm_ours.min(), osm_base.min())
    max_val = max(osm_ours.max(), osm_base.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Normal Reference')
    
    ax.set_xlabel('Theoretical Quantiles')
    ax.set_ylabel('Ordered Residual Values') # 修改 Y 轴标签更准确
    ax.set_title('(b) Q-Q Plot Comparison') # 修改标题
    ax.legend()
    ax.grid(True, alpha=0.3)

    # =========================
    # (c) Residuals vs Predicted
    # =========================
    ax = axes[1, 0]
    ax.scatter(y_pred_ours, resid_ours, alpha=0.6, label=model_names[0], s=25)
    ax.scatter(y_pred_base, resid_base, alpha=0.6, label=model_names[1], s=25)
    ax.axhline(y=0, linestyle='--')
    ax.set_xlabel('Predicted ΔG')
    ax.set_ylabel('Residuals')
    ax.legend()
    ax.set_title('(c) Residuals vs Predicted')
    ax.grid(True, alpha=0.3)

    # =========================
    # (d) Absolute Error vs True
    # =========================
    ax = axes[1, 1]
    ax.scatter(y_true, np.abs(resid_ours), alpha=0.6, label=model_names[0], s=25)
    ax.scatter(y_true, np.abs(resid_base), alpha=0.6, label=model_names[1], s=25)
    ax.set_xlabel('True ΔG')
    ax.set_ylabel('|Residuals|')
    ax.legend()
    ax.set_title('(d) Absolute Error vs True Values')
    ax.grid(True, alpha=0.3)

    plt.suptitle(f'Paddle2021 Residual Analysis: {model_names[0]} vs {model_names[1]}', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

    print(f"\n✅ 图已保存: {save_path}")


def main():
    ours_file = "/tmp/AbAgCDR/resultsxin/train_predictions.csv"
    base_file = "/tmp/AbAgCDR/resultsxin/PWAARPEtrain_predictions.csv"
    output_plot = "/tmp/AbAgCDR/resultsxin/train_residual_clean.png"

    print("="*70)
    print("🚀 Residual Analysis（论文精简版）")
    print("="*70)

    df_ours = pd.read_csv(ours_file)
    df_base = pd.read_csv(base_file)

    y_true = df_ours['true_ddg'].values
    y_pred_ours = df_ours['pred_ddg'].values
    y_pred_base = df_base['pred_ddg'].values

    print(f"\n样本数: {len(y_true)}")

    plot_residual_analysis(
        y_true,
        y_pred_ours,
        y_pred_base,
        ['PACA', 'PACARPE'],
        output_plot
    )


if __name__ == "__main__":
    main()

# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
# from scipy import stats
# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# import os

# plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica', 'sans-serif']
# plt.rcParams['axes.unicode_minus'] = False

# def plot_residual_analysis(y_true, y_pred_ours, y_pred_base, model_names, save_path):
#     """绘制残差分析图"""
    
#     fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
#     # 计算残差
#     resid_ours = y_pred_ours - y_true
#     resid_base = y_pred_base - y_true
    
#     # 计算统计量
#     print("\n" + "="*70)
#     print("📊 残差统计")
#     print("="*70)
#     print(f"\n{model_names[0]}:")
#     print(f"  Mean: {np.mean(resid_ours):.4f}")
#     print(f"  Std:  {np.std(resid_ours):.4f}")
#     print(f"  Min:  {np.min(resid_ours):.4f}")
#     print(f"  Max:  {np.max(resid_ours):.4f}")
    
#     print(f"\n{model_names[1]}:")
#     print(f"  Mean: {np.mean(resid_base):.4f}")
#     print(f"  Std:  {np.std(resid_base):.4f}")
#     print(f"  Min:  {np.min(resid_base):.4f}")
#     print(f"  Max:  {np.max(resid_base):.4f}")
    
#     # 1. 残差散点图
#     ax = axes[0, 0]
#     ax.scatter(y_true, resid_ours, alpha=0.6, label=model_names[0], s=30, c='#1f77b4')
#     ax.scatter(y_true, resid_base, alpha=0.6, label=model_names[1], s=30, c='#d62728')
#     ax.axhline(y=0, color='black', linestyle='--', linewidth=2)
#     ax.set_xlabel('True ΔG')
#     ax.set_ylabel('Residuals (Pred - True)')
#     ax.legend()
#     ax.set_title('(a) Residual Plot')
#     ax.grid(True, alpha=0.3)
    
#     # 2. 残差分布直方图
#     ax = axes[0, 1]
#     ax.hist(resid_ours, bins=30, alpha=0.6, label=model_names[0], density=True, color='#1f77b4')
#     ax.hist(resid_base, bins=30, alpha=0.6, label=model_names[1], density=True, color='#d62728')
#     ax.axvline(x=0, color='black', linestyle='--')
#     ax.set_xlabel('Residuals')
#     ax.set_ylabel('Density')
#     ax.legend()
#     ax.set_title('(b) Residual Distribution')
#     ax.grid(True, alpha=0.3)
    
#     # 3. Q-Q图
#     ax = axes[0, 2]
#     stats.probplot(resid_ours, dist="norm", plot=ax)
#     ax.set_title(f'(c) Q-Q Plot ({model_names[0]})')
#     ax.grid(True, alpha=0.3)
    
#     # 4. 残差 vs 预测值
#     ax = axes[1, 0]
#     ax.scatter(y_pred_ours, resid_ours, alpha=0.6, label=model_names[0], s=30, c='#1f77b4')
#     ax.scatter(y_pred_base, resid_base, alpha=0.6, label=model_names[1], s=30, c='#d62728')
#     ax.axhline(y=0, color='black', linestyle='--')
#     ax.set_xlabel('Predicted ΔG')
#     ax.set_ylabel('Residuals')
#     ax.legend()
#     ax.set_title('(d) Residuals vs Predicted')
#     ax.grid(True, alpha=0.3)
    
#     # 5. 绝对残差 vs 真实值
#     ax = axes[1, 1]
#     ax.scatter(y_true, np.abs(resid_ours), alpha=0.6, label=model_names[0], s=30, c='#1f77b4')
#     ax.scatter(y_true, np.abs(resid_base), alpha=0.6, label=model_names[1], s=30, c='#d62728')
#     ax.set_xlabel('True ΔG')
#     ax.set_ylabel('|Residuals|')
#     ax.legend()
#     ax.set_title('(e) Absolute Residuals vs True Values')
#     ax.grid(True, alpha=0.3)
    
#     # 6. 累计误差分布
#     ax = axes[1, 2]
#     sorted_ours = np.sort(np.abs(resid_ours))
#     sorted_base = np.sort(np.abs(resid_base))
#     y_vals = np.arange(1, len(sorted_ours)+1) / len(sorted_ours)
    
#     ax.plot(sorted_ours, y_vals, label=model_names[0], linewidth=2, color='#1f77b4')
#     ax.plot(sorted_base, y_vals, label=model_names[1], linewidth=2, color='#d62728')
#     ax.set_xlabel('Absolute Residual')
#     ax.set_ylabel('Cumulative Probability')
#     ax.legend()
#     ax.set_title('(f) Cumulative Error Distribution')
#     ax.grid(True, alpha=0.3)
    
#     plt.suptitle(f'Residual Analysis: {model_names[0]} vs {model_names[1]}', fontsize=16)
#     plt.tight_layout()
#     plt.savefig(save_path, dpi=300, bbox_inches='tight')
#     plt.show()
#     plt.close()
    
#     print(f"\n✅ 残差分析图已保存至: {save_path}")

# def main():
#     # 文件路径
#     ours_file = "/tmp/AbAgCDR/resultsxin/sabdab_predictions.csv"
#     base_file = "/tmp/AbAgCDR/resultsxin/lightgbm_sabdab_predictions.csv"
#     output_dir = "/tmp/AbAgCDR/resultsxin"
#     output_plot = os.path.join(output_dir, "residual_sabdab_analysis_final.png")
    
#     print("="*70)
#     print("🚀 残差分析 - PACA vs LightGBM")
#     print("="*70)
    
#     # 加载数据
#     df_ours = pd.read_csv(ours_file)
#     df_base = pd.read_csv(base_file)
    
#     # 按Index合并
#     merged = pd.merge(
#         df_ours[['Index', 'true_ddg', 'pred_ddg']], 
#         df_base[['Index', 'pred_ddg']], 
#         on='Index', 
#         suffixes=('_ours', '_base')
#     )
#     merged.columns = ['Index', 'true_ddg', 'pred_ddg_ours', 'pred_ddg_base']
    
#     print(f"\n✅ 合并后样本数: {len(merged)}")
    
#     # 提取数据
#     y_true = merged['true_ddg'].values
#     y_pred_ours = merged['pred_ddg_ours'].values
#     y_pred_base = merged['pred_ddg_base'].values
    
#     # 运行残差分析
#     plot_residual_analysis(
#         y_true, y_pred_ours, y_pred_base,
#         ['PACA', 'LightGBM'],
#         output_plot
#     )

# if __name__ == "__main__":
#     main()