
# import pandas as pd
# import numpy as np

# print("="*70)
# print("🔍 快速诊断 - 检查数据")
# print("="*70)

# # 加载数据
# df_ours = pd.read_csv("/tmp/AbAgCDR/resultsxin/sabdab_predictions.csv")
# df_base = pd.read_csv("/tmp/AbAgCDR/resultsxin/lightgbm_sabdab_predictions.csv")

# print("\n📊 Ours文件前5行:")
# print(df_ours.head(10))

# print("\n📊 Baseline文件前5行:")
# print(df_base.head(10))

# # 检查数值
# print("\n📈 数值统计:")
# print(f"Ours - true_ddg: mean={df_ours['true_ddg'].mean():.4f}, std={df_ours['true_ddg'].std():.4f}")
# print(f"Ours - pred_ddg: mean={df_ours['pred_ddg'].mean():.4f}, std={df_ours['pred_ddg'].std():.4f}")
# print(f"Base - true_ddg: mean={df_base['true_ddg'].mean():.4f}, std={df_base['true_ddg'].std():.4f}")
# print(f"Base - pred_ddg: mean={df_base['pred_ddg'].mean():.4f}, std={df_base['pred_ddg'].std():.4f}")

# # 检查PACA预测是否等于真实值
# diff = np.abs(df_ours['true_ddg'] - df_ours['pred_ddg'])
# print(f"\n⚠️ PACA预测与真实值差异:")
# print(f"   平均绝对差: {diff.mean():.6f}")
# print(f"   最大绝对差: {diff.max():.6f}")
# print(f"   完全相等的样本数: {(diff < 1e-10).sum()} / {len(df_ours)}")

# if diff.mean() < 1e-10:
#     print("\n❌ 问题确认: PACA的pred_ddg完全等于true_ddg！")
#     print("   说明数据加载时列对应错误")

# import pandas as pd
# import numpy as np
# from sklearn.metrics import mean_squared_error, mean_absolute_error

# # 加载数据
# df_ours = pd.read_csv("/root/autodl-tmp/AbAgCDR/resultsxin2/abbind_predictions_seed_42.csv")
# df_base = pd.read_csv("/root/autodl-tmp/AbAgCDR/resultsxin2/PWAARPEpaddle_predictions_seed_42.csv")

# # 合并
# merged = pd.merge(df_ours, df_base, on='Index', suffixes=('_ours', '_base'))

# y_true = merged['true_ddg_ours'].values
# y_pred_ours = merged['pred_ddg_ours'].values
# y_pred_base = merged['pred_ddg_base'].values

# # 计算残差
# resid_ours = y_pred_ours - y_true
# resid_base = y_pred_base - y_true

# print("="*70)
# print("📊 残差统计")
# print("="*70)

# print(f"\nPACA 残差:")
# print(f"  Mean: {np.mean(resid_ours):.4f}")
# print(f"  Std:  {np.std(resid_ours):.4f}")
# print(f"  Median: {np.median(resid_ours):.4f}")
# print(f"  MAE: {np.mean(np.abs(resid_ours)):.4f}")

# print(f"\nPACARPE 残差:")
# print(f"  Mean: {np.mean(resid_base):.4f}")
# print(f"  Std:  {np.std(resid_base):.4f}")
# print(f"  Median: {np.median(resid_base):.4f}")
# print(f"  MAE: {np.mean(np.abs(resid_base)):.4f}")

# # 按结合强度分组
# percentiles = np.percentile(y_true, [25, 50, 75])
# print(f"\n" + "="*70)
# print("📊 按结合强度分组")
# print("="*70)

# groups = [
#     (y_true <= percentiles[0], 'Strong Binding (lowest 25%)'),
#     ((y_true > percentiles[0]) & (y_true <= percentiles[2]), 'Medium Binding (25-75%)'),
#     (y_true > percentiles[2], 'Weak Binding (highest 25%)')
# ]

# for mask, label in groups:
#     print(f"\n{label} (n={mask.sum()}):")
#     print(f"  PACA MAE: {np.mean(np.abs(resid_ours[mask])):.4f}")
#     print(f"  PACARPE MAE: {np.mean(np.abs(resid_base[mask])):.4f}")


# import pandas as pd
# import numpy as np
# from sklearn.metrics import mean_squared_error, mean_absolute_error

# def analyze_dataset(ours_path, base_path, dataset_name):
#     """通用分析函数"""
    
#     df_ours = pd.read_csv(ours_path)
#     df_base = pd.read_csv(base_path)
    
#     merged = pd.merge(df_ours, df_base, on='Index', suffixes=('_ours', '_base'))
    
#     y_true = merged['true_ddg_ours'].values
#     y_pred_ours = merged['pred_ddg_ours'].values
#     y_pred_base = merged['pred_ddg_base'].values
    
#     resid_ours = y_pred_ours - y_true
#     resid_base = y_pred_base - y_true
    
#     print("="*70)
#     print(f"📊 残差统计 - {dataset_name}")
#     print("="*70)
    
#     print(f"\nPACA 残差:")
#     print(f"  Mean:   {np.mean(resid_ours):.4f}")
#     print(f"  Std:    {np.std(resid_ours):.4f}")
#     print(f"  Median: {np.median(resid_ours):.4f}")
#     print(f"  MAE:    {np.mean(np.abs(resid_ours)):.4f}")
    
#     print(f"\nPACARPE 残差:")
#     print(f"  Mean:   {np.mean(resid_base):.4f}")
#     print(f"  Std:    {np.std(resid_base):.4f}")
#     print(f"  Median: {np.median(resid_base):.4f}")
#     print(f"  MAE:    {np.mean(np.abs(resid_base)):.4f}")
    
#     # 按结合强度分组
#     percentiles = np.percentile(y_true, [25, 50, 75])
#     print(f"\n" + "="*70)
#     print("📊 按结合强度分组")
#     print("="*70)
    
#     groups = [
#         (y_true <= percentiles[0], 'Strong Binding (lowest 25%)'),
#         ((y_true > percentiles[0]) & (y_true <= percentiles[2]), 'Medium Binding (25-75%)'),
#         (y_true > percentiles[2], 'Weak Binding (highest 25%)')
#     ]
    
#     for mask, label in groups:
#         if mask.sum() > 0:
#             print(f"\n{label} (n={mask.sum()}):")
#             print(f"  PACA MAE:     {np.mean(np.abs(resid_ours[mask])):.4f}")
#             print(f"  PACARPE MAE:  {np.mean(np.abs(resid_base[mask])):.4f}")
#             improvement = (1 - np.mean(np.abs(resid_ours[mask])) / np.mean(np.abs(resid_base[mask]))) * 100
#             print(f"  Improvement:  {improvement:.2f}%")
    
#     print("\n" + "-"*70 + "\n")
    
#     return {
#         'dataset': dataset_name,
#         'paca_mae': np.mean(np.abs(resid_ours)),
#         'base_mae': np.mean(np.abs(resid_base)),
#         'groups': {
#             'strong': {
#                 'n': (y_true <= percentiles[0]).sum(),
#                 'paca_mae': np.mean(np.abs(resid_ours[y_true <= percentiles[0]])),
#                 'base_mae': np.mean(np.abs(resid_base[y_true <= percentiles[0]])),
#             },
#             'medium': {
#                 'n': ((y_true > percentiles[0]) & (y_true <= percentiles[2])).sum(),
#                 'paca_mae': np.mean(np.abs(resid_ours[(y_true > percentiles[0]) & (y_true <= percentiles[2])])),
#                 'base_mae': np.mean(np.abs(resid_base[(y_true > percentiles[0]) & (y_true <= percentiles[2])])),
#             },
#             'weak': {
#                 'n': (y_true > percentiles[2]).sum(),
#                 'paca_mae': np.mean(np.abs(resid_ours[y_true > percentiles[2]])),
#                 'base_mae': np.mean(np.abs(resid_base[y_true > percentiles[2]])),
#             }
#         }
#     }


# # ======================================================
# # 运行分析（修正路径）
# # ======================================================

# datasets = [
#     ('paddle', '/root/autodl-tmp/AbAgCDR/resultsxin2/paddle_predictions_seed_42.csv', '/root/autodl-tmp/AbAgCDR/resultsxin2/PWAARPEpaddle_predictions_seed_42.csv'),
#     ('abbind', '/root/autodl-tmp/AbAgCDR/resultsxin2/abbind_predictions_seed_42.csv', '/root/autodl-tmp/AbAgCDR/resultsxin2/PWAARPEabbind_predictions_seed_42.csv'),
#     ('sabdab', '/root/autodl-tmp/AbAgCDR/resultsxin2/sabdab_predictions_seed_42.csv', '/root/autodl-tmp/AbAgCDR/resultsxin2/PWAARPEsabdab_predictions_seed_42.csv'),
#     ('skempi', '/root/autodl-tmp/AbAgCDR/resultsxin2/skempi_predictions_seed_42.csv', '/root/autodl-tmp/AbAgCDR/resultsxin2/PWAARPEskempi_predictions_seed_42.csv'),
# ]

# results = {}
# for name, ours_path, base_path in datasets:
#     results[name] = analyze_dataset(ours_path, base_path, name.capitalize())


# # ======================================================
# # 打印汇总表格（直接用于论文）
# # ======================================================
# print("\n" + "="*70)
# print("📊 汇总表格（可直接用于论文 Table/Figure）")
# print("="*70)

# print("\n| Dataset | Group | n | PACA MAE | PACARPE MAE | Improvement |")
# print("|---------|-------|---|----------|-------------|-------------|")

# for name, res in results.items():
#     for group_name in ['strong', 'medium', 'weak']:
#         g = res['groups'][group_name]
#         imp = (1 - g['paca_mae'] / g['base_mae']) * 100
#         print(f"| {name.capitalize()} | {group_name.capitalize()} | {g['n']} | {g['paca_mae']:.4f} | {g['base_mae']:.4f} | {imp:.2f}% |")

# 预测的是真实值
# import pandas as pd
# import numpy as np
# from sklearn.metrics import mean_squared_error, mean_absolute_error


# def analyze_dataset(ours_path, base_path, dataset_name):
#     """通用分析函数：MAE + RMSE，按结合强度分三组统计"""
#     df_ours = pd.read_csv(ours_path)
#     df_base = pd.read_csv(base_path)

#     merged = pd.merge(df_ours, df_base, on='Index', suffixes=('_ours', '_base'))

#     y_true = merged['true_ddg_ours'].values
#     y_pred_ours = merged['pred_ddg_ours'].values
#     y_pred_base = merged['pred_ddg_base'].values

#     resid_ours = y_pred_ours - y_true
#     resid_base = y_pred_base - y_true

#     # 全局指标
#     mae_ours = np.mean(np.abs(resid_ours))
#     mae_base = np.mean(np.abs(resid_base))
#     rmse_ours = np.sqrt(mean_squared_error(y_true, y_pred_ours))
#     rmse_base = np.sqrt(mean_squared_error(y_true, y_pred_base))

#     print("=" * 70)
#     print(f"📊 残差统计 - {dataset_name}")
#     print("=" * 70)

#     print(f"\nPACA (PACA‑Affinity):")
#     print(f"  Mean Resid: {np.mean(resid_ours):.4f}")
#     print(f"  Std Resid:  {np.std(resid_ours):.4f}")
#     print(f"  Median Res:{np.median(resid_ours):.4f}")
#     print(f"  MAE:       {mae_ours:.4f}")
#     print(f"  RMSE:      {rmse_ours:.4f}")

#     print(f"\nPACARPE (PACA+RPE):")
#     print(f"  Mean Resid: {np.mean(resid_base):.4f}")
#     print(f"  Std Resid:  {np.std(resid_base):.4f}")
#     print(f"  Median Res:{np.median(resid_base):.4f}")
#     print(f"  MAE:       {mae_base:.4f}")
#     print(f"  RMSE:      {rmse_base:.4f}")

#     # 按真实ddg分结合强度分组：25% / 50% /75%分位数
#     percentiles = np.percentile(y_true, [25, 50, 75])
#     mask_strong = y_true <= percentiles[0]
#     mask_medium = (y_true > percentiles[0]) & (y_true <= percentiles[2])
#     mask_weak = y_true > percentiles[2]

#     print(f"\n" + "=" * 70)
#     print("📊 按结合强度分组 (MAE & RMSE)")
#     print("=" * 70)

#     group_info_list = [
#         (mask_strong, 'Strong Binding (lowest 25%)'),
#         (mask_medium, 'Medium Binding (25‑75%)'),
#         (mask_weak, 'Weak Binding (highest 25%)')
#     ]

#     group_dict_out = {}
#     for mask, label in group_info_list:
#         n_sample = int(mask.sum())
#         if n_sample <= 0:
#             continue
#         # 分组指标
#         sub_ytrue = y_true[mask]
#         sub_pred_ours = y_pred_ours[mask]
#         sub_pred_base = y_pred_base[mask]

#         g_mae_ours = np.mean(np.abs(sub_pred_ours - sub_ytrue))
#         g_mae_base = np.mean(np.abs(sub_pred_base - sub_ytrue))
#         g_rmse_ours = np.sqrt(mean_squared_error(sub_ytrue, sub_pred_ours))
#         g_rmse_base = np.sqrt(mean_squared_error(sub_ytrue, sub_pred_base))

#         imp_mae = (1 - g_mae_ours / g_mae_base) * 100

#         print(f"\n{label} (n={n_sample}):")
#         print(f"  PACA‑Affinity   MAE:{g_mae_ours:.4f} | RMSE:{g_rmse_ours:.4f}")
#         print(f"  PACA+RPE        MAE:{g_mae_base:.4f} | RMSE:{g_rmse_base:.4f}")
#         print(f"  MAE Improvement:{imp_mae:.2f}%")

#         group_dict_out[label.split()[0].lower()] = {
#             "n": n_sample,
#             "paca_mae": g_mae_ours,
#             "base_mae": g_mae_base,
#             "paca_rmse": g_rmse_ours,
#             "base_rmse": g_rmse_base
#         }

#     print("\n" + "-" * 70 + "\n")

#     return {
#         'dataset': dataset_name,
#         'paca_mae': mae_ours,
#         'base_mae': mae_base,
#         'paca_rmse': rmse_ours,
#         'base_rmse': rmse_base,
#         'groups': group_dict_out
#     }


# # ======================================================
# # 运行分析
# # ======================================================
# datasets = [
#     ('paddle', '/root/autodl-tmp/AbAgCDR/resultsxin2/paddle_predictions_seed_42.csv',
#      '/root/autodl-tmp/AbAgCDR/resultsxin2/PWAARPEpaddle_predictions_seed_42.csv'),
#     ('abbind', '/root/autodl-tmp/AbAgCDR/resultsxin2/abbind_predictions_seed_42.csv',
#      '/root/autodl-tmp/AbAgCDR/resultsxin2/PWAARPEabbind_predictions_seed_42.csv'),
#     ('sabdab', '/root/autodl-tmp/AbAgCDR/resultsxin2/sabdab_predictions_seed_42.csv',
#      '/root/autodl-tmp/AbAgCDR/resultsxin2/PWAARPEsabdab_predictions_seed_42.csv'),
#     ('skempi', '/root/autodl-tmp/AbAgCDR/resultsxin2/skempi_predictions_seed_42.csv',
#      '/root/autodl-tmp/AbAgCDR/resultsxin2/PWAARPEskempi_predictions_seed_42.csv'),
# ]

# results = {}
# for name, ours_path, base_path in datasets:
#     results[name] = analyze_dataset(ours_path, base_path, name.capitalize())

# # ======================================================
# # Markdown汇总表格（MAE+RMSE，论文直接复制）
# # ======================================================
# print("\n" + "=" * 85)
# print("📊 汇总表格（MAE + RMSE，可直接复制到论文）")
# print("=" * 85)
# print("\n| Dataset | Group | n | PACA‑Affinity MAE | PACA+RPE MAE | MAE Imp. | PACA‑Affinity RMSE | PACA+RPE RMSE |")
# print("|---------|-------|---|-------------------|--------------|----------|--------------------|---------------|")

# for name, res in results.items():
#     for group_name in ['strong', 'medium', 'weak']:
#         g = res['groups'][group_name]
#         imp_mae = (1 - g['paca_mae'] / g['base_mae']) * 100
#         print(
#             f"| {name.capitalize()} | {group_name.capitalize()} | {g['n']} | "
#             f"{g['paca_mae']:.4f} | {g['base_mae']:.4f} | {imp_mae:.2f}% | "
#             f"{g['paca_rmse']:.4f} | {g['base_rmse']:.4f} |"
#         )

# 使用的是归一化后的
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error


def analyze_dataset(ours_path, base_path, dataset_name):
    """通用分析函数：MAE + RMSE，按结合强度分三组统计（归一化版本）"""
    df_ours = pd.read_csv(ours_path)
    df_base = pd.read_csv(base_path)

    merged = pd.merge(df_ours, df_base, on='Index', suffixes=('_ours', '_base'))

    # 适配归一化后的列名
    y_true = merged['true_ddg_normalized_ours'].values
    y_pred_ours = merged['pred_ddg_normalized_ours'].values
    y_pred_base = merged['pred_ddg_normalized_base'].values

    resid_ours = y_pred_ours - y_true
    resid_base = y_pred_base - y_true

    # 全局指标
    mae_ours = np.mean(np.abs(resid_ours))
    mae_base = np.mean(np.abs(resid_base))
    rmse_ours = np.sqrt(mean_squared_error(y_true, y_pred_ours))
    rmse_base = np.sqrt(mean_squared_error(y_true, y_pred_base))

    print("=" * 70)
    print(f"📊 残差统计 - {dataset_name} (归一化空间)")
    print("=" * 70)

    print(f"\nPACA (Normalized):")
    print(f"  Mean Resid: {np.mean(resid_ours):.4f}")
    print(f"  Std Resid:  {np.std(resid_ours):.4f}")
    print(f"  Median Res:{np.median(resid_ours):.4f}")
    print(f"  MAE:       {mae_ours:.4f}")
    print(f"  RMSE:      {rmse_ours:.4f}")

    print(f"\nPACARPE (Normalized):")
    print(f"  Mean Resid: {np.mean(resid_base):.4f}")
    print(f"  Std Resid:  {np.std(resid_base):.4f}")
    print(f"  Median Res:{np.median(resid_base):.4f}")
    print(f"  MAE:       {mae_base:.4f}")
    print(f"  RMSE:      {rmse_base:.4f}")

    # 按真实ddg分结合强度分组：25% / 50% /75%分位数
    percentiles = np.percentile(y_true, [25, 50, 75])
    mask_strong = y_true <= percentiles[0]
    mask_medium = (y_true > percentiles[0]) & (y_true <= percentiles[2])
    mask_weak = y_true > percentiles[2]

    print(f"\n" + "=" * 70)
    print("📊 按结合强度分组 (MAE & RMSE)")
    print("=" * 70)

    group_info_list = [
        (mask_strong, 'Strong Binding (lowest 25%)'),
        (mask_medium, 'Medium Binding (25‑75%)'),
        (mask_weak, 'Weak Binding (highest 25%)')
    ]

    group_dict_out = {}
    for mask, label in group_info_list:
        n_sample = int(mask.sum())
        if n_sample <= 0:
            continue
        # 分组指标
        sub_ytrue = y_true[mask]
        sub_pred_ours = y_pred_ours[mask]
        sub_pred_base = y_pred_base[mask]

        g_mae_ours = np.mean(np.abs(sub_pred_ours - sub_ytrue))
        g_mae_base = np.mean(np.abs(sub_pred_base - sub_ytrue))
        g_rmse_ours = np.sqrt(mean_squared_error(sub_ytrue, sub_pred_ours))
        g_rmse_base = np.sqrt(mean_squared_error(sub_ytrue, sub_pred_base))

        imp_mae = (1 - g_mae_ours / g_mae_base) * 100

        print(f"\n{label} (n={n_sample}):")
        print(f"  PACA (Norm)   MAE:{g_mae_ours:.4f} | RMSE:{g_rmse_ours:.4f}")
        print(f"  PACARPE(Norm) MAE:{g_mae_base:.4f} | RMSE:{g_rmse_base:.4f}")
        print(f"  MAE Improvement:{imp_mae:.2f}%")

        group_dict_out[label.split()[0].lower()] = {
            "n": n_sample,
            "paca_mae": g_mae_ours,
            "base_mae": g_mae_base,
            "paca_rmse": g_rmse_ours,
            "base_rmse": g_rmse_base
        }

    print("\n" + "-" * 70 + "\n")

    return {
        'dataset': dataset_name,
        'paca_mae': mae_ours,
        'base_mae': mae_base,
        'paca_rmse': rmse_ours,
        'base_rmse': rmse_base,
        'groups': group_dict_out
    }


# ======================================================
# 运行分析
# ======================================================
datasets = [
    ('paddle', '/root/autodl-tmp/AbAgCDR/resultsxin3guiyihua/paddle_predictions_normalized_seed_42.csv',
     '/root/autodl-tmp/AbAgCDR/resultsxin3guiyihua/PWAARPEpaddle_predictions_normalized_seed_42.csv'),
    ('abbind', '/root/autodl-tmp/AbAgCDR/resultsxin3guiyihua/abbind_predictions_normalized_seed_42.csv',
     '/root/autodl-tmp/AbAgCDR/resultsxin3guiyihua/PWAARPEabbind_predictions_normalized_seed_42.csv'),
    ('sabdab', '/root/autodl-tmp/AbAgCDR/resultsxin3guiyihua/sabdab_predictions_normalized_seed_42.csv',
     '/root/autodl-tmp/AbAgCDR/resultsxin3guiyihua/PWAARPEsabdab_predictions_normalized_seed_42.csv'),
    ('skempi', '/root/autodl-tmp/AbAgCDR/resultsxin3guiyihua/skempi_predictions_normalized_seed_42.csv',
     '/root/autodl-tmp/AbAgCDR/resultsxin3guiyihua/PWAARPEskempi_predictions_normalized_seed_42.csv'),
]

results = {}
for name, ours_path, base_path in datasets:
    results[name] = analyze_dataset(ours_path, base_path, name.capitalize())

# ======================================================
# Markdown汇总表格（MAE+RMSE，论文直接复制）
# ======================================================
print("\n" + "=" * 85)
print("📊 汇总表格（归一化空间 MAE + RMSE，可直接复制到论文）")
print("=" * 85)
print("\n| Dataset | Group | n | PACA‑Affinity MAE | PACA+RPE MAE | MAE Imp. | PACA‑Affinity RMSE | PACA+RPE RMSE |")
print("|---------|-------|---|-------------------|--------------|----------|--------------------|---------------|")

for name, res in results.items():
    for group_name in ['strong', 'medium', 'weak']:
        g = res['groups'][group_name]
        imp_mae = (1 - g['paca_mae'] / g['base_mae']) * 100
        print(
            f"| {name.capitalize()} | {group_name.capitalize()} | {g['n']} | "
            f"{g['paca_mae']:.4f} | {g['base_mae']:.4f} | {imp_mae:.2f}% | "
            f"{g['paca_rmse']:.4f} | {g['base_rmse']:.4f} |"
        )