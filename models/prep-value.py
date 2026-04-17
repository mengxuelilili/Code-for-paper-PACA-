# import pandas as pd
# import numpy as np
# from scipy import stats
# import os

# # =================配置区域=================
# # 你的模型 (PACA) 文件路径
# MY_MODEL_PATH = '/tmp/AbAgCDR/resultsxin/skempi_predictions.csv'
# MY_MODEL_PRED_COL = 'pred_ddg'   # 预测值列名
# MY_MODEL_TRUE_COL = 'true_ddg'  # 真实值列名
# MY_MODEL_IDX_COL = 'Index'      # 索引列名

# # Baseline 模型 (PACARPE) 文件路径
# BASELINE_PATH = '/tmp/AbAgCDR/resultsxin/PWAARPEskempi_predictions.csv'
# BASELINE_PRED_COL = 'pred_ddg'   # 预测值列名
# BASELINE_TRUE_COL = 'true_ddg'  # 真实值列名
# BASELINE_IDX_COL = 'Index'      # 索引列名

# # 输出结果保存路径
# OUTPUT_TXT = '/tmp/AbAgCDR/resultsxin/skempisignificance_test_report.txt'
# # =========================================

# def run_significance_test():
#     print("🚀 开始统计显著性分析...")
    
#     # 1. 加载数据
#     if not os.path.exists(MY_MODEL_PATH):
#         raise FileNotFoundError(f"❌ 找不到文件: {MY_MODEL_PATH}")
#     if not os.path.exists(BASELINE_PATH):
#         raise FileNotFoundError(f"❌ 找不到文件: {BASELINE_PATH}")

#     df_my = pd.read_csv(MY_MODEL_PATH)
#     df_base = pd.read_csv(BASELINE_PATH)

#     print(f"📂 加载 PACA 数据: {len(df_my)} 行")
#     print(f"📂 加载 PACARPE 数据: {len(df_base)} 行")

#     # 2. 数据清洗与对齐 (基于 Index)
#     # 确保索引列是整数
#     df_my[MY_MODEL_IDX_COL] = pd.to_numeric(df_my[MY_MODEL_IDX_COL], errors='coerce')
#     df_base[BASELINE_IDX_COL] = pd.to_numeric(df_base[BASELINE_IDX_COL], errors='coerce')

#     # 按 Index 排序并合并
#     # 使用 merge 确保只有两个文件都有的 Index 才会被纳入计算 (Inner Join)
#     merged_df = pd.merge(
#         df_my[[MY_MODEL_IDX_COL, MY_MODEL_PRED_COL, MY_MODEL_TRUE_COL]], 
#         df_base[[BASELINE_IDX_COL, BASELINE_PRED_COL, BASELINE_TRUE_COL]],
#         left_on=MY_MODEL_IDX_COL, 
#         right_on=BASELINE_IDX_COL,
#         suffixes=('_my', '_base')
#     ).sort_values(MY_MODEL_IDX_COL)

#     if len(merged_df) == 0:
#         raise ValueError("❌ 合并后数据为空！请检查两个文件中的 'Index' 列是否匹配。")
    
#     if len(merged_df) != len(df_my) or len(merged_df) != len(df_base):
#         print(f"⚠️ 警告：部分 Index 未匹配。原始 PACA:{len(df_my)}, 原始 Base:{len(df_base)}, 匹配后:{len(merged_df)}")

#     n_samples = len(merged_df)
#     print(f"✅ 成功对齐 {n_samples} 个样本用于配对检验。")

#     # 提取数组
#     pred_my = merged_df[f'{MY_MODEL_PRED_COL}_my'].values.astype(float)
#     pred_base = merged_df[f'{BASELINE_PRED_COL}_base'].values.astype(float)
#     true_val = merged_df[f'{MY_MODEL_TRUE_COL}_my'].values.astype(float) # 假设两个文件的真实值是一样的

#     # 计算绝对误差 (Absolute Error)
#     # 误差越小越好
#     error_my = np.abs(pred_my - true_val)
#     error_base = np.abs(pred_base - true_val)

#     # ================= 检验 1: 预测值分布差异 =================
#     # 目的：看两个模型的预测数值本身是否有显著不同
#     t_stat_pred, p_val_pred = stats.ttest_rel(pred_my, pred_base)
    
#     # ================= 检验 2: 预测误差差异 (核心性能指标) =================
#     # 目的：看 PACA 的误差是否显著小于 PACARPE
#     # 原假设 H0: 误差均值相等 (性能无差异)
#     # 备择假设 H1: 误差均值不等 (或 PACA 误差更小)
#     t_stat_err, p_val_err = stats.ttest_rel(error_my, error_base)
    
#     # 计算误差降低的幅度
#     mean_err_my = np.mean(error_my)
#     mean_err_base = np.mean(error_base)
#     err_reduction = mean_err_base - mean_err_my # 正数表示 PACA 误差更小
#     err_reduction_pct = (err_reduction / mean_err_base) * 100 if mean_err_base > 0 else 0

#     # ================= 生成报告 =================
#     report_lines = []
#     report_lines.append("="*60)
#     report_lines.append("📊 统计显著性分析报告 (Paired t-test)")
#     report_lines.append(f"数据集样本量 (N): {n_samples}")
#     report_lines.append("="*60)
    
#     # 1. 描述性统计
#     report_lines.append("\n【1. 描述性统计】")
#     report_lines.append(f"PACA (My Model)  Mean Pred : {np.mean(pred_my):.4f} ± {np.std(pred_my):.4f}")
#     report_lines.append(f"PACARPE (Base)   Mean Pred : {np.mean(pred_base):.4f} ± {np.std(pred_base):.4f}")
#     report_lines.append("-" * 30)
#     report_lines.append(f"PACA (My Model)  Mean Error (MAE): {mean_err_my:.4f}")
#     report_lines.append(f"PACARPE (Base)   Mean Error (MAE): {mean_err_base:.4f}")
#     report_lines.append(f"误差降低幅度: {err_reduction:.4f} ({err_reduction_pct:.2f}%)")

#     # 2. 预测值差异检验
#     report_lines.append("\n【2. 预测值分布差异检验 (Prediction Distribution)】")
#     report_lines.append(f"t-statistic: {t_stat_pred:.4f}")
#     report_lines.append(f"p-value    : {p_val_pred:.6e}")
#     sig_pred = "显著" if p_val_pred < 0.05 else "不显著"
#     report_lines.append(f"结论       : 两个模型的预测值分布差异 {sig_pred} (p {'<' if p_val_pred<0.05 else '>='} 0.05)")

#     # 3. 误差差异检验 (关键)
#     report_lines.append("\n【3. 预测误差性能差异检验 (Performance / MAE Comparison)】")
#     report_lines.append("*(此检验直接反映模型谁更准)*")
#     report_lines.append(f"t-statistic: {t_stat_err:.4f}")
#     report_lines.append(f"p-value    : {p_val_err:.6e}")
    
#     significance_level = ""
#     if p_val_err < 0.001:
#         significance_level = "⭐⭐⭐ 极显著 (p < 0.001)"
#     elif p_val_err < 0.01:
#         significance_level = "⭐⭐ 非常显著 (p < 0.01)"
#     elif p_val_err < 0.05:
#         significance_level = "⭐ 显著 (p < 0.05)"
#     else:
#         significance_level = "❌ 不显著 (p >= 0.05)"
    
#     report_lines.append(f"显著性判定 : {significance_level}")
    
#     if p_val_err < 0.05:
#         if mean_err_my < mean_err_base:
#             report_lines.append("🎉 结论: PACA 的预测误差显著低于 PACARPE，性能显著提升！")
#         else:
#             report_lines.append("⚠️ 结论: PACA 的预测误差显著高于 PACARPE，性能显著下降。")
#     else:
#         report_lines.append("💡 结论: 两个模型的性能差异在统计学上不显著。")

#     report_lines.append("="*60)

#     # 打印到控制台
#     for line in report_lines:
#         print(line)

#     # 保存到文件
#     os.makedirs(os.path.dirname(OUTPUT_TXT), exist_ok=True)
#     with open(OUTPUT_TXT, 'w', encoding='utf-8') as f:
#         f.write('\n'.join(report_lines))
    
#     print(f"\n💾 详细报告已保存至: {OUTPUT_TXT}")

# if __name__ == "__main__":
#     try:
#         run_significance_test()
#     except Exception as e:
#         print(f"❌ 发生错误: {e}")
#         import traceback
#         traceback.print_exc()


import pandas as pd
import numpy as np
from scipy import stats
import os

# ================= 配置区域 =================

BASE_DIR = '/tmp/AbAgCDR/resultsxin/'

SEEDS = [0, 1, 2, 3, 42]

# 多数据集配置（按你的实际文件名改）
DATASETS = {
    "SKEMPI2.0": {
        "my": "skempi_predictions_seed_{seed}.csv",
        "base": "PWAARPEskempi_predictions_seed_{seed}.csv"
    },
    "AB-Bind": {
        "my": "abbind_predictions_seed_{seed}.csv",
        "base": "PWAARPEabbind_predictions_seed_{seed}.csv"
    },
    "SAbDab": {
        "my": "sabdab_predictions_seed_{seed}.csv",
        "base": "PWAARPEsabdab_predictions_seed_{seed}.csv"
    },
    "PaddlePaddle2021": {
        "my": "train_predictions_seed_{seed}.csv",
        "base": "PWAARPEtrain_predictions_seed_{seed}.csv"
    }
}

IDX_COL = 'Index'
PRED_COL = 'pred_ddg'
TRUE_COL = 'true_ddg'

OUTPUT_PATH = '/tmp/AbAgCDR/resultsxin/final_significance_results.txt'

# ===========================================


def load_and_align(df_my, df_base):
    """严格按 Index 对齐（核心！）"""
    df_my[IDX_COL] = pd.to_numeric(df_my[IDX_COL], errors='coerce')
    df_base[IDX_COL] = pd.to_numeric(df_base[IDX_COL], errors='coerce')

    merged = pd.merge(
        df_my[[IDX_COL, PRED_COL, TRUE_COL]],
        df_base[[IDX_COL, PRED_COL, TRUE_COL]],
        on=IDX_COL,
        suffixes=('_my', '_base')
    )

    return merged


def compute_mae_from_merged(df):
    """基于对齐后的数据计算 MAE"""
    err_my = np.abs(df[f"{PRED_COL}_my"] - df[f"{TRUE_COL}_my"])
    err_base = np.abs(df[f"{PRED_COL}_base"] - df[f"{TRUE_COL}_base"])

    return np.mean(err_my), np.mean(err_base)


def compute_cohens_d(diff):
    """Cohen's d（paired）"""
    std = np.std(diff, ddof=1)
    if std == 0:
        return 0
    return np.mean(diff) / std


def run_dataset(dataset_name, config):
    print(f"\n📊 Processing Dataset: {dataset_name}")

    mae_my_list = []
    mae_base_list = []

    for seed in SEEDS:
        path_my = os.path.join(BASE_DIR, config["my"].format(seed=seed))
        path_base = os.path.join(BASE_DIR, config["base"].format(seed=seed))

        if not os.path.exists(path_my) or not os.path.exists(path_base):
            print(f"⚠️ Missing seed {seed}, skip")
            continue

        df_my = pd.read_csv(path_my)
        df_base = pd.read_csv(path_base)

        merged = load_and_align(df_my, df_base)

        if len(merged) == 0:
            print(f"❌ Seed {seed} merge failed")
            continue

        mae_my, mae_base = compute_mae_from_merged(merged)

        mae_my_list.append(mae_my)
        mae_base_list.append(mae_base)

        print(f"Seed {seed}: PACA={mae_my:.4f}, BASE={mae_base:.4f}")

    mae_my_arr = np.array(mae_my_list)
    mae_base_arr = np.array(mae_base_list)

    # t-test
    t_stat, p_val = stats.ttest_rel(mae_my_arr, mae_base_arr)

    # effect size
    diff = mae_base_arr - mae_my_arr
    cohen_d = compute_cohens_d(diff)

    # mean ± std
    mean_my = np.mean(mae_my_arr)
    std_my = np.std(mae_my_arr, ddof=1)

    mean_base = np.mean(mae_base_arr)
    std_base = np.std(mae_base_arr, ddof=1)

    improvement = (mean_base - mean_my) / mean_base * 100

    return {
        "dataset": dataset_name,
        "paca": f"{mean_my:.4f} ± {std_my:.4f}",
        "base": f"{mean_base:.4f} ± {std_base:.4f}",
        "delta": f"{improvement:.2f}%",
        "t": f"{t_stat:.4f}",
        "p": f"{p_val:.2e}",
        "d": f"{cohen_d:.2f}"
    }


def main():
    results = []

    for name, cfg in DATASETS.items():
        res = run_dataset(name, cfg)
        results.append(res)

    print("\n" + "="*90)
    print("📊 FINAL TABLE (Directly usable in paper)")
    print("="*90)

    print(f"{'Dataset':<15} {'PACA MAE':<20} {'Baseline MAE':<20} {'ΔMAE':<10} {'t':<10} {'p':<12} {'d':<6}")
    print("-"*90)

    for r in results:
        print(f"{r['dataset']:<15} {r['paca']:<20} {r['base']:<20} {r['delta']:<10} {r['t']:<10} {r['p']:<12} {r['d']:<6}")

    # 保存
    with open(OUTPUT_PATH, "w") as f:
        for r in results:
            f.write(str(r) + "\n")

    print(f"\n✅ Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()