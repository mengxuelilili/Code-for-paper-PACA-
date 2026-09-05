import pandas as pd
import numpy as np
from scipy import stats

# ======================================================
# 配置：指向归一化后的预测结果 CSV
# ======================================================
# 假设你的文件命名规则是：PWAARPE{name}_predictions_normalized_seed_42.csv
# 请根据实际文件名修改路径
datasets = ['paddle', 'abbind', 'sabdab', 'skempi']
base_dir = "/root/autodl-tmp/AbAgCDR/resultsxin3guiyihua/" # 根据你之前的路径修改

results_table = []

print(f"{'Dataset':<15} | {'PACA MAE':<15} | {'Base MAE':<15} | {'Delta(%)':<10} | {'t-value':<10} | {'p-value':<12} | {'Cohen d':<10}")
print("-" * 100)

for name in datasets:
    # 1. 读取数据
    # 注意：这里假设两个模型预测的是同一批 Index，且列名包含 normalized
    file_paca = f"{base_dir}{name}_predictions_normalized_seed_42.csv" 
    file_base = f"{base_dir}PWAARPE{name}_predictions_normalized_seed_42.csv" 
    
    # 如果文件名前缀不同，请在这里调整，确保读到的是归一化数据
    try:
        df_paca = pd.read_csv(file_paca)
        df_base = pd.read_csv(file_base)
    except FileNotFoundError as e:
        print(f"❌ 文件未找到: {e}")
        continue

    # 2. 计算绝对误差 (Absolute Errors) - 关键步骤
    # 使用归一化列名
    err_paca = np.abs(df_paca['true_ddg_normalized'] - df_paca['pred_ddg_normalized'])
    err_base = np.abs(df_base['true_ddg_normalized'] - df_base['pred_ddg_normalized'])

    # 3. 计算 MAE
    mae_paca = np.mean(err_paca)
    mae_base = np.mean(err_base)
    
    # 计算标准差 (用于表格中的 ± 符号，通常指 MAE 的标准误或样本标准差，这里用样本标准差)
    std_paca = np.std(err_paca, ddof=1)
    std_base = np.std(err_base, ddof=1)

    # 4. 计算提升幅度 Delta
    delta = (1 - mae_paca / mae_base) * 100

    # 5. 配对 t 检验 (Paired t-test)
    # 比较的是误差分布，而不是最终的一个 MAE 数值
    t_stat, p_value = stats.ttest_rel(err_base, err_paca) 
    # 注意：ttest_rel(x, y) 如果 x > y，t值为正。这里我们要看 Base 是否显著大于 Paca

    # 6. Cohen's d (配对样本)
    # d = mean(diff) / std(diff)
    diff = err_base - err_paca
    cohens_d = np.mean(diff) / np.std(diff, ddof=1)

    # 格式化输出
    display_name = name.capitalize()
    if name == 'abbind': display_name = 'AB-Bind'
    if name == 'sabdab': display_name = 'SAbDab'
    if name == 'skempi': display_name = 'SKEMPI2.0'
    if name == 'paddle': display_name = 'Paddle2021'

    print(f"{display_name:<15} | {mae_paca:.4f}±{std_paca:.4f} | {mae_base:.4f}±{std_base:.4f} | {delta:>6.2f}%   | {t_stat:>8.4f} | {p_value:>10.2e} | {cohens_d:>8.2f}")

    results_table.append({
        'Dataset': display_name,
        'PACA_MAE': f"{mae_paca:.4f} ± {std_paca:.4f}",
        'Base_MAE': f"{mae_base:.4f} ± {std_base:.4f}",
        'Delta': f"{delta:.2f}%",
        't_value': f"{t_stat:.4f}",
        'p_value': f"{p_value:.2e}",
        'Cohen_d': f"{cohens_d:.2f}"
    })

# 生成 Markdown 表格方便复制
print("\n\n================ 复制以下内容到论文 =================")
print("| Dataset | PACA-Affinity (MAE) | PACA+RPE (MAE) | Δ (%) | t-value | p-value | Cohen’s d |")
print("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
for row in results_table:
    print(f"| {row['Dataset']} | {row['PACA_MAE']} | {row['Base_MAE']} | {row['Delta']} | {row['t_value']} | {row['p_value']} | {row['Cohen_d']} |")