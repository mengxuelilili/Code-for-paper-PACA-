import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os

def analyze_by_binding_strength(y_true, y_pred_ours, y_pred_baseline, model_names, save_path):
    """按结合强度分组分析"""
    
    # 计算分位数
    percentiles = np.percentile(y_true, [25, 50, 75])
    
    print("\n" + "="*80)
    print("📊 结合强度分组 (基于真实ΔG)")
    print("="*80)
    print(f"强结合阈值: < {percentiles[0]:.2f} (最低25%)")
    print(f"中等结合: {percentiles[0]:.2f} - {percentiles[2]:.2f} (中间50%)")
    print(f"弱结合阈值: > {percentiles[2]:.2f} (最高25%)")
    
    # 分组
    groups = [
        (y_true <= percentiles[0], 'Strong Binding\n(lowest 25%)'),
        ((y_true > percentiles[0]) & (y_true <= percentiles[2]), 'Medium Binding\n(25-75%)'),
        (y_true > percentiles[2], 'Weak Binding\n(highest 25%)')
    ]
    
    results = []
    
    print("\n" + "="*80)
    print("📊 Performance by Binding Strength")
    print("="*80)
    
    for mask, label in groups:
        y_t = y_true[mask]
        y_o = y_pred_ours[mask]
        y_b = y_pred_baseline[mask]
        
        rmse_o = np.sqrt(mean_squared_error(y_t, y_o))
        rmse_b = np.sqrt(mean_squared_error(y_t, y_b))
        mae_o = mean_absolute_error(y_t, y_o)
        mae_b = mean_absolute_error(y_t, y_b)
        
        # 判断哪个模型更好
        better = "PACA" if rmse_o < rmse_b else "PACARPE"
        improvement = ((rmse_b - rmse_o) / rmse_b) * 100
        
        print(f"\n{label} (n={len(y_t)}):")
        print(f"  RMSE - PACA: {rmse_o:.4f}, PACARPE: {rmse_b:.4f}")
        print(f"  MAE  - PACA: {mae_o:.4f}, PACARPE: {mae_b:.4f}")
        print(f"  Improvement: {improvement:+.2f}% ({better} better)")
        
        results.append({
            'Group': label.replace('\n', ' '),
            'Count': len(y_t),
            'RMSE_Ours': rmse_o,
            'RMSE_Baseline': rmse_b,
            'MAE_Ours': mae_o,
            'MAE_Baseline': mae_b,
            'Better': better,
            'Improvement': improvement
        })
    
    # 绘制柱状图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # RMSE柱状图
    x = np.arange(len(results))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, [r['RMSE_Ours'] for r in results], width, 
                    label='PACA', color='#1f77b4', alpha=0.8, edgecolor='black', linewidth=0.5)
    bars2 = ax1.bar(x + width/2, [r['RMSE_Baseline'] for r in results], width, 
                    label='PACARPE', color='#d62728', alpha=0.8, edgecolor='black', linewidth=0.5)
    
    # 添加数值标签
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{height:.2f}', ha='center', va='bottom', fontsize=9)
    
    ax1.set_xlabel('Binding Strength Group', fontsize=12)
    ax1.set_ylabel('RMSE', fontsize=12)
    ax1.set_title('RMSE by Binding Strength', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([r['Group'] for r in results], fontsize=10)
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim(0, max(max(r['RMSE_Ours'] for r in results), 
                        max(r['RMSE_Baseline'] for r in results)) * 1.15)
    
    # MAE柱状图
    bars1 = ax2.bar(x - width/2, [r['MAE_Ours'] for r in results], width, 
                    label='PACA', color='#1f77b4', alpha=0.8, edgecolor='black', linewidth=0.5)
    bars2 = ax2.bar(x + width/2, [r['MAE_Baseline'] for r in results], width, 
                    label='LightGBM', color='#d62728', alpha=0.8, edgecolor='black', linewidth=0.5)
    
    # 添加数值标签
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{height:.2f}', ha='center', va='bottom', fontsize=9)
    
    ax2.set_xlabel('Binding Strength Group', fontsize=12)
    ax2.set_ylabel('MAE', fontsize=12)
    ax2.set_title('MAE by Binding Strength', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels([r['Group'] for r in results], fontsize=10)
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim(0, max(max(r['MAE_Ours'] for r in results), 
                        max(r['MAE_Baseline'] for r in results)) * 1.15)
    
    plt.suptitle('Performance Comparison by Binding Strength\nPACA vs PACARPE on Paddle2021 Dataset', 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    
    print(f"\n✅ 长尾效应分析图保存至: {save_path}")
    
    return pd.DataFrame(results)

def main():
    # 文件路径
    ours_file = "/tmp/AbAgCDR/resultsxin/train_predictions.csv"
    base_file = "/tmp/AbAgCDR/resultsxin/PWAARPEtrain_predictions.csv"
    output_dir = "/tmp/AbAgCDR/resultsxin"
    output_plot = os.path.join(output_dir, "train_binding_strength_analysis.png")
    output_csv = os.path.join(output_dir, "train_binding_strength_results.csv")
    
    print("="*70)
    print("🚀 长尾效应分析 - 按结合强度分组")
    print("="*70)
    print(f"📊 基于数据对齐检查结果: ✅ 269样本完全匹配")
    
    # 加载数据
    df_ours = pd.read_csv(ours_file)
    df_base = pd.read_csv(base_file)
    
    print(f"\n📊 Ours文件列名: {list(df_ours.columns)}")
    print(f"📊 Baseline文件列名: {list(df_base.columns)}")
    
    # 按Index合并
    merged = pd.merge(df_ours, df_base, on='Index', suffixes=('_ours', '_base'))
    print(f"\n✅ 合并后: {len(merged)} 样本")
    print(f"📊 合并后列名: {list(merged.columns)}")
    
    # 提取数据 - 使用正确的列名
    # 真实值在两个文件中相同，任选一个即可
    y_true = merged['true_ddg_ours'].values  # 或 merged['true_ddg_base'].values
    
    y_pred_ours = merged['pred_ddg_ours'].values
    y_pred_base = merged['pred_ddg_base'].values
    
    print(f"\n✅ 数据提取成功:")
    print(f"   y_true 范围: [{y_true.min():.2f}, {y_true.max():.2f}]")
    print(f"   y_pred_ours 范围: [{y_pred_ours.min():.2f}, {y_pred_ours.max():.2f}]")
    print(f"   y_pred_base 范围: [{y_pred_base.min():.2f}, {y_pred_base.max():.2f}]")
    
    # 验证真实值是否一致（可选）
    y_true_base = merged['true_ddg_base'].values
    if np.allclose(y_true, y_true_base):
        print("   ✅ 两个文件的真实值一致")
    else:
        print("   ⚠️ 警告: 两个文件的真实值不完全一致")
    
    # 运行长尾效应分析
    results_df = analyze_by_binding_strength(
        y_true, y_pred_ours, y_pred_base,
        ['PACA', 'PACARPE'],
        output_plot
    )
    
    # 保存结果
    results_df.to_csv(output_csv, index=False)
    print(f"\n✅ 详细结果保存至: {output_csv}")

if __name__ == "__main__":
    main()