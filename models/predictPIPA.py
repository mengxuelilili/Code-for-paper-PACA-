# import tensorflow as tf
# from tensorflow.keras.models import load_model
# import numpy as np
# import pandas as pd
# from sklearn.metrics import mean_squared_error, r2_score
# import matplotlib.pyplot as plt

# # 假设你已经保存了模型，并知道其路径
# MODEL_PATH = 'path_to_saved_model/dg_affinity_regression_model.h5'
# TEST_DATA_PATH = '/tmp/AbAgCDR/data/pairs_seq_abbind2.tsv'  # 示例测试数据集路径

# def load_test_data(file_path):
#     df = pd.read_csv(file_path, sep='\t')
#     return df['heavy'].tolist(), df['light'].tolist(), df['antigen'].tolist(), df['delta_g'].values.astype(np.float32)

# def embed_seqs(seqs, seq2t, seq_size=2000):
#     return np.array([seq2t.embed_normalized(seq, seq_size) for seq in seqs])

# def main():
#     # 加载模型
#     model = load_model(MODEL_PATH)
    
#     # 加载并处理测试数据
#     h_test, l_test, ag_test, y_true = load_test_data(TEST_DATA_PATH)
    
#     # 加载嵌入函数
#     from embeddings.seq2tensor import s2t
#     seq2t = s2t('../../../embeddings/default_onehot.txt')
#     dim = seq2t.dim
    
#     X_h_test = embed_seqs(h_test, seq2t)
#     X_l_test = embed_seqs(l_test, seq2t)
#     X_ag_test = embed_seqs(ag_test, seq2t)
    
#     # 进行预测
#     preds = model.predict([X_h_test, X_l_test, X_ag_test]).flatten()
    
#     # 计算误差
#     mse = mean_squared_error(y_true, preds)
#     rmse = np.sqrt(mse)
#     r2 = r2_score(y_true, preds)
#     print(f"RMSE: {rmse:.4f}, R²: {r2:.4f}")
    
#     # 绘制回归图
#     plt.figure(figsize=(8, 6))
#     plt.scatter(y_true, preds, alpha=0.6, color='#1f77b4', s=20)
#     min_val = min(y_true.min(), preds.min())
#     max_val = max(y_true.max(), preds.max())
#     plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Ideal Fit')
#     plt.xlabel('True ΔΔG', fontsize=12)
#     plt.ylabel('Predicted ΔΔG', fontsize=12)
#     plt.title('Regression: True vs Predicted ΔΔG', fontsize=14)
#     plt.legend()
#     plt.grid(True, linestyle='--', alpha=0.5)
#     plt.tight_layout()
#     plt.show()

# if __name__ == "__main__":
#     main()

# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
"""
抗体-抗原结合亲和力预测脚本（基于已训练的 TensorFlow 模型）
✅ 已修复：强制使用 CPU，避免 CuDNN 版本冲突
输入：TSV 文件（需含 heavy/light/antigen/delta_g 列）
输出：
  - results/predictions.csv
  - results/regression_plot.png
"""

import os
# ⚠️ 必须在 import tensorflow 之前禁用 GPU！
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
print("⚠️ 已禁用 GPU，使用 CPU 运行（避免 CuDNN 版本冲突）")

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# === 添加 embeddings 路径 ===
embed_path = os.path.abspath('../../../embeddings')
if embed_path not in sys.path:
    sys.path.append(embed_path)

try:
    from embeddings.seq2tensor import s2t
except ImportError:
    raise ImportError(f"❌ 未找到 seq2tensor 模块。请确保路径正确: {embed_path}")

# === TensorFlow 导入（此时已确定使用 CPU）===
import tensorflow as tf
from tensorflow.keras.models import load_model

# === 配置参数 ===
SEQ_SIZE = 2000
MODEL_PATH = 'results/dg_affinity_regression_model.h5'      # 训练保存的模型
INPUT_TSV = '/tmp/AbAgCDR/data/pairs_seq_benchmark1.tsv'       # 可根据需要修改
OUTPUT_DIR = 'results'
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_and_preprocess_tsv(file_path):
    """加载 TSV 并标准化列名"""
    df = pd.read_csv(file_path, sep='\t')
    cols_lower = [c.lower() for c in df.columns]
    col_map = {}

    # 自动匹配列名
    for name in ['antibody_seq_a', 'heavy', 'vh', 'h', 'hc']:
        if name in cols_lower:
            col_map['heavy'] = df.columns[cols_lower.index(name)]
            break
    for name in ['antibody_seq_b', 'light', 'vl', 'l', 'lc']:
        if name in cols_lower:
            col_map['light'] = df.columns[cols_lower.index(name)]
            break
    for name in ['antigen_seq', 'antigen', 'ag']:
        if name in cols_lower:
            col_map['antigen'] = df.columns[cols_lower.index(name)]
            break
    for name in ['delta_g', 'ddg', 'deltag', 'binding_affinity', 'affinity']:
        if name in cols_lower:
            col_map['delta_g'] = df.columns[cols_lower.index(name)]
            break

    required = ['heavy', 'light', 'antigen', 'delta_g']
    if not all(k in col_map for k in required):
        raise ValueError(f"TSV 缺少必要列！需包含 heavy/light/antigen/delta_g。可用列: {df.columns.tolist()}")

    df_clean = df[[col_map['heavy'], col_map['light'], col_map['antigen'], col_map['delta_g']]].copy()
    df_clean.columns = ['heavy', 'light', 'antigen', 'delta_g']
    df_clean.dropna(subset=['heavy', 'light', 'antigen', 'delta_g'], inplace=True)
    df_clean['delta_g'] = pd.to_numeric(df_clean['delta_g'], errors='coerce')
    df_clean.dropna(subset=['delta_g'], inplace=True)

    valid_aa = set("ACDEFGHIKLMNPQRSTVWY")
    def is_valid(seq):
        return isinstance(seq, str) and len(seq.strip()) > 0 and all(c in valid_aa for c in seq.upper())
    
    before = len(df_clean)
    df_clean = df_clean[
        df_clean['heavy'].apply(is_valid) &
        df_clean['light'].apply(is_valid) &
        df_clean['antigen'].apply(is_valid)
    ]
    after = len(df_clean)
    print(f"📥 加载 {file_path} → 有效样本: {after} / {before}")
    return df_clean


def embed_seqs(seqs, seq2t, seq_size=2000):
    from tqdm import tqdm
    return np.array([seq2t.embed_normalized(seq, seq_size) for seq in tqdm(seqs, desc="Embedding", leave=False)])


def main():
    print("🔍 开始预测流程...")

    # 1. 加载模型（CPU 模式）
    print(f"📂 加载模型: {MODEL_PATH}")
    model = load_model(MODEL_PATH)
    print("✅ 模型加载成功（CPU 模式）")

    # 2. 加载数据
    df = load_and_preprocess_tsv(INPUT_TSV)
    heavies = df['heavy'].tolist()
    lights = df['light'].tolist()
    antigens = df['antigen'].tolist()
    true_ddg = df['delta_g'].values.astype(np.float32)

    # 3. 嵌入序列（必须与训练一致！）
    print("🔄 序列嵌入中...")
    seq2t = s2t('../../../embeddings/benchmarkdefault_onehot.txt')
    X_h = embed_seqs(heavies, seq2t, SEQ_SIZE)
    X_l = embed_seqs(lights, seq2t, SEQ_SIZE)
    X_ag = embed_seqs(antigens, seq2t, SEQ_SIZE)

    # 4. 预测（使用 CPU）
    print("🧠 进行预测（CPU）...")
    pred_ddg = model.predict([X_h, X_l, X_ag], batch_size=32, verbose=1).flatten()

    # 5. 保存结果
    result_df = pd.DataFrame({
        "Index": np.arange(len(true_ddg)),
        "true_ddg": true_ddg,
        "pred_ddg": pred_ddg
    })
    output_csv = os.path.join(OUTPUT_DIR, "benchmarkPIPApredictions.csv")
    result_df.to_csv(output_csv, index=False)
    print(f"✅ 预测结果已保存至: {output_csv}")

    # 6. 绘制回归图
    print("🎨 绘制回归图...")
    plt.figure(figsize=(8, 6))
    plt.scatter(result_df['true_ddg'], result_df['pred_ddg'], alpha=0.6, color='#1f77b4', s=20, edgecolors='none')
    min_val = min(result_df['true_ddg'].min(), result_df['pred_ddg'].min())
    max_val = max(result_df['true_ddg'].max(), result_df['pred_ddg'].max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Ideal Fit (y=x)')
    plt.xlabel('True ΔΔG', fontsize=12)
    plt.ylabel('Predicted ΔΔG', fontsize=12)
    plt.title('Regression: True vs Predicted ΔΔG', fontsize=14)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()

    plot_path = os.path.join(OUTPUT_DIR, "benchmarkPIPAregression_plot.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"📊 回归图已保存至: {plot_path}")

    # 7. 打印简单指标
    from sklearn.metrics import mean_squared_error, r2_score
    from scipy.stats import pearsonr
    rmse = np.sqrt(mean_squared_error(true_ddg, pred_ddg))
    pcc, _ = pearsonr(true_ddg, pred_ddg)
    r2 = r2_score(true_ddg, pred_ddg)
    print(f"\n📈 预测性能概览:")
    print(f"   RMSE: {rmse:.4f}")
    print(f"   PCC : {pcc:.4f}")
    print(f"   R²  : {r2:.4f}")


if __name__ == "__main__":
    main()