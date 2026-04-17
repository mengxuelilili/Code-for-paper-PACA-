# # -*- coding: utf-8 -*-
# """
# 抗体-抗原结合亲和力回归模型（ΔG 预测）
# 兼容 TensorFlow 2.x，支持多数据集联合训练与分组评估
# 按每个数据集 6:2:2 划分 train/val/test，测试集完全隔离
# """

# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# print("⚠️ 已禁用 GPU，使用 CPU 运行（因 CuDNN 版本不兼容）")

# import sys
# import numpy as np
# import pandas as pd
# from tqdm import tqdm
# from sklearn.model_selection import train_test_split

# # === 添加 embeddings 路径 ===
# embed_path = os.path.abspath('../../../embeddings')
# if embed_path not in sys.path:
#     sys.path.append(embed_path)

# try:
#     from embeddings.seq2tensor import s2t  # 注意：直接导入 seq2tensor，不是 embeddings.seq2tensor
# except ImportError:
#     raise ImportError(f"❌ 未找到 seq2tensor 模块。请确保文件存在: {embed_path}/seq2tensor.py")

# # === TensorFlow 设置 ===
# import tensorflow as tf
# from tensorflow.keras.models import Model
# from tensorflow.keras.layers import (
#     Dense, Input, Conv1D, GlobalAveragePooling1D,
#     concatenate, MaxPooling1D, Bidirectional, GRU, LeakyReLU
# )
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.callbacks import EarlyStopping

# # === 配置参数 ===
# SEQ_SIZE = 2000
# HIDDEN_DIM = 64
# N_EPOCHS = 50
# BATCH_SIZE = 32
# RST_FILE = 'results/dg_affinity_regression_tf2_clean_test.txt'

# PRED_FILES = [
#     ("/tmp/AbAgCDR/data/final_dataset_train.tsv", "Paddle"),
#     ("/tmp/AbAgCDR/data/pairs_seq_abbind2.tsv", "AbBind"),
#     ("/tmp/AbAgCDR/data/pairs_seq_sabdab.tsv", "SAbDab"),
#     ("/tmp/AbAgCDR/data/pairs_seq_skempi.tsv", "Skempi"),
#     ("/tmp/AbAgCDR/data/pairs_seq_benchmark1.tsv", "Benchmark"),
# ]

# # === 数据加载：返回带 dataset 标识的 DataFrame ===
# def load_and_merge_datasets(file_list):
#     all_data = []
#     valid_aa = set("ACDEFGHIKLMNPQRSTVWY")
    
#     for file_path, source_name in file_list:
#         print(f"📥 加载 {source_name} 数据: {file_path}")
#         try:
#             df = pd.read_csv(file_path, sep='\t', low_memory=False)
#             cols_lower = [c.lower() for c in df.columns]
#             col_map = {}
            
#             for name in ['antibody_seq_a', 'heavy', 'vh', 'h', 'hc']:
#                 if name in cols_lower:
#                     col_map['heavy'] = df.columns[cols_lower.index(name)]
#                     break
#             for name in ['antibody_seq_b', 'light', 'vl', 'l', 'lc']:
#                 if name in cols_lower:
#                     col_map['light'] = df.columns[cols_lower.index(name)]
#                     break
#             for name in ['antigen_seq', 'antigen', 'ag']:
#                 if name in cols_lower:
#                     col_map['antigen'] = df.columns[cols_lower.index(name)]
#                     break
#             for name in ['delta_g', 'ddg', 'deltag', 'binding_affinity', 'affinity']:
#                 if name in cols_lower:
#                     col_map['delta_g'] = df.columns[cols_lower.index(name)]
#                     break
            
#             required = ['heavy', 'light', 'antigen', 'delta_g']
#             if not all(k in col_map for k in required):
#                 print(f"   ⚠️ 跳过 {source_name}: 缺少必要列")
#                 continue
            
#             df_clean = df[[col_map['heavy'], col_map['light'], col_map['antigen'], col_map['delta_g']]].copy()
#             df_clean.columns = ['heavy', 'light', 'antigen', 'delta_g']
#             df_clean['dataset'] = source_name  # 标记来源
            
#             df_clean.dropna(subset=['heavy', 'light', 'antigen', 'delta_g'], inplace=True)
#             df_clean['delta_g'] = pd.to_numeric(df_clean['delta_g'], errors='coerce')
#             df_clean.dropna(subset=['delta_g'], inplace=True)
            
#             def is_valid_seq(seq):
#                 if not isinstance(seq, str) or len(seq.strip()) == 0:
#                     return False
#                 return all(c in valid_aa for c in seq.upper())
            
#             before = len(df_clean)
#             df_clean = df_clean[
#                 df_clean['heavy'].apply(is_valid_seq) &
#                 df_clean['light'].apply(is_valid_seq) &
#                 df_clean['antigen'].apply(is_valid_seq)
#             ]
#             after = len(df_clean)
#             print(f"   清洗后: {after} / {before} 条有效数据")
            
#             if after > 0:
#                 all_data.append(df_clean)
                
#         except Exception as e:
#             print(f"   ❌ 加载失败 {file_path}: {e}")
    
#     if not all_data:
#         raise RuntimeError("未加载任何有效数据！")
    
#     merged_df = pd.concat(all_data, ignore_index=True)
#     print(f"\n✅ 总共合并 {len(merged_df)} 条样本")
#     return merged_df

# # === 按 dataset 分组划分 train/val/test (6:2:2) ===
# def split_by_dataset(df, test_size=0.2, val_size=0.25):  # val_size = 0.25 of remaining 80% → 20%
#     train_list, val_list, test_list = [], [], []
    
#     for dataset_name in df['dataset'].unique():
#         subset = df[df['dataset'] == dataset_name].copy()
#         print(f"  - 划分 {dataset_name}: {len(subset)} 条")
        
#         # 先分出 test (20%)
#         train_val, test = train_test_split(subset, test_size=test_size, random_state=42)
#         # 再从 train_val 中分出 val (25% of 80% = 20% overall)
#         train, val = train_test_split(train_val, test_size=val_size, random_state=42)
        
#         train_list.append(train)
#         val_list.append(val)
#         test_list.append(test)
    
#     return (
#         pd.concat(train_list, ignore_index=True),
#         pd.concat(val_list, ignore_index=True),
#         pd.concat(test_list, ignore_index=True)
#     )

# # === 构建模型 ===
# def build_dg_model(input_dim, seq_size=2000, hidden_dim=64):
#     input_h = Input(shape=(seq_size, input_dim), name='heavy')
#     input_l = Input(shape=(seq_size, input_dim), name='light')
#     input_ag = Input(shape=(seq_size, input_dim), name='antigen')

#     def extract_features(x):
#         x = MaxPooling1D(3)(Conv1D(hidden_dim, 3)(x))
#         x = concatenate([Bidirectional(GRU(hidden_dim, return_sequences=True))(x), x])
#         x = MaxPooling1D(3)(Conv1D(hidden_dim, 3)(x))
#         x = concatenate([Bidirectional(GRU(hidden_dim, return_sequences=True))(x), x])
#         x = MaxPooling1D(2)(Conv1D(hidden_dim, 3)(x))
#         x = concatenate([Bidirectional(GRU(hidden_dim, return_sequences=True))(x), x])
#         x = MaxPooling1D(2)(Conv1D(hidden_dim, 3)(x))
#         x = concatenate([Bidirectional(GRU(hidden_dim, return_sequences=True))(x), x])
#         x = MaxPooling1D(2)(Conv1D(hidden_dim, 3)(x))
#         x = concatenate([Bidirectional(GRU(hidden_dim, return_sequences=True))(x), x])
#         x = Conv1D(hidden_dim, 3)(x)
#         x = GlobalAveragePooling1D()(x)
#         return x

#     feat_h = extract_features(input_h)
#     feat_l = extract_features(input_l)
#     feat_ag = extract_features(input_ag)

#     antibody_feat = tf.multiply(feat_h, feat_l)
#     complex_feat = tf.multiply(antibody_feat, feat_ag)

#     x = Dense(hidden_dim)(complex_feat)
#     x = LeakyReLU(alpha=0.3)(x)
#     x = Dense(hidden_dim // 2)(x)
#     x = LeakyReLU(alpha=0.3)(x)
#     output = Dense(1, name='delta_g')(x)

#     model = Model(inputs=[input_h, input_l, input_ag], outputs=output)
#     return model

# # === 主程序 ===
# def main():
#     os.makedirs('results', exist_ok=True)
    
#     # 1. 加载并划分数据
#     df = load_and_merge_datasets(PRED_FILES)
#     df_train, df_val, df_test = split_by_dataset(df, test_size=0.2, val_size=0.25)
    
#     print(f"\n📊 划分结果:")
#     print(f"   训练集: {len(df_train)}")
#     print(f"   验证集: {len(df_val)}")
#     print(f"   测试集: {len(df_test)} (完全隔离，仅用于最终评估)")

#     # 2. 提取序列和标签
#     def get_arrays(df_part):
#         heavies = df_part['heavy'].tolist()
#         lights = df_part['light'].tolist()
#         antigens = df_part['antigen'].tolist()
#         delta_g = df_part['delta_g'].values.astype(np.float32)
#         sources = df_part['dataset'].values
#         return heavies, lights, antigens, delta_g, sources

#     # 3. 嵌入函数
#     print("🔄 将序列转为 One-Hot 张量...")
#     seq2t = s2t('../../../embeddings/default_onehot.txt')
#     dim = seq2t.dim

#     def embed_seqs(seqs):
#         return np.array([seq2t.embed_normalized(seq, SEQ_SIZE) for seq in tqdm(seqs, leave=False)])

#     # 4. 处理训练/验证/测试集
#     h_train, l_train, ag_train, y_train, _ = get_arrays(df_train)
#     h_val, l_val, ag_val, y_val, _ = get_arrays(df_val)
#     h_test, l_test, ag_test, y_test, sources_test = get_arrays(df_test)

#     X_h_train = embed_seqs(h_train)
#     X_l_train = embed_seqs(l_train)
#     X_ag_train = embed_seqs(ag_train)

#     X_h_val = embed_seqs(h_val)
#     X_l_val = embed_seqs(l_val)
#     X_ag_val = embed_seqs(ag_val)

#     X_h_test = embed_seqs(h_test)
#     X_l_test = embed_seqs(l_test)
#     X_ag_test = embed_seqs(ag_test)

#     # 5. 构建模型
#     print("🏗️ 构建模型...")
#     model = build_dg_model(input_dim=dim, seq_size=SEQ_SIZE, hidden_dim=HIDDEN_DIM)
#     model.compile(optimizer=Adam(learning_rate=0.001, amsgrad=True), loss='mse', metrics=['mae'])
#     print(model.summary())

#     # 6. 训练（使用 train + val，val 用于早停）
#     print("🏃 开始训练（CPU 模式）...")
#     history = model.fit(
#         [X_h_train, X_l_train, X_ag_train],
#         y_train,
#         batch_size=BATCH_SIZE,
#         epochs=N_EPOCHS,
#         validation_data=([X_h_val, X_l_val, X_ag_val], y_val),
#         callbacks=[EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)],
#         verbose=1
#     )

#     # 7. 最终评估：仅在隔离的测试集上预测
#     print("🔍 在隔离测试集上评估...")
#     preds = model.predict([X_h_test, X_l_test, X_ag_test]).flatten()

#     # 8. 分组评估
#     from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
#     from scipy.stats import pearsonr

#     def compute_metrics(y_true, y_pred):
#         mae = mean_absolute_error(y_true, y_pred)
#         mse = mean_squared_error(y_true, y_pred)
#         rmse = np.sqrt(mse)
#         pcc, _ = pearsonr(y_true, y_pred)
#         r2 = r2_score(y_true, y_pred)
#         return {"MAE": mae, "MSE": mse, "RMSE": rmse, "PCC": pcc, "R2": r2}

#     results = {}
#     unique_sources = np.unique(sources_test)
#     for src in unique_sources:
#         mask = sources_test == src
#         if np.sum(mask) < 2:
#             continue
#         results[src] = compute_metrics(y_test[mask], preds[mask])
    
#     results["Overall"] = compute_metrics(y_test, preds)

#     # 9. 保存结果
#     with open(RST_FILE, 'w') as f:
#         f.write("Dataset\tMAE\tMSE\tRMSE\tPCC\tR2\n")
#         for src in ["Overall"] + sorted([k for k in results.keys() if k != "Overall"]):
#             m = results[src]
#             f.write(f"{src}\t{m['MAE']:.4f}\t{m['MSE']:.4f}\t{m['RMSE']:.4f}\t{m['PCC']:.4f}\t{m['R2']:.4f}\n")
        
#         f.write("\nTrue\tPred\tSource\n")
#         for t, p, s in zip(y_test, preds, sources_test):
#             f.write(f"{t:.4f}\t{p:.4f}\t{s}\n")

#     print(f"\n✅ 训练完成！结果已保存至: {RST_FILE}")
#     for src in ["Overall"] + sorted([k for k in results.keys() if k != "Overall"]):
#         m = results[src]
#         print(f"{src:>10} | MAE: {m['MAE']:.3f}, RMSE: {m['RMSE']:.3f}, PCC: {m['PCC']:.3f}, R²: {m['R2']:.3f}")

# if __name__ == "__main__":
#     main()

# -*- coding: utf-8 -*-
# """
# 抗体-抗原结合亲和力回归模型（ΔG 预测）
# 兼容 TensorFlow 2.x，支持多数据集联合训练与分组评估
# 按每个数据集 6:2:2 划分 train/val/test，测试集完全隔离
# ✅ 已添加模型保存功能
# """

# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# print("⚠️ 已禁用 GPU，使用 CPU 运行（因 CuDNN 版本不兼容）")

# import sys
# import numpy as np
# import pandas as pd
# from tqdm import tqdm
# from sklearn.model_selection import train_test_split

# # === 添加 embeddings 路径 ===
# embed_path = os.path.abspath('../../../embeddings')
# if embed_path not in sys.path:
#     sys.path.append(embed_path)

# try:
#     from embeddings.seq2tensor import s2t  # 注意：直接导入 seq2tensor，不是 embeddings.seq2tensor
# except ImportError:
#     raise ImportError(f"❌ 未找到 seq2tensor 模块。请确保文件存在: {embed_path}/seq2tensor.py")

# # === TensorFlow 设置 ===
# import tensorflow as tf
# from tensorflow.keras.models import Model
# from tensorflow.keras.layers import (
#     Dense, Input, Conv1D, GlobalAveragePooling1D,
#     concatenate, MaxPooling1D, Bidirectional, GRU, LeakyReLU
# )
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.callbacks import EarlyStopping

# # === 配置参数 ===
# SEQ_SIZE = 2000
# HIDDEN_DIM = 64
# N_EPOCHS = 50
# BATCH_SIZE = 32
# RST_FILE = 'results/dg_affinity_regression_tf2_clean_test.txt'
# MODEL_SAVE_PATH = 'results/dg_affinity_regression_model.h5'  # 👈 新增：模型保存路径

# PRED_FILES = [
#     ("/tmp/AbAgCDR/data/final_dataset_train.tsv", "Paddle"),
#     ("/tmp/AbAgCDR/data/pairs_seq_abbind2.tsv", "AbBind"),
#     ("/tmp/AbAgCDR/data/pairs_seq_sabdab.tsv", "SAbDab"),
#     ("/tmp/AbAgCDR/data/pairs_seq_skempi.tsv", "Skempi"),
#     ("/tmp/AbAgCDR/data/pairs_seq_benchmark1.tsv", "Benchmark"),
# ]

# # === 数据加载：返回带 dataset 标识的 DataFrame ===
# def load_and_merge_datasets(file_list):
#     all_data = []
#     valid_aa = set("ACDEFGHIKLMNPQRSTVWY")
    
#     for file_path, source_name in file_list:
#         print(f"📥 加载 {source_name} 数据: {file_path}")
#         try:
#             df = pd.read_csv(file_path, sep='\t', low_memory=False)
#             cols_lower = [c.lower() for c in df.columns]
#             col_map = {}
            
#             for name in ['antibody_seq_a', 'heavy', 'vh', 'h', 'hc']:
#                 if name in cols_lower:
#                     col_map['heavy'] = df.columns[cols_lower.index(name)]
#                     break
#             for name in ['antibody_seq_b', 'light', 'vl', 'l', 'lc']:
#                 if name in cols_lower:
#                     col_map['light'] = df.columns[cols_lower.index(name)]
#                     break
#             for name in ['antigen_seq', 'antigen', 'ag']:
#                 if name in cols_lower:
#                     col_map['antigen'] = df.columns[cols_lower.index(name)]
#                     break
#             for name in ['delta_g', 'ddg', 'deltag', 'binding_affinity', 'affinity']:
#                 if name in cols_lower:
#                     col_map['delta_g'] = df.columns[cols_lower.index(name)]
#                     break
            
#             required = ['heavy', 'light', 'antigen', 'delta_g']
#             if not all(k in col_map for k in required):
#                 print(f"   ⚠️ 跳过 {source_name}: 缺少必要列")
#                 continue
            
#             df_clean = df[[col_map['heavy'], col_map['light'], col_map['antigen'], col_map['delta_g']]].copy()
#             df_clean.columns = ['heavy', 'light', 'antigen', 'delta_g']
#             df_clean['dataset'] = source_name  # 标记来源
            
#             df_clean.dropna(subset=['heavy', 'light', 'antigen', 'delta_g'], inplace=True)
#             df_clean['delta_g'] = pd.to_numeric(df_clean['delta_g'], errors='coerce')
#             df_clean.dropna(subset=['delta_g'], inplace=True)
            
#             def is_valid_seq(seq):
#                 if not isinstance(seq, str) or len(seq.strip()) == 0:
#                     return False
#                 return all(c in valid_aa for c in seq.upper())
            
#             before = len(df_clean)
#             df_clean = df_clean[
#                 df_clean['heavy'].apply(is_valid_seq) &
#                 df_clean['light'].apply(is_valid_seq) &
#                 df_clean['antigen'].apply(is_valid_seq)
#             ]
#             after = len(df_clean)
#             print(f"   清洗后: {after} / {before} 条有效数据")
            
#             if after > 0:
#                 all_data.append(df_clean)
                
#         except Exception as e:
#             print(f"   ❌ 加载失败 {file_path}: {e}")
    
#     if not all_data:
#         raise RuntimeError("未加载任何有效数据！")
    
#     merged_df = pd.concat(all_data, ignore_index=True)
#     print(f"\n✅ 总共合并 {len(merged_df)} 条样本")
#     return merged_df

# # === 按 dataset 分组划分 train/val/test (6:2:2) ===
# def split_by_dataset(df, test_size=0.2, val_size=0.25):  # val_size = 0.25 of remaining 80% → 20%
#     train_list, val_list, test_list = [], [], []
    
#     for dataset_name in df['dataset'].unique():
#         subset = df[df['dataset'] == dataset_name].copy()
#         print(f"  - 划分 {dataset_name}: {len(subset)} 条")
        
#         # 先分出 test (20%)
#         train_val, test = train_test_split(subset, test_size=test_size, random_state=42)
#         # 再从 train_val 中分出 val (25% of 80% = 20% overall)
#         train, val = train_test_split(train_val, test_size=val_size, random_state=42)
        
#         train_list.append(train)
#         val_list.append(val)
#         test_list.append(test)
    
#     return (
#         pd.concat(train_list, ignore_index=True),
#         pd.concat(val_list, ignore_index=True),
#         pd.concat(test_list, ignore_index=True)
#     )

# # === 构建模型 ===
# def build_dg_model(input_dim, seq_size=2000, hidden_dim=64):
#     input_h = Input(shape=(seq_size, input_dim), name='heavy')
#     input_l = Input(shape=(seq_size, input_dim), name='light')
#     input_ag = Input(shape=(seq_size, input_dim), name='antigen')

#     def extract_features(x):
#         x = MaxPooling1D(3)(Conv1D(hidden_dim, 3)(x))
#         x = concatenate([Bidirectional(GRU(hidden_dim, return_sequences=True))(x), x])
#         x = MaxPooling1D(3)(Conv1D(hidden_dim, 3)(x))
#         x = concatenate([Bidirectional(GRU(hidden_dim, return_sequences=True))(x), x])
#         x = MaxPooling1D(2)(Conv1D(hidden_dim, 3)(x))
#         x = concatenate([Bidirectional(GRU(hidden_dim, return_sequences=True))(x), x])
#         x = MaxPooling1D(2)(Conv1D(hidden_dim, 3)(x))
#         x = concatenate([Bidirectional(GRU(hidden_dim, return_sequences=True))(x), x])
#         x = MaxPooling1D(2)(Conv1D(hidden_dim, 3)(x))
#         x = concatenate([Bidirectional(GRU(hidden_dim, return_sequences=True))(x), x])
#         x = Conv1D(hidden_dim, 3)(x)
#         x = GlobalAveragePooling1D()(x)
#         return x

#     feat_h = extract_features(input_h)
#     feat_l = extract_features(input_l)
#     feat_ag = extract_features(input_ag)

#     antibody_feat = tf.multiply(feat_h, feat_l)
#     complex_feat = tf.multiply(antibody_feat, feat_ag)

#     x = Dense(hidden_dim)(complex_feat)
#     x = LeakyReLU(alpha=0.3)(x)
#     x = Dense(hidden_dim // 2)(x)
#     x = LeakyReLU(alpha=0.3)(x)
#     output = Dense(1, name='delta_g')(x)

#     model = Model(inputs=[input_h, input_l, input_ag], outputs=output)
#     return model

# # === 主程序 ===
# def main():
#     os.makedirs('results', exist_ok=True)  # 确保 results 目录存在
    
#     # 1. 加载并划分数据
#     df = load_and_merge_datasets(PRED_FILES)
#     df_train, df_val, df_test = split_by_dataset(df, test_size=0.2, val_size=0.25)
    
#     print(f"\n📊 划分结果:")
#     print(f"   训练集: {len(df_train)}")
#     print(f"   验证集: {len(df_val)}")
#     print(f"   测试集: {len(df_test)} (完全隔离，仅用于最终评估)")

#     # 2. 提取序列和标签
#     def get_arrays(df_part):
#         heavies = df_part['heavy'].tolist()
#         lights = df_part['light'].tolist()
#         antigens = df_part['antigen'].tolist()
#         delta_g = df_part['delta_g'].values.astype(np.float32)
#         sources = df_part['dataset'].values
#         return heavies, lights, antigens, delta_g, sources

#     # 3. 嵌入函数
#     print("🔄 将序列转为 One-Hot 张量...")
#     seq2t = s2t('../../../embeddings/default_onehot.txt')
#     dim = seq2t.dim

#     def embed_seqs(seqs):
#         return np.array([seq2t.embed_normalized(seq, SEQ_SIZE) for seq in tqdm(seqs, leave=False)])

#     # 4. 处理训练/验证/测试集
#     h_train, l_train, ag_train, y_train, _ = get_arrays(df_train)
#     h_val, l_val, ag_val, y_val, _ = get_arrays(df_val)
#     h_test, l_test, ag_test, y_test, sources_test = get_arrays(df_test)

#     X_h_train = embed_seqs(h_train)
#     X_l_train = embed_seqs(l_train)
#     X_ag_train = embed_seqs(ag_train)

#     X_h_val = embed_seqs(h_val)
#     X_l_val = embed_seqs(l_val)
#     X_ag_val = embed_seqs(ag_val)

#     X_h_test = embed_seqs(h_test)
#     X_l_test = embed_seqs(l_test)
#     X_ag_test = embed_seqs(ag_test)

#     # 5. 构建模型
#     print("🏗️ 构建模型...")
#     model = build_dg_model(input_dim=dim, seq_size=SEQ_SIZE, hidden_dim=HIDDEN_DIM)
#     model.compile(optimizer=Adam(learning_rate=0.001, amsgrad=True), loss='mse', metrics=['mae'])
#     print(model.summary())

#     # 6. 训练（使用 train + val，val 用于早停）
#     print("🏃 开始训练（CPU 模式）...")
#     history = model.fit(
#         [X_h_train, X_l_train, X_ag_train],
#         y_train,
#         batch_size=BATCH_SIZE,
#         epochs=N_EPOCHS,
#         validation_data=([X_h_val, X_l_val, X_ag_val], y_val),
#         callbacks=[EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)],
#         verbose=1
#     )

#     # 7. 最终评估：仅在隔离的测试集上预测
#     print("🔍 在隔离测试集上评估...")
#     preds = model.predict([X_h_test, X_l_test, X_ag_test]).flatten()

#     # 8. 分组评估
#     from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
#     from scipy.stats import pearsonr

#     def compute_metrics(y_true, y_pred):
#         mae = mean_absolute_error(y_true, y_pred)
#         mse = mean_squared_error(y_true, y_pred)
#         rmse = np.sqrt(mse)
#         pcc, _ = pearsonr(y_true, y_pred)
#         r2 = r2_score(y_true, y_pred)
#         return {"MAE": mae, "MSE": mse, "RMSE": rmse, "PCC": pcc, "R2": r2}

#     results = {}
#     unique_sources = np.unique(sources_test)
#     for src in unique_sources:
#         mask = sources_test == src
#         if np.sum(mask) < 2:
#             continue
#         results[src] = compute_metrics(y_test[mask], preds[mask])
    
#     results["Overall"] = compute_metrics(y_test, preds)

#     # 9. 保存评估结果
#     with open(RST_FILE, 'w') as f:
#         f.write("Dataset\tMAE\tMSE\tRMSE\tPCC\tR2\n")
#         for src in ["Overall"] + sorted([k for k in results.keys() if k != "Overall"]):
#             m = results[src]
#             f.write(f"{src}\t{m['MAE']:.4f}\t{m['MSE']:.4f}\t{m['RMSE']:.4f}\t{m['PCC']:.4f}\t{m['R2']:.4f}\n")
        
#         f.write("\nTrue\tPred\tSource\n")
#         for t, p, s in zip(y_test, preds, sources_test):
#             f.write(f"{t:.4f}\t{p:.4f}\t{s}\n")

#     print(f"\n✅ 训练完成！结果已保存至: {RST_FILE}")

#     # 10. ✅ 保存训练好的模型（关键新增！）
#     print(f"💾 正在保存模型到: {MODEL_SAVE_PATH}")
#     model.save(MODEL_SAVE_PATH)
#     print(f"✅ 模型已成功保存！")

#     # 11. 打印评估指标
#     for src in ["Overall"] + sorted([k for k in results.keys() if k != "Overall"]):
#         m = results[src]
#         print(f"{src:>10} | MAE: {m['MAE']:.3f}, RMSE: {m['RMSE']:.3f}, PCC: {m['PCC']:.3f}, R²: {m['R2']:.3f}")

# if __name__ == "__main__":
#     main()

# # """
# # 抗体-抗原结合亲和力回归模型（ΔG 预测）
# # 兼容 TensorFlow 2.x，支持多数据集联合训练与分组评估
# # 按每个数据集 6:2:2 划分 train/val/test，测试集完全隔离
# # ✅ 已添加标签归一化功能
# # ✅ 已添加模型保存功能
# # ✅ 已添加标准化器保存功能
# # """

# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# print("⚠️ 已禁用 GPU，使用 CPU 运行")

# import sys
# import numpy as np
# import pandas as pd
# from tqdm import tqdm
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# import pickle

# # === 添加 embeddings 路径 ===
# embed_path = os.path.abspath('../../../embeddings')
# if embed_path not in sys.path:
#     sys.path.append(embed_path)

# try:
#     from embeddings.seq2tensor import s2t
# except ImportError:
#     raise ImportError(f"❌ 未找到 seq2tensor 模块。请确保文件存在：{embed_path}/seq2tensor.py")

# # === TensorFlow 设置 ===
# import tensorflow as tf
# from tensorflow.keras.models import Model
# from tensorflow.keras.layers import (
#     Dense, Input, Conv1D, GlobalAveragePooling1D,
#     concatenate, MaxPooling1D, Bidirectional, GRU, LeakyReLU, Dropout
# )
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.callbacks import EarlyStopping

# # === 配置参数 ===
# SEQ_SIZE = 2000
# HIDDEN_DIM = 40        # 👈 适中维度
# N_EPOCHS = 32          # 👈 适中轮数
# BATCH_SIZE = 40        # 👈 适中批次
# LEARNING_RATE = 0.0015 # 👈 适中学习率
# DROPOUT_RATE = 0.20    # 👈 适中 Dropout
# RST_FILE = 'results/dg_affinity_regression_tf2.txt'
# MODEL_SAVE_PATH = 'results/dg_affinity_regression_model.h5'
# SCALER_SAVE_PATH = 'results/dg_affinity_scaler.pkl'

# PRED_FILES = [
#     ("/tmp/AbAgCDR/data/final_dataset_train.tsv", "Paddle"),
#     ("/tmp/AbAgCDR/data/pairs_seq_abbind2_clean.tsv", "AbBind"),
#     ("/tmp/AbAgCDR/data/pairs_seq_sabdab_clean.tsv", "SAbDab"),
#     ("/tmp/AbAgCDR/data/pairs_seq_skempi_clean.tsv", "Skempi")
# ]

# # === 数据加载 ===
# def load_and_merge_datasets(file_list):
#     all_data = []
#     valid_aa = set("ACDEFGHIKLMNPQRSTVWY")
    
#     for file_path, source_name in file_list:
#         print(f"📥 加载 {source_name} 数据：{file_path}")
#         try:
#             df = pd.read_csv(file_path, sep='\t', low_memory=False)
#             cols_lower = [c.lower() for c in df.columns]
#             col_map = {}
            
#             for name in ['antibody_seq_a', 'heavy', 'vh', 'h', 'hc']:
#                 if name in cols_lower:
#                     col_map['heavy'] = df.columns[cols_lower.index(name)]
#                     break
#             for name in ['antibody_seq_b', 'light', 'vl', 'l', 'lc']:
#                 if name in cols_lower:
#                     col_map['light'] = df.columns[cols_lower.index(name)]
#                     break
#             for name in ['antigen_seq', 'antigen', 'ag']:
#                 if name in cols_lower:
#                     col_map['antigen'] = df.columns[cols_lower.index(name)]
#                     break
#             for name in ['delta_g', 'ddg', 'deltag', 'binding_affinity', 'affinity']:
#                 if name in cols_lower:
#                     col_map['delta_g'] = df.columns[cols_lower.index(name)]
#                     break
            
#             required = ['heavy', 'light', 'antigen', 'delta_g']
#             if not all(k in col_map for k in required):
#                 print(f"   ⚠️ 跳过 {source_name}: 缺少必要列")
#                 continue
            
#             df_clean = df[[col_map['heavy'], col_map['light'], col_map['antigen'], col_map['delta_g']]].copy()
#             df_clean.columns = ['heavy', 'light', 'antigen', 'delta_g']
#             df_clean['dataset'] = source_name
            
#             df_clean.dropna(subset=['heavy', 'light', 'antigen', 'delta_g'], inplace=True)
#             df_clean['delta_g'] = pd.to_numeric(df_clean['delta_g'], errors='coerce')
#             df_clean.dropna(subset=['delta_g'], inplace=True)
            
#             def is_valid_seq(seq):
#                 if not isinstance(seq, str) or len(seq.strip()) == 0:
#                     return False
#                 return all(c in valid_aa for c in seq.upper())
            
#             before = len(df_clean)
#             df_clean = df_clean[
#                 df_clean['heavy'].apply(is_valid_seq) &
#                 df_clean['light'].apply(is_valid_seq) &
#                 df_clean['antigen'].apply(is_valid_seq)
#             ]
#             after = len(df_clean)
#             print(f"   清洗后：{after} / {before} 条有效数据")
            
#             if after > 0:
#                 all_data.append(df_clean)
                
#         except Exception as e:
#             print(f"   ❌ 加载失败 {file_path}: {e}")
    
#     if not all_data:
#         raise RuntimeError("未加载任何有效数据！")
    
#     merged_df = pd.concat(all_data, ignore_index=True)
#     print(f"\n✅ 总共合并 {len(merged_df)} 条样本")
#     return merged_df

# # === 按 dataset 分组划分 train/val/test (6:2:2) ===
# def split_by_dataset(df, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2):
#     train_list, val_list, test_list = [], [], []
    
#     print(f"\n📊 数据划分配置：train={train_ratio}, val={val_ratio}, test={test_ratio}")
    
#     for dataset_name in df['dataset'].unique():
#         subset = df[df['dataset'] == dataset_name].copy()
#         n_total = len(subset)
        
#         train_val, test = train_test_split(
#             subset, 
#             test_size=test_ratio, 
#             random_state=42
#         )
        
#         val_ratio_adjusted = val_ratio / (train_ratio + val_ratio)
#         train, val = train_test_split(
#             train_val, 
#             test_size=val_ratio_adjusted, 
#             random_state=42
#         )
        
#         train_list.append(train)
#         val_list.append(val)
#         test_list.append(test)
        
#         n_train, n_val, n_test = len(train), len(val), len(test)
#         print(f"  - {dataset_name}: 总={n_total}, 训练={n_train}({n_train/n_total:.2f}), "
#               f"验证={n_val}({n_val/n_total:.2f}), 测试={n_test}({n_test/n_total:.2f})")
    
#     return (
#         pd.concat(train_list, ignore_index=True),
#         pd.concat(val_list, ignore_index=True),
#         pd.concat(test_list, ignore_index=True)
#     )

# # === 构建模型 ===
# def build_dg_model(input_dim, seq_size=2000, hidden_dim=40):
#     input_h = Input(shape=(seq_size, input_dim), name='heavy')
#     input_l = Input(shape=(seq_size, input_dim), name='light')
#     input_ag = Input(shape=(seq_size, input_dim), name='antigen')

#     def extract_features(x):
#         x = Conv1D(hidden_dim, 3, padding='same')(x)
#         x = LeakyReLU(alpha=0.2)(x)
#         x = MaxPooling1D(2)(x)
#         x = Bidirectional(GRU(hidden_dim, return_sequences=False))(x)
#         return x

#     feat_h = extract_features(input_h)
#     feat_l = extract_features(input_l)
#     feat_ag = extract_features(input_ag)

#     x = concatenate([feat_h, feat_l, feat_ag])
#     x = Dense(hidden_dim)(x)
#     x = LeakyReLU(alpha=0.2)(x)
#     x = Dropout(DROPOUT_RATE)(x)
#     x = Dense(hidden_dim // 2)(x)
#     x = LeakyReLU(alpha=0.2)(x)
#     x = Dense(1)(x)

#     model = Model(inputs=[input_h, input_l, input_ag], outputs=x)
#     return model

# # === 主程序 ===
# def main():
#     os.makedirs('results', exist_ok=True)
    
#     # 1. 加载并划分数据
#     df = load_and_merge_datasets(PRED_FILES)
#     df_train, df_val, df_test = split_by_dataset(df, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2)
    
#     print(f"\n📊 划分结果:")
#     print(f"   训练集：{len(df_train)}")
#     print(f"   验证集：{len(df_val)}")
#     print(f"   测试集：{len(df_test)}")

#     # 2. 提取序列和标签
#     def get_arrays(df_part):
#         heavies = df_part['heavy'].tolist()
#         lights = df_part['light'].tolist()
#         antigens = df_part['antigen'].tolist()
#         delta_g = df_part['delta_g'].values.astype(np.float32)
#         sources = df_part['dataset'].values
#         return heavies, lights, antigens, delta_g, sources

#     # 3. 嵌入函数
#     print("\n🔄 将序列转为 One-Hot 张量...")
#     seq2t = s2t('../../../embeddings/default_onehot.txt')
#     dim = seq2t.dim

#     def embed_seqs(seqs):
#         return np.array([seq2t.embed_normalized(seq, SEQ_SIZE) for seq in tqdm(seqs, leave=False)])

#     # 4. 处理训练/验证/测试集
#     h_train, l_train, ag_train, y_train, _ = get_arrays(df_train)
#     h_val, l_val, ag_val, y_val, _ = get_arrays(df_val)
#     h_test, l_test, ag_test, y_test, sources_test = get_arrays(df_test)

#     # 5. 标签标准化
#     print("\n📈 对 ΔG 标签进行标准化...")
#     scaler = StandardScaler()
#     y_train_scaled = scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
#     y_val_scaled = scaler.transform(y_val.reshape(-1, 1)).flatten()
    
#     print(f"   原始 ΔG: mean={y_train.mean():.4f}, std={y_train.std():.4f}")
#     print(f"   标准化后：mean={y_train_scaled.mean():.4f}, std={y_train_scaled.std():.4f}")
    
#     with open(SCALER_SAVE_PATH, 'wb') as f:
#         pickle.dump(scaler, f)
#     print(f"   ✅ 标准化器已保存至：{SCALER_SAVE_PATH}")

#     # 6. 序列嵌入
#     X_h_train = embed_seqs(h_train)
#     X_l_train = embed_seqs(l_train)
#     X_ag_train = embed_seqs(ag_train)

#     X_h_val = embed_seqs(h_val)
#     X_l_val = embed_seqs(l_val)
#     X_ag_val = embed_seqs(ag_val)

#     X_h_test = embed_seqs(h_test)
#     X_l_test = embed_seqs(l_test)
#     X_ag_test = embed_seqs(ag_test)

#     # 7. 构建模型
#     print("\n🏗️ 构建模型...")
#     model = build_dg_model(input_dim=dim, seq_size=SEQ_SIZE, hidden_dim=HIDDEN_DIM)
#     model.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss='mse', metrics=['mae'])
#     model.summary()

#     # 8. 训练
#     print("\n🏃 开始训练...")
#     history = model.fit(
#         [X_h_train, X_l_train, X_ag_train],
#         y_train_scaled,
#         batch_size=BATCH_SIZE,
#         epochs=N_EPOCHS,
#         validation_data=([X_h_val, X_l_val, X_ag_val], y_val_scaled),
#         callbacks=[EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True)],
#         verbose=1
#     )

#     # 9. 测试集评估
#     print("\n🔍 在测试集上评估...")
#     preds_scaled = model.predict([X_h_test, X_l_test, X_ag_test]).flatten()
    
#     # 检查预测值是否有变化
#     print(f"   预测值范围：[{preds_scaled.min():.4f}, {preds_scaled.max():.4f}]")
#     print(f"   预测值标准差：{preds_scaled.std():.4f}")
    
#     preds = scaler.inverse_transform(preds_scaled.reshape(-1, 1)).flatten()

#     # 10. 计算指标
#     from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
#     from scipy.stats import pearsonr

#     def compute_metrics(y_true, y_pred):
#         if np.std(y_pred) < 1e-6 or np.std(y_true) < 1e-6:
#             pcc = 0.0
#         else:
#             pcc, _ = pearsonr(y_true, y_pred)
#         mae = mean_absolute_error(y_true, y_pred)
#         mse = mean_squared_error(y_true, y_pred)
#         rmse = np.sqrt(mse)
#         r2 = r2_score(y_true, y_pred)
#         return {"MAE": mae, "MSE": mse, "RMSE": rmse, "PCC": pcc, "R2": r2}

#     results = {}
#     unique_sources = np.unique(sources_test)
#     for src in unique_sources:
#         mask = sources_test == src
#         if np.sum(mask) < 2:
#             continue
#         results[src] = compute_metrics(y_test[mask], preds[mask])
    
#     results["Overall"] = compute_metrics(y_test, preds)

#     # 11. 保存结果
#     with open(RST_FILE, 'w') as f:
#         f.write("Dataset\tMAE\tMSE\tRMSE\tPCC\tR2\n")
#         for src in ["Overall"] + sorted([k for k in results.keys() if k != "Overall"]):
#             m = results[src]
#             f.write(f"{src}\t{m['MAE']:.4f}\t{m['MSE']:.4f}\t{m['RMSE']:.4f}\t{m['PCC']:.4f}\t{m['R2']:.4f}\n")
        
#         f.write("\nTrue\tPred\tSource\n")
#         for t, p, s in zip(y_test, preds, sources_test):
#             f.write(f"{t:.4f}\t{p:.4f}\t{s}\n")

#     print(f"\n✅ 训练完成！结果已保存至：{RST_FILE}")

#     # 12. 保存模型
#     model.save(MODEL_SAVE_PATH)
#     print(f"💾 模型已保存至：{MODEL_SAVE_PATH}")

#     # 13. 打印指标
#     print("\n" + "="*60)
#     print("📊 评估结果")
#     print("="*60)
#     for src in ["Overall"] + sorted([k for k in results.keys() if k != "Overall"]):
#         m = results[src]
#         print(f"{src:>12} | MAE: {m['MAE']:.3f}, RMSE: {m['RMSE']:.3f}, PCC: {m['PCC']:.3f}, R²: {m['R2']:.3f}")
#     print("="*60)

# if __name__ == "__main__":
#     main()

import os
import sys
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr

# === 1. 环境配置 ===
# 强制使用 CPU，避免 GPU 非确定性导致的结果波动
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # 屏蔽 TensorFlow 警告

print("⚠️ 已禁用 GPU，使用 CPU 运行以确保结果可复现")

# 添加 embeddings 路径
embed_path = os.path.abspath('../../../embeddings')
if embed_path not in sys.path:
    sys.path.append(embed_path)

try:
    from embeddings.seq2tensor import s2t
except ImportError:
    raise ImportError(f"❌ 未找到 seq2tensor 模块。请确保文件存在：{embed_path}/seq2tensor.py")

import tensorflow as tf
# 限制 TF 内存增长，防止占满所有显存（虽然用的是 CPU，但以防万一加载了 GPU 插件）
tf.config.set_soft_device_placement(True)

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Dense, Input, Conv1D, GlobalAveragePooling1D,
    concatenate, MaxPooling1D, Bidirectional, GRU, LeakyReLU, Dropout
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# === 2. 全局配置 ===
SEQ_SIZE = 2000       # 序列固定长度
HIDDEN_DIM = 64       # 稍微增加维度以匹配任务复杂度
N_EPOCHS = 50         # 增加轮数以便收敛
BATCH_SIZE = 32       
LEARNING_RATE = 0.001
DROPOUT_RATE = 0.3    # 增加 Dropout 防止过拟合
RST_FILE = 'results/dg_affinity_regression_tf2_strict.txt'
MODEL_SAVE_PATH = 'results/dg_affinity_regression_model_strict.h5'
SCALER_SAVE_PATH = 'results/dg_affinity_scaler_strict.pkl'

PRED_FILES = [
    ("/tmp/AbAgCDR/data/final_dataset_train.tsv", "Paddle"),
    ("/tmp/AbAgCDR/data/pairs_seq_abbind2_clean.tsv", "AbBind"),
    ("/tmp/AbAgCDR/data/pairs_seq_sabdab_clean.tsv", "SAbDab"),
    ("/tmp/AbAgCDR/data/pairs_seq_skempi_clean.tsv", "Skempi")
]

# === 3. 数据加载与清洗 ===
def load_and_merge_datasets(file_list):
    all_data = []
    valid_aa = set("ACDEFGHIKLMNPQRSTVWY")
    
    for file_path, source_name in file_list:
        if not os.path.exists(file_path):
            print(f"   ⚠️ 文件不存在，跳过：{file_path}")
            continue
            
        print(f"📥 加载 {source_name} ...")
        try:
            df = pd.read_csv(file_path, sep='\t', low_memory=False)
            cols_lower = [str(c).lower() for c in df.columns]
            col_map = {}
            
            # 智能列名映射
            heavy_keys = ['antibody_seq_a', 'heavy', 'vh', 'h', 'hc', 'heavy_chain']
            light_keys = ['antibody_seq_b', 'light', 'vl', 'l', 'lc', 'light_chain']
            antigen_keys = ['antigen_seq', 'antigen', 'ag', 'target_seq']
            label_keys = ['delta_g', 'ddg', 'deltag', 'binding_affinity', 'affinity', 'dg']
            
            def find_col(keys, cols_lower, original_cols):
                for k in keys:
                    if k in cols_lower:
                        return original_cols[cols_lower.index(k)]
                return None

            col_map['heavy'] = find_col(heavy_keys, cols_lower, list(df.columns))
            col_map['light'] = find_col(light_keys, cols_lower, list(df.columns))
            col_map['antigen'] = find_col(antigen_keys, cols_lower, list(df.columns))
            col_map['delta_g'] = find_col(label_keys, cols_lower, list(df.columns))
            
            required = ['heavy', 'light', 'antigen', 'delta_g']
            missing = [k for k in required if col_map[k] is None]
            if missing:
                print(f"   ⚠️ 跳过 {source_name}: 缺少列 {missing}")
                continue
            
            df_clean = df[[col_map['heavy'], col_map['light'], col_map['antigen'], col_map['delta_g']]].copy()
            df_clean.columns = ['heavy', 'light', 'antigen', 'delta_g']
            df_clean['dataset'] = source_name
            
            # 类型转换与清洗
            df_clean['delta_g'] = pd.to_numeric(df_clean['delta_g'], errors='coerce')
            df_clean.dropna(subset=['heavy', 'light', 'antigen', 'delta_g'], inplace=True)
            
            # 序列有效性检查
            def is_valid_seq(seq):
                if not isinstance(seq, str) or len(seq.strip()) == 0:
                    return False
                return all(c.upper() in valid_aa for c in seq)
            
            before = len(df_clean)
            df_clean = df_clean[
                df_clean['heavy'].apply(is_valid_seq) &
                df_clean['light'].apply(is_valid_seq) &
                df_clean['antigen'].apply(is_valid_seq)
            ]
            after = len(df_clean)
            if after > 0:
                print(f"   ✅ {source_name}: {after}/{before} 有效样本")
                all_data.append(df_clean)
            else:
                print(f"   ❌ {source_name}: 无有效样本")
                
        except Exception as e:
            print(f"   ❌ 加载失败 {file_path}: {e}")
    
    if not all_data:
        raise RuntimeError("❌ 未加载到任何有效数据！程序终止。")
    
    merged_df = pd.concat(all_data, ignore_index=True)
    print(f"\n📊 合并后总样本数：{len(merged_df)}")
    return merged_df

# === 4. 严格的数据划分 (防泄露核心) ===
def split_by_dataset_strict(df, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, random_state=42):
    """
    使用 GroupShuffleSplit 确保相同的序列组合不会跨越训练/验证/测试集。
    Group ID 定义为：Heavy + Light + Antigen 的组合。
    这能防止同一复合物的不同突变体被拆分到不同集合。
    """
    print(f"\n🛡️ 执行严格数据划分 (基于序列组合隔离)...")
    
    # 创建 Group ID
    # 注意：将序列转为字符串并拼接，作为分组的依据
    df['group_id'] = (
        df['heavy'].astype(str) + "|" + 
        df['light'].astype(str) + "|" + 
        df['antigen'].astype(str)
    )
    
    train_list, val_list, test_list = [], [], []
    
    for dataset_name in df['dataset'].unique():
        subset = df[df['dataset'] == dataset_name].reset_index(drop=True)
        n_total = len(subset)
        
        if n_total < 10:
            print(f"   ⚠️ {dataset_name} 样本太少 ({n_total})，跳过严格划分，直接归入训练集")
            train_list.append(subset)
            continue
            
        groups = subset['group_id'].values
        indices = np.arange(n_total)
        
        # 第一步：分离测试集 (Test vs Train+Val)
        gss_test = GroupShuffleSplit(
            test_size=test_ratio, 
            n_splits=1, 
            random_state=random_state
        )
        train_val_idx, test_idx = next(gss_test.split(indices, groups=groups))
        
        train_val_subset = subset.iloc[train_val_idx].reset_index(drop=True)
        test_subset = subset.iloc[test_idx].reset_index(drop=True)
        
        # 第二步：分离验证集 (Val vs Train)
        if len(train_val_subset) < 10:
            # 如果剩余太少，全部作为训练集
            train_subset = train_val_subset
            val_subset = pd.DataFrame(columns=train_subset.columns)
        else:
            # 重新计算 Val 在 Train+Val 中的比例
            # 目标：Val / (Train + Val) = val_ratio / (train_ratio + val_ratio)
            adjusted_val_size = val_ratio / (train_ratio + val_ratio)
            
            tv_groups = train_val_subset['group_id'].values
            tv_indices = np.arange(len(train_val_subset))
            
            gss_val = GroupShuffleSplit(
                test_size=adjusted_val_size, 
                n_splits=1, 
                random_state=random_state + 1 # 不同的种子
            )
            train_idx, val_idx = next(gss_val.split(tv_indices, groups=tv_groups))
            
            train_subset = train_val_subset.iloc[train_idx]
            val_subset = train_val_subset.iloc[val_idx]
        
        train_list.append(train_subset)
        val_list.append(val_subset)
        test_list.append(test_subset)
        
        # 打印统计
        n_tr, n_va, n_te = len(train_subset), len(val_subset), len(test_subset)
        print(f"   - {dataset_name}: Total={n_total} | Train={n_tr}, Val={n_va}, Test={n_te}")
        
        # 🔍 泄露检查
        train_groups = set(train_subset['group_id'])
        test_groups = set(test_subset['group_id'])
        val_groups = set(val_subset['group_id']) if len(val_subset) > 0 else set()
        
        overlap_tr_te = train_groups.intersection(test_groups)
        overlap_tr_va = train_groups.intersection(val_groups)
        overlap_va_te = val_groups.intersection(test_groups)
        
        if overlap_tr_te or overlap_tr_va or overlap_va_te:
            print(f"   ❌ 严重错误：发现组间泄露！")
            if overlap_tr_te: print(f"      Train-Test 重叠：{len(overlap_tr_te)} 组")
            if overlap_tr_va: print(f"      Train-Val 重叠：{len(overlap_tr_va)} 组")
            if overlap_va_te: print(f"      Val-Test 重叠：{len(overlap_va_te)} 组")
            raise RuntimeError("数据划分逻辑错误，存在泄露。")
        else:
            print(f"   ✅ 验证通过：{dataset_name} 无组间泄露。")

    final_train = pd.concat(train_list, ignore_index=True).drop(columns=['group_id'])
    final_val = pd.concat(val_list, ignore_index=True).drop(columns=['group_id'])
    final_test = pd.concat(test_list, ignore_index=True).drop(columns=['group_id'])
    
    return final_train, final_val, final_test

# === 5. 模型构建 ===
def build_dg_model(input_dim, seq_size=SEQ_SIZE, hidden_dim=HIDDEN_DIM):
    input_h = Input(shape=(seq_size, input_dim), name='heavy')
    input_l = Input(shape=(seq_size, input_dim), name='light')
    input_ag = Input(shape=(seq_size, input_dim), name='antigen')

    def extract_features(x):
        # Conv1D 提取局部模式
        x = Conv1D(hidden_dim, 5, padding='same')(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = MaxPooling1D(2)(x)
        x = Dropout(0.2)(x)
        
        # BiGRU 捕捉长程依赖
        x = Bidirectional(GRU(hidden_dim, return_sequences=False))(x)
        return x

    feat_h = extract_features(input_h)
    feat_l = extract_features(input_l)
    feat_ag = extract_features(input_ag)

    # 融合
    x = concatenate([feat_h, feat_l, feat_ag])
    x = Dense(hidden_dim * 2)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dropout(DROPOUT_RATE)(x)
    
    x = Dense(hidden_dim)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dropout(DROPOUT_RATE)(x)
    
    output = Dense(1)(x) # 线性输出，回归任务

    model = Model(inputs=[input_h, input_l, input_ag], outputs=output)
    return model

# === 6. 主程序 ===
def main():
    os.makedirs('results', exist_ok=True)
    
    # 1. 加载数据
    df = load_and_merge_datasets(PRED_FILES)
    
    # 2. 严格划分
    df_train, df_val, df_test = split_by_dataset_strict(
        df, 
        train_ratio=0.6, 
        val_ratio=0.2, 
        test_ratio=0.2,
        random_state=42
    )
    
    print(f"\n📊 最终数据集大小:")
    print(f"   训练集：{len(df_train)}")
    print(f"   验证集：{len(df_val)}")
    print(f"   测试集：{len(df_test)}")
    
    if len(df_test) == 0:
        print("⚠️ 测试集为空，无法评估。")
        return

    # 3. 准备数据数组
    def get_arrays(df_part):
        return (
            df_part['heavy'].tolist(),
            df_part['light'].tolist(),
            df_part['antigen'].tolist(),
            df_part['delta_g'].values.astype(np.float32),
            df_part['dataset'].values
        )

    h_train, l_train, ag_train, y_train, src_train = get_arrays(df_train)
    h_val, l_val, ag_val, y_val, src_val = get_arrays(df_val)
    h_test, l_test, ag_test, y_test, src_test = get_arrays(df_test)

    # 4. 标签标准化 ( STRICT: 仅 fit 训练集 )
    print("\n📈 标准化标签 (仅基于训练集统计量)...")
    scaler = StandardScaler()
    y_train_scaled = scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
    
    # Transform 验证集和测试集
    y_val_scaled = scaler.transform(y_val.reshape(-1, 1)).flatten()
    # 测试集标签不需要 scaled 用于训练，但为了对比预测值，我们保留原始 y_test
    # 预测值将在最后反归一化
    
    print(f"   训练集 ΔG: Mean={y_train.mean():.4f}, Std={y_train.std():.4f}")
    
    with open(SCALER_SAVE_PATH, 'wb') as f:
        pickle.dump(scaler, f)

    # 5. 序列嵌入
    print("\n🔄 序列编码 (One-Hot)...")
    try:
        seq2t = s2t('../../../embeddings/default_onehot.txt')
    except Exception:
        # 尝试备用路径或默认
        seq2t = s2t(os.path.join(embed_path, 'default_onehot.txt'))
        
    dim = seq2t.dim
    print(f"   特征维度：{dim}, 序列长度：{SEQ_SIZE}")

    def embed_seqs(seqs):
        return np.array([seq2t.embed_normalized(seq, SEQ_SIZE) for seq in tqdm(seqs, desc="Embedding", leave=False)])

    print("   处理训练集...")
    X_h_train = embed_seqs(h_train)
    X_l_train = embed_seqs(l_train)
    X_ag_train = embed_seqs(ag_train)

    print("   处理验证集...")
    X_h_val = embed_seqs(h_val)
    X_l_val = embed_seqs(l_val)
    X_ag_val = embed_seqs(ag_val)

    print("   处理测试集...")
    X_h_test = embed_seqs(h_test)
    X_l_test = embed_seqs(l_test)
    X_ag_test = embed_seqs(ag_test)

    # 6. 构建与训练模型
    print("\n🏗️ 构建模型...")
    model = build_dg_model(input_dim=dim)
    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss='mse', metrics=['mae'])
    
    # model.summary() # 可选：打印结构

    print("\n🏃 开始训练 (Strict Mode)...")
    early_stop = EarlyStopping(
        monitor='val_loss', 
        patience=8, 
        restore_best_weights=True,
        verbose=1
    )
    
    history = model.fit(
        [X_h_train, X_l_train, X_ag_train],
        y_train_scaled,
        validation_data=([X_h_val, X_l_val, X_ag_val], y_val_scaled),
        epochs=N_EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stop],
        verbose=1
    )

    # 7. 评估
    print("\n🔍 在测试集上评估...")
    preds_scaled = model.predict([X_h_test, X_l_test, X_ag_test]).flatten()
    
    # 反归一化预测值 (使用训练集的统计量)
    preds = scaler.inverse_transform(preds_scaled.reshape(-1, 1)).flatten()
    
    # 计算指标
    def compute_metrics(y_true, y_pred):
        if len(y_true) < 2:
            return {"MAE": 0, "MSE": 0, "RMSE": 0, "PCC": 0, "R2": 0}
        
        # 防止常数预测导致 PCC 报错
        if np.std(y_pred) < 1e-7 or np.std(y_true) < 1e-7:
            pcc = 0.0
        else:
            pcc, _ = pearsonr(y_true, y_pred)
            
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        return {"MAE": mae, "MSE": mse, "RMSE": rmse, "PCC": pcc, "R2": r2}

    results = {}
    unique_sources = np.unique(src_test)
    
    print(f"\n{'Dataset':<15} | {'Count':>5} | {'MSE':>6} | {'RMSE':>6} | {'MAE':>6} | {'PCC':>6} | {'R2':>6}")
    print("-" * 65)
    
    for src in unique_sources:
        mask = src_test == src
        count = np.sum(mask)
        if count < 2:
            continue
        
        m = compute_metrics(y_test[mask], preds[mask])
        results[src] = m
        print(f"{src:<15} | {count:>5} | {m['MSE']:>6.4f} | {m['RMSE']:>6.4f} | {m['MAE']:>6.4f} | {m['PCC']:>6.4f} | {m['R2']:>6.4f}")
    
    # 总体指标
    m_overall = compute_metrics(y_test, preds)
    results["Overall"] = m_overall
    print("-" * 65)
    print(f"{'Overall':<15} | {len(y_test):>5} | {m_overall['MAE']:>6.3f} | {m_overall['RMSE']:>6.3f} | {m_overall['PCC']:>6.3f} | {m_overall['R2']:>6.3f}")

    # 8. 保存结果
    with open(RST_FILE, 'w') as f:
        f.write("Dataset\tCount\tMAE\tMSE\tRMSE\tPCC\tR2\n")
        for src in ["Overall"] + sorted([k for k in results.keys() if k != "Overall"]):
            m = results[src]
            count = len(y_test) if src == "Overall" else np.sum(src_test == src)
            f.write(f"{src}\t{count}\t{m['MAE']:.4f}\t{m['MSE']:.4f}\t{m['RMSE']:.4f}\t{m['PCC']:.4f}\t{m['R2']:.4f}\n")
        
        # 保存详细预测对 (可选，文件可能很大)
        # f.write("\nTrue\tPred\tSource\n")
        # for t, p, s in zip(y_test, preds, src_test):
        #     f.write(f"{t:.4f}\t{p:.4f}\t{s}\n")

    # 9. 保存模型
    model.save(MODEL_SAVE_PATH)
    print(f"\n✅ 完成！模型保存至：{MODEL_SAVE_PATH}")
    print(f"✅ 结果保存至：{RST_FILE}")
    print(f"✅ 标准化器保存至：{SCALER_SAVE_PATH}")

if __name__ == "__main__":
    main()

# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# print("⚠️ 已禁用 GPU，使用 CPU 运行")

# import sys
# import numpy as np
# import pandas as pd
# from tqdm import tqdm
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import RobustScaler, StandardScaler
# import pickle

# # === 导入 ESM2 ===
# try:
#     import esm
#     print("✅ ESM2 库加载成功")
# except ImportError:
#     raise ImportError("❌ 未找到 esm 库。请安装：pip install fair-esm")

# # === TensorFlow 设置 ===
# import tensorflow as tf
# from tensorflow.keras.models import Model
# from tensorflow.keras.layers import (
#     Dense, Input, Conv1D, GlobalAveragePooling1D,
#     concatenate, MaxPooling1D, Bidirectional, GRU, LeakyReLU, Dropout,
#     BatchNormalization
# )
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.callbacks import EarlyStopping

# # === 配置参数 ===
# # ESM2 嵌入维度
# ESM2_DIM = 1280  # esm2_t12_35M_UR50D 的输出维度
# HEAVY_LIGHT_DIM = 532  # 抗体轻/重链目标维度
# ANTIGEN_DIM = 500      # 抗原目标维度

# HIDDEN_DIM = 128       # 👈 增加维度以匹配 ESM2 特征
# N_EPOCHS = 50          # 👈 增加轮数
# BATCH_SIZE = 32        # 👈 适中批次
# LEARNING_RATE = 0.001  # 👈 调整学习率
# DROPOUT_RATE = 0.25    # 👈 适中 Dropout
# RST_FILE = 'results/2dg_affinity_regression_tf2_esm2.txt'
# MODEL_SAVE_PATH = 'results/2dg_affinity_regression_model_esm2.h5'
# SCALER_SAVE_PATH = 'results/2dg_affinity_scaler_robust.pkl'
# FEATURE_SCALER_PATH = 'results/2dg_affinity_feature_scaler.pkl'

# PRED_FILES = [
#     ("/tmp/AbAgCDR/data/final_dataset_train.tsv", "Paddle"),
#     ("/tmp/AbAgCDR/data/pairs_seq_abbind2.tsv", "AbBind"),
#     ("/tmp/AbAgCDR/data/pairs_seq_sabdab.tsv", "SAbDab"),
#     ("/tmp/AbAgCDR/data/pairs_seq_skempi.tsv", "Skempi"),
#     ("/tmp/AbAgCDR/data/pairs_seq_benchmark1.tsv", "Benchmark"),
# ]

# # === ESM2 嵌入提取器 ===
# class ESM2Embedder:
#     """ESM2 蛋白质序列嵌入提取器"""
    
#     def __init__(self, model_name="esm2_t12_35M_UR50D"):
#         print(f"🧠 加载 ESM2 模型 ({model_name})...")
#         self.model_name = model_name
#         self.model, self.alphabet = esm.pretrained.esm2_t12_35M_UR50D()
#         self.device = "cpu"  # 当前脚本禁用 GPU
#         self.model = self.model.to(self.device)
#         self.model.eval()
#         self.batch_converter = self.alphabet.get_batch_converter()
#         self.emb_dim = 1280  # ESM2 t12 35M 的输出维度
#         print(f"✅ ESM2 加载完成 (设备：{self.device}, 嵌入维度：{self.emb_dim})")
    
#     def get_embedding(self, seq):
#         """获取单条序列的嵌入 (Mean Pooling, 1280 维)"""
#         seq = str(seq).upper().strip()
#         valid_aas = "ARNDCQEGHILKMFPSTWYV"
#         seq = ''.join([aa for aa in seq if aa in valid_aas])
#         if not seq:
#             seq = "A"
        
#         data = [("protein", seq)]
#         _, _, batch_tokens = self.batch_converter(data)
#         batch_tokens = batch_tokens.to(self.device)
        
#         with torch.no_grad():
#             results = self.model(batch_tokens, repr_layers=[12])
#             token_repr = results["representations"][12]
        
#         # Mean pooling (去掉 BOS 和 EOS)
#         emb = token_repr[0, 1:-1].mean(dim=0).cpu().numpy()
#         return emb
    
#     def embed_batch(self, sequences, target_dim=None):
#         """
#         批量提取嵌入并可选调整到目标维度
        
#         Args:
#             sequences: 序列列表
#             target_dim: 目标维度 (None 则保持 1280 维)
        
#         Returns:
#             embeddings: (N, target_dim) 或 (N, 1280)
#         """
#         embeddings = []
#         for seq in tqdm(sequences, desc="Extracting ESM2 embeddings", leave=False):
#             emb = self.get_embedding(seq)
#             embeddings.append(emb)
        
#         embeddings = np.array(embeddings, dtype=np.float32)  # (N, 1280)
        
#         # 维度调整
#         if target_dim is not None:
#             if embeddings.shape[1] > target_dim:
#                 embeddings = embeddings[:, :target_dim]
#             elif embeddings.shape[1] < target_dim:
#                 pad = np.zeros((embeddings.shape[0], target_dim - embeddings.shape[1]), dtype=np.float32)
#                 embeddings = np.concatenate([embeddings, pad], axis=1)
        
#         return embeddings


# # === 数据加载 ===
# def load_and_merge_datasets(file_list):
#     all_data = []
#     valid_aa = set("ACDEFGHIKLMNPQRSTVWY")
    
#     for file_path, source_name in file_list:
#         print(f"📥 加载 {source_name} 数据：{file_path}")
#         try:
#             df = pd.read_csv(file_path, sep='\t', low_memory=False)
#             cols_lower = [c.lower() for c in df.columns]
#             col_map = {}
            
#             for name in ['antibody_seq_a', 'heavy', 'vh', 'h', 'hc']:
#                 if name in cols_lower:
#                     col_map['heavy'] = df.columns[cols_lower.index(name)]
#                     break
#             for name in ['antibody_seq_b', 'light', 'vl', 'l', 'lc']:
#                 if name in cols_lower:
#                     col_map['light'] = df.columns[cols_lower.index(name)]
#                     break
#             for name in ['antigen_seq', 'antigen', 'ag']:
#                 if name in cols_lower:
#                     col_map['antigen'] = df.columns[cols_lower.index(name)]
#                     break
#             for name in ['delta_g', 'ddg', 'deltag', 'binding_affinity', 'affinity']:
#                 if name in cols_lower:
#                     col_map['delta_g'] = df.columns[cols_lower.index(name)]
#                     break
            
#             required = ['heavy', 'light', 'antigen', 'delta_g']
#             if not all(k in col_map for k in required):
#                 print(f"   ⚠️ 跳过 {source_name}: 缺少必要列")
#                 continue
            
#             df_clean = df[[col_map['heavy'], col_map['light'], col_map['antigen'], col_map['delta_g']]].copy()
#             df_clean.columns = ['heavy', 'light', 'antigen', 'delta_g']
#             df_clean['dataset'] = source_name
            
#             df_clean.dropna(subset=['heavy', 'light', 'antigen', 'delta_g'], inplace=True)
#             df_clean['delta_g'] = pd.to_numeric(df_clean['delta_g'], errors='coerce')
#             df_clean.dropna(subset=['delta_g'], inplace=True)
            
#             def is_valid_seq(seq):
#                 if not isinstance(seq, str) or len(seq.strip()) == 0:
#                     return False
#                 return all(c in valid_aa for c in seq.upper())
            
#             before = len(df_clean)
#             df_clean = df_clean[
#                 df_clean['heavy'].apply(is_valid_seq) &
#                 df_clean['light'].apply(is_valid_seq) &
#                 df_clean['antigen'].apply(is_valid_seq)
#             ]
#             after = len(df_clean)
#             print(f"   清洗后：{after} / {before} 条有效数据")
            
#             if after > 0:
#                 all_data.append(df_clean)
                
#         except Exception as e:
#             print(f"   ❌ 加载失败 {file_path}: {e}")
    
#     if not all_data:
#         raise RuntimeError("未加载任何有效数据！")
    
#     merged_df = pd.concat(all_data, ignore_index=True)
#     print(f"\n✅ 总共合并 {len(merged_df)} 条样本")
#     return merged_df


# # === 按 dataset 分组划分 train/val/test (6:2:2) ===
# def split_by_dataset(df, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2):
#     train_list, val_list, test_list = [], [], []
    
#     print(f"\n📊 数据划分配置：train={train_ratio}, val={val_ratio}, test={test_ratio}")
    
#     for dataset_name in df['dataset'].unique():
#         subset = df[df['dataset'] == dataset_name].copy()
#         n_total = len(subset)
        
#         train_val, test = train_test_split(
#             subset, 
#             test_size=test_ratio, 
#             random_state=42
#         )
        
#         val_ratio_adjusted = val_ratio / (train_ratio + val_ratio)
#         train, val = train_test_split(
#             train_val, 
#             test_size=val_ratio_adjusted, 
#             random_state=42
#         )
        
#         train_list.append(train)
#         val_list.append(val)
#         test_list.append(test)
        
#         n_train, n_val, n_test = len(train), len(val), len(test)
#         print(f"  - {dataset_name}: 总={n_total}, 训练={n_train}({n_train/n_total:.2f}), "
#               f"验证={n_val}({n_val/n_total:.2f}), 测试={n_test}({n_test/n_total:.2f})")
    
#     return (
#         pd.concat(train_list, ignore_index=True),
#         pd.concat(val_list, ignore_index=True),
#         pd.concat(test_list, ignore_index=True)
#     )


# # === 构建模型 (适配 ESM2 嵌入) ===
# def build_dg_model(input_dim_h, input_dim_l, input_dim_ag, hidden_dim=128):
#     """
#     构建抗体 - 抗原亲和力预测模型
    
#     Args:
#         input_dim_h: 重链输入维度 (ESM2: 532)
#         input_dim_l: 轻链输入维度 (ESM2: 532)
#         input_dim_ag: 抗原输入维度 (ESM2: 500)
#         hidden_dim: 隐藏层维度
#     """
#     # 输入形状：(batch_size, feature_dim) - ESM2 嵌入是固定维度
#     input_h = Input(shape=(input_dim_h,), name='heavy')
#     input_l = Input(shape=(input_dim_l,), name='light')
#     input_ag = Input(shape=(input_dim_ag,), name='antigen')

#     # 添加维度以使用 Conv1D: (batch, 1, feature_dim)
#     def add_dim(x):
#         return tf.expand_dims(x, axis=1)
    
#     h_exp = tf.keras.layers.Lambda(add_dim)(input_h)
#     l_exp = tf.keras.layers.Lambda(add_dim)(input_l)
#     ag_exp = tf.keras.layers.Lambda(add_dim)(input_ag)

#     def extract_features(x, name_prefix):
#         # Conv1D: (batch, 1, feature_dim) → (batch, 1, hidden_dim)
#         x = Conv1D(hidden_dim, 1, padding='same', name=f'{name_prefix}_conv1d')(x)
#         x = LeakyReLU(alpha=0.2)(x)
#         x = BatchNormalization()(x)
#         x = Dropout(0.1)(x)
        
#         # Global pooling: (batch, 1, hidden_dim) → (batch, hidden_dim)
#         x = GlobalAveragePooling1D()(x)
#         return x

#     feat_h = extract_features(h_exp, 'heavy')
#     feat_l = extract_features(l_exp, 'light')
#     feat_ag = extract_features(ag_exp, 'antigen')

#     # 拼接特征
#     x = concatenate([feat_h, feat_l, feat_ag])  # (batch, hidden_dim*3)
    
#     # 全连接层
#     x = Dense(hidden_dim * 2)(x)
#     x = LeakyReLU(alpha=0.2)(x)
#     x = BatchNormalization()(x)
#     x = Dropout(DROPOUT_RATE)(x)
    
#     x = Dense(hidden_dim)(x)
#     x = LeakyReLU(alpha=0.2)(x)
#     x = BatchNormalization()(x)
#     x = Dropout(DROPOUT_RATE)(x)
    
#     x = Dense(hidden_dim // 2)(x)
#     x = LeakyReLU(alpha=0.2)(x)
#     x = Dense(1)(x)

#     model = Model(inputs=[input_h, input_l, input_ag], outputs=x)
#     return model


# # === 主程序 ===
# def main():
#     os.makedirs('results', exist_ok=True)
    
#     # 1. 加载并划分数据
#     df = load_and_merge_datasets(PRED_FILES)
#     df_train, df_val, df_test = split_by_dataset(df, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2)
    
#     print(f"\n📊 划分结果:")
#     print(f"   训练集：{len(df_train)}")
#     print(f"   验证集：{len(df_val)}")
#     print(f"   测试集：{len(df_test)}")

#     # 2. 提取序列和标签
#     def get_arrays(df_part):
#         heavies = df_part['heavy'].tolist()
#         lights = df_part['light'].tolist()
#         antigens = df_part['antigen'].tolist()
#         delta_g = df_part['delta_g'].values.astype(np.float32)
#         sources = df_part['dataset'].values
#         return heavies, lights, antigens, delta_g, sources

#     h_train, l_train, ag_train, y_train, _ = get_arrays(df_train)
#     h_val, l_val, ag_val, y_val, _ = get_arrays(df_val)
#     h_test, l_test, ag_test, y_test, sources_test = get_arrays(df_test)

#     # 3. 初始化 ESM2 嵌入器
#     print("\n" + "="*60)
#     print("🧬 初始化 ESM2 嵌入器")
#     print("="*60)
#     embedder = ESM2Embedder(model_name="esm2_t12_35M_UR50D")

#     # 4. 标签标准化 (使用 RobustScaler)
#     print("\n" + "="*60)
#     print("📈 对 ΔG 标签进行标准化 (RobustScaler)")
#     print("="*60)
#     label_scaler = RobustScaler()
#     y_train_scaled = label_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
#     y_val_scaled = label_scaler.transform(y_val.reshape(-1, 1)).flatten()
#     y_test_scaled = label_scaler.transform(y_test.reshape(-1, 1)).flatten()
    
#     print(f"   原始 ΔG: mean={y_train.mean():.4f}, std={y_train.std():.4f}")
#     print(f"   原始 ΔG 范围: [{y_train.min():.4f}, {y_train.max():.4f}]")
#     print(f"   标准化后：mean={y_train_scaled.mean():.4f}, std={y_train_scaled.std():.4f}")
#     print(f"   标准化后范围：[{y_train_scaled.min():.4f}, {y_train_scaled.max():.4f}]")
    
#     with open(SCALER_SAVE_PATH, 'wb') as f:
#         pickle.dump(label_scaler, f)
#     print(f"   ✅ 标签标准化器已保存至：{SCALER_SAVE_PATH}")

#     # 5. 序列嵌入 (ESM2)
#     print("\n" + "="*60)
#     print("🔄 将序列转为 ESM2 嵌入...")
#     print("="*60)
    
#     print("   处理重链 (Heavy)...")
#     X_h_train = embedder.embed_batch(h_train, target_dim=HEAVY_LIGHT_DIM)
#     X_h_val = embedder.embed_batch(h_val, target_dim=HEAVY_LIGHT_DIM)
#     X_h_test = embedder.embed_batch(h_test, target_dim=HEAVY_LIGHT_DIM)
    
#     print("   处理轻链 (Light)...")
#     X_l_train = embedder.embed_batch(l_train, target_dim=HEAVY_LIGHT_DIM)
#     X_l_val = embedder.embed_batch(l_val, target_dim=HEAVY_LIGHT_DIM)
#     X_l_test = embedder.embed_batch(l_test, target_dim=HEAVY_LIGHT_DIM)
    
#     print("   处理抗原 (Antigen)...")
#     X_ag_train = embedder.embed_batch(ag_train, target_dim=ANTIGEN_DIM)
#     X_ag_val = embedder.embed_batch(ag_val, target_dim=ANTIGEN_DIM)
#     X_ag_test = embedder.embed_batch(ag_test, target_dim=ANTIGEN_DIM)
    
#     print(f"   ✅ 嵌入形状：H={X_h_train.shape}, L={X_l_train.shape}, A={X_ag_train.shape}")

#     # 6. 特征标准化 (对 ESM2 嵌入做 StandardScaler)
#     print("\n" + "="*60)
#     print("📊 对 ESM2 特征进行标准化...")
#     print("="*60)
    
#     feature_scaler_h = StandardScaler()
#     feature_scaler_l = StandardScaler()
#     feature_scaler_ag = StandardScaler()
    
#     X_h_train = feature_scaler_h.fit_transform(X_h_train)
#     X_h_val = feature_scaler_h.transform(X_h_val)
#     X_h_test = feature_scaler_h.transform(X_h_test)
    
#     X_l_train = feature_scaler_l.fit_transform(X_l_train)
#     X_l_val = feature_scaler_l.transform(X_l_val)
#     X_l_test = feature_scaler_l.transform(X_l_test)
    
#     X_ag_train = feature_scaler_ag.fit_transform(X_ag_train)
#     X_ag_val = feature_scaler_ag.transform(X_ag_val)
#     X_ag_test = feature_scaler_ag.transform(X_ag_test)
    
#     # 保存特征标准化器
#     with open(FEATURE_SCALER_PATH, 'wb') as f:
#         pickle.dump({
#             'heavy': feature_scaler_h,
#             'light': feature_scaler_l,
#             'antigen': feature_scaler_ag
#         }, f)
#     print(f"   ✅ 特征标准化器已保存至：{FEATURE_SCALER_PATH}")

#     # 7. 构建模型
#     print("\n" + "="*60)
#     print("🏗️ 构建模型...")
#     print("="*60)
#     model = build_dg_model(
#         input_dim_h=HEAVY_LIGHT_DIM,
#         input_dim_l=HEAVY_LIGHT_DIM,
#         input_dim_ag=ANTIGEN_DIM,
#         hidden_dim=HIDDEN_DIM
#     )
#     model.compile(
#         optimizer=Adam(learning_rate=LEARNING_RATE),
#         loss='mse',
#         metrics=['mae']
#     )
#     model.summary()

#     # 8. 训练
#     print("\n" + "="*60)
#     print("🏃 开始训练...")
#     print("="*60)
    
#     early_stopping = EarlyStopping(
#         monitor='val_loss',
#         patience=8,
#         restore_best_weights=True,
#         verbose=1
#     )
    
#     history = model.fit(
#         [X_h_train, X_l_train, X_ag_train],
#         y_train_scaled,
#         batch_size=BATCH_SIZE,
#         epochs=N_EPOCHS,
#         validation_data=([X_h_val, X_l_val, X_ag_val], y_val_scaled),
#         callbacks=[early_stopping],
#         verbose=1
#     )

#     # 9. 测试集评估
#     print("\n" + "="*60)
#     print("🔍 在测试集上评估...")
#     print("="*60)
    
#     preds_scaled = model.predict([X_h_test, X_l_test, X_ag_test]).flatten()
    
#     print(f"   预测值 (标准化) 范围：[{preds_scaled.min():.4f}, {preds_scaled.max():.4f}]")
#     print(f"   预测值 (标准化) 标准差：{preds_scaled.std():.4f}")
    
#     # 反变换到原始尺度
#     preds = label_scaler.inverse_transform(preds_scaled.reshape(-1, 1)).flatten()
    
#     print(f"   预测值 (原始) 范围：[{preds.min():.4f}, {preds.max():.4f}]")
#     print(f"   真实值范围：[{y_test.min():.4f}, {y_test.max():.4f}]")

#     # 10. 计算指标
#     from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
#     from scipy.stats import pearsonr

#     def compute_metrics(y_true, y_pred):
#         if np.std(y_pred) < 1e-6 or np.std(y_true) < 1e-6:
#             pcc = 0.0
#         else:
#             pcc, _ = pearsonr(y_true, y_pred)
#         mae = mean_absolute_error(y_true, y_pred)
#         mse = mean_squared_error(y_true, y_pred)
#         rmse = np.sqrt(mse)
#         r2 = r2_score(y_true, y_pred)
#         return {"MAE": mae, "MSE": mse, "RMSE": rmse, "PCC": pcc, "R2": r2}

#     results = {}
#     unique_sources = np.unique(sources_test)
#     for src in unique_sources:
#         mask = sources_test == src
#         if np.sum(mask) < 2:
#             continue
#         results[src] = compute_metrics(y_test[mask], preds[mask])
#         print(f"   {src}: PCC={results[src]['PCC']:.4f}, RMSE={results[src]['RMSE']:.4f}")
    
#     results["Overall"] = compute_metrics(y_test, preds)

#     # 11. 保存结果
#     with open(RST_FILE, 'w') as f:
#         f.write("Dataset\tMAE\tMSE\tRMSE\tPCC\tR2\n")
#         for src in ["Overall"] + sorted([k for k in results.keys() if k != "Overall"]):
#             m = results[src]
#             f.write(f"{src}\t{m['MAE']:.4f}\t{m['MSE']:.4f}\t{m['RMSE']:.4f}\t{m['PCC']:.4f}\t{m['R2']:.4f}\n")
        
#         f.write("\nTrue\tPred\tSource\n")
#         for t, p, s in zip(y_test, preds, sources_test):
#             f.write(f"{t:.4f}\t{p:.4f}\t{s}\n")

#     print(f"\n✅ 训练完成！结果已保存至：{RST_FILE}")

#     # 12. 保存模型
#     model.save(MODEL_SAVE_PATH)
#     print(f"💾 模型已保存至：{MODEL_SAVE_PATH}")

#     # 13. 打印指标
#     print("\n" + "="*60)
#     print("📊 评估结果")
#     print("="*60)
#     for src in ["Overall"] + sorted([k for k in results.keys() if k != "Overall"]):
#         m = results[src]
#         print(f"{src:>12} | MAE: {m['MAE']:.3f}, MSE: {m['MSE']:.3f}, RMSE: {m['RMSE']:.3f}, PCC: {m['PCC']:.3f}, R²: {m['R2']:.3f}")
#     print("="*60)
    
#     # 14. 保存配置信息
#     config = {
#         'esm2_model': 'esm2_t12_35M_UR50D',
#         'heavy_light_dim': HEAVY_LIGHT_DIM,
#         'antigen_dim': ANTIGEN_DIM,
#         'hidden_dim': HIDDEN_DIM,
#         'batch_size': BATCH_SIZE,
#         'learning_rate': LEARNING_RATE,
#         'dropout_rate': DROPOUT_RATE,
#         'epochs': N_EPOCHS,
#         'label_scaler': 'RobustScaler',
#         'feature_scaler': 'StandardScaler'
#     }
    
#     config_path = 'results/2model_config_esm2.pkl'
#     with open(config_path, 'wb') as f:
#         pickle.dump(config, f)
#     print(f"📋 模型配置已保存至：{config_path}")


# if __name__ == "__main__":
#     # 需要导入 torch 用于 ESM2
#     import torch
#     main()