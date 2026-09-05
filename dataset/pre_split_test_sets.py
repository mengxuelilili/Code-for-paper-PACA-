# """
# pre_split_test_sets.py
# 先为每个数据集划分出固定的测试集，保存为独立的 .pt 文件
# 主实验只使用 train+val，测试集永久锁死
# """

# import torch
# import os
# import numpy as np
# from sklearn.model_selection import train_test_split

# # ======================================================
# # 配置
# # ======================================================
# DATA_DIR = "/tmp/AbAgCDR/data/"
# OUTPUT_DIR = "/root/autodl-tmp/AbAgCDR/data_split/"
# os.makedirs(OUTPUT_DIR, exist_ok=True)

# # 原始数据文件
# DATASET_FILES = {
#     "paddle": "train_data.pt",
#     "abbind": "abbind_data.pt",
#     "sabdab": "sabdab_data.pt",
#     "skempi": "skempi_data.pt"
# }

# SEED = 42  # 固定种子，确保测试集永远不变

# # ======================================================
# # 主函数
# # ======================================================
# def pre_split_datasets():
#     print("="*60)
#     print("📊 预划分测试集（固定，永不改变）")
#     print("="*60)
    
#     for name, filename in DATASET_FILES.items():
#         filepath = os.path.join(DATA_DIR, filename)
        
#         if not os.path.exists(filepath):
#             print(f"⚠️ 文件不存在: {filepath}")
#             continue
        
#         print(f"\n处理 {name}...")
        
#         # 加载原始数据
#         data = torch.load(filepath, map_location="cpu")
#         X_a = data["X_a"].cpu().numpy()
#         X_b = data["X_b"].cpu().numpy()
#         antigen = data["antigen"].cpu().numpy()
#         y = data["y"].cpu().numpy()
        
#         # 第一次划分：分出 test (20%)
#         X_a_tv, X_a_test, X_b_tv, X_b_test, ag_tv, ag_test, y_tv, y_test = train_test_split(
#             X_a, X_b, antigen, y, test_size=0.2, random_state=SEED
#         )
        
#         # 保存 test 集（永久锁死）
#         test_data = {
#             "X_a": torch.tensor(X_a_test, dtype=torch.float32),
#             "X_b": torch.tensor(X_b_test, dtype=torch.float32),
#             "antigen": torch.tensor(ag_test, dtype=torch.float32),
#             "y": torch.tensor(y_test, dtype=torch.float32)
#         }
#         test_path = os.path.join(OUTPUT_DIR, f"{name}_test.pt")
#         torch.save(test_data, test_path)
#         print(f"  ✅ 测试集保存: {test_path} ({len(y_test)} 样本)")
        
#         # 保存 train+val 合并集（用于主实验）
#         trainval_data = {
#             "X_a": torch.tensor(X_a_tv, dtype=torch.float32),
#             "X_b": torch.tensor(X_b_tv, dtype=torch.float32),
#             "antigen": torch.tensor(ag_tv, dtype=torch.float32),
#             "y": torch.tensor(y_tv, dtype=torch.float32)
#         }
#         trainval_path = os.path.join(OUTPUT_DIR, f"{name}_trainval.pt")
#         torch.save(trainval_data, trainval_path)
#         print(f"  ✅ Train+Val 保存: {trainval_path} ({len(y_tv)} 样本)")
        
#         # 打印信息
#         print(f"    总计: {len(y)} 样本")
#         print(f"    Train+Val: {len(y_tv)} (80%)")
#         print(f"    Test: {len(y_test)} (20%)")
    
#     print("\n" + "="*60)
#     print("✅ 所有数据集预划分完成！")
#     print(f"测试集路径: {OUTPUT_DIR}")
#     print("="*60)


# if __name__ == "__main__":
#     pre_split_datasets()

# """
# split_tsv_with_pt.py
# 使用与 PT 划分完全相同的 SEED 来划分 TSV
# 保证 TSV 和 PT 的划分一一对应
# """

# import os
# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split

# DATA_DIR = "/tmp/AbAgCDR/data/"
# OUTPUT_DIR = "/root/autodl-tmp/AbAgCDR/data_split/"
# SEED = 42  # 必须与 pre_split_test_sets.py 一致

# TSV_FILES = {
#     "paddle": "final_dataset_train.tsv",
#     "abbind": "pairs_seq_abbind2_clean.tsv",
#     "sabdab": "pairs_seq_sabdab_clean.tsv",
#     "skempi": "pairs_seq_skempi_clean.tsv"
# }

# def split_tsv():
#     for name, tsv_file in TSV_FILES.items():
#         tsv_path = os.path.join(DATA_DIR, tsv_file)
#         if not os.path.exists(tsv_path):
#             print(f"⚠️ TSV 不存在: {tsv_path}")
#             continue
        
#         df = pd.read_csv(tsv_path, sep='\t')
#         indices = np.arange(len(df))
        
#         # 使用相同的 SEED 划分
#         trainval_idx, test_idx = train_test_split(
#             indices, test_size=0.2, random_state=SEED
#         )
        
#         trainval_df = df.iloc[trainval_idx].sort_index()
#         test_df = df.iloc[test_idx].sort_index()
        
#         trainval_path = os.path.join(OUTPUT_DIR, f"{name}_trainval.tsv")
#         test_path = os.path.join(OUTPUT_DIR, f"{name}_test.tsv")
        
#         trainval_df.to_csv(trainval_path, sep='\t', index=False)
#         test_df.to_csv(test_path, sep='\t', index=False)
        
#         print(f"{name}: trainval={len(trainval_df)}, test={len(test_df)}")

# if __name__ == "__main__":
#     split_tsv()


"""
re_split_consistently.py
确保PT和TSV划分完全一致
"""

import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

SEED = 42
TEST_SIZE = 0.2

# 路径配置
PT_DIR = "/tmp/AbAgCDR/data/"
TSV_DIR = "/tmp/AbAgCDR/data/"
OUTPUT_DIR = "/root/autodl-tmp/AbAgCDR/data_split/"

# 数据集配置
DATASETS = {
    'paddle': {
        'pt': 'train_data.pt',
        'tsv': 'final_dataset_train.tsv',
    },
    'abbind': {
        'pt': 'abbind_data.pt',
        'tsv': 'pairs_seq_abbind2_clean.tsv',
    },
    'sabdab': {
        'pt': 'sabdab_data.pt',
        'tsv': 'pairs_seq_sabdab_clean.tsv',
    },
    'skempi': {
        'pt': 'skempi_data.pt',
        'tsv': 'pairs_seq_skempi_clean.tsv',
    }
}

def re_split_dataset(name, config):
    print(f"\n{'='*50}")
    print(f"📊 重新划分 {name.upper()}")
    print('='*50)
    
    # ========== 1. 先划分TSV，获取索引 ==========
    tsv_path = os.path.join(TSV_DIR, config['tsv'])
    if not os.path.exists(tsv_path):
        print(f"  ⚠️ TSV不存在: {tsv_path}")
        return
    
    df = pd.read_csv(tsv_path, sep='\t')
    total_samples = len(df)
    indices = np.arange(total_samples)
    
    trainval_idx, test_idx = train_test_split(
        indices, test_size=TEST_SIZE, random_state=SEED
    )
    
    print(f"  TSV总样本: {total_samples}")
    print(f"  Train+Val: {len(trainval_idx)} (80%)")
    print(f"  Test: {len(test_idx)} (20%)")
    
    # 保存TSV划分
    trainval_df = df.iloc[trainval_idx].sort_index()
    test_df = df.iloc[test_idx].sort_index()
    
    trainval_tsv_path = os.path.join(OUTPUT_DIR, f"{name}_trainval.tsv")
    test_tsv_path = os.path.join(OUTPUT_DIR, f"{name}_test.tsv")
    
    trainval_df.to_csv(trainval_tsv_path, sep='\t', index=False)
    test_df.to_csv(test_tsv_path, sep='\t', index=False)
    print(f"  ✅ TSV划分保存成功")
    
    # ========== 2. 用相同的索引划分PT ==========
    pt_path = os.path.join(PT_DIR, config['pt'])
    if not os.path.exists(pt_path):
        print(f"  ⚠️ PT不存在: {pt_path}")
        return
    
    pt_data = torch.load(pt_path, map_location="cpu")
    X_a = pt_data["X_a"].numpy()
    X_b = pt_data["X_b"].numpy()
    antigen = pt_data["antigen"].numpy()
    y = pt_data["y"].numpy()
    label_scaler = pt_data.get("label_scaler", None)
    
    print(f"  PT总样本: {len(y)}")
    
    # 使用相同的索引划分
    X_a_trainval = X_a[trainval_idx]
    X_b_trainval = X_b[trainval_idx]
    antigen_trainval = antigen[trainval_idx]
    y_trainval = y[trainval_idx]
    
    X_a_test = X_a[test_idx]
    X_b_test = X_b[test_idx]
    antigen_test = antigen[test_idx]
    y_test = y[test_idx]
    
    # 保存PT划分
    trainval_pt_data = {
        "X_a": torch.tensor(X_a_trainval, dtype=torch.float32),
        "X_b": torch.tensor(X_b_trainval, dtype=torch.float32),
        "antigen": torch.tensor(antigen_trainval, dtype=torch.float32),
        "y": torch.tensor(y_trainval, dtype=torch.float32),
    }
    if label_scaler is not None:
        trainval_pt_data["label_scaler"] = label_scaler
    
    test_pt_data = {
        "X_a": torch.tensor(X_a_test, dtype=torch.float32),
        "X_b": torch.tensor(X_b_test, dtype=torch.float32),
        "antigen": torch.tensor(antigen_test, dtype=torch.float32),
        "y": torch.tensor(y_test, dtype=torch.float32),
    }
    if label_scaler is not None:
        test_pt_data["label_scaler"] = label_scaler
    
    trainval_pt_path = os.path.join(OUTPUT_DIR, f"{name}_trainval.pt")
    test_pt_path = os.path.join(OUTPUT_DIR, f"{name}_test.pt")
    
    torch.save(trainval_pt_data, trainval_pt_path)
    torch.save(test_pt_data, test_pt_path)
    
    print(f"  ✅ PT划分保存成功")
    print(f"  ✅ {name.upper()} 重新划分完成！")


def verify_consistency(name):
    """验证PT和TSV划分是否一致"""
    print(f"\n  🔍 验证 {name.upper()}")
    
    # 加载PT测试集
    pt_path = os.path.join(OUTPUT_DIR, f"{name}_test.pt")
    pt_data = torch.load(pt_path, map_location="cpu")
    y_pt = pt_data["y"].numpy()
    
    # 加载TSV测试集
    tsv_path = os.path.join(OUTPUT_DIR, f"{name}_test.tsv")
    tsv_df = pd.read_csv(tsv_path, sep='\t')
    y_tsv = tsv_df['delta_g'].values
    
    print(f"    PT样本数: {len(y_pt)}")
    print(f"    TSV样本数: {len(y_tsv)}")
    
    if len(y_pt) == len(y_tsv):
        print(f"    ✅ 样本数一致")
        
        # 检查数值是否匹配（排序后比较）
        # 注意：PT中y是归一化值，TSV中delta_g是原始值
        # 需要先获取label_scaler
        train_pt_path = os.path.join(OUTPUT_DIR, f"{name}_trainval.pt")
        train_data = torch.load(train_pt_path, map_location="cpu")
        label_scaler = train_data.get("label_scaler", None)
        
        if label_scaler is not None:
            y_pt_real = label_scaler.inverse_transform(y_pt.reshape(-1, 1)).flatten()
            
            # 排序后比较
            y_pt_sorted = np.sort(y_pt_real)
            y_tsv_sorted = np.sort(y_tsv)
            
            diff = np.mean(np.abs(y_pt_sorted - y_tsv_sorted))
            print(f"    排序后平均差异: {diff:.6f}")
            
            if diff < 0.01:
                print(f"    ✅ PT和TSV数据一致！")
                return True
            else:
                print(f"    ⚠️ 差异较大: {diff:.4f}")
                return False
        else:
            print(f"    ⚠️ 无法验证（label_scaler不存在）")
            return False
    else:
        print(f"    ❌ 样本数不一致！")
        return False


# ======================================================
# 主程序
# ======================================================

print("="*70)
print("🔄 重新划分数据集（确保PT和TSV一致）")
print("="*70)

os.makedirs(OUTPUT_DIR, exist_ok=True)

for name, config in DATASETS.items():
    re_split_dataset(name, config)

print("\n" + "="*70)
print("🔍 验证所有数据集")
print("="*70)

all_consistent = True
for name in DATASETS.keys():
    if not verify_consistency(name):
        all_consistent = False

if all_consistent:
    print("\n" + "="*70)
    print("✅ 所有数据集划分一致！可以继续画图8。")
    print("="*70)
else:
    print("\n" + "="*70)
    print("⚠️ 部分数据集不一致，请检查。")
    print("="*70)