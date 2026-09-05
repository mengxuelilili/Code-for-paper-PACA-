"""
pre_split_test_sets.py
先为每个数据集划分出固定的测试集，保存为独立的 .pt 文件
主实验只使用 train+val，测试集永久锁死
"""

import torch
import os
import numpy as np
from sklearn.model_selection import train_test_split

# ======================================================
# 配置
# ======================================================
DATA_DIR = "/tmp/AbAgCDR/data/"
OUTPUT_DIR = "/root/autodl-tmp/AbAgCDR/data_split/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 原始数据文件
DATASET_FILES = {
    "paddle": "train_data.pt",
    "abbind": "abbind_data.pt",
    "sabdab": "sabdab_data.pt",
    "skempi": "skempi_data.pt"
}

SEED = 42  # 固定种子，确保测试集永远不变

# ======================================================
# 主函数
# ======================================================
def pre_split_datasets():
    print("="*60)
    print("📊 预划分测试集（固定，永不改变）")
    print("="*60)
    
    for name, filename in DATASET_FILES.items():
        filepath = os.path.join(DATA_DIR, filename)
        
        if not os.path.exists(filepath):
            print(f"⚠️ 文件不存在: {filepath}")
            continue
        
        print(f"\n处理 {name}...")
        
        # 加载原始数据
        data = torch.load(filepath, map_location="cpu")
        X_a = data["X_a"].cpu().numpy()
        X_b = data["X_b"].cpu().numpy()
        antigen = data["antigen"].cpu().numpy()
        y = data["y"].cpu().numpy()
        
        # 第一次划分：分出 test (20%)
        X_a_tv, X_a_test, X_b_tv, X_b_test, ag_tv, ag_test, y_tv, y_test = train_test_split(
            X_a, X_b, antigen, y, test_size=0.2, random_state=SEED
        )
        
        # 保存 test 集（永久锁死）
        test_data = {
            "X_a": torch.tensor(X_a_test, dtype=torch.float32),
            "X_b": torch.tensor(X_b_test, dtype=torch.float32),
            "antigen": torch.tensor(ag_test, dtype=torch.float32),
            "y": torch.tensor(y_test, dtype=torch.float32)
        }
        test_path = os.path.join(OUTPUT_DIR, f"{name}_test.pt")
        torch.save(test_data, test_path)
        print(f"  ✅ 测试集保存: {test_path} ({len(y_test)} 样本)")
        
        # 保存 train+val 合并集（用于主实验）
        trainval_data = {
            "X_a": torch.tensor(X_a_tv, dtype=torch.float32),
            "X_b": torch.tensor(X_b_tv, dtype=torch.float32),
            "antigen": torch.tensor(ag_tv, dtype=torch.float32),
            "y": torch.tensor(y_tv, dtype=torch.float32)
        }
        trainval_path = os.path.join(OUTPUT_DIR, f"{name}_trainval.pt")
        torch.save(trainval_data, trainval_path)
        print(f"  ✅ Train+Val 保存: {trainval_path} ({len(y_tv)} 样本)")
        
        # 打印信息
        print(f"    总计: {len(y)} 样本")
        print(f"    Train+Val: {len(y_tv)} (80%)")
        print(f"    Test: {len(y_test)} (20%)")
    
    print("\n" + "="*60)
    print("✅ 所有数据集预划分完成！")
    print(f"测试集路径: {OUTPUT_DIR}")
    print("="*60)


if __name__ == "__main__":
    pre_split_datasets()

"""
split_tsv_with_pt.py
使用与 PT 划分完全相同的 SEED 来划分 TSV
保证 TSV 和 PT 的划分一一对应
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

DATA_DIR = "/tmp/AbAgCDR/data/"
OUTPUT_DIR = "/root/autodl-tmp/AbAgCDR/data_split/"
SEED = 42  # 必须与 pre_split_test_sets.py 一致

TSV_FILES = {
    "paddle": "final_dataset_train.tsv",
    "abbind": "pairs_seq_abbind2_clean.tsv",
    "sabdab": "pairs_seq_sabdab_clean.tsv",
    "skempi": "pairs_seq_skempi_clean.tsv"
}

def split_tsv():
    for name, tsv_file in TSV_FILES.items():
        tsv_path = os.path.join(DATA_DIR, tsv_file)
        if not os.path.exists(tsv_path):
            print(f"⚠️ TSV 不存在: {tsv_path}")
            continue
        
        df = pd.read_csv(tsv_path, sep='\t')
        indices = np.arange(len(df))
        
        # 使用相同的 SEED 划分
        trainval_idx, test_idx = train_test_split(
            indices, test_size=0.2, random_state=SEED
        )
        
        trainval_df = df.iloc[trainval_idx].sort_index()
        test_df = df.iloc[test_idx].sort_index()
        
        trainval_path = os.path.join(OUTPUT_DIR, f"{name}_trainval.tsv")
        test_path = os.path.join(OUTPUT_DIR, f"{name}_test.tsv")
        
        trainval_df.to_csv(trainval_path, sep='\t', index=False)
        test_df.to_csv(test_path, sep='\t', index=False)
        
        print(f"{name}: trainval={len(trainval_df)}, test={len(test_df)}")

if __name__ == "__main__":
    split_tsv()


