import pandas as pd
import os
from anarci import run_anarci
from pathlib import Path

# ---------------------------
# 配置：输入输出路径
# ---------------------------
INPUT_DIR = "/tmp/AbAgCDR/data"
OUTPUT_DIR = "/tmp/AbAgCDR/models/data_annotated"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TSV_FILES = [
    "final_dataset_train.tsv",
    "pairs_seq_skempi_clean.tsv",
    "pairs_seq_sabdab_clean.tsv",
    "pairs_seq_abbind2_clean.tsv"
]

# 假设原始列名为：heavy_chain, antigen_seq, delta_g
HEAVY_COL = "antibody_seq_b"
ANTIGEN_COL = "antibody_seq_a"
LABEL_COL = "delta_g"

def annotate_heavy_chain(seq):
    """使用 ANARCI (IMGT scheme) 注释重链，返回7个区域字典"""
    if not isinstance(seq, str) or len(seq) < 20:
        return None
    try:
        # ANARCI 输入格式: [(name, sequence)]
        results = run_anarci([("H", seq)], scheme="imgt", output=False)
        numbering, alignment_details = results[0][0][0], results[0][1][0]
        
        regions = {
            'H-FR1': [], 'H-CDR1': [], 'H-FR2': [],
            'H-CDR2': [], 'H-FR3': [], 'H-CDR3': [], 'H-FR4': []
        }
        
        for pos, aa in numbering:
            if aa == '-' or not aa.isalpha():
                continue
            imgt_num = pos[0]  # IMGT 编号（整数）
            if 1 <= imgt_num <= 26:
                regions['H-FR1'].append(aa)
            elif 27 <= imgt_num <= 38:
                regions['H-CDR1'].append(aa)
            elif 39 <= imgt_num <= 55:
                regions['H-FR2'].append(aa)
            elif 56 <= imgt_num <= 65:
                regions['H-CDR2'].append(aa)
            elif 66 <= imgt_num <= 104:
                regions['H-FR3'].append(aa)
            elif 105 <= imgt_num <= 117:
                regions['H-CDR3'].append(aa)
            elif 118 <= imgt_num <= 128:
                regions['H-FR4'].append(aa)
            # 超出范围的残基忽略（如 tail）
        
        # 合并为字符串
        return {k: ''.join(v) for k, v in regions.items()}
    
    except Exception as e:
        print(f"ANARCI failed for sequence: {seq[:20]}... Error: {e}")
        return None

def process_tsv(input_path, output_path):
    print(f"Processing {input_path}...")
    df = pd.read_csv(input_path, sep='\t', engine='python', on_bad_lines='skip')
    
    # 检查必需列是否存在
    if HEAVY_COL not in df.columns:
        raise ValueError(f"Column '{HEAVY_COL}' not found in {input_path}. Available: {list(df.columns)}")
    
    region_data = []
    valid_indices = []
    for idx, row in df.iterrows():
        heavy_seq = row[HEAVY_COL]
        ann = annotate_heavy_chain(heavy_seq)
        if ann is not None:
            region_data.append(ann)
            valid_indices.append(idx)
        else:
            print(f"Skipping row {idx} due to ANARCI failure.")
    
    # 构建新 DataFrame
    regions_df = pd.DataFrame(region_data, index=valid_indices)
    df_annotated = pd.concat([df.loc[valid_indices].reset_index(drop=True), regions_df.reset_index(drop=True)], axis=1)
    
    # 保存
    df_annotated.to_csv(output_path, sep='\t', index=False)
    print(f"Saved {len(df_annotated)} / {len(df)} samples to {output_path}")

# ---------------------------
# 执行所有文件
# ---------------------------
if __name__ == "__main__":
    for tsv in TSV_FILES:
        input_path = os.path.join(INPUT_DIR, tsv)
        output_path = os.path.join(OUTPUT_DIR, tsv)
        if os.path.exists(input_path):
            process_tsv(input_path, output_path)
        else:
            print(f"Warning: {input_path} does not exist, skipping.")
    
    print("\n✅ All files processed! Use data in:", OUTPUT_DIR)