import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from pathlib import Path


# === 配置氨基酸编码 ===
AA_LIST = list("ACDEFGHIKLMNPQRSTVWY")  # 20 standard amino acids
AA_TO_IDX = {aa: idx for idx, aa in enumerate(AA_LIST)}
AA_TO_IDX["X"] = len(AA_LIST)  # unknown / rare AA
PAD_IDX = len(AA_TO_IDX)       # padding token (not used in one-hot, but for clarity)

# 假设使用 one-hot 编码 → 特征维度 = 21
PT_FEATURE_SIZE = len(AA_TO_IDX)  # 21


def seq_to_onehot(seq: str, max_len: int) -> np.ndarray:
    """
    将氨基酸序列转为 one-hot 矩阵 (max_len, 21)
    超长截断，不足补零（padding 用全 0 向量）
    """
    seq = seq.upper()
    L = min(len(seq), max_len)
    arr = np.zeros((max_len, PT_FEATURE_SIZE), dtype=np.float32)
    for i, aa in enumerate(seq[:L]):
        idx = AA_TO_IDX.get(aa, AA_TO_IDX["X"])
        arr[i, idx] = 1.0
    return arr


class TriProtDataset(Dataset):
    def __init__(
        self,
        tsv_path: str,
        max_heavy_len: int = 220,
        max_light_len: int = 220,
        max_antigen_len: int = 500,
    ):
        """
        Args:
            tsv_path: 如 '/tmp/AbAgCDR/data/final_dataset_train.tsv'
            max_heavy_len: 重链最大长度
            max_light_len: 轻链最大长度
            max_antigen_len: 抗原最大长度
        """
        self.df = pd.read_csv(tsv_path, sep='\t')
        assert {"antibody_seq_a", "antibody_seq_b", "antigen_seq", "delta_g"} <= set(self.df.columns)

        self.max_heavy_len = max_heavy_len
        self.max_light_len = max_light_len
        self.max_antigen_len = max_antigen_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        heavy_seq = row["antibody_seq_a"]
        light_seq = row["antibody_seq_b"]
        antigen_seq = row["antigen_seq"]
        delta_g = row["delta_g"]

        # 转为 one-hot 特征矩阵
        heavy_tensor = seq_to_onehot(heavy_seq, self.max_heavy_len)  # (H, 21)
        light_tensor = seq_to_onehot(light_seq, self.max_light_len)  # (L, 21)
        antigen_tensor = seq_to_onehot(antigen_seq, self.max_antigen_len)  # (A, 21)

        label = np.array(delta_g, dtype=np.float32)

        return (
            heavy_tensor.astype(np.float32),
            light_tensor.astype(np.float32),
            antigen_tensor.astype(np.float32),
            label,
        )