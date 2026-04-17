# dataset.py
import pandas as pd
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import torch


class AntibodyAntigenDataset(Dataset):
    def __init__(self, csv_file, esm_model_name="facebook/esm2_t33_650M_UR50D", max_length=1024):
        self.df = pd.read_csv(csv_file, sep='\t')
        self.tokenizer = AutoTokenizer.from_pretrained(esm_model_name)
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        heavy_seq = str(row['antibody_seq_b'])   # 重链
        light_seq = str(row['antibody_seq_a'])   # 轻链
        antigen_seq = str(row['antigen_seq'])
        delta_g = float(row['delta_g'])

        def tokenize(seq):
            tokens = self.tokenizer(
                seq,
                return_tensors="pt",
                padding="max_length",
                max_length=self.max_length,
                truncation=True,
                add_special_tokens=True
            )["input_ids"].squeeze(0)
            return tokens

        return (
            tokenize(heavy_seq),
            tokenize(light_seq),
            tokenize(antigen_seq),
            torch.tensor(delta_g, dtype=torch.float32)
        )