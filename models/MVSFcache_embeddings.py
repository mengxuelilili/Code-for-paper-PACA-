import os
import torch
import pandas as pd
import numpy as np
from pathlib import Path

# 创建缓存目录
CACHE_DIR = Path("/tmp/AbAgCDR/embeddings")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

print("🧠 Loading ESM-2 (35M)...")
import esm
esm_model, alphabet = esm.pretrained.esm2_t12_35M_UR50D()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
esm_model = esm_model.to(device)
esm_model.eval()
batch_converter = alphabet.get_batch_converter()

def get_esm_embedding(seq):
    seq = str(seq).upper().strip()
    valid_aas = "ARNDCQEGHILKMFPSTWYV"
    seq = ''.join([aa for aa in seq if aa in valid_aas])
    if len(seq) == 0:
        seq = "A"
    data = [("protein", seq)]
    _, _, batch_tokens = batch_converter(data)
    batch_tokens = batch_tokens.to(device)
    with torch.no_grad():
        results = esm_model(batch_tokens, repr_layers=[12])
        token_repr = results["representations"][12]
    emb = token_repr[0, 1:-1].mean(dim=0).cpu().numpy()
    return emb

# 所有训练 TSV 路径
train_tsvs = [
    "/tmp/AbAgCDR/data/final_dataset_train.tsv",
    "/tmp/AbAgCDR/data/pairs_seq_skempi_clean.tsv",
    "/tmp/AbAgCDR/data/pairs_seq_sabdab_clean.tsv",
    "/tmp/AbAgCDR/data/pairs_seq_abbind2_clean.tsv",
]

all_samples = []
sample_sources = []  # 记录每个样本来自哪个文件

for tsv_path in train_tsvs:
    print(f"Processing {tsv_path}...")
    df = pd.read_csv(tsv_path, sep='\t', on_bad_lines='skip')
    required_cols = ['antibody_seq_a', 'antibody_seq_b', 'antigen_seq', 'delta_g']
    for col in required_cols:
        if col not in df.columns:
            continue
    for _, row in df.iterrows():
        a = str(row['antibody_seq_a'])
        b = str(row['antibody_seq_b'])
        ag = str(row['antigen_seq'])
        dg = float(row['delta_g'])
        if a and b and ag and not pd.isna(dg):
            all_samples.append((a, b, ag, dg))
            sample_sources.append(Path(tsv_path).stem)

print(f"Total samples: {len(all_samples)}")

# 提取所有唯一序列（去重加速）
unique_seqs = set()
for a, b, ag, _ in all_samples:
    unique_seqs.update([a, b, ag])

print(f"Unique sequences: {len(unique_seqs)}")

# 批量提取嵌入
seq_to_emb = {}
unique_list = list(unique_seqs)
for i in range(0, len(unique_list), 64):
    batch = unique_list[i:i+64]
    batch_data = [("protein", seq.upper()) for seq in batch]
    _, _, tokens = batch_converter(batch_data)
    tokens = tokens.to(device)
    with torch.no_grad():
        results = esm_model(tokens, repr_layers=[12])
        embs = results["representations"][12][:, 1:-1].mean(dim=1).cpu().numpy()
    for j, seq in enumerate(batch):
        seq_to_emb[seq] = embs[j]

# 为每个样本保存嵌入
l_embs, h_embs, ag_embs, labels = [], [], [], []
for a, b, ag, dg in all_samples:
    l_embs.append(seq_to_emb[a])
    h_embs.append(seq_to_emb[b])
    ag_embs.append(seq_to_emb[ag])
    labels.append(dg)

# 保存
np.save(CACHE_DIR / "lchain.npy", np.array(l_embs))
np.save(CACHE_DIR / "hchain.npy", np.array(h_embs))
np.save(CACHE_DIR / "ag.npy", np.array(ag_embs))
np.save(CACHE_DIR / "labels.npy", np.array(labels))
np.save(CACHE_DIR / "sources.npy", np.array(sample_sources))

print("✅ Embeddings cached successfully!")