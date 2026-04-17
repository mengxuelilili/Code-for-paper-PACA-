# # preprocess_embeddings.py
# import os
# import torch
# import pandas as pd
# import pickle
# import lmdb
# from igfold import IgFoldRunner
# import esm
# from cfg_ab import AminoAcid_Vocab

# os.makedirs("/AntiBinder/embeddings", exist_ok=True)

# # 加载数据
# df = pd.read_csv("/AntiBinder/datasets/xx")
# df = df.dropna(subset=['H-FR1','H-CDR1','H-FR2','H-CDR2','H-FR3','H-CDR3','H-FR4', 'Antigen Sequence'])

# # 初始化模型（仅一次）
# esm_model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
# esm_model.eval().cuda()
# batch_converter = alphabet.get_batch_converter()

# igfold = IgFoldRunner()

# # LMDB for antibody structure
# env = lmdb.open('/AntiBinder/datasets/fold_emb/fold_emb_for_train', map_size=1024*1024*1024*50, lock=False)

# for idx, row in df.iterrows():
#     try:
#         sample_id = f"{idx}_{row['Antigen']}"

#         # === Antigen ESM embedding ===
#         antigen_seq = row['Antigen Sequence']
#         if len(antigen_seq) > 1024:
#             antigen_seq = antigen_seq[:1024]
#         batch_labels, batch_strs, antigen_tensor = batch_converter([("antigen", antigen_seq)])
#         with torch.no_grad():
#             antigen_repr = esm_model(antigen_tensor.cuda(), repr_layers=[33])['representations'][33].cpu()  # [1, L, 1280]
#         torch.save(antigen_repr.squeeze(0), f"/AntiBinder/embeddings/antigen_{sample_id}.pt")

#         # === Antibody sequence & type ===
#         vh_seq = row['H-FR1'] + row['H-CDR1'] + row['H-FR2'] + row['H-CDR2'] + row['H-FR3'] + row['H-CDR3'] + row['H-FR4']
#         antibody_token = torch.tensor([AminoAcid_Vocab.get(aa, 0) for aa in vh_seq])
#         torch.save(antibody_token, f"/AntiBinder/embeddings/antibody_token_{sample_id}.pt")

#         # region type
#         HF1 = [1]*len(row['H-FR1'])
#         HCDR1 = [3]*len(row['H-CDR1'])
#         HF2 = [1]*len(row['H-FR2'])
#         HCDR2 = [4]*len(row['H-CDR2'])
#         HF3 = [1]*len(row['H-FR3'])
#         HCDR3 = [5]*len(row['H-CDR3'])
#         HF4 = [1]*len(row['H-FR4'])
#         at_type = torch.tensor(HF1 + HCDR1 + HF2 + HCDR2 + HF3 + HCDR3 + HF4)
#         torch.save(at_type, f"/AntiBinder/embeddings/at_type_{sample_id}.pt")

#         # === Antibody structure (IgFold) ===
#         with env.begin(write=True) as txn:
#             if txn.get(vh_seq.encode()) is None:
#                 emb = igfold.embed(sequences={"H": vh_seq})
#                 struct_emb = emb.structure_embs.detach().cpu()  # [1, L, 64]
#                 txn.put(vh_seq.encode(), pickle.dumps(struct_emb))

#         # Save metadata
#         torch.save({
#             'label': row['ANT_Binding'],
#             'antigen_len': len(antigen_seq),
#             'antibody_len': len(vh_seq),
#             'sample_id': sample_id
#         }, f"/AntiBinder/embeddings/meta_{sample_id}.pt")

#         print(f"Processed {idx}")

#     except Exception as e:
#         print(f"Error at {idx}: {e}")
#         continue

# env.close()

import os
import torch
import pandas as pd
import pickle
import lmdb
import numpy as np

# 尝试导入 IgFold 和 ESM，如果未安装会报错提示
try:
    from igfold import IgFoldRunner
except ImportError:
    print("❌ 错误: 未找到 igfold。请运行: pip install igfold")
    exit(1)

try:
    import esm
except ImportError:
    print("❌ 错误: 未找到 esm。请运行: pip install fair-esm")
    exit(1)

# ==============================
# 1. 定义词汇表 (替代 cfg_ab.AminoAcid_Vocab)
# ==============================
# 标准 20 种氨基酸 + 未知 token (0)
AMINO_ACIDS = "ARNDCQEGHILKMFPSTWYV"
AminoAcid_Vocab = {aa: i + 1 for i, aa in enumerate(AMINO_ACIDS)}
# 0 保留给未知字符或 padding
AminoAcid_Vocab['X'] = 0 
AminoAcid_Vocab['-'] = 0 
AminoAcid_Vocab['U'] = 0 # Selenocysteine 等罕见氨基酸映射为 0 或单独处理

def get_amino_token(seq):
    return [AminoAcid_Vocab.get(aa.upper(), 0) for aa in seq]

# ==============================
# 2. 配置路径
# ==============================
DATA_PATH = "/AntiBinder/datasets/xx"  # 请确认此文件存在
EMB_DIR = "/AntiBinder/embeddings"
LMDB_PATH = '/AntiBinder/datasets/fold_emb/fold_emb_for_train'

os.makedirs(EMB_DIR, exist_ok=True)
os.makedirs(os.path.dirname(LMDB_PATH), exist_ok=True)

print(f"🚀 开始预处理...")
print(f"📂 数据源: {DATA_PATH}")
print(f"💾 嵌入保存目录: {EMB_DIR}")
print(f"🗄️ 结构数据库: {LMDB_PATH}")

# ==============================
# 3. 加载数据
# ==============================
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"数据文件不存在: {DATA_PATH}")

df = pd.read_csv(DATA_PATH)
required_cols = ['H-FR1','H-FR1','H-CDR1','H-FR2','H-CDR2','H-FR3','H-CDR3','H-FR4', 'Antigen Sequence', 'ANT_Binding']
# 检查列是否存在 (注意原代码 H-FR1 写了两次，这里修正)
check_cols = ['H-FR1','H-CDR1','H-FR2','H-CDR2','H-FR3','H-CDR3','H-FR4', 'Antigen Sequence']
missing = [c for c in check_cols if c not in df.columns]
if missing:
    raise ValueError(f"CSV 缺少必要列: {missing}")

# 删除含有空值的行
df = df.dropna(subset=check_cols + ['ANT_Binding'])
df = df.reset_index(drop=True)
print(f"✅ 有效数据行数: {len(df)}")

# ==============================
# 4. 初始化模型
# ==============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🧠 使用设备: {device}")

# 加载 ESM-2 (650M)
print("⏳ 加载 ESM-2 模型...")
esm_model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
esm_model = esm_model.to(device)
esm_model.eval()
batch_converter = alphabet.get_batch_converter()

# 加载 IgFold
print("⏳ 加载 IgFold 模型...")
igfold = IgFoldRunner()
igfold.model.eval() # 确保 eval 模式

# 初始化 LMDB
# map_size: 50GB (根据需求调整)
env = lmdb.open(LMDB_PATH, map_size=1024*1024*1024*50, lock=False, writemap=True)

# ==============================
# 5. 主循环
# ==============================
for idx, row in df.iterrows():
    try:
        # 生成唯一 ID
        # 使用索引和抗原名称的一部分避免过长，确保唯一性
        antigen_name = str(row.get('Antigen', 'Unknown')).replace('/', '_').replace('\\', '_')[:50]
        sample_id = f"idx{idx}_{antigen_name}"
        
        # --- A. 抗原序列处理 (ESM) ---
        antigen_seq = str(row['Antigen Sequence']).strip().upper()
        if len(antigen_seq) == 0:
            continue
        if len(antigen_seq) > 1024:
            antigen_seq = antigen_seq[:1024] # 截断以防显存爆炸
        
        batch_labels, batch_strs, antigen_tensor = batch_converter([("antigen", antigen_seq)])
        antigen_tensor = antigen_tensor.to(device)
        
        with torch.no_grad():
            results = esm_model(antigen_tensor, repr_layers=[33])
            # [1, L, 1280] -> [L, 1280]
            antigen_repr = results['representations'][33].squeeze(0).cpu()
        
        torch.save(antigen_repr, os.path.join(EMB_DIR, f"antigen_{sample_id}.pt"))

        # --- B. 抗体序列构建 & Tokenization ---
        vh_parts = [
            str(row['H-FR1']), str(row['H-CDR1']), str(row['H-FR2']), 
            str(row['H-CDR2']), str(row['H-FR3']), str(row['H-CDR3']), str(row['H-FR4'])
        ]
        vh_seq = "".join(vh_parts).strip().upper()
        
        if len(vh_seq) == 0:
            continue

        # 转换为 Token IDs
        token_ids = get_amino_token(vh_seq)
        antibody_token = torch.tensor(token_ids, dtype=torch.long)
        torch.save(antibody_token, os.path.join(EMB_DIR, f"antibody_token_{sample_id}.pt"))

        # --- C. 区域类型编码 (Region Type) ---
        # 编码方案: FR=1, CDR1=3, CDR2=4, CDR3=5 (根据你的原逻辑)
        region_lengths = [
            len(row['H-FR1']), len(row['H-CDR1']), len(row['H-FR2']), 
            len(row['H-CDR2']), len(row['H-FR3']), len(row['H-CDR3']), len(row['H-FR4'])
        ]
        region_types = [1, 3, 1, 4, 1, 5, 1] # 对应上面的部分
        
        at_type_list = []
        for length, r_type in zip(region_lengths, region_types):
            at_type_list.extend([r_type] * length)
            
        at_type = torch.tensor(at_type_list, dtype=torch.long)
        torch.save(at_type, os.path.join(EMB_DIR, f"at_type_{sample_id}.pt"))

        # --- D. 抗体结构嵌入 (IgFold) ---
        # 修正：使用 sample_id 作为 key，而不是 vh_seq，防止序列重复导致覆盖
        with env.begin(write=True) as txn:
            lmdb_key = sample_id.encode()
            
            # 检查是否已存在 (可选，如果想跳过已处理的)
            if txn.get(lmdb_key) is not None:
                print(f"⏭️ 跳过已存在结构: {sample_id}")
            else:
                # IgFold 输入格式通常是字典 {"H": seq} 或列表
                # 注意：IgFold 可能需要完整的 Fv 或单链，这里假设输入 VH 即可
                try:
                    # igfold.embed 返回对象，具体属性视版本而定
                    # 某些版本返回 {'structure_embs': tensor, 'coords': tensor}
                    emb_result = igfold.embed(sequences={"H": vh_seq})
                    
                    # 提取结构嵌入 (通常是 [1, L, 64] 或类似)
                    if hasattr(emb_result, 'structure_embs'):
                        struct_emb = emb_result.structure_embs.detach().cpu()
                    elif isinstance(emb_result, dict) and 'structure_embs' in emb_result:
                        struct_emb = emb_result['structure_embs'].detach().cpu()
                    else:
                        # 兼容旧版本或直接返回 tensor 的情况
                        struct_emb = emb_result.detach().cpu() if torch.is_tensor(emb_result) else None
                        
                    if struct_emb is not None:
                        txn.put(lmdb_key, pickle.dumps(struct_emb))
                    else:
                        print(f"⚠️ IgFold 返回空嵌入: {sample_id}")
                        
                except Exception as ie:
                    print(f"❌ IgFold 处理失败 {sample_id}: {ie}")
                    # 可以选择存入一个全零标记表示失败，或者跳过
                    txn.put(lmdb_key, pickle.dumps(None))

        # --- E. 保存元数据 ---
        meta_data = {
            'label': float(row['ANT_Binding']),
            'antigen_len': len(antigen_seq),
            'antibody_len': len(vh_seq),
            'sample_id': sample_id,
            'original_antigen_name': antigen_name
        }
        torch.save(meta_data, os.path.join(EMB_DIR, f"meta_{sample_id}.pt"))

        if (idx + 1) % 10 == 0:
            print(f"✅ 已处理: {idx + 1}/{len(df)} (当前样本: {sample_id})")
            # 定期清理显存
            torch.cuda.empty_cache()

    except Exception as e:
        print(f"❌ 严重错误 at index {idx}: {e}")
        import traceback
        traceback.print_exc()
        continue

env.close()
print("🎉 所有数据处理完成！")