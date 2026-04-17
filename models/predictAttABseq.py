# -*- coding: utf-8 -*-
"""
AttABseq 预测脚本
输入：TSV 文件（antibody_seq_a, antibody_seq_b, antigen_seq, delta_g）
输出：CSV 文件（Index, delta_g, predicted_delta_g）
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import pickle
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# 尝试导入 ESM2
try:
    import esm
    ESM_AVAILABLE = True
    print("✅ ESM2 可用")
except ImportError:
    ESM_AVAILABLE = False
    print("⚠️ ESM2 不可用，请安装：pip install fair-esm")


# ============================================================================
# 序列编码模块
# ============================================================================

class SequenceEncoder:
    def __init__(self, use_esm=True):
        self.use_esm = use_esm and ESM_AVAILABLE
        self.valid_aa = "ACDEFGHIKLMNPQRSTVWY"
        
        if self.use_esm:
            print("🧠 加载 ESM2 模型...")
            self.model, self.alphabet = esm.pretrained.esm2_t12_35M_UR50D()
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model = self.model.to(self.device)
            self.model.eval()
            self.batch_converter = self.alphabet.get_batch_converter()
            print(f"✅ ESM2 加载完成 (设备：{self.device})")
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def _clean_seq(self, seq):
        seq = str(seq).upper().strip()
        return ''.join([aa for aa in seq if aa in self.valid_aa])
    
    def get_embedding(self, seq):
        seq = self._clean_seq(seq)
        if not seq:
            seq = "A"
        
        if self.use_esm:
            data = [("protein", seq)]
            _, _, batch_tokens = self.batch_converter(data)
            batch_tokens = batch_tokens.to(self.device)
            
            with torch.no_grad():
                results = self.model(batch_tokens, repr_layers=[12])
                token_repr = results["representations"][12]
            emb = token_repr[0, 1:-1].mean(dim=0).cpu().numpy()
        else:
            emb = np.zeros((len(seq), 20))
            aa_map = {aa: i for i, aa in enumerate(self.valid_aa)}
            for i, aa in enumerate(seq):
                emb[i, aa_map[aa]] = 1
            emb = emb.mean(axis=0)
        
        return emb.astype(np.float32)
    
    def encode_batch(self, sequences, target_dim=None):
        embeddings = []
        for seq in tqdm(sequences, desc="Encoding", leave=False):
            emb = self.get_embedding(seq)
            embeddings.append(emb)
        
        embeddings = np.array(embeddings, dtype=np.float32)
        
        if target_dim is not None:
            if embeddings.shape[1] > target_dim:
                embeddings = embeddings[:, :target_dim]
            elif embeddings.shape[1] < target_dim:
                pad = np.zeros((embeddings.shape[0], target_dim - embeddings.shape[1]), dtype=np.float32)
                embeddings = np.concatenate([embeddings, pad], axis=1)
        
        return embeddings


# ============================================================================
# 模型架构
# ============================================================================

class SelfAttention(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout, device):
        super().__init__()
        self.hid_dim = hid_dim
        self.n_heads = n_heads
        assert hid_dim % n_heads == 0
        
        self.w_q = nn.Linear(hid_dim, hid_dim)
        self.w_k = nn.Linear(hid_dim, hid_dim)
        self.w_v = nn.Linear(hid_dim, hid_dim)
        self.fc = nn.Linear(hid_dim, hid_dim)
        self.do = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim // n_heads])).to(device)
    
    def forward(self, query, key, value, mask=None):
        bsz = query.shape[0]
        Q = self.w_q(query)
        K = self.w_k(key)
        V = self.w_v(value)
        
        Q = Q.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)
        K = K.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)
        V = V.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)
        
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale
        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)
        
        attention = self.do(F.softmax(energy, dim=-1))
        x = torch.matmul(attention, V)
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(bsz, -1, self.hid_dim)
        x = self.fc(x)
        return x


class Encoder(nn.Module):
    def __init__(self, protein_dim, hid_dim, n_layers, kernel_size, dropout, device):
        super().__init__()
        assert kernel_size % 2 == 1
        
        self.input_dim = protein_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.device = device
        self.scale = torch.sqrt(torch.FloatTensor([0.5])).to(device)
        
        self.convs1 = nn.ModuleList([
            nn.Conv1d(hid_dim, 2*hid_dim, kernel_size, padding=(kernel_size-1)//2) 
            for _ in range(self.n_layers)
        ])
        self.convs2 = nn.ModuleList([
            nn.Conv1d(hid_dim, 2*hid_dim, 5, padding=2) 
            for _ in range(self.n_layers)
        ])
        self.convs3 = nn.ModuleList([
            nn.Conv1d(hid_dim, 2*hid_dim, 7, padding=3) 
            for _ in range(self.n_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(self.input_dim, self.hid_dim)
        self.project = nn.Linear(hid_dim * 3, hid_dim)
        self.ln = nn.LayerNorm(hid_dim)
    
    def forward(self, protein):
        conv_input = self.fc(protein)
        conv_input1 = conv_input.permute(0, 2, 1)
        conv_input2 = conv_input.permute(0, 2, 1)
        conv_input3 = conv_input.permute(0, 2, 1)
        
        for conv in self.convs1:
            conved = (F.glu(conv(self.dropout(conv_input1)), dim=1) + conv_input1) * self.scale
            conv_input1 = conved
        
        for conv in self.convs2:
            conved = (F.glu(conv(self.dropout(conv_input2)), dim=1) + conv_input2) * self.scale
            conv_input2 = conved
        
        for conv in self.convs3:
            conved = (F.glu(conv(self.dropout(conv_input3)), dim=1) + conv_input3) * self.scale
            conv_input3 = conved
        
        conved = torch.cat((conv_input1, conv_input2, conv_input3), 1)
        conved = conved.permute(0, 2, 1)
        conved = self.project(conved)
        conved = self.ln(conved)
        return conved


class DecoderLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, pf_dim, self_attention, positionwise_feedforward, dropout, device):
        super().__init__()
        self.ln = nn.LayerNorm(hid_dim)
        self.sa = self_attention(hid_dim, n_heads, dropout, device)
        self.ea = self_attention(hid_dim, n_heads, dropout, device)
        self.pf = positionwise_feedforward(hid_dim, pf_dim, dropout)
        self.do = nn.Dropout(dropout)
    
    def forward(self, trg, src, trg_mask=None, src_mask=None):
        trg = self.ln(trg + self.do(self.sa(trg, trg, trg, trg_mask)))
        trg = self.ln(trg + self.do(self.ea(trg, src, src, src_mask)))
        trg = self.ln(trg + self.do(self.pf(trg)))
        return trg


class PositionwiseFeedforward(nn.Module):
    def __init__(self, hid_dim, pf_dim, dropout):
        super().__init__()
        self.fc_1 = nn.Conv1d(hid_dim, pf_dim, 1)
        self.fc_2 = nn.Conv1d(pf_dim, hid_dim, 1)
        self.do = nn.Dropout(dropout)
    
    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.fc_2(self.do(F.relu(self.fc_1(x))))
        x = x.permute(0, 2, 1)
        return x


class AttABseqModel(nn.Module):
    def __init__(self, heavy_dim=532, light_dim=532, antigen_dim=500,
                 hid_dim=128, n_layers=3, n_heads=8, pf_dim=256,
                 dropout=0.2, device=None):
        super().__init__()
        
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        
        self.heavy_encoder = Encoder(heavy_dim, hid_dim, n_layers, 3, dropout, device)
        self.light_encoder = Encoder(light_dim, hid_dim, n_layers, 3, dropout, device)
        self.antigen_encoder = Encoder(antigen_dim, hid_dim, n_layers, 3, dropout, device)
        
        self_attention = SelfAttention
        positionwise_feedforward = PositionwiseFeedforward
        decoder_layer = DecoderLayer
        
        self.ag_ab_decoder = nn.ModuleList([
            decoder_layer(hid_dim, n_heads, pf_dim, self_attention, 
                         positionwise_feedforward, dropout, device)
            for _ in range(n_layers)
        ])
        
        self.ab_ag_decoder = nn.ModuleList([
            decoder_layer(hid_dim, n_heads, pf_dim, self_attention, 
                         positionwise_feedforward, dropout, device)
            for _ in range(n_layers)
        ])
        
        self.fc_agg = nn.Linear(hid_dim * 2, hid_dim)
        self.fc1 = nn.Linear(hid_dim, hid_dim // 2)
        self.fc2 = nn.Linear(hid_dim // 2, 64)
        self.fc3 = nn.Linear(64, 16)
        self.fc_out = nn.Linear(16, 1)
        
        self.dropout = nn.Dropout(dropout)
        self.gn = nn.GroupNorm(8, 64)
    
    def _make_mask(self, seq_len):
        return torch.ones((seq_len, seq_len), device=self.device)
    
    def _decode(self, decoder_layers, trg, src, trg_mask, src_mask):
        for layer in decoder_layers:
            trg = layer(trg, src, trg_mask, src_mask)
        return trg
    
    def _global_pool(self, x, mask=None):
        if mask is not None:
            x = x * mask.unsqueeze(-1)
            return x.sum(dim=1) / mask.sum(dim=1, keepdim=True)
        return x.mean(dim=1)
    
    def forward(self, heavy, light, antigen):
        heavy = heavy.unsqueeze(1)
        light = light.unsqueeze(1)
        antigen = antigen.unsqueeze(1)
        
        enc_heavy = self.heavy_encoder(heavy)
        enc_light = self.light_encoder(light)
        enc_antigen = self.antigen_encoder(antigen)
        
        enc_ab = (enc_heavy + enc_light) / 2
        
        ab_mask = self._make_mask(enc_ab.shape[1]).unsqueeze(0).unsqueeze(3)
        ag_mask = self._make_mask(enc_antigen.shape[1]).unsqueeze(0).unsqueeze(3)
        
        ag_ab = self._decode(self.ag_ab_decoder, enc_antigen, enc_ab, ag_mask, ab_mask)
        ab_ag = self._decode(self.ab_ag_decoder, enc_ab, enc_antigen, ab_mask, ag_mask)
        
        ag_ab_pool = self._global_pool(ag_ab)
        ab_ag_pool = self._global_pool(ab_ag)
        
        x = torch.cat([ag_ab_pool, ab_ag_pool], dim=-1)
        x = self.dropout(F.relu(self.fc_agg(x)))
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.gn(x)
        x = self.dropout(F.relu(self.fc3(x)))
        x = self.fc_out(x)
        
        return x.squeeze(-1)


# ============================================================================
# 预测主函数
# ============================================================================

def predict(input_tsv, model_path, scaler_path, output_csv, batch_size=32, use_esm=True):
    """
    执行预测并保存结果
    
    Args:
        input_tsv: 输入 TSV 文件路径
        model_path: 模型权重路径 (.pth)
        scaler_path: 标准化器路径 (.pkl)
        output_csv: 输出 CSV 文件路径
        batch_size: 批次大小
        use_esm: 是否使用 ESM2
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  使用设备：{device}")
    
    # ========== 1. 加载模型 ==========
    print("\n" + "="*60)
    print("🏗️ 加载模型")
    print("="*60)
    
    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint.get('config', {})
    
    model = AttABseqModel(
        heavy_dim=config.get('heavy_dim', 532),
        light_dim=config.get('light_dim', 532),
        antigen_dim=config.get('antigen_dim', 500),
        hid_dim=config.get('hidden_dim', 128),
        n_layers=config.get('n_layers', 3),
        n_heads=config.get('n_heads', 8),
        dropout=config.get('dropout', 0.2),
        device=device
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    print(f"✅ 模型已加载：{model_path}")
    
    # ========== 2. 加载标准化器 ==========
    print("\n" + "="*60)
    print("📈 加载标准化器")
    print("="*60)
    
    with open(scaler_path, 'rb') as f:
        label_scaler = pickle.load(f)
    print(f"✅ 标准化器已加载：{scaler_path}")
    
    # ========== 3. 加载输入数据 ==========
    print("\n" + "="*60)
    print("📊 加载输入数据")
    print("="*60)
    print(f"输入文件：{input_tsv}")
    
    df = pd.read_csv(input_tsv, sep='\t')
    print(f"原始数据量：{len(df)} 条")
    
    # 列名映射（支持多种命名）
    col_map = {}
    cols_lower = [c.lower() for c in df.columns]
    
    for name in ['antibody_seq_a', 'light', 'vl', 'l']:
        if name in cols_lower:
            col_map['light'] = df.columns[cols_lower.index(name)]
            break
    
    for name in ['antibody_seq_b', 'heavy', 'vh', 'h']:
        if name in cols_lower:
            col_map['heavy'] = df.columns[cols_lower.index(name)]
            break
    
    for name in ['antigen_seq', 'antigen', 'ag']:
        if name in cols_lower:
            col_map['antigen'] = df.columns[cols_lower.index(name)]
            break
    
    has_delta_g = False
    for name in ['delta_g', 'ddg', 'deltag', 'binding_affinity']:
        if name in cols_lower:
            col_map['delta_g'] = df.columns[cols_lower.index(name)]
            has_delta_g = True
            break
    
    # 提取序列和标签
    light_seqs = df[col_map['light']].astype(str).tolist()
    heavy_seqs = df[col_map['heavy']].astype(str).tolist()
    antigen_seqs = df[col_map['antigen']].astype(str).tolist()
    
    if has_delta_g:
        delta_g_values = df[col_map['delta_g']].astype(float).tolist()
    else:
        delta_g_values = [0.0] * len(df)
    
    # 清洗序列
    valid_aa = set("ACDEFGHIKLMNPQRSTVWY")
    def clean_seq(seq):
        seq = str(seq).upper().strip()
        return ''.join([aa for aa in seq if aa in valid_aa])
    
    valid_indices = []
    for i in range(len(df)):
        if (clean_seq(light_seqs[i]) and 
            clean_seq(heavy_seqs[i]) and 
            clean_seq(antigen_seqs[i])):
            valid_indices.append(i)
    
    light_seqs = [light_seqs[i] for i in valid_indices]
    heavy_seqs = [heavy_seqs[i] for i in valid_indices]
    antigen_seqs = [antigen_seqs[i] for i in valid_indices]
    delta_g_values = [delta_g_values[i] for i in valid_indices]
    
    print(f"有效数据量：{len(valid_indices)} 条")
    print(f"是否包含真实值：{'是' if has_delta_g else '否'}")
    
    # ========== 4. 序列编码 ==========
    print("\n" + "="*60)
    print("🧬 序列编码 (ESM2)")
    print("="*60)
    
    encoder = SequenceEncoder(use_esm=use_esm)
    
    print("编码重链 (antibody_seq_b)...")
    X_heavy = encoder.encode_batch(heavy_seqs, target_dim=532)
    
    print("编码轻链 (antibody_seq_a)...")
    X_light = encoder.encode_batch(light_seqs, target_dim=532)
    
    print("编码抗原 (antigen_seq)...")
    X_antigen = encoder.encode_batch(antigen_seqs, target_dim=500)
    
    # ========== 5. 创建 DataLoader ==========
    print("\n" + "="*60)
    print("📦 创建 DataLoader")
    print("="*60)
    
    class PredDataset(Dataset):
        def __init__(self, X_h, X_l, X_a, y):
            self.X_h = X_h
            self.X_l = X_l
            self.X_a = X_a
            self.y = y
        
        def __len__(self):
            return len(self.X_h)
        
        def __getitem__(self, idx):
            return (
                torch.tensor(self.X_h[idx], dtype=torch.float32),
                torch.tensor(self.X_l[idx], dtype=torch.float32),
                torch.tensor(self.X_a[idx], dtype=torch.float32),
                torch.tensor(self.y[idx], dtype=torch.float32)
            )
    
    dataset = PredDataset(X_heavy, X_light, X_antigen, delta_g_values)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # ========== 6. 执行预测 ==========
    print("\n" + "="*60)
    print("🔮 执行预测")
    print("="*60)
    
    all_preds_scaled = []
    
    with torch.no_grad():
        for heavy, light, antigen, delta_g in tqdm(data_loader, desc="Predicting"):
            heavy = heavy.to(device)
            light = light.to(device)
            antigen = antigen.to(device)
            
            preds = model(heavy, light, antigen)
            all_preds_scaled.extend(preds.cpu().numpy())
    
    all_preds_scaled = np.array(all_preds_scaled)
    
    # ========== 7. 反标准化 ==========
    print("\n" + "="*60)
    print("🔄 反标准化")
    print("="*60)
    
    all_preds_orig = label_scaler.inverse_transform(all_preds_scaled.reshape(-1, 1)).flatten()
    
    # ========== 8. 保存结果 ==========
    print("\n" + "="*60)
    print("💾 保存结果")
    print("="*60)
    
    # 创建结果 DataFrame
    result_df = pd.DataFrame({
        'Index': list(range(len(all_preds_orig))),
        'delta_g': delta_g_values,
        'predicted_delta_g': all_preds_orig
    })
    
    # 确保输出目录存在
    output_dir = os.path.dirname(os.path.abspath(output_csv))
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存 CSV
    result_df.to_csv(output_csv, index=False)
    
    print(f"✅ 结果已保存：{output_csv}")
    print(f"   总样本数：{len(result_df)}")
    
    # ========== 9. 统计信息 ==========
    print("\n" + "="*60)
    print("📊 预测统计")
    print("="*60)
    print(f"   预测值范围：[{all_preds_orig.min():.4f}, {all_preds_orig.max():.4f}]")
    print(f"   预测值均值：{all_preds_orig.mean():.4f}")
    print(f"   预测值标准差：{all_preds_orig.std():.4f}")
    
    if has_delta_g:
        from scipy.stats import pearsonr
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        
        y_true = np.array(delta_g_values)
        y_pred = all_preds_orig
        
        pcc, _ = pearsonr(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        
        print(f"\n   ✅ 评估指标:")
        print(f"   PCC:  {pcc:.4f}")
        print(f"   MAE:  {mae:.4f}")
        print(f"   RMSE: {rmse:.4f}")
        print(f"   R²:   {r2:.4f}")
    
    print("\n" + "="*60)
    print("✅ 预测完成！")
    print("="*60)
    
    return result_df


# ============================================================================
# 命令行入口
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="AttABseq 预测脚本")
    parser.add_argument("--input", type=str, 
                       default="/tmp/AbAgCDR/data/pairs_seq_skempi.tsv",
                       help="输入 TSV 文件路径")
    parser.add_argument("--model", type=str, 
                       default="/tmp/AbAgCDR/models/runs/best_model.pth",
                       help="模型权重路径 (.pth)")
    parser.add_argument("--scaler", type=str, 
                       default="/tmp/AbAgCDR/models/runs/label_scaler.pkl",
                       help="标准化器路径 (.pkl)")
    parser.add_argument("--output", type=str, 
                       default="/tmp/AbAgCDR/models/runs/AttABseqpredictions_skempi.csv",
                       help="输出 CSV 文件路径")
    parser.add_argument("--batch_size", type=int, default=32,
                       help="批次大小 (默认：32)")
    parser.add_argument("--no_esm", action="store_true",
                       help="不使用 ESM2，使用 One-Hot 编码")
    
    args = parser.parse_args()
    
    predict(
        input_tsv=args.input,
        model_path=args.model,
        scaler_path=args.scaler,
        output_csv=args.output,
        batch_size=args.batch_size,
        use_esm=not args.no_esm
    )