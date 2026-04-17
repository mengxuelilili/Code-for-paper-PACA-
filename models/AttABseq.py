# -*- coding: utf-8 -*-
"""
IMMS-AB 改进版 - 多数据集训练 + 独立测试
输入：antibody_seq_a(轻链), antibody_seq_b(重链), antigen_seq, delta_g
数据集划分：每个数据集 6:2:2 (train:val:test)
Benchmark：单独测试集，不参与训练

修复：Encoder 输出维度投影 (384 → 128)
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
import pickle
from tqdm import tqdm
import copy

# 尝试导入 ESM2
try:
    import esm
    ESM_AVAILABLE = True
    print("✅ ESM2 可用")
except ImportError:
    ESM_AVAILABLE = False
    print("⚠️ ESM2 不可用，将使用 One-Hot 编码")


# ============================================================================
# 随机种子
# ============================================================================

def set_seed(seed=42):
    """设置随机种子"""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================================
# 序列编码模块
# ============================================================================

class SequenceEncoder:
    """序列编码器（支持 ESM2 或 One-Hot）"""
    
    def __init__(self, use_esm=True, max_length=1024):
        self.use_esm = use_esm and ESM_AVAILABLE
        self.max_length = max_length
        self.valid_aa = "ACDEFGHIKLMNPQRSTVWY"
        
        if self.use_esm:
            print("🧠 加载 ESM2 模型...")
            self.model, self.alphabet = esm.pretrained.esm2_t12_35M_UR50D()
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model = self.model.to(self.device)
            self.model.eval()
            self.batch_converter = self.alphabet.get_batch_converter()
            self.embed_dim = 1280
            print(f"✅ ESM2 加载完成 (维度：{self.embed_dim}, 设备：{self.device})")
        else:
            self.embed_dim = 20
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"⚠️ 使用 One-Hot 编码 (维度：{self.embed_dim})")
    
    def _clean_seq(self, seq):
        """清洗序列"""
        seq = str(seq).upper().strip()
        return ''.join([aa for aa in seq if aa in self.valid_aa])
    
    def get_embedding(self, seq):
        """获取单条序列嵌入"""
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
        """批量编码序列"""
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
# 数据集类
# ============================================================================

class AntibodyDataset(Dataset):
    """抗体 - 抗原数据集"""
    
    def __init__(self, samples, label_scaler=None):
        self.samples = samples
        self.label_scaler = label_scaler
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # 支持 4 或 5 元组
        if len(sample) == 4:
            heavy, light, antigen, delta_g = sample
            delta_g_scaled = delta_g
        elif len(sample) == 5:
            heavy, light, antigen, delta_g, delta_g_scaled = sample
        else:
            raise ValueError(f"样本格式错误，期望 4 或 5 个值，实际 {len(sample)} 个")
        
        return {
            'heavy': torch.tensor(heavy, dtype=torch.float32),
            'light': torch.tensor(light, dtype=torch.float32),
            'antigen': torch.tensor(antigen, dtype=torch.float32),
            'delta_g': torch.tensor(delta_g, dtype=torch.float32),
            'delta_g_scaled': torch.tensor(delta_g_scaled, dtype=torch.float32)
        }


def collate_fn(batch):
    """DataLoader 的 collate 函数"""
    heavy = torch.stack([x['heavy'] for x in batch])
    light = torch.stack([x['light'] for x in batch])
    antigen = torch.stack([x['antigen'] for x in batch])
    delta_g = torch.stack([x['delta_g'] for x in batch])
    delta_g_scaled = torch.stack([x['delta_g_scaled'] for x in batch])
    return heavy, light, antigen, delta_g, delta_g_scaled


# ============================================================================
# 数据加载函数
# ============================================================================

def load_tsv_data(tsv_path):
    """加载 TSV 数据"""
    print(f"📥 加载：{tsv_path}")
    df = pd.read_csv(tsv_path, sep='\t')
    
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
    
    for name in ['delta_g', 'ddg', 'deltag', 'binding_affinity']:
        if name in cols_lower:
            col_map['delta_g'] = df.columns[cols_lower.index(name)]
            break
    
    required = ['light', 'heavy', 'antigen', 'delta_g']
    if not all(k in col_map for k in required):
        raise ValueError(f"缺少必要列，需要：{required}")
    
    df = df[[col_map['light'], col_map['heavy'], col_map['antigen'], col_map['delta_g']]].copy()
    df.columns = ['antibody_seq_a', 'antibody_seq_b', 'antigen_seq', 'delta_g']
    
    valid_aa = set("ACDEFGHIKLMNPQRSTVWY")
    def is_valid_seq(seq):
        if not isinstance(seq, str) or len(seq.strip()) == 0:
            return False
        return all(c in valid_aa for c in seq.upper())
    
    before = len(df)
    df = df[
        df['antibody_seq_a'].apply(is_valid_seq) &
        df['antibody_seq_b'].apply(is_valid_seq) &
        df['antigen_seq'].apply(is_valid_seq)
    ]
    df['delta_g'] = pd.to_numeric(df['delta_g'], errors='coerce')
    df = df.dropna(subset=['delta_g'])
    after = len(df)
    
    print(f"   清洗后：{after} / {before} 条有效数据")
    
    return df


def split_dataset(df, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, seed=42):
    """划分数据集为 train/val/test (6:2:2)"""
    train_val_df, test_df = train_test_split(df, test_size=test_ratio, random_state=seed)
    val_ratio_adjusted = val_ratio / (train_ratio + val_ratio)
    train_df, val_df = train_test_split(train_val_df, test_size=val_ratio_adjusted, random_state=seed)
    return train_df, val_df, test_df


# ============================================================================
# 模型架构（✅ 已修复维度问题）
# ============================================================================

class SelfAttention(nn.Module):
    """自注意力机制"""
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
    """
    蛋白质特征提取（多尺度 CNN + 残差）
    ✅ 修复：添加投影层将 384 维映射回 hid_dim
    """
    def __init__(self, protein_dim, hid_dim, n_layers, kernel_size, dropout, device):
        super().__init__()
        
        assert kernel_size % 2 == 1, "Kernel size must be odd"
        
        self.input_dim = protein_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.device = device
        self.scale = torch.sqrt(torch.FloatTensor([0.5])).to(device)
        
        # 多尺度 CNN (3, 5, 7)
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
        
        # ✅ 修复：投影层将 3*hid_dim 映射回 hid_dim
        self.project = nn.Linear(hid_dim * 3, hid_dim)
        self.ln = nn.LayerNorm(hid_dim)
    
    def forward(self, protein):
        # protein = [batch, seq_len, protein_dim]
        conv_input = self.fc(protein)  # [batch, seq_len, hid_dim]
        
        # 转换为 Conv1d 输入格式
        conv_input1 = conv_input.permute(0, 2, 1)
        conv_input2 = conv_input.permute(0, 2, 1)
        conv_input3 = conv_input.permute(0, 2, 1)
        
        # 多尺度卷积
        for conv in self.convs1:
            conved = (F.glu(conv(self.dropout(conv_input1)), dim=1) + conv_input1) * self.scale
            conv_input1 = conved
        
        for conv in self.convs2:
            conved = (F.glu(conv(self.dropout(conv_input2)), dim=1) + conv_input2) * self.scale
            conv_input2 = conved
        
        for conv in self.convs3:
            conved = (F.glu(conv(self.dropout(conv_input3)), dim=1) + conv_input3) * self.scale
            conv_input3 = conved
        
        # 拼接多尺度特征：[batch, 3*hid_dim, seq_len]
        conved = torch.cat((conv_input1, conv_input2, conv_input3), 1)
        conved = conved.permute(0, 2, 1)  # [batch, seq_len, 3*hid_dim]
        
        # ✅ 修复：投影回 hid_dim
        conved = self.project(conved)  # [batch, seq_len, hid_dim]
        conved = self.ln(conved)
        
        return conved


class DecoderLayer(nn.Module):
    """解码器层"""
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
    """位置前馈网络"""
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


class IMMSABModel(nn.Module):
    """IMMS-AB 模型"""
    def __init__(self, 
                 heavy_dim=532, 
                 light_dim=532, 
                 antigen_dim=500,
                 hid_dim=128, 
                 n_layers=3, 
                 n_heads=8, 
                 pf_dim=256,
                 dropout=0.2,
                 device=None):
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
        
        self.fc_agg = nn.Linear(hid_dim * 2, hid_dim)  # 修复：2 条链拼接
        self.fc1 = nn.Linear(hid_dim, hid_dim // 2)
        self.fc2 = nn.Linear(hid_dim // 2, 64)
        self.fc3 = nn.Linear(64, 16)
        self.fc_out = nn.Linear(16, 1)
        
        self.dropout = nn.Dropout(dropout)
        self.gn = nn.GroupNorm(8, 64)
    
    def _make_mask(self, seq_len):
        mask = torch.ones((seq_len, seq_len), device=self.device)
        return mask
    
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
        # heavy, light, antigen = [batch, feature_dim]
        
        # 添加序列维度用于 Encoder: [batch, 1, feature_dim]
        heavy = heavy.unsqueeze(1)
        light = light.unsqueeze(1)
        antigen = antigen.unsqueeze(1)
        
        # 编码：[batch, 1, hid_dim]
        enc_heavy = self.heavy_encoder(heavy)
        enc_light = self.light_encoder(light)
        enc_antigen = self.antigen_encoder(antigen)
        
        # 抗体特征融合：[batch, 1, hid_dim]
        enc_ab = (enc_heavy + enc_light) / 2
        
        # 创建掩码
        ab_mask = self._make_mask(enc_ab.shape[1]).unsqueeze(0).unsqueeze(3)
        ag_mask = self._make_mask(enc_antigen.shape[1]).unsqueeze(0).unsqueeze(3)
        
        # 抗体 - 抗原交互
        ag_ab = self._decode(self.ag_ab_decoder, enc_antigen, enc_ab, ag_mask, ab_mask)
        ab_ag = self._decode(self.ab_ag_decoder, enc_ab, enc_antigen, ab_mask, ag_mask)
        
        # 全局池化：[batch, hid_dim]
        ag_ab_pool = self._global_pool(ag_ab)
        ab_ag_pool = self._global_pool(ab_ag)
        
        # 拼接：[batch, hid_dim * 2]
        x = torch.cat([ag_ab_pool, ab_ag_pool], dim=-1)
        
        # 全连接预测
        x = self.dropout(F.relu(self.fc_agg(x)))
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.gn(x)
        x = self.dropout(F.relu(self.fc3(x)))
        x = self.fc_out(x)  # [batch, 1]
        
        return x.squeeze(-1)


# ============================================================================
# 训练器
# ============================================================================

class Trainer:
    """训练器"""
    def __init__(self, model, lr=1e-4, weight_decay=1e-5, device=None):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.model = model.to(device)
        
        self.optimizer = optim.AdamW(
            model.parameters(), 
            lr=lr, 
            weight_decay=weight_decay
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6
        )
        
        self.criterion = nn.HuberLoss(delta=1.0)
    
    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        for heavy, light, antigen, delta_g, delta_g_scaled in tqdm(train_loader, desc="Training", leave=False):
            heavy = heavy.to(self.device)
            light = light.to(self.device)
            antigen = antigen.to(self.device)
            delta_g_scaled = delta_g_scaled.to(self.device)
            
            self.optimizer.zero_grad()
            preds = self.model(heavy, light, antigen)
            loss = self.criterion(preds, delta_g_scaled)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            all_preds.extend(preds.detach().cpu().numpy())
            all_labels.extend(delta_g_scaled.detach().cpu().numpy())
        
        avg_loss = total_loss / len(train_loader)
        pcc, _ = pearsonr(all_preds, all_labels)
        
        return avg_loss, pcc
    
    def evaluate(self, data_loader, label_scaler=None):
        """评估"""
        self.model.eval()
        total_loss = 0
        all_preds_scaled = []
        all_labels_scaled = []
        all_labels_orig = []
        
        with torch.no_grad():
            for heavy, light, antigen, delta_g, delta_g_scaled in tqdm(data_loader, desc="Evaluating", leave=False):
                heavy = heavy.to(self.device)
                light = light.to(self.device)
                antigen = antigen.to(self.device)
                delta_g_scaled = delta_g_scaled.to(self.device)
                
                preds = self.model(heavy, light, antigen)
                loss = self.criterion(preds, delta_g_scaled)
                
                total_loss += loss.item()
                all_preds_scaled.extend(preds.cpu().numpy())
                all_labels_scaled.extend(delta_g_scaled.cpu().numpy())
                all_labels_orig.extend(delta_g.cpu().numpy())
        
        avg_loss = total_loss / len(data_loader)
        
        all_preds_scaled = np.array(all_preds_scaled)
        all_labels_scaled = np.array(all_labels_scaled)
        pcc_scaled, _ = pearsonr(all_preds_scaled, all_labels_scaled)
        mae_scaled = mean_absolute_error(all_labels_scaled, all_preds_scaled)
        rmse_scaled = np.sqrt(mean_squared_error(all_labels_scaled, all_preds_scaled))
        r2_scaled = r2_score(all_labels_scaled, all_preds_scaled)
        
        if label_scaler is not None:
            all_preds_orig = label_scaler.inverse_transform(all_preds_scaled.reshape(-1, 1)).flatten()
            all_labels_orig = np.array(all_labels_orig)
            pcc_orig, _ = pearsonr(all_labels_orig, all_preds_orig)
            mae_orig = mean_absolute_error(all_labels_orig, all_preds_orig)
            rmse_orig = np.sqrt(mean_squared_error(all_labels_orig, all_preds_orig))
            r2_orig = r2_score(all_labels_orig, all_preds_orig)
        else:
            pcc_orig, mae_orig, rmse_orig, r2_orig = pcc_scaled, mae_scaled, rmse_scaled, r2_scaled
        
        return {
            'loss': avg_loss,
            'PCC': pcc_orig,
            'MAE': mae_orig,
            'RMSE': rmse_orig,
            'R2': r2_orig,
            'MSE': rmse_orig ** 2,
            'preds': all_preds_orig if label_scaler else all_preds_scaled,
            'labels': all_labels_orig
        }


# ============================================================================
# 主程序
# ============================================================================

def main():
    set_seed(42)
    
    TRAIN_TSVS = {
        "final_dataset_train": ("/tmp/AbAgCDR/data/final_dataset_train.tsv", 1.0),
        "skempi": ("/tmp/AbAgCDR/data/pairs_seq_skempi_clean.tsv", 1.0),
        "sabdab": ("/tmp/AbAgCDR/data/pairs_seq_sabdab_clean.tsv", 1.0),
        "abbind": ("/tmp/AbAgCDR/data/pairs_seq_abbind2_clean.tsv", 1.0),
    }
    
    # BENCHMARK_TSV = "/tmp/AbAgCDR/data/pairs_seq_benchmark1.tsv"
    
    CONFIG = {
        "train_ratio": 0.6,
        "val_ratio": 0.2,
        "test_ratio": 0.2,
        "seed": 42,
        "run_dir": "/tmp/AbAgCDR/models/runs",
        "use_esm": True,
        "hidden_dim": 128,
        "n_layers": 3,
        "n_heads": 8,
        "dropout": 0.2,
        "batch_size": 32,
        "n_epochs": 50,
        "learning_rate": 1e-4,
        "weight_decay": 1e-5,
        "patience": 10,
    }
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  使用设备：{device}")
    
    os.makedirs(CONFIG["run_dir"], exist_ok=True)
    model_save_path = os.path.join(CONFIG["run_dir"], "best_model.pth")
    scaler_save_path = os.path.join(CONFIG["run_dir"], "label_scaler.pkl")
    results_path = os.path.join(CONFIG["run_dir"], "results.txt")
    
    # ========================================================================
    # 1. 加载并划分所有训练数据集
    # ========================================================================
    print("\n" + "="*70)
    print("📊 加载并划分数据集 (6:2:2)")
    print("="*70)
    
    all_train_samples = []
    all_val_samples = []
    all_test_samples = {}
    
    for name, (tsv_path, weight) in TRAIN_TSVS.items():
        print(f"\n{'='*50}")
        print(f"处理数据集：{name}")
        print(f"{'='*50}")
        
        if not os.path.exists(tsv_path):
            print(f"⚠️  跳过 {name}: 文件不存在")
            continue
        
        df = load_tsv_data(tsv_path)
        train_df, val_df, test_df = split_dataset(
            df, 
            train_ratio=CONFIG["train_ratio"],
            val_ratio=CONFIG["val_ratio"],
            test_ratio=CONFIG["test_ratio"],
            seed=CONFIG["seed"]
        )
        
        print(f"   划分结果：train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
        
        all_test_samples[name] = {
            'df': test_df,
            'weight': weight,
            'samples': None
        }
        
        for _, row in train_df.iterrows():
            all_train_samples.append((
                row['antibody_seq_b'],
                row['antibody_seq_a'],
                row['antigen_seq'],
                row['delta_g'],
                weight
            ))
        
        for _, row in val_df.iterrows():
            all_val_samples.append((
                row['antibody_seq_b'],
                row['antibody_seq_a'],
                row['antigen_seq'],
                row['delta_g']
            ))
    
    print(f"\n✅ 总训练样本：{len(all_train_samples)}")
    print(f"✅ 总验证样本：{len(all_val_samples)}")
    
    # # ========================================================================
    # # 2. 加载 Benchmark 数据集
    # # ========================================================================
    # print("\n" + "="*70)
    # print("📊 加载 Benchmark 数据集")
    # print("="*70)
    
    # benchmark_data = None
    # if os.path.exists(BENCHMARK_TSV):
    #     df = load_tsv_data(BENCHMARK_TSV)
    #     benchmark_data = {
    #         'df': df,
    #         'samples': None
    #     }
    #     print(f"✅ Benchmark 样本数：{len(df)}")
    # else:
    #     print(f"⚠️  Benchmark 文件不存在：{BENCHMARK_TSV}")
    
    # ========================================================================
    # 3. 标签标准化
    # ========================================================================
    print("\n" + "="*70)
    print("📈 标签标准化 (RobustScaler)")
    print("="*70)
    
    train_labels = np.array([s[3] for s in all_train_samples]).reshape(-1, 1)
    
    label_scaler = RobustScaler()
    label_scaler.fit(train_labels)
    
    print(f"   标准化器均值：{label_scaler.center_[0]:.4f}")
    print(f"   标准化器尺度：{label_scaler.scale_[0]:.4f}")
    print(f"   原始标签范围：[{train_labels.min():.4f}, {train_labels.max():.4f}]")
    
    with open(scaler_save_path, 'wb') as f:
        pickle.dump(label_scaler, f)
    print(f"   ✅ 标准化器已保存：{scaler_save_path}")
    
    # ========================================================================
    # 4. 序列编码
    # ========================================================================
    print("\n" + "="*70)
    print("🧬 序列编码 (ESM2)")
    print("="*70)
    
    encoder = SequenceEncoder(use_esm=CONFIG["use_esm"], max_length=1024)
    
    def encode_samples(samples, label_scaler=None):
        """编码样本 - 统一输出格式为 (heavy, light, antigen, delta_g, delta_g_scaled)"""
        heavies = [s[0] for s in samples]
        lights = [s[1] for s in samples]
        antigens = [s[2] for s in samples]
        labels = [s[3] for s in samples]
        
        print("   编码重链...")
        X_h = encoder.encode_batch(heavies, target_dim=532)
        print("   编码轻链...")
        X_l = encoder.encode_batch(lights, target_dim=532)
        print("   编码抗原...")
        X_a = encoder.encode_batch(antigens, target_dim=500)
        
        labels_scaled = label_scaler.transform(np.array(labels).reshape(-1, 1)).flatten()
        
        encoded = [(X_h[i], X_l[i], X_a[i], labels[i], labels_scaled[i]) 
                   for i in range(len(labels))]
        
        return encoded
    
    print("\n编码训练集...")
    train_samples_encoded = encode_samples(all_train_samples, label_scaler)
    
    print("编码验证集...")
    val_samples_encoded = encode_samples(all_val_samples, label_scaler)
    
    for name, data in all_test_samples.items():
        print(f"编码 {name} 测试集...")
        samples = [(row['antibody_seq_b'], row['antibody_seq_a'], row['antigen_seq'], row['delta_g']) 
                   for _, row in data['df'].iterrows()]
        data['samples'] = encode_samples(samples, label_scaler)
    
    # if benchmark_data:
    #     print("编码 Benchmark...")
    #     samples = [(row['antibody_seq_b'], row['antibody_seq_a'], row['antigen_seq'], row['delta_g']) 
    #                for _, row in benchmark_data['df'].iterrows()]
    #     benchmark_data['samples'] = encode_samples(samples, label_scaler)
    
    # ========================================================================
    # 5. 创建 DataLoader
    # ========================================================================
    print("\n" + "="*70)
    print("📦 创建 DataLoader")
    print("="*70)
    
    train_dataset = AntibodyDataset(train_samples_encoded, label_scaler=None)
    
    train_weights = [s[4] for s in all_train_samples]
    train_weights_tensor = torch.tensor(train_weights, dtype=torch.float32)
    sampler = torch.utils.data.WeightedRandomSampler(
        train_weights_tensor, 
        len(train_weights_tensor), 
        replacement=True
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=CONFIG["batch_size"], 
        sampler=sampler,
        collate_fn=collate_fn,
        num_workers=0
    )
    
    val_dataset = AntibodyDataset(val_samples_encoded, label_scaler=None)
    val_loader = DataLoader(
        val_dataset, 
        batch_size=CONFIG["batch_size"], 
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )
    
    test_loaders = {}
    for name, data in all_test_samples.items():
        test_dataset = AntibodyDataset(data['samples'], label_scaler=None)
        test_loaders[name] = DataLoader(
            test_dataset, 
            batch_size=CONFIG["batch_size"], 
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0
        )
    
    bench_loader = None
    # if benchmark_data:
    #     bench_dataset = AntibodyDataset(benchmark_data['samples'], label_scaler=None)
    #     bench_loader = DataLoader(
    #         bench_dataset, 
    #         batch_size=CONFIG["batch_size"], 
    #         shuffle=False,
    #         collate_fn=collate_fn,
    #         num_workers=0
    #     )
    
    # ========================================================================
    # 6. 创建模型
    # ========================================================================
    print("\n" + "="*70)
    print("🏗️ 创建模型")
    print("="*70)
    
    model = IMMSABModel(
        heavy_dim=532,
        light_dim=532,
        antigen_dim=500,
        hid_dim=CONFIG["hidden_dim"],
        n_layers=CONFIG["n_layers"],
        n_heads=CONFIG["n_heads"],
        dropout=CONFIG["dropout"],
        device=device
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   总参数量：{total_params:,}")
    print(f"   可训练参数：{trainable_params:,}")
    
    # ========================================================================
    # 7. 训练
    # ========================================================================
    print("\n" + "="*70)
    print("🏃 开始训练")
    print("="*70)
    
    trainer = Trainer(
        model, 
        lr=CONFIG["learning_rate"], 
        weight_decay=CONFIG["weight_decay"],
        device=device
    )
    
    best_val_pcc = -np.inf
    best_model_state = None
    patience_counter = 0
    
    for epoch in range(CONFIG["n_epochs"]):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch+1}/{CONFIG['n_epochs']}")
        print(f"{'='*50}")
        
        train_loss, train_pcc = trainer.train_epoch(train_loader)
        val_metrics = trainer.evaluate(val_loader, label_scaler)
        
        trainer.scheduler.step(val_metrics['loss'])
        
        print(f"   Train Loss: {train_loss:.4f}, Train PCC: {train_pcc:.4f}")
        print(f"   Val Loss: {val_metrics['loss']:.4f}, Val PCC: {val_metrics['PCC']:.4f}")
        print(f"   Val MAE: {val_metrics['MAE']:.4f}, Val RMSE: {val_metrics['RMSE']:.4f}, Val R²: {val_metrics['R2']:.4f}")
        
        if val_metrics['PCC'] > best_val_pcc:
            best_val_pcc = val_metrics['PCC']
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
            
            torch.save({
                'model_state_dict': best_model_state,
                'epoch': epoch,
                'val_pcc': val_metrics['PCC'],
                'config': CONFIG
            }, model_save_path)
            print(f"   🎉 新最佳模型已保存 (Val PCC: {val_metrics['PCC']:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= CONFIG["patience"]:
                print(f"   ⏹️  早停 (patience={patience_counter})")
                break
    
    # ========================================================================
    # 8. 测试评估
    # ========================================================================
    print("\n" + "="*70)
    print("🧪 测试评估")
    print("="*70)
    
    model.load_state_dict(best_model_state)
    trainer = Trainer(model, lr=CONFIG["learning_rate"], device=device)
    
    test_results = {}
    for name, loader in test_loaders.items():
        print(f"\n评估 {name} 测试集...")
        metrics = trainer.evaluate(loader, label_scaler)
        test_results[name] = metrics
        
        print(f"   {name.upper()} TEST → "
              f"R²: {metrics['R2']:.4f}, "
              f"MSE: {metrics['MSE']:.4f}, "
              f"RMSE: {metrics['RMSE']:.4f}, "
              f"MAE: {metrics['MAE']:.4f}, "
              f"PCC: {metrics['PCC']:.4f}")
    
    bench_metrics = None
    if bench_loader:
        print(f"\n评估 Benchmark 测试集...")
        bench_metrics = trainer.evaluate(bench_loader, label_scaler)
        
        print(f"\n🎯 BENCHMARK TEST → "
              f"R²: {bench_metrics['R2']:.4f}, "
              f"MSE: {bench_metrics['MSE']:.4f}, "
              f"RMSE: {bench_metrics['RMSE']:.4f}, "
              f"MAE: {bench_metrics['MAE']:.4f}, "
              f"PCC: {bench_metrics['PCC']:.4f}")
    
    # ========================================================================
    # 9. 保存结果
    # ========================================================================
    print("\n" + "="*70)
    print("💾 保存结果")
    print("="*70)
    
    with open(results_path, 'w') as f:
        f.write("IMMS-AB 测试结果\n")
        f.write("="*70 + "\n\n")
        
        f.write("验证集最佳 PCC: {:.4f}\n\n".format(best_val_pcc))
        
        f.write("各数据集测试集指标:\n")
        f.write("-"*70 + "\n")
        f.write(f"{'Dataset':<20} {'PCC':<10} {'R²':<10} {'RMSE':<10} {'MAE':<10} {'MSE':<10}\n")
        f.write("-"*70 + "\n")
        
        for name, metrics in test_results.items():
            f.write(f"{name:<20} {metrics['PCC']:<10.4f} {metrics['R2']:<10.4f} "
                   f"{metrics['RMSE']:<10.4f} {metrics['MAE']:<10.4f} {metrics['MSE']:<10.4f}\n")
        
        if bench_metrics:
            f.write("-"*70 + "\n")
            f.write(f"{'BENCHMARK':<20} {bench_metrics['PCC']:<10.4f} {bench_metrics['R2']:<10.4f} "
                   f"{bench_metrics['RMSE']:<10.4f} {bench_metrics['MAE']:<10.4f} {bench_metrics['MSE']:<10.4f}\n")
        
        f.write("="*70 + "\n")
        
        for name, metrics in test_results.items():
            f.write(f"\n{name}:\n")
            f.write("True\tPred\n")
            for t, p in zip(metrics['labels'], metrics['preds']):
                f.write(f"{t:.4f}\t{p:.4f}\n")
        
        if bench_metrics:
            f.write(f"\nBENCHMARK:\n")
            f.write("True\tPred\n")
            for t, p in zip(bench_metrics['labels'], bench_metrics['preds']):
                f.write(f"{t:.4f}\t{p:.4f}\n")
    
    print(f"   ✅ 结果已保存：{results_path}")
    print(f"   ✅ 模型已保存：{model_save_path}")
    print(f"   ✅ 标准化器已保存：{scaler_save_path}")
    
    print("\n" + "="*70)
    print("✅ 训练完成！")
    print("="*70)
    print(f"\n📊 最终结果汇总:")
    print(f"{'='*70}")
    print(f"{'Dataset':<20} {'PCC':<10} {'R²':<10} {'RMSE':<10} {'MSE':<10} {'MAE':<10}")
    print(f"{'='*70}")
    
    for name, metrics in test_results.items():
        print(f"{name:<20} {metrics['PCC']:<10.4f} {metrics['R2']:<10.4f} "
              f"{metrics['RMSE']:<10.4f} {metrics['MSE']:<10.4f} {metrics['MAE']:<10.4f}")
    
    if bench_metrics:
        print(f"{'='*70}")
        print(f"{'BENCHMARK':<20} {bench_metrics['PCC']:<10.4f} {bench_metrics['R2']:<10.4f} "
              f"{bench_metrics['RMSE']:<10.4f} {bench_metrics['MSE']:<10.4f} {bench_metrics['MAE']:<10.4f}")
    
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
