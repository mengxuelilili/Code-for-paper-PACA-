# model.py
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from transformers import AutoModel


# class HuberLoss(nn.Module):
#     def __init__(self, delta=1.0):
#         super().__init__()
#         self.delta = delta

#     def forward(self, pred, target):
#         pred = pred.view(-1)
#         target = target.view(-1)
#         diff = pred - target
#         abs_diff = torch.abs(diff)
#         quadratic = 0.5 * diff ** 2
#         linear = self.delta * abs_diff - 0.5 * self.delta ** 2
#         loss = torch.where(abs_diff <= self.delta, quadratic, linear)
#         return loss.mean()


# class LogCoshLoss(nn.Module):
#     def forward(self, pred, target):
#         return torch.mean(torch.log(torch.cosh(pred - target)))


# class AntibodyAffinityModel(nn.Module):
#     def __init__(self, ab_embed_dim=532, ag_embed_dim=500, cnn_out_channels=64, dropout=0.3):
#         super().__init__()

#         def make_cnn(in_dim, out_channels=cnn_out_channels):
#             return nn.Sequential(
#                 nn.Conv1d(in_dim, out_channels, kernel_size=5, padding=2),
#                 nn.ReLU(),
#                 nn.AdaptiveMaxPool1d(1)
#             )

#         # 抗体（heavy/light）使用 ab_embed_dim=532
#         self.cnn_heavy = make_cnn(ab_embed_dim)
#         self.cnn_light = make_cnn(ab_embed_dim)
#         # 抗原（antigen）使用 ag_embed_dim=500
#         self.cnn_antigen = make_cnn(ag_embed_dim)

#         self.fc = nn.Sequential(
#             nn.Linear(cnn_out_channels * 3, 512),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(512, 1)
#         )

#     def forward(self, heavy, light, antigen):
#         # heavy, light: [B, L, 532] → permute to [B, 532, L]
#         # antigen:       [B, L, 500] → permute to [B, 500, L]
#         h = self.cnn_heavy(heavy.permute(0, 2, 1))      # [B, 64, 1]
#         l = self.cnn_light(light.permute(0, 2, 1))      # [B, 64, 1]
#         g = self.cnn_antigen(antigen.permute(0, 2, 1))  # [B, 64, 1]

#         out = torch.cat([h.squeeze(-1), l.squeeze(-1), g.squeeze(-1)], dim=1)  # [B, 192]
#         return self.fc(out).squeeze(-1)

import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from torch import optim
import torchmetrics
from transformers import EsmTokenizer, EsmModel
from typing import List, Tuple, Dict
import pickle

# === 随机种子 ===
def set_seed(seed=22):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# === 损失函数 ===
class HuberLoss(nn.Module):
    def __init__(self, delta=1.0):
        super().__init__()
        self.delta = delta
    
    def forward(self, pred, target):
        pred = pred.view(-1, 1)
        target = target.view(-1, 1)
        diff = pred - target
        abs_diff = torch.abs(diff)
        condition = abs_diff <= self.delta
        quadratic = 0.5 * diff ** 2
        linear = self.delta * abs_diff - 0.5 * self.delta ** 2
        loss = torch.where(condition, quadratic, linear)
        return loss.mean()


class LogCoshLoss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, pred, target):
        loss = torch.log(torch.cosh(pred - target))
        return torch.mean(loss)


# === 数据集类 ===
class AntibodyAntigenDataset(Dataset):
    """抗体 - 抗原序列数据集"""
    
    def __init__(self, df: pd.DataFrame, tokenizer, max_length: int = 1024):
        self.df = df
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # 确保列存在
        required_cols = ['heavy', 'light', 'antigen', 'delta_g']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"缺少列：{col}")
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # 获取序列
        heavy_seq = str(row['heavy']).upper().strip()
        light_seq = str(row['light']).upper().strip()
        antigen_seq = str(row['antigen']).upper().strip()
        delta_g = float(row['delta_g'])
        
        # Tokenize (ESM使用字母表，不需要传统tokenizer)
        # 这里返回原始序列，在collate_fn中处理
        return {
            'heavy': heavy_seq,
            'light': light_seq,
            'antigen': antigen_seq,
            'delta_g': delta_g
        }


def collate_fn(batch: List[Dict], alphabet, max_length: int = 1024):
    """
    自定义collate函数：将序列批量转换为ESM输入
    """
    heavy_seqs = [item['heavy'] for item in batch]
    light_seqs = [item['light'] for item in batch]
    antigen_seqs = [item['antigen'] for item in batch]
    delta_g = torch.tensor([item['delta_g'] for item in batch], dtype=torch.float32)
    
    # 使用ESM字母表转换
    batch_converter = alphabet.get_batch_converter()
    
    def process_seqs(seqs):
        data = [(f"seq_{i}", seq) for i, seq in enumerate(seqs)]
        _, _, batch_tokens = batch_converter(data)
        # 截断/填充到max_length
        if batch_tokens.shape[1] > max_length:
            batch_tokens = batch_tokens[:, :max_length, :]
        return batch_tokens
    
    heavy_tokens = process_seqs(heavy_seqs)
    light_tokens = process_seqs(light_seqs)
    antigen_tokens = process_seqs(antigen_seqs)
    
    return heavy_tokens, light_tokens, antigen_tokens, delta_g


# === 模型 ===
class ClassifierNet(pl.LightningModule):
    def __init__(self, 
                 esm_model_name: str = "facebook/esm2_t12_35M_UR50D",
                 hidden_dim: int = 128,
                 learning_rate: float = 1e-4,
                 max_length: int = 1024):
        super(ClassifierNet, self).__init__()
        self.save_hyperparameters()
        
        set_seed(22)
        
        # 加载ESM2模型和字母表
        print(f"🧠 加载 ESM2 模型：{esm_model_name}")
        self.esm, self.alphabet = EsmModel.from_pretrained(esm_model_name), None
        # 获取字母表
        from transformers import AutoTokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(esm_model_name)
        except:
            self.tokenizer = None
        
        # 使用fair-esm的字母表
        import esm
        self.esm_model, self.alphabet = esm.pretrained.esm2_t12_35M_UR50D()
        self.esm_antigen, _ = esm.pretrained.esm2_t12_35M_UR50D()
        
        # 冻结ESM参数 (可选)
        for param in self.esm_model.parameters():
            param.requires_grad = False
        for param in self.esm_antigen.parameters():
            param.requires_grad = False
        
        self.esm_dim = 1280  # ESM2 t12 35M 的输出维度
        self.max_length = max_length
        
        # 重链特征提取
        self.heavy_layer1 = nn.Sequential(
            nn.Conv1d(self.esm_dim, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.heavy_layer2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        
        # 轻链特征提取
        self.light_layer1 = nn.Sequential(
            nn.Conv1d(self.esm_dim, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.light_layer2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        
        # 抗原特征提取
        self.antigen_layer1 = nn.Sequential(
            nn.Conv1d(self.esm_dim, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.antigen_layer2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        
        # 注意力机制
        self.attention_h = nn.MultiheadAttention(embed_dim=128, num_heads=8, batch_first=True)
        self.attention_l = nn.MultiheadAttention(embed_dim=128, num_heads=8, batch_first=True)
        self.attention_ag = nn.MultiheadAttention(embed_dim=128, num_heads=8, batch_first=True)
        
        # 全连接层
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(128 * 3, 512)  # 3条链拼接
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 1)
        
        # 评估指标
        self.mse = torchmetrics.MeanSquaredError()
        self.mae = torchmetrics.MeanAbsoluteError()
        self.rmse = torchmetrics.MeanSquaredError(squared=False)
        self.r2score = torchmetrics.R2Score()
        self.pearson = torchmetrics.PearsonCorrCoef()
        
        self.learning_rate = learning_rate
    
    def forward(self, heavy_tokens, light_tokens, antigen_tokens):
        """
        Args:
            heavy_tokens: (B, SeqLen) 长整型token
            light_tokens: (B, SeqLen)
            antigen_tokens: (B, SeqLen)
        """
        device = self.device
        
        # 获取ESM嵌入 (使用Mean Pooling)
        def get_embedding(model, tokens):
            tokens = tokens.to(device)
            with torch.no_grad():  # 冻结ESM
                results = model(tokens.to(device), repr_layers=[12])
                token_repr = results["representations"][12]
                # Mean pooling (去掉BOS和EOS)
                emb = token_repr[:, 1:-1, :].mean(dim=1)  # (B, 1280)
            return emb
        
        # 注意：这里需要正确转换token格式
        # fair-esm需要 (B, SeqLen) 的long tensor
        emb_h = get_embedding(self.esm_model, heavy_tokens)  # (B, 1280)
        emb_l = get_embedding(self.esm_model, light_tokens)
        emb_ag = get_embedding(self.esm_antigen, antigen_tokens)
        
        # 添加序列维度用于Conv1d: (B, 1280) → (B, 1280, 1)
        emb_h = emb_h.unsqueeze(-1)
        emb_l = emb_l.unsqueeze(-1)
        emb_ag = emb_ag.unsqueeze(-1)
        
        # 特征提取
        h = self.heavy_layer1(emb_h)
        h = self.heavy_layer2(h)  # (B, 128, 1)
        
        l = self.light_layer1(emb_l)
        l = self.light_layer2(l)
        
        ag = self.antigen_layer1(emb_ag)
        ag = self.antigen_layer2(ag)
        
        # 去掉序列维度: (B, 128, 1) → (B, 128)
        h = h.squeeze(-1)
        l = l.squeeze(-1)
        ag = ag.squeeze(-1)
        
        # 拼接
        x = torch.cat([h, l, ag], dim=-1)  # (B, 384)
        
        # 全连接
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # (B, 1)
        
        return x
    
    def training_step(self, batch, batch_idx):
        heavy_tokens, light_tokens, antigen_tokens, labels = batch
        
        # 前向传播
        preds = self(heavy_tokens, light_tokens, antigen_tokens)
        
        # 计算损失
        labels = labels.view(-1, 1)
        loss = HuberLoss(delta=1.0)(preds, labels)
        
        # 记录指标
        self.log('train_loss', loss, on_epoch=True, prog_bar=True)
        self.log('train_mse', self.mse(preds, labels), on_epoch=True)
        self.log('train_mae', self.mae(preds, labels), on_epoch=True)
        self.log('train_pearson', self.pearson(preds, labels), on_epoch=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        heavy_tokens, light_tokens, antigen_tokens, labels = batch
        
        preds = self(heavy_tokens, light_tokens, antigen_tokens)
        labels = labels.view(-1, 1)
        
        loss = HuberLoss(delta=1.0)(preds, labels)
        
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        self.log('val_mse', self.mse(preds, labels), on_epoch=True)
        self.log('val_pearson', self.pearson(preds, labels), on_epoch=True)
        
        return {'val_loss': loss, 'preds': preds, 'labels': labels}
    
    def test_step(self, batch, batch_idx):
        heavy_tokens, light_tokens, antigen_tokens, labels = batch
        
        preds = self(heavy_tokens, light_tokens, antigen_tokens)
        labels = labels.view(-1, 1)
        
        loss = HuberLoss(delta=1.0)(preds, labels)
        
        self.log('test_loss', loss, on_epoch=True)
        self.log('test_mse', self.mse(preds, labels), on_epoch=True)
        self.log('test_mae', self.mae(preds, labels), on_epoch=True)
        self.log('test_rmse', self.rmse(preds, labels), on_epoch=True)
        self.log('test_r2', self.r2score(preds, labels), on_epoch=True)
        self.log('test_pearson', self.pearson(preds, labels), on_epoch=True)
        
        return {'test_loss': loss, 'preds': preds, 'labels': labels}
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',
                'frequency': 1
            }
        }


# === 数据加载函数 ===
def load_data(tsv_path: str) -> pd.DataFrame:
    """加载TSV数据"""
    df = pd.read_csv(tsv_path, sep='\t')
    
    # 列名映射
    col_map = {}
    cols_lower = [c.lower() for c in df.columns]
    
    for name in ['antibody_seq_a', 'heavy', 'vh', 'h']:
        if name in cols_lower:
            col_map['heavy'] = df.columns[cols_lower.index(name)]
            break
    
    for name in ['antibody_seq_b', 'light', 'vl', 'l']:
        if name in cols_lower:
            col_map['light'] = df.columns[cols_lower.index(name)]
            break
    
    for name in ['antigen_seq', 'antigen', 'ag']:
        if name in cols_lower:
            col_map['antigen'] = df.columns[cols_lower.index(name)]
            break
    
    for name in ['delta_g', 'ddg', 'deltag', 'binding_affinity']:
        if name in cols_lower:
            col_map['delta_g'] = df.columns[cols_lower.index(name)]
            break
    
    # 重命名列
    df = df[[col_map['heavy'], col_map['light'], col_map['antigen'], col_map['delta_g']]].copy()
    df.columns = ['heavy', 'light', 'antigen', 'delta_g']
    
    # 清洗数据
    valid_aa = set("ACDEFGHIKLMNPQRSTVWY")
    def is_valid_seq(seq):
        if not isinstance(seq, str) or len(seq.strip()) == 0:
            return False
        return all(c in valid_aa for c in seq.upper())
    
    df = df[
        df['heavy'].apply(is_valid_seq) &
        df['light'].apply(is_valid_seq) &
        df['antigen'].apply(is_valid_seq)
    ]
    df['delta_g'] = pd.to_numeric(df['delta_g'], errors='coerce')
    df = df.dropna(subset=['delta_g'])
    
    return df


# === 主程序 ===
def main():
    set_seed(22)
    
    # 配置
    DATA_DIR = "/tmp/AbAgCDR/data"
    TRAIN_FILE = f"{DATA_DIR}/final_dataset_train.tsv"
    MODEL_SAVE_PATH = "results/lightning_model_esm2"
    
    # 加载数据
    print("📥 加载数据...")
    df = load_data(TRAIN_FILE)
    print(f"✅ 加载 {len(df)} 条样本")
    
    # 划分数据集
    from sklearn.model_selection import train_test_split
    train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
    
    print(f"   训练集：{len(train_df)}, 验证集：{len(val_df)}, 测试集：{len(test_df)}")
    
    # 创建数据集
    import esm
    _, alphabet = esm.pretrained.esm2_t12_35M_UR50D()
    
    train_dataset = AntibodyAntigenDataset(train_df, alphabet)
    val_dataset = AntibodyAntigenDataset(val_df, alphabet)
    test_dataset = AntibodyAntigenDataset(test_df, alphabet)
    
    # 创建DataLoader
    train_loader = DataLoader(
        train_dataset, 
        batch_size=32, 
        shuffle=True, 
        collate_fn=lambda x: collate_fn(x, alphabet, max_length=1024),
        num_workers=4
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=32, 
        shuffle=False, 
        collate_fn=lambda x: collate_fn(x, alphabet, max_length=1024),
        num_workers=4
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=32, 
        shuffle=False, 
        collate_fn=lambda x: collate_fn(x, alphabet, max_length=1024),
        num_workers=4
    )
    
    # 创建模型
    model = ClassifierNet(
        esm_model_name="esm2_t12_35M_UR50D",
        hidden_dim=128,
        learning_rate=1e-4,
        max_length=1024
    )
    
    # 训练回调
    early_stopping = pl.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        mode='min',
        verbose=True
    )
    
    checkpoint = pl.callbacks.ModelCheckpoint(
        dirpath=MODEL_SAVE_PATH,
        filename='best_model-{epoch:02d}-{val_loss:.4f}',
        monitor='val_loss',
        mode='min',
        save_top_k=3
    )
    
    # 训练
    trainer = pl.Trainer(
        max_epochs=50,
        accelerator='auto',  # 自动选择CPU/GPU
        devices=1,
        callbacks=[early_stopping, checkpoint],
        log_every_n_steps=10,
        gradient_clip_val=1.0
    )
    
    print("\n🏃 开始训练...")
    trainer.fit(model, train_loader, val_loader)
    
    # 测试
    print("\n🧪 测试评估...")
    trainer.test(model, test_loader)
    
    print(f"\n✅ 训练完成！模型保存至：{MODEL_SAVE_PATH}")


if __name__ == "__main__":
    main()