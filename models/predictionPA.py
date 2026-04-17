import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from roformercnn import CombinedModel

# ======================================================
# CDR 区域定义（Chothia）
# ======================================================
def getCDRPos(_loop, cdr_scheme='chothia'):
    CDRS = {
        'L1': ['24','25','26','27','28','29','30','30A','30B','30C','30D','30E','30F','30G','30H','30I','31','32','33','34'],
        'L2': ['50','51','51A','52','52A','52B','52C','52D','53','54','55','56'],
        'L3': ['89','90','91','92','93','94','95','95A','95B','95C','95D','95E','95F','95G','95H','95I','95J','96','97'],
        'H1': ['26','27','28','29','30','31','31A','31B','31C','31D','31E','31F','31G','31H','31I','31J','32'],
        'H2': ['52','52A','52B','52C','52D','52E','52F','52G','52H','52I','52J','52K','52L','52M','52N','52O','53','54','55','56'],
        'H3': ['95','96','97','98','99','100','100A','100B','100C','100D','100E','100F','100G','100H',
               '100I','100J','100K','100L','100M','100N','100O','100P','100Q','100R','100S','100T',
               '100U','100V','100W','100X','100Y','100Z','101','102']
    }
    return CDRS[_loop]


# ======================================================
# Dataset & Collate
# ======================================================
class ListDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_fn(batch):
    X_a_list = [torch.tensor(item[0], dtype=torch.float32) if not isinstance(item[0], torch.Tensor) else item[0] for item in batch]
    X_b_list = [torch.tensor(item[1], dtype=torch.float32) if not isinstance(item[1], torch.Tensor) else item[1] for item in batch]
    ag_list = [torch.tensor(item[2], dtype=torch.float32) if not isinstance(item[2], torch.Tensor) else item[2] for item in batch]
    y_list = [torch.tensor(item[3], dtype=torch.float32) if not isinstance(item[3], torch.Tensor) else item[3] for item in batch]

    max_len = max(
        max(x.shape[0] for x in X_a_list),
        max(x.shape[0] for x in X_b_list),
        max(x.shape[0] for x in ag_list)
    )

    def pad_to_len(x, L):
        if x.shape[0] < L:
            pad = torch.zeros(L - x.shape[0], x.shape[1], dtype=x.dtype)
            return torch.cat([x, pad], dim=0)
        else:
            return x[:L]

    X_a_padded = torch.stack([pad_to_len(x, max_len) for x in X_a_list])
    X_b_padded = torch.stack([pad_to_len(x, max_len) for x in X_b_list])
    ag_padded = torch.stack([pad_to_len(x, max_len) for x in ag_list])
    y_tensor = torch.stack(y_list)

    return X_a_padded, X_b_padded, ag_padded, y_tensor


# ======================================================
# Main Prediction + Plot Function
# ======================================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Paths ---
    model_path = "/tmp/AbAgCDR/model/best_modelxin.pth"
    embed_path = "/tmp/AbAgCDR/data/abbind_data.pt"
    tsv_path = "/tmp/AbAgCDR/data/pairs_seq_abbind2_clean.tsv"
    output_dir = "/tmp/AbAgCDR/resultsxin"
    os.makedirs(output_dir, exist_ok=True) 
    output_csv = os.path.join(output_dir, "abbind_predictions.csv")    # abbind_predictions.csv abbind_predictions_seed_42.csv
    plot_path = os.path.join(output_dir, "abbindregression_plot.png")  # abbind_regression_plot.png abbind_regression_plot_seed_42.png

    # --- Load model and scaler ---
    checkpoint = torch.load(model_path, map_location="cpu")
    label_scaler = checkpoint.get("label_scaler", None)
    if label_scaler is None:
        raise ValueError("❌ label_scaler not found in model checkpoint! Cannot inverse transform predictions.")

    model = CombinedModel(
        [getCDRPos("H1"), getCDRPos("H2"), getCDRPos("H3")],
        [getCDRPos("L1"), getCDRPos("L2"), getCDRPos("L3")],
        num_heads=2,
        embed_dim=532,
        antigen_embed_dim=500
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    # --- Load original TSV to get TRUE (unnormalized) delta_g ---
    df_tsv = pd.read_csv(tsv_path, sep="\t")
    if "delta_g" not in df_tsv.columns:
        raise KeyError("❌ TSV file must contain 'delta_g' column for true values.")
    true_ddg_original = df_tsv["delta_g"].values

    # --- Load embedding data ---
    data = torch.load(embed_path, map_location="cpu")
    X_a = data["X_a"].cpu().numpy()
    X_b = data["X_b"].cpu().numpy()
    antigen = data["antigen"].cpu().numpy()
    y_normalized = data["y"].cpu().numpy()

    assert len(true_ddg_original) == len(y_normalized), \
        f"TSV rows ({len(true_ddg_original)}) != embedding samples ({len(y_normalized)})"

    # --- Prepare dataset ---
    samples = [(X_a[i], X_b[i], antigen[i], y_normalized[i]) for i in range(len(y_normalized))]
    dataloader = DataLoader(
        ListDataset(samples),
        batch_size=32,
        shuffle=False,
        collate_fn=collate_fn
    )

    # --- Run prediction ---
    pred_normalized_list = []
    with torch.no_grad():
        for X_a_batch, X_b_batch, ag_batch, _ in dataloader:
            X_a_batch = X_a_batch.to(device)
            X_b_batch = X_b_batch.to(device)
            ag_batch = ag_batch.to(device)
            pred = model(X_b_batch, X_a_batch, ag_batch).view(-1)
            pred_normalized_list.extend(pred.cpu().numpy())

    pred_normalized = np.array(pred_normalized_list).reshape(-1, 1)
    pred_ddg_original = label_scaler.inverse_transform(pred_normalized).flatten()

    # --- Save results ---
    result_df = pd.DataFrame({
        "Index": np.arange(len(true_ddg_original)),
        "true_ddg": true_ddg_original,
        "pred_ddg": pred_ddg_original
    })
    result_df.to_csv(output_csv, index=False)
    print(f"✅ Predictions saved to: {output_csv}")

    # --- Plot regression figure ---
    plt.figure(figsize=(8, 6))
    plt.scatter(result_df['true_ddg'], result_df['pred_ddg'], alpha=0.6, color='#1f77b4', s=20)
    min_val = min(result_df['true_ddg'].min(), result_df['pred_ddg'].min())
    max_val = max(result_df['true_ddg'].max(), result_df['pred_ddg'].max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Ideal Fit')
    plt.xlabel('True ΔG', fontsize=14)
    plt.ylabel('Predicted ΔG', fontsize=14)
    plt.title('AB-Bind Regression: True vs Predicted ΔG', fontsize=16)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=330)
    print(f"🎨 Regression plot saved to: {plot_path}")

    # Optional: show plot if running locally
    # plt.show()

if __name__ == "__main__":
    main()

# # PWAA+RPE预测脚本
# # -*- coding: utf-8 -*-
# """
# 预测脚本 - 适配 CombinedModel 训练脚本
# 模型输入顺序：(antibody_light, antibody_heavy, antigen)
# """

# import torch
# import torch.nn as nn
# from torch.utils.data import Dataset, DataLoader
# import numpy as np
# import pandas as pd
# import os
# import matplotlib.pyplot as plt
# from scipy.stats import pearsonr
# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# from roformercnn import CombinedModel  # ✅ 确保与训练脚本同一文件

# # ======================================================
# # CDR 区域定义（Chothia）
# # ======================================================
# def getCDRPos(_loop, cdr_scheme='chothia'):
#     CDRS = {
#         'L1': ['24','25','26','27','28','29','30','30A','30B','30C','30D','30E','30F','30G','30H','30I','31','32','33','34'],
#         'L2': ['50','51','51A','52','52A','52B','52C','52D','53','54','55','56'],
#         'L3': ['89','90','91','92','93','94','95','95A','95B','95C','95D','95E','95F','95G','95H','95I','95J','96','97'],
#         'H1': ['26','27','28','29','30','31','31A','31B','31C','31D','31E','31F','31G','31H','31I','31J','32'],
#         'H2': ['52','52A','52B','52C','52D','52E','52F','52G','52H','52I','52J','52K','52L','52M','52N','52O','53','54','55','56'],
#         'H3': ['95','96','97','98','99','100','100A','100B','100C','100D','100E','100F','100G','100H',
#                '100I','100J','100K','100L','100M','100N','100O','100P','100Q','100R','100S','100T',
#                '100U','100V','100W','100X','100Y','100Z','101','102']
#     }
#     return CDRS[_loop]

# # ======================================================
# # Dataset & Collate - ✅ 修复维度处理
# # ======================================================
# class ListDataset(Dataset):
#     def __init__(self, samples):
#         self.samples = samples

#     def __len__(self):
#         return len(self.samples)

#     def __getitem__(self, idx):
#         return self.samples[idx]

# def collate_fn(batch):
#     """✅ 修复：与训练脚本完全一致"""
#     X_a_list = [torch.tensor(item[0], dtype=torch.float32) if not isinstance(item[0], torch.Tensor) else item[0] for item in batch]
#     X_b_list = [torch.tensor(item[1], dtype=torch.float32) if not isinstance(item[1], torch.Tensor) else item[1] for item in batch]
#     ag_list = [torch.tensor(item[2], dtype=torch.float32) if not isinstance(item[2], torch.Tensor) else item[2] for item in batch]
#     y_list = [torch.tensor(item[3], dtype=torch.float32) if not isinstance(item[3], torch.Tensor) else item[3] for item in batch]

#     max_len = max(
#         max(x.shape[0] for x in X_a_list),
#         max(x.shape[0] for x in X_b_list),
#         max(x.shape[0] for x in ag_list)
#     )

#     def pad_to_len(x, L):
#         if x.shape[0] < L:
#             # ✅ 修复：使用 *x.shape[1:] 支持任意维度
#             pad = torch.zeros(L - x.shape[0], *x.shape[1:], dtype=x.dtype, device=x.device)
#             return torch.cat([x, pad], dim=0)
#         else:
#             return x[:L]

#     X_a_padded = torch.stack([pad_to_len(x, max_len) for x in X_a_list])
#     X_b_padded = torch.stack([pad_to_len(x, max_len) for x in X_b_list])
#     ag_padded = torch.stack([pad_to_len(x, max_len) for x in ag_list])
#     y_tensor = torch.stack(y_list)

#     # ✅ 确保 3D tensor
#     if X_a_padded.dim() == 2:
#         X_a_padded = X_a_padded.unsqueeze(0)
#     if X_b_padded.dim() == 2:
#         X_b_padded = X_b_padded.unsqueeze(0)
#     if ag_padded.dim() == 2:
#         ag_padded = ag_padded.unsqueeze(0)

#     return X_a_padded, X_b_padded, ag_padded, y_tensor

# # ======================================================
# # Main Prediction Function
# # ======================================================
# def main():
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"🖥️  Using device: {device}")

#     # --- Paths - ✅ 修复：与训练脚本保存路径一致 ---
#     model_path = "/tmp/AbAgCDR/model/PWAA_RPE_best_model.pth"  # ✅ 修改
#     embed_path = "/tmp/AbAgCDR/data/train_data.pt"
#     tsv_path = "/tmp/AbAgCDR/data/final_dataset_train.tsv"
#     output_dir = "/tmp/AbAgCDR/resultsxin"
#     os.makedirs(output_dir, exist_ok=True)
#     output_csv = os.path.join(output_dir, "PWAARPEtrain_predictions_seed_42.csv") # PWAARPEskempi_predictions.csv
#     plot_path = os.path.join(output_dir, "PWAARPEtrain_regression_plot_seed_42.png")

#     # --- Load model and scaler ---
#     print("\n" + "="*60)
#     print("🏗️ 加载模型")
#     print("="*60)
    
#     checkpoint = torch.load(model_path, map_location="cpu")
#     label_scaler = checkpoint.get("label_scaler", None)
#     config = checkpoint.get("config", {})
    
#     print(f"模型配置：{config}")
    
#     if label_scaler is None:
#         print("⚠️  label_scaler not found in checkpoint, predictions will be normalized")

#     # --- ✅ 修复：CDR 边界顺序与训练脚本一致 ---
#     print("\n" + "="*60)
#     print("🧬 初始化模型")
#     print("="*60)
    
#     model = CombinedModel(
#         cdr_boundaries_light=[getCDRPos("L1"), getCDRPos("L2"), getCDRPos("L3")],  # ✅ 第一个参数是 light
#         cdr_boundaries_heavy=[getCDRPos("H1"), getCDRPos("H2"), getCDRPos("H3")],  # ✅ 第二个参数是 heavy
#         num_heads=config.get("num_heads", 2),
#         embed_dim=config.get("embed_dim", 532),
#         antigen_embed_dim=config.get("antigen_embed_dim", 500),
#         hidden_dim=config.get("hidden_dim", 256)
#     )
    
#     model.load_state_dict(checkpoint["model_state_dict"])
#     model.to(device)
#     model.eval()
    
#     total_params = sum(p.numel() for p in model.parameters())
#     print(f"✅ 模型已加载，总参数量：{total_params:,}")

#     # --- Load original TSV ---
#     print("\n" + "="*60)
#     print("📊 加载数据")
#     print("="*60)
    
#     df_tsv = pd.read_csv(tsv_path, sep="\t")
#     if "delta_g" not in df_tsv.columns:
#         raise KeyError("❌ TSV file must contain 'delta_g' column")
#     true_ddg_original = df_tsv["delta_g"].values

#     # --- Load embedding data ---
#     data = torch.load(embed_path, map_location="cpu")
#     X_a = data["X_a"].cpu().numpy()  # light chain
#     X_b = data["X_b"].cpu().numpy()  # heavy chain
#     antigen = data["antigen"].cpu().numpy()
#     y_normalized = data["y"].cpu().numpy()

#     assert len(true_ddg_original) == len(y_normalized), \
#         f"TSV rows ({len(true_ddg_original)}) != embedding samples ({len(y_normalized)})"

#     print(f"✅ 数据加载完成：{len(y_normalized)} 样本")

#     # --- Prepare dataset ---
#     samples = [(X_a[i], X_b[i], antigen[i], y_normalized[i]) for i in range(len(y_normalized))]
#     dataloader = DataLoader(
#         ListDataset(samples),
#         batch_size=32,
#         shuffle=False,
#         collate_fn=collate_fn,
#         num_workers=0
#     )

#     # --- Run prediction - ✅ 修复：输入顺序与训练一致 ---
#     print("\n" + "="*60)
#     print("🔮 执行预测")
#     print("="*60)
    
#     pred_normalized_list = []
#     with torch.no_grad():
#         for X_a_batch, X_b_batch, ag_batch, _ in dataloader:
#             X_a_batch = X_a_batch.to(device)
#             X_b_batch = X_b_batch.to(device)
#             ag_batch = ag_batch.to(device)
            
#             # ✅ 修复：输入顺序 (light, heavy, antigen) 与训练脚本一致
#             pred = model(X_a_batch, X_b_batch, ag_batch).view(-1)
#             pred_normalized_list.extend(pred.cpu().numpy())

#     pred_normalized = np.array(pred_normalized_list).reshape(-1, 1)
    
#     # --- Inverse transform ---
#     if label_scaler is not None:
#         pred_ddg_original = label_scaler.inverse_transform(pred_normalized).flatten()
#     else:
#         pred_ddg_original = pred_normalized.flatten()
#         print("⚠️  无 label_scaler，输出为归一化值")

#     # --- Save results ---
#     print("\n" + "="*60)
#     print("💾 保存结果")
#     print("="*60)
    
#     result_df = pd.DataFrame({
#         "Index": np.arange(len(true_ddg_original)),
#         "true_ddg": true_ddg_original,
#         "pred_ddg": pred_ddg_original
#     })
#     result_df.to_csv(output_csv, index=False)
#     print(f"✅ 预测结果已保存：{output_csv}")

#     # --- Calculate metrics ---
#     print("\n" + "="*60)
#     print("📊 评估指标")
#     print("="*60)
    
#     pcc, _ = pearsonr(true_ddg_original, pred_ddg_original)
#     mse = mean_squared_error(true_ddg_original, pred_ddg_original)
#     rmse = np.sqrt(mse)
#     mae = mean_absolute_error(true_ddg_original, pred_ddg_original)
#     r2 = r2_score(true_ddg_original, pred_ddg_original)
    
#     print(f"   PCC:  {pcc:.4f}")
#     print(f"   MSE:  {mse:.4f}")
#     print(f"   RMSE: {rmse:.4f}")
#     print(f"   MAE:  {mae:.4f}")
#     print(f"   R²:   {r2:.4f}")

#     # --- Plot regression figure ---
#     print("\n" + "="*60)
#     print("🎨 绘制回归图")
#     print("="*60)
    
#     plt.figure(figsize=(8, 6))
#     plt.scatter(result_df['true_ddg'], result_df['pred_ddg'], alpha=0.6, color='#1f77b4', s=20)
#     min_val = min(result_df['true_ddg'].min(), result_df['pred_ddg'].min())
#     max_val = max(result_df['true_ddg'].max(), result_df['pred_ddg'].max())
#     plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Ideal Fit')
    
#     # 添加指标文本
#     textstr = f'PCC: {pcc:.4f}\nRMSE: {rmse:.4f}\nMAE: {mae:.4f}\nR²: {r2:.4f}'
#     plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=10,
#              verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
#     plt.xlabel('True ΔΔG', fontsize=12)
#     plt.ylabel('Predicted ΔΔG', fontsize=12)
#     plt.title('SKEMPI2.0 Regression: True vs Predicted ΔΔG', fontsize=14)
#     plt.legend()
#     plt.grid(True, linestyle='--', alpha=0.5)
#     plt.tight_layout()
#     plt.savefig(plot_path, dpi=300)
#     print(f"✅ 回归图已保存：{plot_path}")

#     print("\n" + "="*60)
#     print("✅ 预测完成！")
#     print("="*60)

# if __name__ == "__main__":
#     main()