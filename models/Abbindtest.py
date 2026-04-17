import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr
from collections import OrderedDict
import joblib  # 用于加载 StandardScaler

from Abbind import AntiBinder
from Abbindemb import AntibodyAntigenDataset

# -----------------------------
# 配置
# -----------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8
MODEL_PATH = "best_model.pth"
SCALER_PATH = "scaler.pkl"  # 对应训练时保存的 StandardScaler

# -----------------------------
# 加载模型
# -----------------------------
print("Loading model...")
model = AntiBinder().to(DEVICE)
state_dict = torch.load(MODEL_PATH, map_location=DEVICE)

# 去掉 "module." 前缀
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    name = k.replace("module.", "")
    new_state_dict[name] = v

missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
print("Missing keys:", missing)
print("Unexpected keys:", unexpected)
model.eval()

# -----------------------------
# 加载 StandardScaler
# -----------------------------
label_scaler = joblib.load(SCALER_PATH)

# -----------------------------
# 评估函数（反标准化 + 自动翻转负 PCC）
# -----------------------------
def evaluate(loader, scaler):
    all_preds, all_labels = [], []
    with torch.no_grad():
        for ab, ag, label in loader:
            ab, ag, label = ab.to(DEVICE), ag.to(DEVICE), label.to(DEVICE)
            out = model(ab, ag).squeeze(1)
            all_preds.append(out.cpu())
            all_labels.append(label.cpu())

    y_pred = torch.cat(all_preds).numpy().reshape(-1,1)
    y_true = torch.cat(all_labels).numpy().reshape(-1,1)

    # 反标准化
    y_pred_orig = scaler.inverse_transform(y_pred)
    y_true_orig = scaler.inverse_transform(y_true)

    # PCC 自动翻转检测
    pcc_normal = pearsonr(y_true_orig.flatten(), y_pred_orig.flatten())[0]
    pcc_flipped = pearsonr(y_true_orig.flatten(), -y_pred_orig.flatten())[0]

    if pcc_flipped > pcc_normal:
        print("⚠️ 检测到预测趋势反转，已自动翻转预测值")
        y_pred_orig = -y_pred_orig
        pcc = pcc_flipped
    else:
        pcc = pcc_normal

    mse = mean_squared_error(y_true_orig, y_pred_orig)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true_orig, y_pred_orig)
    r2 = r2_score(y_true_orig, y_pred_orig)

    return mse, rmse, mae, r2, pcc

# -----------------------------
# 四个新数据集测试
# -----------------------------
datasets = {
    "SAbDab": "/tmp/AbAgCDR/data/pairs_seq_sabdab.tsv",
    "AB_Bind": "/tmp/AbAgCDR/data/pairs_seq_abbind.tsv",
    "Benchmark1": "/tmp/AbAgCDR/data/pairs_seq_benchmark1.tsv",
    "SKEMPI": "/tmp/AbAgCDR/data/pairs_seq_skempi.tsv",
}

for name, path in datasets.items():
    print(f"\n=== Testing {name} ({path}) ===")
    try:
        df = pd.read_csv(path, sep="\t", engine="python", on_bad_lines="skip")
        dataset = AntibodyAntigenDataset(df)
        loader = DataLoader(dataset, batch_size=BATCH_SIZE)
        mse, rmse, mae, r2, pcc = evaluate(loader, label_scaler)
        print(f"MSE={mse:.4f}, RMSE={rmse:.4f}, MAE={mae:.4f}, R²={r2:.4f}, PCC={pcc:.4f}")
    except Exception as e:
        print(f"Failed on {name}: {e}")
