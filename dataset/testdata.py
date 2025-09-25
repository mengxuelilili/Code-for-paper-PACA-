import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr
from models.roformercnn import CombinedModel
import os
import warnings

warnings.filterwarnings('ignore')


# ==============================================================
# 数据集类
# ==============================================================
class CustomDataset(Dataset):
    """自定义数据集类，支持 dict/list，自动过滤不可索引对象"""
    def __init__(self, data):
        if isinstance(data, dict):
            self.data = {k: v for k, v in data.items() if hasattr(v, "__getitem__")}
            if not self.data:
                raise ValueError("数据字典中没有有效的张量！")
            self.length = len(next(iter(self.data.values())))
            self.is_dict = True
        elif isinstance(data, list):
            self.data = data
            self.length = len(data)
            self.is_dict = False
        else:
            raise ValueError(f"不支持的数据格式: {type(data)}")

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.is_dict:
            return {k: v[idx] for k, v in self.data.items()}
        else:
            return self.data[idx]


# ==============================================================
# 数据加载（统一 key 命名）
# ==============================================================
def load_test_data(data_path):
    """加载测试数据，保留 scaler，并统一 key 命名"""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"数据文件不存在: {data_path}")

    data = torch.load(data_path, map_location="cpu")
    print(f"成功加载: {data_path}, 类型={type(data)}")

    scalers = {}
    if isinstance(data, dict):
        keys_to_remove = []
        for k, v in data.items():
            if not (isinstance(v, torch.Tensor) or isinstance(v, np.ndarray)):
                scalers[k] = v
                keys_to_remove.append(k)
        for k in keys_to_remove:
            data.pop(k)

        # === 统一 key 命名 ===
        rename_map = {
            "X_a_test": "ab_light",
            "X_b_test": "ab_heavy",
            "antigen_test": "antigen",
            "y_test": "y",
        }
        for old, new in rename_map.items():
            if old in data:
                data[new] = data.pop(old)

        print(f"有效键: {list(data.keys())}, 样本数={len(next(iter(data.values())))}")

    return data, scalers


# ==============================================================
# 动态 padding
# ==============================================================
def dynamic_pad_collate(batch):
    if not batch:
        return {}

    keys = batch[0].keys()
    padded_batch = {}

    for key in keys:
        key_data = [item[key] for item in batch]
        first_item = key_data[0]

        # 处理标量 (y / delta_g)
        if isinstance(first_item, (int, float)) or (
            isinstance(first_item, torch.Tensor) and first_item.dim() == 0
        ):
            padded_batch[key] = torch.tensor(
                [float(x) for x in key_data], dtype=torch.float32
            )
            continue

        # 处理序列
        seqs = []
        max_len, feat_dim = 0, 1
        for x in key_data:
            t = torch.tensor(x, dtype=torch.float32) if not isinstance(x, torch.Tensor) else x
            if t.dim() == 1:
                t = t.unsqueeze(-1)
            seqs.append(t)
            max_len = max(max_len, t.shape[0])
            feat_dim = max(feat_dim, t.shape[1])

        padded = torch.zeros(len(seqs), max_len, feat_dim, dtype=torch.float32)
        for i, t in enumerate(seqs):
            padded[i, : t.shape[0], : t.shape[1]] = t
        padded_batch[key] = padded

    return padded_batch


# ==============================================================
# 模型相关
# ==============================================================
def get_cdr_positions(loop_name, cdr_scheme='chothia'):
    """获取CDR区域的位置信息"""
    if cdr_scheme == 'chothia':
        cdrs = {
            'L1': ['24', '25', '26', '27', '28', '29', '30', '30A', '30B', '30C', '30D', '30E', '30F',
                   '30G', '30H', '30I', '31', '32', '33', '34'],
            'L2': ['50', '51', '51A', '52', '52A', '52B', '52C', '52D', '53', '54', '55', '56'],
            'L3': ['89', '90', '91', '92', '93', '94', '95', '95A', '95B', '95C', '95D', '95E', '95F',
                   '95G', '95H', '95I', '95J', '96', '97'],
            'H1': ['26', '27', '28', '29', '30', '31', '31A', '31B', '31C', '31D', '31E', '31F', '31G', '31H',
                   '31I', '31J', '32'],
            'H2': ['52', '52A', '52B', '52C', '52D', '52E', '52F', '52G', '52H', '52I', '52J', '52K', '52L', '52M',
                   '52N', '52O', '53', '54', '55', '56'],
            'H3': ['95', '96', '97', '98', '99', '100', '100A', '100B', '100C', '100D',
                   '100E', '100F', '100G', '100H', '100I', '100J', '100K', '100L', '100M', '100N', '100O', '100P',
                   '100Q', '100R', '100S', '100T', '100U', '100V', '100W', '100X', '100Y', '100Z', '101', '102']
        }
        return cdrs.get(loop_name, [])
    return []


def load_model(model_path, device):
    ckpt = torch.load(model_path, map_location=device)
    model = CombinedModel(
        cdr_boundaries_light=[get_cdr_positions(s) for s in ["L1", "L2", "L3"]],
        cdr_boundaries_heavy=[get_cdr_positions(s) for s in ["H1", "H2", "H3"]],
        num_heads=2,
        embed_dim=532,
        antigen_embed_dim=500,
    ).to(device)

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
        print("加载 checkpoint 模型参数")
    else:
        model.load_state_dict(ckpt)
        print("加载完整模型参数")

    model.eval()
    return model


# ==============================================================
# 批量预测
# ==============================================================
def predict_batch(model, batch, device):
    try:
        ab_light = batch["ab_light"].to(device)
        ab_heavy = batch["ab_heavy"].to(device)
        antigen = batch["antigen"].to(device)
        labels = batch["y"].to(device).float()
    except KeyError as e:
        raise KeyError(f"缺少输入或标签: {e}, batch keys={list(batch.keys())}")

    with torch.no_grad():
        preds = model(ab_light, ab_heavy, antigen).squeeze()

    return preds, labels


# ==============================================================
# 指标
# ==============================================================
def calculate_metrics(preds, labels):
    preds, labels = np.array(preds), np.array(labels)
    mask = ~(np.isnan(preds) | np.isnan(labels) | np.isinf(preds) | np.isinf(labels))
    preds, labels = preds[mask], labels[mask]

    if len(preds) == 0:
        return None

    mse = mean_squared_error(labels, preds)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(labels, preds)
    r2 = r2_score(labels, preds)
    pearson = pearsonr(labels, preds)[0] if len(labels) > 1 else 0.0

    return {"MSE": mse, "RMSE": rmse, "MAE": mae, "R2": r2, "Pearson": pearson, "N": len(labels)}


# ==============================================================
# 单数据集评估
# ==============================================================
def evaluate_dataset(model, data_path, device, batch_size=32):
    data, scalers = load_test_data(data_path)
    dataset = CustomDataset(data)
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=dynamic_pad_collate)

    preds, labels = [], []
    for batch in loader:
        try:
            p, l = predict_batch(model, batch, device)
            preds.append(p.cpu().numpy())
            labels.append(l.cpu().numpy())
        except Exception as e:
            print(f"跳过 batch: {e}")

    if not preds:
        return None

    preds, labels = np.concatenate(preds), np.concatenate(labels)
    return calculate_metrics(preds, labels)


# ==============================================================
# 主程序
# ==============================================================
def main():
    model_path = "/tmp/AbAgCDR/model/stable_bestmodel_weighted2.pth"
    datasets = {
        "Benchmark": "/tmp/AbAgCDR/data/benchmark_data.pt"
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(model_path, device)

    results = {}
    for name, path in datasets.items():
        if not os.path.exists(path):
            continue
        print(f"\n=== 测试 {name} ===")
        m = evaluate_dataset(model, path, device)
        if m:
            results[name] = m
            print(f"{name}: " + ", ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in m.items()))

    if results:
        print("\n" + "=" * 50)
        print(f"{'Dataset':<12} {'MSE':<8} {'RMSE':<8} {'MAE':<8} {'R2':<8} {'Pearson':<8} {'N':<6}")
        for k, v in results.items():
            print(f"{k:<12} {v['MSE']:<8.4f} {v['RMSE']:<8.4f} {v['MAE']:<8.4f} {v['R2']:<8.4f} {v['Pearson']:<8.4f} {v['N']:<6}")


if __name__ == "__main__":
    main()
