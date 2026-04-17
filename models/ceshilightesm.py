# import joblib
#
# pipeline = joblib.load("esm2_cdr_attention_lgbm_pipeline.pkl")
# print(pipeline.keys())
# import torch
# import matplotlib.pyplot as plt
# import numpy as np
#
# # 数据路径
# train_path = "/tmp/AbAgCDR/data/train_data.pt"
# test_datasets = {
#     "SAbDab": "/tmp/AbAgCDR/data/sabdab_data.pt",
#     "AB_Bind": "/tmp/AbAgCDR/data/abbind_data.pt",
#     "SKEMPI": "/tmp/AbAgCDR/data/skempi_data.pt",
#     "Benchmark": "/tmp/AbAgCDR/data/benchmark_data.pt"
# }
#
# def load_labels(path):
#     """从 .pt 文件加载标签，并保证转换为 numpy"""
#     data = torch.load(path)
#     for key in ['y_train', 'y_test', 'y']:
#         if key in data:
#             y = data[key]
#             return y.cpu().numpy() if y.is_cuda else y.numpy()
#     raise KeyError(f"No label found in {path}. Available keys: {list(data.keys())}")
#
#
# def plot_distributions(train_labels, test_labels_dict):
#     plt.figure(figsize=(10, 6))
#
#     # 训练集
#     mu, sigma = np.mean(train_labels), np.std(train_labels)
#     plt.hist(train_labels, bins=40, alpha=0.5, label=f"Train (μ={mu:.2f}, σ={sigma:.2f})")
#
#     # 各测试集
#     for name, labels in test_labels_dict.items():
#         mu, sigma = np.mean(labels), np.std(labels)
#         plt.hist(labels, bins=40, alpha=0.5, label=f"{name} (μ={mu:.2f}, σ={sigma:.2f})")
#
#     plt.xlabel("Label value")
#     plt.ylabel("Frequency")
#     plt.title("训练集 vs 测试集 标签分布对比")
#     plt.legend()
#     plt.tight_layout()
#     plt.show()
#
# def main():
#     # 加载训练集标签
#     train_labels = load_labels(train_path)
#
#     # 加载测试集标签
#     test_labels_dict = {name: load_labels(path) for name, path in test_datasets.items()}
#
#     # 画图
#     plot_distributions(train_labels, test_labels_dict)
#
# if __name__ == "__main__":
#     main()

# import torch
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# import os
#
# # ----------------------------
# # 配置
# # ----------------------------
# train_path = "/tmp/AbAgCDR/data/train_data.pt"
# test_datasets = {
#     "SAbDab": "/tmp/AbAgCDR/data/sabdab_data.pt",
#     "AB_Bind": "/tmp/AbAgCDR/data/abbind_data.pt",
#     "SKEMPI": "/tmp/AbAgCDR/data/skempi_data.pt",
#     "Benchmark": "/tmp/AbAgCDR/data/benchmark_data.pt"
# }
# output_dir = "/tmp/AbAgCDR/feature_distribution_plots"
# os.makedirs(output_dir, exist_ok=True)
#
# # ----------------------------
# # 数据加载与零填充对齐
# # ----------------------------
# def pad_to_max_length(arr_list):
#     """将一组二维数组（[N, L_i]）零填充到最大长度 L_max"""
#     max_len = max([arr.shape[1] for arr in arr_list])
#     padded_list = []
#     for arr in arr_list:
#         pad_width = max_len - arr.shape[1]
#         if pad_width > 0:
#             arr = np.pad(arr, ((0,0), (0,pad_width)), mode='constant', constant_values=0)
#         padded_list.append(arr)
#     return padded_list
#
# def load_data(path, is_train=False):
#     data = torch.load(path)
#
#     if is_train:
#         X_a = data.get("X_a_train", data.get("X_a"))
#         X_b = data.get("X_b_train", data.get("X_b"))
#         antigen = data.get("antigen_train", data.get("antigen"))
#         y = data.get("y_train", data.get("y"))
#     else:
#         X_a = data.get("X_a_test", data.get("X_a"))
#         X_b = data.get("X_b_test", data.get("X_b"))
#         antigen = data.get("antigen_test", data.get("antigen"))
#         y = data.get("y_test", data.get("y"))
#
#     # 转 CPU + numpy
#     def to_numpy(x):
#         if isinstance(x, torch.Tensor):
#             return x.detach().cpu().numpy()
#         return x
#
#     X_a, X_b, antigen = to_numpy(X_a), to_numpy(X_b), to_numpy(antigen)
#
#     # 零填充到最大长度
#     X_a, X_b, antigen = pad_to_max_length([X_a, X_b, antigen])
#
#     features = np.concatenate([X_a, X_b, antigen], axis=1)
#     return features, to_numpy(y)
#
# # ----------------------------
# # 加载训练和测试数据
# # ----------------------------
# train_features, train_y = load_data(train_path, is_train=True)
#
# test_features_dict = {}
# for name, path in test_datasets.items():
#     test_features, test_y = load_data(path, is_train=False)
#     test_features_dict[name] = test_features
#
# # ----------------------------
# # 单特征维度 KDE 图
# # ----------------------------
# num_features = train_features.shape[1]
# for i in range(num_features):
#     plt.figure(figsize=(8,5))
#     sns.kdeplot(train_features[:, i], label=f"Train (train_data)", fill=True, alpha=0.5)
#     for name, feat in test_features_dict.items():
#         sns.kdeplot(feat[:, i], label=f"Test ({name})", fill=True, alpha=0.5)
#     plt.title(f"Feature Dimension {i}")
#     plt.xlabel(f"Feature {i} value")
#     plt.ylabel("Density")
#     plt.legend()
#     plt.tight_layout()
#     plt.savefig(os.path.join(output_dir, f"feature_{i}_distribution.png"))
#     plt.close()
#
# # ----------------------------
# # 汇总图：箱线图 + 均值热图
# # ----------------------------
# # 合并训练和测试数据
# all_features = [train_features]
# all_labels = ["Train (train_data)"] * train_features.shape[0]
#
# for name, feat in test_features_dict.items():
#     all_features.append(feat)
#     all_labels.extend([f"Test ({name})"] * feat.shape[0])
#
# all_features = np.vstack(all_features)
#
# # 箱线图
# plt.figure(figsize=(15,6))
# sns.boxplot(data=all_features, whis=1.5)
# plt.title("Overall Feature Distribution (Boxplot)")
# plt.xlabel("Feature Dimension")
# plt.ylabel("Value")
# plt.tight_layout()
# plt.savefig(os.path.join(output_dir, "overall_boxplot.png"))
# plt.close()
#
# # 均值热图
# mean_features = all_features.mean(axis=0, keepdims=True)
# plt.figure(figsize=(15,2))
# sns.heatmap(mean_features, cmap="viridis")
# plt.title("Overall Feature Mean Heatmap")
# plt.xlabel("Feature Dimension")
# plt.tight_layout()
# plt.savefig(os.path.join(output_dir, "overall_mean_heatmap.png"))
# plt.close()
#
# print(f"所有特征分布图及汇总图已保存到 {output_dir} 文件夹下。")
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ----------------------------
# 配置
# ----------------------------
train_path = "/tmp/AbAgCDR/data/train_data.pt"
test_datasets = {
    "SAbDab": "/tmp/AbAgCDR/data/sabdab_data.pt",
    "AB_Bind": "/tmp/AbAgCDR/data/abbind_data.pt",
    "SKEMPI": "/tmp/AbAgCDR/data/skempi_data.pt",
    "Benchmark": "/tmp/AbAgCDR/data/benchmark_data.pt"
}
output_dir = "/tmp/AbAgCDR/feature_distribution_plots"
os.makedirs(output_dir, exist_ok=True)

# ----------------------------
# 数据加载与零填充对齐
# ----------------------------
def pad_to_max_length(arr_list):
    # 过滤掉 None
    arr_list = [arr for arr in arr_list if arr is not None]
    max_len = max([arr.shape[1] for arr in arr_list])
    padded_list = []
    for arr in arr_list:
        pad_width = max_len - arr.shape[1]
        if pad_width > 0:
            pad_shape = ((0,0), (0,pad_width)) + ((0,0),)*(arr.ndim-2)
            arr = np.pad(arr, pad_shape, mode='constant', constant_values=0)
        padded_list.append(arr)
    return padded_list

def load_data(path, is_train=False):
    data = torch.load(path)

    if is_train:
        X_a = data["X_a_train"] if "X_a_train" in data else data.get("X_a")
        X_b = data["X_b_train"] if "X_b_train" in data else data.get("X_b")
        antigen = data["antigen_train"] if "antigen_train" in data else data.get("antigen")
        y = data["y_train"] if "y_train" in data else data.get("y")
    else:
        X_a = data["X_a_test"] if "X_a_test" in data else data.get("X_a")
        X_b = data["X_b_test"] if "X_b_test" in data else data.get("X_b")
        antigen = data["antigen_test"] if "antigen_test" in data else data.get("antigen")
        y = data["y_test"] if "y_test" in data else data.get("y")

    def to_numpy(x):
        if x is None:
            return None
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return x

    X_a, X_b, antigen = to_numpy(X_a), to_numpy(X_b), to_numpy(antigen)
    X_list = [arr for arr in [X_a, X_b, antigen] if arr is not None]
    X_list = pad_to_max_length(X_list)

    def flatten_last(arr):
        if arr.ndim > 2:
            return arr.reshape(arr.shape[0], -1)
        return arr

    features = np.concatenate([flatten_last(arr) for arr in X_list], axis=1)
    return features, to_numpy(y)


# ----------------------------
# 绘制综合大图
# ----------------------------
def plot_feature_distributions(train_features, test_features_dict, max_features=50):
    # 如果维度太大，只画前 max_features 个维度
    num_features = min(train_features.shape[1], max_features)
    fig, axes = plt.subplots(num_features, 1, figsize=(12, num_features*1.5), sharex=True)
    if num_features == 1:
        axes = [axes]

    for i in range(num_features):
        axes[i].hist(train_features[:, i], bins=50, alpha=0.5, label="Train")
        for name, feat in test_features_dict.items():
            axes[i].hist(feat[:, i], bins=50, alpha=0.5, label=name)
        axes[i].legend(loc="upper right")
        axes[i].set_ylabel(f"Feature {i}")

    axes[-1].set_xlabel("Feature value")
    plt.tight_layout()
    plt.show()

# ----------------------------
# 主流程
# ----------------------------
train_features, train_y = load_data(train_path, is_train=True)
test_features_dict = {}
for name, path in test_datasets.items():
    feat, y = load_data(path, is_train=False)
    test_features_dict[name] = feat

plot_feature_distributions(train_features, test_features_dict)

