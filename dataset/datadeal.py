
import torch
import pandas as pd
from antiberty import AntiBERTyRunner
import esm
from sklearn.preprocessing import StandardScaler
import torch.nn.utils.rnn as rnn_utils
import torch.nn.functional as F
import numpy as np

# 确定设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 读取TSV文件
def load_data_from_tsv(file_path):
    data = pd.read_csv(file_path, sep='\t')
    antibody_sequences_a = data['antibody_seq_a'].tolist()  # 轻链序列
    antibody_sequences_b = data['antibody_seq_b'].tolist()  # 重链序列
    antigen_sequences = data['antigen_seq'].tolist()  # 抗原序列
    delta_g = data['delta_g'].tolist()  # 亲和力值
    return antibody_sequences_a, antibody_sequences_b, antigen_sequences, delta_g

def pwaa_encode_sequence(sequences, L=None):
    pwaa_features = []

    for seq in sequences:
        seq_length = len(seq)

        # 若未指定 L，则默认使用一半序列长度
        if L is None:
            L = seq_length // 2

        # 计算氨基酸的一热编码
        one_hot = torch.zeros((seq_length, 20), device=device)  # 20表示20种氨基酸
        for i, aa in enumerate(seq):
            index = ord(aa) - ord('A')  # 字符转为索引
            if 0 <= index < 20:  # 确保索引在有效范围内
                one_hot[i, index] = 1

        # 计算权重分布，符合图片中的公式
        weights = torch.tensor(
            [(j + abs(j) / L) for j in range(-L, L + 1)], device=device)
        weights = weights / (L * (L + 1))  # 归一化权重，与公式中的 L(L+1)一致

        # 调整权重长度以匹配序列长度
        if len(weights) > seq_length:
            weights = weights[:seq_length]
        elif len(weights) < seq_length:
            weights = torch.cat([weights, torch.zeros(seq_length - len(weights), device=device)])

        # 将位置权重应用于一热编码特征
        weighted_feature = one_hot * weights.unsqueeze(1)
        pwaa_features.append(weighted_feature)

    # 将所有序列的 PWAA 特征堆叠为一个张量
    pwaa_tensor = rnn_utils.pad_sequence(pwaa_features, batch_first=True)
    return pwaa_tensor.to(device)

# 使用 ESM 处理抗原序列（保持原始长度，便于后续使用注意力机制）
def process_antigen_sequences(antigen_sequences, esm_model, alphabet):
    embeddings = []
    for antigen_seq in antigen_sequences:
        # 使用 ESM 的分词器对序列进行分词
        tokens = alphabet.encode(antigen_seq)
        tokens_tensor = torch.tensor(tokens, device=device).unsqueeze(0)  # 添加 batch 维度并移动到设备上

        with torch.no_grad():
            results = esm_model(tokens_tensor, repr_layers=[11])  # 使用第11层的表示
        embedding = results['representations'][11].squeeze(0)  # 去除 batch 维度，保持序列长度
        # 将未压缩的嵌入添加到列表
        embeddings.append(embedding)
    # 使用 rnn_utils.pad_sequence 来填充嵌入序列，确保序列长度一致
    embeddings_padded = rnn_utils.pad_sequence(embeddings, batch_first=True)
    # 将张量转换到设备上（如果适用）
    return embeddings_padded.to(device)

# 加载 ESM 模型和分词器
esm_model, alphabet = esm.pretrained.esm2_t12_35M_UR50D()
esm_model = esm_model.to(device)

# === 滑动窗口工具函数 ===
def sliding_window(seq, window_size=128, stride=64):
    """生成序列的滑动窗口切片"""
    for i in range(0, len(seq), stride):
        yield seq[i:i + window_size]

# 使用 AntiBERTy 处理抗体序列（重链和轻链），保持 [L, D] 格式
def process_antibody_sequences(antibody_sequences_a, antibody_sequences_b, antiberty_model,
                               window_size=128, stride=128):
    """使用AntiBERTy处理抗体序列 + 滑动窗口，不做 pooling，保持 [L, D]"""
    print("开始处理抗体序列...")
    embeddings_a = []
    embeddings_b = []

    for i, (heavy_seq, light_seq) in enumerate(zip(antibody_sequences_a, antibody_sequences_b)):
        if i % 50 == 0:
            print(f"处理抗体序列进度: {i + 1}/{len(antibody_sequences_a)}")

        try:
            if not light_seq or not isinstance(light_seq, str):
                print(f"警告: 第{i + 1}个轻链序列无效")
                continue
            if not heavy_seq or not isinstance(heavy_seq, str):
                print(f"警告: 第{i + 1}个重链序列无效")
                continue

            # 轻链处理
            light_chunks = []
            for chunk in sliding_window(light_seq, window_size, stride):
                emb, _ = antiberty_model.embed([chunk], return_attention=True)
                light_chunks.append(emb[0].detach().to(device))
            embedding_a_tensor = torch.cat(light_chunks, dim=0)  # 保持 [L, D]

            # 重链处理
            heavy_chunks = []
            for chunk in sliding_window(heavy_seq, window_size, stride):
                emb, _ = antiberty_model.embed([chunk], return_attention=True)
                heavy_chunks.append(emb[0].detach().to(device))
            embedding_b_tensor = torch.cat(heavy_chunks, dim=0)  # 保持 [L, D]

            embeddings_a.append(embedding_a_tensor)
            embeddings_b.append(embedding_b_tensor)

        except Exception as e:
            print(f"处理第{i + 1}个抗体序列时出错: {e}")
            continue

    if not embeddings_a or not embeddings_b:
        raise ValueError("没有成功处理任何抗体序列")

    embeddings_a_padded = rnn_utils.pad_sequence(embeddings_a, batch_first=True).to(device)
    embeddings_b_padded = rnn_utils.pad_sequence(embeddings_b, batch_first=True).to(device)

    print(f"轻链嵌入形状: {embeddings_a_padded.shape}")
    print(f"重链嵌入形状: {embeddings_b_padded.shape}")

    return embeddings_a_padded, embeddings_b_padded
# 标准化嵌入
def normalize_embeddings(embeddings, name=""):
    print(f"开始标准化{name}嵌入...")
    emb_np = embeddings.detach().cpu().numpy()
    orig_shape = emb_np.shape
    emb_flat = emb_np.reshape(-1, emb_np.shape[-1])
    scaler = StandardScaler()
    emb_scaled = scaler.fit_transform(emb_flat)
    emb_scaled = torch.tensor(emb_scaled.reshape(orig_shape), dtype=torch.float32, device=device)
    print(f"{name}嵌入标准化完成，形状: {emb_scaled.shape}")
    return emb_scaled, scaler
# 修改 prepare_dataset
def prepare_dataset(file_path, antiberty_model, esm_model, alphabet, device):
    # 1. 加载数据 (抗体序列 + 抗原序列 + ΔG)
    antibody_sequences_a, antibody_sequences_b, antigen_sequences, delta_g = load_data_from_tsv(file_path)

    # ========== Helper: 标准化并返回 torch.Tensor ==========
    def standardize_embeddings(embeddings):
        embeddings_numpy = embeddings.detach().cpu().numpy()
        flat = embeddings_numpy.reshape(-1, embeddings_numpy.shape[-1])
        scaled = StandardScaler().fit_transform(flat)
        scaled_tensor = torch.tensor(scaled, dtype=torch.float32).reshape(embeddings_numpy.shape)
        return scaled_tensor.to(device)

    # ========== 2. 抗体嵌入 ==========
    embeddings_a, embeddings_b = process_antibody_sequences(antibody_sequences_a, antibody_sequences_b, antiberty_model)
    embeddings_a_scaled = standardize_embeddings(embeddings_a)
    embeddings_b_scaled = standardize_embeddings(embeddings_b)

    pwaa_a = pwaa_encode_sequence(antibody_sequences_a, L=5).to(device)
    pwaa_b = pwaa_encode_sequence(antibody_sequences_b, L=5).to(device)

    # Pad 对齐 A/B
    max_seqa_len = max(pwaa_a.shape[1], embeddings_a_scaled.shape[1])
    max_seqb_len = max(pwaa_b.shape[1], embeddings_b_scaled.shape[1])

    pwaa_a_padded = torch.nn.functional.pad(pwaa_a, (0, 0, 0, max_seqa_len - pwaa_a.shape[1]), "constant", 0)
    emb_a_padded  = torch.nn.functional.pad(embeddings_a_scaled, (0, 0, 0, max_seqa_len - embeddings_a_scaled.shape[1]), "constant", 0)
    embeddings_a_fused = torch.cat((pwaa_a_padded, emb_a_padded), dim=2)

    pwaa_b_padded = torch.nn.functional.pad(pwaa_b, (0, 0, 0, max_seqb_len - pwaa_b.shape[1]), "constant", 0)
    emb_b_padded  = torch.nn.functional.pad(embeddings_b_scaled, (0, 0, 0, max_seqb_len - embeddings_b_scaled.shape[1]), "constant", 0)
    embeddings_b_fused = torch.cat((pwaa_b_padded, emb_b_padded), dim=2)

    # ========== 3. 抗原嵌入 ==========
    antigen_embeddings = process_antigen_sequences(antigen_sequences, esm_model, alphabet)
    antigen_embeddings_scaled = standardize_embeddings(antigen_embeddings)

    pwaa_antigen = pwaa_encode_sequence(antigen_sequences, L=5).to(device)

    max_seqg_len = max(pwaa_antigen.shape[1], antigen_embeddings_scaled.shape[1])
    pwaa_g_padded = torch.nn.functional.pad(pwaa_antigen, (0, 0, 0, max_seqg_len - pwaa_antigen.shape[1]), "constant", 0)
    emb_g_padded  = torch.nn.functional.pad(antigen_embeddings_scaled, (0, 0, 0, max_seqg_len - antigen_embeddings_scaled.shape[1]), "constant", 0)

    antigen_fused = torch.cat((pwaa_g_padded, emb_g_padded), dim=2)

    # ========== 4. ΔG 归一化 ==========
    delta_g = np.array(delta_g).reshape(-1, 1)
    label_scaler = StandardScaler()
    delta_g_scaled = label_scaler.fit_transform(delta_g)
    delta_g_tensor = torch.tensor(delta_g_scaled, dtype=torch.float32, device=device).squeeze()

    # ========== 5. 返回 ==========
    return embeddings_a_fused, embeddings_b_fused, antigen_fused, delta_g_tensor, label_scaler

# 修改 save_dataset
def save_dataset(test_data, test_path):
    torch.save({
        'X_a_test': test_data[0],
        'X_b_test': test_data[1],
        'antigen_test': test_data[2],
        'y_test': test_data[3],
        'label_scaler': test_data[4]   # 保存标签归一化器
    }, test_path)
    print(f"训练集保存到 {test_path}")

# 主函数
if __name__ == "__main__":
    file_path = '/tmp/AbAgCDR/data/pairs_seq_benchmark1.tsv'
    test_path = '/tmp/AbAgCDR/data/benchmark_data.pt'

    antiberty_model = AntiBERTyRunner()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    X_a_test, X_b_test, antigen_test, y_test, label_scaler = prepare_dataset(
        file_path, antiberty_model, esm_model, alphabet, device
    )

    save_dataset((X_a_test, X_b_test, antigen_test, y_test, label_scaler), test_path)
