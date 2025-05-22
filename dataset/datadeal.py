
import torch
import pandas as pd
from antiberty import AntiBERTyRunner
import esm
from sklearn.preprocessing import StandardScaler
import torch.nn.utils.rnn as rnn_utils
import torch.nn.functional as F
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


# 使用 AntiBERTy 处理抗体序列（重链和轻链）训练时用的
def process_antibody_sequences(antibody_sequences_a, antibody_sequences_b, antiberty_model):
    embeddings_a = []
    embeddings_b = []
    for heavy_seq, light_seq in zip(antibody_sequences_a, antibody_sequences_b):
        embedding_a, _ = antiberty_model.embed([light_seq], return_attention=True)  # 生成轻链嵌入
        embedding_b, _ = antiberty_model.embed([heavy_seq], return_attention=True)  # 生成重链嵌入
        embeddings_a.append(embedding_a[0].clone().detach())  # 取出第一个样本并转换为张量
        embeddings_b.append(embedding_b[0].clone().detach())  # 取出第一个样本并转换为张量

    # 填充序列以使它们具有相同的长度
    embeddings_a_padded = rnn_utils.pad_sequence(embeddings_a, batch_first=True)
    embeddings_b_padded = rnn_utils.pad_sequence(embeddings_b, batch_first=True)
    # 将张量转换到设备上（如果适用）
    return embeddings_a_padded.to(device), embeddings_b_padded.to(device)

def prepare_dataset(file_path, antiberty_model, esm_model, alphabet, device):
# def prepare_dataset(test_file_path, antiberty_model, esm_model, alphabet, device):
    # # 1. 加载数据带有亲和力值
    antibody_sequences_a, antibody_sequences_b, antigen_sequences, delta_g = load_data_from_tsv(file_path)
    # antibody_sequences_a, antibody_sequences_b, antigen_sequences = load_data_from_tsv(test_file_path)

    # 2. 生成抗体嵌入
    embeddings_a, embeddings_b = process_antibody_sequences(antibody_sequences_a, antibody_sequences_b, antiberty_model)
    pwaa_a = pwaa_encode_sequence(antibody_sequences_a, L=5)  # 计算轻链的PWAA特征
    pwaa_b = pwaa_encode_sequence(antibody_sequences_b, L=5)  # 计算重链的PWAA特征
    print("Shape of pwaa_a:", pwaa_a.shape)
    print("Shape of pwaa_b:", pwaa_b.shape)
    print("Shape of embeddings_a:", embeddings_a.shape)
    print("Shape of embeddings_b:", embeddings_b.shape)

    # 归一化抗体嵌入（轻链和重链）
    embeddings_a_numpy = embeddings_a.detach().cpu().numpy()  # 轻链嵌入
    embeddings_b_numpy = embeddings_b.detach().cpu().numpy()  # 重链嵌入

    # 初始化标准化器
    scaler_a = StandardScaler()  # 标准化轻链嵌入
    scaler_b = StandardScaler()  # 标准化重链嵌入

    # 展平嵌入数据以符合标准化的要求
    embeddings_a_flat = embeddings_a_numpy.reshape(-1, embeddings_a_numpy.shape[-1])  # 展平为 (样本数, 特征数)
    embeddings_b_flat = embeddings_b_numpy.reshape(-1, embeddings_b_numpy.shape[-1])  # 展平为 (样本数, 特征数)

    # 进行标准化
    embeddings_a_scaled = scaler_a.fit_transform(embeddings_a_flat)  # 标准化轻链嵌入
    embeddings_b_scaled = scaler_b.fit_transform(embeddings_b_flat)  # 标准化重链嵌入

    # 重新调整形状为原始张量的形状
    embeddings_a_scaled_tensor = torch.tensor(embeddings_a_scaled, dtype=torch.float32).reshape(embeddings_a_numpy.shape)  # 还原为 (batch_size, seq_length, embedding_dim)
    embeddings_b_scaled_tensor = torch.tensor(embeddings_b_scaled, dtype=torch.float32).reshape(embeddings_b_numpy.shape)  # 还原为 (batch_size, seq_length, embedding_dim)

    # 将所有张量移动到指定设备
    pwaa_a = pwaa_a.to(device)
    pwaa_b = pwaa_b.to(device)
    embeddings_a_scaled_tensor = embeddings_a_scaled_tensor.to(device)
    embeddings_b_scaled_tensor = embeddings_b_scaled_tensor.to(device)

    # 找到最大序列长度
    max_seqa_length = max(pwaa_a.shape[1], embeddings_a_scaled_tensor.shape[1])
    max_seqb_length = max(pwaa_b.shape[1], embeddings_b_scaled_tensor.shape[1])

    # 对PWAA特征和嵌入特征进行填充
    pwaa_a_padded = torch.nn.functional.pad(pwaa_a, (0, 0, 0, max_seqa_length - pwaa_a.shape[1]), "constant", 0)
    pwaa_b_padded = torch.nn.functional.pad(pwaa_b, (0, 0, 0, max_seqb_length - pwaa_b.shape[1]), "constant", 0)
    embeddings_a_padded = torch.nn.functional.pad(embeddings_a_scaled_tensor, (0, 0, 0, max_seqa_length - embeddings_a_scaled_tensor.shape[1]), "constant", 0)
    embeddings_b_padded = torch.nn.functional.pad(embeddings_b_scaled_tensor, (0, 0, 0, max_seqb_length - embeddings_b_scaled_tensor.shape[1]), "constant", 0)

    # 融合PWAA特征和嵌入特征
    embeddings_a_fused = torch.cat((pwaa_a_padded, embeddings_a_padded), dim=2)
    embeddings_b_fused = torch.cat((pwaa_b_padded, embeddings_b_padded), dim=2)
    print("Shape of embeddings_a_fused:", embeddings_a_fused.shape)
    print("Shape of embeddings_b_fused:", embeddings_b_fused.shape)

    # 3. 生成抗原嵌入并融合PWAA特征
    antigen_embeddings = process_antigen_sequences(antigen_sequences, esm_model, alphabet)  # 处理抗原序列

    pwaa_antigen = pwaa_encode_sequence(antigen_sequences, L=3)  # 计算抗原序列的PWAA特征
    print("Shape of pwaa_antigen:", pwaa_antigen.shape)
    print("Shape of antigen_embeddings:", antigen_embeddings.shape)
    # 特征融合

    antigen_embed_dim = torch.cat((pwaa_antigen, antigen_embeddings), dim=2)  # 合并在第三维度
    print("Shape of antigen_embed_dim:", antigen_embed_dim.shape)

    # 4. 转换亲和力值为张量 (delta_g 直接用于回归任务)
    delta_g_tensor = torch.tensor(delta_g, dtype=torch.float32, device=device)

    #5. 返回所有数据作为训练集
    return embeddings_a_fused, embeddings_b_fused, antigen_embed_dim, delta_g_tensor

# 保存测试数据集
def save_dataset(test_data, test_path):
    # 将训练集保存为 .pt 文件
    torch.save({
        'X_a_test': test_data[0],
        'X_b_test': test_data[1],
        'antigen_test': test_data[2],
        'y_test': test_data[3]
    }, test_path)
    print(f"训练集保存到 {test_path}")



# 主函数
if __name__ == "__main__":
    # file_path = '/tmp/AbAgCDR/data/final_dataset_train.tsv'  # 更新为您的数据文件路径
    # train_path = '/tmp/AbAgCDR/data/train3.1_data_xin.pt'  # 更新为保存的训练集路径
    # file_path = '/tmp/AbAgCDR/data/pairs_seq_sabdab.tsv'  # 更新为您的数据文件路径
    # test_path = '/tmp/AbAgCDR/data/sabdab1_test_data.pt'  # 更新为保存的训练集路径
    file_path = '/tmp/AbAgCDR/data/pairs_seq_benchmark1.tsv'  # 更新为您的数据文件路径
    test_path = '/tmp/AbAgCDR/data/pairs_seq_benchmark1.1_test.pt'  # 更新为保存的训练集路径
    # file_path = '/tmp/AbAgCDR/data/pairs_seq_skempi.tsv'  # 更新为您的数据文件路径
    # test_path = '/tmp/AbAgCDR/data/pairs_seq_skempi1.1_test.pt'  # 更新为保存的训练集路径
    # test_file_path = '/tmp/AbAgCDR/data/pairs_seq_benchmark.tsv'  # 更新为您的数据文件路径
    # test_path = '/tmp/AbAgCDR/data/benchmark1.3.pt'  # 更新为保存的训练集路径
    # 初始化 AntiBERTy 模型
    antiberty_model = AntiBERTyRunner()  # 使用您的 AntiBERTy 模型路径
    # 获取当前设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 准备数据集
    # X_a_train, X_b_train, antigen_train, y_train = prepare_dataset(file_path, antiberty_model, esm_model, alphabet, device)
    # 准备测试数据集
    X_a_test, X_b_test, antigen_test, y_test = prepare_dataset(file_path, antiberty_model, esm_model, alphabet, device)
    # 保存数据集
    # save_dataset((X_a_train, X_b_train, antigen_train, y_train), train_path)
    # 保存测试数据集
    save_dataset((X_a_test, X_b_test, antigen_test, y_test), test_path)

    # # 准备数据集测试没有真实亲和力值的
    # X_a_test, X_b_test, antigen_test = prepare_dataset(test_file_path, antiberty_model, esm_model, alphabet, device)
    # # 保存测试数据集
    # save_test_dataset((X_a_test, X_b_test, antigen_test), test_path)