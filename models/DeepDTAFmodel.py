import torch
import torch.nn as nn
from dataset import PT_FEATURE_SIZE  # 确保这个值匹配你的氨基酸特征维度（如 one-hot=21）

CHAR_PROT_SET_LEN = None  # 不再需要 SMILES 字符集

class Squeeze(nn.Module):
    def forward(self, x: torch.Tensor):
        return x.squeeze(-1)  # 只 squeeze 最后一维（AdaptiveMaxPool1d 输出 (N, C, 1)）

class CDilated(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride=1, d=1):
        super().__init__()
        padding = int((kSize - 1) / 2) * d
        self.conv = nn.Conv1d(nIn, nOut, kSize, stride=stride, padding=padding, bias=False, dilation=d)

    def forward(self, x):
        return self.conv(x)

class DilatedParallelResidualBlockA(nn.Module):  # 修正拼写：Parllel → Parallel
    def __init__(self, nIn, nOut, add=True):
        super().__init__()
        n = int(nOut / 5)
        n1 = nOut - 4 * n
        self.c1 = nn.Conv1d(nIn, n, 1)
        self.br1 = nn.Sequential(nn.BatchNorm1d(n), nn.PReLU())
        self.d1 = CDilated(n, n1, 3, 1, 1)
        self.d2 = CDilated(n, n, 3, 1, 2)
        self.d4 = CDilated(n, n, 3, 1, 4)
        self.d8 = CDilated(n, n, 3, 1, 8)
        self.d16 = CDilated(n, n, 3, 1, 16)
        self.br2 = nn.Sequential(nn.BatchNorm1d(nOut), nn.PReLU())

        self.add = add and (nIn == nOut)

    def forward(self, x):
        out = self.br1(self.c1(x))
        d1 = self.d1(out)
        d2 = self.d2(out)
        d4 = self.d4(out)
        d8 = self.d8(out)
        d16 = self.d16(out)

        add1 = d2
        add2 = add1 + d4
        add3 = add2 + d8
        add4 = add3 + d16

        combine = torch.cat([d1, add1, add2, add3, add4], dim=1)
        if self.add:
            combine = x + combine
        return self.br2(combine)


class TriProtDTA(nn.Module):
    def __init__(self, embed_size=128, out_channels=128):
        super().__init__()
        self.embed = nn.Linear(PT_FEATURE_SIZE, embed_size)  # 共享嵌入层（也可独立）

        # 为三条序列分别创建编码器（结构相同，参数不共享）
        self.encoder_h = self._make_encoder(embed_size, out_channels)
        self.encoder_l = self._make_encoder(embed_size, out_channels)
        self.encoder_a = self._make_encoder(embed_size, out_channels)

        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Sequential(
            nn.Linear(out_channels * 3, 128),
            nn.Dropout(0.5),
            nn.PReLU(),
            nn.Linear(128, 64),
            nn.Dropout(0.5),
            nn.PReLU(),
            nn.Linear(64, 1),
            nn.PReLU()
        )

    def _make_encoder(self, in_ch, out_ch):
        layers = []
        ic = in_ch
        for oc in [32, 64, out_ch]:
            layers.append(DilatedParallelResidualBlockA(ic, oc))
            ic = oc
        layers.append(nn.AdaptiveMaxPool1d(1))
        layers.append(Squeeze())
        return nn.Sequential(*layers)

    def forward(self, heavy, light, antigen):
        """
        Args:
            heavy:   (N, Lh, D)
            light:   (N, Ll, D)
            antigen: (N, La, D)
        Returns:
            output: (N, 1)
        """
        # 嵌入 + 转置为 (N, C, L)
        h = self.embed(heavy).transpose(1, 2)
        l = self.embed(light).transpose(1, 2)
        a = self.embed(antigen).transpose(1, 2)

        # 编码
        h_feat = self.encoder_h(h)   # (N, 128)
        l_feat = self.encoder_l(l)   # (N, 128)
        a_feat = self.encoder_a(a)   # (N, 128)

        # 融合
        fused = torch.cat([h_feat, l_feat, a_feat], dim=1)  # (N, 384)
        fused = self.dropout(fused)
        out = self.classifier(fused)  # (N, 1)
        return out