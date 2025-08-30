import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Temporal_Average_Pooling(nn.Module):
    def __init__(self, dim):
        super().__init__()

    def forward(self, x):
        # x: (B, T, D)

        # 1) 均值 μ：沿 time-step 維度平均
        mu = x.mean(dim=1)             # → (B, D)

        return mu


class Temporal_Statistics_Pooling(nn.Module):
    def __init__(self, dim):
        super().__init__()

    def forward(self, x):
        # x: (B, T, D)

        # 1) 均值 μ：沿 time-step 維度平均
        mu = x.mean(dim=1)             # → (B, D)

        # 2) 標準差 σ：sqrt(E[h^2] - μ^2)
        #    用 mean 而不是 sum/T 也可以，結果一樣
        sigma2 = (x**2).mean(dim=1) - mu.pow(2)      # → (B, D)
        sigma = torch.sqrt(sigma2.clamp(min=1e-5))   # → (B, D)

        # 3) 串接成 (B, 2*D)
        out = torch.cat([mu, sigma], dim=1)
        return out

class Self_Attentive_Pooling(nn.Module):
    def __init__(self, dim):
        super().__init__()        
        # Self-Attentive Pooling
        self.sap_linear = nn.Linear(dim, dim)

        # 注意力向量要初始化，不然一開始就可能拿到垃圾值
        self.attention = nn.Parameter(torch.empty(dim, 1))
        nn.init.xavier_uniform_(self.attention)  

    def forward(self, x):
        """
        x: (B, T, D)
        returns: (B, D)
        """

        # 3) 注意力分數計算完全一樣，只是改寫得更直觀：
        h = torch.tanh(self.sap_linear(x))      # → (B, T, D)
        w = torch.matmul(h, self.attention)     # → (B, T, 1)
        w = F.softmax(w.squeeze(-1), dim=1)     # → (B, T)
        w = w.unsqueeze(-1)                     # → (B, T, 1)

        # 4) 帶權和 μ，以及計算加權標準差 σ；加上 clamp(min=1e-5) 避免
        #    由於浮點誤差算出小負值導致 sqrt(<0) 的 NaN
        mu = torch.sum(x * w, dim=1)            # → (B, D)

        return mu

class Attentive_Statistics_Pooling(nn.Module):
    def __init__(self, dim):
        super().__init__()        
        # Self-Attentive Pooling

        self.sap_linear = nn.Linear(dim, dim)
        # 注意力向量要初始化，不然一開始就可能拿到垃圾值
        self.attention = nn.Parameter(torch.empty(dim, 1))
        nn.init.xavier_uniform_(self.attention)  

    def forward(self, x):
        """
        x: (B, T, D)
        returns: (B, 2*D)
        """

        # 3) 注意力分數計算完全一樣，只是改寫得更直觀：
        h = torch.tanh(self.sap_linear(x))      # → (B, T, D)
        w = torch.matmul(h, self.attention)     # → (B, T, 1)
        w = F.softmax(w.squeeze(-1), dim=1)     # → (B, T)
        w = w.unsqueeze(-1)                     # → (B, T, 1)

        # 4) 帶權和 μ，以及計算加權標準差 σ；加上 clamp(min=1e-5) 避免
        #    由於浮點誤差算出小負值導致 sqrt(<0) 的 NaN
        mu = torch.sum(x * w, dim=1)            # → (B, D)
        sigma = torch.sqrt(
            (torch.sum((x**2) * w, dim=1) - mu**2)
            .clamp(min=1e-5)
        )                                       # → (B, D)

        # 5) 串接成 (B, 2*D)
        out = torch.cat([mu, sigma], dim=1)

        return out

class General_Self_Attentive_Pooling(nn.Module):
    def __init__(self, dim, segment_size=1):
        """
        General ASP
        Args:
            dim (int): 輸入特徵的維度 (D)
            segment_size (int): 每個 segment 包含多少個 frame (n)
        """
        super().__init__()
        self.segment_size = segment_size
        
        # Self-Attentive Pooling 的線性層，與原版相同
        self.sap_linear = nn.Linear(dim, dim)
        
        # 注意力向量，與原版相同
        self.attention = nn.Parameter(torch.empty(dim, 1))
        nn.init.xavier_uniform_(self.attention)

    def forward(self, x):
        """
        x: (B, T, D)
        returns: (B, D)
        """
        
        B, T, D = x.shape
        S = self.segment_size

        # 1) Padding: 如果總幀數 T 無法被 segment_size 整除，則在序列尾部補 0
        if T % S != 0:
            padding_needed = S - (T % S)
            # F.pad 的參數是 (左邊補, 右邊補, 上面補, 下面補)，我們要補在 T 維度上
            x = F.pad(x, (0, 0, 0, padding_needed))
        
        new_T = x.shape[1]

        # 2) 分組與局部池化 (Local Pooling):
        #    將 (B, new_T, D) 重塑為 (B, num_segments, segment_size, D)
        #    然後對每個 segment 內的 frame 取平均，得到 segment 的代表性特徵
        x_segments = x.view(B, new_T // S, S, D)
        x_pooled = x_segments.mean(dim=2)       # → (B, num_segments, D)
        
        # --- 注意力計算改在 segment 層級上 ---

        # 3) 注意力分數計算: 現在輸入是 segment 代表性特徵
        h = torch.tanh(self.sap_linear(x_pooled))   # → (B, num_segments, D)
        w = torch.matmul(h, self.attention)         # → (B, num_segments, 1)
        w = F.softmax(w.squeeze(-1), dim=1)         # → (B, num_segments)
        w = w.unsqueeze(-1)                         # → (B, num_segments, 1)

        # 4) 計算加權平均 μ 與加權標準差 σ
        mu = torch.sum(x_pooled * w, dim=1)         # → (B, D)

        return mu

class General_Attentive_Statistics_Pooling(nn.Module):
    def __init__(self, dim, segment_size=1):
        """
        General ASP
        Args:
            dim (int): 輸入特徵的維度 (D)
            segment_size (int): 每個 segment 包含多少個 frame (n)
        """
        super().__init__()
        self.segment_size = segment_size
        
        # Self-Attentive Pooling 的線性層，與原版相同
        self.sap_linear = nn.Linear(dim, dim)
        
        # 注意力向量，與原版相同
        self.attention = nn.Parameter(torch.empty(dim, 1))
        nn.init.xavier_uniform_(self.attention)

    def forward(self, x):
        """
        x: (B, T, D)
        returns: (B, 2*D)
        """
        
        B, T, D = x.shape
        S = self.segment_size

        # 1) Padding: 如果總幀數 T 無法被 segment_size 整除，則在序列尾部補 0
        if T % S != 0:
            padding_needed = S - (T % S)
            # F.pad 的參數是 (左邊補, 右邊補, 上面補, 下面補)，我們要補在 T 維度上
            x = F.pad(x, (0, 0, 0, padding_needed))
        
        new_T = x.shape[1]

        # 2) 分組與局部池化 (Local Pooling):
        #    將 (B, new_T, D) 重塑為 (B, num_segments, segment_size, D)
        #    然後對每個 segment 內的 frame 取平均，得到 segment 的代表性特徵
        x_segments = x.view(B, new_T // S, S, D)
        x_pooled = x_segments.mean(dim=2)       # → (B, num_segments, D)
        
        # --- 注意力計算改在 segment 層級上 ---

        # 3) 注意力分數計算: 現在輸入是 segment 代表性特徵
        h = torch.tanh(self.sap_linear(x_pooled))   # → (B, num_segments, D)
        w = torch.matmul(h, self.attention)         # → (B, num_segments, 1)
        w = F.softmax(w.squeeze(-1), dim=1)         # → (B, num_segments)
        w = w.unsqueeze(-1)                         # → (B, num_segments, 1)

        # 4) 計算加權平均 μ 與加權標準差 σ
        mu = torch.sum(x_pooled * w, dim=1)         # → (B, D)
        sigma = torch.sqrt(
            (torch.sum((x_pooled**2) * w, dim=1) - mu**2)
            .clamp(min=1e-5)
        )                                           # → (B, D)

        # 7) 串接成 (B, 2*D)
        out = torch.cat([mu, sigma], dim=1)

        return out

class Dual_Resolution_Attentive_Pooling(nn.Module):
    """
    結合 Temporal_Average_Pooling 與 General_Self_Attentive_Pooling，
    以可學習的 alpha/beta 加權和形式輸出最終音訊向量。
    """
    def __init__(self, dim, segment_size=1, alpha=1.0, beta=0.0):
        super().__init__()
        # 平均池化分支
        self.tap  = Temporal_Average_Pooling(dim)
        # General Self-Attentive 池化分支
        self.gsap = General_Self_Attentive_Pooling(dim, segment_size=segment_size)
        # 融合係數（可學習）
        self.alpha = nn.Parameter(torch.tensor(alpha))
        self.beta  = nn.Parameter(torch.tensor(beta))
        # 兩分支輸出皆為 (B, D)，維度一致

    def forward(self, x):
        # x: (B, T, D)
        rep_avg  = self.tap(x)    # → (B, D)
        rep_attn = self.gsap(x)   # → (B, D)
        # 加權融合
        out = self.alpha * rep_avg + self.beta * rep_attn
        # a = torch.sigmoid(self.alpha)
        # out = a * rep_avg + (1-a) * rep_attn
        return out

class Dual_Resolution_Statistics_Pooling(nn.Module):
    """
    結合 Temporal_Statistics_Pooling 與 General_Attentive_Statistics_Pooling，
    以可學習的 alpha/beta 加權和形式輸出最終音訊向量 (維度 2*D)。
    """
    def __init__(self, dim, segment_size=1, alpha=1.0, beta=0.0):
        super().__init__()
        # 統計池化分支 (輸出 (B, 2*D))
        self.tsp  = Temporal_Statistics_Pooling(dim)
        # General Attentive Stats 池化分支 (輸出 (B, 2*D))
        self.gasp = General_Attentive_Statistics_Pooling(dim, segment_size=segment_size)
        # 融合係數（可學習）
        self.alpha = nn.Parameter(torch.tensor(alpha))
        self.beta  = nn.Parameter(torch.tensor(beta))

    def forward(self, x):
        # x: (B, T, D)
        rep_stat = self.tsp(x)    # → (B, 2*D)
        rep_attn = self.gasp(x)   # → (B, 2*D)
        # 加權融合
        out = self.alpha * rep_stat + self.beta * rep_attn
        # a = torch.sigmoid(self.alpha)
        # out = a * rep_stat + (1-a) * rep_attn1
        return out

