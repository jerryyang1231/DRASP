from torch import nn
import torch
import torch.nn.functional as F

class Temporal_Average_Pooling(nn.Module):
    def __init__(self, dim, eps: float = 1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        """
        x:    [B, T, D]
        mask: [B, T, 1], dtype float, 值為 1.0 表示 valid frame，0.0 表示要忽略
        """
        if mask is None:
            # fallback: unmasked pooling
            mu = x.mean(dim=1)  # → [B, D]
        else:
            # 計算有效 frame 的數量： [B, 1]
            mask_sum = mask.sum(dim=1).clamp(min=1.0)
            # masked mean: sum(x * mask) / sum(mask)
            mu = (x * mask).sum(dim=1) / mask_sum

        return mu

class Temporal_Statistics_Pooling(nn.Module):
    def __init__(self, dim, eps: float = 1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        """
        x:    [B, T, D]
        mask: [B, T, 1], dtype float, 值為 1.0 表示 valid frame，0.0 表示要忽略
        """
        if mask is None:
            # fallback: unmasked pooling
            mu = x.mean(dim=1)  # → [B, D]
            sigma2 = (x**2).mean(dim=1) - mu.pow(2)
        else:
            # 計算有效 frame 的數量： [B, 1]
            mask_sum = mask.sum(dim=1).clamp(min=1.0)
            # masked mean: sum(x * mask) / sum(mask)
            mu = (x * mask).sum(dim=1) / mask_sum
            # masked E[x^2]: sum(x^2 * mask) / sum(mask)
            ex2 = (x**2 * mask).sum(dim=1) / mask_sum
            sigma2 = ex2 - mu.pow(2)

        # sqrt of variance
        sigma = torch.sqrt(sigma2.clamp(min=self.eps))

        # concat → [B, 2*D]
        out = torch.cat([mu, sigma], dim=1)
        return out

class Self_Attentive_Pooling(nn.Module):
    def __init__(self, dim, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        # Self-Attentive Pooling
        self.sap_linear = nn.Linear(dim, dim)
        self.attention = nn.Parameter(torch.empty(dim, 1))
        nn.init.xavier_uniform_(self.attention)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):                
        """
        x:    [B, T, D]
        mask: [B, T, 1] bool/float, 1=valid frame, 0=pad
        """
        # 1) attention scores
        h = torch.tanh(self.sap_linear(x))         # [B, T, D]
        w = torch.matmul(h, self.attention).squeeze(-1)  # [B, T]

        # 2) mask before softmax
        if mask is not None:
            mask2 = mask.squeeze(-1).to(torch.bool)     # [B, T]
            w = w.masked_fill(~mask2, float('-1e9'))
        w = torch.softmax(w, dim=1).unsqueeze(-1)       # [B, T, 1]

        # 3) weighted mean
        mu = torch.sum(x * w, dim=1)                   # [B, D]

        return mu

class Attentive_Statistics_Pooling(nn.Module):
    def __init__(self, dim, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        # Self-Attentive Pooling
        self.sap_linear = nn.Linear(dim, dim)
        self.attention = nn.Parameter(torch.empty(dim, 1))
        nn.init.xavier_uniform_(self.attention)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):                
        """
        x:    [B, T, D]
        mask: [B, T, 1] bool/float, 1=valid frame, 0=pad
        """
        # 1) attention scores
        h = torch.tanh(self.sap_linear(x))         # [B, T, D]
        w = torch.matmul(h, self.attention).squeeze(-1)  # [B, T]

        # 2) mask before softmax
        if mask is not None:
            mask2 = mask.squeeze(-1).to(torch.bool)     # [B, T]
            w = w.masked_fill(~mask2, float('-1e9'))
        w = torch.softmax(w, dim=1).unsqueeze(-1)       # [B, T, 1]

        # 3) weighted mean
        mu = torch.sum(x * w, dim=1)                   # [B, D]
        # weighted second moment
        ex2 = torch.sum((x**2) * w, dim=1) / 1.0       # [B, D]
        sigma2 = ex2 - mu.pow(2)
        sigma = torch.sqrt(sigma2.clamp(min=self.eps))    # [B, D]

        # 4) concat
        out = torch.cat([mu, sigma], dim=1)               # [B, 2*D]
        return out

class General_Self_Attentive_Pooling(nn.Module):
    def __init__(self, dim, segment_size=1, eps: float = 1e-5):
        super().__init__()
        self.segment_size = segment_size
        self.eps = eps
        # Self-Attentive Pooling 的線性層 + attention 向量
        self.sap_linear = nn.Linear(dim, dim)
        self.attention = nn.Parameter(torch.empty(dim, 1))
        nn.init.xavier_uniform_(self.attention)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        """
        x:    [B, T, D]
        mask: [B, T, 1] 或 None，1=valid, 0=pad
        returns: [B, D]
        """
        B, T, D = x.shape
        S = self.segment_size

        # 1) 自動 padding 到可被 S 整除
        pad_len = (S - (T % S)) % S
        if pad_len > 0:
            x = F.pad(x, (0, 0, 0, pad_len), value=0.0)
            if mask is not None:
                mask = F.pad(mask, (0, 0, 0, pad_len), value=0.0)
        new_T = T + pad_len
        N = new_T // S

        # 2) reshape 分段
        x_seg = x.view(B, N, S, D)  # [B, N, S, D]
        if mask is not None:
            m_seg = mask.view(B, N, S, 1).to(x.dtype)
            seg_counts = m_seg.sum(dim=2).clamp(min=self.eps)
            x_pooled = (x_seg * m_seg).sum(dim=2) / seg_counts  # [B, N, D]
        else:
            x_pooled = x_seg.mean(dim=2)  # [B, N, D]

        # 3) segment-level attention
        h = torch.tanh(self.sap_linear(x_pooled))     # [B, N, D]
        w = torch.matmul(h, self.attention).squeeze(-1)  # [B, N]
        if mask is not None:
            valid_seg = (seg_counts.squeeze(-1) > self.eps)
            w = w.masked_fill(~valid_seg, float('-1e9'))
        w = F.softmax(w, dim=1).unsqueeze(-1)  # [B, N, 1]

        # 4) 全局加權統計
        mu = torch.sum(x_pooled * w, dim=1)           # [B, D]

        return mu

class General_Attentive_Statistics_Pooling(nn.Module):
    def __init__(self, dim, segment_size=1, eps: float = 1e-5):
        super().__init__()
        self.segment_size = segment_size
        self.eps = eps
        # Self-Attentive Pooling 的線性層 + attention 向量
        self.sap_linear = nn.Linear(dim, dim)
        self.attention = nn.Parameter(torch.empty(dim, 1))
        nn.init.xavier_uniform_(self.attention)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        """
        x:    [B, T, D]
        mask: [B, T, 1] 或 None，1=valid, 0=pad
        returns: [B, 2*D]
        """
        B, T, D = x.shape
        S = self.segment_size

        # 1) 自動 padding 到可被 S 整除
        pad_len = (S - (T % S)) % S
        if pad_len > 0:
            x = F.pad(x, (0, 0, 0, pad_len), value=0.0)
            if mask is not None:
                mask = F.pad(mask, (0, 0, 0, pad_len), value=0.0)
        new_T = T + pad_len
        N = new_T // S

        # 2) reshape 分段
        x_seg = x.view(B, N, S, D)  # [B, N, S, D]
        if mask is not None:
            m_seg = mask.view(B, N, S, 1).to(x.dtype)
            seg_counts = m_seg.sum(dim=2).clamp(min=self.eps)
            x_pooled = (x_seg * m_seg).sum(dim=2) / seg_counts  # [B, N, D]
        else:
            x_pooled = x_seg.mean(dim=2)  # [B, N, D]

        # 3) segment-level attention
        h = torch.tanh(self.sap_linear(x_pooled))     # [B, N, D]
        w = torch.matmul(h, self.attention).squeeze(-1)  # [B, N]
        if mask is not None:
            valid_seg = (seg_counts.squeeze(-1) > self.eps)
            w = w.masked_fill(~valid_seg, float('-1e9'))
        w = F.softmax(w, dim=1).unsqueeze(-1)  # [B, N, 1]

        # 4) 全局加權統計
        mu = torch.sum(x_pooled * w, dim=1)           # [B, D]
        ex2 = torch.sum((x_pooled ** 2) * w, dim=1)
        sigma = torch.sqrt((ex2 - mu**2).clamp(min=self.eps))  # [B, D]

        # 4) concat
        out = torch.cat([mu, sigma], dim=1)               # [B, 2*D]
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

    def forward(self, x, mask=None):
        # x: (B, T, D)
        rep_avg  = self.tap(x, mask)    # → (B, D)
        rep_attn = self.gsap(x, mask)   # → (B, D)
        # 加權融合
        return self.alpha * rep_avg + self.beta * rep_attn

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

    def forward(self, x, mask=None):
        # x: (B, T, D)
        rep_stat = self.tsp(x, mask)    # → (B, 2*D)
        rep_attn = self.gasp(x, mask)   # → (B, 2*D)
        # 加權融合
        return self.alpha * rep_stat + self.beta * rep_attn
