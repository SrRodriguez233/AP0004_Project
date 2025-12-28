import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import random

class RelativePositionBias(nn.Module):
    """
    简化版 T5 风格相对位置偏置：给每个注意力 head 添加一个基于相对距离的可学习偏置。
    """
    def __init__(self, num_buckets=32, max_distance=128, num_heads=8):
        super().__init__()
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.num_heads = num_heads
        self.relative_attention_bias = nn.Embedding(num_buckets, num_heads)

    def _relative_position_bucket(self, relative_position):
        # relative_position: [L_q, L_k]
        sign = (relative_position > 0).to(torch.long)
        n = torch.abs(relative_position)
        max_exact = self.num_buckets // 2
        is_small = n < max_exact
        val_if_large = max_exact + (
            (torch.log(n.float() / max_exact) / math.log(self.max_distance / max_exact) * (self.num_buckets - max_exact)).to(torch.long)
        )
        val_if_large = torch.clamp(val_if_large, max=max_exact + self.num_buckets - max_exact - 1)
        buckets = torch.where(is_small, n, val_if_large)
        buckets = buckets + sign * max_exact
        # 保护：确保 bucket 索引始终在 [0, num_buckets-1] 范围内，避免 Embedding 越界
        buckets = torch.clamp(buckets, min=0, max=self.num_buckets - 1)
        return buckets

    def forward(self, q_len, k_len, device):
        # 生成 [L_q, L_k] 的相对位置矩阵
        context_position = torch.arange(q_len, dtype=torch.long, device=device)[:, None]
        memory_position = torch.arange(k_len, dtype=torch.long, device=device)[None, :]
        relative_position = memory_position - context_position  # [L_q, L_k]
        rp_bucket = self._relative_position_bucket(relative_position)
        # [L_q, L_k, num_heads]
        values = self.relative_attention_bias(rp_bucket)
        return values.permute(2, 0, 1)  # [num_heads, L_q, L_k]

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.scale = dim ** -0.5
        self.eps = eps
        self.g = nn.Parameter(torch.ones(dim)) # 可学习的缩放参数 gamma

    def forward(self, x):
        # x: [..., dim]
        # 计算 RMS
        norm = torch.norm(x, dim=-1, keepdim=True) * self.scale
        return x / (norm + self.eps) * self.g

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)  # [max_len, d_model]

    def forward(self, x):
        # x: [B, L, D]
        L = x.size(1)
        return x + self.pe[:L].unsqueeze(0)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1, use_relative_bias=False, num_buckets=32, max_distance=128):
        super().__init__()
        assert d_model % nhead == 0
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.use_relative_bias = use_relative_bias

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

        if use_relative_bias:
            self.rel_bias = RelativePositionBias(num_buckets=num_buckets, max_distance=max_distance, num_heads=nhead)

    def forward(self, query, key, value, attn_mask=None, key_padding_mask=None):
        B, L_q, _ = query.size()
        L_k = key.size(1)

        # 确保掩码为 bool 类型，避免 masked_fill 报 dtype 错误
        if attn_mask is not None and attn_mask.dtype is not torch.bool:
            attn_mask = attn_mask.to(torch.bool)
        if key_padding_mask is not None and key_padding_mask.dtype is not torch.bool:
            key_padding_mask = key_padding_mask.to(torch.bool)

        q = self.q_proj(query).view(B, L_q, self.nhead, self.head_dim).transpose(1, 2)  # [B, H, L_q, D]
        k = self.k_proj(key).view(B, L_k, self.nhead, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(B, L_k, self.nhead, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # [B, H, L_q, L_k]

        if self.use_relative_bias:
            rel = self.rel_bias(L_q, L_k, device=query.device)  # [H, L_q, L_k]
            scores = scores + rel.unsqueeze(0)

        if attn_mask is not None:
            # attn_mask: bool, True=keep, False=mask. Shape [L_q, L_k] or [B, L_q, L_k]
            if attn_mask.dim() == 2:
                scores = scores.masked_fill(~attn_mask.unsqueeze(0).unsqueeze(1), float('-inf'))
            else:
                scores = scores.masked_fill(~attn_mask.unsqueeze(1), float('-inf'))

        if key_padding_mask is not None:
            # key_padding_mask: bool [B, L_k], True=keep (non-pad)
            scores = scores.masked_fill(~key_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf'))

        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)  # [B, H, L_q, D]
        out = out.transpose(1, 2).contiguous().view(B, L_q, self.d_model)
        out = self.out_proj(out)
        return out

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout, norm_type='layer', use_relative_bias=False):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, nhead, dropout=dropout, use_relative_bias=use_relative_bias)

        if norm_type == 'rms':
            self.norm1 = RMSNorm(d_model)
            self.norm2 = RMSNorm(d_model)
        else:
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        src2 = self.norm1(src)
        src2 = self.self_attn(src2, src2, src2, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2)

        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout, norm_type='layer', use_relative_bias=False):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, nhead, dropout=dropout, use_relative_bias=use_relative_bias)
        self.cross_attn = MultiHeadAttention(d_model, nhead, dropout=dropout, use_relative_bias=False)

        if norm_type == 'rms':
            self.norm1 = RMSNorm(d_model)
            self.norm2 = RMSNorm(d_model)
            self.norm3 = RMSNorm(d_model)
        else:
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)
            self.norm3 = nn.LayerNorm(d_model)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        # Masked self-attention
        t2 = self.norm1(tgt)
        t2 = self.self_attn(t2, t2, t2, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)
        tgt = tgt + self.dropout1(t2)

        # Cross attention
        t2 = self.norm2(tgt)
        t2 = self.cross_attn(t2, memory, memory, attn_mask=memory_mask, key_padding_mask=memory_key_padding_mask)
        tgt = tgt + self.dropout2(t2)

        # FFN
        t2 = self.norm3(tgt)
        t2 = self.linear2(self.dropout(F.relu(self.linear1(t2))))
        tgt = tgt + self.dropout3(t2)
        return tgt

class TransformerNMT(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=256, nhead=4, num_encoder_layers=4, num_decoder_layers=4,
                 dim_feedforward=512, dropout=0.1, norm_type='layer', use_relative_bias=False, use_abs_pos=True):
        super().__init__()
        self.src_embed = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embed = nn.Embedding(tgt_vocab_size, d_model)
        self.use_abs_pos = use_abs_pos
        self.pos_encoder = SinusoidalPositionalEncoding(d_model) if use_abs_pos else None

        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, norm_type=norm_type, use_relative_bias=use_relative_bias)
            for _ in range(num_encoder_layers)
        ])
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, norm_type=norm_type, use_relative_bias=use_relative_bias)
            for _ in range(num_decoder_layers)
        ])
        self.generator = nn.Linear(d_model, tgt_vocab_size)

    def _generate_square_subsequent_mask(self, sz: int, device):
        # True = keep, False = mask future tokens
        upper = torch.triu(torch.ones(sz, sz, device=device, dtype=torch.bool), diagonal=1)
        return ~upper

    def encode(self, src, src_mask=None, src_key_padding_mask=None):
        x = self.src_embed(src)
        if self.pos_encoder is not None:
            x = self.pos_encoder(x)
        for layer in self.encoder_layers:
            x = layer(x, src_mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        return x

    def decode(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        x = self.tgt_embed(tgt)
        if self.pos_encoder is not None:
            x = self.pos_encoder(x)
        for layer in self.decoder_layers:
            x = layer(x, memory, tgt_mask=tgt_mask, memory_mask=memory_mask,
                      tgt_key_padding_mask=tgt_key_padding_mask, memory_key_padding_mask=memory_key_padding_mask)
        return x

    def forward(self, src, src_key_padding_mask, tgt_inp, tgt_key_padding_mask):
        memory = self.encode(src, src_key_padding_mask=src_key_padding_mask)
        tgt_mask = self._generate_square_subsequent_mask(tgt_inp.size(1), device=src.device)
        out = self.decode(tgt_inp, memory, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask,
                          memory_key_padding_mask=src_key_padding_mask)
        logits = self.generator(out)
        return logits