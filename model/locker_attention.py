import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .attention import *


class LockerAttention(nn.Module):
    def __init__(self, h, h_l, d_k, kernel_size=3):
        super().__init__()
        self.h_g = h - h_l
        self.h_l = h_l
        self.d_k = d_k
        self.kernel_size = kernel_size
        self.convs = nn.ModuleList([self.init_conv(d_k, kernel_size) for _ in range(self.h_l)])

    def init_conv(self, channels, kernel_size=3):
        assert (kernel_size - 1) % 2 == 0
        kernel_size = int(kernel_size)
        return nn.Sequential(
            torch.nn.Conv1d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=kernel_size,
                padding=(kernel_size-1)//2
            ),
            torch.nn.ReLU()
        )

    def forward(self, query, key, value, mask=None, dropout=None):
        b, h, l, d_k = query.size()
        query_g, key_g, value_g = query[:, :self.h_g, ...], key[:, :self.h_g, ...], value[:, :self.h_g, ...]
        query_l, key_l, value_l = query[:, self.h_g:, ...], key[:, self.h_g:, ...], value[:, self.h_g:, ...]

        scores_g = torch.matmul(query_g, key_g.transpose(-2, -1)) \
            / math.sqrt(query_g.size(-1))
        if mask is not None:
            scores_g = scores_g.masked_fill(mask == 0, -1e9)
        
        p_attn_g = F.softmax(scores_g, dim=-1)
        if dropout is not None:
            p_attn_g = dropout(p_attn_g)
        value_g = torch.matmul(p_attn_g, value_g)

        value_l = torch.cat([self.convs[i](value_l[:, i, ...].squeeze().permute( \
            0, 2, 1)).unsqueeze(1).permute(0, 1, 3, 2) for i in range(self.h_l)], dim=1) 
        if dropout is not None:
            value_l = dropout(value_l)
        return torch.cat([value_g, value_l], dim=1)


class LockerMultiHeadedAttention(nn.Module):
    def __init__(self, h, h_l, d_model, kernel_size, head_size=None, dropout=0.1):
        super().__init__()
        assert d_model % h == 0

        self.h = h
        self.h_l = h_l
        self.d_k = d_model // h
        assert self.h > self.h_l and self.h_l > 0

        if head_size is not None:
            self.head_size = head_size
        else:
            self.head_size = d_model // h

        self.linear_layers = nn.ModuleList(
            [nn.Linear(d_model, self.h * self.head_size) for _ in range(3)])
        self.attention = LockerAttention(h, h_l, self.d_k, kernel_size)
        self.dropout = nn.Dropout(p=dropout)
        self.output_linear = nn.Linear(self.h * self.head_size, d_model)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # 1) do all the linear projections in batch from d_model => h x d_k
        query, key, value = [l(x).view(batch_size, -1, self.h, self.head_size).transpose(1, 2)
                             for l, x in zip(self.linear_layers, (query, key, value))]
        
        # 2) apply attention on all the projected vectors in batch.
        x = self.attention(
            query, key, value, mask=mask, dropout=self.dropout)

        # 3) "concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(
            batch_size, -1, self.h * self.head_size)
        return self.output_linear(x)


class LockerTransformerBlock(nn.Module):
    def __init__(self, hidden, attn_heads, local_heads, kernel_size, head_size, feed_forward_hidden, dropout, attn_dropout=0.1):
        super().__init__()
        self.attention = LockerMultiHeadedAttention(
            h=attn_heads, h_l=local_heads, d_model=hidden, kernel_size=kernel_size, head_size=head_size, dropout=attn_dropout)
        self.feed_forward = PositionwiseFeedForward(
            d_model=hidden, d_ff=feed_forward_hidden)
        self.input_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.output_sublayer = SublayerConnection(size=hidden, dropout=dropout)

    def forward(self, x, mask):
        x = self.input_sublayer(
            x, lambda _x: self.attention.forward(_x, _x, _x, mask=mask))
        x = self.output_sublayer(x, self.feed_forward)
        return x