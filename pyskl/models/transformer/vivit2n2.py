import torch
import torch.nn as nn

from torch import einsum
from torch import Tensor
from einops import rearrange
from ...utils import Graph
from ..builder import BACKBONES
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, attention_dropout=0.1, projection_dropout=0.1):
        super().__init__()
        self.heads = num_heads
        head_dim = dim // self.heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.attn_drop = nn.Dropout(attention_dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(projection_dropout)

    def forward(self, x):
        B, N, C = x.shape

        qkv = self.qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        q = q * self.scale

        attn = einsum('b h i d, b h j d -> b h i j', q, k)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = einsum('b h i j, b h j d -> b h i d', attn, v)
        x = rearrange(x, 'b h n d -> b n (h d)')

        return self.proj_drop(self.proj(x))


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x):
        batch, drop_prob, device, dtype = x.shape[0], self.drop_prob, x.device, x.dtype

        if drop_prob <= 0. or not self.training:
            return x

        keep_prob = 1 - self.drop_prob
        shape = (batch, *((1,) * (x.ndim - 1)))

        keep_mask = torch.zeros(shape, device=device).float().uniform_(0, 1) < keep_prob
        output = x.div(keep_prob) * keep_mask.float()
        return output


class PositionalEncoding(nn.Module):
    def __init__(self, hidden_dim, dropout=0.1, max_position_embeddings=512):
        super().__init__()
        self.register_buffer("position_ids", torch.arange(max_position_embeddings).expand((1, -1)))
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor):
        """
        :param x: (B, T, C)
        :return: (B, T, C)
        """
        seq_length = x.size(1)
        position_ids = self.position_ids[:, :seq_length]
        x = x + self.position_embeddings(position_ids)
        return self.dropout(x)


def _init_weights(module):
    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


class TransformerEncoderLayer(nn.Module):
    """
    Inspired by torch.nn.TransformerEncoderLayer and
    rwightman's timm package.
    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 attention_dropout=0.1, drop_path_rate=0.1):
        super().__init__()

        self.pre_norm = nn.LayerNorm(d_model)
        self.self_attn = Attention(dim=d_model, num_heads=nhead,
                                   attention_dropout=attention_dropout, projection_dropout=dropout)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout2 = nn.Dropout(dropout)

        self.drop_path = DropPath(drop_path_rate)

        self.activation = F.gelu

    def forward(self, src, *args, **kwargs):
        src = src + self.drop_path(self.self_attn(self.pre_norm(src)))
        src = self.norm1(src)
        src2 = self.linear2(self.dropout1(self.activation(self.linear1(src))))
        src = src + self.drop_path(self.dropout2(src2))
        return src


@BACKBONES.register_module()
class ViViT2n2(nn.Module):
    def __init__(
            self,
            graph_cfg,
            in_channels=3,
            dim=256,
            depth=5,
            heads=4,
            dropout=0.1,
            scale_dim=4,
            max_position_embeddings_1=26,
            max_position_embeddings_2=65,
            attention_dropout=0.1,
            stochastic_depth_rate=0.1,
    ):
        super().__init__()

        graph = Graph(**graph_cfg)
        A = torch.tensor(graph.A, dtype=torch.float32, requires_grad=False)
        self.data_bn = nn.BatchNorm1d(in_channels * A.size(1))
        self.norm = nn.LayerNorm(dim)
        dpr = [x.item() for x in torch.linspace(0, stochastic_depth_rate, depth)]
        dpr_iter = iter(dpr)
        # self.Transformer = nn.ModuleList([])
        # for _ in range(depth):
        #     self.Transformer.append(nn.ModuleList([
        #         TransformerEncoderLayer(d_model=dim, nhead=heads,
        #                                 dim_feedforward=dim * scale_dim, dropout=dropout,
        #                                 attention_dropout=attention_dropout, drop_path_rate=next(dpr_iter))
        #     ]))

        self.blocks = nn.ModuleList([
            TransformerEncoderLayer(d_model=dim, nhead=heads,
                                    dim_feedforward=dim * scale_dim, dropout=dropout,
                                    attention_dropout=attention_dropout, drop_path_rate=dpr_iter)
            for _ in depth])

        self.enc_pe_1 = PositionalEncoding(dim, dropout, max_position_embeddings_2)
        self.enc_pe_2 = PositionalEncoding(dim, dropout, max_position_embeddings_1)
        self.space_token = nn.Parameter(torch.randn(1, 1, dim))
        # self.space_transformer = self.Transformer(dim, depth, heads, dim_head, dim * scale_dim, dropout)
        self.temporal_token = nn.Parameter(torch.randn(1, 1, dim))
        # self.temporal_transformer = self.Transformer(dim, depth, heads, dim_head, dim * scale_dim, dropout)
        self.to_embedding = nn.Linear(in_channels, dim)

        self.init_weights()

    def init_weights(self):
        if self.space_token is not None:
            nn.init.trunc_normal_(self.space_token, std=.02)
        if self.temporal_token is not None:
            nn.init.trunc_normal_(self.temporal_token, std=.02)
        self.apply(_init_weights)

    def forward(self, x):
        N, M, T, V, C = x.size()
        # x = x.permute(0, 1, 3, 4, 2).contiguous()
        # x = self.data_bn(x.view(N * M, V * C, T))
        # x = x.view(N, M, V, C, T).permute(0, 1, 4, 3, 2).contiguous().view(N * M * V, T, C)

        x = x.permute(0, 1, 3, 2, 4).contiguous().view(N * M * V, T, C)

        # 维度从 N * M * V, T, C 变为 N * M * V, T, dim
        x = self.to_embedding(x)

        # cls 变为 N * M * V，1，dim
        cls_space_tokens = self.space_token.expand(x.size(0), -1, -1)

        # x变为 N * M * V, 1+T, dim
        x = torch.cat((cls_space_tokens, x), dim=1)

        # 输出为 N * M * V, 1+T, dim
        x = self.enc_pe_1(x)

        # 输出为 N * M * V, 1+T, dim
        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)

        # 提取cls，x维度变为 N * M * V，dim
        x = x[:, 0].view(N * M, V, -1)

        # cls 变为 N * M，1，dim
        cls_temporal_tokens = self.temporal_token.expand(x.size(0), -1, -1)

        # 输出为 N * M, 1+V, dim
        x = torch.cat((cls_temporal_tokens, x), dim=1)

        x = self.enc_pe_2(x)

        # 输出为 N * M, 1+V, dim
        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)

        # 输出为 N, M, dim
        x = x[:, 0].view(N, M, -1)

        return x

# x = torch.randn(2,6,1,2,3)
# model = ViViT2()
# output = model.forward(x)
# print(output.shape)
