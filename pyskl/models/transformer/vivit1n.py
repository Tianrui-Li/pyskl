import torch
import torch.nn as nn

from torch import einsum
from torch import Tensor
from einops import rearrange
from ...utils import Graph
from ..builder import BACKBONES


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()

        inner_dim = dim_head * heads

        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, d, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = dots.softmax(dim=-1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)

        return out  # input dim (b n d), output dim (b n d)


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
    if isinstance(module, (nn.Linear, nn.Embedding)):
        nn.init.trunc_normal_(module.weight, std=.02)
    elif isinstance(module, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
        nn.init.zeros_(module.bias)
        nn.init.ones_(module.weight)
    if isinstance(module, nn.Linear) and module.bias is not None:
        nn.init.zeros_(module.bias)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.norm = nn.LayerNorm(dim)
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)


@BACKBONES.register_module()
class ViViT1n(nn.Module):
    def __init__(
            self,
            graph_cfg,
            in_channels=3,
            dim=192,
            dropout=0.,
            max_position_embeddings=512,
            depth=4,
            heads=3,
            dim_head=64,
            scale_dim=4,
            ):
        super().__init__()
        graph = Graph(**graph_cfg)
        A = torch.tensor(graph.A, dtype=torch.float32, requires_grad=False)
        self.data_bn = nn.BatchNorm1d(in_channels * A.size(1))

        self.enc_pe = PositionalEncoding(dim, dropout, max_position_embeddings)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.st_transformer = Transformer(dim, depth, heads, dim_head, dim * scale_dim, dropout)
        self.to_embedding = nn.Linear(in_channels, dim)

        self.init_weights()  # initialization

    def init_weights(self):
        nn.init.trunc_normal_(self.cls_token, std=.02)
        self.apply(_init_weights)

    def forward(self, x):
        N, M, T, V, C = x.size()
        # print(x.size())
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = self.data_bn(x.view(N * M, V * C, T))
        x = x.view(N, M, V, C, T).permute(0, 1, 4, 3, 2).contiguous().view(N * M, T * V, C)

        x = self.to_embedding(x)  # output(N*M,T*V,dim)

        # cls-token dim (N*M,1,dim)
        cls_st_tokens = self.cls_token.expand(x.size(0), -1, -1)

        # output(N*M,1+T*V,dim)
        x = torch.cat((cls_st_tokens, x), dim=1)

        # output (N*M,1+T*V,dim)
        x_input = self.enc_pe(x)
        x = self.st_transformer(x_input)

        # N*M,1,dim ->N,M,dim
        # 只保留第二维的0索引向量2,3,4->2,4
        x = x[:, 0].view(N, M, -1)

        return x

# x = torch.randn(24,2,96,25,3)
# model = ViViT1n()
# output = model.forward(x)
# print(output.shape)
