import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn, einsum

from ..builder import BACKBONES


# from ...utils import Graph


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, attention_dropout=0.1, projection_dropout=0.1):
        super().__init__()
        self.heads = num_heads
        self.head_dim = dim // self.heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.attn_drop = nn.Dropout(attention_dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(projection_dropout)

    def forward(self, x):
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


class PreNorm(nn.Module):
    def __init__(self, dim, fn, drop_path_rate=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
        self.drop_path = DropPath(drop_path_rate)

    def forward(self, x, **kwargs):
        return self.drop_path(self.fn(self.norm(x), **kwargs))


class FeedForward(nn.Module):
    def __init__(self, d_model, dim_feedforward=2048, dropout=0.1, drop_path_rate=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.drop_path = DropPath(drop_path_rate)
        self.activation = F.gelu

    def forward(self, src):
        src = self.norm1(src)
        src2 = self.linear2(self.dropout1(self.activation(self.linear1(src))))
        src = src + self.drop_path(self.dropout2(src2))
        return src


class TransformerEncoder(nn.Module):
    """
    Inspired by torch.nn.TransformerEncoderLayer and
    rwightman's timm package.
    """

    def __init__(self, depth, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 attention_dropout=0.1, stochastic_depth_rate=0.1):
        super().__init__()
        self.layers = nn.ModuleList([])
        dpr = [x.item() for x in torch.linspace(0, stochastic_depth_rate, depth)]
        dpr_iter = iter(dpr)
        for _ in range(depth):
            dpr_now = next(dpr_iter)
            self.layers.append(nn.ModuleList(
                [PreNorm(d_model, Attention(dim=d_model, num_heads=nhead, attention_dropout=attention_dropout,
                                            projection_dropout=dropout), dpr_now),
                 PreNorm(d_model, Attention(dim=d_model, num_heads=nhead, attention_dropout=attention_dropout,
                                            projection_dropout=dropout), dpr_now),
                 FeedForward(d_model, dim_feedforward=dim_feedforward, dropout=dropout, drop_path_rate=dpr_now)
                 ]))

    def forward(self, x):
        b = x.shape[0]  # NM,T,V,C
        x = torch.flatten(x, start_dim=0, end_dim=1)  # NMT,V,C

        for sp_attn, temp_attn, ff in self.layers:
            sp_attn_x = sp_attn(x) + x  # Spatial attention # NMT,V,C

            # Reshape tensors for temporal attention
            sp_attn_x = sp_attn_x.chunk(b, dim=0)  # 切割张量块为b个，T V C
            sp_attn_x = [temp[None] for temp in sp_attn_x]  # 1 T V C
            sp_attn_x = torch.cat(sp_attn_x, dim=0).transpose(1, 2)  # B V T C
            sp_attn_x = torch.flatten(sp_attn_x, start_dim=0, end_dim=1)  # BV T C

            temp_attn_x = temp_attn(sp_attn_x) + sp_attn_x  # Temporal attention

            x = ff(temp_attn_x) + temp_attn_x  # MLP

            # Again reshape tensor for spatial attention
            x = x.chunk(b, dim=0)
            x = [temp[None] for temp in x]
            x = torch.cat(x, dim=0).transpose(1, 2)
            x = torch.flatten(x, start_dim=0, end_dim=1)  # BT V C

        # Reshape vector to [b, t*v, dim]
        x = x.chunk(b, dim=0)
        x = [temp[None] for temp in x]
        x = torch.cat(x, dim=0)
        x = torch.flatten(x, start_dim=1, end_dim=2)

        # 输出N*M，T*V，dim
        return x


@BACKBONES.register_module()
class ST_AttnST(nn.Module):
    def __init__(self,
                 # graph_cfg,
                 in_channels=3,
                 hidden_dim=64,
                 depth=10,
                 num_heads=8,
                 mlp_ratio=4,
                 attention_dropout=0.1,
                 stochastic_depth_rate=0.1,
                 dropout_rate=0.,
                 layer_norm_eps=1e-6,
                 max_position_embeddings_1=25,
                 max_position_embeddings_2=64,
                 norm_first=True,
                 ):
        super().__init__()
        # graph = Graph(**graph_cfg)
        # A = torch.tensor(graph.A, dtype=torch.float32, requires_grad=False)
        # self.data_bn = nn.BatchNorm1d(in_channels * A.size(1))

        self.dim = hidden_dim * num_heads
        self.to_embedding = nn.Linear(in_channels, self.dim)
        self.norm = (nn.LayerNorm(512, eps=layer_norm_eps)
                     if norm_first else None)

        # repeat same spatial position encoding temporally

        self.v = max_position_embeddings_1
        self.t = max_position_embeddings_2
        self.pos_embedding = nn.Parameter(torch.randn(1, 1, self.v, self.dim, device=torch.device('cuda'))).repeat(1,
                                                                                                                   self.t,
                                                                                                                   1, 1)
        self.dropout = nn.Dropout(dropout_rate)
        self.transformer = TransformerEncoder(depth=depth, d_model=self.dim, nhead=num_heads,
                                              dim_feedforward=self.dim * mlp_ratio, dropout=0.1,
                                              attention_dropout=attention_dropout,
                                              stochastic_depth_rate=stochastic_depth_rate)

        self.init_weights()  # initialization

    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        """ x is a video: (b, C, T, H, W) """
        N, M, T, V, C = x.size()
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        # x = self.data_bn(x.view(N * M, V * C, T))
        x = x.view(N, M, V, C, T).permute(0, 1, 4, 3, 2).contiguous().view(N * M, T, V, C)

        tokens = self.to_embedding(x)
        tokens += self.pos_embedding
        tokens = self.dropout(tokens)
        x = self.transformer(tokens)

        if self.norm is not None:
            x = self.norm(x)

        x = x.mean(dim=1)
        x = x.view(N, M, -1)

        return x
