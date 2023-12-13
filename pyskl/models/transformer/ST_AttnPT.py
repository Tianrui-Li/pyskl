import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn, einsum

from ..builder import BACKBONES


# from ...utils import Graph


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


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, attention_dropout=0.1, projection_dropout=0.1, num_space=25, num_time=32,
                 attn_type=None):
        super().__init__()

        self.heads = num_heads
        self.head_dim = dim // self.heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.attn_drop = nn.Dropout(attention_dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(projection_dropout)

        self.attn_type = attn_type
        self.num_space = num_space
        self.num_time = num_time

    def forward_attention(self, x):
        qkv = self.qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        q = q * self.scale
        attn = einsum('b h i d, b h j d -> b h i j', q, k)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = einsum('b h i j, b h j d -> b h i d', attn, v)
        x = rearrange(x, 'b h n d -> b n (h d)')

        return self.proj_drop(self.proj(x))

    def forward_space(self, x):

        t = self.num_time
        n = self.num_space

        # hide time dimension into batch dimension
        x = rearrange(x, 'b t n d -> (b t) n d')  # (bt, n, d)

        # apply self-attention
        out = self.forward_attention(x)  # (bt, n, d)

        # recover time dimension and merge it into space
        out = rearrange(out, '(b t) n d -> b (t n) d', t=t, n=n)  # (b, tn, d)

        return out

    def forward_time(self, x):

        t = self.num_time
        n = self.num_space

        # hide time dimension into batch dimension
        x = x.permute(0, 2, 1, 3)  # (b, n, t, d)
        x = rearrange(x, 'b n t d -> (b n) t d')  # (bn, t, d)

        # apply self-attention
        out = self.forward_attention(x)  # (bn, t, d)

        # recover time dimension and merge it into space
        out = rearrange(out, '(b n) t d -> b (t n) d', t=t, n=n)  # (b, tn, d)

        return out

    def forward(self, x):

        t = self.num_time
        n = self.num_space

        # reshape to reveal dimensions of space and time
        x = rearrange(x, 'b (t n) d -> b t n d', t=t, n=n)

        if self.attn_type == 'space':
            out = self.forward_space(x)  # (b, tn, d)
        else:
            out = self.forward_time(x)  # (b, tn, d)

        return out


class TransformerEncoder(nn.Module):
    def __init__(self, d_model, nhead, num_space, num_time, dim_feedforward=2048, dropout=0.1,
                 attention_dropout=0.1, stochastic_depth_rate=0.1):
        super().__init__()

        self.num_space = num_space
        self.num_time = num_time
        self.dim_heads = d_model // nhead

        heads_half = int(nhead / 2.0)

        self.attention_space = PreNorm(d_model, Attention(dim=d_model, num_heads=heads_half,
                                                          attention_dropout=attention_dropout,
                                                          projection_dropout=dropout,
                                                          num_space=num_space, num_time=num_time,
                                                          attn_type='space'), stochastic_depth_rate
                                       )
        self.attention_time = PreNorm(d_model, Attention(dim=d_model, num_heads=heads_half,
                                                         attention_dropout=attention_dropout,
                                                         projection_dropout=dropout,
                                                         num_space=num_space, num_time=num_time,
                                                         attn_type='time'), stochastic_depth_rate
                                      )

        inner_dim = d_model * 2
        self.linear = nn.Sequential(nn.Linear(inner_dim, d_model), nn.Dropout(dropout))
        self.mlp = FeedForward(d_model, dim_feedforward=dim_feedforward, dropout=dropout,
                               drop_path_rate=stochastic_depth_rate)

    def forward(self, x):
        # self-attention
        xs = self.attention_space(x)
        xt = self.attention_time(x)
        out_att = torch.cat([xs, xt], dim=2)

        # linear after self-attention
        out_att = self.linear(out_att)

        # residual connection for self-attention
        out_att += x

        # mlp after attention
        out_mlp = self.mlp(out_att)

        # residual for mlp
        out_mlp += out_att

        return out_mlp


@BACKBONES.register_module()
class ST_AttnPT(nn.Module):
    def __init__(
            self,
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
            norm_first=True,
            num_space=25,
            num_time=64,
            ):
        super().__init__()
        self.depth = depth
        self.dim = hidden_dim * num_heads
        self.to_embedding = nn.Linear(in_channels, self.dim)
        self.norm = (nn.LayerNorm(self.dim, eps=layer_norm_eps)
                     if norm_first else None)

        self.num_space = num_space
        self.num_time = num_time
        self.num_patches = self.num_time * self.num_space

        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches, self.dim))
        self.dropout = nn.Dropout(dropout_rate)

        dpr = [x.item() for x in torch.linspace(0, stochastic_depth_rate, depth)]
        dpr_iter = iter(dpr)
        self.transformers = nn.ModuleList([])
        for _ in range(self.depth):
            self.transformers.append(
                TransformerEncoder(d_model=self.dim, nhead=num_heads,
                                   num_space=self.num_space, num_time=self.num_time,
                                   dim_feedforward=self.dim * mlp_ratio, dropout=0.1,
                                   attention_dropout=attention_dropout,
                                   stochastic_depth_rate=next(dpr_iter),
                                   ))

        self.init_weights()

    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):

        N, M, T, V, C = x.size()
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        # x = self.data_bn(x.view(N * M, V * C, T))
        x = x.view(N, M, V, C, T).permute(0, 1, 4, 3, 2).contiguous().view(N * M, T * V, C)

        x = self.to_embedding(x)  # (b,t*v,dim)

        # add position embedding
        x += self.pos_embedding  # (b, t*v, d)
        x = self.dropout(x)  # (b, t*v, d)

        # # layers of transformers
        for transformer in self.transformers:
            x = transformer(x)  # (b, t*v, d)

        if self.norm is not None:
            x = self.norm(x)

        x = x.view(N, M, T, V, -1).contiguous()
        x = torch.mean(x, [2, 3])

        return x

