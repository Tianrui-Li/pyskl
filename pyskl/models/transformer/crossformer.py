import torch
from torch import nn, einsum
from einops import rearrange
from ...utils import Graph
from ..builder import BACKBONES


def cast_tuple(val, length=1):
    return val if isinstance(val, tuple) else ((val,) * length)


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
    def __init__(
            self,
            dim,
            attn_type,
            window_size,
            window_size2,
            dim_head=32,
            dropout=0.
    ):
        super().__init__()
        assert attn_type in {'short', 'long'}, 'attention type must be one of local or distant'
        heads = dim // dim_head
        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = dim_head * heads

        self.attn_type = attn_type
        self.window_size = window_size
        self.window_size2 = window_size2
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Conv2d(dim, inner_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(inner_dim, dim, 1)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2).contiguous()
        *_, height, width, heads, wsz, wsz2, device = *x.shape, self.heads, self.window_size, self.window_size2, x.device
        if self.attn_type == 'short':
            x = rearrange(x, 'b d (h s1) (w s2) -> (b h w) d s1 s2', s1=wsz, s2=wsz2)
        elif self.attn_type == 'long':
            x = rearrange(x, 'b d (l1 h) (l2 w) -> (b h w) d l1 l2', l1=wsz, l2=wsz2)
        q, k, v = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, 'b (h d) x y -> b h (x y) d', h=heads), (q, k, v))
        q = q * self.scale
        sim = einsum('b h i d, b h j d -> b h i j', q, k)
        attn = sim.softmax(dim=-1)
        attn = self.dropout(attn)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x=wsz, y=wsz2)
        out = self.to_out(out)
        if self.attn_type == 'short':
            out = rearrange(out, '(b h w) d s1 s2 -> b d (h s1) (w s2)', h=height // wsz, w=width // wsz2)
        elif self.attn_type == 'long':
            out = rearrange(out, '(b h w) d l1 l2 -> b d (l1 h) (l2 w)', h=height // wsz, w=width // wsz2)
        out = out.permute(0, 2, 3, 1).contiguous()

        return out


class Transformer(nn.Module):
    def __init__(
            self,
            dim,
            *,
            local_window_size,
            local_window_size2,
            global_window_size,
            global_window_size2,
            depth=4,
            dim_head=32,
            attn_dropout=0.,
            ff_dropout=0.,
            scale_dim=4,
    ):
        super().__init__()
        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, attn_type='short', window_size=local_window_size,
                                       window_size2=local_window_size2,
                                       dim_head=dim_head,
                                       dropout=attn_dropout), ),
                PreNorm(dim, FeedForward(dim, dim*scale_dim, dropout=ff_dropout), ),
                PreNorm(dim, Attention(dim, attn_type='long', window_size=global_window_size, window_size2=global_window_size2,
                          dim_head=dim_head,
                          dropout=attn_dropout), ),
                PreNorm(dim, FeedForward(dim, dim*scale_dim, dropout=ff_dropout)),
            ]))

    def forward(self, x):
        for short_attn, short_ff, long_attn, long_ff in self.layers:
            x = short_attn(x) + x
            x = short_ff(x) + x
            x = long_attn(x) + x
            x = long_ff(x) + x

        return x


def _init_weights(module) -> None:
    if isinstance(module, (nn.Linear, nn.Embedding)):
        nn.init.trunc_normal_(module.weight, std=.02)
    elif isinstance(module, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
        nn.init.zeros_(module.bias)
        nn.init.ones_(module.weight)
    if isinstance(module, nn.Linear) and module.bias is not None:
        nn.init.zeros_(module.bias)


# classes
@BACKBONES.register_module()
class CrossFormer(nn.Module):
    def __init__(
            self,
            graph_cfg,
            *,
            dim=(64, 128, 256, 512),
            depth=(2, 2, 8, 2),
            global_window_size=(8, 4, 2, 1),
            global_window_size2=(5, 1, 1, 1),
            local_window_size=6,
            local_window_size2=5,
            attn_dropout=0.,
            ff_dropout=0.,
            channels=3
    ):
        super().__init__()

        graph = Graph(**graph_cfg)
        A = torch.tensor(graph.A, dtype=torch.float32, requires_grad=False)
        self.data_bn = nn.BatchNorm1d(channels * A.size(1))

        dim = cast_tuple(dim, 4)
        depth = cast_tuple(depth, 4)
        global_window_size = cast_tuple(global_window_size, 4)
        global_window_size2 = cast_tuple(global_window_size2, 4)
        local_window_size = cast_tuple(local_window_size, 4)
        local_window_size2 = cast_tuple(local_window_size2, 4)

        assert len(dim) == 4
        assert len(depth) == 4
        assert len(global_window_size) == 4
        assert len(local_window_size) == 4

        # dimensions

        dims = [channels, *dim]
        dim_in_and_out = tuple(zip(dims[:-1], dims[1:]))

        # layers

        self.layers = nn.ModuleList([])

        for (dim_in, dim_out), layers, global_wsz, global_wsz2, local_wsz, local_wsz2 in zip(dim_in_and_out, depth,
                                                                                             global_window_size,
                                                                                             global_window_size2,
                                                                                             local_window_size,
                                                                                             local_window_size2):
            self.layers.append(nn.ModuleList([
                nn.Linear(dim_in, dim_out),
                Transformer(dim_out, local_window_size=local_wsz, local_window_size2=local_wsz2,
                            global_window_size=global_wsz, global_window_size2=global_wsz2, depth=layers,
                            attn_dropout=attn_dropout, ff_dropout=ff_dropout)
            ]))

        self.pos_embedding = nn.Parameter(torch.randn(1, 1, 25, channels, device=torch.device(
            'cuda'))).repeat(1, 96, 1, 1)
        self.init_weights()  # initialization

    def init_weights(self):
        self.apply(_init_weights)

    def forward(self, x):
        N, M, T, V, C = x.size()
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = self.data_bn(x.view(N * M, V * C, T))
        x = x.view(N, M, V, C, T).permute(0, 1, 4, 3, 2).contiguous().view(N * M, T, V, C)

        x += self.pos_embedding

        for cel, transformer in self.layers:
            x = cel(x)
            x = transformer(x)

        x = torch.flatten(x, start_dim=1, end_dim=2)
        x = x.mean(dim=1)
        x = x.view(N, M, -1)

        return x

# x = torch.randn(4, 2, 96, 25, 3)
# model = CrossFormer()
# output = model.forward(x)
# print(output.shape)
