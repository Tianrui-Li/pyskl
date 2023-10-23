import torch
import torch.nn as nn

from torch import einsum

from einops import rearrange
# from ...utils import Graph
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


class FSAttention(nn.Module):
    def __init__(self, dim, heads=3, dim_head=64, dropout=0.):
        super().__init__()

        inner_dim = dim_head * heads

        project_out = not (heads == 1 and dim_head == dim)

        self.attend = nn.Softmax(dim=-1)

        self.heads = heads
        self.scale = dim_head ** -0.5
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(dots)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)

        return out


def _init_weights(module) -> None:
    if isinstance(module, (nn.Linear, nn.Embedding)):
        nn.init.trunc_normal_(module.weight, std=.02)
    elif isinstance(module, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
        nn.init.zeros_(module.bias)
        nn.init.ones_(module.weight)
    if isinstance(module, nn.Linear) and module.bias is not None:
        nn.init.zeros_(module.bias)


class FSATransformerEncoder(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.norm = nn.LayerNorm(dim)
        for _ in range(depth):
            self.layers.append(nn.ModuleList(
                [PreNorm(dim, FSAttention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                 PreNorm(dim, FSAttention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                 PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
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
class ViViT3(nn.Module):
    """ Model-3 backbone of ViViT """

    def __init__(self,
                 # graph_cfg,
                 dim=512,
                 depth=8,
                 heads=8,
                 dim_head=64,
                 in_channels=3,
                 emb_dropout=0.,
                 dropout=0.,
                 scale_dim=4,
                 max_position_embeddings_1=25,
                 max_position_embeddings_2=64,
                 ):
        super().__init__()
        # graph = Graph(**graph_cfg)
        # A = torch.tensor(graph.A, dtype=torch.float32, requires_grad=False)
        # self.data_bn = nn.BatchNorm1d(in_channels * A.size(1))

        self.to_embedding = nn.Linear(in_channels, dim)

        # repeat same spatial position encoding temporally

        self.v = max_position_embeddings_1
        self.t = max_position_embeddings_2
        # self.pos_embedding = nn.Parameter(torch.randn(1, 1, self.v, dim)).repeat(1, self.t, 1, 1)
        # self.pos_embedding = self.pos_embedding.to(torch.device('cuda'))
        self.pos_embedding = nn.Parameter(torch.randn(1, 1, self.v, dim, device=torch.device('cuda'))).repeat(1, self.t,
                                                                                                              1, 1)
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = FSATransformerEncoder(dim, depth, heads, dim_head, dim * scale_dim, dropout)
        self.to_latent = nn.Identity()

        self.init_weights()  # initialization

    def init_weights(self):
        self.apply(_init_weights)

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

        x = x.mean(dim=1)
        # x = self.to_latent(x)
        # x = x.view(N, M*T*V, -1) # 效果好，但需要更大的显存
        x = x.view(N, M, -1)

        return x

# x = torch.randn(16,2,32,25,3)
# model = ViViT3()
# output = model.forward(x)
# print(output.shape)
