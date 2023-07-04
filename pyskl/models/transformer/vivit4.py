import torch
from torch import nn, einsum, optim
from einops import rearrange
from ..builder import BACKBONES


def accuracy(y_pred, y_true):
    n_y = y_true.size(0)
    idx_predicted = torch.argmax(y_pred, 1)
    acc = (idx_predicted == y_true).sum().item()
    acc = acc / float(n_y)
    return acc


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(hidden_dim, out_dim), nn.Dropout(dropout))

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0, num_space=None, num_time=None, attn_type=None):
        super().__init__()

        assert attn_type in ['space', 'time'], 'Attention type should be one of the following: space, time.'

        self.attn_type = attn_type
        self.num_space = num_space
        self.num_time = num_time

        inner_dim = dim_head * heads
        self.scale = dim_head ** -0.5
        self.heads = heads

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

    def forward(self, x):

        t = self.num_time
        n = self.num_space

        # reshape to reveal dimensions of space and time
        x = rearrange(x, 'b (t n) d -> b t n d', t=t, n=n)

        if self.attn_type == 'space':
            out = self.forward_space(x)  # (b, tn, d)
        elif self.attn_type == 'time':
            out = self.forward_time(x)  # (b, tn, d)
        else:
            raise Exception('Unknown attention type: %s' % (self.attn_type))

        return out

    def forward_space(self, x):
        """
        x: (b, t, n, d)
        """

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
        """
        x: (b, t, n, d)
        """

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

    def forward_attention(self, x):
        h = self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = self.attend(dots)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')

        return out


def _init_weights(module) -> None:
    if isinstance(module, (nn.Linear, nn.Embedding)):
        nn.init.trunc_normal_(module.weight, std=.02)
    elif isinstance(module, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
        nn.init.zeros_(module.bias)
        nn.init.ones_(module.weight)
    if isinstance(module, nn.Linear) and module.bias is not None:
        nn.init.zeros_(module.bias)


class Transformer(nn.Module):
    def __init__(self, dim, heads, dim_head, mlp_dim, dropout, num_space, num_time):
        super().__init__()

        self.num_space = num_space
        self.num_time = num_time
        heads_half = int(heads / 2.0)

        assert dim % 2 == 0

        self.attention_space = PreNorm(dim, Attention(dim, heads=heads_half, dim_head=dim_head, dropout=dropout,
                                                      num_space=num_space, num_time=num_time,
                                                      attn_type='space'))
        self.attention_time = PreNorm(dim, Attention(dim, heads=heads_half, dim_head=dim_head, dropout=dropout,
                                                     num_space=num_space, num_time=num_time,
                                                     attn_type='time'))

        inner_dim = dim_head * heads_half * 2
        self.linear = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
        self.mlp = PreNorm(dim, FeedForward(dim, mlp_dim, dim, dropout=dropout))

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
class ViViT4(nn.Module):
    def __init__(self,
                 clip_size=32,
                 ):
        super().__init__()
        self.scale_dim = 4
        self.heads = 3
        self.V = 25
        self.clip_size = clip_size
        self.dim = 192
        self.dim_head = 64
        self.dropout_ratio = 0.1
        self.depth = 4
        self.emb_dropout_ratio = 0.1
        self.in_channels = 3
        self.mlp_dim = self.dim * self.scale_dim
        self.num_time = self.clip_size
        self.num_space = self.V
        self.num_patches = self.num_time * self.num_space

        # init layers of the classifier
        self._init_layers()

        # init loss, metric, and optimizer
        self._loss_fn = nn.CrossEntropyLoss()
        self._metric_fn = accuracy
        # self._optimizer = optim.Adam(self.parameters(), 0.001)
        self._optimizer = optim.SGD(self.parameters(), 0.1)

        self.init_weights()  # initialization

    def init_weights(self):
        self.apply(_init_weights)

    def _init_layers(self):

        self.to_embedding = nn.Linear(self.in_channels, self.dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches, self.dim))
        self.dropout = nn.Dropout(self.emb_dropout_ratio)

        self.transformers = nn.ModuleList([])
        for _ in range(self.depth):
            self.transformers.append(
                Transformer(self.dim, self.heads, self.dim_head, self.mlp_dim, self.dropout_ratio, self.num_space,
                            self.num_time))

    def forward(self, x):

        # b, c, t, h, w = x.shape
        #
        # # hide time inside batch
        # x = x.permute(0, 2, 1, 3, 4)  # (b, t, c, h, w)
        # x = rearrange(x, 'b t c h w -> (b t) c h w')  # (b*t, c, h, w)
        #
        # # input embedding to get patch token
        # 变成N*M*T，V，dim
        # x = self.to_patch_embedding(x)  # (b*t, n, d)

        N, M, T, V, C = x.size()
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(N, M, V, C, T).permute(0, 1, 4, 3, 2).contiguous().view(N * M * T, V, C)
        x = self.to_embedding(x)  # (b*t, n, dim)

        # concat patch token and class token
        x = rearrange(x, '(b t) n d -> b (t n) d', b=N * M, t=T)  # (b, tn, d)

        # add position embedding
        x += self.pos_embedding  # (b, tn, d)
        x = self.dropout(x)  # (b, tn, d)

        # layers of transformers
        for transformer in self.transformers:
            x = transformer(x)  # (b, tn, d)

        # space-time pooling
        x = x.mean(dim=1)
        x = x.view(N, M, -1)

        # classification
        # x = x[:, 0]
        # classifier
        # x = self.mlp_head(x)

        return x


# x = torch.randn(16, 2, 32, 25, 3)
# model = ViViT4()
# output = model.forward(x)
# print(output.shape)
# # torch.Size([16, 2, 192])
