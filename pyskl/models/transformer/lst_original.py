"""Locality-aware Spatiotemporal Transformer
This implementation is based on Pytorch transformer version.
"""

import torch
from torch import nn, einsum
from einops import rearrange, pack
import math
# from ...utils import Graph

from ..builder import BACKBONES
from .utils import PositionalEncoding
from ..gcns import unit_tcn
import torch.nn.functional as F
from rotary_embedding_torch import RotaryEmbedding

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

        self.rotary_emb = RotaryEmbedding(dim=head_dim)

    def forward(self, x):
        B, N, C = x.shape

        qkv = self.qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        q = q * self.scale

        # 应用旋转位置编码
        q = self.rotary_emb.rotate_queries_or_keys(q)
        k = self.rotary_emb.rotate_queries_or_keys(k)

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

        keep_mask = torch.zeros(shape, device = device).float().uniform_(0, 1) < keep_prob
        output = x.div(keep_prob) * keep_mask.float()
        return output



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

        self.linear1  = nn.Linear(d_model, dim_feedforward)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1    = nn.LayerNorm(d_model)
        self.linear2  = nn.Linear(dim_feedforward, d_model)
        self.dropout2 = nn.Dropout(dropout)

        self.drop_path = DropPath(drop_path_rate)

        self.activation = F.gelu

    def forward(self, src, *args, **kwargs):
        src = src + self.drop_path(self.self_attn(self.pre_norm(src)))
        src = self.norm1(src)
        src2 = self.linear2(self.dropout1(self.activation(self.linear1(src))))
        src = src + self.drop_path(self.dropout2(src2))
        return src





def sliding_window_attention_mask(
        seq_len, window_size, neighborhood_size, dtype, with_cls=True):
    """
    Generate sliding window attention mask with a quadratic window shape.

    Args:
    - seq_len: The total sequence length
    - window_size: Size of the attention window
    - neighborhood_size: Number of neighboring windows around the central window

    Returns:
    - mask: A seq_len x seq_len mask where 0 indicates positions that can be attended to, and -1e9 (or a very large negative value) indicates positions that should not be attended to.
    """

    def fn(seq_len, window_size, neighborhood_size):
        mask = torch.full((seq_len, seq_len), torch.finfo(dtype).min)

        num_chunks = math.ceil(seq_len / window_size)
        q_chunks = torch.arange(seq_len).chunk(num_chunks)
        for i, q_chunk in enumerate(q_chunks):
            q_start, q_end = q_chunk[0], q_chunk[-1] + 1
            k_start = max(0, (i - neighborhood_size) * window_size)
            k_end = min(seq_len, (i + neighborhood_size + 1) * window_size)
            mask[q_start: q_end, k_start: k_end] = 0

        return mask

    if not with_cls:
        return fn(seq_len, window_size, neighborhood_size)

    orig_len = seq_len
    seq_len = seq_len - 1
    # mask without considering cls token
    mask = fn(seq_len, window_size, neighborhood_size)
    mask = torch.cat([torch.full((seq_len, 1), torch.finfo(dtype).min), mask], dim=1)
    mask = torch.cat([torch.zeros(1, orig_len, dtype=dtype), mask], dim=0)
    return mask


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        if hasattr(m, 'weight'):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
        if hasattr(m, 'bias') and m.bias is not None and isinstance(m.bias, torch.Tensor):
            nn.init.constant_(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        if hasattr(m, 'weight') and m.weight is not None:
            m.weight.data.normal_(1.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            m.bias.data.fill_(0)


class TemporalConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1):
        super(TemporalConv, self).__init__()
        pad = (kernel_size + (kernel_size - 1) * (dilation - 1) - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, 1),
            padding=(pad, 0),
            stride=(stride, 1),
            dilation=(dilation, 1),
        )

        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class MultiScale_TemporalConv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 dilations=[1, 2, 3, 4],
                 residual=False,
                 residual_kernel_size=1):

        super().__init__()
        assert out_channels % (len(dilations) + 2) == 0, '# out channels should be multiples of # branches'

        # Multiple branches of temporal convolution
        self.num_branches = len(dilations) + 2
        branch_channels = out_channels // self.num_branches
        if type(kernel_size) == list:
            assert len(kernel_size) == len(dilations)
        else:
            kernel_size = [kernel_size] * len(dilations)
        # Temporal Convolution branches
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    branch_channels,
                    kernel_size=1,
                    padding=0
                ),
                nn.BatchNorm2d(branch_channels),
                nn.ReLU(inplace=True),
                TemporalConv(
                    branch_channels,
                    branch_channels,
                    kernel_size=ks,
                    stride=stride,
                    dilation=dilation
                ),
            )
            for ks, dilation in zip(kernel_size, dilations)
        ])

        # Additional Max & 1x1 branch
        self.branches.append(nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3, 1), stride=(stride, 1), padding=(1, 0)),
            nn.BatchNorm2d(branch_channels)  # 为什么还要加bn
        ))

        self.branches.append(nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, padding=0, stride=(stride, 1)),
            nn.BatchNorm2d(branch_channels)
        ))

        # Residual connection
        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = TemporalConv(in_channels, out_channels, kernel_size=residual_kernel_size, stride=stride)
        # print(len(self.branches))
        # initialize
        self.apply(weights_init)

    def forward(self, x):
        # Input dim: (N,C,T,V)
        res = self.residual(x)
        branch_outs = []
        for tempconv in self.branches:
            out = tempconv(x)
            branch_outs.append(out)

        out = torch.cat(branch_outs, dim=1)
        out += res
        return out


class TemporalPooling(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size, stride, pooling=False,
                 with_cls=True, dilations=[1, 2]):
        super().__init__()
        self.pooling = pooling
        self.with_cls = with_cls
        if dim_in == dim_out:
            self.temporal_pool = nn.Identity()
        elif pooling:
            if with_cls:
                self.cls_mapping = nn.Linear(dim_in, dim_out)
            self.temporal_pool = unit_tcn(dim_in, dim_out, kernel_size, stride)
            # self.temporal_pool = MultiScale_TemporalConv(dim_in, dim_out, kernel_size=kernel_size, stride=stride,
            #                                              dilations=dilations,
            #                                              #  residual=True has worse performance in the end
            #                                              residual=False)
        else:
            self.temporal_pool = nn.Linear(dim_in, dim_out)

    def forward(self, x, v=25):
        # x in B, T, C
        if isinstance(self.temporal_pool, nn.Identity):
            return self.temporal_pool(x)

        if self.pooling:
            # Split cls token and map cls token dimension if any
            if self.with_cls:
                cls, x = x[:, :1, :], x[:, 1:, :]
                cls = self.cls_mapping(cls)

            # TCN
            x = rearrange(x, 'b (t v) c -> b c t v', v=v)
            x = self.temporal_pool(x)
            res = rearrange(x, 'b c t v -> b (t v) c')

            # Concat cls token if any
            if self.with_cls:
                res, pack_shape = pack([cls, res], 'b * c')
        else:
            res = self.temporal_pool(x)
        return res


@BACKBONES.register_module()
class LST_original(nn.Module):
    """Locality-aware Spatial-Temporal Transformer
    """

    def __init__(
            self,
            # graph_cfg,
            in_channels=3,
            hidden_dim=64,
            dim_mul_layers=(4, 7),
            dim_mul_factor=2,
            depth=10,
            num_heads=4,
            mlp_ratio=4,
            norm_first=False,
            activation='relu',
            attention_dropout=0.1,
            stochastic_depth_rate=0.1,
            dropout=0.1,
            dropout_rate=0.,
            use_cls=True,
            layer_norm_eps=1e-6,
            max_joints=25,
            max_frames=100,
            temporal_pooling=True,
            sliding_window=False,
            stride1=2,
            kernel_size1=3,
    ):
        super().__init__()

        # # Batch_normalization
        # graph = Graph(**graph_cfg)
        # A = torch.tensor(graph.A, dtype=torch.float32, requires_grad=False)
        # self.data_bn = nn.BatchNorm1d(in_channels * A.size(1))

        self.embd_layer = nn.Linear(in_channels, hidden_dim)
        self.norm_first = norm_first
        self.sliding_window = sliding_window
        self.use_cls = use_cls

        # cls token
        self.cls_token = (nn.Parameter(torch.zeros(1, 1, hidden_dim))
                          if use_cls else None)
        self.pos_embed_cls = (nn.Parameter(torch.zeros(1, 1, hidden_dim))
                              if use_cls else None)

        # # We use two embeddings, one for joints and one for frames
        # self.joint_pe = PositionalEncoding(hidden_dim, max_joints)
        # self.frame_pe = PositionalEncoding(hidden_dim, max_frames)

        # Variable hidden dim
        hidden_dims = []
        dim_in = hidden_dim
        for i in range(depth):
            if dim_mul_layers is not None and i in dim_mul_layers:
                dim_out = dim_in * dim_mul_factor
            else:
                dim_out = dim_in
            hidden_dims.append((dim_in, dim_out))
            dim_in = dim_out

        dpr = [x.item() for x in torch.linspace(0, stochastic_depth_rate, depth)]
        dpr_iter = iter(dpr)

        # Transformer Encoder
        self.layers = nn.ModuleList([])
        mlp_ratio = int(mlp_ratio)

        for dim_in, dim_out in hidden_dims:
            self.layers.append(nn.ModuleList([
                # Temporal pool
                TemporalPooling(
                    dim_in, dim_out, kernel_size1, stride1,
                    pooling=temporal_pooling, with_cls=use_cls),
                # # Transformer encoder
                # nn.TransformerEncoderLayer(
                #     dim_out, num_heads, dim_out * mlp_ratio, dropout,
                #     activation, layer_norm_eps, batch_first=True,
                #     norm_first=norm_first)
                TransformerEncoderLayer(d_model=dim_out, nhead=num_heads,
                                        dim_feedforward=dim_out * mlp_ratio, dropout=dropout_rate,
                                        attention_dropout=attention_dropout, drop_path_rate=next(dpr_iter))

            ]))

        # Variable locality-aware mask
        if sliding_window:
            nb_size = 0
            self.neighborhood_sizes = [nb_size]
            for dim_in, dim_out in hidden_dims[1:]:
                nb_size = nb_size + (dim_in == dim_out)
                self.neighborhood_sizes.append(nb_size)

        self.norm = (nn.LayerNorm(hidden_dims[-1][-1], eps=layer_norm_eps)
                     if norm_first else None)

        self.init_weights()

    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        N, M, T, V, C = x.size()

        # x = x.permute(0, 1, 3, 4, 2).contiguous()
        # x = self.data_bn(x.view(N * M, V * C, T))
        # x = x.view(N, M, V, C, T).permute(0, 1, 4, 3, 2).contiguous().view(N * M, T, V, C)

        x = rearrange(x, 'n m t v c -> (n m) t v c')

        # embed the inputs, orig dim -> hidden dim
        x_embd = self.embd_layer(x)

        # # add positional embeddings
        # x_input = self.joint_pe(x_embd)  # joint-wise
        # x_input = self.frame_pe(rearrange(x_input, 'b t v c -> b v t c'))  # frame wise
        # # convert to required dim order
        # x_input = rearrange(x_input, 'b v t c -> b (t v) c')

        # 旋转位置编码
        x_input = rearrange(x_embd, 'b t v c -> b (t v) c')

        # prepend the cls token for source if needed
        if self.cls_token is not None:
            cls_token = self.cls_token.expand(x_input.size(0), -1, -1)
            cls_token = cls_token + self.pos_embed_cls
            x_input = torch.cat((cls_token, x_input), dim=1)

        hidden_state = x_input
        attn_mask = None
        for i, (temporal_pool, encoder) in enumerate(self.layers):
            # Temporal pooling
            hidden_state = temporal_pool(hidden_state, v=V)

            # Construct attention mask if required
            if self.sliding_window:
                attn_mask = sliding_window_attention_mask(
                    seq_len=hidden_state.size(1),
                    window_size=V,
                    neighborhood_size=self.neighborhood_sizes[i],
                    dtype=hidden_state.dtype,
                    with_cls=self.use_cls,
                ).to(hidden_state.device)
            hidden_state = encoder(hidden_state, attn_mask)

        if self.norm is not None:
            hidden_state = self.norm(hidden_state)

        if self.cls_token is not None:
            hidden_state = hidden_state[:, :1, :]

        hidden_state = rearrange(
            hidden_state, '(n m) tv c -> n m tv c', n=N)

        return hidden_state