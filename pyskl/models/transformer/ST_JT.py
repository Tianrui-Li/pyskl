import math
import torch
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange, pack
from .utils import PositionalEncoding
from ..gcns import unit_tcn
from ..builder import BACKBONES
from rotary_embedding_torch import RotaryEmbedding

# from ...utils import Graph


class DynamicPosBias(nn.Module):
    def __init__(self, dim, num_heads, residual):
        super().__init__()
        self.residual = residual
        self.num_heads = num_heads
        self.pos_dim = dim // 4
        self.pos_proj = nn.Linear(2, self.pos_dim)
        self.pos1 = nn.Sequential(
            nn.LayerNorm(self.pos_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.pos_dim, self.pos_dim),
        )
        self.pos2 = nn.Sequential(
            nn.LayerNorm(self.pos_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.pos_dim, self.pos_dim)
        )
        self.pos3 = nn.Sequential(
            nn.LayerNorm(self.pos_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.pos_dim, self.num_heads)
        )

    def forward(self, biases):
        if self.residual:
            pos = self.pos_proj(biases)
            pos = pos + self.pos1(pos)
            pos = pos + self.pos2(pos)
            pos = self.pos3(pos)
        else:
            pos = self.pos3(self.pos2(self.pos1(self.pos_proj(biases))))
        return pos

    def flops(self, N):
        flops = N * 2 * self.pos_dim
        flops += N * self.pos_dim * self.pos_dim
        flops += N * self.pos_dim * self.pos_dim
        flops += N * self.pos_dim * self.num_heads
        return flops


class Attention(nn.Module):
    """ Multi-head self attention module with dynamic position bias.
    Args:
        dim (int): Number of input channels.
        group_size (tuple[int]): The T and V of the group.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """
    def __init__(self, dim, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.,
                 position_bias=True):
        super().__init__()
        self.dim = dim

        # 根据dim的变化，改变T,V的大小
        if dim == 64:
            self.group_size = (64, 25)
            self.num_heads = 2
        elif dim == 128:
            self.group_size = (32, 25)
            self.num_heads = 4
        elif dim == 256:
            self.group_size = (16, 25)
            self.num_heads = 8
        elif dim == 512:
            self.group_size = (8, 25)
            self.num_heads = 16

        # self.num_heads = num_heads

        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.position_bias = position_bias

        if position_bias:
            self.pos = DynamicPosBias(self.dim // 4, self.num_heads, residual=False)

            # generate mother-set
            position_bias_h = torch.arange(1 - self.group_size[0], self.group_size[0])
            position_bias_w = torch.arange(1 - self.group_size[1], self.group_size[1])
            biases = torch.stack(torch.meshgrid([position_bias_h, position_bias_w]))  # 2, 2Wh-1, 2W2-1
            biases = biases.flatten(1).transpose(0, 1).float()
            self.register_buffer("biases", biases)

            # get pair-wise relative position index for each token inside the group
            coords_h = torch.arange(self.group_size[0])
            coords_w = torch.arange(self.group_size[1])
            coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
            coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
            relative_coords[:, :, 0] += self.group_size[0] - 1  # shift to start from 0
            relative_coords[:, :, 1] += self.group_size[1] - 1
            relative_coords[:, :, 0] *= 2 * self.group_size[1] - 1
            relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
            self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_groups*B, N, C)
            mask: (0/-inf) mask with shape of (num_groups, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        if self.position_bias:
            pos = self.pos(self.biases)  # 2Wh-1 * 2Ww-1, heads
            # select position bias
            relative_position_bias = pos[self.relative_position_index.view(-1)].view(
                self.group_size[0] * self.group_size[1], self.group_size[0] * self.group_size[1], -1)  # Wh*Ww,Wh*Ww,nH
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
            attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x



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


def _trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    # if (mean < a - 2 * std) or (mean > b + 2 * std):
    #     warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
    #                   "The distribution of values may be incorrect.",
    #                   stacklevel=2)

    # Values are generated by using a truncated uniform distribution and
    # then using the inverse CDF for the normal distribution.
    # Get upper and lower cdf values
    l = norm_cdf((a - mean) / std)
    u = norm_cdf((b - mean) / std)

    # Uniformly fill tensor with values from [l, u], then translate to
    # [2l-1, 2u-1].
    tensor.uniform_(2 * l - 1, 2 * u - 1)

    # Use inverse cdf transform for normal distribution to get truncated
    # standard normal
    tensor.erfinv_()

    # Transform to proper mean, std
    tensor.mul_(std * math.sqrt(2.))
    tensor.add_(mean)

    # Clamp to ensure it's in the proper range
    tensor.clamp_(min=a, max=b)
    return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    r"""Fills the input Tensor with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \leq \text{mean} \leq b`.

    NOTE: this impl is similar to the PyTorch trunc_normal_, the bounds [a, b] are
    applied while sampling the normal with mean/std applied, therefore a, b args
    should be adjusted to match the range of mean, std args.

    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value
    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.trunc_normal_(w)
    """
    with torch.no_grad():
        return _trunc_normal_(tensor, mean, std, a, b)


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
                                   attn_drop=attention_dropout, proj_drop=dropout)

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
class ST_JT(nn.Module):
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
            use_cls=False,
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
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if hasattr(m, 'weight'):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if hasattr(m, 'bias') and m.bias is not None and isinstance(m.bias, torch.Tensor):
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                if hasattr(m, 'weight') and m.weight is not None:
                    m.weight.data.normal_(1.0, 0.02)
                if hasattr(m, 'bias') and m.bias is not None:
                    m.bias.data.fill_(0)
            elif isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

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

        if self.norm is not None:
            hidden_state = self.norm(hidden_state)

        if self.cls_token is not None:
            hidden_state = hidden_state[:, :1, :]

        hidden_state = rearrange(
            hidden_state, '(n m) tv c -> n m tv c', n=N)

        return hidden_state
