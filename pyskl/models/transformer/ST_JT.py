
import torch
from torch import nn
from einops import rearrange, pack
from ..builder import BACKBONES
from ..gcns import unit_tcn
import torch.nn.functional as F
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
            pos = self.pos_proj(biases)  # 2Wh-1 * 2Ww-1, heads
            pos = pos + self.pos1(pos)
            pos = pos + self.pos2(pos)
            pos = self.pos3(pos)
        else:
            pos = self.pos3(self.pos2(self.pos1(self.pos_proj(biases))))
        return pos

    # def flops(self, N):
    #     flops = N * 2 * self.pos_dim
    #     flops += N * self.pos_dim * self.pos_dim
    #     flops += N * self.pos_dim * self.pos_dim
    #     flops += N * self.pos_dim * self.num_heads
    #     return flops


class Attention(nn.Module):
    """ Multi-head self attention module with dynamic position bias.
    Args:
        dim (int): Number of input channels.
        group_size (tuple[int]): The height and width of the group.
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

        # dim的变化
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
                 with_cls=True, dilations=None):
        super().__init__()
        if dilations is None:
            dilations = [1, 2]
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
            attention_dropout=0.1,
            stochastic_depth_rate=0.1,
            dropout_rate=0.,
            use_cls=False,
            layer_norm_eps=1e-6,
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

        self.norm = (nn.LayerNorm(512, eps=layer_norm_eps)
                     if norm_first else None)

        self.init_weights()

    # def init_weights(self):
    #     for p in self.parameters():
    #         if p.dim() > 1:
    #             nn.init.xavier_uniform_(p)

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
                nn.init.trunc_normal_(m.weight, std=.02)
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

            # encoder
            hidden_state = encoder(hidden_state, attn_mask)

        if self.norm is not None:
            hidden_state = self.norm(hidden_state)

        if self.cls_token is not None:
            hidden_state = hidden_state[:, :1, :]

        hidden_state = rearrange(
            hidden_state, '(n m) tv c -> n m tv c', n=N)

        return hidden_state
