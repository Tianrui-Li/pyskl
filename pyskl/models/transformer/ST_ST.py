
import torch
from torch import nn, einsum
from einops import rearrange
from ..builder import BACKBONES
import torch.nn.functional as F
# from ...utils import Graph


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

        qkv = self.qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

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

    def forward(self, x):
        """
        :param x: (B, T, C)
        :return: (B, T, C)
        """
        seq_length = x.size(1)
        position_ids = self.position_ids[:, :seq_length]
        x = x + self.position_embeddings(position_ids)
        return self.dropout(x)


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
class ST_ST(nn.Module):
    def __init__(
            self,
            # graph_cfg,
            in_channels=3,
            hidden_dim=64,
            depth=5,  # 10//2=5
            num_heads=8,
            mlp_ratio=4,
            attention_dropout=0.1,
            stochastic_depth_rate=0.1,
            dropout_rate=0.,
            layer_norm_eps=1e-6,
            max_position_embeddings_1=26,
            max_position_embeddings_2=65,
            norm_first=True,
    ):
        super().__init__()

        # # Batch_normalization
        # graph = Graph(**graph_cfg)
        # A = torch.tensor(graph.A, dtype=torch.float32, requires_grad=False)
        # self.data_bn = nn.BatchNorm1d(in_channels * A.size(1))

        self.embd_layer = nn.Linear(in_channels, hidden_dim)
        self.dim = hidden_dim * num_heads

        dpr = [x.item() for x in torch.linspace(0, stochastic_depth_rate, depth)]
        dpr_iter = iter(dpr)

        # Transformer Encoder
        self.spatial_blocks = nn.ModuleList([
            TransformerEncoderLayer(d_model=self.dim, nhead=num_heads,
                                    dim_feedforward=self.dim * mlp_ratio, dropout=dropout_rate,
                                    attention_dropout=attention_dropout, drop_path_rate=next(dpr_iter))
            for _ in range(depth)])

        self.temporal_blocks = nn.ModuleList([
            TransformerEncoderLayer(d_model=self.dim, nhead=num_heads,
                                    dim_feedforward=self.dim * mlp_ratio, dropout=dropout_rate,
                                    attention_dropout=attention_dropout, drop_path_rate=next(dpr_iter))
            for _ in range(depth)])

        self.enc_pe_1 = PositionalEncoding(self.dim, dropout_rate, max_position_embeddings_2)
        self.enc_pe_2 = PositionalEncoding(self.dim, dropout_rate, max_position_embeddings_1)
        self.space_token = nn.Parameter(torch.randn(1, 1, self.dim))
        self.temporal_token = nn.Parameter(torch.randn(1, 1, self.dim))
        self.to_embedding = nn.Linear(in_channels, self.dim)

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

        x = x.permute(0, 1, 3, 2, 4).contiguous().view(N * M * V, T, C)

        # embed the inputs, orig dim -> hidden dim
        x = self.to_embedding(x)

        # cls 变为 N * M * V，1，dim,先temporal
        cls_temporal_tokens = self.temporal_token.expand(x.size(0), -1, -1)

        # x变为 N * M * V, 1+T, dim
        x = torch.cat((cls_temporal_tokens, x), dim=1)

        # 输出为 N * M * V, 1+T, dim
        x = self.enc_pe_1(x)

        # 输出为 N * M * V, 1+T, dim
        for blk in self.temporal_blocks:
            x = blk(x)

        if self.norm is not None:
            x = self.norm(x)

        # 提取cls，x维度变为 N * M * V，dim
        x = x[:, 0].view(N * M, V, -1)

        # cls 变为 N * M，1，dim,后space
        cls_space_tokens = self.space_token.expand(x.size(0), -1, -1)

        # 输出为 N * M, 1+V, dim
        x = torch.cat((cls_space_tokens, x), dim=1)

        x = self.enc_pe_2(x)

        # 输出为 N * M, 1+V, dim
        for blk in self.spatial_blocks:
            x = blk(x)

        if self.norm is not None:
            x = self.norm(x)

        # 输出为 N, M, dim
        x = x[:, 0].view(N, M, -1)

        return x
