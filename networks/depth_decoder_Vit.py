import time
import math
import copy
from functools import partial
from typing import Optional, Callable

import timm
import numpy as np
from layers import ConvBlock, Conv3x3, upsample
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange, repeat
from timm.models.layers import DropPath, trunc_normal_

DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"
# 创建 Sobel 卷积核
sobel_kernel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]], dtype=torch.float32).unsqueeze(
    0).unsqueeze(0)
sobel_kernel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]], dtype=torch.float32).unsqueeze(
    0).unsqueeze(0)


# Sobel 算子函数
def sobel_filter(tensor):
    # 将卷积核扩展到多通道
    sobel_kernel_x_3ch = sobel_kernel_x.expand(3, 1, 3, 3)
    sobel_kernel_y_3ch = sobel_kernel_y.expand(3, 1, 3, 3)

    # 水平梯度
    Gx = F.conv2d(tensor, sobel_kernel_x_3ch, padding=1, groups=3)
    # 垂直梯度
    Gy = F.conv2d(tensor, sobel_kernel_y_3ch, padding=1, groups=3)

    # 梯度幅值
    G = torch.sqrt(Gx ** 2 + Gy ** 2)

    return G


class PatchExpand(nn.Module):
    def __init__(self, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.dim_scale = dim_scale
        # Assuming dim_scale is 2, which means increasing spatial dimensions by a factor of 2
        if dim_scale == 2:
            self.pixel_shuffle = nn.PixelShuffle(upscale_factor=2)
            # Adjusting the number of channels after pixel shuffle
            self.adjust_channels = nn.Conv2d(dim // 4, dim // dim_scale, kernel_size=1, stride=1, padding=0, bias=False)
        else:
            # If no scaling is needed, use an identity mapping
            self.pixel_shuffle = nn.Identity()
            self.adjust_channels = nn.Identity()
        self.norm = norm_layer(dim // dim_scale)

    def forward(self, x):
        # Pixel shuffle expects the input in the format (B, C, H, W)
        x = rearrange(x, 'b h w c -> b c h w')
        if self.dim_scale == 2:
            x = self.pixel_shuffle(x)
            x = self.adjust_channels(x)
        # Convert back to the original format for normalization
        x = rearrange(x, 'b c h w -> b h w c')
        x = self.norm(x)
        return x


class ViTBlock(nn.Module):
    def __init__(
            self,
            d_model,
            dropout=0.,
            attn_drop=0.,
            d_state=16,  # Not used, kept for compatibility
            **kwargs
    ):
        super().__init__()
        self.d_model = d_model
        self.d_inner = d_model  # Using same dimension for ViT

        # Position embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, self.d_inner, 1, 1))

        # Transformer layer
        self.norm1 = nn.LayerNorm(self.d_inner)
        self.attention = nn.MultiheadAttention(
            embed_dim=self.d_inner,
            num_heads=max(1, self.d_inner // 64),  # Ensure at least 1 head
            dropout=attn_drop,
            batch_first=True
        )
        self.norm2 = nn.LayerNorm(self.d_inner)
        self.mlp = nn.Sequential(
            nn.Linear(self.d_inner, self.d_inner * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.d_inner * 4, self.d_inner),
            nn.Dropout(dropout)
        )
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    def forward(self, x: torch.Tensor, **kwargs):
        B, H, W, C = x.shape
        x = x.permute(0, 3, 1, 2)  # (B, C, H, W)

        # Position embedding (interpolate to current size)
        pos_embed = F.interpolate(self.pos_embed, size=(H, W), mode='bilinear', align_corners=False)
        pos_embed = pos_embed.view(1, C, H * W).permute(0, 2, 1)  # (1, H*W, C)

        # Prepare input sequence
        x_flat = x.view(B, C, H * W).permute(0, 2, 1)  # (B, H*W, C)
        x_flat = x_flat + pos_embed

        # Transformer process
        residual = x_flat
        x_flat = self.norm1(x_flat)
        attn_output, _ = self.attention(x_flat, x_flat, x_flat)
        x_flat = residual + attn_output

        residual = x_flat
        x_flat = self.norm2(x_flat)
        mlp_output = self.mlp(x_flat)
        x_flat = residual + mlp_output

        # Reshape back to image
        x_out = x_flat.permute(0, 2, 1).view(B, C, H, W)
        x_out = x_out.permute(0, 2, 3, 1)  # (B, H, W, C)

        return x_out


class VSSBlock(nn.Module):
    def __init__(
            self,
            hidden_dim: int = 0,
            drop_path: float = 0,
            norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
            attn_drop_rate: float = 0,
            d_state: int = 16,
            layer: int = 1,
            **kwargs,
    ):
        super().__init__()
        factor = 2.0
        d_model = int(hidden_dim // factor)  # 这里使用内置int函数
        self.down = nn.Linear(hidden_dim, d_model)
        self.up = nn.Linear(d_model, hidden_dim)
        self.ln_1 = norm_layer(d_model)
        self.self_attention = ViTBlock(d_model=d_model, dropout=attn_drop_rate, d_state=d_state, **kwargs)
        self.drop_path = DropPath(drop_path)
        self.layer = layer

    def forward(self, input: torch.Tensor):
        input_x = self.down(input)
        input_x = input_x + self.drop_path(self.self_attention(self.ln_1(input_x)))
        x = self.up(input_x) + input
        return x


class VSSLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.
    Args:
        dim (int): Number of input channels.
        depth (int): Number of blocks.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(
            self,
            dim,
            depth,
            attn_drop=0.,
            drop_path=0.,
            norm_layer=nn.LayerNorm,
            downsample=None,
            use_checkpoint=False,
            d_state=16,
            **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.use_checkpoint = use_checkpoint

        self.blocks = nn.ModuleList([
            VSSBlock(
                hidden_dim=dim,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                attn_drop_rate=attn_drop,
                d_state=d_state,
            )
            for i in range(depth)])

        if True:
            def _init_weights(module: nn.Module):
                for name, p in module.named_parameters():
                    if name in ["out_proj.weight"]:
                        p = p.clone().detach_()
                        nn.init.kaiming_uniform_(p, a=math.sqrt(5))

            self.apply(_init_weights)

        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)

        if self.downsample is not None:
            x = self.downsample(x)

        return x


class VSSLayer_up(nn.Module):
    """ A basic Swin Transformer layer for one stage.
    Args:
        dim (int): Number of input channels.
        depth (int): Number of blocks.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        Upsample (nn.Module | None, optional): Upsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(
            self,
            dim,
            depth,
            attn_drop=0.,
            drop_path=0.,
            norm_layer=nn.LayerNorm,
            upsample=None,
            use_checkpoint=False,
            d_state=16,
            layer=1,
            **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.use_checkpoint = use_checkpoint

        self.blocks = nn.ModuleList([
            VSSBlock(
                hidden_dim=dim,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                attn_drop_rate=attn_drop,
                d_state=d_state,
                layer=layer,
            )
            for i in range(depth)])

        if True:
            def _init_weights(module: nn.Module):
                for name, p in module.named_parameters():
                    if name in ["out_proj.weight"]:
                        p = p.clone().detach_()
                        nn.init.kaiming_uniform_(p, a=math.sqrt(5))

            self.apply(_init_weights)

        if upsample is not None:
            self.upsample = PatchExpand(dim, dim_scale=2, norm_layer=nn.LayerNorm)
        else:
            self.upsample = None

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)

        if self.upsample is not None:
            x = self.upsample(x)

        return x


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d,
                 bias=False):
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            norm_layer(out_channels),
            nn.ReLU6()
        )


class Conv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, bias=False):
        super(Conv, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2)
        )


class BasicConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, bias=True, norm=False, relu=True, transpose=False):
        super(BasicConv, self).__init__()
        if bias and norm:
            bias = False

        padding = kernel_size // 2
        layers = list()
        if transpose:
            padding = kernel_size // 2 - 1
            layers.append(
                nn.ConvTranspose2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        else:
            layers.append(
                nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        if norm:
            layers.append(nn.BatchNorm2d(out_channel))
        if relu:
            layers.append(nn.ReLU(inplace=True))
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


class AFF(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(AFF, self).__init__()
        self.conv = nn.Sequential(
            BasicConv(in_channel, out_channel, kernel_size=1, stride=1, relu=True),
            BasicConv(out_channel, out_channel, kernel_size=3, stride=1, relu=False)
        )

    def forward(self, x1, x2, x4):
        x = torch.cat([x1, x2, x4], dim=1)

        return self.conv(x)


class ChannelAttentionModule(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super(ChannelAttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttentionModule(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttentionModule, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class FusionConv(nn.Module):
    def __init__(self, in_channels, out_channels, factor=4.0):
        super(FusionConv, self).__init__()
        dim = int(out_channels // factor)
        self.down = nn.Conv2d(in_channels, dim, kernel_size=1, stride=1)
        self.conv_3x3 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1)
        self.conv_5x5 = nn.Conv2d(dim, dim, kernel_size=5, stride=1, padding=2)
        self.conv_7x7 = nn.Conv2d(dim, dim, kernel_size=7, stride=1, padding=3)
        self.spatial_attention = SpatialAttentionModule()
        self.channel_attention = ChannelAttentionModule(dim)
        self.up = nn.Conv2d(dim, out_channels, kernel_size=1, stride=1)
        self.down_2 = nn.Conv2d(in_channels, dim, kernel_size=1, stride=1)

    def forward(self, x1, x2, x4):
        x_fused = torch.cat([x1, x2, x4], dim=1)
        x_fused = self.down(x_fused)
        x_fused_c = x_fused * self.channel_attention(x_fused)
        x_3x3 = self.conv_3x3(x_fused)
        x_5x5 = self.conv_5x5(x_fused)
        x_7x7 = self.conv_7x7(x_fused)
        x_fused_s = x_3x3 + x_5x5 + x_7x7
        x_fused_s = x_fused_s * self.spatial_attention(x_fused_s)

        x_out = self.up(x_fused_s + x_fused_c)

        return x_out


class MSAA(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MSAA, self).__init__()
        self.fusion_conv = FusionConv(in_channels, out_channels)

    def forward(self, x1, x2, x4, last=False):
        x_fused = self.fusion_conv(x1, x2, x4)
        return x_fused


class ViTDepthDecoder(nn.Module):
    def __init__(self, depths=[2, 2, 2, 2, 2], dims=[64, 64, 128, 256, 512],
                 d_state=16, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm,
                 use_checkpoint=False, ):
        super().__init__()
        self.num_layers = len(depths)
        base_dims = 64
        self.embed_dim = dims[0]
        self.num_features = dims[-1]
        # 修复错误：避免使用 int() 函数
        self.num_features_up = dims[0] * 2  # 直接使用乘法
        self.dims = dims
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        self.layers_up = nn.ModuleList()
        self.concat_back_dim = nn.ModuleList()

        for i_layer in range(self.num_layers):
            concat_linear = nn.Linear(self.dims[5 - i_layer], self.dims[4 - i_layer]
                                      ) if i_layer > 0 else nn.Identity()
            if i_layer == 0:
                layer_up = nn.Sequential(
                    VSSLayer(
                        dim=self.dims[4],
                        depth=2,
                        d_state=math.ceil(dims[0] / 6) if d_state is None else d_state,
                        drop=drop_rate,
                        attn_drop=attn_drop_rate,
                        drop_path=dpr[sum(depths[:-1]):sum(depths[:])],
                        norm_layer=norm_layer,
                        downsample=None,
                        use_checkpoint=use_checkpoint),
                    PatchExpand(dim=self.dims[4], dim_scale=2,
                                norm_layer=norm_layer)
                )
            elif i_layer < (self.num_layers - 1):
                layer_up = VSSLayer_up(
                    dim=self.dims[4 - i_layer],
                    depth=depths[(self.num_layers - 1 - i_layer)],
                    d_state=math.ceil(dims[0] / 6) if d_state is None else d_state,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[sum(depths[:(self.num_layers - 1 - i_layer)]):sum(
                        depths[:(self.num_layers - 1 - i_layer) + 1])],
                    norm_layer=norm_layer,
                    upsample=PatchExpand if (i_layer < self.num_layers - 1) else None,
                    use_checkpoint=use_checkpoint,
                    layer=i_layer,
                )
            else:
                # 修复错误：避免使用 int() 函数
                combined_dim = self.dims[4 - i_layer] + self.dims[5 - i_layer] // 2
                layer_up = VSSLayer_up(
                    dim=combined_dim,
                    depth=depths[(self.num_layers - 1 - i_layer)],
                    d_state=math.ceil(dims[0] / 6) if d_state is None else d_state,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[sum(depths[:(self.num_layers - 1 - i_layer)]):sum(
                        depths[:(self.num_layers - 1 - i_layer) + 1])],
                    norm_layer=norm_layer,
                    upsample=PatchExpand,
                    use_checkpoint=use_checkpoint,
                    layer=i_layer,
                )
            self.layers_up.append(layer_up)
            self.concat_back_dim.append(concat_linear)
        del self.concat_back_dim[-1]
        self.norm = norm_layer(self.num_features)
        self.norm_up = norm_layer(self.embed_dim)

        self.sigmoid = nn.Sigmoid()
        self.dispconv0 = Conv3x3(0.5 * (self.dims[0] + 0.5 * self.dims[1]), 1)
        self.dispconv1 = Conv3x3(0.5 * self.dims[0], 1)
        self.dispconv2 = Conv3x3(self.dims[1], 1)
        self.dispconv3 = Conv3x3(self.dims[2], 1)

        hidden_dim = int(base_dims // 4)
        self.AFFs = nn.ModuleList([
            MSAA(3 + hidden_dim * 2, base_dims),
            MSAA(hidden_dim * 4, base_dims),
            MSAA(hidden_dim * 7, base_dims * 2),
            MSAA(hidden_dim * 14, base_dims * 4),
        ])

        self.transfer = nn.ModuleList(
            [
                nn.Conv2d(base_dims, hidden_dim, 1, bias=False),
                nn.Conv2d(base_dims, hidden_dim, 1, bias=False),
                nn.Conv2d(base_dims * 2, hidden_dim * 2, 1, bias=False),
                nn.Conv2d(base_dims * 4, hidden_dim * 4, 1, bias=False),
                nn.Conv2d(base_dims * 8, hidden_dim * 8, 1, bias=False),
            ]
        )

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    # Dencoder and Skip connection
    def forward_up_features(self, x_downsample):
        outputs = {}
        x = x_downsample[-1]
        for inx, layer_up in enumerate(self.layers_up):
            if inx == 0:
                x = layer_up(x)
            else:
                x = torch.cat([x, x_downsample[4 - inx]], -1)
                if inx < 4:
                    x = self.concat_back_dim[inx](x)
                x = layer_up(x)

            if inx == 1:
                tmp = torch.permute(x, (0, 3, 1, 2))
                outputs[("disp", 3)] = self.sigmoid(self.dispconv3(tmp))
            if inx == 2:
                tmp = torch.permute(x, (0, 3, 1, 2))
                outputs[("disp", 2)] = self.sigmoid(self.dispconv2(tmp))
            if inx == 3:
                tmp = torch.permute(x, (0, 3, 1, 2))
                outputs[("disp", 1)] = self.sigmoid(self.dispconv1(tmp))
            if inx == 4:
                tmp = torch.permute(x, (0, 3, 1, 2))
                outputs[("disp", 0)] = self.sigmoid(self.dispconv0(tmp))

        return outputs

    def forward_downfeatures(self, x_downsample, edges):
        x_down_last = x_downsample[-1]
        x_downsample_2 = x_downsample
        x_downsample = []
        for idx, feat in enumerate(x_downsample_2):
            feat = self.transfer[idx](feat)
            x_downsample.append(feat)

        x_down__1_0 = F.interpolate(edges, scale_factor=0.5, mode="bilinear", align_corners=True)
        x_down_1_0 = F.interpolate(x_downsample[1], scale_factor=2.0, mode="bilinear", align_corners=True)

        x_down_0_1 = F.interpolate(x_downsample[0], scale_factor=0.5, mode="bilinear", align_corners=True)
        x_down_2_1 = F.interpolate(x_downsample[2], scale_factor=2.0, mode="bilinear", align_corners=True)

        x_down_1_2 = F.interpolate(x_downsample[1], scale_factor=0.5, mode="bilinear", align_corners=True)
        x_down_3_2 = F.interpolate(x_downsample[3], scale_factor=2.0, mode="bilinear", align_corners=True)

        x_down_2_3 = F.interpolate(x_downsample[2], scale_factor=0.5, mode="bilinear", align_corners=True)
        x_down_4_3 = F.interpolate(x_downsample[4], scale_factor=2.0, mode="bilinear", align_corners=True)

        x_down_0 = self.AFFs[0](x_downsample[0], x_down__1_0, x_down_1_0)
        x_down_1 = self.AFFs[1](x_downsample[1], x_down_0_1, x_down_2_1)
        x_down_2 = self.AFFs[2](x_downsample[2], x_down_1_2, x_down_3_2)
        x_down_3 = self.AFFs[3](x_downsample[3], x_down_2_3, x_down_4_3)

        x_down_0 = torch.permute(x_down_0, (0, 2, 3, 1))
        x_down_1 = torch.permute(x_down_1, (0, 2, 3, 1))
        x_down_2 = torch.permute(x_down_2, (0, 2, 3, 1))
        x_down_3 = torch.permute(x_down_3, (0, 2, 3, 1))
        x_down_last = torch.permute(x_down_last, (0, 2, 3, 1))
        return [x_down_0, x_down_1, x_down_2, x_down_3, x_down_last]

    def forward(self, input_features, edges):
        x_downsample = self.forward_downfeatures(input_features, edges)
        outputs = self.forward_up_features(x_downsample)
        return outputs


if __name__ == "__main__":
    # 测试代码
    from resnet_encoder import ResnetEncoder, ResnetEncoderMatching

    encoder = ResnetEncoder(num_layers=18, pretrained=True, num_input_images=1).to('cuda')

    # 修复：避免使用 'int' 作为变量名
    input_tensor = torch.randn(1, 3, 320, 1024).cuda()  # 将 'int' 改为 'input_tensor'
    edges = torch.randn(1, 3, 320, 1024).cuda()

    model = ViTDepthDecoder().to('cuda')
    feats = encoder(input_tensor)  # 使用新变量名
    out = model(feats, edges)
    print(out['disp', 0].size())