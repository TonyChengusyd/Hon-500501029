import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from math import sqrt
import cv2
from torchvision.models import resnet34, resnet50
# from network.attention import Attention0
from backbone import ResNet50
import numpy as np

"""
Multiscale Sparse Cross-Attention Network
"""
# dynamic sparse cross attention multi-scale feature fusion TANet
class DSCAMSFF_TANet(ResNet50):
    def __init__(self, nclass, aux=True, qkv_VFA_flag=np.array([1, 1, 1]), dim=[256, 512, 1024, 2048], k1=2, k2=3, g=8, **kwargs):
        super(DSCAMSFF_TANet, self).__init__(nclass)
        self.dscamsff = DSCAMSFF(dim=dim, k1=k1, k2=k2, g=g)  # can be changed
        self.head = TABlock(dim[-1], nclass, aux, qkv_VFA_flag, **kwargs)
        self.aux = True
        self.__setattr__('exclusive', ['head'])

    def forward(self, x, loss_factor_map):
        size = x.size()[2:]
        # print(x.shape)
        feature_map, _ = self.base_forward(x)
        # print(feature_map[3].shape)
        x = self.dscamsff(feature_map)

        # only the c4(b,c,h,w)->(b,2048,64,32) is used
        outputs = []
        x = self.head(x, loss_factor_map)
        x0 = F.interpolate(x[0], size, mode='bilinear', align_corners=True)
        outputs.append(x0)

        if self.aux:
            # print('x[1]:{}'.format(x[1].shape))
            x1 = F.interpolate(x[1], size, mode='bilinear', align_corners=True)
            x2 = F.interpolate(x[2], size, mode='bilinear', align_corners=True)
            outputs.append(x1)
            outputs.append(x2)
        return outputs


# Channel attention block
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(in_channels // reduction_ratio, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        avg_out = self.avg_pool(x).view(x.size(0), -1)
        channel_attention = self.fc(avg_out).view(x.size(0), x.size(1), 1, 1)
        return x * channel_attention


# Spatial attention block
class SpatialAttention(nn.Module):
    def __init__(self, in_channels):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        attention = self.conv(x)
        attention = self.sigmoid(attention)
        x = x * attention
        return x


# Channel and Spatial attention block
class CBAM(nn.Module):
    def __init__(self, in_channels, reduction_ratio=4):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(in_channels)

    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


# Channel grounp attention block
class GroupCBAMEnhancer(nn.Module):
    def __init__(self, channel, group=8, cov1=1, cov2=1):
        super().__init__()
        self.cov1 = None
        self.cov2 = None
        if cov1 != 0:
            self.cov1 = nn.Conv2d(channel, channel, kernel_size=1)
        self.group = group
        cbam = []
        for i in range(self.group):
            cbam_ = CBAM(channel // group)
            cbam.append(cbam_)

        self.cbam = nn.ModuleList(cbam)
        self.sigomid = nn.Sigmoid()
        if cov2 != 0:
            self.cov2 = nn.Conv2d(channel, channel, kernel_size=1)

    def forward(self, x):
        x0 = x
        if self.cov1 is not None:
            x = self.cov1(x)
        y = torch.split(x, x.size(1) // self.group, dim=1)
        mask = []
        for y_, cbam in zip(y, self.cbam):
            y_ = cbam(y_)
            y_ = self.sigomid(y_)

            mk = y_

            mean = torch.mean(y_, [1, 2, 3])
            mean = mean.view(-1, 1, 1, 1)

            # mean = torch.mean(y_,[2,3])
            # mean = mean.view(mean.size(0),mean.size(1),1,1)

            gate = torch.ones_like(y_) * mean
            mk = torch.where(y_ > gate, 1, y_)

            mask.append(mk)
        mask = torch.cat(mask, dim=1)
        # print(mask.shape)
        x = x * mask
        if self.cov2 is not None:
            x = self.cov2(x)
        x = x + x0
        return x


# dynamic sparse cross attention...
class DSCA(nn.Module):
    def __init__(self, dim, num_heads=8, avp_kernel=[3, 5, 7], avp_s=[1, 1, 1], avp_pad=[1, 2, 3],
                 qkv_bias=False, qk_scale=None, attn_drop_ratio=0., proj_drop_ratio=0):
        super(DSCA, self).__init__()
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)

        self.dynamic_k = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // 4, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // 4, num_heads, kernel_size=1),
            nn.Softmax(dim=1)
        )
        self.avgpool1 = nn.AvgPool2d(kernel_size=avp_kernel[0], stride=avp_s[0], padding=avp_pad[0])
        self.avgpool2 = nn.AvgPool2d(kernel_size=avp_kernel[1], stride=avp_s[1], padding=avp_pad[1])

        self.avgpool3 = nn.AvgPool2d(kernel_size=avp_kernel[2], stride=avp_s[2], padding=avp_pad[2])
        self.maxpool1 = nn.MaxPool2d(kernel_size=avp_kernel[0], stride=avp_s[0], padding=avp_pad[0])

        self.maxpool2 = nn.MaxPool2d(kernel_size=avp_kernel[1], stride=avp_s[1], padding=avp_pad[1])
        self.maxpool3 = nn.MaxPool2d(kernel_size=avp_kernel[2], stride=avp_s[2], padding=avp_pad[2])

        self.layer_norm = nn.LayerNorm(dim)

    def forward(self, x, y):
        b, c, h, w = x.shape
        y1 = self.avgpool1(y) + self.maxpool1(y)
        y2 = self.avgpool2(y) + self.maxpool2(y)
        y3 = self.avgpool3(y) + self.maxpool3(y)
        # y = torch.cat([y1.flatten(-2,-1),y2.flatten(-2,-1),y3.flatten(-2,-1)],dim = -1)
        y = y1 + y2 + y3
        y = y.flatten(-2, -1)
        y = y.transpose(1, 2)
        y = self.layer_norm(y)
        # Compute dynamic top-k sparsity for each head
        k_values = self.dynamic_k(x).view(b, self.num_heads, 1, 1)

        x = rearrange(x, 'b c h w -> b (h w) c')
        # y = rearrange(y,'b c h w -> b (h w) c')
        B, N1, C = y.shape
        # print(y.shape)
        kv = self.kv(y).reshape(B, N1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        # qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        k_values = (k_values * attn.size(-1)).long().clamp(1, attn.size(-1))

        sparse_attn = []
        for head_idx in range(self.num_heads):
            # 动态调整 top-k 稀疏性
            k_values_for_head = k_values[:, head_idx, 0, 0]  # 提取当前 head 的 k 值 
            attn_head = attn[:, head_idx]  # 当前 head 的注意力 (b, seq_len, seq_len)
            mask = torch.zeros_like(attn_head)

            for batch_idx in range(b):
                k_value = k_values_for_head[batch_idx].item()
                topk_indices = torch.topk(attn_head[batch_idx], k=k_value, dim=-1)[1]
                mask[batch_idx].scatter_(-1, topk_indices, 1.0)

            sparse_attn.append(attn_head * mask)
        attn = torch.stack(sparse_attn, dim=1).softmax(dim=-1)
        out = attn @ v
        out = out.transpose(1, 2).reshape(B, N, C)

        out = self.proj(out)
        out = self.proj_drop(out)
        out = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        return out


# dynamic sparse cross attention multi-scale feature fusion
class DSCAMSFF(nn.Module):
    def __init__(self, dim=[256, 512, 1024, 2048], k1=2, k2=3, g=8):
        """

        Args:
            dim: list
            k1:
            k2:
            g:
        """
        super(DSCAMSFF, self).__init__()
        # print(f'att = {att}')

        self.dim = dim
        dim_b = self.dim[1]
        dim_fc = self.dim[-1]

        self.msc1 = DSCA(dim=dim_b)
        self.msc2 = DSCA(dim=dim_b)
        self.msc3 = DSCA(dim=dim_b)
        self.gce = GroupCBAMEnhancer(dim_fc, g, 0, 0)
        self.cov1 = nn.Conv2d(dim[0], dim_b, 3, 2, 1)
        self.cov2 = nn.Conv2d(dim[1], dim_b, 3, 2, 1)
        self.cov3 = nn.Conv2d(dim[2], dim_b, 3, 2, 1)
        self.cov4 = nn.Conv2d(dim[3], dim_b, 1)

    def forward(self, feature_map_list):
        x1 = feature_map_list[0]
        x1 = self.cov1(x1)
        x2 = feature_map_list[1]
        x2 = self.cov2(x2)
        x3 = feature_map_list[2]
        x3 = self.cov3(x3)
        x4 = feature_map_list[3]
        x4 = self.cov4(x4)

        x1 = self.msc1(x4, x1)
        x2 = self.msc2(x4, x2)
        x3 = self.msc3(x4, x3)

        x = torch.cat([x4, x1, x2, x3], dim=1)
        x = self.gce(x)

        return x


#  VFA attention block, in the original paper by An Yan,
class _VisualFieldAttentionModule(nn.Module):
    """ Position attention module"""

    def __init__(self, in_channels, qkv_vfa_flag=np.array([1, 1, 1]), **kwargs):
        super(_VisualFieldAttentionModule, self).__init__()
        self.conv_b = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.conv_c = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.conv_d = nn.Conv2d(in_channels, in_channels, 1)
        self.alpha = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)
        self.qkv_vfa_flag = qkv_vfa_flag

    def forward(self, x, loss_factor_map):
        # print(x.size())
        batch_size, _, height, width = x.size()
        query_conv = self.conv_b(x)
        key_conv = self.conv_c(x)
        value_conv = self.conv_d(x)
        # +-----------------------------------+
        # print(type(query_conv))
        # print(query_conv.shape)
        # a = query_conv[0]
        # print(type(a), a.shape)
        # target_map = query_conv[0][0]
        # print(target_map)
        # print(type(target_map), target_map.shape)
        # channel_size = query_conv.shape[1]
        # for batch_idx in range(batch_size):
        #     for channel_idx in range(channel_size):
        #         cur_map = query_conv[batch_idx][channel_idx]
        #         # for h in range(width):
        #         #     for w in range(height):
        #         #         cur_map_pixel = cur_map[w][h]
        #         #         cur_loss_factor = loss_factor_map[h][w]
        #         #         if cur_loss_factor < -0.2:
        #         #             pass
        #         #         else:
        #         #             cur_map_pixel = 0.0
        #         #         cur_map[w][h] = cur_map_pixel
        loss_factor_map = torch.from_numpy(loss_factor_map).view(1, 1, height, width).float().to('cuda:0')
        # loss_factor_map = torch.from_numpy(loss_factor_map).view(1, 1, height, width).float()
        QKV_loss_factor_map = []
        for idx in range(len(self.qkv_vfa_flag)):
            QKV_loss_factor_map.append(torch.ones_like(loss_factor_map))
            if self.qkv_vfa_flag[idx] == 1:
                QKV_loss_factor_map[idx] = loss_factor_map

        query_conv = query_conv * QKV_loss_factor_map[0]
        key_conv = key_conv * QKV_loss_factor_map[1]
        value_conv = value_conv * QKV_loss_factor_map[2]
        # +-----------------------------------+
        feat_b = query_conv.view(batch_size, -1, height * width).permute(0, 2, 1)
        feat_c = key_conv.view(batch_size, -1, height * width)
        # print(query_conv.shape, feat_b.shape)
        # print(key_conv.shape, feat_c.shape)
        attention_s = self.softmax(torch.bmm(feat_b, feat_c))
        feat_d = value_conv.view(batch_size, -1, height * width)
        # print(attention_s.shape, feat_d.shape)
        feat_e = torch.bmm(feat_d, attention_s.permute(0, 2, 1)).view(batch_size, -1, height, width)
        out = self.alpha * feat_e + x

        return out


#  This is the spatial attention block
class _PositionAttentionModule(nn.Module):
    """ Position attention module"""

    def __init__(self, in_channels, **kwargs):
        super(_PositionAttentionModule, self).__init__()
        self.conv_b = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.conv_c = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.conv_d = nn.Conv2d(in_channels, in_channels, 1)
        self.alpha = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, _, height, width = x.size()
        query_conv = self.conv_b(x)
        feat_b = query_conv.view(batch_size, -1, height * width).permute(0, 2, 1)
        # feat_b = self.conv_b(x).view(batch_size, -1, height * width).permute(0, 2, 1)
        feat_c = self.conv_c(x).view(batch_size, -1, height * width)
        attention_s = self.softmax(torch.bmm(feat_b, feat_c))
        feat_d = self.conv_d(x).view(batch_size, -1, height * width)
        feat_e = torch.bmm(feat_d, attention_s.permute(0, 2, 1)).view(batch_size, -1, height, width)
        out = self.alpha * feat_e + x

        return out


# the channel attention in the DANet and the TANet
class _ChannelAttentionModule(nn.Module):
    """Channel attention module"""

    def __init__(self, **kwargs):
        super(_ChannelAttentionModule, self).__init__()
        self.beta = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, _, height, width = x.size()
        feat_a = x.view(batch_size, -1, height * width)
        feat_a_transpose = x.view(batch_size, -1, height * width).permute(0, 2, 1)
        attention = torch.bmm(feat_a, feat_a_transpose)
        attention_new = torch.max(attention, dim=-1, keepdim=True)[0].expand_as(attention) - attention
        attention = self.softmax(attention_new)

        feat_e = torch.bmm(attention, feat_a).view(batch_size, -1, height, width)
        out = self.beta * feat_e + x

        return out


# This block contains the spatial, channel and VFA attention
class TABlock(nn.Module):
    def __init__(self, in_channels, nclass, aux=True, qkv_VFA_flag=np.array([1, 1, 1]), norm_layer=nn.BatchNorm2d, norm_kwargs=None, **kwargs):
        super(TABlock, self).__init__()
        self.aux = aux
        inter_channels = in_channels // 4

        self.pam = _PositionAttentionModule(inter_channels, **kwargs)
        self.cam = _ChannelAttentionModule(**kwargs)
        self.vfam = _VisualFieldAttentionModule(inter_channels, qkv_VFA_flag, **kwargs)
        self.conv_p1 = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                     norm_layer(inter_channels, **({} if norm_kwargs is None else norm_kwargs)),
                                     nn.ReLU(True))
        self.conv_p2 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                     norm_layer(inter_channels, **({} if norm_kwargs is None else norm_kwargs)),
                                     nn.ReLU(True))
        self.conv_c1 = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                     norm_layer(inter_channels, **({} if norm_kwargs is None else norm_kwargs)),
                                     nn.ReLU(True))
        self.conv_c2 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                     norm_layer(inter_channels, **({} if norm_kwargs is None else norm_kwargs)),
                                     nn.ReLU(True))
        self.conv_v1 = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                     norm_layer(inter_channels, **({} if norm_kwargs is None else norm_kwargs)),
                                     nn.ReLU(True))
        self.conv_v2 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                     norm_layer(inter_channels, **({} if norm_kwargs is None else norm_kwargs)),
                                     nn.ReLU(True))
        self.out = nn.Sequential(nn.Dropout(0.1), nn.Conv2d(inter_channels, nclass, 1))
        if aux:
            self.conv_p3 = nn.Sequential(nn.Dropout(0.1), nn.Conv2d(inter_channels, nclass, 1))
            self.conv_c3 = nn.Sequential(nn.Dropout(0.1), nn.Conv2d(inter_channels, nclass, 1))

    def forward(self, x, loss_factor_map):
        feat_p = self.conv_p1(x)
        feat_p = self.pam(feat_p)
        feat_p = self.conv_p2(feat_p)

        feat_c = self.conv_c1(x)
        feat_c = self.cam(feat_c)
        feat_c = self.conv_c2(feat_c)

        feat_visual = self.conv_v1(x)
        feat_visual = self.vfam(feat_visual, loss_factor_map)
        feat_visual = self.conv_v2(feat_visual)

        feat_fusion = feat_p + feat_c + feat_visual

        outputs = []
        fusion_out = self.out(feat_fusion)
        outputs.append(fusion_out)
        if self.aux:
            p_out = self.conv_p3(feat_p)
            c_out = self.conv_c3(feat_c)
            outputs.append(p_out)
            outputs.append(c_out)

        return tuple(outputs)


def get_model(backbone='resnet50', pretrained_base=True, qkv_VFA_flag=np.array([1, 1, 1]), dim=[256, 512, 1024, 2048], k1=2, k2=3, g=8, **kwargs):
    cityspaces_numclass = 19
    model = DSCAMSFF_TANet(cityspaces_numclass, backbone=backbone, pretrained_base=pretrained_base, qkv_VFA_flag=qkv_VFA_flag, dim=dim, k1=k1, k2=k2, g=g, **kwargs)
    return model


def check_grayscale(x, normal):
    result = None

    if x == 0:
        result = 0
    elif -normal <= x < 0:
        result = x / normal
    elif 0 < x <= normal + 10:
        result = 0
    else:
        result = 255

    return result


if __name__ == '__main__':
    print('--------net test--------')
    net = get_model().to('cuda')
    input_x = torch.randn(1, 3, 1024, 512).cuda()
    loss_factor_map = np.random.rand(32, 64)
    output_x = net(input_x, loss_factor_map)
    print("The shape of the input: ", input_x.shape)
    print("The shape of the output: ", output_x[0].shape)



