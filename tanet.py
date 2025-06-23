"""Dual Attention Network"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from backbone import ResNet50
import cv2
import json
import numpy as np


class DANet(ResNet50):
    r"""Pyramid Scene Parsing Network

    Parameters
    ----------
    nclass : int
        Number of categories for the training dataset.
    backbone : string
        Pre-trained dilated backbone network type (default:'resnet50'; 'resnet50',
        'resnet101' or 'resnet152').
    norm_layer : object
        Normalization layer used in backbone network (default: :class:`mxnet.gluon.nn.BatchNorm`;
        for Synchronized Cross-GPU BachNormalization).
    aux : bool
        Auxiliary loss.
    Reference:
        Jun Fu, Jing Liu, Haijie Tian, Yong Li, Yongjun Bao, Zhiwei Fang,and Hanqing Lu.
        "Dual Attention Network for Scene Segmentation." *CVPR*, 2019
    """

    def __init__(self, nclass, aux=True, qkv_VFA_flag=np.array([1, 1, 1]), **kwargs):
        super(DANet, self).__init__(nclass)
        self.head = _DAHead(2048, nclass, aux, qkv_VFA_flag, **kwargs)
        self.aux = True
        self.__setattr__('exclusive', ['head'])

    def forward(self, x, loss_factor_map):
        size = x.size()[2:]
        # print(x.shape)
        feature_map, _ = self.base_forward(x)
        # print(feature_map[3].shape)
        c3, c4 = feature_map[2], feature_map[3]

        # only the c4(b,c,h,w)->(b,2048,64,32) is used
        outputs = []
        x = self.head(c4, loss_factor_map)
        x0 = F.interpolate(x[0], size, mode='bilinear', align_corners=True)
        outputs.append(x0)

        if self.aux:
            # print('x[1]:{}'.format(x[1].shape))
            x1 = F.interpolate(x[1], size, mode='bilinear', align_corners=True)
            x2 = F.interpolate(x[2], size, mode='bilinear', align_corners=True)
            outputs.append(x1)
            outputs.append(x2)
        return outputs


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


# This is the spatial attention
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


class _DAHead(nn.Module):
    def __init__(self, in_channels, nclass, aux=True, qkv_VFA_flag=np.array([1, 1 ,1]), norm_layer=nn.BatchNorm2d, norm_kwargs=None, **kwargs):
        super(_DAHead, self).__init__()
        self.aux = aux
        inter_channels = in_channels // 4
        self.conv_p1 = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
            norm_layer(inter_channels, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True)
        )
        self.conv_c1 = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
            norm_layer(inter_channels, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True)
        )
        self.pam = _PositionAttentionModule(inter_channels, **kwargs)
        self.cam = _ChannelAttentionModule(**kwargs)
        self.vfam = _VisualFieldAttentionModule(inter_channels, qkv_VFA_flag, **kwargs)
        self.conv_p2 = nn.Sequential(
            nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
            norm_layer(inter_channels, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True)
        )
        self.conv_c2 = nn.Sequential(
            nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
            norm_layer(inter_channels, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True)
        )
        self.out = nn.Sequential(
            nn.Dropout(0.1),
            nn.Conv2d(inter_channels, nclass, 1)
        )
        if aux:
            self.conv_p3 = nn.Sequential(
                nn.Dropout(0.1),
                nn.Conv2d(inter_channels, nclass, 1)
            )
            self.conv_c3 = nn.Sequential(
                nn.Dropout(0.1),
                nn.Conv2d(inter_channels, nclass, 1)
            )

    def forward(self, x, loss_factor_map):
        feat_p = self.conv_p1(x)
        feat_p = self.pam(feat_p)
        feat_p = self.conv_p2(feat_p)

        feat_c = self.conv_c1(x)
        feat_c = self.cam(feat_c)
        feat_c = self.conv_c2(feat_c)

        feat_visual = self.conv_p1(x)
        feat_visual = self.vfam(feat_visual, loss_factor_map)
        feat_visual = self.conv_p2(feat_visual)


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


def get_tanet(backbone='resnet50', pretrained_base=True, qkv_VFA_flag=np.array([1, 1, 1]), **kwargs):
    cityspaces_numclass = 19
    model = DANet(cityspaces_numclass, backbone=backbone, pretrained_base=pretrained_base, qkv_VFA_flag=qkv_VFA_flag, **kwargs)
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
    img = torch.randn(2, 3, 512, 1024)
    with open("vrf_tools/alldata.json") as fin:
        dat = json.loads(fin.read())
    
    value_map = np.array(dat['data']['18']['R'][0]['hvf'])
    loss_map = np.array(dat['data']['18']['R'][0]['td'])
    loss_map[3][7] = -30
    loss_map[4][7] = -30
    normal_map = value_map - loss_map
    
    img_height = img.shape[2]
    img_width = img.shape[3]
    inter_method = cv2.INTER_LINEAR
    inter_size = (img_width // 16, img_height // 16)
    loss_map = cv2.resize(loss_map, inter_size, interpolation = inter_method)
    value_map = cv2.resize(value_map, inter_size, interpolation = inter_method)
    normal_map = value_map - loss_map

    loss_map_height = loss_map.shape[0]
    loss_map_width = loss_map.shape[1]
    loss_factor_map = np.empty((loss_map_height, loss_map_width))
    for m in range(loss_map_height):
        for n in range(loss_map_width):
            cur_loss = loss_map[m][n]
            cur_normal = normal_map[m][n]

            loss_factor = check_grayscale(cur_loss, cur_normal)
            if loss_factor < -0.2:
                loss_factor = 1
            else:
                loss_factor = 0
            loss_factor_map[m][n] = loss_factor
    
    # print(loss_factor_map.shape, loss_factor_map[0][0])
    model = get_tanet()
    outputs = model(img, loss_factor_map)
    print()
    print(outputs[0].shape)
