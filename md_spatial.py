import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


def split_layer(total_channels, num_groups):
    split = [int(np.ceil(total_channels / num_groups)) for _ in range(num_groups)]
    split[num_groups - 1] += total_channels - sum(split)
    return split


class DepthwiseConv2D(nn.Module):
    def __init__(self, in_channels, kernal_size, stride, bias=False):
        super(DepthwiseConv2D, self).__init__()
        padding = (kernal_size - 1) // 2

        self.depthwise_conv = nn.Conv2d(in_channels, in_channels, kernel_size=kernal_size, padding=padding, stride=stride, groups=in_channels, bias=bias)

    def forward(self, x):
        out = self.depthwise_conv(x)
        return out

class SpatialWeighting(nn.Module):

    def __init__(self,
                 channels,
                 ratio=16,
                 conv_cfg=None,
                 act_cfg=(dict(type='ReLU'), dict(type='Sigmoid'))):
        super().__init__()

        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Sequential(nn.Conv2d(
            in_channels=channels,
            out_channels=int(channels / ratio),
            kernel_size=1,
            stride=1),
            nn.ReLU(inplace=True))


        self.conv2 = nn.Sequential(nn.Conv2d(
            in_channels=int(channels / ratio),
            out_channels=channels,
            kernel_size=1,
            stride=1),
            nn.Sigmoid())

    def forward(self, x):
        out = self.global_avgpool(x)
        out = self.conv1(out)
        out = self.conv2(out)
        return x * out


class GroupConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, n_chunks=1, bias=False):
        super(GroupConv2D, self).__init__()
        self.n_chunks = n_chunks
        self.split_in_channels = split_layer(in_channels, n_chunks)
        split_out_channels = split_layer(out_channels, n_chunks)

        if n_chunks == 1:
            self.group_conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias)
        else:
            self.group_layers = nn.ModuleList()
            for idx in range(n_chunks):
                self.group_layers.append(nn.Conv2d(self.split_in_channels[idx], split_out_channels[idx], kernel_size=kernel_size, bias=bias))

    def forward(self, x):
        if self.n_chunks == 1:
            return self.group_conv(x)
        else:
            split = torch.split(x, self.split_in_channels, dim=1)
            out = torch.cat([layer(s) for layer, s in zip(self.group_layers, split)], dim=1)
            return out


class MDConv(nn.Module):
    def __init__(self, out_channels, n_chunks, stride=1, bias=False):
        super(MDConv, self).__init__()
        self.n_chunks = n_chunks
        self.split_out_channels = split_layer(out_channels, n_chunks)

        self.layers = nn.ModuleList()
        self.spatts = nn.ModuleList()
        for idx in range(self.n_chunks):
            kernel_size = 2 * idx + 3
            self.layers.append(DepthwiseConv2D(self.split_out_channels[idx], kernal_size=kernel_size, stride=stride, bias=bias))
            self.spatts.append(SpatialWeighting(self.split_out_channels[idx],4))

    def forward(self, x):
        split = torch.split(x, self.split_out_channels, dim=1)
        out = torch.cat([layer(s)*spatt(s) for layer,spatt, s in zip(self.layers,self.spatts, split)], dim=1)
        return out

import torchvision
temp = torch.randn((16, 16, 32, 32))
group = MDConv(16, n_chunks=2)
print('group=',group)
print(group(temp).size())
# trace_model = torch.jit.trace(group, temp)
# trace_model.save("mtrace.pth")
# import netron
# netron.start("E:/软件安装/code/Dite-HRNet-main/tools/mtrace.pth")
