
import torch
import torch.nn as nn
from typing import List
from collections import OrderedDict

from . import util as utils
from .modules import Flatten, Activation


class EncoderMixin:
    """Add encoder functionality such as:
        - output channels specification of feature tensors (produced by encoder)
        - patching first convolution for arbitrary input channels
    """

    @property
    def out_channels(self):
        """Return channels dimensions for each tensor of forward output of encoder"""
        return self._out_channels[: self._depth + 1]

    def set_in_channels(self, in_channels):
        """Change first convolution chennels"""
        if in_channels == 3:
            return

        self._in_channels = in_channels
        if self._out_channels[0] == 3:
            self._out_channels = tuple(
                [in_channels] + list(self._out_channels)[1:])

        utils.patch_first_conv(model=self, in_channels=in_channels)

    def get_stages(self):
        """Method should be overridden in encoder"""
        raise NotImplementedError

    def make_dilated(self, stage_list, dilation_list):
        stages = self.get_stages()
        for stage_indx, dilation_rate in zip(stage_list, dilation_list):
            utils.replace_strides_with_dilation(
                module=stages[stage_indx],
                dilation_rate=dilation_rate,
            )


class SegmentationHeads(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, activation=None, upsampling=1):
        super().__init__()
        assert (len(in_channels) == len(out_channels))
        self.segmentationheads = []
        for in_channel, out_channel in zip(in_channels, out_channels):
            self.segmentationheads.append(SegmentationHead_3x1(in_channel, out_channel, activation, upsampling))
        self.segmentationheads = nn.ModuleList(self.segmentationheads)

    def forward(self, src):
        outputs = [segmentationhead(src) for segmentationhead in self.segmentationheads]
        return torch.cat(outputs, dim=1)
        

class SegmentationHead_3x1(nn.Module):
    
    def __init__(self, in_channels, out_channels, activation=None, upsampling=1, with_bn=False):
        super().__init__()
        self.conv2d_0 = nn.Conv2d(in_channels, in_channels, (3, 3), padding=(1, 1), bias=not with_bn)
        self.bn = nn.BatchNorm2d(in_channels) if with_bn else nn.Sequential()
        self.relu = nn.ReLU()
        self.conv2d_1 = nn.Conv2d(in_channels, out_channels, (1, 1))
        self.upsampling = nn.UpsamplingBilinear2d(
            scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        self.activation = Activation(activation)
    
    def forward(self, src):
        out = self.conv2d_0(src)
        out = self.bn(out)
        out = self.relu(out)
        out = self.conv2d_1(out)
        out = self.upsampling(out)
        out = self.activation(out)
        return out


class SegmentationHead(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size=3, activation=None, upsampling=1):
        conv2d = nn.Conv2d(in_channels, out_channels,
                           kernel_size=kernel_size, padding=kernel_size // 2)
        upsampling = nn.UpsamplingBilinear2d(
            scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        activation = Activation(activation)
        super().__init__(conv2d, upsampling, activation)


class ClassificationHead(nn.Sequential):

    def __init__(self, in_channels, classes, pooling="avg", dropout=0.2, activation=None):
        if pooling not in ("max", "avg"):
            raise ValueError(
                "Pooling should be one of ('max', 'avg'), got {}.".format(pooling))
        pool = nn.AdaptiveAvgPool2d(
            1) if pooling == 'avg' else nn.AdaptiveMaxPool2d(1)
        flatten = Flatten()
        dropout = nn.Dropout(
            p=dropout, inplace=True) if dropout else nn.Identity()
        linear = nn.Linear(in_channels, classes, bias=True)
        activation = Activation(activation)
        super().__init__(pool, flatten, dropout, linear, activation)
