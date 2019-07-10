import torch
import torch.nn as nn
from torch.autograd import Variable


class GroupConv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 *args,
                 **kwargs):

        super(GroupConv, self).__init__()

        if not isinstance(kernel_size, tuple): kernel_size = (kernel_size,) * 3
        if not isinstance(stride, tuple): stride = (stride,) * 3

        self.conv = nn.Conv3d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              stride=stride)
        self.pool = nn.MaxPool3d(kernel_size=(1, 3, 3),
                                 stride=(1, 2, 2))
        self.relu = nn.ReLU()

        self.bnorm = nn.BatchNorm3d(out_channels, affine=True)

        self.groupconv = nn.Sequential(self.conv,
                                       self.pool,
                                       self.relu,
                                       self.bnorm)

    def forward(self, x):
        return self.groupconv(x)