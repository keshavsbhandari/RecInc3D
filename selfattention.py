import torch
import torch.nn as nn
from torch.autograd import Variable
import os



class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.f = nn.Conv3d(in_channels=in_channels,
                           out_channels=in_channels // 8,
                           kernel_size=1)
        self.g = nn.Conv3d(in_channels=in_channels,
                           out_channels=in_channels // 8,
                           kernel_size=1)
        self.h = nn.Conv3d(in_channels=in_channels,
                           out_channels=in_channels,
                           kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        m_batchsize, C, depth, width, height = x.size()
        self.fx = self.f(x).view(m_batchsize, -1, depth * width * height).permute(0, 2, 1)
        self.gx = self.g(x).view(m_batchsize, -1, depth * width * height)

        energy = torch.bmm(self.fx, self.gx)
        attention = self.softmax(energy)
        self.hx = self.h(x).view(m_batchsize, -1, depth * height * width)

        out = torch.bmm(self.hx, attention.permute(0, 2, 1))

        out = out.view(m_batchsize, C, depth, width, height)

        out = self.gamma * out + x
        return out, attention