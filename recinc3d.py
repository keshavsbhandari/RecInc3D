import torch
import torch.nn as nn
from torch.autograd import Variable

from groupconv import GroupConv
from lstmincell import LSTMInCell
from selfattention import SelfAttention
import os

class RecInc3D(nn.Module):
    def __init__(self,
                 cin=3,
                 cout=32,
                 lin=48,
                 num_classes=101,
                 is3d = True):
        super(RecInc3D, self).__init__()

        self.num_classes = num_classes

        if is3d:
            self.G1 = GroupConv(cin, cout, (7, 7, 7), (1, 2, 2),batchnorm = True)
            self.G2 = GroupConv(lin, cout * 2, (5, 5, 5), 1, batchnorm = False)
            self.G3 = GroupConv(lin * 2, cout * 4, (2, 3, 3), 1, batchnorm = True)
        else:
            self.G1 = GroupConv(cin, cout, (1, 7, 7), (1, 2, 2))
            self.G2 = GroupConv(lin, cout * 2, (1, 5, 5), 1)
            self.G3 = GroupConv(lin * 2, cout * 4, (1, 3, 3), 1)

        self.C1 = LSTMInCell(cout, lin)
        self.C2 = LSTMInCell(cout * 2, lin * 2)
        self.C3 = LSTMInCell(cout * 4, lin * 4)

        self.G4 = GroupConv(lin * 4, cout * 8, (1, 2, 2), 1, batchnorm = False)
        self.C4 = LSTMInCell(cout * 8, lin * 8)

        self.conv = nn.Conv3d(lin * 8, cout * 16, (1, 2, 2), 1)
        self.maxp = nn.MaxPool3d((1, 3, 3), 1)

        self.attn = SelfAttention(cout * 16)

        self.fc = nn.Linear(5120, self.num_classes)

        self.probs = nn.Softmax(dim=0)

    def forward(self, x, step):
        # Note That framesequence is a list of tensor with dimension
        # [batch,channel,depth,width,height]


        # for step in range(steps):
        #     frame = framesequence[:,step,:]
        #     x = self.C1(self.G1(frame), step)
        #     x = self.C2(self.G2(x), step)
        #     x = self.C3(self.G3(x), step)
        #     x = self.C4(self.G4(x), step)


        x = self.C1(self.G1(x), step)
        x = self.C2(self.G2(x), step)
        x = self.C3(self.G3(x), step)
        x = self.C4(self.G4(x), step)

        x = self.conv(x)
        x = self.maxp(x)
        x, attnx = self.attn(x)
        x = self.fc(x.view(x.shape[0], -1))
        x = self.probs(x)
        return x