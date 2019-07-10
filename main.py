import torch
from recinc3d import RecInc3D
import os
from train import RecIn3DTrain

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if __name__ == '__main__':
    # model = RecInc3D(3,is3d=False).cuda()
    # x = torch.randn(10,3,1,256,256).cuda()
    # out = model([x]*10)
    # print(out.shape)
    r = RecIn3DTrain()
    r.run()


