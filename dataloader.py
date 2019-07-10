import torch

getdata = lambda :(torch.stack([torch.randn((1,3,1,256,256))]*10),torch.randint(0,10,(1,)))