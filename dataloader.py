import torch
from PIL import Image
import json
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import os
import random
from pathlib import Path
import imageio
from utils import getXorY, getdatalist, data


getdata = lambda :(torch.stack([torch.randn((1,3,1,256,256))]*100),torch.randint(0,10,(1,)))

class VideoDataSet(Dataset):
    def __init__(self,
                 data_list,
                 resize = 256,
                 sequence_step = 10,
                 transform = None,
                 train = True):

        self.data = data_list
        self.resize = resize
        self.transform = transform
        self.train = train
        self.sequence_step = sequence_step

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.getframes(self.data[idx])

    def getframes(self, videopath):
        frames = getXorY(videopath, self.sequence_step, self.train)
        stacks = []
        for frame in frames.frames:
            stacks.append(self.transform(Image.open(frame)))
        X = torch.stack(stacks)
        return data(frames = X, label = frames.label)

class VideoDataLoader():
    def __init__(self,
                 batch_size = 10,
                 num_workers = 4,
                 sequence_step = 10,
                 pin_memory = True,
                 resize = 256):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.sequence_step = sequence_step
        self.resize = resize

    def __call__(self, *args, **kwargs):
        return self.load(train = True), self.load(train = False)

    def load(self,train):
        self.data_list = getdatalist(train = train)
        dataset = VideoDataSet(data_list = self.data_list,
                               transform = transforms.Compose([transforms.Resize(self.resize),
                                                              transforms.ToTensor(),
                                                              transforms.Lambda(lambda x:x.unsqueeze(1))]),
                               resize = self.resize,
                               train = train,
                               sequence_step = self.sequence_step)
        print('===>{} Data : {}'.format({True:'Training',False:'Testing'}.get(train),
                                        len(self.data_list)))

        return DataLoader(dataset = dataset,
                          batch_size = {False:1,True:self.batch_size}.get(train),
                          shuffle = train,
                          num_workers = self.num_workers,
                          pin_memory = self.pin_memory)


if __name__ == '__main__':

    dataloader = VideoDataLoader()
    train, test = dataloader()

    print(len(train))



