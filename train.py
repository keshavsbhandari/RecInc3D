from recinc3d import RecInc3D
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.backends.cudnn as cudnn
from dataloader import getdata
from torch.autograd import Variable
from dataloader import VideoDataLoader
# from tqdm import tqdm
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class RecIn3DTrain():
    def __init__(self,
                 batch_size = 10,
                 num_workers = 4,
                 sequence_step = 20,
                 pin_memory = True,
                 resize = 128,
                 nb_epochs = 1,
                 lr = 1e-4,
                 resume = None,
                 start_epoch = None,
                 evaluate = None,
                 momentum = 0.9,
                 pretrained_weight = None):

        self.pretrained_weight = pretrained_weight
        self.resume = resume
        self.start_epoch = start_epoch
        self.nb_epochs = nb_epochs
        self.dataloader = VideoDataLoader(batch_size = batch_size,
                                        num_workers = num_workers,
                                        sequence_step = sequence_step,
                                        pin_memory = pin_memory,
                                        resize = resize)
        self.evaluate = evaluate
        self.sequence_step = sequence_step
        self.trainloader, self.testloader = self.dataloader()
        self.lr = lr
        self.momentum = momentum
        self.step = 0

    def load_pre_trained(self, weights):
        self.model.load_state_dict(torch.load(weights))

    def build_model(self):
        self.model = RecInc3D(3, is3d = False, num_classes = 101).cuda()
        self.model = torch.nn.DataParallel(self.model)

        # if self.pretrained_weight: self.load_pre_trained(self.pretrained_weight)

        self.criterion = nn.CrossEntropyLoss().cuda()
        self.optimizer = torch.optim.Adam(self.model.parameters())
        self.scheduler = ReduceLROnPlateau(self.optimizer, 'min', patience = 1, verbose = True)

    def resume_and_evaluate(self):
        if self.resume:
            pass
        else:
            pass

    def run(self):
        self.build_model()
        # self.resume_and_evaluate()

        cudnn.benchmark = True

        self.train()

        # for self.epoch in range(self.start_epoch, self.nb_epochs):
        #     self.train()
            #validate to get accuracy and loss
            #acc,val = self.validate()
            #isbest = acc>self.bestprec

            #lr_schedule
            #if is_best only save the model

    def train(self):
        for epochs in range(self.nb_epochs):
            for i,data in enumerate(self.trainloader):
                x,y = data.frames, data.label.view(-1)
                x = Variable(x).cuda()
                y = Variable(y).cuda()
                for step in range(self.sequence_step):
                    y_ = self.model(x[:,step,:],step)
                    loss = self.criterion(y_, y)
                    self.optimizer.zero_grad()
                    loss.backward(retain_graph=True)
                    self.optimizer.step()
                    print("#Epoch : \t{}\tstep : \t{}\tloss : \t{}\tseq# : \t{}".format(epochs, i, loss, step))

    def test(self):
        pass
