from recinc3d import RecInc3D
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.backends.cudnn as cudnn
from dataloader import getdata
from torch.autograd import Variable
# from tqdm import tqdm


class RecIn3DTrain():
    def __init__(self,
                 nb_epochs = None,
                 lr = None,
                 batch_size = None,
                 resume = None,
                 start_epoch = None,
                 evaluate = None,
                 train_loader = None,
                 test_loader = None,
                 val_loader = None,
                 pretrained_weight = None):

        self.pretrained_weight = None
        self.resume = None
        self.epoch = None
        self.start_epoch = None
        self.nb_epochs = None

    def load_pre_trained(self, weights):
        self.model.load_state_dict(torch.load(weights))

    def build_model(self):
        self.model = RecInc3D(3, is3d = False).cuda()
        self.model = torch.nn.DataParallel(self.model)

        # if self.pretrained_weight: self.load_pre_trained(self.pretrained_weight)

        self.criterion = nn.CrossEntropyLoss().cuda()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr = 0.001, momentum = 0.9)
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
        for i in range(150):
            x,y = getdata()
            x = Variable(x).cuda()
            y = Variable(y).cuda()
            y_ = self.model(x)
            loss = self.criterion(y_, y)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            print(i,loss)


