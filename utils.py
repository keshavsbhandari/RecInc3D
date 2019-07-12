from pathlib import Path
from bidict import bidict
from os.path import join as J
import torch
from itertools import cycle
from collections import namedtuple
import re
import random
import os



data = namedtuple('data',['frames','label'])

datapath = "/data/keshav/ucf/jpegs_256/"
idx = "/data/keshav/ucf/ucflist/classInd.txt"
testlist = "/data/keshav/ucf/ucflist/testlist01.txt"
trainlist = "/data/keshav/ucf/ucflist/trainlist01.txt"

getid = lambda x:int(re.sub('\D','',x))
order = lambda x:sorted(x,key = lambda u:getid(u.name))

def getdatalist(train = True):
    if train:
        return open(trainlist,'r').readlines()
    else:
        return open(testlist,'r').readlines()

def getMapper(idxpath = "/data/keshav/ucf/ucflist/classInd.txt"):
    indexes = [*map(lambda x: x.strip(), open(idxpath, 'r').readlines())]
    return bidict({y: torch.tensor([int(x)-1]) for x, y in map(lambda i: i.split(), indexes)})

mapper = getMapper()


def randomSequenceChunk(x, n):
    start = random.randint(0,len(x)-n)
    end = start + n
    return x[start:end]

# x is single instance from open(testlist,'r').readlines()
getactualtestpath = lambda testpath:J(datapath,testpath.strip().replace('.avi',''))
getactualtrainpath = lambda trainpath:J(datapath,trainpath.split('.avi ')[0])
getframesfrompath = lambda x,n,pathgetter:randomSequenceChunk((order([*Path(pathgetter(x)).glob("*.jpg")])),n)
getactualpath = {True:getactualtrainpath, False:getactualtestpath}

# path : Instance from trainlist or testlist
# Returns n random frames in sequence from a video
def getXorY(path,n = 10, train = True):
    frames = getframesfrompath(path, n, getactualpath.get(train))
    label = mapper.get(frames[0].parent.parent.name)
    return data(frames = frames, label = label)