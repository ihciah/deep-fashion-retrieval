# -*- coding: utf-8 -*-
# Debug script, useless...

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.autograd import Variable
import torchvision
from config import *


def fun(m, i, o):
    m.test.copy_(o.data.squeeze())

model_conv = torchvision.models.resnet50(pretrained=True)
num_ftrs = model_conv.fc.in_features
model_conv.fc = nn.Linear(num_ftrs, INTER_DIM)
model_conv.add_module('fc2', nn.Linear(INTER_DIM, CATEGORIES))

fc_layer = model_conv._modules.get('fc')
fc_layer.register_buffer("test", torch.zeros(INTER_DIM))
fc_layer.register_forward_hook(fun)
model_conv(Variable(torch.zeros(1, 3, 224, 244)))
print(1)
