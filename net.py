# -*- coding: utf-8 -*-
from torch.autograd import Variable
from torchvision import models
from torchvision import transforms
from PIL import Image
import argparse
import torch
import torchvision
import torch.nn as nn
import numpy as np
import os
from config import *


model_conv = torchvision.models.resnet50(pretrained=True)
for param in model_conv.parameters():
    param.requires_grad = False

num_ftrs = model_conv.fc.in_features
model_conv.fc = nn.Linear(num_ftrs, CATEGORIES)
model_conv = model_conv.cuda(GPU_ID)
