# -*- coding: utf-8 -*-
# Debug script, useless...

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.autograd import Variable
import torchvision
from config import *


a = torch.zeros(32,3,224,224)
print(torch.cat((a,a,a),0))
