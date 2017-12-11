# -*- coding: utf-8 -*-
# Debug script, useless...

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.autograd import Variable
import torchvision
from config import *
import numpy as np


from scipy import spatial

# dataSetI = [3, 45, 7, 2]
# dataSetII = [2, 54, 13, 15]
# result = 1 - spatial.distance.cosine(dataSetI, dataSetII)
# print(result)
#
# from sklearn.metrics.pairwise import cosine_similarity
# from scipy import sparse
# import numpy as np
# import scipy
#
# A = np.array([[0, 1, 0, 0, 1]])
# B = np.array([[0, 2, 0, 0, 2], [0, 4, 0, 0, 2]])
# result = 1-scipy.spatial.distance.cdist(A, B, metric='cosine')
# print(result)

x = np.random.random((1, 3, 224, 224))
x = torch.from_numpy(x).float()
var = Variable(x)
backbone = torchvision.models.resnet50(pretrained=True)
backbone(var)
