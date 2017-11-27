# -*- coding: utf-8 -*-

import os
from config import *
from utils import *
from net import gen_model
from torchvision import transforms
from torch.autograd import Variable
import torch.nn as nn
from data import Fashion


class FeatureExtractor(nn.Module):
    def __init__(self, submodule):
        super(FeatureExtractor, self).__init__()
        self.submodule = submodule

    def forward(self, x):
        for name, module in self.submodule._modules.items()[:-1]:
            x = module(x)
        return x
model = gen_model(model_path=DUMPED_MODEL).cuda(GPU_ID)
extractor = FeatureExtractor(model)

data_transform_test = transforms.Compose([
    transforms.Scale(CROP_SIZE),
    transforms.CenterCrop(CROP_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


test_loader = torch.utils.data.DataLoader(
    Fashion(image_only=True, transform=data_transform_test),
    batch_size=TEST_BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=True
)

for batch_idx, (data, data_path) in enumerate(test_loader):
    data = Variable(data).cuda()
    result = extractor(data)
    result = result.cpu().data.numpy()
    for i in range(len(data_path)):
        path = data_path[i]
        feature = result[i].squeeze()
        dump_feature(feature, path)
    if batch_idx % LOG_INTERVAL == 0:
        print("{} / {}".format(batch_idx * TEST_BATCH_SIZE, len(test_loader.dataset)))
