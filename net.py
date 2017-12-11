# -*- coding: utf-8 -*-

import torchvision
import torch.nn as nn
import torch.nn.functional as F
from config import *
from utils import *
from torch.autograd import Variable


class f_model(nn.Module):
    '''
    input: N * 3 * 224 * 224
    output: N * num_classes, N * inter_dim, N * C' * 7 * 7
    '''
    def __init__(self, freeze_param=False, inter_dim=INTER_DIM, num_classes=CATEGORIES, model_path=None):
        super(f_model, self).__init__()
        self.backbone = torchvision.models.resnet50(pretrained=True)
        state_dict = self.backbone.state_dict()
        num_features = self.backbone.fc.in_features
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])
        model_dict = self.backbone.state_dict()
        model_dict.update({k: v for k, v in state_dict.items() if k in model_dict})
        self.backbone.load_state_dict(model_dict)
        if freeze_param:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.avg_pooling = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(num_features, inter_dim)
        self.fc2 = nn.Linear(inter_dim, num_classes)
        state = load_model(model_path)
        if state:
            new_state = self.state_dict()
            new_state.update({k: v for k, v in state.items() if k in new_state})
            self.load_state_dict(new_state)

    def forward(self, x):
        x = self.backbone(x)
        pooled = self.avg_pooling(x)
        inter_out = self.fc(pooled.view(pooled.size(0), -1))
        out = self.fc2(inter_out)
        return out, inter_out, x


class c_model(nn.Module):
    '''
    input: N * C * 224 * 224
    output: N * C * 7 * 7
    '''
    def __init__(self, pooling_size=32):
        super(c_model, self).__init__()
        self.pooling = nn.AvgPool2d(pooling_size)

    def forward(self, x):
        return self.pooling(x)


class p_model(nn.Module):
    '''
    input: N * C * W * H
    output: N * 1 * W * H
    '''
    def __init__(self):
        super(p_model, self).__init__()

    def forward(self, x):
        n, c, w, h = x.size()
        x = x.view(n, c, w * h).permute(0, 2, 1)
        pooled = F.avg_pool1d(x, c)
        return pooled.view(n, 1, w, h)


if __name__ == "__main__":
    model = f_model()
    # model = c_model()
    print(1)
