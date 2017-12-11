# -*- coding: utf-8 -*-

import torchvision
import torch.nn as nn
from config import *
from utils import *
from torch.autograd import Variable


class f_model(nn.Module):
    def __init__(self, freeze_param=False, inter_dim=INTER_DIM, num_classes=CATEGORIES, model_path=None):
        super(f_model, self).__init__()
        self.backbone = torchvision.models.resnet50(pretrained=True)
        state_dict = self.backbone.state_dict()
        num_features = self.backbone.fc.in_features
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
        self.backbone.load_state_dict(dict([(k, v) for k, v in state_dict if k in self.backbone.state_dict().keys()]))
        if freeze_param:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.fc = nn.Linear(num_features, inter_dim)
        self.fc2 = nn.Linear(inter_dim, num_classes)
        state = load_model(model_path)
        if state:
            state = dict([(k, v) for k, v in state if k in self.state_dict().keys()])
            self.load_state_dict(state)

    def forward(self, x):
        conv_out = self.backbone(x)
        inter_out = self.fc(conv_out)
        out = self.fc2(inter_out)
        return out, inter_out


if __name__ == "__main__":
    model = f_model()
    print(1)
