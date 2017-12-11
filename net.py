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
        model_dict = self.backbone.state_dict()
        model_dict.update({k: v for k, v in state_dict.items() if k in model_dict})
        self.backbone.load_state_dict(model_dict)
        if freeze_param:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.fc = nn.Linear(num_features, inter_dim)
        self.fc2 = nn.Linear(inter_dim, num_classes)
        state = load_model(model_path)
        if state:
            new_state = self.state_dict()
            new_state.update({k: v for k, v in state.items() if k in new_state})
            self.load_state_dict(new_state)

    def forward(self, x):
        conv_out = self.backbone(x)
        _, C, H, W = conv_out.data.size()
        inter_out = self.fc(conv_out.view(-1, C * H * W))
        out = self.fc2(inter_out)
        return out, inter_out


if __name__ == "__main__":
    model = f_model()
    print(1)
