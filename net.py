# -*- coding: utf-8 -*-

import torchvision
import torch.nn as nn
from config import *
from utils import *


def gen_model(freeze_param=False, model_path=None):
    model_conv = torchvision.models.resnet50(pretrained=True)
    if freeze_param:
        for param in model_conv.parameters():
            param.requires_grad = False
    num_ftrs = model_conv.fc.in_features
    model_conv.fc = nn.Linear(num_ftrs, INTER_DIM)
    model_conv.add_module('fc2', nn.Linear(INTER_DIM, CATEGORIES))
    if TRIPLET_WEIGHT:
        # Create a buffer to store feature
        model_conv.fc.register_buffer("feature", torch.zeros(INTER_DIM))
        # On every forward, copy feature to buffer
        model_conv.fc.register_forward_hook(lambda m, i, o: m.feature.copy_(o.data.squeeze()))
    state = load_model(model_path)
    if state:
        model_conv.load_state_dict(state)
    return model_conv

