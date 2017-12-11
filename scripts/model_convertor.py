# -*- coding: utf-8 -*-

import torch
import os
from config import DATASET_BASE, DUMPED_MODEL
from collections import OrderedDict

NEW_MODEL_NAME = "model_5_final_converted.pth.tar"

if __name__ == '__main__':
    d = torch.load(os.path.join(DATASET_BASE, DUMPED_MODEL))
    d = OrderedDict([(k, v) if k != 'backbone.fc.weight' else ('fc.weight', v) for k, v in d.items()])
    d = OrderedDict([(k, v) if k != 'backbone.fc.bias' else ('fc.bias', v) for k, v in d.items()])
    torch.save(d, os.path.join(DATASET_BASE, NEW_MODEL_NAME))
