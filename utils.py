# -*- coding: utf-8 -*-

from config import *
import os
import torch
import numpy as np


def dump_model(model, epoch, batch_idx="final"):
    dump_folder = os.path.join(DATASET_BASE, 'models')
    if not os.path.isdir(dump_folder):
        os.mkdir(dump_folder)
    save_path = os.path.join(dump_folder, "model_{}_{}.pth.tar".format(epoch, batch_idx))
    torch.save(model.state_dict(), save_path)
    return save_path


def load_model(path=None):
    if not path:
        return None
    full = os.path.join(DATASET_BASE, 'models', path)
    for i in [path, full]:
        if os.path.isfile(i):
            return torch.load(i)
    return None


def dump_feature(feat, img_path):
    feat_folder = os.path.join(DATASET_BASE, 'features')
    if not os.path.isdir(feat_folder):
        os.mkdir(feat_folder)
    np_path = img_path.replace("/", "+")
    np_path = os.path.join(feat_folder, np_path)
    np.save(np_path, feat)
