# -*- coding: utf-8 -*-
import os
import numpy as np
from scipy.spatial.distance import cdist
from config import *
from utils import *


def read_lines(path):
    with open(path) as fin:
        lines = fin.readlines()[2:]
        lines = list(filter(lambda x: len(x) > 0, lines))
        names = list(map(lambda x: x.strip().split()[0], lines))
    return names


def naive_query(feature):
    if feature is None:
        print("Input feature is None")
        return
    feat_all = os.path.join(DATASET_BASE, 'all_feat.npy')
    feat_list = os.path.join(DATASET_BASE, 'all_feat.list')
    feats = np.load(feat_all)
    with open(feat_list) as f:
        labels = list(map(lambda x: x.strip(), f.readlines()))
    print("Feats load done {}".format(len(labels)))
    dist = cdist(np.expand_dims(feature, axis=0), feats)[0]
    print("Distance calc done.")
    ind = np.argpartition(dist, -5)[-5:]
    print(ind)
    for i in ind:
        print(labels[i])


f = load_feature("img/1981_Graphic_Ringer_Tee/img_00000033.jpg")
naive_query(f)
