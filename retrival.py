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
    list_eval_partition = os.path.join(DATASET_BASE, r'Eval', r'list_eval_partition.txt')
    names = read_lines(list_eval_partition)
    feats = np.array([])
    labels = []
    for i in names:
        f = load_feature(i)
        if f:
            labels.append(i)
            feats = np.vstack([feats, f])
    print("Feats load done {}".format(len(labels)))
    dist = cdist(np.expand_dims(feature, axis=0), feats)
    print("Distance calc done.")

f = load_feature("img/Sheer_Pleated-Front_Blouse/img_00000001.jpg")
naive_query(f)
