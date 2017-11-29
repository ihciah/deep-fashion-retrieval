# -*- coding: utf-8 -*-
import os
import sys
import numpy as np
from scipy.spatial.distance import cdist
from torch.autograd import Variable
from config import *
from utils import *
from data import Fashion
from net import f_model
import time


@timer_with_task("Loading model")
def load_test_model():
    if not os.path.isfile(DUMPED_MODEL) and not os.path.isfile(os.path.join(DATASET_BASE, "models", DUMPED_MODEL)):
        print("No trained model file!")
        return
    model = f_model(model_path=DUMPED_MODEL).cuda(GPU_ID)
    model.eval()
    extractor = FeatureExtractor(model)
    return extractor


@timer_with_task("Loading feature database")
def load_feat_db():
    feat_all = os.path.join(DATASET_BASE, 'all_feat.npy')
    feat_list = os.path.join(DATASET_BASE, 'all_feat.list')
    if not os.path.isfile(feat_list) or not os.path.isfile(feat_all):
        print("No feature db file! Please run feature_extractor.py first.")
        return
    feats = np.load(feat_all)
    with open(feat_list) as f:
        labels = list(map(lambda x: x.strip(), f.readlines()))
    return feats, labels


def read_lines(path):
    with open(path) as fin:
        lines = fin.readlines()[2:]
        lines = list(filter(lambda x: len(x) > 0, lines))
        names = list(map(lambda x: x.strip().split()[0], lines))
    return names


@timer_with_task("Doing query")
def naive_query(feature, feats, labels, retrieval_top_n=5):
    if feature is None:
        print("Input feature is None")
        return
    dist = 1 - cdist(np.expand_dims(feature, axis=0), feats)[0]
    ind = np.argpartition(dist, -retrieval_top_n)[-retrieval_top_n:][::-1]
    return zip([labels[i] for i in ind], dist[ind])


@timer_with_task("Extracting image feature")
def dump_single_feature(img_path, extractor):
    paths = [img_path, os.path.join(DATASET_BASE, img_path)]
    for i in paths:
        if not os.path.isfile(i):
            continue
        single_loader = torch.utils.data.DataLoader(
            Fashion(type="single", img_path=i, transform=data_transform_test),
            batch_size=1, num_workers=NUM_WORKERS, pin_memory=True
        )
        data = list(single_loader)[0]
        data = Variable(data).cuda(GPU_ID)
        result = extractor(data)
        result = result.cpu().data.numpy()[0].squeeze()
        return result
    return None


def visualize(original, result, cols=1):
    import matplotlib.pyplot as plt
    import cv2
    n_images = len(result) + 1
    titles = ["Original"] + ["Score: {:.4f}".format(v) for k, v in result]
    images = [original] + [k for k, v in result]
    images = list(map(lambda x: cv2.cvtColor(cv2.imread(os.path.join(DATASET_BASE, x)), cv2.COLOR_BGR2RGB), images))
    fig = plt.figure()
    for n, (image, title) in enumerate(zip(images, titles)):
        a = fig.add_subplot(cols, np.ceil(n_images / float(cols)), n + 1)
        plt.imshow(image)
        a.set_title(title)
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_images * 0.25)
    plt.show()

if __name__ == "__main__":
    example = "img/Sheer_Pleated-Front_Blouse/img_00000005.jpg"
    if len(sys.argv) > 1 and sys.argv[1].endswith("jpg"):
        example = sys.argv[1]

    extractor = load_test_model()
    feats, labels = load_feat_db()
    f = dump_single_feature("img/Sheer_Pleated-Front_Blouse/img_00000005.jpg", extractor)
    result = naive_query(f, feats, labels, 5)

    print(result)
    visualize(example, result)
