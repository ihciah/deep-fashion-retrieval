# -*- coding: utf-8 -*-

import os
from config import *
from utils import *
from torch.autograd import Variable
from data import Fashion_attr_prediction, Fashion_inshop
from net import f_model, c_model, p_model


main_model = f_model(model_path=DUMPED_MODEL).cuda(GPU_ID)
color_model = c_model().cuda(GPU_ID)
pooling_model = p_model().cuda(GPU_ID)
extractor = FeatureExtractor(main_model, color_model, pooling_model)


def dump_dataset(loader, deep_feats, color_feats, labels):
    for batch_idx, (data, data_path) in enumerate(loader):
        data = Variable(data).cuda(GPU_ID)
        deep_feat, color_feat = extractor(data)
        for i in range(len(data_path)):
            path = data_path[i]
            feature_n = deep_feat[i].squeeze()
            color_feature_n = color_feat[i]
            # dump_feature(feature, path)

            deep_feats.append(feature_n)
            color_feats.append(color_feature_n)
            labels.append(path)

        if batch_idx % LOG_INTERVAL == 0:
            print("{} / {}".format(batch_idx * EXTRACT_BATCH_SIZE, len(loader.dataset)))


def dump():
    all_loader = torch.utils.data.DataLoader(
        Fashion_attr_prediction(type="all", transform=data_transform_test),
        batch_size=EXTRACT_BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=True
    )
    deep_feats = []
    color_feats = []
    labels = []
    dump_dataset(all_loader, deep_feats, color_feats, labels)

    if ENABLE_INSHOP_DATASET:
        inshop_loader = torch.utils.data.DataLoader(
            Fashion_inshop(type="all", transform=data_transform_test),
            batch_size=EXTRACT_BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=True
        )
        dump_dataset(inshop_loader, deep_feats, color_feats, labels)

    feat_all = os.path.join(DATASET_BASE, 'all_feat.npy')
    color_feat_all = os.path.join(DATASET_BASE, 'all_color_feat.npy')
    feat_list = os.path.join(DATASET_BASE, 'all_feat.list')
    with open(feat_list, "w") as fw:
        fw.write("\n".join(labels))
    np.save(feat_all, np.vstack(deep_feats))
    np.save(color_feat_all, np.vstack(color_feats))
    print("Dumped to all_feat.npy, all_color_feat.npy and all_feat.list.")


if __name__ == "__main__":
    dump()


