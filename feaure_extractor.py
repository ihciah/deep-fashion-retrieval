# -*- coding: utf-8 -*-

import os
from config import *
from utils import *
from torch.autograd import Variable
from data import Fashion_attr_prediction
from net import f_model, c_model, p_model


main_model = f_model(model_path=DUMPED_MODEL).cuda(GPU_ID)
color_model = c_model().cuda(GPU_ID)
pooling_model = p_model().cuda(GPU_ID)
extractor = FeatureExtractor(main_model, color_model, pooling_model)


all_loader = torch.utils.data.DataLoader(
    Fashion_attr_prediction(type="all", transform=data_transform_test),
    batch_size=EXTRACT_BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=True
)


def dump():
    deep_feats = None
    color_feats = None
    labels = []
    for batch_idx, (data, data_path) in enumerate(all_loader):
        data = Variable(data).cuda(GPU_ID)
        deep_feat, color_feat = extractor(data)
        deep_feat = deep_feat.cpu().data.numpy()
        for i in range(len(data_path)):
            path = data_path[i]
            feature_n = deep_feat[i].squeeze()
            color_feature_n = color_feat[i]
            # dump_feature(feature, path)

            if deep_feats is None:
                deep_feats = feature_n
                color_feats = color_feature_n
            else:
                deep_feats = np.vstack([deep_feats, feature_n])
                color_feats = np.vstack([color_feats, color_feature_n])
            labels.append(path)

        if batch_idx % LOG_INTERVAL == 0:
            print("{} / {}".format(batch_idx * EXTRACT_BATCH_SIZE, len(all_loader.dataset)))

    feat_all = os.path.join(DATASET_BASE, 'all_feat.npy')
    color_feat_all = os.path.join(DATASET_BASE, 'all_color_feat.npy')
    feat_list = os.path.join(DATASET_BASE, 'all_feat.list')
    with open(feat_list, "w") as fw:
        fw.write("\n".join(labels))
    np.save(feat_all, deep_feats)
    np.save(color_feat_all, color_feats)
    print("Dumped to all_feat.npy, all_color_feat.npy and all_feat.list.")


if __name__ == "__main__":
    dump()


