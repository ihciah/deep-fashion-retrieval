# -*- coding: utf-8 -*-

import os
from config import *
from utils import *
from torch.autograd import Variable
from data import Fashion
from net import f_model


model = f_model(model_path=DUMPED_MODEL).cuda(GPU_ID)
model.eval()
extractor = FeatureExtractor(model)


all_loader = torch.utils.data.DataLoader(
    Fashion(type="all", transform=data_transform_test),
    batch_size=TEST_BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=True
)


def dump():
    feats = None
    labels = []
    for batch_idx, (data, data_path) in enumerate(all_loader):
        data = Variable(data).cuda(GPU_ID)
        result = extractor(data)
        result = result.cpu().data.numpy()
        for i in range(len(data_path)):
            path = data_path[i]
            feature = result[i].squeeze()
            # dump_feature(feature, path)

            if feats is None:
                feats = feature
            else:
                feats = np.vstack([feats, feature])
            labels.append(path)

        if batch_idx % LOG_INTERVAL == 0:
            print("{} / {}".format(batch_idx * TEST_BATCH_SIZE, len(all_loader.dataset)))

    feat_all = os.path.join(DATASET_BASE, 'all_feat.npy')
    feat_list = os.path.join(DATASET_BASE, 'all_feat.list')
    with open(feat_list, "w") as fw:
        fw.write("\n".join(labels))
    np.save(feat_all, feats)
    print("Dumped to all_feat.npy and all_feat.list.")
    print("Dumped to feature files.")


if __name__ == "__main__":
    dump()


