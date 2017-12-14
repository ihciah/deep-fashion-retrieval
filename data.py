# -*- coding: utf-8 -*-

import torch.utils.data as data
import torch
from config import *
import os
from PIL import Image
import random


class Fashion_attr_prediction(data.Dataset):
    def __init__(self, type="train", transform=None, target_transform=None, crop=False, img_path=None):
        self.transform = transform
        self.target_transform = target_transform
        self.crop = crop
        # type_all = ["train", "test", "all", "triplet", "single"]
        self.type = type
        if type == "single":
            self.img_path = img_path
            return
        self.train_list = []
        self.train_dict = {i: [] for i in range(CATEGORIES)}
        self.test_list = []
        self.all_list = []
        self.bbox = dict()
        self.anno = dict()
        self.read_partition_category()
        self.read_bbox()

    def __len__(self):
        if self.type == "all":
            return len(self.all_list)
        elif self.type == "train":
            return len(self.train_list)
        elif self.type == "test":
            return len(self.test_list)
        else:
            return 1

    def read_partition_category(self):
        list_eval_partition = os.path.join(DATASET_BASE, r'Eval', r'list_eval_partition.txt')
        list_category_img = os.path.join(DATASET_BASE, r'Anno', r'list_category_img.txt')
        partition_pairs = self.read_lines(list_eval_partition)
        category_img_pairs = self.read_lines(list_category_img)
        for k, v in category_img_pairs:
            v = int(v)
            if v <= 20:
                self.anno[k] = v - 1
        for k, v in partition_pairs:
            if k in self.anno:
                if v == "train":
                    self.train_list.append(k)
                    self.train_dict[self.anno[k]].append(k)
                else:
                    # Test and Val
                    self.test_list.append(k)
        self.all_list = self.test_list + self.train_list
        random.shuffle(self.train_list)
        random.shuffle(self.test_list)
        random.shuffle(self.all_list)

    def read_bbox(self):
        list_bbox = os.path.join(DATASET_BASE, r'Anno', r'list_bbox.txt')
        pairs = self.read_lines(list_bbox)
        for k, x1, y1, x2, y2 in pairs:
            self.bbox[k] = [x1, y1, x2, y2]

    def read_lines(self, path):
        with open(path) as fin:
            lines = fin.readlines()[2:]
            lines = list(filter(lambda x: len(x) > 0, lines))
            pairs = list(map(lambda x: x.strip().split(), lines))
        return pairs

    def read_crop(self, img_path):
        img_full_path = os.path.join(DATASET_BASE, img_path)
        with open(img_full_path, 'rb') as f:
            with Image.open(f) as img:
                img = img.convert('RGB')
        if self.crop:
            x1, y1, x2, y2 = self.bbox[img_path]
            if x1 < x2 <= img.size[0] and y1 < y2 <= img.size[1]:
                img = img.crop((x1, y1, x2, y2))
        return img

    def __getitem__(self, index):
        if self.type == "triplet":
            img_path = self.train_list[index]
            target = self.anno[img_path]
            img_p = random.choice(self.train_dict[target])
            img_n = random.choice(self.train_dict[random.choice(list(filter(lambda x: x != target, range(20))))])
            img = self.read_crop(img_path)
            img_p = self.read_crop(img_p)
            img_n = self.read_crop(img_n)
            if self.transform is not None:
                img = self.transform(img)
                img_p = self.transform(img_p)
                img_n = self.transform(img_n)
            return img, img_p, img_n

        if self.type == "single":
            img_path = self.img_path
            img = self.read_crop(img_path)
            if self.transform is not None:
                img = self.transform(img)
            return img

        if self.type == "all":
            img_path = self.all_list[index]
        elif self.type == "train":
            img_path = self.train_list[index]
        else:
            img_path = self.test_list[index]
        target = self.anno[img_path]
        img = self.read_crop(img_path)

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, img_path if self.type == "all" else target


class Fashion_inshop(data.Dataset):
    def __init__(self, type="train", transform=None):
        self.transform = transform
        self.type = type
        self.train_dict = {}
        self.test_dict = {}
        self.train_list = []
        self.test_list = []
        self.all_path = []
        self.cloth = self.readcloth()
        self.read_train_test()

    def read_lines(self, path):
        with open(path) as fin:
            lines = fin.readlines()[2:]
            lines = list(filter(lambda x: len(x) > 0, lines))
            pairs = list(map(lambda x: x.strip().split(), lines))
        return pairs

    def readcloth(self):
        lines = self.read_lines(os.path.join(DATASET_BASE, 'in_shop', 'list_bbox_inshop.txt'))
        valid_lines = list(filter(lambda x: x[1] == '1', lines))
        names = set(list(map(lambda x: x[0], valid_lines)))
        return names

    def read_train_test(self):
        lines = self.read_lines(os.path.join(DATASET_BASE, 'in_shop', 'list_eval_partition.txt'))
        valid_lines = list(filter(lambda x: x[0] in self.cloth, lines))
        for line in valid_lines:
            s = self.train_dict if line[2] == 'train' else self.test_dict
            if line[1] not in s:
                s[line[1]] = [line[0]]
            else:
                s[line[1]].append(line[0])

        def clear_single(d):
            keys_to_delete = []
            for k, v in d.items():
                if len(v) < 2:
                    keys_to_delete.append(k)
            for k in keys_to_delete:
                d.pop(k, None)
        clear_single(self.train_dict)
        clear_single(self.test_dict)
        self.train_list, self.test_list = list(self.train_dict.keys()), list(self.test_dict.keys())
        for v in list(self.train_dict.values()):
            self.all_path += v
        self.train_len = len(self.all_path)
        for v in list(self.test_dict.values()):
            self.all_path += v
        self.test_len = len(self.all_path) - self.train_len

    def process_img(self, img_path):
        img_full_path = os.path.join(DATASET_BASE, 'in_shop', img_path)
        with open(img_full_path, 'rb') as f:
            with Image.open(f) as img:
                img = img.convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img

    def __len__(self):
        if self.type == 'train':
            return len(self.train_list)
        elif self.type == 'test':
            return len(self.test_list)
        else:
            return len(self.all_path)

    def __getitem__(self, item):
        if self.type == 'all':
            img_path = self.all_path[item]
            img = self.process_img(img_path)
            return img, img_path
        s_d = self.train_dict if self.type == 'train' else self.test_dict
        s_l = self.train_list if self.type == 'train' else self.test_list
        imgs = s_d[s_l[item]]
        img_triplet = random.sample(imgs, 2)
        img_other_id = random.choice(list(range(0, item)) + list(range(item + 1, len(s_l))))
        img_other = random.choice(s_d[s_l[img_other_id]])
        img_triplet.append(img_other)
        return list(map(self.process_img, img_triplet))
