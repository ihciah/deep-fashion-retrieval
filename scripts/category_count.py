# -*- coding: utf-8 -*-

from config import *
import os

list_category_img = os.path.join(DATASET_BASE, r'Anno', r'list_category_img.txt')
output = os.path.join(DATASET_BASE, r'Anno', r'analytic.txt')
with open(list_category_img) as fin:
    lines = fin.readlines()[2:]
    lines = list(filter(lambda x: len(x) > 0, lines))
    numbers = list(map(lambda x: int(x.strip().split()[1]), lines))

with open(output, "w") as fw:
    for i in range(1, 51):
        fw.write("%d %d\n" % (i, numbers.count(i)))
