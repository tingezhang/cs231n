#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os
import sys
import warnings
import numpy as np
from sklearn import neighbors
from data_utils import load_CIFAR10

warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")


if __name__ == '__main__':
    if len(sys.argv) >= 2:
        cifar10_root = sys.argv[1]
    else:
        cifar10_root = '/data/work/cs231n/assignment1/cifar-10-batches-py'

    Xtr, Ytr, Xte, Yte = load_CIFAR10(cifar10_root)

    cli = neighbors.KNeighborsClassifier(n_neighbors=3, n_jobs=7)

    cli.fit(Xtr, Ytr)

    Y_pred = cli.predict(Xte)

    cnt_total = len(Y_pred)
    cnt_match = 0

    Y_diff = Y_pred - Yte
    Y_cnt = [1 for a in Y_diff if a == 0]


    """
    3303
    10000
    3303
    10000
    0.3303

    real    11m0.131s
    user    74m13.070s
    sys     0m2.462s
    """
    cnt_match = np.sum(Y_cnt)
    print(cnt_match)
    print(cnt_total)

    cnt_match = 0
    for idx in range(cnt_total):
        if Y_pred[idx] == Yte[idx]:
            cnt_match += 1

    print(cnt_match)
    print(cnt_total)

    print(cnt_match / cnt_total)

