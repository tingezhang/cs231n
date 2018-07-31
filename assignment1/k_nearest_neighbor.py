#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import rpdb
import sys
import numpy as np


class KNearestNeighbor(object):
    "a KNN classifer with L2 distance"

    def __init__(self):
        pass

    def fit(self, X, Y):
        self.X_train = X
        self.Y_train = Y

    def predict(self, X, k=1):
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]

        dists = np.zeros((num_test, num_train))
        y_pred = np.zeros(num_test)

        for idx in range(num_test):
            dists[idx, :] = np.sqrt(np.sum(np.square(self.X_train - X[idx, :]), axis=1))

        #rpdb.set_trace()
        y_pred = np.zeros(num_test)

        index_1 = np.argsort(dists, axis=1)[:, -1]

        y_pred = Y_train[index_1]
        return y_pred

if __name__ == '__main__':
    from data_utils import load_CIFAR10

    test_count = 100

    if len(sys.argv) >= 2:
        cifar10_root = sys.argv[1]
    else:
        cifar10_root = '/Users/tinge/work/cs231n_working/cifar-10-batches-py'
    #Xtr, Ytr, Xte, Yte = load_CIFAR10('/data/work/cs231n/assignment1/cifar-10-batches-py')
    Xtr, Ytr, Xte, Yte = load_CIFAR10(cifar10_root)

    X_train, Y_train, X_test, Y_test = load_CIFAR10(cifar10_root)

    classifier = KNearestNeighbor()
    classifier.fit(X_train, Y_train)

    y_pred = classifier.predict(X_test[:test_count, :])
    print(y_pred)
    print(Y_test[:test_count])

    good_count = 0
    for idx in range(test_count):
        if y_pred[idx] == Y_test[idx]:
            good_count += 1

    print(good_count)
    print(good_count * 1.0 / test_count)

