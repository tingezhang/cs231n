#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import pickle
import numpy as np
import os

def load_CIFAR_batch(filename):
    """ load single cifar batch file """
    with open(filename, 'rb') as f_in:
        data_dict = pickle.load(f_in, encoding='latin1')
        X = data_dict['data']
        Y = data_dict['labels']
        X = X.astype('float')
        Y = np.array(Y)
        return X, Y

def load_CIFAR10(ROOT):
    """ load all of cifar """
    xs = []
    ys = []
    for b in range(1, 6):
        f_in_name = os.path.join(ROOT, 'data_batch_{}'.format(b))
        X, Y = load_CIFAR_batch(f_in_name)
        xs.append(X)
        ys.append(Y)
    Xtr = np.concatenate(xs)
    Ytr = np.concatenate(ys)

    Xte, Yte = load_CIFAR_batch(os.path.join(ROOT, 'test_batch'))
    return Xtr, Ytr, Xte, Yte


if __name__ == '__main__':
    print('do some test in data_utils')
    Xtr, Ytr, Xte, Yte = load_CIFAR10('/Users/tinge/work/cs231n_working/cifar-10-batches-py')
    print(Xtr, Ytr, Xte, Yte)
