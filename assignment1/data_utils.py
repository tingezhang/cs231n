#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os
import sys
import pickle
import numpy as np

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


    if len(sys.argv) >= 2:
        cifar10_root = sys.argv[1]
    else:
        cifar10_root = '/Users/tinge/work/cs231n_working/cifar-10-batches-py'
    #Xtr, Ytr, Xte, Yte = load_CIFAR10('/data/work/cs231n/assignment1/cifar-10-batches-py')
    Xtr, Ytr, Xte, Yte = load_CIFAR10(cifar10_root)
    print(Xtr, Ytr, Xte, Yte)
