#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import random
import numpy as np
from data_utils import load_CIFAR10



cifar10_dir = '/Users/tinge/work/cs231n_working/cifar-10-batches-py'


X_train, Y_train, X_test, Y_test = load_CIFAR10(cifar10_dir)

print('Training data shape: {}'.format(X_train.shape))
print('Training labels shape: {}'.format(Y_train.shape))
print('Test data shape: {}'.format(X_test.shape))
print('Test labels shape: {}'.format(Y_test.shape))


classes = ['plane', 'car', 'bird', 'cat', 'dog', 'frog', 'horse', 'ship', 'truck']
num_classes = len(classes)





