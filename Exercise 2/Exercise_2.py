#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sept 18 16:15:33 2020

@author: umarzubair
"""

import pickle
import numpy as np
import time
from skimage import exposure
from scipy.spatial.distance import cdist


def unpickle(file):
    with open(file, 'rb') as f:
        dict = pickle.load(f, encoding="latin1")
    return dict


def zca_whitening(img_data):
    img_data = img_data - img_data.mean(axis=0)
    img_data = img_data / np.sqrt((img_data ** 2).sum(axis=1))[:, None]

    # compute the covariance of the image data
    cov = np.cov(img_data, rowvar=True)
    U, S, V = np.linalg.svd(cov)
    # build the ZCA matrix
    epsilon = 1e-5
    zca_matrix = np.dot(U, np.dot(np.diag(1.0 / np.sqrt(S + epsilon)), U.T))
    zca_matrix = np.dot(zca_matrix, img_data)  # zca is (N, 3072)
    return zca_matrix


def histogram_equalization(data):
    # Histogram equalization for image enhancement
    # Motivation: http://www.sci.utah.edu/~acoste/uou/Image/project1/Arthur_COSTE_Project_1_report.html#:~:text=number%20of%20pixel.-,B%3A%20Equalization%20of%20a%20Histogram,function%20associated%20to%20the%20image.
    return exposure.equalize_hist(data)


def distance(x):
    return cdist(x.reshape(1, -1), x_train, 'correlation')


def class_acc(pred, gt):
    accuracy = (np.sum(pred == gt) / len(pred)) * 100
    return round(accuracy, 2)


def cifar10_classifier_random(x):
    return np.random.randint(1, 10, len(x))


trainData1 = unpickle('cifar-10-batches-py/data_batch_1')
trainData2 = unpickle('cifar-10-batches-py/data_batch_2')
trainData3 = unpickle('cifar-10-batches-py/data_batch_3')
trainData4 = unpickle('cifar-10-batches-py/data_batch_4')
trainData5 = unpickle('cifar-10-batches-py/data_batch_5')

datadict = unpickle('cifar-10-batches-py/test_batch')

x_train = np.concatenate((zca_whitening(trainData1["data"]), zca_whitening(trainData2["data"]),
                          zca_whitening(trainData3["data"]), zca_whitening(trainData4["data"]),
                          zca_whitening(trainData5["data"])))
y_train = np.concatenate(
    (trainData1["labels"], trainData2["labels"], trainData3["labels"], trainData4["labels"], trainData5["labels"]))

x_test = zca_whitening(datadict["data"])
y_test = np.array(datadict["labels"])

# Applying histogram equalization
x_train = histogram_equalization(x_train)
x_test = histogram_equalization(x_test)

# Random classifier
time_initial_random = time.time()
randomLabels = cifar10_classifier_random(x_test)
randomClassifierAccuracy = class_acc(randomLabels, y_test)
timeFinal = (time.time() - time_initial_random) * 1000
print("Random classifier accuracy: ", round(randomClassifierAccuracy, 2), " %")
print("Time taken for random classifier: ", round(timeFinal, 3), " ms")

# KNN Implementation
print('Size of Training data: ', len(x_train))
print('Size of Testing data: ', len(x_test))

time_initial = time.time()
k = 45  # Value of k for knn
PredictedValue = np.array([])

for sample in x_test:
    idx = np.argpartition(distance(sample).reshape(-1), k)
    idx = idx[:k]
    PredictedValue = np.append(PredictedValue, np.bincount(y_train[idx].flatten()).argmax())

time_final = (time.time() - time_initial)
print("Time taken for ", k, "NN classifier: ", time_final)
print("Accuracy is ", class_acc(np.array(PredictedValue), y_test.flatten()), ' %')
