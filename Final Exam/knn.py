#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 14 16:31:13 2020

@author: umarzubair
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

# Loading data
X_train = np.loadtxt('X_train.txt')
X_test = np.loadtxt('X_test.txt')
y_train = np.loadtxt('Y_train.txt')
y_test = np.loadtxt('Y_test.txt')

print('Shape of Training data: ', np.shape(X_train))
print('Shape of Testing data: ', np.shape(X_test))
y_test = y_test.astype('int32')
y_train = y_train.astype('int32')


def calculate_distance(x):
    return cdist(x.reshape(1, -1), X_train, 'chebyshev')


def class_acc(pred, gt):
    accuracy = (np.sum(pred == gt) / len(pred)) * 100
    return round(accuracy, 2)


def k_nearest_neighbor(k):
    predicted_values = np.array([])
    # Getting the index of smallest distance by matching each point of X_test with every point in X_test
    for sample in X_test:
        # Getting the indexes of last k smallest values
        idx = np.argpartition(calculate_distance(sample).reshape(-1), k)
        idx = idx[:k]
        # Getting the labels for those indexes
        predicted_values = np.append(predicted_values, np.bincount(y_train[idx].flatten()).argmax())
    return class_acc(np.array(predicted_values), y_test.flatten())


k_values = [1, 2, 3, 5, 10, 20]
accuracy_values = []

for k in k_values:
    accuracy_values.append(k_nearest_neighbor(k))

print('K values', k_values)
print('Accuracy per k values:', accuracy_values)

plt.plot(k_values, accuracy_values, marker='o', markerfacecolor='blue', markersize=12, color='skyblue', linewidth=4,
         label="Nearest Neighbor Classification")
plt.legend()
plt.grid(True)
plt.axis([0, 22, 60, 100])

my_xticks = ['1', '2', '3', '5', '10', '20']
plt.xticks(k_values, my_xticks)

for k, accuracy in zip(k_values, accuracy_values):
    if k == 2:
        plt.annotate(str(accuracy), xy=(k - .3, accuracy - 2))
    else:
        plt.annotate(str(accuracy), xy=(k - .5, accuracy + 1.5))

plt.xlabel('Value of K')
plt.ylabel('Accuracy (%)')
plt.gcf().set_size_inches(10, 5)
plt.savefig('Zubair_050519664_knn.png', dpi=300)
plt.show()
