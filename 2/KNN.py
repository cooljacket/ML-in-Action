# !/usr/bin/env	python2
# -*- coding: utf-8 -*-
__author__ = 'jacket'


# standard module
import sys

# third-party module
import numpy as np


def getDataSet():
	X = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
	Y = ['A', 'A', 'B', 'B']
	return (X, Y)


def insertSort(array, data, len_upper_bound=None):
	"""
	insertion sort hold on a complexity: O(NK), 
	N is the total data size, while K is the len_upper_bound
	"""
	tail = len(array) - 1
	pos = tail
	key = data[0]

	if (not len_upper_bound) or (len(array) < len_upper_bound):
		array.append(data)
		tail += 1

	while pos >= 0 and key < array[pos][0]:
		if pos < tail:
			array[pos+1] = array[pos]
		pos -= 1

	if pos < tail:
			array[pos+1] = data


def classifyOne(train_x, train_y, test_x, K):
	# first calculate the distance between test_x with each train_x
	# Euclid distance = sqrt(sum([(x1-x2)**2 for (x1, x2) in zip(X1, X2)])) / 2
	# use numpy's ufunc to speed up the calculation!
	distances = np.sqrt(np.sum((train_x - test_x)**2, axis=1)) / 2
	k_shortest = []

	# use insertion sort to select the k_shortest distances
	for i in range(distances.shape[0]):
		insertSort(k_shortest, (distances[i], train_y[i]), K)

	# vote for each label corresponding to the k_shortest distances
	votes = dict.fromkeys(train_y, 0)
	for (_, y) in k_shortest:
		votes[y] += 1

	# select the most frequent label to be the predict category
	[most_label, max_votes] = [None, 0]
	for label in votes:
		if votes[label] > max_votes:
			max_votes = votes[label]
			most_label = label

	return (most_label, max_votes)


def fit(train_x, train_y, test_x, K):
	results = []
	for x in test_x:
		results.append(classifyOne(train_x, train_y, x, K))

	return results