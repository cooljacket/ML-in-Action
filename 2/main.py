# !/usr/bin/env	python2
# -*- coding: utf-8 -*-
__author__ = 'jacket'


# standard module
import sys

# self-define module
import KNN


def main(args):
	[train_x, train_y] = KNN.getDataSet()
	results = KNN.fit(train_x, train_y, train_x, 2)

	for (predict, y) in zip(results, train_y):
		print('Actual: {0} | Predict: {1}, with support {2}'.format(y, predict[0], predict[1]))


if __name__ == '__main__':
	exit(main(sys.argv[1:]))