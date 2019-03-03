#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    Time    : 下午3:49
    Author  : wangjf
    GitHub  : https://github.com/wjf0627
"""
import numpy as np


def rmse(predictions, targets):
	# 真实值与目标值的误差
	differences = predictions - targets
	differences_squared = differences ** 2
	mean_of_differences_squared = differences_squared.mean()
	# 取平方根
	rms_val = np.sqrt(mean_of_differences_squared)
	return rms_val


y_hat = np.array([0.000, 0.166, 0.333])
y_true = np.array([0.000, 0.254, 0.998])


# print("d is: " + str(["%.8f" % elem for elem in y_hat]))
# print("p is: " + str(["%.8f" % elem for elem in y_true]))
# rmse_val = rmse(y_hat, y_true)
# print("rms error is: " + str(rmse_val))


def mae(predictions, targets):
	differences = predictions - targets
	absolute_differences = np.absolute(differences)
	mean_absolute_differences = absolute_differences.mean()
	return mean_absolute_differences


# mae_val = mae(y_hat, y_true)
# print("mae error is: " + str(mae_val))


def mbe(predictions, targets):
	differences = predictions - targets
	mean_absolute_differences = differences.mean()
	return mean_absolute_differences


# mbe_val = mbe(y_hat, y_true)
# print("mbe error is: " + str(mbe_val))


def hinge_loss(predictions, label):
	"""
    hinge_loss = max(0, s_j - s_yi +1)
    :param predictions:
    :param label:
    :return:
    """
	result = 0.0
	pred_value = predictions[label]
	for i, val in enumerate(predictions):
		if i == label:
			continue
		tmp = val - pred_value + 1
		result += max(0, tmp)
	return result


image1 = np.array([-0.39, 1.49, 4.21])
image2 = np.array([-4.61, 3.28, 1.46])
image3 = np.array([1.03, -2.37, -2.27])
result1 = hinge_loss(image1, 0)
result2 = hinge_loss(image2, 1)
result3 = hinge_loss(image3, 2)
print('image1,hinge loss={}'.format(result1))
print('image2,hinge loss={}'.format(result2))
print('image3,hinge loss={}'.format(result3))
