# -*- coding: utf-8 -*-
# @Time    : 2018/4/28 11:57
# @Author  : Jiahao Yang
# @Email   : yangjh39@uw.edu

import numpy as np
import matplotlib.pyplot as plt

# -------------------------------- Problem 2 -------------------------------- #
def gradientDescent(gradFunc, stepSize, eps):
    w0 = 1
    w1 = [w0 - stepSize * gradFunc(w0)]
    iterNum = 0

    while abs(w1[-1]) > eps:
        w0 = w1[-1]
        w1.append(w0 - stepSize * gradFunc(w0))
        iterNum += 1

        if iterNum > 200:
            break

    return np.array(w1)

def adaptiveGD(gradFunc, c, eps):
    w0 = 1
    gradSquareSum = 0
    iterNum = 0

    grad = gradFunc(w0)
    gradSquareSum = gradSquareSum + grad ** 2
    w1 = [w0 - c/np.sqrt(gradSquareSum) * grad]

    while abs(w1[-1]) > eps:
        w0 = w1[-1]
        grad = gradFunc(w0)
        gradSquareSum = gradSquareSum + grad ** 2
        w1.append(w0 - c / np.sqrt(gradSquareSum) * grad)

        iterNum += 1

        if iterNum > 200:
            break

    return np.array(w1)

# 2.1 GD vs. Adaptive GD for smooth convex function f(w) = 1/2 w^2
def func1(w):
    """f(w) = 1/2 w^2"""
    return 0.5 * w ** 2.

def gradFunc1(w):
    """f^\prime(w) = w"""
    return w

# Calculate w_k and f(w_k)
wkGD1 = gradientDescent(gradFunc1, 3./4., 1e-7)
wkAdaptGD1 = adaptiveGD(gradFunc1,  3./4., 1e-7)

ykGD1 = func1(wkGD1)
ykAdaptGD1 = func1(wkAdaptGD1)

# Plot the learning curve
plt.figure(1)
gd1, = plt.plot(np.arange(len(ykGD1)), np.log(ykGD1))
agd1, = plt.plot(np.arange(len(ykAdaptGD1)), np.log(ykAdaptGD1))

plt.xlabel('iteration numbers')
plt.ylabel('log of loss function value')
plt.legend([gd1, agd1], ["Gradient Descent", "Adaptive Gradient Descent"])
# plt.title('Plot of loss function values v.s. step numbers')

# 2.2 GD vs. Adaptive GD for non-smooth convex function f(w) = |w|
def func2(w):
    return np.abs(w)

def gradFunc2(w):

    if w == 0:
        return None

    return 1 if w > 0 else -1

# Calculate w_k and f(w_k)
wkGD2 = gradientDescent(gradFunc2, 3./4., 1e-7)
wkAdaptGD2 = adaptiveGD(gradFunc2,  3./4., 1e-7)

ykGD2 = func2(wkGD2)
ykAdaptGD2 = func2(wkAdaptGD2)

# Plot the learning curve
plt.figure(2)
gd2, = plt.plot(np.arange(len(ykGD2)), ykGD2)
agd2, = plt.plot(np.arange(len(ykAdaptGD2)), ykAdaptGD2)

plt.xlabel('iteration numbers')
plt.ylabel('loss function value')
plt.legend([gd2, agd2], ["Gradient Descent", "Adaptive Gradient Descent"])
# plt.title('Plot of loss function values v.s. step numbers')



