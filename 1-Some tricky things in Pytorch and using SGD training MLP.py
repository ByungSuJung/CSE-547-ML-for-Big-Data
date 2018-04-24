#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 14 17:07:14 2018

@author: yangjh39
"""

import torch
from torch.autograd import Variable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --------------------------------------- Problem 4 -------------------------------------- #
def func1(w):
    """
    Piecewise Function: when w>1, f(w) = 2w+5 when w=1, f(w) = 6, else f(w)=5
    """
    return ((w > 1.).type(torch.FloatTensor)*2.*w
            + (w == 1.).type(torch.FloatTensor)*6.
            + (w != 1.).type(torch.FloatTensor)*5.)

# 4.1: Draw this piecewise function func1
w = torch.arange(-2, 2, 0.01)
fw = func1(w)

plt.scatter(w, fw)
plt.xlabel("w")
plt.ylabel("f(w)")

# 4.2: When our input w=1, the PyTorch returns a derivative of 0 at this point
w = Variable(torch.FloatTensor([1]), requires_grad=True)
y = func1(w) 
y.backward()
w.grad

# 4.3: Define the same function in another way which returns a derivatve of 2 at w=1 point
def func2(w):
    """
    Piecewise Function: when w>1, f(w) = 2w+5, when w=1, f(w) = 2w+4, else f(w)=5
    """
    return ((w >= 1.).type(torch.FloatTensor)*2.*w
            + (w == 1.).type(torch.FloatTensor)*4.
            + (w != 1.).type(torch.FloatTensor)*5.)

w = Variable(torch.FloatTensor([1]), requires_grad=True)
y = func2(w) 
y.backward()
w.grad

def func3(w):
    """
    Differentiable function with different derivative at the same point
    Function: f(w) = w
    """
    return w

w = torch.arange(-2, 2, 0.01)
fw = func3(w)

plt.scatter(w, fw)
plt.xlabel("w")
plt.ylabel("f(w)")

w = Variable(torch.FloatTensor([0]), requires_grad=True)
y = func3(w)
y.backward()
w.grad

def func4(w):
    """
    Differentiable function with different derivative at the same point
    Function: when w!= 0, f(w) = w, when w=0, f(w) =0
    """
    return (w == 0.).type(torch.FloatTensor)*0. \
           + (w != 0.).type(torch.FloatTensor)*w

w = Variable(torch.FloatTensor([0]), requires_grad=True)
y = func4(w)
y.backward()
w.grad

def func5(w):
    """
    Absolute value function which is not differentiable at point w=0
    """
    return torch.abs(w)

w = Variable(torch.FloatTensor([0]), requires_grad=True)
y = func5(w)
y.backward()
w.grad

# --------------------------------------- Problem 6 -------------------------------------- #
# 6.1: SGD and Linear Regression

def objective_func1(w, lambd, batch_x, batch_y):
    """
    Objective function: L(w) = lambda/2*||w||^2 + 1/n sum(1/2 * (y_i - <w, x_i>)^2)
    The data type of all these input is torch.autograd.Variable so that we can use autograd in Pytorch
    """
    if len(batch_x.size()) == 1:
        return lambd / 2 * w.norm(2) ** 2 + 1. / len(batch_y) * \
                                            sum(1. / 2. * (batch_y - batch_x.dot(w)) ** 2)
    else:
        return lambd/2*w.norm(2)**2 + 1./len(batch_y)*sum(1./2.*(batch_y - batch_x.mv(w))**2)


def batch_iter(y, tx, batch_size, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    """
    data_size = len(y)
    num_batches = np.ceil(data_size/batch_size).astype(int)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx

    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)

        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]


def compute_gradient(w, lambd, batch_x, batch_y):
    """
    Compute a gradient for batch data using autograd in Pytorch
    """
    grad = objective_func1(w, lambd, batch_x, batch_y)
    grad.backward()

    return w.grad


def stochastic_gradient_descent(y, tx, cvy, cvx, testy, testx, batch_size, gamma, max_iter_num, mode, lambd=1):
    """
    Stochastic gradient descent algorithm.
    """
    initial_w = np.ones(tx.shape[1])

    ws = Variable(torch.FloatTensor(initial_w), requires_grad=True)
    lambd = Variable(torch.FloatTensor([lambd]), requires_grad=True)

    n_iter = 0
    train_loss = []
    train_misclassification_err = []

    cv_loss = []
    cv_misclassification_err = []

    if mode == "test":
        test_loss = []
        test_misclassification_err = []

    while n_iter < max_iter_num:
        batch_iter_instant = batch_iter(y, tx, batch_size)

        for minibatch_y, minibatch_x in batch_iter_instant:
            minibatch_y = Variable(torch.FloatTensor(minibatch_y), requires_grad=True)
            minibatch_x = Variable(torch.FloatTensor(minibatch_x[0]), requires_grad=True)

            w = ws - gamma * compute_gradient(ws, lambd, minibatch_x, minibatch_y)
            ws = Variable(torch.FloatTensor(w.data), requires_grad=True)
            n_iter += 1

            if n_iter % 500 == 0:
                train_x = Variable(torch.FloatTensor(tx), requires_grad=True)
                train_y = Variable(torch.FloatTensor(y), requires_grad=True)
                train_loss.append(objective_func1(w, lambd, train_x, train_y).data.numpy()[0])

                train_misclassification_err.append(np.mean(y != np.sign((train_x.mv(w)).data.numpy())))

                cv_x = Variable(torch.FloatTensor(cvx), requires_grad=True)
                cv_y = Variable(torch.FloatTensor(cvy), requires_grad=True)
                cv_loss.append(objective_func1(w, lambd, cv_x, cv_y).data.numpy()[0])

                cv_misclassification_err.append(np.mean(cvy != np.sign((cv_x.mv(w)).data.numpy())))

                if mode == "test":
                    test_x = Variable(torch.FloatTensor(testx), requires_grad=True)
                    test_y = Variable(torch.FloatTensor(testy), requires_grad=True)
                    test_loss.append(objective_func1(w, lambd, test_x, test_y).data.numpy()[0])

                    test_misclassification_err.append(np.mean(testy != np.sign((test_x.mv(w)).data.numpy())))

                    print("Step " + str(n_iter) + ": the test loss is " + str(test_loss[-1])
                          + ", the test misclassification error is " + str(test_misclassification_err[-1]))

                print("Step " + str(n_iter) + ": the train loss is " + str(train_loss[-1])
                      + ", the train misclassification error is " + str(train_misclassification_err[-1]))

                print("Step " + str(n_iter) + ": the cv loss is " + str(cv_loss[-1])
                      + ", the cv misclassification error is " + str(cv_misclassification_err[-1]))

        #     if abs(train_loss[-1] - train_loss[-2]) < 0.005:
        #         break
        #
        # if abs(train_loss[-1] - train_loss[-2]) < 0.005:
        #     break

    if mode == "test":
        return train_loss, train_misclassification_err, cv_loss, cv_misclassification_err, test_loss, test_misclassification_err
    else:
        return cv_loss, cv_misclassification_err


def cross_validation(y, tx, cvy, cvx, testy, testx, batch_size, gamma, max_iter_num, lambd):
    """
    Cross-validation to find the best learning rate gamma and best regularization parameter lambd
    """
    cv_misclassification_err = 1
    best_gamma = 0
    best_lambd = 0
    cv_loss = 0

    for i in np.arange(len(gamma)):
        for j in np.arange(len(lambd)):
            cl, cm = stochastic_gradient_descent(y, tx, cvy, cvx, testy, testx, batch_size,
                                                 gamma[i], max_iter_num, "cv", lambd[j])
            if min(cm) < cv_misclassification_err:
                best_gamma = gamma[i]
                best_lambd = lambd[j]
                cv_loss = min(cl)
                cv_misclassification_err = min(cm)

    return best_gamma, best_lambd, cv_loss, cv_misclassification_err


# 6.2 Implement a MLP
def objective_func_mlp(w1, w2, batch_x, batch_y):
    """
    Objective function: L(w) = 1/n*sum(1/2 * (y_i - <w_2, relu(w_1^T.dot(x_i))>)^2)
    The data type of all these input is torch.autograd.Variable so that we can use autograd in Pytorch
    """
    if len(batch_x.size()) == 1:
        return 1. / len(batch_y) * sum(1. / 2. * (batch_y - w2.dot((w1.transpose(0, 1).mv(batch_x) > 0).type(torch.FloatTensor) * (w1.transpose(0, 1).mv(batch_x)))) ** 2)
    else:
        return 1. / len(batch_y) * sum(1. / 2. * (batch_y - ((batch_x.mm(w1) > 0).type(torch.FloatTensor) * (batch_x.mm(w1))).mv(w2)) ** 2)


def stochastic_gradient_descent_mlp(y, tx, cvy, cvx, testy, testx, batch_size, gamma, max_iter_num, mode, hid_node_num):
    """
    Stochastic gradient descent algorithm for MLP.
    """
    w1 = torch.randn([tx.shape[1], hid_node_num])/20.
    w2 = torch.randn([hid_node_num])/20.

    w1 = Variable(w1, requires_grad=True)
    w2 = Variable(w2, requires_grad=True)

    n_iter = 0
    train_loss = []
    train_misclassification_err = []

    cv_loss = []
    cv_misclassification_err = []

    if mode == "test":
        test_loss = []
        test_misclassification_err = []

    while n_iter < max_iter_num:
        batch_iter_instant = batch_iter(y, tx, batch_size)

        for minibatch_y, minibatch_x in batch_iter_instant:
            minibatch_y = Variable(torch.FloatTensor(minibatch_y), requires_grad=True)
            minibatch_x = Variable(torch.FloatTensor(minibatch_x[0]), requires_grad=True)

            temp_y = objective_func_mlp(w1, w2, minibatch_x, minibatch_y)
            temp_y.backward()

            ws1 = w1 - gamma * w1.grad
            ws2 = w2 - gamma * w2.grad
            w1 = Variable(torch.FloatTensor(ws1.data), requires_grad=True)
            w2 = Variable(torch.FloatTensor(ws2.data), requires_grad=True)
            n_iter += 1

            if n_iter % 500 == 0:
                train_x = Variable(torch.FloatTensor(tx), requires_grad=True)
                train_y = Variable(torch.FloatTensor(y), requires_grad=True)
                train_loss.append(objective_func_mlp(w1, w2, train_x, train_y).data.numpy()[0])

                train_misclassification_err.append(np.mean(y != np.sign((((train_x.mm(w1) > 0).type(torch.FloatTensor) * (train_x.mm(w1))).mv(w2)).data.numpy())))

                cv_x = Variable(torch.FloatTensor(cvx), requires_grad=True)
                cv_y = Variable(torch.FloatTensor(cvy), requires_grad=True)
                cv_loss.append(objective_func_mlp(w1, w2, cv_x, cv_y).data.numpy()[0])

                cv_misclassification_err.append(np.mean(cvy != np.sign((((cv_x.mm(w1) > 0).type(torch.FloatTensor) * (cv_x.mm(w1))).mv(w2)).data.numpy())))

                if mode == "test":
                    test_x = Variable(torch.FloatTensor(testx), requires_grad=True)
                    test_y = Variable(torch.FloatTensor(testy), requires_grad=True)
                    test_loss.append(objective_func_mlp(w1, w2, test_x, test_y).data.numpy()[0])

                    test_misclassification_err.append(np.mean(testy != np.sign((((test_x.mm(w1) > 0).type(torch.FloatTensor) * (test_x.mm(w1))).mv(w2)).data.numpy())))

                    print("Step " + str(n_iter) + ": the test loss is " + str(test_loss[-1])
                          + ", the test misclassification error is " + str(test_misclassification_err[-1]))

                print("Step " + str(n_iter) + ": the train loss is " + str(train_loss[-1])
                      + ", the train misclassification error is " + str(train_misclassification_err[-1]))

                print("Step " + str(n_iter) + ": the cv loss is " + str(cv_loss[-1])
                      + ", the cv misclassification error is " + str(cv_misclassification_err[-1]))

        #     if abs(train_loss[-1] - train_loss[-2]) < 0.005:
        #         break
        #
        # if abs(train_loss[-1] - train_loss[-2]) < 0.005:
        #     break

    if mode == "test":
        return train_loss, train_misclassification_err, cv_loss, cv_misclassification_err, test_loss, test_misclassification_err
    else:
        return cv_loss, cv_misclassification_err


if __name__ == '__main__':
    path = "/home/yangjh39/548/data"

    img_supercat_train = pd.read_csv(path+"/features_small/img_supercat_train2014.csv").iloc[:, 1::]
    img_supercat_cv = pd.read_csv(path+"/features_small/img_supercat_val2014.csv").iloc[:, 1::]
    img_supercat_test = pd.read_csv(path+"/features_small/img_supercat_test2014.csv").iloc[:, 1::]

    features_train = pd.read_csv(path+"/features_small/feats_train2014.csv").iloc[:, 1::]
    features_cv = pd.read_csv(path+"/features_small/feats_val2014.csv").iloc[:, 1::]
    features_test = pd.read_csv(path+"/features_small/feats_test2014.csv").iloc[:, 1::]

    # Transform labels to -1(animal),1(vehicle)
    img_supercat_train = -1 + 2 * np.array((img_supercat_train == "vehicle").astype(int).iloc[:, 0])
    features_train = np.array(features_train)
    features_train = features_train/np.max(features_train)

    img_supercat_cv = -1 + 2 * np.array((img_supercat_cv == "vehicle").astype(int).iloc[:, 0])
    features_cv = np.array(features_cv)
    features_cv = features_cv/np.max(features_cv)

    img_supercat_test = -1 + 2 * np.array((img_supercat_test == "vehicle").astype(int).iloc[:, 0])
    features_test = np.array(features_test)
    features_test = features_test/np.max(features_test)

    # All possible gamma and lambda
    # gamma = np.arange(0.00001, 0.01, 0.002).tolist()
    # lambd = np.arange(6, 15., 2.).tolist()
    #
    # bgamma, blambd, cv_l, cv_m = cross_validation(img_supercat_train, features_train, img_supercat_cv, features_cv,
    #                                           img_supercat_test, features_test, 1, gamma, 80000, lambd)

    train_loss, train_misclassification_err, cv_loss, cv_misclassification_err, test_loss, test_misclassification_err \
        = stochastic_gradient_descent(img_supercat_train, features_train, img_supercat_cv, features_cv,
                                      img_supercat_test, features_test, 1, 1e-4, 10000, "test", 8)

    train_loss_mlp, train_misclassification_err_mlp, cv_loss_mlp, \
    cv_misclassification_err_mlp, test_loss_mlp, test_misclassification_err_mlp \
        = stochastic_gradient_descent_mlp(img_supercat_train, features_train, img_supercat_cv, features_cv,
                                      img_supercat_test, features_test, 1, 1e-3, 10000, "test", 10)

    step_num = np.arange(0, 10000, 500)
    plt.figure(1)
    train_loss_plt, = plt.plot(step_num, train_loss)
    cv_loss_plt, = plt.plot(step_num, cv_loss)
    test_loss_plt, = plt.plot(step_num, test_loss)

    plt.xlabel('step numbers')
    plt.ylabel('loss function value')
    plt.xlim([2500, 10000])
    plt.ylim([0.1, 1])
    plt.legend([train_loss_plt, cv_loss_plt, test_loss_plt], ["train loss", "cross-validation loss", "test loss"])
    # plt.title('Plot of loss function values v.s. step numbers')

    plt.figure(2)
    train_miserr_plt, = plt.plot(step_num, train_misclassification_err)
    cv_miserr_plt, = plt.plot(step_num, cv_misclassification_err)
    test_miserr_plt, = plt.plot(step_num, test_misclassification_err)

    plt.xlabel('step numbers')
    plt.ylabel('misclassification error')
    plt.legend([train_miserr_plt, cv_miserr_plt, test_miserr_plt], ["train error", "cross-validation error", "test error"])
    # plt.title('Plot of misclassification errors v.s. step numbers')

    plt.figure(3)
    train_loss_plt1, = plt.plot(step_num, train_loss_mlp)
    cv_loss_plt1, = plt.plot(step_num, cv_loss_mlp)
    test_loss_plt1, = plt.plot(step_num, test_loss_mlp)

    plt.xlabel('step numbers')
    plt.ylabel('loss function value')
    plt.legend([train_loss_plt1, cv_loss_plt1, test_loss_plt1], ["train loss", "cross-validation loss", "test loss"])
    # plt.title('Plot of loss function values v.s. step numbers')

    plt.figure(4)
    train_miserr_plt1, = plt.plot(step_num, train_misclassification_err_mlp)
    cv_miserr_plt1, = plt.plot(step_num, cv_misclassification_err_mlp)
    test_miserr_plt1, = plt.plot(step_num, test_misclassification_err_mlp)

    plt.xlabel('step numbers')
    plt.ylabel('misclassification error')
    plt.legend([train_miserr_plt1, cv_miserr_plt1, test_miserr_plt1], ["train error", "cross-validation error", "test error"])
    # plt.title('Plot of misclassification errors v.s. step numbers')








