# -*- coding: utf-8 -*-
# @Time    : 2018/4/29 22:27
# @Author  : Jiahao Yang
# @Email   : yangjh39@uw.edu

import torch
from torch.autograd import Variable
from sklearn.metrics import average_precision_score

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------------- Problem 4 -------------------------------- #
class LinearRegression:
    def __init__(self, regular_parameter, step_size, batch_size, initial_weight):
        self.lambd = Variable(torch.FloatTensor([regular_parameter]), requires_grad=True)
        self.stepSize = step_size
        self.batchSize = batch_size
        self.w = Variable(torch.FloatTensor(initial_weight), requires_grad=True)

        self.trainLoss = []
        self.trainAPScore = []
        self.cvLoss = []
        self.cvAPScore = []

        self.testAPScore = []

    def objectiveLinearFunc(self, w, lambd, batch_x, batch_y, loss_type="square"):
        """
        Objective function: L(w) = lambda/2*||w||^2 + 1/n sum(f(y_i, <w, x_i>))
        The data type of all these input is torch.autograd.Variable so that we can use autograd in Pytorch
        """
        if loss_type == "square":
            return lambd / 2 * w.norm(2) ** 2 + 1. / len(batch_y) * sum(sum(1. / 2. * (batch_y - batch_x.mm(w)) ** 2))

        if loss_type == "logistic":
            return lambd / 2 * w.norm(2) ** 2 + 1. / len(batch_y) * sum(sum(batch_y * torch.log(1 + torch.exp(-batch_x.mm(w)))
                                                                            + (1 - batch_y) * torch.log(1 + torch.exp(batch_x.mm(w)))))

    def lossFunc(self, w, lambd, batch_x, batch_y, loss_type="square"):
        """
        Numpy version objective function
        Objective function: L(w) = lambda/2*||w||^2 + 1/n sum(f(y_i, <w, x_i>))
        The data type of all these input is torch.autograd.Variable so that we can use autograd in Pytorch
        """
        wn = w.data.numpy()
        if loss_type == "square":
            loss = lambd.data.numpy()[0] / 2 * w.norm(2).data.numpy()[0] ** 2 + 1. / len(batch_y) * sum(
                sum(1. / 2. * (batch_y - batch_x.dot(wn)) ** 2))
            return loss

        if loss_type == "logistic":
            loss = lambd.data.numpy()[0] / 2 * w.norm(2).data.numpy()[0] ** 2 + 1. / len(batch_y) * sum(sum(
                batch_y * np.log(1 + np.exp(-batch_x.dot(wn))) + (1 - batch_y) * np.log(1 + np.exp(batch_x.dot(wn)))))
            return loss

    def batch_iter(self, x, y, batchSize, shuffle=True):
        """
        Generate a minibatch iterator for a dataset.
        """
        data_size = len(y)
        num_batches = np.ceil(data_size / batchSize).astype(int)

        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_y = y[shuffle_indices]
            shuffled_x = x[shuffle_indices]
        else:
            shuffled_y = y
            shuffled_x = x

        for batch_num in range(num_batches):
            start_index = batch_num * batchSize
            end_index = min((batch_num + 1) * batchSize, data_size)

            if start_index != end_index:
                yield shuffled_y[start_index:end_index], shuffled_x[start_index:end_index]

    def minibatch_gradient_descent(self, x, y, cvx, cvy, loss_type, max_iter_num=10000):
        """
        Mini-batch gradient descent algorithm.
        """

        n_iter = 0

        while n_iter < max_iter_num:
            batch_iter_instant = self.batch_iter(x, y, self.batchSize)

            for minibatch_y, minibatch_x in batch_iter_instant:
                minibatch_y = Variable(torch.FloatTensor(minibatch_y), requires_grad=False)
                minibatch_x = Variable(torch.FloatTensor(minibatch_x), requires_grad=False)

                temp_y = self.objectiveLinearFunc(self.w, self.lambd, minibatch_x, minibatch_y, loss_type)
                temp_y.backward()

                w_temp = self.w - self.stepSize * self.w.grad
                self.w = Variable(torch.FloatTensor(w_temp.data), requires_grad=True)
                n_iter += 1
                # print(n_iter)

                if n_iter % np.floor(y.shape[0]/(2. * self.batchSize)) == 0:
                    train_loss = self.lossFunc(self.w, self.lambd, x, y, loss_type)
                    self.trainLoss.append(train_loss)
                    self.trainAPScore.append(average_precision_score(y, x.dot(self.w.data.numpy())))

                    cv_loss = self.lossFunc(self.w, self.lambd, cvx, cvy, loss_type)
                    self.cvLoss.append(cv_loss)
                    self.cvAPScore.append(average_precision_score(cvy, cvx.dot(self.w.data.numpy())))

                    print("Step " + str(n_iter) + ", " + loss_type + " loss" + ": the train loss is " + str(self.trainLoss[-1])
                          + ", the train AP score is " + str(self.trainAPScore[-1]))

                    print("Step " + str(n_iter) + ", " + loss_type + " loss" + ": the cv loss is " + str(self.cvLoss[-1])
                          + ", the cv AP score is " + str(self.cvAPScore[-1]))

    def heavy_ball(self, x, y, cvx, cvy, loss_type, max_iter_num=10000):
        """
        Momentum gradient descent algorithm.(Heavy ball)
        """

        n_iter = 0
        v = self.w

        while n_iter < max_iter_num:
            batch_iter_instant = self.batch_iter(x, y, self.batchSize)

            for minibatch_y, minibatch_x in batch_iter_instant:
                minibatch_y = Variable(torch.FloatTensor(minibatch_y), requires_grad=False)
                minibatch_x = Variable(torch.FloatTensor(minibatch_x), requires_grad=False)

                temp_y = self.objectiveLinearFunc(self.w, self.lambd, minibatch_x, minibatch_y, loss_type)
                temp_y.backward()

                v_temp = 0.9 * v + self.stepSize * self.w.grad
                w_temp = self.w - v_temp
                v = Variable(torch.FloatTensor(v_temp.data), requires_grad=False)
                self.w = Variable(torch.FloatTensor(w_temp.data), requires_grad=True)

                n_iter += 1

                if n_iter % np.floor(y.shape[0] / (2. * self.batchSize)) == 0:
                    train_loss = self.lossFunc(self.w, self.lambd, x, y, loss_type)
                    self.trainLoss.append(train_loss)
                    self.trainAPScore.append(average_precision_score(y, x.dot(self.w.data.numpy())))

                    cv_loss = self.lossFunc(self.w, self.lambd, cvx, cvy, loss_type)
                    self.cvLoss.append(cv_loss)
                    self.cvAPScore.append(average_precision_score(cvy, cvx.dot(self.w.data.numpy())))

                    print("Step " + str(n_iter) + ", " + loss_type + " loss" + ": the train loss is " + str(
                        self.trainLoss[-1])
                          + ", the train AP score is " + str(self.trainAPScore[-1]))

                    print(
                        "Step " + str(n_iter) + ", " + loss_type + " loss" + ": the cv loss is " + str(self.cvLoss[-1])
                        + ", the cv AP score is " + str(self.cvAPScore[-1]))

    def adagrad(self, x, y, cvx, cvy, loss_type, max_iter_num=6000):
        """
        Adaptive gradient descent algorithm.(Adagrad)
        """

        # set initial parameter(suggested by the author of Adam)
        gradSquareSum = Variable(torch.FloatTensor(torch.zeros(self.w.data.size())), requires_grad=False)

        n_iter = 0

        while n_iter < max_iter_num:
            batch_iter_instant = self.batch_iter(x, y, self.batchSize)

            for minibatch_y, minibatch_x in batch_iter_instant:
                minibatch_y = Variable(torch.FloatTensor(minibatch_y), requires_grad=False)
                minibatch_x = Variable(torch.FloatTensor(minibatch_x), requires_grad=False)

                temp_y = self.objectiveLinearFunc(self.w, self.lambd, minibatch_x, minibatch_y, loss_type)
                temp_y.backward()

                gradSquareSum = gradSquareSum + self.w.grad ** 2
                w_temp = self.w - self.stepSize / (torch.sqrt(gradSquareSum)) * self.w.grad

                self.w = Variable(torch.FloatTensor(w_temp.data), requires_grad=True)

                n_iter += 1
                # print(n_iter)

                if n_iter % np.floor(y.shape[0] / (2. * self.batchSize)) == 0:
                    train_loss = self.lossFunc(self.w, self.lambd, x, y, loss_type)
                    self.trainLoss.append(train_loss)
                    self.trainAPScore.append(average_precision_score(y, x.dot(self.w.data.numpy())))

                    cv_loss = self.lossFunc(self.w, self.lambd, cvx, cvy, loss_type)
                    self.cvLoss.append(cv_loss)
                    self.cvAPScore.append(average_precision_score(cvy, cvx.dot(self.w.data.numpy())))

                    print("Step " + str(n_iter) + ", " + loss_type + " loss" + ": the train loss is " + str(
                        self.trainLoss[-1])
                          + ", the train AP score is " + str(self.trainAPScore[-1]))

                    print(
                        "Step " + str(n_iter) + ", " + loss_type + " loss" + ": the cv loss is " + str(self.cvLoss[-1])
                        + ", the cv AP score is " + str(self.cvAPScore[-1]))

    def train(self, x, y, cvx, cvy, loss_type, method="SGD"):
        """
        Several improved gradient descent algorithm.
        """
        if method == "SGD":
            print("Using mini-batch gradient descent algorithm...")
            self.minibatch_gradient_descent(x, y, cvx, cvy, loss_type)
            print("Model: lambda=" + str(self.lambd.data.numpy()[0]) + ", step size=" + str(self.stepSize)
                  + ", batch size=" + str(self.batchSize) + ", loss type=" + loss_type
                  + ", train method=" + method + "......done")

        elif method == "HB":
            print("Using Heavy Ball gradient descent algorithm...")
            self.heavy_ball(x, y, cvx, cvy, loss_type)
            print("Model: lambda=" + str(self.lambd.data.numpy()[0]) + ", step size=" + str(self.stepSize)
                  + ", batch size=" + str(self.batchSize) + ", loss type=" + loss_type
                  + ", train method=" + method + "......done")

        else:
            print("Using Adaptive gradient descent algorithm...")
            self.adagrad(x, y, cvx, cvy, loss_type)
            print("Model: lambda=" + str(self.lambd.data.numpy()[0]) + ", step size=" + str(self.stepSize)
                  + ", batch size=" + str(self.batchSize) + ", loss type=" + loss_type
                  + ", train method=" + method + "......done")

    def predict(self, new_x, new_y):
        newx = Variable(torch.FloatTensor(new_x), requires_grad=True)

        yscore = newx.mm(self.w).data.numpy()

        for i in np.arange(yscore.shape[1]):
            self.testAPScore.append(average_precision_score(new_y[:, i], yscore[:, i]))


class MLP:
    def __init__(self, regular_parameter, step_size, batch_size, initial_weight1, initial_weight2):
        self.lambd = Variable(torch.FloatTensor([regular_parameter]), requires_grad=True)
        self.stepSize = step_size
        self.batchSize = batch_size
        self.w1 = Variable(torch.FloatTensor(initial_weight1), requires_grad=True)
        self.w2 = Variable(torch.FloatTensor(initial_weight2), requires_grad=True)

        self.trainLoss = []
        self.trainAPScore = []
        self.cvLoss = []
        self.cvAPScore = []

        self.testAPScore = []

    def objectiveMLPFunc(self, w1, w2, lambd, batch_x, batch_y, loss_type="square"):
        """
        Objective function: L(w) = lambda/2*||w||^2 + 1/n sum(f(y_i, <w, x_i>))
        The data type of all these input is torch.autograd.Variable so that we can use autograd in Pytorch
        """

        yhat = ((batch_x.mm(w1) > 0).type(torch.FloatTensor) * (batch_x.mm(w1))).mm(w2)

        if loss_type == "square":
            return lambd / 2 * (w1.norm(2) ** 2 + w2.norm(2) ** 2) \
                   + 1. / len(batch_y) * sum(sum(1. / 2. * (batch_y - yhat) ** 2))

        if loss_type == "logistic":
            return lambd / 2 * (w1.norm(2) ** 2 + w2.norm(2) ** 2) \
                   + 1. / len(batch_y) * sum(sum(batch_y * torch.log(1 + torch.exp(-yhat))
                                                 + (1 - batch_y) * torch.log(1 + torch.exp(yhat))))

    def lossFunc(self, w1, w2, lambd, batch_x, batch_y, loss_type="square"):
        """
        Numpy version objective function
        Objective function: L(w) = lambda/2*||w||^2 + 1/n sum(f(y_i, <w, x_i>))
        The data type of all these input is torch.autograd.Variable so that we can use autograd in Pytorch
        """
        wn1 = w1.data.numpy()
        wn2 = w2.data.numpy()

        yhat = ((batch_x.dot(wn1) > 0) * (batch_x.dot(wn1))).dot(wn2)

        if loss_type == "square":
            loss = lambd.data.numpy()[0] / 2 * (w1.norm(2).data.numpy()[0] ** 2 + w2.norm(2).data.numpy()[0] ** 2) \
                   + 1. / len(batch_y) * sum(sum(1. / 2. * (batch_y - yhat) ** 2))
            return loss

        if loss_type == "logistic":
            loss = lambd.data.numpy()[0] / 2 * (w1.norm(2).data.numpy()[0] ** 2 + w2.norm(2).data.numpy()[0] ** 2) \
                   + 1. / len(batch_y) * sum(sum(batch_y * np.log(1 + np.exp(-yhat))
                                                 + (1 - batch_y) * np.log(1 + np.exp(yhat))))
            return loss

    def batch_iter(self, x, y, batchSize, shuffle=True):
        """
        Generate a minibatch iterator for a dataset.
        """
        data_size = len(y)
        num_batches = np.ceil(data_size / batchSize).astype(int)

        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_y = y[shuffle_indices]
            shuffled_x = x[shuffle_indices]
        else:
            shuffled_y = y
            shuffled_x = x

        for batch_num in range(num_batches):
            start_index = batch_num * batchSize
            end_index = min((batch_num + 1) * batchSize, data_size)

            if start_index != end_index:
                yield shuffled_y[start_index:end_index], shuffled_x[start_index:end_index]

    def minibatch_gradient_descent(self, x, y, cvx, cvy, loss_type, max_iter_num=10000):
        """
        Mini-batch gradient descent algorithm.
        """

        n_iter = 0

        while n_iter < max_iter_num:
            batch_iter_instant = self.batch_iter(x, y, self.batchSize)

            for minibatch_y, minibatch_x in batch_iter_instant:
                minibatch_y = Variable(torch.FloatTensor(minibatch_y), requires_grad=False)
                minibatch_x = Variable(torch.FloatTensor(minibatch_x), requires_grad=False)

                temp_y = self.objectiveMLPFunc(self.w1, self.w2, self.lambd, minibatch_x, minibatch_y, loss_type)
                temp_y.backward()

                w_temp1 = self.w1 - self.stepSize * self.w1.grad
                self.w1 = Variable(torch.FloatTensor(w_temp1.data), requires_grad=True)
                w_temp2 = self.w2 - self.stepSize * self.w2.grad
                self.w2 = Variable(torch.FloatTensor(w_temp2.data), requires_grad=True)

                n_iter += 1

                if n_iter % np.floor(y.shape[0]/(2. * self.batchSize)) == 0:
                    train_loss = self.lossFunc(self.w1, self.w2, self.lambd, x, y, loss_type)
                    self.trainLoss.append(train_loss)
                    self.trainAPScore.append(average_precision_score(y, ((x.dot(self.w1.data.numpy()) > 0) *
                                                                         (x.dot(self.w1.data.numpy()))).dot(self.w2.data.numpy())))

                    cv_loss = self.lossFunc(self.w1, self.w2, self.lambd, cvx, cvy, loss_type)
                    self.cvLoss.append(cv_loss)
                    self.cvAPScore.append(average_precision_score(cvy, ((cvx.dot(self.w1.data.numpy()) > 0) *
                                                                        (cvx.dot(self.w1.data.numpy()))).dot(self.w2.data.numpy())))

                    print("Step " + str(n_iter) + ", " + loss_type + " loss" + ": the train loss is " + str(self.trainLoss[-1])
                          + ", the train AP score is " + str(self.trainAPScore[-1]))

                    print("Step " + str(n_iter) + ", " + loss_type + " loss" + ": the cv loss is " + str(self.cvLoss[-1])
                          + ", the cv AP score is " + str(self.cvAPScore[-1]))

    def heavy_ball(self, x, y, cvx, cvy, loss_type, max_iter_num=10000):
        """
        Momentum gradient descent algorithm.(Heavy ball)
        """

        n_iter = 0
        v1 = self.w1
        v2 = self.w2

        while n_iter < max_iter_num:
            batch_iter_instant = self.batch_iter(x, y, self.batchSize)

            for minibatch_y, minibatch_x in batch_iter_instant:
                minibatch_y = Variable(torch.FloatTensor(minibatch_y), requires_grad=False)
                minibatch_x = Variable(torch.FloatTensor(minibatch_x), requires_grad=False)

                temp_y = self.objectiveMLPFunc(self.w1, self.w2, self.lambd, minibatch_x, minibatch_y, loss_type)
                temp_y.backward()

                v_temp1 = 0.9 * v1 + self.stepSize * self.w1.grad
                w_temp1 = self.w1 - v_temp1
                v1 = Variable(torch.FloatTensor(v_temp1.data), requires_grad=False)
                self.w1 = Variable(torch.FloatTensor(w_temp1.data), requires_grad=True)

                v_temp2 = 0.9 * v2 + self.stepSize * self.w2.grad
                w_temp2 = self.w2 - v_temp2
                v2 = Variable(torch.FloatTensor(v_temp2.data), requires_grad=False)
                self.w2 = Variable(torch.FloatTensor(w_temp2.data), requires_grad=True)

                n_iter += 1

                if n_iter % np.floor(y.shape[0]/(2. * self.batchSize)) == 0:
                    train_loss = self.lossFunc(self.w1, self.w2, self.lambd, x, y, loss_type)
                    self.trainLoss.append(train_loss)
                    self.trainAPScore.append(average_precision_score(y, ((x.dot(self.w1.data.numpy()) > 0) *
                                                                         (x.dot(self.w1.data.numpy()))).dot(self.w2.data.numpy())))

                    cv_loss = self.lossFunc(self.w1, self.w2, self.lambd, cvx, cvy, loss_type)
                    self.cvLoss.append(cv_loss)
                    self.cvAPScore.append(average_precision_score(cvy, ((cvx.dot(self.w1.data.numpy()) > 0) *
                                                                        (cvx.dot(self.w1.data.numpy()))).dot(self.w2.data.numpy())))

                    print("Step " + str(n_iter) + ", " + loss_type + " loss" + ": the train loss is " + str(self.trainLoss[-1])
                          + ", the train AP score is " + str(self.trainAPScore[-1]))

                    print("Step " + str(n_iter) + ", " + loss_type + " loss" + ": the cv loss is " + str(self.cvLoss[-1])
                          + ", the cv AP score is " + str(self.cvAPScore[-1]))

    def adagrad(self, x, y, cvx, cvy, loss_type, max_iter_num=12500):
        """
        Adaptive gradient descent algorithm.(Adagrad)
        """

        # set initial parameter(suggested by the author of Adam)
        gradSquareSum1 = Variable(torch.FloatTensor(torch.zeros(self.w1.data.size())), requires_grad=False)
        gradSquareSum2 = Variable(torch.FloatTensor(torch.zeros(self.w2.data.size())), requires_grad=False)

        eps = 1e-8

        n_iter = 0

        while n_iter < max_iter_num:
            batch_iter_instant = self.batch_iter(x, y, self.batchSize)

            for minibatch_y, minibatch_x in batch_iter_instant:
                minibatch_y = Variable(torch.FloatTensor(minibatch_y), requires_grad=False)
                minibatch_x = Variable(torch.FloatTensor(minibatch_x), requires_grad=False)

                temp_y = self.objectiveMLPFunc(self.w1, self.w2, self.lambd, minibatch_x, minibatch_y, loss_type)
                temp_y.backward()

                gradSquareSum1 = gradSquareSum1 + self.w1.grad ** 2
                w_temp1 = self.w1 - self.stepSize / (torch.sqrt(gradSquareSum1 + eps)) * self.w1.grad

                self.w1 = Variable(torch.FloatTensor(w_temp1.data), requires_grad=True)

                gradSquareSum2 = gradSquareSum2 + self.w2.grad ** 2
                w_temp2 = self.w2 - self.stepSize / (torch.sqrt(gradSquareSum2 + eps)) * self.w2.grad

                self.w2 = Variable(torch.FloatTensor(w_temp2.data), requires_grad=True)

                n_iter += 1

                if n_iter % np.floor(y.shape[0]/(2. * self.batchSize)) == 0:
                    train_loss = self.lossFunc(self.w1, self.w2, self.lambd, x, y, loss_type)
                    self.trainLoss.append(train_loss)
                    self.trainAPScore.append(average_precision_score(y, ((x.dot(self.w1.data.numpy()) > 0) *
                                                                         (x.dot(self.w1.data.numpy()))).dot(self.w2.data.numpy())))

                    cv_loss = self.lossFunc(self.w1, self.w2, self.lambd, cvx, cvy, loss_type)
                    self.cvLoss.append(cv_loss)
                    self.cvAPScore.append(average_precision_score(cvy, ((cvx.dot(self.w1.data.numpy()) > 0) *
                                                                        (cvx.dot(self.w1.data.numpy()))).dot(self.w2.data.numpy())))

                    print("Step " + str(n_iter) + ", " + loss_type + " loss" + ": the train loss is " + str(self.trainLoss[-1])
                          + ", the train AP score is " + str(self.trainAPScore[-1]))

                    print("Step " + str(n_iter) + ", " + loss_type + " loss" + ": the cv loss is " + str(self.cvLoss[-1])
                          + ", the cv AP score is " + str(self.cvAPScore[-1]))

    def train(self, x, y, cvx, cvy, loss_type, method="SGD"):
        """
        Several improved gradient descent algorithm.
        """
        if method == "SGD":
            print("Using mini-batch gradient descent algorithm...")
            self.minibatch_gradient_descent(x, y, cvx, cvy, loss_type)
            print("Model: lambda=" + str(self.lambd.data.numpy()[0]) + ", step size=" + str(self.stepSize)
                  + ", batch size=" + str(self.batchSize) + ", loss type=" + loss_type
                  + ", train method=" + method + "......done")

        elif method == "HB":
            print("Using Heavy Ball gradient descent algorithm...")
            self.heavy_ball(x, y, cvx, cvy, loss_type)
            print("Model: lambda=" + str(self.lambd.data.numpy()[0]) + ", step size=" + str(self.stepSize)
                  + ", batch size=" + str(self.batchSize) + ", loss type=" + loss_type
                  + ", train method=" + method + "......done")

        else:
            print("Using Adaptive gradient descent algorithm...")
            self.adagrad(x, y, cvx, cvy, loss_type)
            print("Model: lambda=" + str(self.lambd.data.numpy()[0]) + ", step size=" + str(self.stepSize)
                  + ", batch size=" + str(self.batchSize) + ", loss type=" + loss_type
                  + ", train method=" + method + "......done")

    def predict(self, new_x, new_y):
        yscore = ((new_x.dot(self.w1.data.numpy()) > 0) * (new_x.dot(self.w1.data.numpy()))).dot(self.w2.data.numpy())

        for i in np.arange(yscore.shape[1]):
            self.testAPScore.append(average_precision_score(new_y[:, i], yscore[:, i]))


if __name__ == "__main__":
    path = "/home/yangjh39/548/data/features2_small"

    img_cat_train = np.array(pd.read_csv(path + "/img_cat_train2014.csv").iloc[:, 1::])
    img_cat_cv = np.array(pd.read_csv(path + "/img_cat_val2014.csv").iloc[:, 1::])
    img_cat_test = np.array(pd.read_csv(path + "/img_cat_test2014.csv").iloc[:, 1::])

    features_train = np.array(pd.read_csv(path + "/feats_train2014.csv").iloc[:, 1::])
    features_cv = np.array(pd.read_csv(path + "/feats_val2014.csv").iloc[:, 1::])
    features_test = np.array(pd.read_csv(path + "/feats_test2014.csv").iloc[:, 1::])

    print("Import data....successfully")
    # features_train = features_train / np.max(abs(features_train))
    # features_cv = features_cv / np.max(abs(features_cv))
    # features_test = features_test / np.max(abs(features_test))

    # 4.1 SGD
    lm1 = LinearRegression(200, 1e-6, 16, np.zeros([features_train.shape[1], img_cat_train.shape[1]]))
    lm1.train(features_train, img_cat_train, features_cv, img_cat_cv, "square", "SGD")

    lm2 = LinearRegression(20, 5e-5, 16, np.zeros([features_train.shape[1], img_cat_train.shape[1]]))
    lm2.train(features_train, img_cat_train, features_cv, img_cat_cv, "logistic", "Adagrad")

    # Plot of lm
    plt.figure(1)
    square_cv_aps_plt, = plt.plot(np.arange(0.5, len(lm1.cvAPScore)*0.5+0.5, 0.5), lm1.cvAPScore)
    logis_cv_aps_plt, = plt.plot(np.arange(0.5, len(lm2.cvAPScore)*0.5+0.5, 0.5), lm2.cvAPScore)

    plt.xlabel('echos of dataset ')
    plt.ylabel('AP score')
    plt.legend([square_cv_aps_plt, logis_cv_aps_plt], ["square loss", "logistic loss"])
    plt.title("Comparison between square loss and logistic loss")

    plt.figure(2)
    train_loss_plt, = plt.plot(np.arange(0.5, len(lm2.trainLoss) * 0.5 + 0.5, 0.5), lm2.trainLoss)
    cv_loss_plt, = plt.plot(np.arange(0.5, len(lm2.cvLoss) * 0.5 + 0.5, 0.5), lm2.cvLoss)

    plt.xlabel('echos of dataset ')
    plt.ylabel('loss')
    plt.legend([train_loss_plt, cv_loss_plt], ["train loss", "development loss"])
    # plt.title("Loss value of SGD with logistic loss function")

    plt.figure(3)
    train_aps_plt, = plt.plot(np.arange(0.5, len(lm2.trainAPScore) * 0.5 + 0.5, 0.5), lm2.trainAPScore)
    cv_aps_plt, = plt.plot(np.arange(0.5, len(lm2.cvAPScore) * 0.5 + 0.5, 0.5), lm2.cvAPScore)

    plt.xlabel('echos of dataset ')
    plt.ylabel('AP score')
    plt.legend([train_loss_plt, cv_loss_plt], ["train AP score", "development AP score"])
    # plt.title("AP score of SGD with logistic loss function")

    lm2.predict(features_test, img_cat_test)
    pd.DataFrame(lm2.testAPScore).to_csv("linear_adg_testaps.csv")

    np.max(lm2.cvAPScore)

    hidden_nodes_nums = 150
    ini_w1 = np.random.normal(0, 1, [features_train.shape[1], hidden_nodes_nums]) / 100.
    ini_w2 = np.random.normal(0, 1, [hidden_nodes_nums, img_cat_train.shape[1]]) / 100.
    mlp2 = MLP(0.5, 1e-3, 16, ini_w1, ini_w2)
    mlp2.train(features_train, img_cat_train, features_cv, img_cat_cv, "logistic", "SGD")

    # Plot of mlp
    plt.figure(4)
    mlp_train_loss_plt, = plt.plot(np.arange(0.5, len(mlp2.trainLoss) * 0.5 + 0.5, 0.5), mlp2.trainLoss)
    mlp_cv_loss_plt, = plt.plot(np.arange(0.5, len(mlp2.cvLoss) * 0.5 + 0.5, 0.5), mlp2.cvLoss)

    plt.xlabel('echos of dataset ')
    plt.ylabel('loss')
    # plt.ylim([0, 60])
    # plt.xlim([6, 16])
    plt.legend([mlp_train_loss_plt, mlp_cv_loss_plt], ["train loss", "development loss"])
    # plt.title("Loss value of SGD with logistic loss function")

    plt.figure(5)
    mlp_train_aps_plt, = plt.plot(np.arange(0.5, len(mlp2.trainAPScore) * 0.5 + 0.5, 0.5), mlp2.trainAPScore)
    mlp_cv_aps_plt, = plt.plot(np.arange(0.5, len(mlp2.cvAPScore) * 0.5 + 0.5, 0.5), mlp2.cvAPScore)

    plt.xlabel('echos of dataset ')
    plt.ylabel('AP score')
    plt.legend([mlp_train_aps_plt, mlp_cv_aps_plt], ["train AP score", "development AP score"])
    # plt.title("AP score of SGD with logistic loss function")

    mlp2.predict(features_test, img_cat_test)
    pd.DataFrame(mlp2.testAPScore).to_csv("mlp_adg_testaps.csv")

    np.max(mlp2.cvAPScore)






    # Plot of mlp
    plt.figure(4)
    mlp_train_loss_plt, = plt.plot(np.arange(0.5, len(a) * 0.5 + 0.5, 0.5), a)
    mlp_cv_loss_plt, = plt.plot(np.arange(0.5, len(b) * 0.5 + 0.5, 0.5), b)

    plt.xlabel('echos of dataset ')
    plt.ylabel('loss')
    # plt.ylim([0, 60])
    # plt.xlim([6, 16])
    plt.legend([mlp_train_loss_plt, mlp_cv_loss_plt], ["train loss", "development loss"])
    # plt.title("Loss value of SGD with logistic loss function")

    plt.figure(5)
    mlp_train_aps_plt, = plt.plot(np.arange(0.5, len(mlp2.trainAPScore) * 0.5 + 0.5, 0.5), mlp2.trainAPScore)
    mlp_cv_aps_plt, = plt.plot(np.arange(0.5, len(mlp2.cvAPScore) * 0.5 + 0.5, 0.5), mlp2.cvAPScore)

    plt.xlabel('echos of dataset ')
    plt.ylabel('AP score')
    plt.legend([mlp_train_aps_plt, mlp_cv_aps_plt], ["train AP score", "development AP score"])
    # plt.title("AP score of SGD with logistic loss function")

    mlp2.predict(features_test, img_cat_test)
    pd.DataFrame(mlp2.testAPScore).to_csv("mlp_adg_testaps.csv")

    np.max(mlp2.cvAPScore)


