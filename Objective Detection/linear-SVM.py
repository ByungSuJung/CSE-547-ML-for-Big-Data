# -*- coding: utf-8 -*-
# @Time    : 2018/5/14 11:20
# @Author  : Jiahao Yang
# @Email   : yangjh39@uw.edu

from pycocotools.coco import COCO
import _pickle as cPickle

from multiprocessing.dummy import Pool as ThreadPool
import gc

import os
import numpy as np
import random
from PIL import Image
import time
from math import ceil, floor
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.autograd import Variable
from sklearn.metrics import average_precision_score


class Featurizer:
    dim = 11776  # for small features

    def __init__(self):
        # pyramidal pooling of sizes 1, 3, 6
        self.pool1 = nn.AdaptiveMaxPool2d(1)
        self.pool3 = nn.AdaptiveMaxPool2d(3)
        self.pool6 = nn.AdaptiveMaxPool2d(6)
        self.lst = [self.pool1, self.pool3, self.pool6]

    def featurize(self, projected_bbox, image_features):
        # projected_bbox: bbox projected onto final layer
        # image_features: C x W x H tensor : output of conv net
        full_image_features = torch.from_numpy(image_features)
        x, y, x1, y1 = projected_bbox
        crop = full_image_features[:, x:x1, y:y1]
        #         return torch.cat([self.pool1(crop).view(-1), self.pool3(crop).view(-1),
        #                           self.pool6(crop).view(-1)], dim=0) # returns torch Variable
        return torch.cat([self.pool1(crop).view(-1), self.pool3(crop).view(-1),
                          self.pool6(crop).view(-1)], dim=0).data.numpy()  # returns numpy array


class load_dataset:
    def __init__(self, data_dir):
        self.dataDir = data_dir
        
        self.labelForBboxAllImageTrain = 0
        self.imageIdAndBboxTrain = 0
        self.img_idsTrain = 0
        self.featsTrain = 0
        
        self.labelForBboxAllImageCv = 0
        self.imageIdAndBboxCv = 0
        self.img_idsCv = 0
        self.featsCv = 0
        
        print("loading labels for bounding box...")
        with open(os.path.join(dataDir, 'bboxes', 'labelForBboxAllImage_train2014.p'), 'rb') as fp:
            self.labelForBboxAllImageTrain = cPickle.load(fp)
        fp.close()
        print("load labels for bboxes successfully...")
        
        self.labelForBboxAllImageTrain = np.array(self.labelForBboxAllImageTrain)
        
        # load image ids and the candidate bounding box respectively
        print("loading image ids and the candidate bounding box respectively...")
        self.imageIdAndBboxTrain = cPickle.load(open(os.path.join(dataDir, 'bboxes', 'train2014_bboxes.p'), 'rb'), encoding='latin1')
        print("load image ids and the candidate bounding box successfully...")
        
        # load features
        print("loading features...")
        [self.img_idsTrain, self.featsTrain] = cPickle.load(open(os.path.join(dataDir, 'features_small', 'train2014.p'),'rb'), encoding='latin1')
        print("load features successfully...")
        
        print("loading labels for bounding box...")
        with open(os.path.join(dataDir, 'bboxes', 'labelForBboxAllImage_val2014.p'), 'rb') as fp:
            self.labelForBboxAllImageCv = cPickle.load(fp)
        fp.close()
        print("load labels for bboxes successfully...")
        
        self.labelForBboxAllImageCv = np.array(self.labelForBboxAllImageCv)
        
        # load image ids and the candidate bounding box respectively
        print("loading image ids and the candidate bounding box respectively...")
        self.imageIdAndBboxCv = cPickle.load(open(os.path.join(dataDir, 'bboxes', 'val2014_bboxes.p'), 'rb'), encoding='latin1')
        print("load image ids and the candidate bounding box successfully...")
        
        # load features
        print("loading features...")
        [self.img_idsCv, self.featsCv] = cPickle.load(open(os.path.join(dataDir, 'features_small', 'val2014.p'),'rb'), encoding='latin1')
        print("load features successfully...")
    

class LinearSVM:
    def __init__(self, regular_parameter, step_size, batch_size, initial_weight, train_size):
        self.lambd = Variable(torch.FloatTensor([regular_parameter]), requires_grad=True)
        self.stepSize = step_size
        self.batchSize = batch_size
        self.w = Variable(torch.FloatTensor(initial_weight), requires_grad=True)
        self.maxiternum = np.floor(8 * train_size / batch_size)
        
        self.trainLoss = []
        self.trainAPScore = []
        self.cvLoss = []
        self.cvAPScore = []

        self.testAPScore = []
        
        self.topIndex = 0

    def objectiveLinearFunc(self, w, lambd, batch_x, batch_y):
        """
        Objective function: L(w) = lambda/2*||w||^2 + 1/n sum(f(y_i, <w, x_i>))
        The data type of all these input is torch.autograd.Variable so that we can use autograd in Pytorch
        """
        batch_y = batch_y.contiguous().view([-1,1])

        return lambd / 2. * torch.norm(w) ** 2. + 1. / (2. * len(batch_y)) * torch.sum(torch.clamp(1. - (2. * batch_y - 1.) * batch_x.mm(w), min=0) ** 2.)

    def lossFunc(self, w, lambd, batch_x, batch_y):
        """
        Numpy version objective function
        Objective function: L(w) = lambda/2*||w||^2 + 1/n sum(f(y_i, <w, x_i>))
        The data type of all these input is torch.autograd.Variable so that we can use autograd in Pytorch
        """        
        batch_y = batch_y.reshape([-1,1])
        
        wn = w.data.numpy()
        loss = lambd.data.numpy()[0] / 2. * torch.norm(w).data.numpy()[0] ** 2. + 1. / (2. * len(batch_y)) * (np.clip(1. - (2. * batch_y - 1.) * batch_x.dot(wn), 0., None) ** 2.).sum()

        return loss

    def batch_iter(self, x, y, batchSize, shuffle=True):
        """
        Generate a minibatch iterator for a dataset.
        """
        data_size = len(y)
        num_batches = np.ceil(data_size / batchSize).astype(int)

        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))

        else:
            shuffle_indices = np.arange(data_size)

        for batch_num in range(num_batches):
            start_index = batch_num * batchSize
            end_index = min((batch_num + 1) * batchSize, data_size)

            if start_index != end_index:
                yield y[shuffle_indices[start_index:end_index]], x[shuffle_indices[start_index:end_index]]

    def minibatch_gradient_descent(self, x, y, cvx, cvy, max_iter_num):
        """
        Mini-batch gradient descent algorithm.
        """

        n_iter = 0

        while n_iter < max_iter_num:
            batch_iter_instant = self.batch_iter(x, y, self.batchSize)

            for minibatch_y, minibatch_x in batch_iter_instant:
                minibatch_y = Variable(torch.FloatTensor(minibatch_y), requires_grad=False)
                minibatch_x = Variable(torch.FloatTensor(minibatch_x), requires_grad=False)
                
                n_iter += 1
                
                temp_y = self.objectiveLinearFunc(self.w, self.lambd, minibatch_x, minibatch_y)
                temp_y.backward()
                
                w_temp = self.w - self.stepSize * self.w.grad
                self.w = Variable(torch.FloatTensor(w_temp.data), requires_grad=True)
                
                #reset gradient
                if self.w.grad is not None: 
                    self.w.grad.data.zero_()
                
                # print(n_iter)

                if n_iter % np.floor(y.shape[0]/(2. * self.batchSize)) == 0:
                    train_loss = self.lossFunc(self.w, self.lambd, x, y)
                    self.trainLoss.append(train_loss)
                    self.trainAPScore.append(average_precision_score(y, x.dot(self.w.data.numpy())))

                    cv_loss = self.lossFunc(self.w, self.lambd, cvx, cvy)
                    self.cvLoss.append(cv_loss)
                    self.cvAPScore.append(average_precision_score(cvy, cvx.dot(self.w.data.numpy())))
                    

                    print("Step " + str(n_iter) + ": the train loss is " + str(self.trainLoss[-1])
                          + ", the train AP score is " + str(self.trainAPScore[-1]))

                    print("Step " + str(n_iter) + ": the cv loss is " + str(self.cvLoss[-1])
                          + ", the cv AP score is " + str(self.cvAPScore[-1]))

    def heavy_ball(self, x, y, cvx, cvy, max_iter_num):
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

                temp_y = self.objectiveLinearFunc(self.w, self.lambd, minibatch_x, minibatch_y)
                temp_y.backward()

                v_temp = 0.9 * v + self.stepSize * self.w.grad
                w_temp = self.w - v_temp
                v = Variable(torch.FloatTensor(v_temp.data), requires_grad=False)
                self.w = Variable(torch.FloatTensor(w_temp.data), requires_grad=True)
                
                #reset gradient
                if self.w.grad is not None: 
                    self.w.grad.data.zero_()

                n_iter += 1

                if n_iter % np.floor(y.shape[0] / (2. * self.batchSize)) == 0:
                    train_loss = self.lossFunc(self.w, self.lambd, x, y)
                    self.trainLoss.append(train_loss)
                    self.trainAPScore.append(average_precision_score(y, x.dot(self.w.data.numpy())))

                    cv_loss = self.lossFunc(self.w, self.lambd, cvx, cvy)
                    self.cvLoss.append(cv_loss)
                    self.cvAPScore.append(average_precision_score(cvy, cvx.dot(self.w.data.numpy())))

                    print("Step " + str(n_iter) + ": the train loss is " + str(self.trainLoss[-1])
                          + ", the train AP score is " + str(self.trainAPScore[-1]))

                    print("Step " + str(n_iter) + ": the cv loss is " + str(self.cvLoss[-1])
                          + ", the cv AP score is " + str(self.cvAPScore[-1]))

    def train(self, x, y, cvx, cvy, method="SGD"):
        """
        Several improved gradient descent algorithm.
        """
        if method == "SGD":
            print("Using mini-batch gradient descent algorithm...")
            self.minibatch_gradient_descent(x, y, cvx, cvy, self.maxiternum)
            print("Model: lambda=" + str(self.lambd.data.numpy()[0]) + ", step size=" + str(self.stepSize)
                  + ", batch size=" + str(self.batchSize) + ", train method=" + method + "......done")

        elif method == "HB":
            print("Using Heavy Ball gradient descent algorithm...")
            self.heavy_ball(x, y, cvx, cvy, self.maxiternum)
            print("Model: lambda=" + str(self.lambd.data.numpy()[0]) + ", step size=" + str(self.stepSize)
                  + ", batch size=" + str(self.batchSize) + ", train method=" + method + "......done")

    def predict(self, new_x, new_y):
        newx = Variable(torch.FloatTensor(new_x), requires_grad=True)

        yscore = newx.mm(self.w).data.numpy()
        self.topIndex = sorted(range(len(yscore)), key=lambda i: yscore[i])[-int(np.ceil(len(yscore)/2)):]
        
        self.testAPScore.append(average_precision_score(new_y, yscore))


# nearest neighbor in 1-based indexing
def _nnb_1(x):
    x1 = int(floor((x + 8) / 16.0))
    x1 = max(1, min(x1, 13))
    return x1


def project_onto_feature_space(rect, image_dims):
    # project bounding box onto conv net
    # @param rect: (x, y, w, h)
    # @param image_dims: (imgx, imgy), the size of the image
    # output bbox: (x, y, x'+1, y'+1) where the box is x:x', y:y'

    # For conv 5, center of receptive field of i is i_0 = 16 i for 1-based indexing
    imgx, imgy = image_dims
    x, y, w, h = rect
    # scale to 224 x 224, standard input size.
    x1, y1 = ceil((x + w) * 224 / imgx), ceil((y + h) * 224 / imgy)
    x, y = floor(x * 224 / imgx), floor(y * 224 / imgy)
    px = _nnb_1(x + 1) - 1 # inclusive
    py = _nnb_1(y + 1) - 1 # inclusive
    px1 = _nnb_1(x1 + 1) # exclusive
    py1 = _nnb_1(y1 + 1) # exclusive

    return [px, py, px1, py1]


# For each category, extract features for n positive bboxes and 2n negative bboxes
def extract_features_cat(catIndex, labelForBboxAllImage, imageIdAndBbox, img_ids, feats, dataDir, dataType):
    annFile = '{}/annotations/instances_{}.json'.format(dataDir, dataType)

    coco = COCO(annFile)
    positiveNum = len(np.where(labelForBboxAllImage[:, catIndex] == 1.0)[0])
    
    positiveImageAndBbox = labelForBboxAllImage[np.where(labelForBboxAllImage[:, catIndex] == 1.0)[0], 18:20]
    
    if dataType != "test2014":
        negativeImageAndBbox = labelForBboxAllImage[random.sample(np.where(labelForBboxAllImage[:, catIndex] != 1.0)[0].tolist(), 2*positiveNum), 18:20]
    else:
        negativeImageAndBbox = labelForBboxAllImage[np.where(labelForBboxAllImage[:, catIndex] != 1.0)[0], 18:20]
    
    featurizer = Featurizer()
    
    featureLabels = np.zeros([3*positiveNum, 11777])
    for k in np.arange(positiveNum):
        i = positiveImageAndBbox[k, 0]
        j = positiveImageAndBbox[k, 1]
        
        idx1 = imageIdAndBbox[0].index(i)
        bbox = imageIdAndBbox[1][idx1][int(j)]
        
        # load image
        img = coco.loadImgs([i])[0]  # make sure image ID exists in the dataset given to you.
        img_pil = Image.open('%s/%s/%s'%(dataDir, dataType, img['file_name']))  # make sure data dir is correct
    
        projectedBbox = project_onto_feature_space(bbox, img_pil.size)
        
        idx2 = img_ids.index(i)
        img_feats = feats[idx2]
        bbox_feats = featurizer.featurize(projectedBbox, img_feats)
        
        featureLabels[k, 0:-1], featureLabels[k, -1] = bbox_feats, 1
    
    for m in np.arange(2 * positiveNum):
        i2 = negativeImageAndBbox[m, 0]
        j2 = negativeImageAndBbox[m, 1]
        
        idx3 = imageIdAndBbox[0].index(i2)
        bbox2 = imageIdAndBbox[1][idx3][int(j2)]
        
        # load image
        img2 = coco.loadImgs([i2])[0]  # make sure image ID exists in the dataset given to you.
        img_pil2 = Image.open('%s/%s/%s'%(dataDir, dataType, img2['file_name']))  # make sure data dir is correct
    
        projectedBbox2 = project_onto_feature_space(bbox2, img_pil2.size)
        
        idx4 = img_ids.index(i2)
        img_feats2 = feats[idx4]
        bbox_feats2 = featurizer.featurize(projectedBbox2, img_feats2)
        
        featureLabels[k+m+1, 0:-1], featureLabels[k+m+1, -1] = bbox_feats2, 0
    
    return featureLabels


def extract_n_negative(catIndex, labelForBboxAllImage, imageIdAndBbox, img_ids, feats, dataDir, dataType):
    annFile = '{}/annotations/instances_{}.json'.format(dataDir, dataType)

    coco = COCO(annFile)
    positiveNum = len(np.where(labelForBboxAllImage[:, catIndex] == 1.0)[0])

    negativeImageAndBbox = labelForBboxAllImage[random.sample(np.where(labelForBboxAllImage[:, catIndex] != 1.0)[0].tolist(), positiveNum), 18:20]
    
    featurizer = Featurizer()

    featureLabels = np.zeros([positiveNum, 11776])

    for m in np.arange(positiveNum):
        i = negativeImageAndBbox[m, 0]
        j = negativeImageAndBbox[m, 1]

        idx = imageIdAndBbox[0].index(i)
        bbox = imageIdAndBbox[1][idx][int(j)]

        # load image
        img = coco.loadImgs([i])[0]  # make sure image ID exists in the dataset given to you.
        img_pil = Image.open('%s/%s/%s' % (dataDir, dataType, img['file_name']))  # make sure data dir is correct

        projectedBbox = project_onto_feature_space(bbox, img_pil.size)

        idx = img_ids.index(i)
        img_feats = feats[idx]
        bbox_feats = featurizer.featurize(projectedBbox, img_feats)

        featureLabels[m, :]= bbox_feats

    return featureLabels


def classifier(dataDir, categoryId, lambd, learnrt, batchsz, loader,iternum=5):
    # create train/validation/test data set for classifier

    print("creating train data set...")
    trainDat = extract_features_cat(categoryId, loader.labelForBboxAllImageTrain, loader.imageIdAndBboxTrain, loader.img_idsTrain, loader.featsTrain, dataDir, "train2014")
    print("create train data set successfully...")
    
    print("creating val data set...")
    cvDat = extract_features_cat(categoryId, loader.labelForBboxAllImageCv, loader.imageIdAndBboxCv, loader.img_idsCv, loader.featsCv, dataDir, "val2014")
    print("create val data set successfully...")
    
    features_train = trainDat[:, 0:-1]
    img_cat_train = trainDat[:, -1] 
    # features_test = testDat[:, 0:-1]
    # img_cat_test = testDat[:, -1]
    features_cv = cvDat[:, 0:-1]
    img_cat_cv = cvDat[:, -1]
    del trainDat, cvDat
    gc.collect()
    
    # ini_w = np.random.normal(0, 1, [features_train.shape[1], 1]) / 100.
    ini_w = np.zeros([features_train.shape[1], 1])
    
    trainLoss = []
    cvLoss = []
    trainApscore = []
    cvApscore = []
    weight = []
    
    # hard negative mining
    for i in np.arange(iternum):
        if i==0:
            lsvm = LinearSVM(lambd, learnrt, batchsz, ini_w, features_train.shape[0])
        else:
            lsvm = LinearSVM(lambd, learnrt, batchsz, lsvm.w.data.numpy(), features_train.shape[0])
        
        lsvm.train(features_train, img_cat_train, features_cv, img_cat_cv, "HB")
        
        trainLoss = trainLoss + lsvm.trainLoss
        cvLoss = cvLoss + lsvm.cvLoss
        trainApscore = trainApscore + lsvm.trainAPScore
        cvApscore = cvApscore + lsvm.cvAPScore
        weight.append(lsvm.w.data.numpy())
        
        if i != iternum-1:
            lsvm.predict(features_train[np.where(img_cat_train == 0), :][0], img_cat_train[np.where(img_cat_train == 0)])
            
            # create new train data set for classifier
            print("creating new train data set...")
            features_train_new2 = extract_n_negative(categoryId, loader.labelForBboxAllImageTrain, loader.imageIdAndBboxTrain, loader.img_idsTrain, loader.featsTrain, dataDir, "train2014")

            features_train = np.vstack((features_train[np.where(img_cat_train == 1), :][0], np.vstack((features_train[np.where(img_cat_train == 0), :][0][lsvm.topIndex, :], features_train_new2))))
            del features_train_new2
            gc.collect()
            print("create new train data set successfully...")
            
    result = [trainLoss, cvLoss, trainApscore, cvApscore, weight]
        
    with open(os.path.join(dataDir, 'result', 'resultforcategory_{}.p'.format(categoryId)), 'wb') as fpout:
        cPickle.dump(result, fpout)
    fpout.close()
    

# For each category, extract features for n positive bboxes and 2n negative bboxes
def calculate_test_ap(dataDir, coco, hardnum):
    print("loading labels for bounding box...")
    with open(os.path.join(dataDir, 'bboxes', 'labelForBboxAllImage_test2014.p'), 'rb') as fp:
        labelForBboxAllImageTest = cPickle.load(fp)
    fp.close()
    print("load labels for bboxes successfully...")
    
    labelForBboxAllImageTest = np.array(labelForBboxAllImageTest)
    
    # load image ids and the candidate bounding box respectively
    print("loading image ids and the candidate bounding box respectively...")
    imageIdAndBboxTest = cPickle.load(open(os.path.join(dataDir, 'bboxes', 'test2014_bboxes.p'), 'rb'), encoding='latin1')
    print("load image ids and the candidate bounding box successfully...")
    
    # load features
    print("loading features...")
    [img_idsTest, featsTest] = cPickle.load(open(os.path.join(dataDir, 'features_small', 'test2014.p'),'rb'), encoding='latin1')
    print("load features successfully...")
    
    featurizer = Featurizer()
    
    maxnum = labelForBboxAllImageTest.shape[0]
    stepnum = 0
    stepsz = 2000
        
    wx = np.zeros([maxnum, 18])
    while stepnum*stepsz < maxnum:
        samplenum = np.min([stepsz, maxnum - stepnum*stepsz])
        
        TestImageAndBbox = labelForBboxAllImageTest[(stepnum*stepsz):(stepnum*stepsz+samplenum) , 18:20]
        
        featureLabels = np.zeros([samplenum, 11776])
        for k in np.arange(samplenum): 
            i = TestImageAndBbox[k, 0]
            j = TestImageAndBbox[k, 1]
            
            idx1 = imageIdAndBboxTest[0].index(i)
            bbox = imageIdAndBboxTest[1][idx1][int(j)]
            
            # load image
            img = coco.loadImgs([i])[0]  # make sure image ID exists in the dataset given to you.
            img_pil = Image.open('%s/%s/%s'%(dataDir, "test2014", img['file_name']))  # make sure data dir is correct
        
            projectedBbox = project_onto_feature_space(bbox, img_pil.size)
        
            idx2 = img_idsTest.index(i)
            img_feats = featsTest[idx2]
            bbox_feats = featurizer.featurize(projectedBbox, img_feats)
        
            featureLabels[k, :] = bbox_feats

        for categoryId in np.arange(18):
            with open(os.path.join(dataDir, 'result', 'resultforcategory_{}.p'.format(categoryId)), 'rb') as fpin:
                result = cPickle.load(fpin)
                fpin.close()
        
            wx[(stepnum*stepsz):(stepnum*stepsz+samplenum), categoryId] = featureLabels.dot(result[-1][hardnum])[:,0]
        
        stepnum += 1
        print(stepnum)
    
    testaps = np.zeros(18)
    for i in np.arange(18):
        testaps[i] = average_precision_score(labelForBboxAllImageTest[:, i], wx[:, i])
    
    with open(os.path.join(dataDir, 'result', 'testaps_{}.p'.format(hardnum)), 'wb') as fpout:
        cPickle.dump(testaps, fpout)
    fpout.close()
    
    # return testaps
    
# data and annotation directories
dataDir = 'E:/U Washington/Study/STAT 548(Machine Learning for Big Data)/Data'  # MAKE SURE THIS IS CORRECT!!!!
# dataType = 'val2014'
# dataType = 'test2014'
# dataType = 'train2014'

# ---------------------------------- train 19 classifier ---------------------------------- #
loader = load_dataset(dataDir)

time_start=time.time()
pool = ThreadPool(3)
results = pool.map(lambda catnum: classifier(dataDir, catnum, 500, 5e-8, 16, loader, 5), [i for i in [0, 3, 4, 5, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 2, 6, 1, 8]])
pool.close()
pool.join()
print('Time spent: ', time.time()-time_start, 's')


# -------------------------------- calculate test ap score ------------------------------#
annFile = '{}/annotations/instances_{}.json'.format(dataDir, "test2014")
coco = COCO(annFile)

for hardnum in np.arange(5):
    calculate_test_ap(dataDir, coco, hardnum)

# show test apscore
testaps = []
for hardnum in np.arange(5):
    with open(os.path.join(dataDir, 'result', 'testaps_{}.p'.format(hardnum)), 'rb') as fpin:
        testaps.append(cPickle.load(fpin))
    fpin.close()

# ------------------------------------ making plots ------------------------------------#
for categoryId in np.arange(18):
    with open(os.path.join(dataDir, 'result', 'resultforcategory_{}.p'.format(categoryId)), 'rb') as fpin:
        result = cPickle.load(fpin)
    fpin.close()

    fig = plt.figure(categoryId)
    fig.set_size_inches(16, 5)
    plt.subplot(1, 2, 1)
    train_loss_plt, = plt.plot(np.arange(0.5, len(result[0])*0.5+0.5, 0.5), result[0])
    cv_loss_plt, = plt.plot(np.arange(0.5, len(result[1])*0.5+0.5, 0.5), result[1])

    plt.xlabel('epoch of dataset')
    plt.ylabel('loss')
    plt.legend([train_loss_plt, cv_loss_plt], ["train loss", "cv loss"])
    plt.title("Loss value of category " + str(categoryId))

    plt.subplot(1, 2, 2)
    train_aps_plt, = plt.plot(np.arange(0.5, len(result[2])*0.5+0.5, 0.5), result[2])
    cv_aps_plt, = plt.plot(np.arange(0.5, len(result[3])*0.5+0.5, 0.5), result[3])

    plt.xlabel('epoch of dataset')
    plt.ylabel('apscore')
    plt.legend([train_aps_plt, cv_aps_plt], ["train ap score", "cv ap score"])
    plt.title("Ap score of category " + str(categoryId))

    fig.savefig('plt_{}.png'.format(categoryId), bbox_inches='tight')
