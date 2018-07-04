# -*- coding: utf-8 -*-
# @Time    : 2018/5/14 19:37
# @Author  : Jiahao Yang
# @Email   : yangjh39@uw.edu

from pycocotools.coco import COCO

import _pickle as cPickle
import os
import time
import numpy as np

# IoU
def iou(rect1, rect2): # rect = [x, y, w, h]
    x1, y1, w1, h1 = rect1
    X1, Y1 = x1+w1, y1 + h1
    x2, y2, w2, h2 = rect2
    X2, Y2 = x2+w2, y2 + h2
    a1 = (X1 - x1 + 1) * (Y1 - y1 + 1)
    a2 = (X2 - x2 + 1) * (Y2 - y2 + 1)
    x_int = max(x1, x2)
    X_int = min(X1, X2)
    y_int = max(y1, y2)
    Y_int = min(Y1, Y2)
    a_int = (X_int - x_int + 1) * (Y_int - y_int + 1) * 1.0
    if x_int > X_int or y_int > Y_int:
        a_int = 0.0
    return a_int / (a1 + a2 - a_int)

# data and annotation directories
dataDir = '/Users/yangjh39/Desktop/548/Data2'  # MAKE SURE THIS IS CORRECT!!!!
dataType = 'val2014'
# dataType = 'test2014'
# dataType = 'train2014'
annFile = '{}/annotations/instances_{}.json'.format(dataDir, dataType)

coco = COCO(annFile)

# load image ids and the candidate bounding box respectively
imageIdAndBbox = cPickle.load(open(os.path.join(dataDir, 'bboxes', '{}_bboxes.p'.format(dataType)), 'rb'), encoding='latin1')

# the indexes of 18 categories
catId = np.concatenate((np.arange(2, 10), np.arange(16, 26)))

# get the ground truth label for each bounding box in each image
labelForBboxAllImage = []
t = time.time()
print("start calculate ground truth label for each bounding box in each image...")
for i in np.arange(len(imageIdAndBbox[0])):
    annIds = coco.getAnnIds(imgIds=imageIdAndBbox[0][i], iscrowd=None)
    anns = coco.loadAnns(annIds)

    if imageIdAndBbox[1][i] is not None:
        for j in np.arange(len(imageIdAndBbox[1][i])):
            labelForBboxPerImage = np.zeros([20])
            labelForBboxPerImage[18:20] = imageIdAndBbox[0][i], j  # image id and bounding box id respectively

            for ann in anns:
                if iou(ann['bbox'], imageIdAndBbox[1][i][j].tolist()) > 0.5:
                    labelForBboxPerImage[np.where(catId == ann['category_id'])] = 1.0

            labelForBboxAllImage.append(labelForBboxPerImage)

print("get ground truth labels successfully...use " + str(time.time()-t) + "s.")

with open(os.path.join(dataDir, 'bboxes', 'labelForBboxAllImage_{}.p'.format(dataType)), 'wb') as fp:
    cPickle.dump(labelForBboxAllImage, fp)
fp.close()

