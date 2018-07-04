#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 14 22:48:26 2018

@author: yangjh39
"""

from __future__ import division
import os, sys, time
import _pickle as cPickle
import itertools

from pycocotools.coco import COCO
import numpy as np
import pandas as pd
import skimage.io as io
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = (8.0, 10.0)

import cv2
from IPython.display import set_matplotlib_formats
set_matplotlib_formats('jpg')
from PIL import Image, ImageDraw

# data and annotation directories
dataDir = '/home/yangjh39/548/data'  #### TODO: Ensure this is correct

dataType='val2014' # uncomment to access the validation set
# dataType='train2014' # uncomment to access the train set
# dataType='test2014' # uncomment to access the test set
annFile='{}/annotations/instances_{}.json'.format(dataDir,dataType) # annotations

# directory structure for this demo
print('Expected directory structure:')
print('-'*60)
for path, dirs, files in os.walk(dataDir):
    if path.split("/")[-1] != '.ipynb_checkpoints': 
        # do not disply jupyter related files
        print(path)
    if path.split("/")[-1] in ['features_small', 'features_tiny']:
        for f in files:
            print('-'*8, f)

# initialize COCO api for instance annotations. This step takes several seconds each time.
coco=COCO(annFile)  

cats = coco.loadCats(coco.getCatIds()) # categories
cat_id_to_name = {cat['id']: cat['name'] for cat in cats} # category id to name mapping
cat_name_to_id = {cat['name']: cat['id'] for cat in cats} # category name to id mapping

cat_to_supercat = {cat['name']: cat['supercategory'] for cat in cats}
cat_id_to_supercat = {cat['id']: cat['supercategory'] for cat in cats}

# print supercategory and categories in each supercategory
supercat_to_cats = {}
for key, group in itertools.groupby(sorted([(sc, c) for (c, sc) in cat_to_supercat.items()]), lambda x: x[0]):
    lst = [thing[1] for thing in group]
    print(key, ":", '{1}{0}'.format("\n----".join(lst), "\n----"), '\n' )
    supercat_to_cats[key] = lst

# Load extacted features:
t1 = time.time()
[img_list, feats] = cPickle.load(open(os.path.join(dataDir, 'features_small', '{}.p'.format(dataType)),'rb'),encoding='latin1')
print('time to load features =', time.time() - t1, 'sec')
print('num images =', len(img_list))
print('shape of features =', feats.shape)


# Find supercategories associated with each image
img_supercat = []

for i in np.arange(len(img_list)):
    img_id = img_list[i]  # any image ID to find supercategory
    annIds = coco.getAnnIds(imgIds=img_id, iscrowd=None)
    anns = coco.loadAnns(annIds)

    categories = set([ann['category_id'] for ann in anns])
    supercategories = set([cat_id_to_supercat[ann['category_id']] for ann in anns])

    img_supercat.append(list(supercategories)[0])

    print(supercategories) # expect singleton

pd.DataFrame(img_supercat).to_csv(dataDir + "/features_small/img_supercat_" + dataType + ".csv")
pd.DataFrame(img_list).to_csv(dataDir + "/features_small/img_list_" + dataType + ".csv")
feats = feats.reshape(feats.shape[0], -1)
pd.DataFrame(feats.reshape(feats.shape[0], -1)).to_csv(dataDir + "/features_small/feats_" + dataType + ".csv")

