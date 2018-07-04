These codes are about the CSE 547/ STAT 548 homework 3.

Before you run the script: linear-SVM.py, you need to use feature-label-matrix.py to prepare 3 pickle data files.
The steps are shown below.

1. Change the "dataDir" in read_coco.py to the path that you put your data file on your own computer
2. Uncommand "dataType" to access to train data set to generate the labelForBboxAllImage_train2014.p files
3. Uncommand "dataType" to access to test data set to generate the labelForBboxAllImage_test2014.p files, rerun feature-label-matrix.py
4. Uncommand "dataType" to access to cv data set to generate the labelForBboxAllImage_val2014.p files, rerun feature-label-matrix.py
5. Let the "path" in linear-SVM.py to be the same as the "dataDir" in feature-label-matrix.py
6. Run linear-SVM.py and we can get the results.