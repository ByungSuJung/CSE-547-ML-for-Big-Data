These codes are about the CSE 547/ STAT 548 final report.

Before you run the script: linear-SVM.py, you need to use feature-label-matrix.py and 
feature-label-matrix-lsh.py to prepare 6 pickle data files.
The steps are shown below.

1. Change the "dataDir" in read_coco.py to the path that you put your data file on your own computer
2. Uncommand "dataType" to access to train data set to generate the labelForBboxAllImage_train2014.p files
3. Uncommand "dataType" to access to test data set to generate the labelForBboxAllImage_test2014.p files, rerun feature-label-matrix.py
4. Uncommand "dataType" to access to cv data set to generate the labelForBboxAllImage_val2014.p files, rerun feature-label-matrix.py
5. repeat 2-4 steps for feature-label-matrix-lsh.py
6. Let the "path" in linear-SVM.py and LSH.py to be the same as the "dataDir" in feature-label-matrix.py
7. Run linear-SVM.py and LSH.py and we can get the results.