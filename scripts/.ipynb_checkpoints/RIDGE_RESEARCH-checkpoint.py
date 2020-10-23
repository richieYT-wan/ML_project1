
import numpy as np
import matplotlib.pyplot as plt
from implementations import *
from preprocessing import *
from train_tune import *
from proj1_helpers import *

DATA_TRAIN_PATH = '../data/train.csv' # TODO: download train data and supply path here 

#loading data
print("::: ALTERNATIVE DATA PROCESSING :::\n")
y, tx_train, ids = load_csv_data(DATA_TRAIN_PATH)
names= get_feature_names(DATA_TRAIN_PATH)
name2num,num2name = mapping(DATA_TRAIN_PATH)
#Train preprocessing
tx0, y0, tx1, y1, tx2, y2, tx3, y3, id0, id1, id2,id3= cluster_preprocessing_train(tx_train,y,num2name)

#K-fold CV
k_fold=10

lambdas = np.logspace(-9,-1,11)
degrees = np.array(range(1,10))

print("#======== CV for Cluster 0 ========#")
w_opt0, d_opt0, la0, train0, test0 = crossval_ridge_gridsearch(y0,tx0,k_fold,
                                                             lambdas,degrees,
                                                             loss=True)
cv_viz(d_opt0,lambdas,train0[d_opt0-1,:],test0[d_opt0-1,:],save="ridge_clust0REM0")
#----------------1
print("#======== CV for Cluster 1 ========#")
w_opt1, d_opt1, la1, train1, test1 = crossval_ridge_gridsearch(y1,tx1,k_fold,
                                                             lambdas,degrees,
                                                             loss=True)
cv_viz(d_opt1,lambdas,train1[d_opt1-1,:],test0[d_opt1-1,:],save="ridge_clust1REM0")

#----------------2
print("#======== CV for Cluster 2 ========#")
w_opt2, d_opt2, la2, train2, test2 = crossval_ridge_gridsearch(y2,tx2,k_fold,
                                                             lambdas,degrees,
                                                             loss=True)
cv_viz(d_opt2,lambdas,train2[d_opt2-1,:],test0[d_opt2-1,:],save="ridge_clust2REM0")

#----------------3
print("#======== CV for Cluster 3 ========#")
w_opt3, d_opt3, la3, train3, test3 = crossval_ridge_gridsearch(y3,tx3,k_fold,
                                                             lambdas,degrees,
                                                             loss=True)
cv_viz(d_opt3,lambdas,train3[d_opt3-1,:],test0[d_opt3-1,:],save="ridge_clust3REM0")

degs=[d_opt0,d_opt1,d_opt2,d_opt3]
####

print("TEST DATA")
DATA_TEST_PATH = '../data/test.csv' # TODO: download test data and supply path here 
_, tX_test, ids_test = load_csv_data(DATA_TEST_PATH)
OUTPUT_PATH = '../submissions/' # TODO: fill in desired name of output file for submission

#Test preprocessing
test0, i0, test1, i1, test2, i2, test3, i3 = cluster_preprocessing_test(tX_test, id0, id1,id2,id3 degs, num2name)
#Prediction

yclusterpred_opt = cluster_predict(w_opt0,w_opt1,w_opt2,w_opt3,
                               test0,test1,test2,test3,
                               i0,i1,i2,i3,how="normal")

create_csv_submission(ids_test, yclusterpred_opt, OUTPUT_PATH+"CV_GRIDSEARCH_alt_ridge.csv")


best = "degs = {},{},{},{} \n lambdas = {:e}, \n {:e}, \n {:e} \n {:e}".format(d_opt0,d_opt1,d_opt2,d_opt3,la0,la1,la2,la3)
file1 = open("bestparams_ridgeREMOVE0.txt","w") 
file1.write(best)
file1.close()