
import numpy as np
import matplotlib.pyplot as plt
from implementations import *
from preprocessing import *
from train_tune import *
from proj1_helpers import *
#import pandas as pd

DATA_TRAIN_PATH = '../data/train.csv' # TODO: download train data and supply path here 

#loading data
print("::: LOADING DATA :::\n")
y, tx_train, ids = load_csv_data(DATA_TRAIN_PATH)
names= get_feature_names(DATA_TRAIN_PATH)
name2num,num2name = mapping(DATA_TRAIN_PATH)
#Train preprocessing
tx0, y0, tx1, y1, tx2, y2, tx3, y3, id0, id1, id2, id3 = cluster_preprocessing_train(tx_train,y,num2name)

#K-fold CV
k_fold=5
#USING FEWER ITERATIONS DUE TO HOW MASSIVE THIS GRIDSEARCH IS.
n_iters = 1500
gamma = 2.2e-6
lambdas = np.logspace(-7,2,4)
degrees = np.array(range(1,3))
file1 = open("bestparams_regulogANOTHER.txt","w") 

print("#======== CV for Cluster 0 ========#")
wlog0, dlog0, la0, train0, test0 = crossval_regulog_gridsearch(y0,tx0,k_fold,
                                                             lambdas,degrees,
                                                             n_iters,gamma,loss=True,tol=1e-6)
cv_viz(dlog0,lambdas,train0[dlog0-1,:],test0[dlog0-1,:],save="regulog_clust0")
#----------------1
print("#======== CV for Cluster 1 ========#")
wlog1, dlog1, la1, train1, test1 = crossval_regulog_gridsearch(y1,tx1,k_fold,
                                                             lambdas,degrees,
                                                             n_iters,gamma,loss=True,tol=1e-6)
cv_viz(dlog1,lambdas,train1[dlog1-1,:],test0[dlog1-1,:],save="regulog_clust1")

#----------------2
print("#======== CV for Cluster 2 ========#")
wlog2, dlog2, la2, train2, test2 = crossval_regulog_gridsearch(y2,tx2,k_fold,
                                                             lambdas,degrees,
                                                             n_iters,gamma,loss=True,tol=1e-6)
cv_viz(dlog2,lambdas,train2[dlog2-1,:],test0[dlog2-1,:],save="regulog_clust2")

#----------------3
print("#======== CV for Cluster 3 ========#")
wlog3, dlog3, la3, train3, test3 = crossval_regulog_gridsearch(y3,tx3,k_fold,
                                                             lambdas,degrees,
                                                             n_iters,gamma,loss=True,tol=1e-6)
cv_viz(dlog3,lambdas,train3[dlog3-1,:],test0[dlog3-1,:],save="regulog_clust3")

degs=[dlog0,dlog1,dlog2,dlog3]
####

print("TEST DATA")
DATA_TEST_PATH = '../data/test.csv' # TODO: download test data and supply path here 
_, tX_test, ids_test = load_csv_data(DATA_TEST_PATH)
OUTPUT_PATH = '../submissions/' # TODO: fill in desired name of output file for submission

#Test preprocessing
test0, i0, test1, i1, test2, i2, test3, i3 = cluster_preprocessing_test(tX_test, id0, id1,
                                                                 id2, id3, degs, num2name)
#Prediction

yclusterpred_log = cluster_predict(wlog0,wlog1,wlog2,wlog3,
                               test0,test1,test2,test3,
                               i0,i1,i2,i3,how="log")

create_csv_submission(ids_test, yclusterpred_log, OUTPUT_PATH+"CV_GRIDSEARCH_regulog.csv")


best = "degs = {},{},{},{} \n lambdas = {:e}, \n {:e}, \n {:e} \n {:e}".format(dlog0,dlog1,dlog2,dlog3,la0,la1,la2,la3)
file1.write(best)
file1.close()