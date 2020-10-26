
import numpy as np
import matplotlib.pyplot as plt
from implementations import *
from preprocessing import *
from train_tune import *
from proj1_helpers import *
#import pandas as pd
import seaborn as sns

print("::: LOADING DATA :::\n")
DATA_TRAIN_PATH = '../data/train.csv' # TODO: download train data and supply path here 
y, tx_train, ids = load_csv_data(DATA_TRAIN_PATH)
names= get_feature_names(DATA_TRAIN_PATH)
name2num,num2name = mapping(DATA_TRAIN_PATH)
#Train preprocessing
tx0, y0, tx1, y1, tx2, y2, tx3, y3, id0, id1, id2, id3 =cluster_preprocessing_train(tx_train,y,num2name,f="median")
degrees = [1,2,3,4]
lambdas = np.logspace(-9,-3,5)
gamma = 2e-6
k_fold = 5
n_iters = 1250
SAVEPATH = "../output/xd/"

file1 = open(SAVEPATH+"bestparamsregulogfinalscript.txt","w")
tol = 1e-8
print("#======== CV for Cluster 0 ========#")
wlog0, dlog0, la0, train0, test0 = crossval_regulog_gridsearch(y0,tx0,k_fold,
                                                             lambdas,degrees,
                                                             n_iters,gamma,loss=True,tol=tol)
cv_viz(dlog0,lambdas,train0[dlog0-1,:],test0[dlog0-1,:],save=SAVEPATH+"REGULOG_C0_LOGFTSFXD")
file1.write("Cluster 0 : best deg {}. lam {:e} \n".format(dlog0,la0))
#----------------1
print("#======== CV for Cluster 1 ========#")
wlog1, dlog1, la1, train1, test1 = crossval_regulog_gridsearch(y1,tx1,k_fold,
                                                             lambdas,degrees,
                                                             n_iters,gamma,loss=True,tol=tol)
cv_viz(dlog1,lambdas,train1[dlog1-1,:],test0[dlog1-1,:],save=SAVEPATH+"REGULOG_C1_LOGFTSFXD")
file1.write("Cluster 1 : best deg {}. lam {:e} \n".format(dlog1,la1))

#----------------2
print("#======== CV for Cluster 2 ========#")
wlog2, dlog2, la2, train2, test2 = crossval_regulog_gridsearch(y2,tx2,k_fold,
                                                             lambdas,degrees,
                                                             n_iters,gamma,loss=True,tol=tol)
cv_viz(dlog2,lambdas,train2[dlog2-1,:],test0[dlog2-1,:],save=SAVEPATH+"REGULOG_C2_LOGFTSFXD")
file1.write("Cluster 2 : best deg {}. lam {:e} \n".format(dlog2,la2))

#----------------3
print("#======== CV for Cluster 3 ========#")
wlog3, dlog3, la3, train3, test3 = crossval_regulog_gridsearch(y3,tx3,k_fold,
                                                             lambdas,degrees,
                                                             n_iters,gamma,loss=True,tol=tol)
cv_viz(dlog3,lambdas,train3[dlog3-1,:],test0[dlog3-1,:],save=SAVEPATH+"REGULOG_C3_LOGFTSFXD")
file1.write("Cluster 3 : best deg {}. lam {:e} \n".format(dlog3,la3))

file1.close()
degs=[dlog0,dlog1,dlog2,dlog3]

DATA_TEST_PATH = '../data/test.csv' # TODO: download test data and supply path here 
_, tX_test, ids_test = load_csv_data(DATA_TEST_PATH)
OUTPUT_PATH = '../finalsubmit/' 

test0, i0, test1, i1, test2, i2, test3, i3 = cluster_preprocessing_test(tX_test, id0, id1,
                                                                 id2, id3, degs, num2name, how="median")
#Prediction
file1.close()
yclusterpred_log = cluster_predict(wlog0,wlog1,wlog2,wlog3,
                               test0,test1,test2,test3,
                               i0,i1,i2,i3,how="log")

create_csv_submission(ids_test, yclusterpred_log, SAVEPATH+"REGULOGFINAL_LOGFIXED_MEDIAN.csv")