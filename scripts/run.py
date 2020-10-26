import numpy as np
import matplotlib.pyplot as plt
from implementations import *
from preprocessing import *
from train_tune import *
from proj1_helpers import *


#loading data
try:
      DATA_TRAIN_PATH = '../data/train.csv'
      y, tx_train, ids = load_csv_data(DATA_TRAIN_PATH)
      print("#===========::LOADING TRAIN DATA::===========#")
except: 
      print("\nTHE FILE : train.csv HAS NOT BEEN FOUND IN ../data/ \nPlease make sure you unzipped it.\n")
names= get_feature_names(DATA_TRAIN_PATH)
name2num,num2name = mapping(DATA_TRAIN_PATH)
#Train preprocessing
tx0, y0, tx1, y1, tx2, y2, tx3, y3, id0, id1, id2, id3, means, stds = cluster_preprocessing_train(tx_train,y,f="median")
SAVEPATH = "../output/"

#K-fold CV
print("#===========::10-fold Crossvalidation::===========# \n")
k_fold=10
lambdas = np.logspace(-12,-3,9)
degrees = np.array(range(1,15))

print("#===========::CV for Cluster 0::===========#")
file1 = open(SAVEPATH+"ridgecrossvalidation_params.txt","w")

w_opt0, d_opt0, la0= crossval_ridge_gridsearch(y0,tx0,k_fold,
                                                             lambdas,degrees,
                                                             loss=False)
file1.write("Cluster 0 : best deg {}. lam {:e} \n".format(d_opt0,la0))

#----------------1
print("\n#===========::CV for Cluster 1::===========#")
w_opt1, d_opt1, la1 = crossval_ridge_gridsearch(y1,tx1,k_fold,
                                                             lambdas,degrees,
                                                             loss=False)
file1.write("Cluster 1 : best deg {}. lam {:e} \n".format(d_opt1,la1))

#----------------2
print("\n#===========::CV for Cluster 2::===========#")
w_opt2, d_opt2, la2= crossval_ridge_gridsearch(y2,tx2,k_fold,
                                                             lambdas,degrees,
                                                             loss=False)

file1.write("Cluster 2 : best deg {}. lam {:e} \n".format(d_opt2,la2))

#----------------3
print("\n#===========::CV for Cluster 3::===========#")
w_opt3, d_opt3, la3 = crossval_ridge_gridsearch(y3,tx3,k_fold,
                                                             lambdas,degrees,
                                                             loss=False)

file1.write("Cluster 3 : best deg {}. lam {:e} \n".format(d_opt3,la3))


degs=[d_opt0,d_opt1,d_opt2,d_opt3]
file1.close()

try:
      DATA_TEST_PATH = '../data/test.csv' # TODO: download test data and supply path here 
      _, tX_test, ids_test = load_csv_data(DATA_TEST_PATH)
      print("\n#===========::LOADING TEST DATA::===========# \n")
except: 
      print("The file test.csv has not been found in ../data/ \nPlease make sure you unzipped it.")

OUTPUT_PATH = '../output/' # TODO: fill in desired name of output file for submission
#Test preprocessing
test0, i0, test1, i1, test2, i2, test3, i3 = cluster_preprocessing_test(tX_test, id0, id1, id2, id3, means,stds, degs,f="median")
#Prediction
yclusterpred_opt = cluster_predict(w_opt0,w_opt1,w_opt2,w_opt3,
                               test0,test1,test2,test3,
                               i0,i1,i2,i3,how="normal")

create_csv_submission(ids_test, yclusterpred_opt, OUTPUT_PATH+"final_prediction.csv")
print("#===========::Prediction done::===========#")
