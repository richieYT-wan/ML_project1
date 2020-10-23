import numpy as np
import matplotlib.pyplot as plt
from implementations import *
from preprocessing import *
from train_tune import *
from proj1_helpers import *

print("::: ALTERNATIVE DATA PROCESSING :::\n")
y, tx_train, ids = load_csv_data(DATA_TRAIN_PATH)
names= get_feature_names(DATA_TRAIN_PATH)
name2num,num2name = mapping(DATA_TRAIN_PATH)
#Train preprocessing
tx0, y0, tx1, y1, tx2, y2, tx3, y3, id0, id1= cluster_preprocessing_train_alt(tx_train,y,num2name)
degs = [1,2,3,4]
tx0 = build_poly(tx0, 1)
tx1 = build_poly(tx1, 2)
tx2 = build_poly(tx2, 3)
tx3 = build_poly(tx3, 4)

l0, l1, l2, l3 = 4.000000e-05, 1.000000e-02, 2.000000e-08, 3.000000e-03

w_opt0 = ridge_regression(y0, tx0, l0)
w_opt1 = ridge_regression(y1, tx1, l1)
w_opt2 = ridge_regression(y2, tx2, l2)
w_opt3 = ridge_regression(y3, tx3, l2)

print("TEST DATA")
DATA_TEST_PATH = '../data/test.csv' # TODO: download test data and supply path here 
_, tX_test, ids_test = load_csv_data(DATA_TEST_PATH)
OUTPUT_PATH = '../submissions/' # TODO: fill in desired name of output file for submission

#Test preprocessing
test0, i0, test1, i1, test2, i2, test3, i3 = cluster_preprocessing_test_alt(tX_test, id0, id1, degs, num2name)
#Prediction

yclusterpred_opt = cluster_predict(w_opt0,w_opt1,w_opt2,w_opt3,
                               test0,test1,test2,test3,
                               i0,i1,i2,i3,how="normal")

print("Submission created")
create_csv_submission(ids_test, yclusterpred_opt, OUTPUT_PATH+"CV_GRIDSEARCH_alt_ridge.csv")
