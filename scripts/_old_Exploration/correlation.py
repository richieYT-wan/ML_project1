import numpy as np
import matplotlib.pyplot as plt
from implementations import *
from preprocessing import *
from train_tune import *
from proj1_helpers import *

DATA_TRAIN_PATH = '../data/train.csv'

#loading data
print("#===========::LOADING TRAIN DATA::===========#")
y, tx_train, ids = load_csv_data(DATA_TRAIN_PATH)
names= get_feature_names(DATA_TRAIN_PATH)
name2num,num2name = mapping(DATA_TRAIN_PATH)
#Train preprocessing
tx0, _,tx1,_,_,_,_,_ = prijetnum_clustering(tx_train,y)
#tx0, y0, tx1, y1, tx2, y2, tx3, y3, id0, id1, id2, id3, means, stds = cluster_preprocessing_train(tx_train,y,f="median")
SAVEPATH = "../output/"

view_correlation(tx1, "cluster1", save="corr1")