# -*- coding: utf-8 -*-
"""some helper functions for project 1."""
import csv
import numpy as np
import matplotlib.pyplot as plt
from costs import sigmoid

def load_csv_data(data_path, sub_sample=False):
    """Loads data and returns y (class labels), tX (features) and ids (event ids)"""
    y = np.genfromtxt(data_path, delimiter=",", skip_header=1, dtype=str, usecols=1)
    x = np.genfromtxt(data_path, delimiter=",", skip_header=1)
    ids = x[:, 0].astype(np.int)
    input_data = x[:, 2:]

    # convert class labels from strings to binary (-1,1)
    yb = np.ones(len(y))
    yb[np.where(y=='b')] = -1
    
    # sub-sample
    if sub_sample:
        yb = yb[::50]
        input_data = input_data[::50]
        ids = ids[::50]

    return yb, input_data, ids


def predict_labels(weights, data):
    """Generates class predictions given weights, and a test data matrix"""
    y_pred = np.dot(data, weights)
    y_pred[np.where(y_pred <= 0)] = -1
    y_pred[np.where(y_pred > 0)] = 1
    
    return y_pred

def predict_labels_log(weights, data):
    """Implements the quantize step in logistic regression.
    Generates class predictions given weights obtained from logistic regression
    , and a test data matrix. Prediction is based on the probability of y = 1 
    given x (data) and w (weights) with p(y=1|x)= sigmoid(x.T.dot(w))
    being more/less than 0.5."""
    y_pred = sigmoid(data.dot(weights))
    #fancy indexing
    y_pred[y_pred >0.5] = 1
    y_pred[y_pred <=0.5] = -1
    
    return np.squeeze(y_pred)

def create_csv_submission(ids, y_pred, name):
    """
    Creates an output file in csv format for submission to kaggle
    Arguments: ids (event ids associated with each prediction)
               y_pred (predicted class labels)
               name (string name of .csv output file to be created)
    """
    with open(name, 'w') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({'Id':int(r1),'Prediction':int(r2)})

            
def histogram_clusters(tx0,tx1,tx2,tx3,data_path="../data/train.csv"):
    """
    Function to visualize the distribution of all features after having been clustered. 
    Input : datamatrix of each cluster (tx0 to tx3)
    
    """
    fig, axes = plt.subplots(10,3,figsize=(8,20),sharex=False,sharey =False,)
    a = axes.ravel()
    #Getting the header.
    x = np.genfromtxt(data_path, delimiter = ",", skip_header = 0,
                 names = True,max_rows=1)
    header = x.dtype.names[2:]
    for idx,ax in enumerate(a[:]):        
        
        ax.hist(tx0[:,idx],bins=500,stacked=True, histtype='stepfilled',alpha=0.5, label = "Cluster 0", color="cornflowerblue")
        ax.hist(tx1[:,idx],bins=500,stacked=True, histtype='stepfilled',alpha=0.5, label = "Cluster 1", color = "red")
        ax.hist(tx2[:,idx],bins=500,stacked=True, histtype='stepfilled',alpha=0.5, label = "Cluster 2", color="cyan")
        ax.hist(tx3[:,idx],bins=500,stacked=True, histtype='stepfilled',alpha=0.5, label = "Cluster 3", color ="purple")
        #ax.set_adjustable(self, adjustable, share=False)
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(loc='best',fontsize="x-small")
        ax.autoscale()

        ax.set_title(header[idx])
    plt.tight_layout() 
    plt.show()

