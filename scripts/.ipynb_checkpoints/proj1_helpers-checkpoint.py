# -*- coding: utf-8 -*-
"""some helper functions for project 1.
Expanded over the base functions.
Implements functions to visualize data (features, output, etc.)
Main collaborator : Richie Yat-tsai Wan (258934)

"""
import csv
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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


def cluster_predict(w0,w1,w2,w3, tx0,tx1,tx2,tx3, i0,i1,i2,i3, how = "normal"):
    """
    Takes the clusterzied weights, sets, and IDs and performs a prediction, 
    re-merging them into a single array with the correct indexing.
    Input : w0, w1, w2, w3 (np.array) : Sets of weights for each cluster 
            tx0, tx1, tx2, tx3 (np.array) : Set of test data separated into clusters (see prijetnum_clustering in prepocessing.py)
            i0, i1, i2, i3 (indices) : Set of indices from the clusters (same as above)
            how (string) : normal or log, returns the prediction for a given method
    """
    if how == "normal":
        print("Normal prediction")
        ypred_0 = predict_labels(w0, tx0)
        ypred_1 = predict_labels(w1, tx1)
        ypred_2 = predict_labels(w2, tx2)
        ypred_3 = predict_labels(w3, tx3)
        ypred = np.ones(i0.shape[0])
        ypred[i0]=ypred_0
        ypred[i1]=ypred_1
        ypred[i2]=ypred_2
        ypred[i3]=ypred_3
        return ypred
    
    elif how == "log":
        print("Prediction for log regression")
        ypred_0 = predict_labels_log(w0, tx0)
        ypred_1 = predict_labels_log(w1, tx1)
        ypred_2 = predict_labels_log(w2, tx2)
        ypred_3 = predict_labels_log(w3, tx3)
        ypred = np.ones(i0.shape[0])
        ypred[i0]=ypred_0
        ypred[i1]=ypred_1
        ypred[i2]=ypred_2
        ypred[i3]=ypred_3        
        return ypred
    #ridge cluster prediction

def save_single_histogram(tx0,tx1,tx2,tx3,data_path,feat,bins=1000,
                     save=False,log=False,xlim=False,ylim=False):
    """
    helper function to get single histogram with cluster datas.
    Mostly used during feature selection, as to whether or not to take the log of a feature.
    Input : tx0, tx1, tx2, tx3 (np.array) : clusterized input data
            data_path (string) : used to get the feature names for the plot
            bins (int): bin size for the histogram
            save (string or False), proposed formatting : "_suffix", as the file is automatically saved as feat+number+suffx
            log  (bool) : shows the histogram/distribution of the logged input. (deprecated, used during exploration)
            xlim, ylim (lists of len 2, [min, max]) : sets the axis limit for x and y respectively
    """
    #SINGLEPLOTS
    fig, axes = plt.subplots(1,1,figsize=(15,15),sharex=False,sharey =False,)
    plt.rc('xtick',labelsize=18)
    plt.rc('ytick',labelsize=18)
    #Getting the header/feature names
    x = np.genfromtxt(data_path, delimiter = ",", skip_header = 0, names = True, max_rows=1)
    names = x.dtype.names[2:] 
    plot0 = tx0[:,feat]
    plot1 = tx1[:,feat]
    plot2 = tx2[:,feat]
    plot3 = tx3[:,feat]
    title = "Feat{},{},{}".format(feat,names[feat],save)
    if log:
        plot0 = np.log(plot0+0.001)
        plot1 = np.log(plot1+0.001)
        plot2 = np.log(plot2+0.001)
        plot3 = np.log(plot3+0.001)
        #title = title+"LOG"
        
    axes.hist(plot0,bins=bins,stacked=True, histtype='stepfilled',alpha=0.5, 
              label = "Cluster 0", color="cornflowerblue")
    axes.hist(plot1,bins=bins,stacked=True, histtype='stepfilled',alpha=0.5, 
              label = "Cluster 1", color = "red")
    axes.hist(plot2,bins=bins,stacked=True, histtype='stepfilled',alpha=0.5, 
              label = "Cluster 2", color="cyan")
    axes.hist(plot3,bins=bins,stacked=True, histtype='stepfilled',alpha=0.5, 
              label = "Cluster 3", color ="purple")
    axes.set_title(title, fontsize = 26,weight ="bold")
    if xlim:
        axes.set_xlim(xlim)
    if ylim:
        axes.set_ylim(ylim)
    handles, labels = axes.get_legend_handles_labels()
    axes.legend(loc='best',fontsize="x-large",prop={"weight":"bold","size":24})
    axes.autoscale()
    if xlim:
        axes.set_xlim(xlim)
    plt.tight_layout() 
    start="../output/histograms/"
    ends=".png"
    if save:
        featnum= str(feat)
        name = "feat"+featnum+save
        print("Saving under :",start+name+ends)
        plt.savefig(start+name+ends)
    plt.show()

def histogram_clusters(tx0,tx1,tx2,tx3,data_path="../data/train.csv", save = False):
    """
    Function to visualize the distribution of all features after having been clustered. 
    Input : tx0, tx1, tx2, tx3 (np.array) : Input data for each cluster
            data_path (string) :  Filepath of input, to read feature names
            save (string or False) : String specifying filepath if wanting to save the figure
    
    """
    fig, axes = plt.subplots(10,3,figsize=(20,38),sharex=False,sharey =False,)
    a = axes.ravel()
    #Getting the header/feature names
    x = np.genfromtxt(data_path, delimiter = ",", skip_header = 0, names = True, max_rows=1)
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

        ax.set_title("Feat:{},{} ".format(idx,header[idx]))
    plt.tight_layout() 
    if save and (type(save)==str): 
        print("Saving file under",save)
        plt.savefig(save)
    plt.show()
    

def view_correlation(tx, name, save=False):
    """
    Gets the correlation matrix and plots and saves it.
    Input : tx (np.array), data to view
            name (str), name of the clustered dataset (Can be cluster0, for example.)
            save (str or False), specifies whether to save the figure.
    """
    corr = np.corrcoef(tx.T)
    colors = sns.color_palette("icefire", as_cmap = True)
    plt.figure(figsize=(10,10))
    sns.heatmap(corr,cmap=colors, square=True)
    plt.title("Clustering w.r.t PRI_jet_num value. \n Correlation matrix for {}".format(save),weight=575)
    plt.xlabel("Feature number #")
    plt.xticks(rotation = "horizontal")
    plt.ylabel("Feature number #")
    if save:
        filename = "../output/corr"+name+".png"
        print("Saving under :"+filename)
        plt.savefig(filename)



        
def get_feature_names(data_path ="../data/train.csv"):
    """
    Get the names of features from the header.
    input : data_path (string) 
    output : names (np.array) : contains as string the name of each features
    """
    data = np.genfromtxt(data_path, delimiter = ",", skip_header = 0,
                         names = True,max_rows=1);
    names = np.array(data.dtype.names[2:])
    return names


def mapping(data_path ="../data/train.csv"):
    """
    Mapping of index number to and from feature names.
    output : returns two dictionaries mapping number to names and vice-versa
    """
    names = get_feature_names(data_path)
    numbers = range(30)
    map_to_names = dict(zip(names,numbers))
    map_to_numbers = dict(zip(numbers,names))
    return map_to_names, map_to_numbers


def cv_viz(degree,lambds, mse_tr, mse_te,save=False):
    """visualization the curves of mse_tr and mse_te, used during cross-validation.
    input : Degree (int) : Degree of polynomial augmentation
            lambds (np.logspace) : logspace of lambdas used during CV
            mse_tr, mse_te : RMSE arrays for train and test.
    """
    plt.semilogx(lambds, mse_tr, marker=".", color='b', label='train error')
    plt.semilogx(lambds, mse_te, marker=".", color='r', label='test error')
    plt.xlabel("lambda")
    plt.ylabel("rmse")
    plt.title("Cross validation for {} : Degree={}".format(save, degree))
    plt.legend(loc=2)
    plt.grid(True)
    if save:
        plt.savefig("../output/"+save+".png")
        plt.close()
