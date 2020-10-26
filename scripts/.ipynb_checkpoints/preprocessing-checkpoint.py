import numpy as np
import matplotlib.pyplot as plt

"""Implements functions that can modify data in any way.

   In particular, "utility functions" will help with sampling, shuffling, splitting
   without actually pre-processing the data. 
   
   "Pre-processing" functions will be able to modify or add features, filter, 
   remove corrupted data such as entries with -999 values, etc.
   
"""

#=========================================================================#
#========                    Utility functions                    ========#
#=========================================================================#


def batch_iter(y, tx, batch_size=1, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y`   and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]

            
def split_data(x, y, ratio, myseed=None):
    """split the dataset based on the split ratio.
    """
    # set seed, None by default
    np.random.seed(myseed)
    # generate random indices
    num_row = len(y)
    indices = np.random.permutation(num_row)
    index_split = int(np.floor(ratio * num_row))
    index_tr = indices[: index_split]
    index_te = indices[index_split:]
    # create split
    x_tr = x[index_tr]
    x_te = x[index_te]
    y_tr = y[index_tr]
    y_te = y[index_te]
    return x_tr, x_te, y_tr, y_te


def sample_data(y, x, size_samples):
    """sample from dataset."""
    num_observations = y.shape[0]
    random_permuted_indices = np.random.permutation(num_observations)
    y = y[random_permuted_indices]
    x = x[random_permuted_indices]
    return y[:size_samples], x[:size_samples]

#=========================================================================#
#========                 Pre-processing functions                ========#
#=========================================================================#

def standardize(x):
    """Standardize the data-set to have 0 mean and unit variance"""
    mean_x = np.mean(x)
    x = x - mean_x
    std_x = np.std(x)
    x = x / std_x
    return x, mean_x, std_x

def add_bias(tx):
    """Adds a bias at the beginning of an dataset.
       (Unused, as build_poly with degree 1 also adds a bias.)
       Input : tx, np.array of dim N x D
       Output : tx_biased, np.array of dim N x (D+1)
    """
    return  np.c_[np.ones((tx.shape[0], 1)),tx]

def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    
    #adding bias
    poly = np.ones((len(x), 1))
    
    for deg in range(1, degree+1):
        poly = np.c_[poly, np.power(x, deg)]
        
    return poly


def convert_label(y):
    """converts the labels into 0 or 1 for log reg
    Input : y : target labels
    Output : y : converted target labels with all -1 replaced with 0.
    """
    #copy to prevent unwanted inplace value assignment
    bin_y = y.copy()
    #using fancy numpy indexing
    bin_y[bin_y==-1]=0
    return bin_y

def replace_999_nan(tx):
    """Replaces all -999 values by NaN, allows easier processing below"""
    #copy to prevent unwanted inplace value assignment
    tx_out = tx.copy()
    tx_out[tx_out==-999]= np.nan
    return tx_out


def replace_999_mean(tx):
    """Replaces all -999 values by the *mean* of their column.
       First replaces all abherrant values by NaN, then compute the *mean*,
       ignoring those values, then replacing NaNs by the computed *mean*.
    """
    tx_out = replace_999_nan(tx) #replace -999 by NaN
    mean_of_feat = np.nanmean(tx_out,axis = 0) #mean of columns 
    inds = np.where(np.isnan(tx_out)) #calculate index
    tx_out[inds] = np.take(mean_of_feat, inds[1]) #replace NaN by mean
    return tx_out


def replace_999_median(tx):
    """Replaces all -999 values by the *median* of their column.
       First replaces all abherrant values by NaN, then compute the *median*,
       ignoring those values, then replacing NaNs by the computed *median*.
    """
    tx_out = replace_999_nan(tx) #replace -999 by NaN
    med_of_feat = np.nanmedian(tx_out,axis = 0) #median of columns 
    inds = np.where(np.isnan(tx_out)) #calculate index
    tx_out[inds] = np.take(med_of_feat, inds[1]) #replace NaN by median
    return tx_out


def prijetnum_indexing(tx, jetcol=22):
    """
    Gets the indices for the various clusters according to their PRI_jet_num.
    Input : tx (np.array) : entire train (or test) data set
    Output: ic0, ic1, ic2, ic3 (indices) : Indices of entries that belongs to each cluster (PRI_jet_num of 0 to 3)
    """
    pjn_arr = tx[:,jetcol]
    ic0, ic1, ic2, ic3 = (pjn_arr==0),(pjn_arr==1),(pjn_arr==2),(pjn_arr==3)
    return ic0, ic1, ic2, ic3

def prijetnum_clustering(tx,y=None,jetcol=22):
    """
    Clusters the data into four groups, according to their PRI_jet_num value.
    PRI_jet_num is found in column 22, can change if needed.
    Input : tx, y (training set and target), or only tx (test_set)
    Output : split dataset (clusters) of tx0 to tx3
    Additional ouput : Clusterized targets if it is a training set, i.e.
                       (Y is not None)
                       Indices if it is a test set (Y is None)
    """
    #Values of PRI_jet_num are found in column 22 of tx.
    id0,id1,id2,id3 = prijetnum_indexing(tx,jetcol)
    #getting indices, clusters for train data and targets 
    tx0 = tx[id0]
    tx1 = tx[id1]
    tx2 = tx[id2]
    tx3 = tx[id3]
    if y is not None:
        y0 = y[id0]
        y1 = y[id1]
        y2 = y[id2]
        y3 = y[id3]
        print("Prediction targets detected. Using a training set. \n Returning clusterized dataset and targets. \n")
        return tx0, y0, tx1, y1, tx2, y2, tx3, y3
    #When y is None, i.e. when only input is a test-set
    #Returns the clustermust also return indices
    #to use for prediction
    elif y is None:
        print("No targets detected. Using a test-set. \n Returning clusterized dataset and indices. \n")
        return tx0, id0, tx1, id1, tx2, id2, tx3, id3
    
def delete_features(tx):
    """
    If the entire column is equal to -999, 
    the entire column is deleted and the index is registered in a list "idx_taken_out" for the future prediction.
    """
    x_df = tx.copy()
    idx_taken_out=[]
    for i in range(len(x_df[0])):
        if np.all(x_df[:,i] == -999):
            idx_taken_out.append(i)
    x_df=np.delete(x_df, idx_taken_out, 1)
    #print(len(idx_taken_out), 'features deleted')
    return  x_df, idx_taken_out


#=========================================================================#
#========              Cluster-processing functions               ========#
#=========================================================================#


"""The functions below applies the pre-processing functions from above to each cluster."""

def cluster_log(tx0,tx1,tx2,tx3, feat):
    """
    Returns the data sets with a natural logarithm applied to the selected features
    Input : tx0, tx1, tx2, tx3 (np.array) : Input separated into clusters
            feat (list of indices) : List of features to be logged.
    Output : same as input but with logged feature.

    NOTE : Initially, the same set of features were logged in all group, hence the need for this function.
           Afterwards, it was found that cluster0 and 1 needed its own set of features while 2 and 3 shared
           the same features, so this function was kept.

    """
    t0 = np.copy(tx0)
    t1 = np.copy(tx1)
    t2 = np.copy(tx2)
    t3 = np.copy(tx3)    
    t0[:,feat] = np.log(t0[:,feat]+0.01)
    t1[:,feat] = np.log(t1[:,feat]+0.01)
    t2[:,feat] = np.log(t2[:,feat]+0.01)
    t3[:,feat] = np.log(t3[:,feat]+0.01)
    return t0, t1, t2, t3

def cluster_std(t0,t1,t2,t3):
    """
    Standardizes each of the separated dataset.
    """
    t0, mean0, std0 = standardize(t0)
    t1, mean1, std1 = standardize(t1)
    t2, mean2, std2 = standardize(t2)
    t3, mean3, std3 = standardize(t3)
    return t0, t1, t2, t3, mean0, mean1, mean2, mean3, std0, std1, std2, std3

def cluster_replace(t0,t1,t2,t3,f="median"):
    """
    Replaces remaining -999 values for all sets, using f. f is median by default
    Should be used after delete_features.
    """
    if f == "mean":
        print("::Replacing remaining -999 values with mean")
        t0=replace_999_mean(t0)
        t1=replace_999_mean(t1)
        t2=replace_999_mean(t2)
        t3=replace_999_mean(t3)
    if f == "median":
        print("::Replacing remaining -999 values with median")
        t0=replace_999_median(t0)
        t1=replace_999_median(t1)
        t2=replace_999_median(t2)
        t3=replace_999_median(t3)
    if f!="mean" and f!="median":
        print("Invalid f detected. Returning un-processed datasets")
    return t0, t1, t2, t3



def cluster_buildpoly(t0,t1,t2,t3,degs):
    """
    build_poly() function call for all clusters w.r.t to their optimal degree found during crossvalidation
    """
    t0 = build_poly(t0,degs[0])
    t1 = build_poly(t1,degs[1])
    t2 = build_poly(t2,degs[2])
    t3 = build_poly(t3,degs[3])
    
    return t0, t1, t2, t3



def cluster_preprocessing_train(tx_train,y, f="median"):
    """
    Pre-process whole training dataset. Clusters them w.r.t. PRIjetnum, applying log to wanted features,
    Removing features with all -999 rows, replacing remaning -999 values with f (median by default)
    Standardizes and returns all sets, targets, and deleted column indices.

    input : tx_train (np.array), whole training set
            y (np.array), whole training target
            f (str), = "mean" or "median" or write anything else to ignore

    Output: tx_df0 to tx_df0 (np.array) : Split(cluster) and preprocessed train datasets
            y0 to y3 (np.array) : Corresponding split targets
            id_del0 to id_del3 : Indices of deleted columns to be used in cluster_preprocessing_test

    """
    
    print("PREPROCESSING TRAIN DATA \n::Clustering data w.r.t. to PRI_jet_num numbers \n")
    tx0, y0, tx1, y1, tx2, y2, tx3, y3 = prijetnum_clustering(tx_train,y)
    
    #getting a placeholder to use for cluster_log 
    ph = np.ones((1,30))

    
    #Logarithm of selected features with long-tail distribution
    log_features_0 = [ 1, 2, 3, 7, 8, 9, 10, 13, 16, 19, 21]
    log_features_1 = [ 1, 2, 3, 7, 8, 9, 10, 13, 16, 19, 21, 23, 29]
    log_features_23 = [ 1, 2, 3, 5, 7, 8, 9, 10, 13, 16, 19, 21, 23, 26, 29]
    
    #OLDFEATURES
    #log_features = [1,2,3,7,8,9,10,13,16,19,21]
    print("::Getting indices of columns to remove before taking log.")
    _, id_del0 = delete_features(tx0)
    _, id_del1 = delete_features(tx1)
    _, id_del2 = delete_features(tx2)
    _, id_del3 = delete_features(tx3)
    
    print("::Taking the log of selected features")
    tx_df0, _, _, _ = cluster_log(tx0,ph,ph,ph,log_features_0)
    _, tx_df1, _, _ = cluster_log(ph, tx1, ph, ph, log_features_1)
    _, _, tx_df2, tx_df3 = cluster_log(ph, ph, tx2, tx3,log_features_23)

    #Deleting features with all -999 rows
    print("::Removing features where all rows are -999 or 0. Returning indices for later")
    tx_df0 = np.delete(tx_df0, id_del0, axis =1)
    tx_df1 = np.delete(tx_df1, id_del1, axis =1)
    tx_df2 = np.delete(tx_df2, id_del2, axis =1)
    tx_df3 = np.delete(tx_df3, id_del3, axis =1)
    tx_df0 = np.delete(tx_df0,-1,1)
    ##Replacing remaining -999 values with the mean or median of that feature
    tx_df0, tx_df1, tx_df2, tx_df3 = cluster_replace(tx_df0, tx_df1, tx_df2, tx_df3, f)
    
    #Standardizing
    print("::Standardizing : Setting mean to 0 and variance to 1")
    tx_df0, tx_df1, tx_df2, tx_df3, mean0, mean1, mean2, mean3, std0, std1, std2, std3 = cluster_std(tx_df0, tx_df1, tx_df2, tx_df3)
    means = [mean0, mean1, mean2, mean3]
    stds = [std0, std1, std2, std3]
    
    print("::Preprocessing done \n")
    return tx_df0, y0, tx_df1, y1, tx_df2, y2, tx_df3, y3, id_del0, id_del1, id_del2, id_del3, means, stds



def cluster_preprocessing_test(tX_test, id_del0, id_del1, id_del2, id_del3, means, stds, degs,f="median"):
    """
    Pre-process whole training dataset. Clusters them w.r.t. PRIjetnum, applying log to wanted features,
    Removing features with all -999 rows, replacing remaning -999 values with f (mean by default)
    Standardizes and returns all sets, targets, and deleted column indices.

    input : tx_train (np.array), whole training set
            id_del0, ..., id_del3, indices of deleted columns returned by 
            degs (list), degrees for build_poly found during crossvalidation gridsearch
            f (str), = "mean" or "median" or write anything else to ignore.

    Output: test0 to test3 : Split(cluster) and preprocessed train datasets
            i0 to i3 : indices used for cluster_predict (see proj1_helpers.py)
    """    
    print("PREPROCESSING TEST DATA \n::Clustering w.r.t. to PRI_jet_num numbers")
    test0, i0, test1, i1, test2, i2, test3, i3, = prijetnum_clustering(tX_test)
    
    ##Logarithm of selected features with long-tail distribution
    log_features_0 = [ 1, 2, 3, 7, 8, 9, 10, 13, 16, 19, 21]
    log_features_1 = [ 1, 2, 3, 7, 8, 9, 10, 13, 16, 19, 21, 23, 29]
    log_features_23 = [ 1, 2, 3, 5, 7, 8, 9, 10, 13, 16, 19, 21, 23, 26, 29]
    
    #placeholder for clusterlog
    ph = np.ones((1,30))
    #OLDFEATURES
    #log_features = [0,1,2,3,7,8,9,10,13,16,19,21]
    print("::Taking the log of selected features")
    test0, _, _, _ = cluster_log(test0, ph, ph, ph, log_features_0)
    _, test1, _, _ = cluster_log(ph, test1, ph, ph, log_features_1)
    _, _, test2, test3 = cluster_log(ph, ph, test2, test3, log_features_23)

    #Deleting features with all -999 rows, ID from pre_processing_train
    print("::Deleting features where all rows are -999 or 0.")
    test0 = np.delete(test0, id_del0, axis = 1 )
    test1 = np.delete(test1, id_del1, axis = 1 ) 
    test2 = np.delete(test2, id_del2, axis = 1 ) 
    test3 = np.delete(test3, id_del3, axis = 1 ) 
    test0 = np.delete(test0,-1,1)
    ##Replacing remaining -999 values with the mean or median of that feature
    test0, test1, test2, test3 = cluster_replace(test0, test1, test2, test3, f)
    #Standardizing
    print("::Standardizing : Using means and stds computed on the train set")
    #xd
    #test0, test1, test2, test3 = cluster_std(test0, test1, test2, test3)
    test0 = (test0-means[0])/stds[0]
    test1 = (test1-means[1])/stds[1]
    test2 = (test2-means[2])/stds[2]
    test3 = (test3-means[3])/stds[3]

    #Augmenting features w.r.t. optimal degrees found during CV.
    print("::Augmenting features")
    test0, test1, test2, test3 = cluster_buildpoly(test0,test1,test2,test3,degs)
    
    print("Preprocessing done, returning clusterized test set and indices \n")
    
    return test0, i0, test1, i1, test2, i2, test3, i3