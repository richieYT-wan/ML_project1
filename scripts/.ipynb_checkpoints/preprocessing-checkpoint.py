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


def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    poly = np.ones((len(x), 1))
    for deg in range(1, degree+1):
        poly = np.c_[poly, np.power(x, deg)]
        #np.c_ adds (stacks) an extra column to poly, 
        #extra column = np.power(x,deg)
        #takes all entries of x, power it.
        #each row is the new data entry with *degree* extra features 
    return poly


def convert_label(y):
    """converts the labels into 0 or 1 for log reg"""
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


def replace_999_mean(tx):
    """Replaces all -999 values by the *median* of their column.
       First replaces all abherrant values by NaN, then compute the *median*,
       ignoring those values, then replacing NaNs by the computed *median*.
    """
    tx_out = replace_999_nan(tx) #replace -999 by NaN
    med_of_feat = np.nanmedian(tx_out,axis = 0) #median of columns 
    inds = np.where(np.isnan(tx_out)) #calculate index
    tx_out[inds] = np.take(med_of_feat, inds[1]) #replace NaN by median
    return tx_out


def replace_outliers(tx, conf_level = 1):
    """Replaces outliers that aren't in the defined confidence interval by the median
       Input : tx (np.array), 
               conf_level (int), takes values : 0 (68%), 1 (95%), 2 (99.7%)
       Output : tx (np.array), without outliers
    """
    if conf_level in ~[0,1,2]:
        conf_level = 1;
    #Computing mean, standard deviation, median of all features column-wise
    mean_of_feat = np.nanmean( tx, axis = 0)
    std_of_feat = np.nanstd( tx, axis = 0)
    med_of_feat = np.nanmedian( tx, axis = 0)
    #Getting the boundaries of the confidence interval
    max_conf_int = mean_of_feat + (int_conf+1) * std_of_feat / np.sqrt( len( tx[0] ) )
    min_conf_int = mean_of_feat - (int_conf+1) * std_of_feat / np.sqrt( len( tx[0] ) )
    
    for i in range( len( tx[0] ) ):
        print('in feature index', i, np.count_nonzero( ( tx[i] > max_conf_int[i]) | (tx[i] < min_conf_int[i] ) ), 'outliers, with confidence intervalle', int_conf ) #can be put in comment
        tx_train_without_out = np.where( (tx[i] > max_conf_int[i]) | (tx[i] < min_conf_int[i]) , med_of_feat[i], tx) #replace values if it isn't in Confidence intervalle
        
    return tx_train_without_out 

def prijetnum_indexing(tx, jetcol=22):
    """
    Gets the indices for the various clusters according to their PRI_jet_num.
    """
    pjn_arr = tx[:,jetcol]
    return (pjn_arr==0),(pjn_arr==1),(pjn_arr==2),(pjn_arr==3)

def prijetnum_clustering(tx,y=None,jetcol=22):
    """
    Clusters the data into four groups, according to their PRI_jet_num value.
    PRI_jet_num is found in column 22, can change if needed.
    Input : tx, y (training set and target), or only tx (test_set)
    Output : split dataset (clusters). 
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
        print("Using a training set. \n Returning clusterized dataset and targets. \n")
        return tx0, y0, tx1, y1, tx2, y2, tx3, y3
    #When y is None, i.e. when only input is a test-set
    #Returns the clustermust also return indices
    #to use for prediction
    elif y is None:
        print("Using a test-set. \n Returning clusterized dataset and indices. \n")
        return tx0, id0, tx1, id1, tx2, id2, tx3, id3
    
