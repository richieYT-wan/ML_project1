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

#detect the amount of -999 values of input tx_train (not really necessary)
def detect_999(tx_train):
    print('There is', np.count_nonzero(tx_train == -999))
    index_999 = np.where(tx_train == -999)
    return index_999

#Delete all the values of -999, whitout flattening tx_train, but will shift all dimensions, not good
def treat_999_delete(tx_train):
    tx_train_deleted = np.array([i[i != -999] for i in tx_train])#delete the -999, will reduce vector dimensions, CAUTION
    return tx_train_deleted 

#Replace -999 values to NaN
def treat_999_nan(tx_train):
    tx_train_nan = np.where(tx_train == -999, np.nan, tx_train) #replace -999 by NaN
    return tx_train_nan

#A TESTER, voir si on obtient de meilleurs résultats
#Replace -999 values by the mean of the all features values, using first treat_999_nannt de meilleurs résultats
def treat_999_nan_to_mean(tx_train):
    tx_train_nan = treat_999_nan(tx_train) #replace -999 by NaN
    mean_of_feat = np.nanmean(tx_train_nan,axis = 0) #mean of columns
    inds = np.where(np.isnan(tx_train_nan)) #calculate index
    tx_train_nan[inds] = np.take(mean_of_feat, inds[1]) #replace NaN by mean
    return  tx_train_nan

#Replace -999 values by the median of the all features values, using first treat_999_nan
def treat_999_nan_to_median(tx_train):
    tx_train_nan = treat_999_nan(tx_train) #replace -999 by NaN
    med_of_feat = np.nanmedian(tx_train_nan,axis = 0) #median of columns
    inds = np.where(np.isnan(tx_train_nan)) #calculate index
    tx_train_nan[inds] = np.take(med_of_feat, inds[1]) #replace NaN by median
    return  tx_train_nan

#Replace outliers in features by median of the features.
#(int_conf = 3 -> 99,7%, int_conf = 2 -> 95%, int_conf = 1 -> 68%) 
def outliers(tx_train_1):
    int_conf = 1 #choix confidence intervalle
    mean_of_feat = np.nanmean( tx_train_1, axis = 0)#mean of all features, per column
    std_of_feat = np.nanstd( tx_train_1, axis = 0)#standard deviation of all features, per column
    med_of_feat = np.nanmedian( tx_train_1, axis = 0)#median deviation of all features, per column
    max_conf_int = mean_of_feat + int_conf * std_of_feat / np.sqrt( len( tx_train_1[0] ) ) #maximum of confidence interval
    min_conf_int = mean_of_feat - int_conf * std_of_feat / np.sqrt( len( tx_train_1[0] ) ) #min of confidence interval
    
    for i in range( len( tx_train_1[0] ) ):
        print('in feature index', i, np.count_nonzero( ( tx_train_1[i] > max_conf_int[i]) | (tx_train_1[i] < min_conf_int[i] ) ), 'outliers, with confidence intervalle', int_conf ) #can be put in comment
        tx_train_without_out = np.where( (tx_train_1[i] > max_conf_int[i]) | (tx_train_1[i] < min_conf_int[i]) , med_of_feat[i], tx_train_1) #replace values if it isn't in Confidence intervalle
        
    return tx_train_without_out 


