import numpy as np
import matplotlib.pyplot as plt 


#=========================================================================#
#========                   Required functions                    ========#
#=========================================================================#


def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """Linear regression using standard Gradient Descent"""
    
    # initializing variables: 
    loss = 0
    w = initial_w
    
    for i in range(max_iters):
        #computing the gradient
        gradient = compute_gradient(y, tx, w)
        #weight update step
        w = w - gamma*gradient
    #computes the MSE loss
    loss = compute_MSE_loss(y, tx, w)
    return w, loss

def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    """Linear regression using Stochastic Gradient Descent
    Using a minibatch size of 1, random and shuffled minibatches 
    are generated using the batch_iter function 
    located in the Utility functions below."""
    
    w = initial_w
    loss = 0
    
    for i in range(max_iters):
        for y_batch, tx_batch in batch_iter(y, tx, batch_size=1, num_batches=1):
            gradient = compute_stoch_gradient(y_batch, tx_batch, w)
            w = w - gamma* gradient
        
    loss = compute_MSE_loss(y, tx, w)
        
    return w, loss


def least_squares(y, tx):
    """Least squares regression using normal equations    
    With X.T(y-X*w)=0,
    X.T * y = (X.T * X)*w
    a*w = b"""
    
    a = tx.T.dot(tx)
    b = tx.T.dot(y)
    #using numpy's linalg.solve to get 
    #the solution to a*w = b
    w = np.linalg.solve(a, b)
    loss = compute_MSE_loss(y, tx, w)
    
    return w, loss


def ridge_regression(y, tx, lambda_):
    """Ridge regression using normal equations
    Similar to the implementation of least squares, 
    with an added parameter lambda_prime = 2*N*lambda
    (X.T X + lambda_prime*I)*w = X.T * y
    i.e. a*w = b"""
    
    #computing lambda_prime times Identity
    lambda_prime = 2*tx.shape[0]*lambda_
    aI = lambda_prime * np.eye(tx.shape[1])
    #Recreating both sides of the equation
    # a*w = b
    a = tx.T.dot(tx) + aI
    b = tx.T.dot(y)
    w = np.linalg.solve(a, b)
    loss = compute_MSE_loss(y, tx, w)
    return w, loss


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Logistic regression using gradient descent or SGD"""
    return w, loss


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """Regularized logistic regression using gradient descent or SGD"""
    return w, loss


#=========================================================================#
#========                    Utility functions                    ========#
#=========================================================================#

#-------------------
# Gradient functions
#-------------------
def compute_gradient(y, tx, w):
    """Computes the gradient """
    
    err = y - tx.dot(w)
    grad = -tx.T.dot(err) / len(err)
    return grad


def compute_stoch_gradient(y, tx ,w):
    """Computes the stochastic gradient"""
    
    err = y - tx.dot(w)
    grad = -tx.T.dot(err) / len(err)
    return grad 


### Cost functions 
def compute_MSE_loss(y, tx, w):
    """Computes the loss using MSE"""
    
    e = y - tx.dot(w)
    mse = (1/2)*np.mean(e**2)
    return mse


def compute_MAE_loss(y, tx, w):
    """Computes the loss using MAE"""
    
    e = y - tx.dot(w)
    mae = np.mean(np.abs(e))
    return mae 


def compute_RMSE(y, tx, w):
    """Computes the RMSE for Ridge and Lasso regression"""
    
    return np.sqrt(2*compute_MSE_loss(y,tx,w))


#-----------------
# Helper functions 
#-----------------

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

            
def standardize(x):
    """Standardize the data-set to have 0 mean and unit variance"""
    mean_x = np.mean(x)
    x = x - mean_x
    std_x = np.std(x)
    x = x / std_x
    return x, mean_x, std_x


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

#def build_poly(x, degree):
#    """polynomial basis functions for input data x, for j=0 up to j=degree."""
#    poly = np.ones((len(x), 1))
#    for deg in range(1, degree+1):
#        poly = np.c_[poly, np.power(x, deg)]
#        #np.c_ adds (stacks) an extra column to poly, 
#        #extra column = np.power(x,deg)
#        #takes all entries of x, power it.
#        #each row is the new data entry with *degree* extra features 
#    return poly
#

