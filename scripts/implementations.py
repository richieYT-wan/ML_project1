import numpy as np
import matplotlib.pyplot as plt 


#=====================================================================#
#=======                  Required functions                   =======#
#=====================================================================#


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
    
    return w, loss

def least_squares(y, tx):
    """Least squares regression using normal equations"""
    return w, loss

def ridge_regression(y, tx, lambda_):
    """Ridge regression using normal equations"""
    return w, loss

def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Logistic regression using gradient descent or SGD"""
    return w, loss

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """Regularized logistic regression using gradient descent or SGD"""
    return w, loss


#=====================================================================#
#=======                   Utility functions                   =======#
#=====================================================================#


def compute_gradient(y, tx, w):
    """Computes the gradient """
    
    err = y - tx.dot(w)
    grad = -tx.T.dot(err) / len(err)
    return grad

def compute_MSE_loss(y, tx, w):
    """Computes the loss using MSE"""
    
    e = y - tx.dot(w)
    return (1/2)*np.mean(e**2)

def compute_MAE_loss(y, tx, w):
    """Computes the loss using MAE"""
    
    e = y - tx.dot(w)
    return np.mean(np.abs(e))

def compute_stoch_gradient(y, tx ,w):
    """Computes the stochastic gradient (in mini-batch)"""
    
    err = y - tx.dot(w)
    grad = -tx.T.dot(err) / len(err)
    return grad 

def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
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