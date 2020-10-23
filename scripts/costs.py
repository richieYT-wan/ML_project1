# -*- coding: utf-8 -*-
"""
    Implements the functions used to compute the loss and gradient of various
    methods. Also contains functions used for gradient steps such as sigmoid,
    calculate_hessian for logistic regression.
"""

import numpy as np

#=========================================================================#
#========                   Gradient functions                    ========#
#=========================================================================#


def compute_gradient(y, tx, w):
    """Computes the gradient 
        Using the definition of grad for MSE"""
    
    err = y - tx.dot(w)
    grad = -tx.T.dot(err) / len(err)
    return grad


def compute_stoch_gradient(y, tx ,w):
    """Computes the stochastic gradient"""
    
    err = y - tx.dot(w)
    grad = -tx.T.dot(err) / len(err)
    return grad 

def compute_log_gradient(y, tx, w):
    """Computes the gradient for logistic regression
    """
    sig = sigmoid(tx.dot(w))
    #reshaping, we had broadcasting issues.
    grad = (tx.T.dot(sig-(y.reshape(sig.shape))))
    return grad


#=========================================================================#
#========                     Cost functions                      ========#
#=========================================================================#


def compute_mse(y, tx, w):
    """Computes the loss using MSE"""
    
    e = y - tx.dot(w)
    mse = (1/2)*np.mean(e**2)
    return mse


def compute_mae(y, tx, w):
    """Computes the loss using MAE"""
    
    e = y - tx.dot(w)
    mae = np.mean(np.abs(e))
    return mae 


def compute_rmse(y, tx, w):
    """Computes the RMSE for Ridge and Lasso regression"""
    
    return np.sqrt(2*compute_mse(y,tx,w))

def sigmoid(t):
    """Computes the sigmoid function on input z. 
       z may be of the form tx.dot(w). Numerically stable
       For entries that are negative, the form exp(z)/(1+exp(z)) is used,
       whereas for positive entries, the form 1/(1+exp(-z)) is used.
       """
    #Copy to avoid unwanted inplace modifications
    z = t.copy()
    #Fancy np indexing
    z[z>0] = np.divide(1, 1+np.exp(-(z[z>0])) )
    z[z<=0] = np.divide(np.exp(z[z<=0]), 1+np.exp(z[z<=0]))
    return z

def compute_logloss(y, tx, w):
    """Computes the loss function for logistic regression.
       Using the negative log likelihood criterion
    """
    sig = sigmoid(tx.dot(w))
    #Add a infinitesimally small value to avoid log(0)
    loss = -1*( y.T.dot(np.log(sig+1e-8))+(1-y).T.dot(np.log(1-sig+1e-8)))
    
    #dividing by N, can be changed later, doesnt really change anything.
    return np.squeeze(loss)/len(y)
