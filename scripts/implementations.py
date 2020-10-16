import numpy as np
import matplotlib.pyplot as plt 
from proj1_helpers import *
from costs import *
from preprocessing import *

"""Implements the functions ("ML methods") required by the project 1 guidelines"""
#=========================================================================#
#========                   Required functions                    ========#
#=========================================================================#


def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """Linear regression using standard Gradient Descent"""
            
    # initializing variables: 
    loss = 0
    w = initial_w
    
    for i in range(max_iters):
        #computing the gradient & weight update step
        w = w - gamma*compute_gradient(y, tx, w)
    #computes the MSE loss
    loss = compute_mse(y, tx, w)
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
        
    loss = compute_mse(y, tx, w)
        
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
    loss = compute_mse(y, tx, w)
    
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
    loss = compute_mse(y, tx, w)
    return w, loss


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Logistic regression using gradient descent or SGD
    Labels must be binary, i.e. y : {0;1}
    """
    w = initial_w
    
    for i in range(max_iters):
        w = w - gamma*compute_log_gradient(y, tx, w)
    loss = compute_log_oss(y, tx, w)
    return w, loss


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """Regularized logistic regression using gradient descent or SGD"""
    return w, loss




