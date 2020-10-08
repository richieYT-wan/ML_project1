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