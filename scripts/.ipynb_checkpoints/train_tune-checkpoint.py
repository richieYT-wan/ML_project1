import numpy as np
import matplotlib.pyplot as plt
from implementations import *

"""
Implements functions needed for training, hyper-parameters tuning, cross-validation, 
as well as demo functions to look at output.
Unique (sole) author : Richie Yat-tsai Wan (258934)
"""

def build_k_indices(y, k_fold=4):#, seed=667):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    #np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval] for k in range(k_fold)]
    return np.array(k_indices)


def k_split(y, x, k_indices, k):
    """Returns the split datasets for k-fold cross validation"""
    
    # get k'th subgroup in test, others in train
    test_indice = k_indices[k]
    train_indice = k_indices[~(np.arange(k_indices.shape[0]) == k)]
    train_indice = train_indice.reshape(-1)
    #splitting targets and datasets
    y_test = y[test_indice]
    y_train = y[train_indice]
    tx_test = x[test_indice]
    tx_train = x[train_indice]
    return tx_train, y_train, tx_test, y_test

def cross_validation_ridge(y, tx, k_indices, k, lambda_):
    """return the losses and weights of ridge regression for a given 
       value of lambda"""
    # get k'th subgroup in test, others in train
    tx_tr, y_tr, tx_te, y_te = k_split(y, tx, k_indices, k)
    
    # ridge regression
    w, _ = ridge_regression(y_tr, tx_tr, lambda_)
    
    # calculate the loss for train and test data

    loss_tr = compute_rmse(y_tr, tx_tr, w)
    loss_te = compute_rmse(y_te, tx_te, w)
    return loss_tr, loss_te, w


def crossval_ridge_gridsearch(y, tx_clust, k_fold, lambdas, degrees, loss=False):
    """
    Input : y (target), np.array
            tx (dataset), np.array
            k_fold, int fold for Cross validation
            lambdas, degrees : np.arrays containing the values to be tested.
            loss, bool : If true, will return the train-test loss arrays, to be used in viz for plotting.
            
    Returns the lambda, degree and associated weight that optimizes
    the RMSE loss for Ridge Regression using K-fold cross validation.
    It is advisable to use clusterized and pre-processed tx as input 
    in order to maximize data points in each of the cluster/k'th subgroup.
    
    Computes the best degree of polynomial expansion and its associated
    optimal value of lambda.
    """
    #Initializing values. 
    k_indices = build_k_indices(y, k_fold)
    #Get arrays of train and test loss for each degree (in axis=0), and each lambdas(in axis=1)
    total_train_loss = np.empty((len(degrees),len(lambdas)))
    total_test_loss = np.empty((len(degrees),len(lambdas)))
    #Gridsearch loops.
    for id_deg, degree in enumerate(degrees):
        print("Iterating. Testing {} lambdas for current degree = {}".format(len(lambdas),degree))
        tx_poly = build_poly(tx_clust,degree)
        lambda_train_loss = []
        lambda_test_loss = []
        for id_lam, lambda_ in enumerate(lambdas):
            train_loss_tmp = []
            test_loss_tmp = []
            for k in range(k_fold):
                train_loss, test_loss, _ = cross_validation_ridge(y,tx_poly,
                                                                   k_indices,
                                                                   k,lambda_)
                train_loss_tmp.append(train_loss)
                test_loss_tmp.append(test_loss)
                
            lambda_train_loss.append(np.mean(train_loss_tmp))
            lambda_test_loss.append(np.mean(test_loss_tmp))
            
        total_train_loss[id_deg,:]=lambda_train_loss
        total_test_loss[id_deg,:]=lambda_test_loss
        
    #Getting the best degree, values.
    print("Getting best degree and lambda")
    best_id = np.argwhere(total_test_loss==np.min(total_test_loss))[0,]
    best_degree = degrees[best_id[0]]
    best_lambda = lambdas[best_id[1]]
    
    #Getting the optimized weights.
    print("Ridge regression : getting optimal weights with best degree ({}), lambda ({})".format(best_degree,best_lambda))
    
    tx_poly_best = build_poly(tx_clust,best_degree)
    w_opt, _ = ridge_regression(y,tx_poly_best,best_lambda)
    if loss==False:
        print("Done, returning optimal weight, degree, lambda.")
        return w_opt, best_degree, best_lambda
    elif loss:
        print("Done, returning optimal weight, degree, lambda \n And train and test loss arrays for visualization")
        return w_opt, best_degree, best_lambda, total_train_loss, total_test_loss
    
    

def cross_validation_regulog(y, tx, k_indices, k, lambda_, max_iters=1500,gamma=2.5e-6,tol = 1.25e-5):
    """return the losses and weights of regularized log regression for a given 
       value of lambda, gamma,"""
    # get k'th subgroup in test, others in train
    tx_tr, y_tr, tx_te, y_te = k_split(y, tx, k_indices, k)
    initial_w = np.random.randn(tx_tr.shape[1],1)

    # Regulog regression
    w, _, losses = reg_logistic_regression(y_tr,tx_tr,lambda_,
                                           initial_w, max_iters, gamma,tol)
    
    # calculate the loss for train and test data

    loss_tr = compute_logloss(y_tr, tx_tr, w)
    loss_te = compute_logloss(y_te, tx_te, w)
    return loss_tr, loss_te, w, losses


def crossval_regulog_gridsearch(y, tx_clust, k_fold, lambdas, degrees, 
                                max_iters= 1500, gamma = 2.5e-6, loss= False,
                                tol = 1.25e-6):
    """
    Input : y (target), np.array
            tx (dataset), np.array
            k_fold, int, fold for Cross validation
            lambdas, degrees : np.arrays containing the values to be tested.
            loss, bool : If true, will return the train-test loss arrays, to be used in viz for plotting.
            
    Returns the lambda, degree and associated weight that optimizes
    the RMSE loss for Ridge Regression using K-fold cross validation.
    It is advisable to use clusterized and pre-processed tx as input 
    in order to maximize data points in each of the cluster/k'th subgroup.
    
    Computes the best degree of polynomial expansion and its associated
    optimal value of lambda.
    """
    #Initializing values. 
    k_indices = build_k_indices(y, k_fold)
    #Converting to binary labels for log prediction
    y = convert_label(y)
    #Get arrays of train and test loss for each degree (in axis=0), and each lambdas(in axis=1)
    
    total_train_loss = np.empty((len(degrees),len(lambdas)))
    total_test_loss = np.empty((len(degrees),len(lambdas)))
    #Creates a massive array for losses. massive memory also issue with early termination?
   
    #Gridsearch loops.
    for id_deg, degree in enumerate(degrees):
        print("Iterating. Testing {} lambdas for current degree = {}".format(len(lambdas),degree))
        tx_poly = build_poly(tx_clust,degree)
        lambda_train_loss = []
        lambda_test_loss = []
        losses_graphs=np.empty
    
        for id_lam, lambda_ in enumerate(lambdas):
            train_loss_tmp = []
            test_loss_tmp = []
            print("newlambda")
            for k in range(k_fold):
                train_loss, test_loss, _, _ = cross_validation_regulog(y, tx_poly, k_indices, k,
                                                                   lambda_, max_iters, gamma,tol)
                
                train_loss_tmp.append(train_loss)
                test_loss_tmp.append(test_loss)
                
            lambda_train_loss.append(np.mean(train_loss_tmp))
            lambda_test_loss.append(np.mean(test_loss_tmp))
                
        total_train_loss[id_deg,:]=lambda_train_loss
        total_test_loss[id_deg,:]=lambda_test_loss
        
    #Getting the best degree, values.
    print("Getting best degree and lambda")
    best_id = np.argwhere(total_test_loss==np.min(total_test_loss))[0,]
    best_degree = degrees[best_id[0]]
    best_lambda = lambdas[best_id[1]]
    
    #Getting optimized weight
    print("ReguLog regression : getting optimal weights with best degree ({}), lambda ({})".format(best_degree,best_lambda))
    
    tx_poly_best = build_poly(tx_clust,best_degree)
    initial_w = np.random.randn(tx_poly_best.shape[1],1)
    #More iterations + lower tolerance to optimize more (if possible).
    w_opt, _, _ = reg_logistic_regression(y,tx_poly_best, best_lambda,
                                    initial_w, 2000, gamma, tol=1e-7)
    
    
    if loss==False:
        print("Done, returning optimal weight, degree, lambda.")
        return w_opt, best_degree, best_lambda
    
    elif loss:
        print("Done, returning optimal weight, degree, lambda \n And train and test loss arrays for visualization")
        return w_opt, best_degree, best_lambda, total_train_loss, total_test_loss
        
    