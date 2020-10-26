# Machine Learning - Project 1: Finding the Higgs Boson using Machine Learning 

**Authors:**

- Richie Yat Tsai Wan, 258934
- Luis da Silva, 259985
- Gaëlle Wavre, 260965

## Running the program

The data is available in the `data` folder in the zipped folders `train.csv.zip` and `test.csv.zip`. It must be unzipped into `train.csv` and `test.csv` for our code to run smoothly.
To the run the script, make sure to have unzipped the data in `data`, then open a terminal and go into the corresponding folder : 
```bash
cd project1/scripts
python run.py
```

The predictions and the best hyperparameters are generated in the `output` folder

All of the functions and scripts were written using :
    *CPython 3.7.5*

    *IPython 5.8.0*

    *numpy 1.19.1*

    *seaborn 0.11.0*
    
    *jupyterlab 2.2.6*

## Folder Structure 

*All .py files contain the necessary docstrings to use each and every function implemented, see below for a quick description of each method.
The jupyter notebooks (.ipynb) used for exploratory data analysis were not included. However, the functions used for EDA were implemented and are located in proj1_helpers.
For any information on how we did the EDA, please refer to `report_project1.pdf`*

- `data/` : contains train and test data in as .zip
- `output/` : Folder in which all outputs of `run.py` will be saved
- `scripts/` : Contains all the functions implemented
    - `old_hyperparams_tuning/` : folder containing scripts previously used for tuning of hyperparameters. (Deprecated)
    - `costs.py` : Contains the functions needed to compute costs, gradients for all regression methods as well as the (numerically stable) sigmoid function.
    - `implementations.py` : Contains the 6 functions required in the project1 guidelines.
    - `preprocessing.py` : Implements the methods needed to preprocess the data. It is split into 3 parts : 
        - *utility*, which deals with shuffling, splitting, sampling data 
        - *Pre-processing*, which will preprocess a given input tx
        - *Cluster pre-processing*, which contains functions that apply pre-processing to clustered datasets, as well as two functions that will do every steps of the pre-processing required.
        
    - `proj1_helpers` : helper functions.
        * Functions needed to load data and generate predictions **(not implemented by us)**
        * Helper functions used to view data (histograms, clusters, cross validation losses)
        * Functions used to generate predictions in clusters
    - `run.py` : required script. Used to run and generate the model parameters as well as final_prediction in the `./output/` folder.
    - `train_tune` : implements all functions used for hyperparameters tuning, such as cross-validation.
- `report_project1.pdf` : Report in .pdf format



## Functions description

### `proj1_helpers.py` 

- load_csv_data: loads data 
- predict_labels: generates class predictions given weights and a test data matrix
- predict_labels_log: implements the quantize step in logistic regression
- create_csv_submission: creates output file in csv format for submission to kaggle
- cluster_predict: takes clusterized weights, sets and IDs and performs a prediction
- histogram_clusters: visualization of the distribution of all features after having been clustered
- single_histogram: visualization of single histogram with cluster data
- view_correlation: Gets and plots the correlation matrix for a given dataset
- get_feature_names: gets the names of features from the header
- mapping: mapping of index number to and from feature names
- cv_viz: visualization of the curves of MSE of train and test sets


### `implementations.py`
- least_squares_GD: linear regression using standard gradient descent
- least_squares_SGD: linear regression using stochastic gradient descent
- least_squares: least squares regression using normal equations
- ridge_regression: ridge regression using normal equations with parameter lambda
- logistic_regression: logistic regression using GD or SGD
- reg_logistic_regression: regularized logistic regression using GD or SGD


### `costs.py` 

- Gradient functions:
    - compute_gradient: computes the gradient using the definition of gradient for MSE
    - compute_stoch_gradient: computes the stochastic gradient
    - compute_log_gradient: computes the gradient for logistic regression

- Cost functions:
    - compute_mse: computes the loss using mean square error
    - compute_mae: computes the loss using mean absolute error
    - compute_rmse: computes the root mean square error for ridge and lasso regression
    - sigmoid: computes the sigmoid function on input z
    - compute_logloss: computes the loss function for logistic regression


### `preprocessing.py` 

- Utility functions:
    - batch_iter: generates a minibatch iterator for a dataset
    - split_data: splits the dataset according to given split ratio
    - sample_data: defines samples from the given dataset

- Pre-processing functions:
    - standardize: standardizes the dataset to have 0 mean and unit variance
    - add_bias: adds a bias at the beginning of the dataset
    - build_poly: builds polynomial basis functions for input data up to a given degree
    - convert_label: converts the labels into 0 or 1 for logistic regression
    - replace_999_nan: replaces all '-999' values by NaN
    - replace_999_mean: replaces all '-999' values by the mean of their column
    - replace_999_median: replaces all '-999' values by the median of their column
    - prijetnum_indexing: obtains the indices for the various clusters according to their `PRI_jet_num` value
    - prijetnum_clustering: clusters the data into four groups according to their PRI_jet_num value
    - delete_features: if the entire column has '-999' values, the data is deleted and the index registered in a list `idx_taken_out` 

- Cluster-processing functions:
    - cluster_log: returns the data sets with a natural logarithm applied to selected features
    - cluster_std: standardizes the clusterized datasets
    - cluster_replace: replaces remaining '-999' values for all sets (by default by the mean)
    - cluster_buildpoly: builds polynomial expansion for all clusters with respect to their optimal degree found during crossvalidation
    - cluster_preprocessing_train: preprocesses whole training set (clusters them w.r.t PRIjetnum, applies log to wanted features, removes features with all '-999' rows, replaces remaining '-999' with the mean, standardizes, and returns all sets, targets, and deleted column indices)
    - cluster_preprocessing_test: identical to `cluster_processing_train` but on the test dataset
    - cluster_preprocessing_train_alt: same processing functions on the train dataset but done before the clustering
    - cluster_preprocessing_test_alt: same processing functions on the test dataset but done before the clustering

### `train_tune.py` 
- build_k_indices: builds k indices for k-fold cross validation
- k_split: returns the split datasets for k_fold cross validation
- cross_validation-ridge: returns the losses and weights of ridge regression for a given value of lambda
- crossval_ridge_gridsearch: computes the best degree of polynomial expansion and its associated optimal value of lambda as well as the associated weight that optimizes the RMSE loss for ridge regression using k-fold cross validation
- cross_validation_regulog: returns the losses and weights of regularized logistic regression for a given value of lambda and gamma
- crossval_regulog_gridsearch: computes the best degree of polynomial expansion and its associated optimal value of lambda as well as the associated weight that optimizes the RMSE loss for regularized logistic regression using k-fold cross validation