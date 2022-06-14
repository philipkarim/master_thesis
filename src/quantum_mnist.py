"""
alt+z to fix word wrap

Rotating the monitor:
xrandr --output DP-1 --rotate right
xrandr --output DP-1 --rotate normal

xrandr --query to find the name of the monitors

"""
#Importing modules
import copy
import time
from dataclasses import replace
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#Importing qiskit learn modules
import qiskit as qk
from qiskit.quantum_info import DensityMatrix, partial_trace

#Importing scikit learn modules
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_curve
from sklearn.datasets import load_digits
from sklearn.utils import shuffle
from sklearn.datasets import fetch_openml
from ml_methods_class import MlMethods

#Importing pytorch modules
import torch.optim as optim_torch
import torch

# Import the other classes and functions
from optimize_loss import optimize
from utils import *
from varQITE import *
from NN_class import *
from BM import *
from train_supervised import train_model

def quantum_mnist(initial_H, ansatz, n_epochs, lr, optim_method, m1=0.7, m2=0.99, \
                v_q=2,layers=None, ml_task='classification', directory='mnist_classification',\
                name=None, init_ww='xavier_normal',QBM=True, samp_400=False, big_mnist=False):
    """
    Function to run fraud classification with the variational Boltzmann machine

    Args:
            initial_H(array):   The Hamiltonian which will be used, the parameters 
                                will be initialized within this function

            ansatz(array):      Ansatz whill be used in the VarQBM

    Returns:    Scores on how the BM performed
    """
    #Importing the data

    #Load the digits 0,1,2,3
    classes=4
    if big_mnist is not True:
        digits = load_digits(n_class=classes)
        X=digits.data
        y=digits.target
    else:
        #mnist = fetch_openml('mnist_784')
        X, y= fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)
        y=y.astype('int')
        index=[]

        if samp_400==False:
            n_each_class=175
        else:
            n_each_class=150

        for i in range(classes):
            index+=list(np.where(y==i)[0][:n_each_class])

        X , y = X[index], y[index]

    print(np.shape(X), np.shape(y))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=461)

    #400 samples
    if samp_400==True:
        X_train=X_train[0:400];  y_train=y_train[0:400]

    
    #Scale the data
    scaler=MinMaxScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    #X_train=X_train[[0]]
    #y_train=y_train[[0]]
    #X_test=X_test[[0]]
    #y_test=y_test[[0]]

    data_mnist=[X_train, y_train, X_test, y_test]
    params_fraud=[n_epochs, optim_method, lr, m1, m2]
    data_test=[X_test, y_test]

    binar=False
    plot_cm=True
    n_epoc=20
    acc=True

    n_epoc=0

    if QBM==True:
        train_model(data_mnist, initial_H, ansatz, params_fraud, visible_q=v_q, task=ml_task, folder=directory, network_coeff=layers, nickname=name, init_w=init_ww)
    elif QBM==False:
        best_params=None
        #best_params=gridsearch_params(data_mnist, 20, binarize_data=binar)

        train_rbm(data_mnist, best_params, plot_acc_vs_epoch=n_epoc, name='mnist', binarize_data=binar,plot_acc=acc, cm=plot_cm, data_val=data_test)
        #print('Mnist')
        #rbm_plot_scores(data_mnist, name='digit2', binarize_input=binar)

    else:
        model=MlMethods(data_mnist[0], data_mnist[1], data_mnist[2], data_mnist[3])
        
        if QBM=='NN':
            model.neural_net(4, 0.01)
        
        elif QBM=='OLS':
            model.ols_reg()
        
        elif QBM=='Ridge':
            model.ridge_reg()
       
        elif QBM=='Lasso':
            model.lasso_reg()
        
        elif QBM=='KNN':
            model.k_nn()
        
        elif QBM=='logreg':
            model.logistic_reg()
        
        else:
            sys.exit('No method defined')