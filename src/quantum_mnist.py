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

#Importing pytorch modules
import torch.optim as optim_torch
import torch

# Import the other classes and functions
from optimize_loss import optimize
from utils import *
from varQITE import *
from NN_class import *

from train_supervised import train_model
from BM import *


def quantum_mnist(initial_H, ansatz, n_epochs, lr, optim_method, m1=0.7, m2=0.99, \
                v_q=2,layers=None, ml_task='classification', directory='mnist_classification',\
                name=None, init_ww='xavier_normal',QBM=True):
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
    digits = load_digits(n_class=classes)
    """
    _, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
    for ax, image, label in zip(axes, digits.images, digits.target):
        ax.set_axis_off()
        ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
        ax.set_title("Training: %i" % label)
    
    plt.show()
    """

    X=digits.data
    y=digits.target

    """
    classes=len(np.unique(y))
    instances=np.zeros(classes)
    for i in range(1000):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=i)
        for j in range(classes):
            instances[j]=len(np.where(y_test==j)[0])
        if np.all(instances == instances[0]):
            print(i)
            break
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=461)


    #Scale the data
    scaler=MinMaxScaler()
    scaler.fit(X_train)
    #print(scaler.data_max_, scaler.data_min_)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)


    ###12 samples, 3x4 digits
    #X_train=np.concatenate((X_train[0:11], np.array([X_train[14]])), axis=0)
    #y_train=np.concatenate((y_train[0:11], np.array([y_train[14]])), axis=None)
    #X_test= np.concatenate((X_test[0:7], X_test[[8,9,13,16,17]]), axis=0)
    #y_test= np.concatenate((y_test[0:7], y_test[[8,9,13,16,17]]), axis=None)
    
    #X_train=np.array([X_train[0]])
    #y_train=np.array([y_train[0]])
    #X_test=np.array([X_test[5]])
    #y_test=np.array([y_test[5]])

    #X_test=[]
    #y_test=[]

    
    data_mnist=[X_train, y_train, X_test, y_test]
    params_fraud=[n_epochs, optim_method, lr, m1, m2]
    data_test=[X_test, y_test]

    binar=False
    plot_cm=True
    n_epoc=20
    acc=True

    if QBM==True:
        train_model(data_mnist, initial_H, ansatz, params_fraud, visible_q=v_q, task=ml_task, folder=directory, network_coeff=layers, nickname=name, init_w=init_ww)
    else:
        best_params=None
        #best_params=gridsearch_params(data_mnist, 20, binarize_data=binar)

        train_rbm(data_mnist, best_params, plot_acc_vs_epoch=n_epoc, name='mnist', binarize_data=binar,plot_acc=acc, cm=plot_cm, data_val=data_test)
        #print('Mnist')
        #rbm_plot_scores(data_mnist, name='digit2', binarize_input=binar)
