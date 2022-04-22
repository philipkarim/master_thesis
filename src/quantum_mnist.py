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

    X=X[0:150]
    y=y[0:150]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=2)
    #Now it is time to scale the data
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

    if QBM==True:
        train_model(data_mnist, initial_H, ansatz, params_fraud, visible_q=v_q, task=ml_task, folder=directory, network_coeff=layers, nickname=name, init_w=init_ww)
    else:
        best_params=None
        test_data=[X_test, y_test]
        #best_params=gridsearch_params(data_mnist, 10)
        train_rbm(data_mnist, best_params, plot_acc_vs_epoch=200, name='mnist')
        #rbm_plot_scores(data_mnist, name='digit2')

    """


    if initial_H==1:
        hamiltonian=[[[0., 'z', 0], [0., 'z', 1]], [[0., 'z', 0]], [[0., 'z', 1]]]
        n_hamilParameters=len(hamiltonian)
    elif initial_H==2:
        hamiltonian=[[[0., 'z', 0], [0., 'z', 1]], [[0., 'z', 0]], [[0., 'z', 1]], [[0.1, 'x', 0]],[[0.1, 'x', 1]]]
        n_hamilParameters=len(hamiltonian)-2
    elif initial_H==3:
        hamiltonian=[[[0., 'z', 0], [0., 'z', 1]], [[0., 'z', 0]], [[0., 'z', 1]], [[0, 'x', 0]],[[0, 'x', 1]]]
        n_hamilParameters=len(hamiltonian)

    else:
        print('Hamiltonian not defined')
        exit()

    init_params=np.array(copy.deepcopy(ansatz))[:, 1].astype('float')
    tracing_q, rotational_indices=getUtilityParameters(ansatz)

    #Initializing the parameters:
    if network_coeff is not None:
        #Initializing the network
        net=Net(network_coeff, X_train[0], n_hamilParameters)
        net.apply(init_weights)
        #Floating the network parameters
        net = net.float()

        if optim_method=='SGD':
            optimizer = optim_torch.SGD(net.parameters(), lr=lr)
            m1=0; m2=0
        elif optim_method=='Adam':
            optimizer = optim_torch.Adam(net.parameters(), lr=lr, betas=[m1, m2])
        elif optim_method=='Amsgrad':
            optimizer = optim_torch.Adam(net.parameters(), lr=lr, betas=[m1, m2], amsgrad=True)
        elif optim_method=='RMSprop':
            optimizer = optim_torch.RMSprop(net.parameters(), lr=lr, alpha=m1)
            m2=0
        else:
            print('Optimizer not defined')
            exit()
        
        H_parameters=net(X_train[0])

    else:
        #Initializing the parameters:
        H_parameters=np.random.uniform(low=-1.0, high=1.0, size=((n_hamilParameters, len(X_train[0]))))
        H_parameters = torch.tensor(H_parameters, dtype=torch.float64 ,requires_grad=True)#.float()

        #print(H_parameters)

        if optim_method=='SGD':
            optimizer = optim_torch.SGD([H_parameters], lr=lr)
            m1=0; m2=0
        elif optim_method=='Adam':
            optimizer = optim_torch.Adam([H_parameters], lr=lr, betas=[m1, m2])
        elif optim_method=='Amsgrad':
            optimizer = optim_torch.Adam([H_parameters], lr=lr, betas=[m1, m2], amsgrad=True)
        elif optim_method=='RMSprop':
            optimizer = optim_torch.RMSprop([H_parameters], lr=lr, alpha=m1)
            m2=0
        else:
            print('Optimizer not defined')
            exit()

    optim=optimize(H_parameters, rotational_indices, tracing_q, learning_rate=lr, method=optim_method, fraud=True)
    varqite_train=varQITE(hamiltonian, ansatz, steps=n_steps, symmetrix_matrices=False)
    varqite_train.initialize_circuits()

    loss_mean=[]
    loss_mean_test=[]

    acc_score_train=[]
    acc_score_test=[]

    for epoch in range(n_epochs):
        print(f'Epoch: {epoch}/{n_epochs}')

        #Lists to save the predictions of the epoch
        train_acc_epoch=[]
        test_acc_epoch=[]
        train_pred_epoch=[]
        test_pred_epoch=[]
        loss_list=[]

        y_train_labels=[]

        #Loops over each sample
        X_train, y_train = shuffle(X_train, y_train, random_state=0)


        for i,sample in enumerate(X_train):
            print('-----TRAINING-------')
            #Updating the Hamiltonian with the correct parameters
            if network_coeff is not None:
                #Network parameters
                output_coef=net(sample)

                for term_H in range(n_hamilParameters):
                    for qub in range(len(hamiltonian[term_H])):
                        hamiltonian[term_H][qub][0]=output_coef[term_H]
            else:
                for term_H in range(n_hamilParameters):
                    for qub in range(len(hamiltonian[term_H])):
                        hamiltonian[term_H][qub][0]=bias_param(sample, H_parameters[term_H])


            #Updating the hamitlonian
            varqite_train.update_H(hamiltonian)
            ansatz=update_parameters(ansatz, init_params)
            omega, d_omega=varqite_train.state_prep(gradient_stateprep=False)
            ansatz=update_parameters(ansatz, omega)
            trace_circ=create_initialstate(ansatz)

            DM=DensityMatrix.from_instruction(trace_circ)
            PT=partial_trace(DM,tracing_q)
            
            #TODO: [0,1] or [2,3]?
            visible_q=[0,1]
            p_QBM = PT.probabilities(visible_q)
            #p_QBM = PT.probabilities([2,3])

            target_data=np.zeros(classes)
            target_data[y_train[i]]=1

            #print(f'Target vector: {target_data}, target sol: {y_train[i]}')
            
            #TODO: Rewrite for multiclass classification
            #Appending predictions and compute
            #train_pred_epoch.append(0) if p_QBM[0]>0.5 else train_pred_epoch.append(1)

            train_pred_epoch.append(np.where(p_QBM==p_QBM.max())[0][0])
            
            loss=optim.fraud_CE(target_data,p_QBM)
            print(f'Sample: {i}/{len(X_train)}')
            print(f'Current AS: {accuracy_score(y_train[:i+1],train_pred_epoch)}, Loss: {loss}')
            print(f'p_QBM: {p_QBM}, target: {target_data}')

            #exit()            
            #Appending loss and epochs
            loss_list.append(loss)

            #TODO: Remember to insert the visible qubit list, might do it
            #automaticly
            gradient_qbm=optim.fraud_grad_ps(hamiltonian, ansatz, d_omega, visible_q)
            gradient_loss=optim.gradient_loss(target_data, p_QBM, gradient_qbm)
            #print(f'gradient_loss: {gradient_loss}')        

            #TODO: fix when diagonal elemetns, also do not compute the
            #gradient if there is no need inside the var qite loop
            #H_coefficients=np.zeros(len(hamiltonian))
            #new_parameters=optim.adam(H_parameters, gradient_loss, discriminative=False, sample=sample)
            optimizer.zero_grad()
            if network_coeff is not None:
                output_coef.backward(torch.tensor(gradient_loss, dtype=torch.float64))
            else:
                gradient=np.zeros((len(gradient_loss),len(sample)))
                for k, grad in enumerate(gradient_loss):
                    for s, samp in enumerate (sample):
                        gradient[k][s]=grad*samp
                
                H_parameters.backward(torch.tensor(gradient, dtype=torch.float64))

            optimizer.step()
        
        #Computes the test scores regarding the test set:
        loss_mean.append(np.mean(loss_list))
        acc_score_train.append(accuracy_score(y_train,train_pred_epoch))

        print(f'Train Epoch complete : mean loss list= {loss_mean}, AS: {acc_score_train}')

        print(f'---------TESTING----------')
        #Creating the correct hamiltonian with the input data as bias
        loss_list=[]
        with torch.no_grad():
            for i,sample in enumerate(X_test):
                #Updating the Hamiltonian with the correct parameters
                if network_coeff is not None:
                    #Network parameters
                    output_coef=net(sample)
                    for term_H in range(n_hamilParameters):
                        for qub in range(len(hamiltonian[term_H])):
                            hamiltonian[term_H][qub][0]=output_coef[term_H]
                else:
                    for term_H in range(n_hamilParameters):
                        for qub in range(len(hamiltonian[term_H])):
                            hamiltonian[term_H][qub][0]=bias_param(sample, H_parameters[term_H])

                #Updating the hamitlonian
                varqite_train.update_H(hamiltonian)
                ansatz=update_parameters(ansatz, init_params)
                omega, not_used=varqite_train.state_prep(gradient_stateprep=True)
                ansatz=update_parameters(ansatz, omega)
                trace_circ=create_initialstate(ansatz)

                DM=DensityMatrix.from_instruction(trace_circ)
                PT=partial_trace(DM,tracing_q)
                p_QBM = PT.probabilities(visible_q)
                test_pred_epoch.append(np.where(p_QBM==p_QBM.max())[0][0])

                target_data=np.zeros(classes)
                target_data[y_test[i]]=1

                loss=optim.fraud_CE(target_data,p_QBM)
                loss_list.append(loss)


        #Computes the test scores regarding the test set:
        acc_score_test.append(accuracy_score(y_test,test_pred_epoch))
        loss_mean_test.append(np.mean(loss_list))

        print(f'Loss: {loss_mean_test[-1]}, Current AS: {acc_score_test[-1]}')
    
    del optim
    del varqite_train

    #Save the scores
    if nickname is not None:
        np.save('results/disc_learning/mnist/acc_test'+nickname+'.npy', np.array(acc_score_test))
        np.save('results/disc_learning/mnist/acc_train'+nickname+'.npy', np.array(acc_score_train))
        np.save('results/disc_learning/mnist/loss_test'+nickname+'.npy', np.array(loss_mean_test))
        np.save('results/disc_learning/mnist/loss_train'+nickname+'.npy', np.array(loss_mean))


    return 
    """


