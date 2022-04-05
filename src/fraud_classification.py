"""
alt+z to fix word wrap

Rotating the monitor:
xrandr --output DP-1 --rotate right
xrandr --output DP-1 --rotate normal

xrandr --query to find the name of the monitors
"""
import copy
from dataclasses import replace
import numpy as np
import qiskit as qk
from qiskit.quantum_info import DensityMatrix, partial_trace
import time
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_curve
from sklearn.utils import shuffle

import torch.optim as optim_torch

# Import the other classes and functions
from varQITE import *
from utils import *
from optimize_loss import optimize
from NN_class import *

import seaborn as sns

#sns.set_style("darkgrid")

def bias_param(x, theta):
    """
    Function which computes the Hamiltonian parameters with supervised fraud dataset
    and datasample as bias

        Args:   
            x(list):        Data sample
            theta(array):   Hamiltonian parameters for 1 parameter

        Return: (float): The dot producted parameter
    """

    return np.dot(x, theta)


def fraud_detection(initial_H, ansatz, n_epochs, n_steps, lr, opt_met, network_coeff=None, bias_val=0.01, nickname=None):
    """
    Function to run fraud classification with the variational Boltzmann machine

    Args:
            initial_H(array):   The Hamiltonian which will be used, the parameters 
                                will be initialized within this function

            ansatz(array):      Ansatz whill be used in the VarQBM

            network_coeff(list): layerwise [input, output, bias], 0 if no bias, 1 with bias

    Returns:    Scores on how the BM performed
    """
    #Importing the data
    fraud_20=False

    if fraud_20==True:
        dataset_fraud=np.load('datasets/time_amount_zip_mcc_1000_instances.npy', allow_pickle=True)
        #Start by normalizing the dataset by subtracting the mean and dividing by the deviation:

        #Split the data into X and y
        X=np.hsplit(dataset_fraud, (len(dataset_fraud[0])-1,len(dataset_fraud[0])))
        y=X[1].astype('int')
        X=X[0]

        ## The reason we split it like this instead of the regular train_test_split is to secure
        ## The right amount of true samples in each set(Probably is a lot more efficient way to do this)

        #Extracts indices of samples which are fraud and not fraud
        true_indices=np.where(y==1)[0]
        false_indices=np.where(y==0)[0]

        #Train: 20% 100/500
        train_true_samples=100
        test_true_samples=50
        train_false_samples=400
        test_false_samples=200

        #CV
        #Liste med kretser, paralellisering, gpu, cluster?

        #Makes sure that each set of data contains the wanted number of true samples
        train_indices=np.random.choice(true_indices, train_true_samples, replace=False)
        true_indices = np.delete(true_indices, np.where(np.in1d(true_indices, train_indices)))
        test_indices=np.random.choice(true_indices, test_true_samples, replace=False)
        val_indices = np.delete(true_indices, np.where(np.in1d(true_indices, test_indices)))

        #Random sampling from the false samples
        train_indices_false=np.random.choice(false_indices, train_false_samples, replace=False)
        false_indices = np.delete(false_indices, np.where(np.in1d(false_indices, train_indices_false)))
        test_indices_false=np.random.choice(false_indices, test_false_samples, replace=False)
        val_indices_false = np.delete(false_indices, np.where(np.in1d(false_indices, test_indices_false)))

        y_train_indices=np.sort(np.concatenate((train_indices, train_indices_false)))
        y_test_indices=np.sort(np.concatenate((test_indices, test_indices_false)))
        y_val_indices=np.sort(np.concatenate((val_indices, val_indices_false)))

        #Splits the samples according to the sampling of y
        X_train=X[y_train_indices]
        X_test=X[y_test_indices]
        X_val=X[y_val_indices]

        y_train=y[y_train_indices]
        y_test=y[y_test_indices]
        y_val=y[y_val_indices]

    else:
        dataset_fraud=np.load('datasets/time_amount_zip_mcc_1000_instances_5050.npy', allow_pickle=True)

        X=np.hsplit(dataset_fraud, (len(dataset_fraud[0])-1,len(dataset_fraud[0])))
        y=X[1].astype('int')
        X=X[0]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
        #Now it is time to scale the data   
    
    scaler=StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    if fraud_20==True:
        X_val = scaler.transform(X_val)

    #print(y_train[9:19])

    #TODO: Remove this when the thing work
    X_train=X_train[0:1]
    y_train=y_train[0:1]

    #print(f'y_train: {y_train}')
    X_test=X_test[0:1]
    y_test=y_test[0:1]


    #print(y_train[0:20])
    #X_train=np.array([X_train[16]])
    #y_train=np.array([y_train[16]])
    #X_test=np.array([X_test[1]])
    #y_test=np.array([y_test[1]])
    
    #X_test=[]
    #y_test=[]


    #scaler=StandardScaler()
    #scaler.fit(y_train)
    #y_train = scaler.transform(y_train)
    #y_test = scaler.transform(y_test)
    #y_val = scaler.transform(y_val)

    #print(X_train_scaled)

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

    
    #Initializing the parameters:
    if network_coeff is not None:
        #TODO: Remember net.eval() when testing


        #Initializing the network
        #TODO: Add activation function?
        net=Net(network_coeff, X_train[0], n_hamilParameters)

        #print(net)
        net.apply(init_weights)


        #Floating the network parameters
        net = net.float()

        #TODO: Insert optimzer list with name, lr, momentum, and everything else needed for optimizer
        #to not insert every thing as arguments
        if opt_met=='SGD':
            optimizer = optim_torch.SGD(net.parameters(), lr=lr)
        elif opt_met=='Adam':
            optimizer = optim_torch.Adam(net.parameters(), lr=lr)
        elif opt_met=='Amsgrad':
            optimizer = optim_torch.Adam(net.parameters(), lr=lr, amsgrad=True)
        else:
            print('optimizer not defined')
            exit()

        H_parameters=net(X_train[0])
        print(f'Hamiltonian params: {H_parameters}')

    else:
        H_parameters=np.random.uniform(low=-1.0, high=1.0, size=((n_hamilParameters, len(X_train[0]))))
            
    init_params=np.array(copy.deepcopy(ansatz))[:, 1].astype('float')

    tracing_q, rotational_indices=getUtilityParameters(ansatz)
    optim=optimize(H_parameters, rotational_indices, tracing_q, learning_rate=lr, method=opt_met, fraud=True)
    
    varqite_train=varQITE(hamiltonian, ansatz, steps=n_steps, symmetrix_matrices=True)
    varqite_train.initialize_circuits()

    loss_mean=[]
    loss_mean_test=[]

    acc_score_train=[]
    acc_score_test=[]

    for epoch in range(n_epochs):
        start_time=time.time()
        print(f'Epoch: {epoch}/{n_epochs}')

        #Lists to save the predictions of the epoch
        train_acc_epoch=[]
        test_acc_epoch=[]
        train_pred_epoch=[]
        test_pred_epoch=[]
        loss_list=[]

        #Loops over each sample
        X_train, y_train = shuffle(X_train, y_train, random_state=0)
        
        for i,sample in enumerate(X_train):
            varqite_time=time.time()
            #Updating the Hamiltonian with the correct parameters
            #print(f'Old hamiltonian {hamiltonian}')
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
            #print(f'New hamiltonian {hamiltonian}')
            ansatz=update_parameters(ansatz, init_params)
            omega, d_omega=varqite_train.state_prep(gradient_stateprep=False)
            
            ansatz=update_parameters(ansatz, omega)
            trace_circ=create_initialstate(ansatz)

            DM=DensityMatrix.from_instruction(trace_circ)
            PT=partial_trace(DM,tracing_q)
            #TODO: Test with this thing
            p_QBM = PT.probabilities([0])
            #Both work I guess, but then use[1,2,3] as tracing qubits
            #p_QBM=np.diag(PT.data).real.astype(float)

            target_data=np.zeros(2)
            target_data[y_train[i]]=1
            
            #TODO: Rewrite for multiclass classification
            #Appending predictions and compute
            train_pred_epoch.append(0) if p_QBM[0]>0.5 else train_pred_epoch.append(1)

            loss=optim.fraud_CE(target_data,p_QBM)

            #loss=-np.sum(p_data*np.log(p_BM))

            #print(f'Sample: {i}/{len(X_train)}')
            print(f'Current AS: {accuracy_score(y_train[:i+1],train_pred_epoch)}, Loss: {loss}')
            #print(f'p_QBM: {p_QBM}, target: {target_data}')

            #Appending loss and epochs
            loss_list.append(loss)
            
            #TODO: Remember to insert the visible qubit list, might do it
            #automaticly by changing the utilized variable function
            gradient_qbm=optim.fraud_grad_ps(hamiltonian, ansatz, d_omega, [0])
            gradient_loss=optim.gradient_loss(target_data, p_QBM, gradient_qbm)
            #print(f'gradient_loss: {gradient_loss}')        

            #TODO: fix when diagonal elemetns, also do not compute the
            #gradient if there is no need inside the var qite loop 
            if network_coeff is not None:
                optimizer.zero_grad()
                output_coef.backward(torch.tensor(gradient_loss, dtype=torch.float64))
                optimizer.step()
            else:     
                new_parameters=optim.adam(H_parameters, gradient_loss, discriminative=False, sample=sample)
            
            #TODO: Time with and without if statement and check time
            print(f'1 sample run: {time.time()-varqite_time}')

        #Computes the test scores regarding the test set:
        loss_mean.append(np.mean(loss_list))
        acc_score_train.append(accuracy_score(y_train,train_pred_epoch))

        print(f'Epoch complete: mean loss list= {loss_mean}')
        print(f'Epoch complete: AS train list= {acc_score_train}')
        print(f'Epoch complete: Hamiltomian= {hamiltonian}')
        print(f'Train 1 sample: {time.time()-start_time}')
        print('Testing model on test data')

        #Creating the correct hamiltonian with the input data as bias
        loss_list=[]
        #net.eval()
        with torch.no_grad():
            for i,sample in enumerate(X_test):
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
                p_QBM = PT.probabilities([0])

                target_data=np.zeros(2)
                target_data[y_test[i]]=1
                
                loss=optim.fraud_CE(target_data,p_QBM)
                loss_list.append(loss)
                #TODO: Rewrite for multiclass classification
                #Appending predictions and compute
                test_pred_epoch.append(0) if p_QBM[0]>0.5 else test_pred_epoch.append(1)

                print(f'Sample: {i}/{len(X_test)}')
                print(f'Current AS: {accuracy_score(y_test[:i+1],test_pred_epoch)} Loss: {loss}')
                print(f'p_QBM: {p_QBM}, target: {target_data}')

            #Computes the test scores regarding the test set:
            acc_score_test.append(accuracy_score(y_test,test_pred_epoch))
            loss_mean_test.append(np.mean(loss_list))

    
    del optim
    del varqite_train

    #Save the scores
    if nickname is not None:
        np.save('results/disc_learning/acc_test_5050'+nickname+'.npy', np.array(acc_score_test))
        np.save('results/disc_learning/acc_train_5050'+nickname+'.npy', np.array(acc_score_train))
        np.save('results/disc_learning/loss_test_5050'+nickname+'.npy', np.array(loss_mean_test))
        np.save('results/disc_learning/loss_train_5050'+nickname+'.npy', np.array(loss_mean))


    return 



