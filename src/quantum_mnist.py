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
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_curve
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt

# Import the other classes and functions
from optimize_loss import optimize
from utils import *
from varQITE import *

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


def quantum_mnist(initial_H, ansatz, n_epochs, n_steps, lr, opt_met):
    """
    Function to run fraud classification with the variational Boltzmann machine

    Args:
            initial_H(array):   The Hamiltonian which will be used, the parameters 
                                will be initialized within this function

            ansatz(array):      Ansatz whill be used in the VarQBM

    Returns:    Scores on how the BM performed
    """
    #Importing the data
    save_scores=False

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
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    #Now it is time to scale the data   
    scaler=MinMaxScaler()
    scaler.fit(X_train)
    #print(scaler.data_max_, scaler.data_min_)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    print(len(X_train))

    #TODO: Remove this when the model works
    X_train=X_train[0:100]
    y_train=y_train[0:100]
    X_test=X_test[0:25]
    y_test=y_test[0:25]
    
    X_train=np.array([X_train[1]])
    y_train=np.array([y_train[1]])
    X_test=np.array([X_test[0]])
    y_test=np.array([y_test[0]])

    X_test=[]
    y_test=[]


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
    H_parameters=np.random.uniform(low=-1.0, high=1.0, size=((n_hamilParameters, len(X_train[0]))))
    
    #print(f'Hamiltonian: {hamiltonian}')
    
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
        print(f'Epoch: {epoch}/{n_epochs}')

        #Lists to save the predictions of the epoch
        train_acc_epoch=[]
        test_acc_epoch=[]
        train_pred_epoch=[]
        test_pred_epoch=[]
        loss_list=[]

        y_train_labels=[]

        #Loops over each sample
        for i,sample in enumerate(X_train):
            #Updating the Hamiltonian with the correct parameters
            #print(f'Old hamiltonian {hamiltonian}')
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

            #print(f'Print {d_omega}')

            DM=DensityMatrix.from_instruction(trace_circ)
            PT=partial_trace(DM,tracing_q)
            
            #TODO: Test both these
            visible_q=[0,1]
            p_QBM = PT.probabilities(visible_q)
            #p_QBM = PT.probabilities([2,3])

            #Both work I guess, but then use[1,2,3] as tracing qubits
            #p_QBM=np.diag(PT.data).real.astype(float)

            #target_data=np.zeros(classes)
            #target_data[y_train[i]]=1

            #print(f'Target vector: {target_data}, target sol: {y_train[i]}')
            
            #TODO: Rewrite for multiclass classification
            #Appending predictions and compute
            #train_pred_epoch.append(0) if p_QBM[0]>0.5 else train_pred_epoch.append(1)
            target_data=np.zeros(classes)
            target_data[y_train[i]]=1

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
            new_parameters=optim.adam(H_parameters, gradient_loss, discriminative=False, sample=sample)
        
        #Computes the test scores regarding the test set:
        loss_mean.append(np.mean(loss_list))
        acc_score_train.append(accuracy_score(y_train,train_pred_epoch))

        print(f'Epoch complete: mean loss list= {loss_mean}')
        print(f'Epoch complete: AS train list= {acc_score_train}')
        print(f'Epoch complete: Hamiltomian= {hamiltonian}')

        print(f'Testing model on test data')

        #Creating the correct hamiltonian with the input data as bias
        loss_list=[]
        for i,sample in enumerate(X_test):
            #Updating the Hamiltonian with the correct parameters
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
            target_data[y_train[i]]=1

            loss=optim.fraud_CE(target_data,p_QBM)
            loss_list.append(loss)

            print(f'Sample: {i}/{len(X_test)}')
            print(f'Current AS: {accuracy_score(y_test[:i+1],test_pred_epoch)} Loss: {loss}')
            print(f'p_QBM: {p_QBM}, target: {target_data}')

        #Computes the test scores regarding the test set:
        acc_score_test.append(accuracy_score(y_test,test_pred_epoch))
        loss_mean_test.append(np.mean(loss_list))

    
    del optim
    del varqite_train

    #Save the scores
    if save_scores==True:
        np.save('results/mnist/quantum_mnist/acc_test.npy', np.array(acc_score_test))
        np.save('results/mnist/quantum_mnist/acc_train.npy', np.array(acc_score_train))
        np.save('results/mnist/quantum_mnist/loss_test.npy', np.array(loss_mean_test))
        np.save('results/mnist/quantum_mnist/loss_train.npy', np.array(loss_mean))


    return 



