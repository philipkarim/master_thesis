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


# Import the other classes and functions
from optimize_loss import optimize
from utils import *
from varQITE import *

import seaborn as sns

sns.set_style("darkgrid")


both=True

plot_fidelity=True


def train_fraudmodel(H_operator, ansatz, n_epochs, target_data, n_steps=10, lr=0.1, optim_method='Adam', plot=True):    
    init_params=np.array(copy.deepcopy(ansatz))[:, 1].astype('float')

    loss_list=[]
    epoch_list=[]
    norm_list=[]
    tracing_q, rotational_indices=getUtilityParameters(ansatz)

    #print(tracing_q, rotational_indices, n_qubits_ansatz)

    optim=optimize(H_operator, rotational_indices, tracing_q, learning_rate=lr, method=optim_method)
    varqite_train=varQITE(H_operator, ansatz, steps=n_steps)
    
    time_intit=time.time()
    varqite_train.initialize_circuits()
    print(f'initialization time: {time.time()-time_intit}')
    for epoch in range(n_epochs):
        print(f'epoch: {epoch}')

        #Stops, memory allocation??? How to check
        ansatz=update_parameters(ansatz, init_params)
        omega, d_omega=varqite_train.state_prep(gradient_stateprep=False)

        optimize_time=time.time()

        ansatz=update_parameters(ansatz, omega)

        print(f' omega: {omega}')
        print(f' d_omega: {d_omega}')

        #Dansity matrix measure, measure instead of computing whole DM
        
        trace_circ=create_initialstate(ansatz)
        DM=DensityMatrix.from_instruction(trace_circ)
        PT=partial_trace(DM,tracing_q)
        p_QBM=np.diag(PT.data).real.astype(float)
        
        print(f'p_QBM: {p_QBM}')
        loss=optim.cross_entropy_new(target_data,p_QBM)
        print(f'Loss: {loss, loss_list}')
        norm=np.linalg.norm((target_data-p_QBM), ord=1)
        #Appending loss and epochs
        norm_list.append(norm)
        loss_list.append(loss)
        epoch_list.append(epoch)

        time_g_ps=time.time()
        gradient_qbm=optim.gradient_ps(H_operator, ansatz, d_omega)
        print(f'Time for ps: {time.time()-time_g_ps}')

        gradient_loss=optim.gradient_loss(target_data, p_QBM, gradient_qbm)
        print(f'gradient_loss: {gradient_loss}')        

        H_coefficients=np.zeros(len(H_operator))

        for ii in range(len(H_operator)):
            H_coefficients[ii]=H_operator[ii][0][0]

        print(f'Old params: {H_coefficients}')
        #new_parameters=optim.adam(H_coefficients, gradient_loss)
        new_parameters=optim.adam(H_coefficients, gradient_loss)

        #new_parameters=optim.gradient_descent_gradient_done(np.array(H)[:,0].astype(float), gradient_loss)
        print(f'New params {new_parameters}')
        #TODO: Try this
        #gradient_descent_gradient_done(self, params, lr, gradient):

        for i in range(len(H_operator)):
            for j in range(len(H_operator[i])):
                H_operator[i][j][0]=new_parameters[i]
        
        varqite_train.update_H(H_operator)
    
    del optim
    del varqite_train

    if plot==True:
        plt.plot(epoch_list, loss_list)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.show()
    
    print(f'Time to optimize: {time.time()-optimize_time}')

    return np.array(loss_list), np.array(norm_list), p_QBM


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


def fraud_detection(initial_H, ansatz, n_epochs, n_steps, lr, opt_met):
    """
    Function to run fraud classification with the variational Boltzmann machine

    Args:
            initial_H(array):   The Hamiltonian which will be used, the parameters 
                                will be initialized within this function

            ansatz(array):      Ansatz whill be used in the VarQBM

    Returns:    Scores on how the BM performed
    """
    #Importing the data
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

    #Just doublechecks that all indices are unique
    #print(any((y_test == x).all() for x in y_train))
    
    #Now it is time to scale the data   
    scaler=StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    X_val = scaler.transform(X_val)

    #TODO: Remove this when the thing work
    X_train=X_train[0:100]
    y_train=y_train[0:100]
    X_test=X_test[0:25]
    y_test=y_test[0:25]

    #TODO: double check if I should use the mean of y_train or X_train
    #TODO: Is it really necessary to scale the target variables,
    # when we are dealing with binary classification? 
    #Doesnt hurt to test and see if it has any affect iguess

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
    H_parameters=np.random.uniform(low=-1.0, high=1.0, size=((n_hamilParameters, len(X_train[0]))))
    
    """
    for term_H in range(n_hamilParameters):
        for qub in range(len(hamiltonian[term_H])):
            hamiltonian[term_H][qub][0]=bias_param(X_train[0], H_init_val[term_H])
    """
    #print(f'Hamiltonian: {hamiltonian}')
    
    init_params=np.array(copy.deepcopy(ansatz))[:, 1].astype('float')

    tracing_q, rotational_indices=getUtilityParameters(ansatz)

    optim=optimize(n_hamilParameters, rotational_indices, tracing_q, learning_rate=lr, method=opt_met)
    varqite_train=varQITE(hamiltonian, ansatz, steps=n_steps)
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

        #Loops over each sample
        for i,sample in enumerate(X_train):
            #Updating the Hamiltonian with the correct parameters
            print(f'Old hamiltonian {hamiltonian}')
            for term_H in range(n_hamilParameters):
                for qub in range(len(hamiltonian[term_H])):
                    hamiltonian[term_H][qub][0]=bias_param(sample, H_parameters[term_H])

            #Updating the hamitlonian
            varqite_train.update_H(hamiltonian)
            print(f'New hamiltonian {hamiltonian}')
            ansatz=update_parameters(ansatz, init_params)
            omega, d_omega=varqite_train.state_prep(gradient_stateprep=False)
            ansatz=update_parameters(ansatz, omega)
            trace_circ=create_initialstate(ansatz)

            #print(f'Print {d_omega}')

            DM=DensityMatrix.from_instruction(trace_circ)
            PT=partial_trace(DM,tracing_q)
            p_QBM = PT.probabilities([0])
            #Both work I guess, but then use[1,2,3] as tracing qubits
            #p_QBM=np.diag(PT.data).real.astype(float)

            target_data=np.zeros(2)
            target_data[y_train[i]]=1
            
            #TODO: Rewrite for multiclass classification
            #Appending predictions and compute
            train_pred_epoch.append(0) if p_QBM[0]>0.5 else train_pred_epoch.append(1)

            loss=optim.fraud_CE(target_data,p_QBM)
            print(f'Sample: {i}/{len(X_train)}')
            print(f'Current AS: {accuracy_score(y_train[:i+1],train_pred_epoch)}, Loss: {loss}')
            print(f'p_QBM: {p_QBM}, target: {target_data}')

            #Appending loss and epochs
            loss_list.append(loss)
            
            #TODO: Remember to insert the visible qubit list, might do it 
            #automaticly
            gradient_qbm=optim.fraud_grad_ps(hamiltonian, ansatz, d_omega, [0])
            gradient_loss=optim.gradient_loss(target_data, p_QBM, gradient_qbm)
            #print(f'gradient_loss: {gradient_loss}')        

            #TODO: fix when diagonal elemetns, also do not compute the
            #gradient if there is no need inside the var qite loop 
            #H_coefficients=np.zeros(len(hamiltonian))
            new_parameters=optim.adam(H_parameters, gradient_loss, discriminative=True)
        
        #Computes the test scores regarding the test set:
        loss_mean.append(np.mean(loss_list))
        acc_score_train.append(accuracy_score(y_train,train_pred_epoch))

        print(f'Epoch complete: mean loss list= {loss_mean}')
        print(f'Epoch complete: AS train list= {acc_score_train}')

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
    np.save('results/fraud/acc_test.npy', np.array(acc_score_test))
    np.save('results/fraud/acc_train.npy', np.array(acc_score_train))
    np.save('results/fraud/loss_test.npy', np.array(loss_mean_test))
    np.save('results/fraud/loss_train.npy', np.array(loss_mean))


    return 



