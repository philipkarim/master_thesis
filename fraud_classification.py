"""
alt+z to fix word wrap

Rotating the monitor:
xrandr --output DP-1 --rotate right
xrandr --output DP-1 --rotate normal

xrandr --query to find the name of the monitors

"""
import copy
import numpy as np
import qiskit as qk
from qiskit.quantum_info import DensityMatrix, partial_trace
import time
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

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
    dataset_fraud=np.load('time_amount_zip_mcc_1000_instances.npy', allow_pickle=True)
    #Start by normalizing the dataset by subtracting the mean and dividing by the deviation:
    scaler=StandardScaler()
    #Transform data
    fraud_data_scaled=scaler.fit_transform(dataset_fraud)

    print(f'Scaled data: {fraud_data_scaled}')

    #TODO: Write the hamiltonians
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

    #This will be used to reinitiate the ansatz parameters each epoch
    #init_params=np.array(copy.deepcopy(ansatz))[:, 1].astype('float')
    H_init_val=np.random.uniform(low=-1.0, high=1.0, size=n_hamilParameters)
    for term_H in range(len(n_hamilParameters)):
        for qub in range(len(initial_H[term_H])):
            hamiltonian[term_H][qub][0]=H_init_val[term_H]

    print(f'Hamiltonian: {hamiltonian}')
    exit()
    
    loss_list=[]
    epoch_list=[]

    tracing_q, rotational_indices=getUtilityParameters(ansatz)

    optim=optimize(initial_H, rotational_indices, tracing_q, learning_rate=lr, method=opt_met)
    varqite_train=varQITE(initial_H, ansatz, steps=n_steps)
    varqite_train.initialize_circuits()

    for epoch in range(n_epochs):
        ansatz=update_parameters(ansatz, init_params)
        omega, d_omega=varqite_train.state_prep(gradient_stateprep=False)
        ansatz=update_parameters(ansatz, omega)
        trace_circ=create_initialstate(ansatz)

        #print(f' omega: {omega}')
        #print(f' d_omega: {d_omega}')

        DM=DensityMatrix.from_instruction(trace_circ)
        PT=partial_trace(DM,tracing_q)
        p_QBM=np.diag(PT.data).real.astype(float)
        
        print(f'p_QBM: {p_QBM}')

        

        #TODO: Do something about the target data
        loss=optim.fraud_CE(p_data,p_QBM)
        print(f'Loss: {loss, loss_list}')

        #Appending loss and epochs
        loss_list.append(loss)
        epoch_list.append(epoch)

        gradient_qbm=optim.gradient_ps(initial_H, ansatz, d_omega)
        gradient_loss=optim.gradient_loss(target_data, p_QBM, gradient_qbm)
        print(f'gradient_loss: {gradient_loss}')        

        H_coefficients=np.zeros(len(initial_H))

        for ii in range(len(initial_H)):
            H_coefficients[ii]=initial_H[ii][0][0]

        print(f'Old params: {H_coefficients}')
        new_parameters=optim.adam(H_coefficients, gradient_loss)

        print(f'New params {new_parameters}')
        for i in range(len(initial_H)):
            for j in range(len(initial_H[i])):
                initial_H[i][j][0]=new_parameters[i]
        
        varqite_train.update_H(initial_H)
    
    del optim
    del varqite_train

    return np.array(loss_list), np.array(norm_list), p_QBM



