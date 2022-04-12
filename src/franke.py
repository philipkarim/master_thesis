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

#Import scikit learn modules
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_curve
from sklearn.utils import shuffle

#Import pytorch modules
import torch.optim as optim_torch
import torch

# Import the other classes and functions
from varQITE import *
from utils import *
from optimize_loss import optimize
from NN_class import *

import seaborn as sns


def franke_function(x, y, noise_sigma=0):
    """
    Frankes function
    """
    x_nine = 9 * x
    y_nine = 9 * y
    first = 0.75 * np.exp(-(x_nine - 2)**2 * 0.25 - (y_nine - 2)**2 * 0.25)
    second = 0.75 * np.exp(-(x_nine + 1)**2 / 49 - (y_nine + 1)**2 * 0.1)
    third = 0.5 * np.exp(-(x_nine - 7)**2 * 0.25 - (y_nine - 3)**2 * 0.25)
    fourth = - 0.2 * np.exp(-(x_nine - 4)**2 - (y_nine - 7)**2)
    if noise_sigma != 0:
        rand = np.random.normal(0, noise_sigma)

        return first + second + third + fourth + rand
    else:
        return first + second + third + fourth

def design_matrix(x, y, d):
    """Function for setting up a design X-matrix with rows [1, x, y, x², y², xy, ...]
    Input: x and y mesh, argument d is the degree.
    """

    if len(x.shape) > 1:
    # reshape input to 1D arrays (easier to work with)
        x = np.ravel(x)
        y = np.ravel(y)

    N = len(x)
    p = int((d+1)*(d+2)/2)	# number of elements in beta
    X = np.ones((N,p))

    for n in range(1,d+1):
        q = int((n)*(n+1)/2)
        for m in range(n+1):
            X[:,q+m] = x**(n-m)*y**m

    return X

def plot_franke(N=20, file_title='real_franke'):
    deg = 2
    x = np.linspace(0, 1, N); y = np.linspace(0, 1, N)
    x, y = np.meshgrid(x, y)
    x = np.ravel(x); y = np.ravel(y)
    X = design_matrix(x, y, deg)
    Y = franke_function(x, y, noise_sigma=0.1)

    fig = plt.figure() 
    ax = plt.axes(projection ='3d') 
    
    # Creating color map
    #my_cmap = plt.get_cmap('cividis')
    #my_cmap = plt.get_cmap('magma')
    my_cmap = plt.get_cmap('YlGnBu')
    
    # Creating plot
    trisurf = ax.plot_trisurf(X[:,1], X[:,2], Y,cmap = my_cmap,
                            linewidth = 0.2,antialiased = True,
                            edgecolor = 'grey') 
    
    # Adding labels
    ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')
    plt.tight_layout()
    plt.savefig('results/disc_learning/franke/'+file_title+'.png')
    plt.clf


def franke(initial_H, ansatz, n_epochs, n_steps, lr, opt_met, m1=0.7, m2=0.99, network_coeff=None, nickname=None):
    """
    Function to run regression of the franke function with the variational Boltzmann machine

    Args:
            initial_H(array):   The Hamiltonian which will be used, the parameters 
                                will be initialized within this function

            ansatz(array):      Ansatz whill be used in the VarQBM

            network_coeff(list): layerwise [input, output, bias], 0 if no bias, 1 with bias

    Returns:    Scores on how the BM performed
    """
    #Importing the data
    N=20

    #Bias variance tradeoff with complexity?
    deg = 5
    x = np.linspace(0, 1, N); y = np.linspace(0, 1, N)
    x, y = np.meshgrid(x, y)
    x = np.ravel(x); y = np.ravel(y)
    X = design_matrix(x, y, deg)
    Y = franke_function(x, y, noise_sigma=0.1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    #Now it is time to scale the data
    #MinMax data due two the values of the qubits will give the target value
    scaler=MinMaxScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    
    target_scaler=MinMaxScaler()
    target_scaler.fit(y_train)
    X_train = scaler.transform(y_train)
    X_test = scaler.transform(y_test)


    #TODO: The general function should start here

    #Some 'default' Hamiltonians
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
        #TODO: Remember net.eval() when testing, or validating?


        #Initializing the network
        net=Net(network_coeff, X_train[0], n_hamilParameters)
        #TODO: Might want to initialize the weights anohther method which reduces the values of the coefficients
        net.apply(init_weights)

        #Floating the network parameters
        net = net.float()

        #TODO: Insert optimzer list with name, lr, momentum, and everything else needed for optimizer
        #to not insert every thing as arguments
        if opt_met=='SGD':
            optimizer = optim_torch.SGD(net.parameters(), lr=lr)
            m1=0; m2=0
        elif opt_met=='Adam':
            optimizer = optim_torch.Adam(net.parameters(), lr=lr, betas=[m1, m2])
        elif opt_met=='Amsgrad':
            optimizer = optim_torch.Adam(net.parameters(), lr=lr, betas=[m1, m2],amsgrad=True)
        elif opt_met=='RMSprop':
            optimizer = optim_torch.RMSprop(net.parameters(), lr=lr, alpha=m1)
            m2=0
        else:
            print('optimizer not defined')
            exit()

        H_parameters=net(X_train[0])

    else:            
        #Initializing the parameters:
        H_parameters=np.random.uniform(low=-1.0, high=1.0, size=((n_hamilParameters, len(X_train[0]))))
        H_parameters = torch.tensor(H_parameters, dtype=torch.float64, requires_grad=True)

        if opt_met=='SGD':
            optimizer = optim_torch.SGD([H_parameters], lr=lr)
            m1=0; m2=0
        elif opt_met=='Adam':
            optimizer = optim_torch.Adam([H_parameters], lr=lr, betas=[m1, m2])
        elif opt_met=='Amsgrad':
            optimizer = optim_torch.Adam([H_parameters], lr=lr, betas=[m1, m2], amsgrad=True)
        elif opt_met=='RMSprop':
            optimizer = optim_torch.RMSprop([H_parameters], lr=lr, alpha=m1)
            m2=0
        else:
            print('Optimizer not defined')
            exit()
    

    init_params=np.array(copy.deepcopy(ansatz))[:, 1].astype('float')

    #TODO: Should only trace visible qubit [0].. or [2]?
    tracing_q, rotational_indices=getUtilityParameters(ansatz)
    optim=optimize(H_parameters, rotational_indices, tracing_q, learning_rate=lr, method=opt_met, fraud=True)
    
    varqite_train=varQITE(hamiltonian, ansatz, steps=n_steps, symmetrix_matrices=False)
    varqite_train.initialize_circuits()

    loss_mean=[]
    loss_mean_test=[]
    H_coefficients=[]

    for epoch in range(n_epochs):
        start_time=time.time()
        #print(f'Epoch: {epoch}/{n_epochs}')

        #Lists to save the predictions of the epoch
        #TODO: What to save?
        train_pred_epoch=[]
        test_pred_epoch=[]
        loss_list=[]

        #Loops over each sample
        X_train, y_train = shuffle(X_train, y_train, random_state=0)
        
        for i,sample in enumerate(X_train):
            varqite_time=time.time()
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
            #TODO: visible_q should be another place, maybe utilities?
            """
            Continue here!
            """
            visible_q=[0]
            p_QBM = PT.probabilities(visible_q)

            #TODO: Rewrite for regression, which loss?
            #Appending predictions and compute
            train_pred_epoch.append(0) if p_QBM[0]>0.5 else train_pred_epoch.append(1)

            #TODO: New loss function?
            loss=optim.fraud_CE(target_data,p_QBM)

            print(f'TRAIN: Loss: {loss}, p_QBM: {p_QBM}, target: {target_data}')

            #Appending loss and epochs
            loss_list.append(loss)
            
            #TODO: Remember to insert the visible qubit list, might do it
            #automaticly by changing the utilized variable function
            gradient_qbm=optim.fraud_grad_ps(hamiltonian, ansatz, d_omega, [0])
            gradient_loss=optim.gradient_loss(target_data, p_QBM, gradient_qbm)

            optimizer.zero_grad()
            if network_coeff is not None:
                output_coef.backward(torch.tensor(gradient_loss, dtype=torch.float64))
            else:
                gradient=np.zeros((len(gradient_loss),len(sample)))
                for ii, grad in enumerate(gradient_loss):
                    for jj, samp in enumerate (sample):
                        gradient[ii][jj]=grad*samp
                
                H_parameters.backward(torch.tensor(gradient, dtype=torch.float64))

            optimizer.step()


            #TODO: Time with and without if statement and check time
            print(f'1 sample run: {time.time()-varqite_time}')

        #Computes the test scores regarding the test set:
        loss_mean.append(np.mean(loss_list))
        print(f'Train Epoch complete : mean loss list= {loss_mean}')

 
        #Creating the correct hamiltonian with the input data as bias
        loss_list=[]
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
                #TODO: Insert correct variable [0]
                p_QBM = PT.probabilities([0])
                
                #TODO: Which loss function?
                loss=optim.fraud_CE(target_data,p_QBM)
                loss_list.append(loss)

                #TODO: Rewrite for regression
                test_pred_epoch.append(0) if p_QBM[0]>0.5 else test_pred_epoch.append(1)

                print(f'TEST: Loss: {loss}')

            #Computes the test scores regarding the test set:
            loss_mean_test.append(np.mean(loss_list))

    
    del optim
    del varqite_train

    #Save the scores
    if nickname is not None:
        #TODO: Also save the best hamiltonian parameters
        np.save('results/disc_learning/franke/loss_test'+nickname+'.npy', np.array(loss_mean_test))
        np.save('results/disc_learning/franke/loss_train'+nickname+'.npy', np.array(loss_mean))


    return 



