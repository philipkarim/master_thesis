"""
alt+z to fix word wrap

Rotating the monitor:
xrandr --output DP-1 --rotate right
xrandr --output DP-1 --rotate normal

xrandr --query to find the name of the monitors

"""
import random
import copy
import numpy as np
import qiskit as qk
from qiskit.circuit import Parameter, ParameterVector
from qiskit.quantum_info import DensityMatrix, partial_trace, state_fidelity
import time
import matplotlib.pyplot as plt
import torch


# Import the other classes and functions
from simulations import *
from optimize_loss import optimize
from utils import *
from varQITE import *
from fraud_classification import fraud_detection
from quantum_mnist import quantum_mnist

import multiprocessing as mp
#import seaborn as sns

#sns.set_style("darkgrid")
#plt.style.use('science')

both=True

plot_fidelity=True

if both==False:
    Hamiltonian=2
    p_data=np.array([0.12, 0.88])

    #Trying to reproduce fig2- Now we know that these params produce a bell state
    if Hamiltonian==1:
        params= [['ry',0, 0],['ry',0, 1], ['cx', 1,0], ['cx', 0, 1],
                    ['ry',np.pi/2, 0],['ry',0, 1], ['cx', 0, 1]]
                    #[gate, value, qubit]
        H=        [[1., 'z', 0]]
    elif Hamiltonian==2:
        params=  [['ry',0, 0], ['ry',0, 1], ['ry',0, 2], ['ry',0, 3], 
                ['cx', 3,0], ['cx', 2, 3],['cx', 1, 2], ['ry', 0, 3],
                ['cx', 0, 1], ['ry', 0, 2], ['ry',np.pi/2, 0], 
                ['ry',np.pi/2, 1], ['cx', 0, 2], ['cx', 1, 3]]
                #[gate, value, qubit]

        #Write qk.z instead of str? then there is no need to use get.atr?
        H=     [[1., 'z', 0], [1., 'z', 1], [-0.2, 'z', 0], 
                [-0.2, 'z', 1],[0.3, 'x', 0], [0.3, 'x', 1]]

    elif Hamiltonian==3:
        params= [['ry',0, 0],['ry',0, 1], ['cx', 1,0], ['cx', 0, 1],
                    ['ry',np.pi/2, 0],['ry',0, 1], ['cx', 0, 1]]
                    #[gate, value, qubit]
        H_init=np.random.uniform(low=-1.0, high=1.0, size=1)
        H=        [[H_init[0], 'z', 0]]

    elif Hamiltonian==4:
        params=  [['ry',0, 0], ['ry',0, 1], ['ry',0, 2], ['ry',0, 3], 
                ['cx', 3,0], ['cx', 2, 3],['cx', 1, 2], ['ry', 0, 3],
                ['cx', 0, 1], ['ry', 0, 2], ['ry',np.pi/2, 0], 
                ['ry',np.pi/2, 1], ['cx', 0, 2], ['cx', 1, 3]]
        
        p_data=np.array([0.5,0, 0, 0.5])
        H_init=np.random.uniform(low=-1.0, high=1.0, size=3)
        print(H_init)
        H=     [[H_init[0], 'z', 0], [H_init[0], 'z', 1], [H_init[1], 'z', 1], [H_init[2], 'z', 0]]


    #make_varQITE object
    start=time.time()
    varqite=varQITE(H, params, steps=10)
    #varqite.initialize_circuits()

    #Testing
    omega, d_omega=varqite.state_prep(gradient_stateprep=True)
    #print(d_omega)
    end=time.time()
    print(f'Time used: {np.around(end-start, decimals=1)} seconds')

    #varqite.dC_circ0(4,0)
    #varqite.dC_circ1(5,0,0)
    #varqite.dC_circ2(4,1,0)

    """
    Investigating the tracing of subsystem b
    """
    params=update_parameters(params, omega)

    #Dansity matrix measure, measure instead of computing whole DM
    trace_circ=create_initialstate(params)
    DM=DensityMatrix.from_instruction(trace_circ)


    #Rewrite this to an arbitrary amount of qubits
    if Hamiltonian==1 or Hamiltonian==2:
        if Hamiltonian==1:
            PT=partial_trace(DM,[1])
            H_analytical=np.array([[0.12, 0],[0, 0.88]])

        elif Hamiltonian==2:
            #What even is this partial trace? thought it was going to be [1,3??]
            #PT=partial_trace(DM,[0,3])=80%
            PT=partial_trace(DM,[2,3])
            
            H_analytical= np.array([[0.10, -0.06, -0.06, 0.01], 
                                    [-0.06, 0.43, 0.02, -0.05], 
                                    [-0.06, 0.02, 0.43, -0.05], 
                                    [0.01, -0.05, -0.05, 0.05]])

        print('---------------------')
        print('Analytical Gibbs state:')
        print(H_analytical)
        print('Computed Gibbs state:')
        print(PT.data)
        print('---------------------')

        H_fidelity=state_fidelity(PT.data, H_analytical, validate=False)

        print(f'Fidelity: {H_fidelity}')
elif both==True:
    pass

else:
    params1= [['ry',0, 0],['ry',0, 1], ['cx', 1,0], ['cx', 0, 1],
                ['ry',np.pi/2, 0],['ry',0, 1], ['cx', 0, 1]]
                #[gate, value, qubit]
    #H1=        [[1., 'z', 0]]
    H1=        [[[1., 'z', 0]]]

    params2=  [['ry',0, 0], ['ry',0, 1], ['ry',0, 2], ['ry',0, 3], 
            ['cx', 3,0], ['cx', 2, 3],['cx', 1, 2], ['ry', 0, 3],
            ['cx', 0, 1], ['ry', 0, 2], ['ry',np.pi/2, 0], 
            ['ry',np.pi/2, 1], ['cx', 0, 2], ['cx', 1, 3]]
            #[gate, value, qubit]

    #Write qk.z instead of str? then there is no need to use get.atr?
    #H2=     [[1., 'z', 0], [1., 'z', 1], [-0.2, 'z', 0], 
    #        [-0.2, 'z', 1],[0.3, 'x', 0], [0.3, 'x', 1]]
    
    H2=     [[[1., 'z', 0], [1., 'z', 1]], [[-0.2, 'z', 0]], 
        [[-0.2, 'z', 1]], [[0.3, 'x', 0]], [[0.3, 'x', 1]]]
    
    #H2=     [[[1., 'z', 0]], [[1., 'z', 1]], [[-0.2, 'z', 0]], 
    #    [[-0.2, 'z', 1]], [[0.3, 'x', 0]], [[0.3, 'x', 1]]]


    """
    Testing
    """
    print('VarQite 1')
    varqite1=varQITE(H1, params1, steps=10)
    varqite1.initialize_circuits()
    start1=time.time()
    omega1, d_omega=varqite1.state_prep(gradient_stateprep=True)
    end1=time.time()


    print('VarQite 2')
    varqite2=varQITE(H2, params2, steps=10)
    varqite2.initialize_circuits()
    start2=time.time()
    omega2, d_omega=varqite2.state_prep(gradient_stateprep=True)
    end2=time.time()
    #print(d_omega)

    print(f'Time used H1: {np.around(end1-start1, decimals=1)} seconds')
    print(f'Time used H2: {np.around(end2-start2, decimals=1)} seconds')

    print(f'omega: {omega2}')

    """
    Investigating the tracing of subsystem b
    """
    params1=update_parameters(params1, omega1)
    params2=update_parameters(params2, omega2)

    #Dansity matrix measure, measure instead of computing whole DM
    trace_circ1=create_initialstate(params1)
    trace_circ2=create_initialstate(params2)

    print(trace_circ2)

    DM1=DensityMatrix.from_instruction(trace_circ1)
    DM2=DensityMatrix.from_instruction(trace_circ2)

    PT1 =partial_trace(DM1,[1])
    H1_analytical=np.array([[0.12, 0],[0, 0.88]])

    PT2=partial_trace(DM2,[2,3])
    #Just to check that the correct parts are subtraced
    PT2_2=partial_trace(DM2,[1,3])
    PT2_3=partial_trace(DM2,[0,1])
    PT2_4=partial_trace(DM2,[0,2])

    H2_analytical= np.array([[0.10, -0.06, -0.06, 0.01], 
                            [-0.06, 0.43, 0.02, -0.05], 
                            [-0.06, 0.02, 0.43, -0.05], 
                            [0.01, -0.05, -0.05, 0.05]])

    print('---------------------')
    print('Analytical Gibbs state:')
    print(H1_analytical)
    print('Computed Gibbs state:')
    print(np.real(PT1.data))
    print('---------------------')

    
    print('---------------------')
    print('Analytical Gibbs state:')
    print(H2_analytical)
    print('Computed Gibbs state:')
    print(np.real(PT2.data))
    print('---------------------')

    H_fidelity1=state_fidelity(PT1.data, H1_analytical, validate=False)
    H_fidelity2=state_fidelity(PT2.data, H2_analytical, validate=False)
    H_fidelity2_2=state_fidelity(PT2_2.data, H2_analytical, validate=False)
    H_fidelity2_3=state_fidelity(PT2_3.data, H2_analytical, validate=False)
    H_fidelity2_4=state_fidelity(PT2_4.data, H2_analytical, validate=False)

    print(f'Fidelity: H1: {np.around(H_fidelity1, decimals=2)}, H2: '
                        f'{np.around(H_fidelity2, decimals=2)}, '
                        f'{np.around(H_fidelity2_2, decimals=2)}, '
                        f'{np.around(H_fidelity2_3, decimals=2)}, '
                        f'{np.around(H_fidelity2_4, decimals=2)}')




def train(H_operator, ansatz, n_epochs, target_data, n_steps=10, lr=0.1, optim_method='Adam', plot=True):
    print('------------------------------------------------------')
    
    init_params=np.array(copy.deepcopy(ansatz))[:, 1].astype('float')

    loss_list=[]
    epoch_list=[]
    norm_list=[]
    tracing_q, rotational_indices=getUtilityParameters(ansatz)

    optim=optimize(H_operator, rotational_indices, tracing_q, learning_rate=lr, method=optim_method) ##Do not call this each iteration, it will mess with the momentum

    varqite_train=varQITE(H_operator, ansatz, steps=n_steps, symmetrix_matrices=True)
    
    time_intit=time.time()
    varqite_train.initialize_circuits()
    print(f'initialization time: {time.time()-time_intit}')
    for epoch in range(n_epochs):
        print(f'epoch: {epoch}')

        ansatz=update_parameters(ansatz, init_params)
        omega, d_omega=varqite_train.state_prep(gradient_stateprep=False)

        optimize_time=time.time()

        ansatz=update_parameters(ansatz, omega)

        #print(f' omega: {omega}')
        #print(f' d_omega: {d_omega}')

        #Dansity matrix measure, measure instead of computing whole DM
        
        trace_circ=create_initialstate(ansatz)
        DM=DensityMatrix.from_instruction(trace_circ)
        PT=partial_trace(DM,tracing_q)
        p_QBM=np.diag(PT.data).real.astype(float)
        
        #print(f'p_QBM: {p_QBM}')
        loss=optim.cross_entropy_new(target_data,p_QBM)
        print(f'Loss: {loss, loss_list}')
        norm=np.linalg.norm((target_data-p_QBM), ord=1)
        #Appending loss and epochs
        norm_list.append(norm)
        loss_list.append(loss)
        epoch_list.append(epoch)

        time_g_ps=time.time()
        gradient_qbm=optim.gradient_ps(H_operator, ansatz, d_omega)
        #print(f'Time for ps: {time.time()-time_g_ps}')

        gradient_loss=optim.gradient_loss(target_data, p_QBM, gradient_qbm)
        #print(f'gradient_loss: {gradient_loss}')        

        H_coefficients=np.zeros(len(H_operator))

        for ii in range(len(H_operator)):
            H_coefficients[ii]=H_operator[ii][0][0]

        #print(f'Old params: {H_coefficients}')
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


def multiple_simulations(n_sims, initial_H, ans, epochs, target_data,opt_met , l_r, steps, names):
    saved_error=np.zeros((n_sims, epochs))
    l1_norm=np.zeros((n_sims, epochs))
    
    qbm_list=[]

    for i in range(n_sims):
        print(f'Seed: {i} of {n_sims}')
        H_init_val=np.random.uniform(low=-1.0, high=1.0, size=len(initial_H))
        print(H_init_val)
        
        for term_H in range(len(initial_H)):
            for qub in range(len(initial_H[term_H])):
                initial_H[term_H][qub][0]=H_init_val[term_H]
        
        time_1epoch=time.time()
        saved_error[i], l1_norm[i], dist=train(initial_H, copy.deepcopy(ans), epochs, target_data, n_steps=steps, lr=l_r, optim_method=opt_met, plot=False)
        qbm_list.append(dist)
        time_1epoch_end=time.time()

        print(f'Time for one loop: {time_1epoch_end-time_1epoch}')
    

    epochs_list=list(range(0,epochs))
    avg_list=np.mean(saved_error, axis=0)
    std_list=np.std(saved_error, axis=0)

    avg_list_norm=np.mean(l1_norm, axis=0)
    std_list_norm=np.std(l1_norm, axis=0)

    #print(l1_norm)
    #np.save('results/arrays/.npy', saved_error, l1_norm, np.array(qbm_list))


    if len(target_data)==4:
        min_error=1000
        max_error=0
        best_index=0
        for j in range(n_sims):
            print(f'saved error: {saved_error[j][-1]}')
            if min_error>saved_error[j][-1]:
                print(f'saved error: {saved_error[j][-1]}')
                min_error=saved_error[j][-1]
                best_pbm=qbm_list[j]
                best_index=j

            if max_error<saved_error[j][-1]:
                worst_pbm=qbm_list[j]
                max_error=saved_error[j][-1]

        print(f'best_pbm {best_pbm}')
        print(f'worst pbm {worst_pbm}')
        print('---------------------')
        print(f'avg_list {avg_list}')
        print(f'std_list {std_list}')
        print(f'error {saved_error}')
        print('---------------------')
        bell_state=[0.5,0,0,0.5]
        barWidth = 0.25
    
        # Set position of bar on X axis
        br1 = np.arange(len(bell_state))
        br2 = [x + barWidth for x in br1]
        br3 = [x + barWidth for x in br2]
        
        # Make the plot
        plt.bar(br1, bell_state, color ='r', width = barWidth,
                edgecolor ='grey', label ='Bell state')
        plt.bar(br2, worst_pbm, color ='g', width = barWidth,
                edgecolor ='grey', label ='Worst trained')
        plt.bar(br3, best_pbm, color ='b', width = barWidth,
                edgecolor ='grey', label ='Best trained')
        plt.xlabel('Sample')
        plt.ylabel('Probability')
        plt.xticks([r + barWidth for r in range(len(bell_state))],['00', '01', '10', '11'])
        plt.legend()
        plt.tight_layout()
        plt.savefig(str(l_r*1000)+str(len(target_data))+names+'_bar.png')
        plt.clf()
        #plt.show()

    plt.errorbar(epochs_list, avg_list, std_list)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    #plt.title('Bell state: Mean loss with standard deviation using 10 seeds')
    plt.savefig('lr'+str(l_r*1000)+str(len(target_data))+names+'_mean.png')
    plt.clf()

    #plt.show()
    if len(target_data)==4:
        plt.plot(epochs_list, saved_error[best_index])
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        #plt.title('Bell state: Best of 10 seeds')
        plt.savefig(str(l_r*1000)+str(len(target_data))+names+'_best.png')
        plt.clf()

    #plt.show()
    for k in range(len(l1_norm)):
        plt.plot(epochs_list, l1_norm[k])

    plt.xlabel('Epoch')
    plt.ylabel('L1 norm')
    #plt.title('Bell state: Random seeds')
    plt.savefig(str(l_r*1000)+str(len(target_data))+names+'_all.png')
    plt.clf()


    for k in range(len(saved_error)):
        plt.plot(epochs_list, saved_error[k])

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    #plt.title('Bell state: Random seeds')
    plt.savefig(str(l_r*1000)+str(len(target_data))+names+'_all.png')
    plt.clf()

        # Plot the subplots
    # Plot 1
    fig, axs = plt.subplots(2, sharex=True)
    #fig.suptitle('QBM training- Target: '+str(target_data))    #fig.xlabel('Epoch')
    fig.suptitle('QBM training- Target: Bell state')    #fig.xlabel('Epoch')

    #plt.figure(figsize=[11, 9])
    axs[1].errorbar(epochs_list, avg_list, std_list)
    axs[1].set(ylabel='Loss', xlabel='Epoch')
    axs[0].errorbar(epochs_list, avg_list_norm, std_list_norm)
    axs[0].set(ylabel='L1 Distance')
    fig.savefig('lr'+str(l_r*1000)+str(len(target_data))+names+'_both.png')

    """
    plt.subplot(2, sharex=True)
    ax1._label('Loss')

    plt.plot(x, y1, 'g', linewidth=2)
    plt.title('Plot 1: 1st Degree curve')

    # Plot 2
    plt.subplot(2, 2, 2)
    plt.scatter(x, y2, color='k')
    plt.title('Plot 2: 2nd Degree curve')

    # Plot 3
    plt.subplot(2, 2, 3)
    plt.plot(x, y3, '-.y', linewidth=3)
    plt.title('Plot 3: 3rd Degree curve')

    # Plot 4
    plt.subplot(2, 2, 4)
    plt.plot(x, y4, '--b', linewidth=3)
    plt.title('Plot 4: 4th Degree curve')

    plt.show()


    fig, (ax1, ax2) = plt.subplots(2, sharex=True)
    ax1.errorbar(epochs_list, avg_list, std_list)
    ax1._label('Loss')
    plt.xlabel('Epoch')

    ax2.errorbar(epochs_list, avg_list_norm, std_list_norm)
    #ax2.xlabel('Epoch')
    ax2._label('L1 norm')

    #plt.title('Bell state: Mean loss with standard deviation using 10 seeds')
    plt.clf()
    """

    return 

"""
def plot_fidelity2(n_steps, name=None):
    rz_add=False

    if rz_add==True:
        params1= [['ry',0, 0],['ry',0, 1], ['cx', 1,0], ['cx', 0, 1],
                    ['ry',np.pi/2, 0],['ry',0, 1], ['cx', 0, 1], ['rz',0, 2]]
        H1=        [[[1., 'z', 0]]]

        params2=  [['ry',0, 0], ['ry',0, 1], ['ry',0, 2], ['ry',0, 3], 
                ['cx', 3,0], ['cx', 2, 3],['cx', 1, 2], ['ry', 0, 3],
                ['cx', 0, 1], ['ry', 0, 2], ['ry',np.pi/2, 0], 
                ['ry',np.pi/2, 1], ['cx', 0, 2], ['cx', 1, 3], ['rz',0, 4]]

        H2=     [[[1., 'z', 0], [1., 'z', 1]], [[-0.2, 'z', 0]], 
            [[-0.2, 'z', 1]], [[0.3, 'x', 0]], [[0.3, 'x', 1]]]

    else:
        params1= [['ry',0, 0],['ry',0, 1], ['cx', 1,0], ['cx', 0, 1],
                    ['ry',np.pi/2, 0],['ry',0, 1], ['cx', 0, 1]]
        H1=        [[[1., 'z', 0]]]

        params2=  [['ry',0, 0], ['ry',0, 1], ['ry',0, 2], ['ry',0, 3], 
                ['cx', 3,0], ['cx', 2, 3],['cx', 1, 2], ['ry', 0, 3],
                ['cx', 0, 1], ['ry', 0, 2], ['ry',np.pi/2, 0], 
                ['ry',np.pi/2, 1], ['cx', 0, 2], ['cx', 1, 3]]

        H2=     [[[1., 'z', 0], [1., 'z', 1]], [[-0.2, 'z', 0]], 
            [[-0.2, 'z', 1]], [[0.3, 'x', 0]], [[0.3, 'x', 1]]]

    
    H1_analytical=np.array([[0.12, 0],[0, 0.88]])
    H2_analytical= np.array([[0.10, -0.06, -0.06, 0.01], 
                            [-0.06, 0.43, 0.02, -0.05], 
                            [-0.06, 0.02, 0.43, -0.05], 
                            [0.01, -0.05, -0.05, 0.05]])

    fidelities1_list=[]
    fidelities2_list=[]
    
    print('VarQite 1')
    varqite1=varQITE(H1, params1, steps=n_steps, symmetrix_matrices=True, plot_fidelity=True)
    varqite1.initialize_circuits()
    omega1, d_omega=varqite1.state_prep(gradient_stateprep=True)
    list_omegas_fielity1=varqite1.fidelity_omega_list()
        
    for i in range(len(list_omegas_fielity1)):
        params1=update_parameters(params1, list_omegas_fielity1[i])
        if rz_add==True:
            trace_circ1=create_initialstate(params1[:-1])
        else:
            trace_circ1=create_initialstate(params1)

        DM1=DensityMatrix.from_instruction(trace_circ1)
        PT1 =partial_trace(DM1,[1])
        fidelities1_list.append(state_fidelity(PT1.data, H1_analytical, validate=False))
        PT1_2 =partial_trace(DM1,[0])
    
    print(f'H1: {fidelities1_list[-1]}, H1_sec:{state_fidelity(PT1_2.data, H1_analytical, validate=False)}')
    
    print('VarQite 2')
    varqite2=varQITE(H2, params2, steps=n_steps, symmetrix_matrices=True, plot_fidelity=True)
    varqite2.initialize_circuits()
    star=time.time()
    omega2, d_omega=varqite2.state_prep(gradient_stateprep=True)
    print(time.time()-star)
    list_omegas_fielity2=varqite2.fidelity_omega_list()

    for j in range(len(list_omegas_fielity2)):
        params2=update_parameters(params2, list_omegas_fielity2[j])
        if rz_add==True:
            trace_circ2=create_initialstate(params2[:-1])
        else:
            trace_circ2=create_initialstate(params2)

        DM2=DensityMatrix.from_instruction(trace_circ2)
        #Switched to 0,1 instead of 2,3
        PT2 =partial_trace(DM2,[2,3])
        fidelities2_list.append(state_fidelity(PT2.data, H2_analytical, validate=False))
        
        PT2_2=np.around(state_fidelity(partial_trace(DM2,[0,1]).data, H2_analytical, validate=False), decimals=4)
        PT2_5=np.around(state_fidelity(partial_trace(DM2,[0,3]).data, H2_analytical, validate=False), decimals=4)
        PT2_6=np.around(state_fidelity(partial_trace(DM2,[1,2]).data, H2_analytical, validate=False), decimals=4)

    print(f'H2: {fidelities2_list[-1]}, H2_2: {PT2_2}, H2_5: {PT2_5}, H2_6: {PT2_6}')

    #print(PT2.data, PT2_2.data)

    #print(PT2.data)
    #print(np.linalg.norm(H2_analytical, 2))
    #print(PT2.data/np.linalg.norm(PT2.data))

    plt.plot(list(range(0, len(fidelities1_list))),fidelities1_list, label='H1')
    plt.plot(list(range(0, len(fidelities2_list))),fidelities2_list, label='H2')
    
    #plt.scatter(list(range(0, len(fidelities1_list))),fidelities1_list, label='H1')
    #plt.scatter(list(range(0, len(fidelities2_list))),fidelities2_list, label='H2')
   
    plt.xlabel('Step')
    plt.ylabel('Fidelity')
    plt.legend()

    if name!=None:
        plt.savefig('results/fidelity/'+name+'.png')
    else:
        plt.show()
        #pass
    
    return
"""

def find_best_alpha(n_steps, alpha_space, name=None):
    params1= [['ry',0, 0],['ry',0, 1], ['cx', 1,0], ['cx', 0, 1],
                ['ry',np.pi/2, 0],['ry',0, 1], ['cx', 0, 1]]
    H1=        [[[1., 'z', 0]]]

    params2=  [['ry',0, 0], ['ry',0, 1], ['ry',0, 2], ['ry',0, 3], 
            ['cx', 3,0], ['cx', 2, 3],['cx', 1, 2], ['ry', 0, 3],
            ['cx', 0, 1], ['ry', 0, 2], ['ry',np.pi/2, 0], 
            ['ry',np.pi/2, 1], ['cx', 0, 2], ['cx', 1, 3]]

    H2=     [[[1., 'z', 0], [1., 'z', 1]], [[-0.2, 'z', 0]], 
        [[-0.2, 'z', 1]], [[0.3, 'x', 0]], [[0.3, 'x', 1]]]
    
    H1_analytical=np.array([[0.12, 0],[0, 0.88]])
    H2_analytical= np.array([[0.10, -0.06, -0.06, 0.01], 
                            [-0.06, 0.43, 0.02, -0.05], 
                            [-0.06, 0.02, 0.43, -0.05], 
                            [0.01, -0.05, -0.05, 0.05]])

    fidelities1_list=[]
    fidelities2_list=[]

    for i in alpha_space:
        varqite1=varQITE(H1, params1, steps=n_steps, alpha=i)
        varqite1.initialize_circuits()
        omega1, d_omega=varqite1.state_prep(gradient_stateprep=True)
    
        params1=update_parameters(params1, omega1)
        trace_circ1=create_initialstate(params1)
        DM1=DensityMatrix.from_instruction(trace_circ1)
        PT1 =partial_trace(DM1,[1])
        fidelities1_list.append(state_fidelity(PT1.data, H1_analytical, validate=False))

        varqite2=varQITE(H2, params2, steps=n_steps, alpha=i)
        varqite2.initialize_circuits()
        omega2, d_omega=varqite2.state_prep(gradient_stateprep=True)
    
        params2=update_parameters(params2, omega2)
        trace_circ2=create_initialstate(params2)
        DM2=DensityMatrix.from_instruction(trace_circ2)
        PT2 =partial_trace(DM2,[2,3])
        fidelities2_list.append(state_fidelity(PT2.data, H2_analytical, validate=False))

    max_index1 = fidelities1_list.index(max(fidelities1_list))
    max_index2 = fidelities2_list.index(max(fidelities2_list))
    print(fidelities1_list)
    print(fidelities2_list)
    print(f'H1, lambda max: {alpha_space[max_index1]}, H2: {alpha_space[max_index2]}')

    plt.plot(alpha_space,fidelities1_list, label='H1')
    plt.plot(alpha_space,fidelities2_list, label='H2')
    plt.xlabel(r'$\lambda$')
    plt.ylabel('Fidelity')
    plt.legend()

    if name!=None:
        plt.savefig('results/fidelity/'+name+'.png')
    else:
        plt.show()

    return


def learningrate_investigation(n_sims, initial_H, ans, epochs, target_data,opt_met , l_r, steps, name):
    seed_error_l1=[]
 
    for i in range(n_sims):
        print(f'Seed: {i} of {n_sims}')
        H_init_val=np.random.uniform(low=-1., high=1., size=len(initial_H))
        print(H_init_val)
        
        for term_H in range(len(initial_H)):
            for qub in range(len(initial_H[term_H])):
                initial_H[term_H][qub][0]=H_init_val[term_H]
        
        time_1epoch=time.time()
        loss_temp, norm_temp, dist=train(initial_H, copy.deepcopy(ans), epochs, target_data, n_steps=steps, lr=l_r, optim_method=opt_met, plot=False)
        time_1epoch_end=time.time()
        seed_error_l1.append(np.array([loss_temp, norm_temp]))
        
        print(f'Time for one loop: {time_1epoch_end-time_1epoch}')
    
    seed_error_l1=np.array(seed_error_l1)
    np.save('results/arrays/learningrate/'+str(l_r)+name+'.npy', seed_error_l1)

    return 

def ite_gs(toy_example=True):

    if toy_example==True:
        H=None
    else:
        g0=0.2252;  g1=0.3435;  g2=-0.4347
        g3=0.5716;  g4=0.0910;  g5=0.0910

        hydrogen_ham=[[[g0, 'z', 0], [g0, 'z', 0]], 
                    [[g1, 'z', 0]],[[g2, 'z', 1]], 
                    [[g3, 'z', 0], [g3, 'z', 1]], 
                    [[g4, 'y', 0], [g4, 'y', 1]], 
                    [[g5, 'x', 0], [g5, 'x', 1]]]

        uniform_params=np.random.uniform(low=-1, high=1, size=8)
        
        ansatz_supp=[['ry',uniform_params[0], 0],['ry',uniform_params[1], 1], 
                    ['rz',uniform_params[2], 0],['rz',uniform_params[3], 1], 
                    ['cx', 0, 1],
                    ['ry',uniform_params[4], 0],['ry',uniform_params[5], 1], 
                    ['rz',uniform_params[6], 0],['rz',uniform_params[7], 1]]

        #eigenvalues=-1, -1, 1 and 1

        #Is each qubit an eigenvalue maybe? No idea
        test_hamiltonian=[[[1,'x', 0]], [[1, 'z', 1]]]

        test_circ=qk.QuantumCircuit(2)
        test_circ.x(0)
        test_circ.z(1)

        print(test_circ)
        #psi=create_initialstate(ansatz_supp)
        #print(psi) 

        varqite_gs=varQITE(test_hamiltonian, ansatz_supp, steps=300, maxTime=4,symmetrix_matrices=True)
        varqite_gs.initialize_circuits()
        omega, trash=varqite_gs.state_prep(gradient_stateprep=True)


        #varqite_gs=varQITE(hydrogen_ham, ansatz_supp, steps=10, symmetrix_matrices=True)
        #varqite_gs.initialize_circuits()
        #omega1, d_omega=varqite_gs.state_prep(gradient_stateprep=True)
        ansatz_supp=update_parameters(ansatz_supp, omega)
        psi=create_initialstate(ansatz_supp)

        backend = qk.Aer.get_backend('unitary_simulator')
        job = qk.execute(psi, backend)
        result = job.result()
        mat_psi=result.get_unitary(psi, decimals=3).data
        print(mat_psi)


        
        backend = qk.Aer.get_backend('unitary_simulator')
        job = qk.execute(test_circ, backend)
        result = job.result()
        mat_H=result.get_unitary(test_circ, decimals=3).data
        print(mat_H)

        print(mat_H@mat_psi)
        final_mat=(mat_H@mat_psi)-mat_psi
        print(final_mat)

        print(np.linalg.eig(np.real(final_mat)))



        #print(np.diag(mat))
        
def isingmodel(ansatz, n_epochs, n_steps=10, lr=0.1, optim_method='Amsgrad', with_field=False):    
    
    #Ising hamiltonian:
    #configurations=4

    if with_field==False:
        #No clue if the Hamiltonians are correct
        H_4=[[[0., 'z', 0], [0., 'z', 1]],
                    [[0., 'z', 0], [0., 'z', 1]], 
                    [[0., 'z', 0], [0., 'z', 1]], 
                    [[0., 'z', 0], [0., 'z', 1]]]

    else:
        H_4=[[[0., 'z', 0], [0., 'z', 1]],
                    [[0., 'z', 0], [0., 'z', 1]], 
                    [[0., 'z', 0], [0., 'z', 1]], 
                    [[0., 'z', 0], [0., 'z', 1]],
                    [[0., 'x', 0]], [[0., 'x', 0]],
                    [[0., 'x', 0]], [[0., 'x', 0]]]
    
    H_init_val=np.random.uniform(low=-1.0, high=1.0, size=len(H_4))
        
    for term_H in range(len(H_4)):
        for qub in range(len(H_4[term_H])):
            H_4[term_H][qub][0]=H_init_val[term_H]
    
    #Without
    """
    target_data=[1,0,0,0]
    random.seed(123)
    H_operator=[]
    for i in range(n_spins):
        spin_init=random.randrange(-1,2,2)
        H_operator.append([[spin_init, 'x', 0]])

    print(H_operator)
    #exit()
    """

    target_data=[0.5, 0, 0, 0.5]
    
    init_params=np.array(copy.deepcopy(ansatz))[:, 1].astype('float')

    loss_list=[]
    epoch_list=[]
    norm_list=[]
    tracing_q, rotational_indices=getUtilityParameters(ansatz)

    #print(tracing_q, rotational_indices, n_qubits_ansatz)

    optim=optimize(H_operator, rotational_indices, tracing_q, learning_rate=lr, method=optim_method) ##Do not call this each iteration, it will mess with the momentum

    varqite_train=varQITE(H_operator, ansatz, steps=n_steps, symmetrix_matrices=True)
    
    time_intit=time.time()
    varqite_train.initialize_circuits()
    print(f'initialization time: {time.time()-time_intit}')
    for epoch in range(n_epochs):
        print(f'epoch: {epoch}')

        ansatz=update_parameters(ansatz, init_params)
        omega, d_omega=varqite_train.state_prep(gradient_stateprep=False)

        optimize_time=time.time()

        ansatz=update_parameters(ansatz, omega)

        #print(f' omega: {omega}')
        #print(f' d_omega: {d_omega}')

        #Dansity matrix measure, measure instead of computing whole DM
        
        trace_circ=create_initialstate(ansatz)
        print(trace_circ)
        DM=DensityMatrix.from_instruction(trace_circ)
        PT=partial_trace(DM,tracing_q)
        
        #TODO: Test with PT.trace instead of np.diag
        

        H_energy=apply_hamiltonian(PT, mini_max_cut=True)
        p_QBM=np.diag(PT.data).real.astype(float)
        print(f'p_QBM: {p_QBM}')

        #loss=optim.cross_entropy_new(target_data,p_QBM)
        print(f'Energy: {H_energy, loss_list}')
        #norm=np.linalg.norm((target_data-p_QBM), ord=1)
        #Appending loss and epochs
        #norm_list.append(norm)
        loss_list.append(H_energy)
        epoch_list.append(epoch)

        #time_g_ps=time.time()
        gradient_qbm=optim.gradient_ps(H_operator, ansatz, d_omega)
        #print(f'Time for ps: {time.time()-time_g_ps}')

        #target_data=[0.2,3,2,1]
        #gradient_loss=optim.gradient_loss(target_data, p_QBM, gradient_qbm)
        #print(f'Gradient first {gradient_loss}')
        gradient_energy=optim.gradient_energy(gradient_qbm, H_energy)
        print(f'gradient_energy: {gradient_energy}')

        #exit()

        H_coefficients=np.zeros(len(H_operator))

        for ii in range(len(H_operator)):
            H_coefficients[ii]=H_operator[ii][0][0]

        #print(f'Old params: {H_coefficients}')
        #new_parameters=optim.adam(H_coefficients, gradient_loss)
        new_parameters=optim.adam(H_coefficients, gradient_energy)
        print(f'New parasm: {new_parameters}')

        #new_parameters=optim.gradient_descent_gradient_done(np.array(H)[:,0].astype(float), gradient_loss)
        #print(f'New params {new_parameters}')
        #TODO: Try this
        #gradient_descent_gradient_done(self, params, lr, gradient):
        
        print(H_operator)

        
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

def find_hamiltonian(ansatz, steps, l_r, opt_met):
    """
    Function to fit an hamiltonian to experimental data, generetaed self.
    """

    #Solution
    #H=[[[0.25, 'x', 0]], [[0.25, 'x', 1]], [[-0.5, 'z', 0], [-0.5, 'z', 0]]]
    #initial_H=[[[0, 'x', 0]], [[0, 'x', 1]], [[0, 'z', 0], [0, 'z', 0]]]
    #0, -0.5, -0.5, -1
    initial_H=       [[[0., 'z', 0], [0., 'z', 1]], 
                [[0., 'z', 0]], [[0., 'z', 1]]]
    target_data=[0.25, 0.5, 0.25, 0]
    #target_data=[0, 0.5, 0.5, 0]

    H_init_val=np.random.uniform(low=-1.0, high=1.0, size=len(initial_H))
    print(H_init_val)

    for term_H in range(len(initial_H)):
        for qub in range(len(initial_H[term_H])):
            initial_H[term_H][qub][0]=H_init_val[term_H]
    
    print(f'Initial H: {initial_H}')

    trash, trash, dont_save=train(initial_H, copy.deepcopy(ansatz), 30, target_data, 
                        n_steps=steps, lr=l_r, optim_method=opt_met, plot=False)



def main():
    np.random.seed(2202)
    #np.random.seed(321)
    torch.manual_seed(2202)
    rz_add=False

    number_of_seeds=10
    learningRate=0.1
    ite_steps=10
    epochs=100
    optimizing_method='Amsgrad'

    """
    [gate, value, qubit]
    """
    if rz_add==False:
        Ham1=       [[[1., 'z', 0]]]
        ansatz1=    [['ry',0, 0],['ry',0, 1], ['cx', 1,0], ['cx', 0, 1],
                    ['ry',np.pi/2, 0],['ry',0, 1], ['cx', 0, 1]]
            
        Ham2=       [[[0., 'z', 0], [0., 'z', 1]], 
                    [[0., 'z', 0]], [[0., 'z', 1]]]

        #Ham2=       [[[ 0.9595, 'z', 0], [ 0.9595, 'z', 1]], 
        #            [[ 0.0943, 'z', 0]], [[-0.0039, 'z', 1]]]

        ansatz2=    [['ry',0, 0], ['ry',0, 1], ['ry',0, 2], ['ry',0, 3], 
                    ['cx', 3,0], ['cx', 2, 3],['cx', 1, 2], ['ry', 0, 3],
                    ['cx', 0, 1], ['ry', 0, 2], ['ry',np.pi/2, 0], 
                    ['ry',np.pi/2, 1], ['cx', 0, 2], ['cx', 1, 3]]

        Ham2_fidelity=      [[[1., 'z', 0], [1., 'z', 1]], [[-0.2, 'z', 0]], 
                            [[-0.2, 'z', 1]], [[0.3, 'x', 0]], [[0.3, 'x', 1]]]

        ansatz_gen_dis=[['ry',0, 0], ['ry',0, 1], ['ry',0, 2], ['ry', 0, 3],
                        ['rz',0, 0], ['rz',0, 1], ['rz',0, 2], ['rz', 0, 3], 
                        ['cx', 1,0], ['ry',np.pi/2, 0],        ['cx', 2, 1],
                        ['rz',0, 0], ['ry', np.pi/2, 1],       ['cx', 3, 2],
                        ['rz',0, 1], ['ry', 0, 2],['ry',0, 3], ['rz', 0, 2],
                        ['rz',0, 3], ['cx', 0, 2],['cx', 1, 3]]
    else:
        Ham1=       [[[1., 'z', 0]]]
        ansatz1=    [['ry',0, 0],['ry',0, 1], ['cx', 1,0], ['cx', 0, 1],
                    ['ry',np.pi/2, 0],['ry',0, 1], ['cx', 0, 1], ['rx', 0, 2]]

        Ham2=       [[[0., 'z', 0], [0., 'z', 1]], 
                    [[0., 'z', 0]], [[0., 'z', 1]]]
        ansatz2=    [['ry',0, 0], ['ry',0, 1], ['ry',0, 2], ['ry',0, 3], 
                    ['cx', 3,0], ['cx', 2, 3],['cx', 1, 2], ['ry', 0, 3],
                    ['cx', 0, 1], ['ry', 0, 2], ['ry',np.pi/2, 0], 
                    ['ry',np.pi/2, 1], ['cx', 0, 2], ['cx', 1, 3], ['rz', 0, 4]]
    
    #print(create_initialstate(ansatz_gen_dis))
    #exit()

    ansatz3=getAnsatz(3)
    ansatz4=getAnsatz(4)

    Ham3=get_Hamiltonian(3)
    Ham4=get_Hamiltonian(4)

    p_data1=np.array([0.5, 0.5])
    p_data2=np.array([0.5, 0, 0, 0.5])
    p_data3=np.array(np.zeros(2**3)); p_data3[0]=0.5; p_data3[-1]=0.5
    p_data4=np.array(np.zeros(2**4)); p_data4[0]=0.5; p_data4[-1]=0.5
    p_data5=np.array(np.zeros(2**5)); p_data5[0]=0.5; p_data5[-1]=0.5

    Ham1=np.array(Ham1, dtype=object)
    Ham2=np.array(Ham2, dtype=object)

    start=time.time()
    
    #ite_gs(toy_example=False)
    #Takning  a break again with the ising thingy
    #isingmodel(ansatz2, epochs, n_steps=ite_steps,lr=0.1, optim_method=optimizing_method)

    #network_coeff=[[10,1], [5,1],[3,1],[4,0]] 
    
    #fraud_detection(1, ansatz2, 30, ite_steps, 0.01, optimizing_method)#000509_40_samples_both_sets')
    #With neural network
    #Remember "with_grad()" for testing cases
    layers=[[['sigmoid'],[32,1],['sigmoid'],[32,1],['sigmoid'], [32,1],['sigmoid']], [1, 3]]

    #fraud_detection(1, ansatz2, 30, ite_steps, 0.1, optimizing_method, m1=0.7, m2=0.99)#000509_40_samples_both_sets')
    fraud_detection(1, ansatz2, 30, ite_steps, 0.01, optimizing_method, m1=0.7, m2=0.99, network_coeff=layers)#000509_40_samples_both_sets')

    #fraud_detection(1, ansatz2, 30, ite_steps, 0.01, optimizing_method, network_coeff)#000509_40_samples_both_sets')
    #quantum_mnist(1, ansatz2, epochs, ite_steps, learningRate, optimizing_method, network_coeff=[[[32,1],[32,1]], [0]])

    #TODO: They use another ansatz to mimic Bell state! Rememebr to switch
    #multiple_simulations(number_of_seeds, Ham1, ansatz1, epochs, p_data1, optimizing_method,l_r=0.1, steps=ite_steps, names='H1_latest_10_seeds')
    #multiple_simulations(number_of_seeds, Ham2, ansatz2, epochs, p_data2, optimizing_method,l_r=0.1, steps=ite_steps, names='H2_latest_10_seeds')
    """Run these"""
    #multiple_simulations(number_of_seeds, Ham3, ansatz3, epochs, p_data3, optimizing_method,l_r=0.1, steps=ite_steps, names='H3_no_seed_new')
    #multiple_simulations(number_of_seeds, Ham4, ansatz4, epochs, p_data4, optimizing_method,l_r=0.1, steps=ite_steps, names='H4_no_seed')
    """
    """
    #learningrate_investigation(1, Ham1, ansatz1, 15, p_data1, optimizing_method,l_r=0.1, steps=ite_steps)
    #learningrate_investigation(number_of_seeds, Ham1, ansatz1, epochs, p_data1, optimizing_method,l_r=0.005, steps=ite_steps, name='09')
    #learningrate_investigation(number_of_seeds, Ham1, ansatz1, epochs, p_data1, optimizing_method,l_r=0.002, steps=ite_steps, name='09')
    #multiple_simulations(number_of_seeds, Ham1, ansatz1, epochs, p_data1, optimizing_method,l_r=0.1, steps=ite_steps, names='H1_10_seed_50_epoch')
    #multiple_simulations(1, Ham1, ansatz1, 100, p_data1, optimizing_method,l_r=0.1, steps=ite_steps, names='test_trash_run_dont_save')
    #multiple_simulations(number_of_seeds, Ham1, ansatz1, epochs, p_data1, optimizing_method,l_r=0.002, steps=ite_steps)
    
    #find_hamiltonian(ansatz2, ite_steps, learningRate, optimizing_method)

    #multiple_simulations(1, Ham2, ansatz_gen_dis, 25, p_data2, optimizing_method,l_r=learningRate, steps=10, names='test_trash_run_dont_save')
    #multiple_simulations(number_of_seeds, Ham1, ansatz1, epochs, p_data1, optimizing_method,l_r=learningRate, steps=ite_steps)
    #multiple_simulations(number_of_seeds, Ham2, ansatz2, epochs, p_data2, optimizing_method,l_r=learningRate, steps=ite_steps)
    
    """
    Fidelity simulations
    """
    #sim_plot_fidelity(ite_steps, rz_add=False, name='Fidelity_dynamic_lmb_without_rz')#, 'Final_fidelity')#, 'after_statevector')#, 'fidelity_H1_H2_new_0_001minC')
    #sim_plot_fidelity(ite_steps, rz_add=True, name='Fidelity_dynamic_lmb_with_rz')#, 'Final_fidelity')#, 'after_statevector')#, 'fidelity_H1_H2_new_0_001minC')

    #sim_lambda_fidelity_search(ite_steps, np.logspace(-12,0,13), rz_add=False, name='without_rz')
    #sim_lambda_fidelity_search(ite_steps, np.logspace(-12,0,13), rz_add=True, name='with_rz')

    """
    Generative learning
    """
    ## Remember to save all arrays in case i run it for a real quantum computer, then save.
    #learning_rate_search(Ham2, ansatz_gen_dis, epochs, p_data2, n_steps=ite_steps, lr=0.1, name=False, optim_method=optimizing_method, plot=False)
    #learning_rate_search(Ham1, ansatz1, epochs, p_data1, n_steps=ite_steps, lr=0.1, name=False, optim_method=optimizing_method, plot=False)
    
    #Use that learning rate to plot for various optimization methods, rms prop, adam, amsgrad, and sgd, each with different momemntum, maybe 2 or 3 momentums, same color of same thing
    #exhaustive_gen_search_paralell(Ham1, ansatz1, epochs, p_data1, n_steps=ite_steps)
    #exhaustive_gen_search_paralell(Ham2, ansatz2, epochs, p_data2, n_steps=ite_steps)
    #final_seed_sim(Ham2, ansatz_gen_dis, epochs, p_data2, n_steps=ite_steps)
    #final_seed_sim(Ham1, ansatz1, epochs, p_data1, n_steps=ite_steps)
    #train_sim(Ham1, ansatz1, epochs, p_data1, n_steps=ite_steps,lr=0.2, optim_method='Amsgrad', m1=0.7, m2=0.99)
    #train(Ham1, ansatz1, epochs, p_data1, n_steps=ite_steps, lr=0.1, optim_method='Amsgrad', plot=False)
    

    #fraud_detection(1, ansatz2, 30, ite_steps, 0.01, optimizing_method, network_coeff=[[[8,1],[8,1]], [0, 1]])#000509_40_samples_both_sets')

    #TODO: Okay new plan:
    #Reproduce the results with increasing learning rate
    #Do the same computations as you did with finding the best optimizers, and then find the best learning rate
    # Then Recompute the shit that is being computed right now, but with another thing instead

    
    #train_sim(Ham1, ansatz1, epochs, p_data1, n_steps=ite_steps,lr=0.5, optim_method='Amsgrad', m1=0.7, m2=0.99)
    #train_sim(Ham2, ansatz2, epochs, p_data2, n_steps=ite_steps,lr=0.1, optim_method='RMSprop', m1=0.99, m2=0.99)

    #train(Ham2, ansatz2, epochs, p_data2, n_steps=ite_steps, lr=0.1, optim_method='Amsgrad', plot=False)


    #Then use the generative, learning thing for q3? With 10 seeds to see that everyone converges
    #Run it again with real computer?
    
    """
    Discriminative learning- Fraud dataset
    """    
    #TODO: Make code with fraud regular- With bias variance?

    #TODO: Make code with network- With bias variance?
    #fraud_sim(1, ansatz2, 30, ite_steps, 0.01, optimizing_method)#000509_40_samples_both_sets')

    #TODO: 
    #   5 activations, (5)

    #   with bias and (3)
    #   without bias  (1)
    #   different biases, 

    #   different lr (4)
    #   different sizes and nodes (?)
    #   
   

    #TODO: What to do about the learning rates and stuff like that?
    #TODO: Layers and node tests?
    #TODO: Test for less samples?

    #TODO: Create code with Franke function to test
    #TODO: And also with mnist

    #Run with all 3 hamiltonians

    ###TASKS:
    
    # Start plot of franke function
    # Run with MNIST 4 samples?
    # Find best seed values
    # Plot the results created
    #Insert lots of results and write a bit on them and everything
    #Then just write methods part asap, like, just write out 5 pages of methods all night
    #should not take that long time since method part is just what I have been  doing during
    #the implementation
    #Preprocessing maybe?


    """
    Discriminative learning- Franke Function
    """

    """
    Discriminative learning- MNIST
    """


    
  
    """
    pid = os.fork()

    if pid > 0 :
        pid=os.fork()
        if pid>0:
            list3=[]
            start1=time.time()
            for i in range(int(5e8)):
                list3.append(i)
            print(f'time1: {time.time()-start1}')

            print("\nI am child process:")
            print("Process ID:", os.getpid())
            print("Parent's process ID:", os.getppid())
        else:
            list1=[]
            start2=time.time()
            for i in range(int(5e8)):
                list1.append(i)
            print(f'time2: {time.time()-start2}')

            print("I am parent process:")
            print("Process ID:", os.getpid())
            print("Child's process ID:", pid)
    else:
        list2=[]
        start3=time.time()
        for i in range(int(5e8)):
            list2.append(i)
        print(f'time3: {time.time()-start3}')
        print("\nI am child process:")
        print("Process ID:", os.getpid())
        print("Parent's process ID:", os.getppid())
    """
    
    #50 epochs

    #end_time=time.time()
    #print(f'Final time: {end_time-start}')

    #find_best_alpha(10, np.logspace(-4,1,5))


if __name__ == "__main__":
    main()





"""
Thoughts:

Todays list:
    - Paralellize code, mpi, fork slurp, spark
    - Run seed
    - Check if the real gradients are fed into the optimizer, like all 3?    
"""
