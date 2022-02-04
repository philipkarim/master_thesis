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

# Import the other classes and functions
from optimize_loss import optimize
from utils import *
from varQITE import *

import multiprocessing as mp
import seaborn as sns

sns.set_style("darkgrid")


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

    loss_list=[]
    epoch_list=[]
    norm_list=[]
    tracing_q, rotational_indices=getUtilityParameters(ansatz)

    #print(tracing_q, rotational_indices, n_qubits_ansatz)

    optim=optimize(H_operator, rotational_indices, tracing_q, learning_rate=lr, method=optim_method) ##Do not call this each iteration, it will mess with the momentum

    varqite_train=varQITE(H_operator, ansatz, steps=n_steps)
    
    time_intit=time.time()
    varqite_train.initialize_circuits()
    print(f'initialization time: {time.time()-time_intit}')
    for epoch in range(n_epochs):
        print(f'epoch: {epoch}')

        #Stops, memory allocation??? How to check
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


def multiple_simulations(n_sims, initial_H, ans, epochs, target_data,opt_met , l_r, steps, names):
    saved_error=np.zeros((n_sims, epochs))
    l1_norm=np.zeros((n_sims, epochs))
    
    qbm_list=[]

    for i in range(n_sims):
        print(f'Seed: {i} of {n_sims}')
        H_init_val=np.random.uniform(low=-0.5, high=0.5, size=len(initial_H))
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

    return 


def plot_fidelity(n_steps, name=None):
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
    varqite1=varQITE(H1, params1, steps=n_steps, plot_fidelity=True)
    varqite1.initialize_circuits()
    omega1, d_omega=varqite1.state_prep(gradient_stateprep=True)
    list_omegas_fielity1=varqite1.fidelity_omega_list()
    
    for i in range(len(list_omegas_fielity1)):
        params1=update_parameters(params1, list_omegas_fielity1[i])
        trace_circ1=create_initialstate(params1)
        DM1=DensityMatrix.from_instruction(trace_circ1)
        PT1 =partial_trace(DM1,[1])
        fidelities1_list.append(state_fidelity(PT1.data, H1_analytical, validate=False))

    print('VarQite 2')
    varqite2=varQITE(H2, params2, steps=n_steps, plot_fidelity=True)
    varqite2.initialize_circuits()
    omega2, d_omega=varqite2.state_prep(gradient_stateprep=True)
    list_omegas_fielity2=varqite2.fidelity_omega_list()    
    
    for j in range(len(list_omegas_fielity2)):
        params2=update_parameters(params2, list_omegas_fielity2[j])
        trace_circ2=create_initialstate(params2)
        DM2=DensityMatrix.from_instruction(trace_circ2)
        PT2 =partial_trace(DM2,[2,3])
        fidelities2_list.append(state_fidelity(PT2.data, H2_analytical, validate=False))

    print(f'H1: {fidelities1_list[-1]}, H2: {fidelities2_list[-1]}')

    plt.plot(list(range(0, len(fidelities1_list))),fidelities1_list, label='H1')
    plt.plot(list(range(0, len(fidelities2_list))),fidelities2_list, label='H2')
    
    #plt.scatter(list(range(0, len(fidelities1_list))),fidelities1_list, label='H1')
    #plt.scatter(list(range(0, len(fidelities2_list))),fidelities2_list, label='H2')
   
    plt.xlabel('Step')
    plt.ylabel('Fidelity')
    plt.legend()

    if name!=None:
        #plt.savefig('results/fidelity/'+name+'.png')
        plt.show()
    else:
        pass
    return


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












def main():
    #np.random.seed(1357)
    np.random.seed(123)

    number_of_seeds=1
    learningRate=0.1
    ite_steps=10
    epochs=100
    optimizing_method='Amsgrad'

    ansatz2=  [['ry',0, 0], ['ry',0, 1], ['ry',0, 2], ['ry',0, 3], 
            ['cx', 3,0], ['cx', 2, 3],['cx', 1, 2], ['ry', 0, 3],
            ['cx', 0, 1], ['ry', 0, 2], ['ry',np.pi/2, 0], 
            ['ry',np.pi/2, 1], ['cx', 0, 2], ['cx', 1, 3]]

    Ham2=   [[[0., 'z', 0], [0., 'z', 1]], 
            [[0., 'z', 0]], [[0., 'z', 1]]]

    #Ham2=     [[[1., 'z', 0], [1., 'z', 1]], [[-0.2, 'z', 0]], 
    #            [[-0.2, 'z', 1]], [[0.3, 'x', 0]], [[0.3, 'x', 1]]]

    p_data2=np.array([0.5, 0, 0, 0.5])
    
    p_data1=np.array([0.70, 0.30])

    ansatz1=    [['ry',0, 0],['ry',0, 1], ['cx', 1,0], ['cx', 0, 1],
                    ['ry',np.pi/2, 0],['ry',0, 1], ['cx', 0, 1]]
                    #[gate, value, qubit]
    Ham1=       [[[1., 'z', 0]]]

    #Testing the optimization
    Ham2=np.array(Ham2, dtype=object)
    Ham1=np.array(Ham1, dtype=object)

    start=time.time()

    #learningrate_investigation(1, Ham1, ansatz1, 15, p_data1, optimizing_method,l_r=0.1, steps=ite_steps)
    #learningrate_investigation(number_of_seeds, Ham1, ansatz1, epochs, p_data1, optimizing_method,l_r=0.005, steps=ite_steps, name='09')
    #learningrate_investigation(number_of_seeds, Ham1, ansatz1, epochs, p_data1, optimizing_method,l_r=0.002, steps=ite_steps, name='09')
    #multiple_simulations(1, Ham2, ansatz2, 25, p_data2, optimizing_method,l_r=0.1, steps=ite_steps, names='pdAH2')
    #multiple_simulations(number_of_seeds, Ham1, ansatz1, epochs, p_data1, optimizing_method,l_r=0.1, steps=ite_steps, names='')
    #multiple_simulations(number_of_seeds, Ham1, ansatz1, epochs, p_data1, optimizing_method,l_r=0.002, steps=ite_steps)


    #multiple_simulations(1, Ham2, ansatz2, 25, p_data2, optimizing_method,l_r=learningRate, steps=10)
    #multiple_simulations(number_of_seeds, Ham1, ansatz1, epochs, p_data1, optimizing_method,l_r=learningRate, steps=ite_steps)
    #multiple_simulations(number_of_seeds, Ham2, ansatz2, epochs, p_data2, optimizing_method,l_r=learningRate, steps=ite_steps)
    
    end_time=time.time()
    print(f'Final time: {end_time-start}')

    plot_fidelity(10)#, 'fidelity_H1_H2_new_0_001minC')
    #find_best_alpha(10, np.logspace(-4,1,5))


if __name__ == "__main__":
    main()





"""
Thoughts:
- Test encoding thing, do some math?
- Okay I might know something, well basicly the code doesnt update all the params,
only half of them actually. But how does it know if it is a controlled gate or not?

Todays list:
    - Fix rot indices loops
    - times -0.5j standard
    - + or - in sums
    - might be due to ridge?

    - where to put H gate

    -Something wrong with C since it is not arbitrary to the X gates hmmmm


H2 best: Non ridge, C:-=, temp.x
H1 best: Non ridge, C:+=, without temp.x

Next list:
    - Complete initialisation of the thing med labels and such
        - Complete C
    - Go through the TODO's
    - Gradietn with initialisation
    - Reproduce results/write code to produce it
    - Numba/paralellization
    - Fix the bug, probably have something to do with omega at index 2,3 and 7 being equal.
        - Okay listen up fam, I think I have some kind of idea to the source of the bug. Basicly V=U_N..U_1, but
        that means U_1 is applied first which makes sense for why C is reversed?
        - The key might be to know why C should be reversed
        - Maybe mixed the arguments some places?
    - Fix H to deal with multiple same coefficients
    - Run multiple circuits in paralell instead of separate
    - Do classical BM
    - Gradient too high, why? Normalize 0,1 instead of pi? learning rate?
    - Always increases, within the righ/left? where it is printed, might be that the parameters is set by running the method, 
    or that it should be copied some place

    - Okay I think I know: Just implement the gradients by ignoring the V_circ when it does not have a derivative and move the
    pauli gate over to the other side. Basicly do just the same thing as in C for 99 and 98 percent lol

    - CV from scikit not ridge CV

    - Why si gradient higher when the loss is lower? Might have a wrong sign. Not always like that
    Why is loss still good after a run? Something isnt reset, the gradient lacks after the run
        - I think this is due to some running of the grads in the algorithm scheme maybe?
        Try transposing it?
    
    - Check why they are the same depending on the coefficients in the gradient loop?
    - Find out what is pulling the predictions so high
    - Normalizing the quantum gates between -1 and 1?

    - Check on the parameters of adam, maybe better with ridge?

    -Params always same size, maybe try with amsgrad with + instead of minus in the x thing

    - Thoughts: dA is quiet high, and the inverse have some values which are quiet low high

    - Should probably normalize the shit
    -Gå gjennom dA step by step og finn ut hvorfor den er drithøy

    - Noe henger igjen som object fra tidligere

    Dagens oppgaver:
    - Undersøke gradient implementasjonsmetodene funker
        Tror det er noe feil med implementasjonen av de to siste coeffisientene
        18.45
    - Sjekke om det er bedre sånn som nå eller gjøre som andre rapporten
        Teste det
        19.30
    - Lage init kretser
        -dc circ og da circ
    - fikse lambda loopen
    - Maybe there is only one error in the article
    -Cyclic property?



    - If loss is larger than 0.5 reduce the alpha by factor 0.1
    Should C be negative?


    -Change three things: seed, momentum, saved momentum title


    - 0.5, 0.1, 0.05. 0.01, 0.005, 0.002
    - 0.7m,                         0.99

    2 loops

"""
