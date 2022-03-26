import numpy as np
import matplotlib.pyplot as plt
from qiskit.quantum_info import DensityMatrix, partial_trace, state_fidelity

from varQITE import *

def sim_plot_fidelity(n_steps, name=None, rz_add=False):
    """
    Function that computes and plots the fidelity between the computed
    states and the analytical states for the Hamiltonian 1 and
    Hamiltonian 2.
    
    Args:
        n_steps(int):   Imaginary time evolution steps.
        name(string):   Filename of plot if the user would want
                        to save the image.
        rz_add(bool):   If an rz gate should be added to count
                        for phase mismatch derivatives.
        """
    print('hmmm')
    if rz_add:
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
        
    for i in list_omegas_fielity1:
        params1=update_parameters(params1, i)
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

    for j in list_omegas_fielity2:
        params2=update_parameters(params2, j)
        if rz_add:
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

    plt.plot(list(range(0, len(fidelities1_list))),fidelities1_list, label='H1')
    plt.plot(list(range(0, len(fidelities2_list))),fidelities2_list, label='H2')
    
    plt.xlabel('Step')
    plt.ylabel('Fidelity')
    plt.legend()

    if name is not None:
        plt.savefig('results/fidelity/'+name+'.png')
    else:
        plt.show()
        #pass

def sim_lambda_fidelity_search(n_steps, lmbs, name=None, rz_add=False):
    """
    Computes the fidelity for a variety of lambda values in search of the best lambda
    value with both Ridge and Lasso regularization.

        n_steps(int):   Imaginary time evolution steps.
        lmbs(list):     List of lmbdas to search over
        name(string):   Filename of plot if the user would want
                        to save the image.
        rz_add(bool):   If an rz gate should be added to count
                        for phase mismatch derivatives.
    """


    H1_ridge_fidelities=[]
    H2_ridge_fidelities=[]
    H1_lasso_fidelities=[]
    H2_lasso_fidelities=[]

    for l in lmbs:
        h1_ridge, h2_ridge=compute_fidelity(n_steps, l, ridge, rz_add)
        h1_lasso, h2_lasso=compute_fidelity(n_steps, l, lasso, rz_add)

        H1_ridge_fidelities.append(h1_ridge)
        H2_ridge_fidelities.append(h2_ridge)
        H1_lasso_fidelities.append(h1_lasso)
        H2_lasso_fidelities.append(h2_lasso)

    plt.plot(lmbs,H1_ridge_fidelities, label=r'\H_1- Ridge')
    plt.plot(lmbs,H2_ridge_fidelities, label=r'\H_2- Ridge')
    plt.plot(lmbs,H1_lasso_fidelities, label=r'\H_1- Lasso')
    plt.plot(lmbs,H1_lasso_fidelities, label=r'\H_2- Lass')
    
    plt.xlabel('Step')
    plt.ylabel('Fidelity')
    plt.legend()

    if name is not None:
        plt.savefig('results/fidelity/'+name+'.png')
    else:
        plt.show()
        #pass

def compute_fidelity(n_steps, lmb, regularizer, rz_add=False):
    """
    Computes after n steps

    Args:
        n_steps(int):       Imaginary time evolution steps.
        lmb(float):         Value of lambda in regularizer
        regularizer(string):Ridge or Lasso  
        rz_add(bool):       If an rz gate should be added to count
                            for phase mismatch derivatives.
    
    Return(floats): fidelity after n steps
    """
    if rz_add:
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


    varqite1=varQITE(H1, params1, steps=n_steps, symmetrix_matrices=True, plot_fidelity=True)
    varqite2=varQITE(H2, params2, steps=n_steps, symmetrix_matrices=True, plot_fidelity=True)
    
    varqite1.initialize_circuits()
    varqite2.initialize_circuits()
    
    omega1, unused=varqite1.state_prep(gradient_stateprep=True)
    omega2, unused=varqite2.state_prep(gradient_stateprep=True)

    params1=update_parameters(params1, omega1)
    params2=update_parameters(params2, omega2)

    #Density matrix measure, measure instead of computing whole DM
    if rz_add==True:
        trace_circ1=create_initialstate(params1[:-1])
        trace_circ2=create_initialstate(params2[:-1])
    else:
        trace_circ1=create_initialstate(params1)
        trace_circ2=create_initialstate(params2)

    DM1=DensityMatrix.from_instruction(trace_circ1)
    DM2=DensityMatrix.from_instruction(trace_circ2)

    PT1=partial_trace(DM1,[1])
    PT2=partial_trace(DM2,[2,3])

    H_fidelity1=state_fidelity(PT1.data, H1_analytical, validate=False)
    H_fidelity2=state_fidelity(PT2.data, H2_analytical, validate=False)

    return H_fidelity1, H_fidelity2