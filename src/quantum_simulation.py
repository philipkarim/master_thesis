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

#sns.set_style("darkgrid")

def trainGS(H_operator, ansatz, n_epochs, n_steps=10, lr=0.1, optim_method='Adam', plot=True):
    init_params=np.array(copy.deepcopy(ansatz))[:, 1].astype('float')
    loss_list=[]
    epoch_list=[]
    tracing_q, rotational_indices=getUtilityParameters(ansatz)

    optim=optimize(H_operator, rotational_indices, tracing_q, learning_rate=lr, method=optim_method) ##Do not call this each iteration, it will mess with the momentum

    varqite_train=varQITE(H_operator, ansatz, steps=n_steps)
    
    time_intit=time.time()
    varqite_train.initialize_circuits()
    print(f'initialization time: {time.time()-time_intit}')
    for epoch in range(n_epochs):
        print(f'epoch: {epoch}')

        #Stops, memory allocation??
        ansatz=update_parameters(ansatz, init_params)
        omega, d_omega=varqite_train.state_prep(gradient_stateprep=True)

        optimize_time=time.time()

        ansatz=update_parameters(ansatz, omega)

        print(f' omega: {omega}')
        print(f' d_omega: {d_omega}')

        #Dansity matrix measure, measure instead of computing whole DM
        
        trace_circ=create_initialstate(ansatz)

        #circuit to matrix? Migh tonly get ancillas? or opposite? I have no idea
        getCircuitMatrix(trace_circ)
        #Transform both to matrices and compute them, shoulf be -1 hartree after enough steps
        


        DM=DensityMatrix.from_instruction(trace_circ)
        PT=partial_trace(DM,tracing_q)
        p_QBM=np.diag(PT.data).real.astype(float)
        
        print(f'p_QBM: {p_QBM}')
        loss=optim.cross_entropy_new(np.array([0.5, 0, 0, 0.5]),p_QBM)
        print(f'Loss: {loss, loss_list}')
        #Appending loss and epochs
        loss_list.append(loss)
        epoch_list.append(epoch)

        time_g_ps=time.time()
        gradient_qbm=optim.gradient_ps(H_operator, ansatz, d_omega)
        print(f'Time for ps: {time.time()-time_g_ps}')

        gradient_loss=optim.gradient_loss(np.array([0.5, 0, 0, 0.5]), p_QBM, gradient_qbm)
        print(f'gradient_loss: {gradient_loss}')

        H_coefficients=np.zeros(len(H_operator))

        for ii in range(len(H_operator)):
            H_coefficients[ii]=H_operator[ii][0][0]

        print(f'Old params: {H_coefficients}')
        new_parameters=optim.adam(H_coefficients, gradient_loss)

        #new_parameters=optim.gradient_descent_gradient_done(np.array(H)[:,0].astype(float), gradient_loss)
        print(f'New params {new_parameters}')
        #gradient_descent_gradient_done(self, params, lr, gradient):
        #TODO: Try this

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

    return np.array(loss_list), np.array(epoch_list)


def computeGS(n_sims, initial_H, ans, epochs,opt_met , l_r, steps, names):

    for i in range(n_sims):
        print(f'Seed: {i} of {n_sims}')
        H_init_val=np.random.uniform(low=-1.0, high=1.0, size=len(initial_H))
        print(H_init_val)
        
        for term_H in range(len(initial_H)):
            for qub in range(len(initial_H[term_H])):
                initial_H[term_H][qub][0]=H_init_val[term_H]
        
        saved_error, epochs_list =trainGS(initial_H, copy.deepcopy(ans), epochs, n_steps=steps, lr=l_r, optim_method=opt_met, plot=False)

    #print(l1_norm)
    #np.save('results/arrays/.npy', saved_error, l1_norm, np.array(qbm_list))


    plt.plot(epochs_list, saved_error)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    #plt.title('Bell state: Random seeds')
    #plt.savefig(str(l_r*1000)+str(len(target_data))+names+'_all.png')
    plt.show()
    #plt.clf()



def gs_VarITE(initial_H, ansatz, steps, final_time, names):
    #Initialising the ansatz with uniform parameters
    random.seed(3)
    for gate in ansatz:
        if gate[0][0]=='r':
            gate[1]=random.uniform(-1, 1)

    varqite=varQITE(initial_H, ansatz, maxTime=final_time, steps=steps, gs_computations=True)
    varqite.initialize_circuits()
    omega, d_omega=varqite.state_prep(gradient_stateprep=True)

    return 


def main():
    """
    Main function
    """
    np.random.seed(1111)

    #time_step=0.225
    ite_steps=1100
    maxTime=11
    rz_add=True

    g0=0.2252;  g1=0.3435;  g2=-0.4347
    g3=0.5716;  g4=0.0910;  g5=0.0910

    hydrogen_ham=[[[g0, 'z', 0], [g0, 'z', 0]],
                [[g1, 'z', 0]],
                [[g2, 'z', 1]],
                [[g3, 'z', 0], [g3, 'z', 1]],
                [[g4, 'y', 0], [g4, 'y', 1]],
                [[g5, 'x', 0], [g5, 'x', 1]]]
    if rz_add:
        hydrogen_ans= [['ry',0, 0],['ry',0, 1],['rz',0, 0],['rz',0, 1], ['cx', 0, 1],['ry',0, 0],['ry',0, 1],['rz',0, 0],['rz',0, 1], ['rz', 0, 2]]
    else:
        hydrogen_ans= [['ry',0, 0],['ry',0, 1],['rz',0, 0],['rz',0, 1], ['cx', 0, 1],['ry',0, 0],['ry',0, 1],['rz',0, 0],['rz',0, 1]]
    
    Ham1=       [[[1., 'z', 0]]]
    ansatz1=    [['ry',0, 0],['ry',0, 1], ['cx', 1,0], ['cx', 0, 1],
                ['ry',np.pi/2, 0],['ry',0, 1], ['cx', 0, 1]]
        
    Ham2=       [[[0., 'z', 0], [0., 'z', 1]], 
                [[0., 'z', 0]], [[0., 'z', 1]]]
    ansatz2=    [['ry',0, 0], ['ry',0, 1], ['ry',0, 2], ['ry',0, 3], 
                ['cx', 3,0], ['cx', 2, 3],['cx', 1, 2], ['ry', 0, 3],
                ['cx', 0, 1], ['ry', 0, 2], ['ry',np.pi/2, 0], 
                ['ry',np.pi/2, 1], ['cx', 0, 2], ['cx', 1, 3]]

    hydrogen_ham=np.array(hydrogen_ham, dtype=object)
    
    gs_VarITE(hydrogen_ham, hydrogen_ans, ite_steps, final_time=maxTime, names='hydrogen_testing')

if __name__ == "__main__":
    main()
