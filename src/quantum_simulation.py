from importlib.metadata import distribution
import random
import copy
from turtle import color
import numpy as np
import qiskit as qk
from qiskit.circuit import Parameter, ParameterVector
from qiskit.quantum_info import DensityMatrix, partial_trace, state_fidelity
import time
import matplotlib.pyplot as plt
import pandas as pd

# Import the other classes and functions
from optimize_loss import optimize
from utils import *
from varQITE import *
from H2_hamiltonian import h2_hamiltonian

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



def gs_VarITE(initial_H, ansatz, steps, final_time, distribution=['U', 1],):
    #Initialising the ansatz with uniform parameters
    random.seed(3)
    for gate in ansatz:
        if gate[0][0]=='r':
            if distribution[0]=='U':
                gate[1]=random.uniform(-distribution[1], distribution[1])
            else:
                gate[1]=random.gauss(0, distribution[1])

    varqite=varQITE(initial_H, ansatz, maxTime=final_time, steps=steps, gs_computations=True, lmbs=np.logspace(-4,-2,7))
    varqite.initialize_circuits()
    omega, d_omega=varqite.state_prep(gradient_stateprep=True)

    return 

def plot_qe(maxTime=10):
    """
    Function to plot energy of quantum system
    """
    #distribution_list=[['U', 1], ['U', 0.5], ['U', 0.1], ['U', 0.01],['N', 1], ['N', 0.5], ['N', 0.1], ['N', 0.01]]
    #steps_list=[0.5, 0.25, 0.1, 0.05, 0.01, 0.005]

    distribution_list=[['U', 1], ['U', 0.5], ['U', 0.1], ['U', 0.01],['N', 1], ['N', 0.5], ['N', 0.1], ['N', 0.01]]
    steps_list=[0.5, 0.25]

    #distribution_list=[['U', 0.5], ['N', 0.5]]
    #distribution_list=[['U', 1], ['U', 0.1], ['U', 0.01],['N', 1], ['N', 0.1], ['N', 0.01]]
    #distribution_list=[['N', 0.1]]

    #steps_list=[0.1]

    plt.figure()
    plt.axhline(y=-1.13711706733, color='black', label='Exact GS', linestyle='--')
    energy_plot=True
    if energy_plot:
        for d in distribution_list:
            for s in steps_list:
                temp_t_e=np.load('results/quantum_systems/H2_MT_latest_'+str(maxTime)+str(d[0])+str(d[1])+str(s)+'.npy')
                #plt.plot(temp_t_e[0], temp_t_e[1], label=d[0]+'- '+str(d[1])+', '+r'$\delta=$'+str(s))
                if len(steps_list)==1:
                    if d[0]=='U':
                        label_n=r'$\mathcal{U}$[-'+str(d[1])+', '+str(d[1])+']'
                    else:
                        label_n=r'$\mathcal{N}$(0,$\sigma=$'+str(d[1])+')'
                else:
                    label_n=r'$\mathcal{\delta}=$'+str(s)

                plt.plot(temp_t_e[0], temp_t_e[1], label=label_n)#label=d[0]+'- '+str(d[1])+', '+r'$\delta=$'+str(s))
           
        plt.ylabel('Energy (Hartree)')
        plt.xlabel(r'Imaginary time $\tau$')
        plt.tight_layout()
        plt.legend(loc="center left")   #plt.legend()
        #plt.legend()
        plt.savefig('results/quantum_systems/energy_H2_search_N01_latest.pdf')
        plt.clf()
    
    else:
        data=[]
        coloumn=[]
        rows=[]

        #for i, s in enumerate(steps_list):
            #data_temp=[]
            #rows.append(r'\delta='+str(s))

            #if distribution_list[i][0]=='U':
                #coloumn.append(r'$\u$[-'+str(distribution_list[i][1])+', '+str(str(distribution_list[i][1]))+']')
            #else:
                #coloumn.append(r'$\N$(0,1)')
                #coloumn.append(r'$\N$(0, r$\sigma$='+str(distribution_list[i][1])+')')

            #for d in distribution_list:
                #data_temp.append(np.load('results/quantum_systems/H2_MT_'+str(maxTime)+str(d[0])+str(d[1])+str(s)+'.npy')[1][-1])
            #data.append(data_temp)
        
        print(coloumn)
        print(rows)
        print(data)

        data = [[1, 5, 10], [2, 6, 9], [3, 7, 8]]

        # Create the pandas DataFrame
        df = pd.DataFrame(data)
        
        # specifying column names
        df.columns = ['Col_1', 'Col_2', 'Col_3']
        
        # print dataframe.
        print(df, "\n")
        
        # transpose of dataframe
        df = df.transpose()
        print("Transpose of above dataframe is-\n", df)
        

def main():
    """
    Main function
    """
    np.random.seed(1111)

    #time_step=0.225
    ite_steps=1500
    maxTime=15
    rz_add=True

    search_params=True
    
    compute_gs_energy.counter=0
    compute_gs_energy.computed_E=[]
    compute_gs_energy.energy_diff=[]
    compute_gs_energy.time_t=[]

    #hamiltonian_JW=h2_hamiltonian(0.7408481486)
    hamiltonian_JW, gs=h2_hamiltonian(1.4)

    #print(hamiltonian_JW)

    jw_H_dict=hamiltonian_JW.terms

    coeffs=[]

    #for term, coefficient in h2_hamiltonian.terms.items():
    #        print(h2_hamiltonian.__class__(term, coefficient))

    names_of_dict=  [(), ((0, 'Z'), (1, 'Z')), ((0, 'Z'), (2, 'Z')),
                    ((1, 'Z'), (2, 'Z')), ((0, 'Z'), (3, 'Z')),
                    ((0, 'X'), (1, 'X') ,(2, 'Y'),(3, 'Y')),  ((0, 'Z'),), ((1, 'Z'),)]

    for i in names_of_dict:
        coeffs.append(np.real(jw_H_dict[i]))
    
    #print(coeffs)

    g1=coeffs[0]-2*coeffs[1]; g2=4*coeffs[5];    g3=-2*coeffs[2]+coeffs[3]+coeffs[4]
    g4=coeffs[6]-coeffs[7];   g5=g4

    hydrogen_ham=[[[g1, 'z', 0], [g1, 'z', 0]],
                [[g2, 'x', 0], [g2, 'x', 1]],
                [[g3, 'z', 0], [g3, 'z', 1]],
                [[g4, 'z', 0]],[[g5, 'z', 1]]]

    #print(hydrogen_ham)


    """
    g0=0.2252;  g1=0.3435;  g2=-0.4347
    g3=0.5716;  g4=0.0910;  g5=0.0910

    hydrogen_ham=[[[g0, 'z', 0], [g0, 'z', 0]],
                [[g1, 'z', 0]],
                [[g2, 'z', 1]],
                [[g3, 'z', 0], [g3, 'z', 1]],
                [[g4, 'y', 0], [g4, 'y', 1]],
                [[g5, 'x', 0], [g5, 'x', 1]]]
    """

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
    if search_params is not True:
        gs_VarITE(hydrogen_ham, hydrogen_ans, ite_steps, final_time=maxTime)
    else:
        #distribution_list=[['U', 1], ['U', 0.5], ['U', 0.1], ['U', 0.01],['N', 1], ['N', 0.5], ['N', 0.1], ['N', 0.01]]
        #steps_list=[0.5, 0.25, 0.1, 0.05, 0.01, 0.005]
        
        distribution_list=[['U', 1], ['U', 0.5], ['U', 0.1], ['U', 0.01],['N', 1], ['N', 0.5], ['N', 0.1], ['N', 0.01]]
        steps_list=[0.5, 0.25]
        

        compute_gs_energy.energies_array=np.zeros((len(distribution_list), len(steps_list)))
        compute_gs_energy.final_t=[]

        for dist in distribution_list:
            for del_step in steps_list:
                compute_gs_energy.counter=0
                compute_gs_energy.time_t=[]
                compute_gs_energy.computed_E=[]
                gs_VarITE(hydrogen_ham, hydrogen_ans, int(maxTime/del_step), final_time=maxTime, distribution=dist)

                np.save('results/quantum_systems/H2_MT_latest_'+str(maxTime)+str(dist[0])+str(dist[1])+str(del_step), np.array([compute_gs_energy.time_t, compute_gs_energy.computed_E]))

if __name__ == "__main__":
    #main()
    plot_qe(15)

