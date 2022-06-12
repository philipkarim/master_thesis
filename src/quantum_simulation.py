# Import library
import random
import numpy as np
import matplotlib.pyplot as plt
from utils import *
from varQITE import *
from H2_hamiltonian import h2_hamiltonian

def gs_VarITE(initial_H, ansatz, steps, final_time, distribution=['U', 1],):
    """
    Ground state energy functions
    """
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
    steps_list=[0.5, 0.25, 0.1, 0.05, 0.01, 0.005]
    
    distribution_list=[['U', 1]]
    #steps_list=[0.01]

    plt.figure()
    plt.axhline(y=-1.137275943080, color='black', label='Exact GS', linestyle='--')

    #Plot the energy using different stepsizes or distribution
    for d in distribution_list:
        for s in steps_list:
            temp_t_e=np.load('results/quantum_systems/H2_MT_latest_'+str(maxTime)+str(d[0])+str(d[1])+str(s)+'.npy')
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
    plt.legend(prop={'size': 8}, loc="upper right", ncol=2)   #plt.legend()
    plt.savefig('results/quantum_systems/energy_H2_search_N01_latest.pdf')
    plt.clf()
            

def main():
    """
    Main function
    """
    #Seeding the randomness
    np.random.seed(1111)

    #Defining parameters of VarITE
    time_step=0.25
    ite_steps=1500
    maxTime=15
    ite_steps= int(maxTime/time_step)
    rz_add=True

    #Search for optimal parameters?
    search_params=False
    
    #Just some global variables to keep track of variables
    compute_gs_energy.counter=0
    compute_gs_energy.computed_E=[]
    compute_gs_energy.energy_diff=[]
    compute_gs_energy.time_t=[]

    #hamiltonian_JW=h2_hamiltonian(0.7408481486)
    hamiltonian_JW, gs=h2_hamiltonian(1.4)
    #print(hamiltonian_JW)

    #4-qubit Hamiltonian
    jw_H_dict=hamiltonian_JW.terms
    coeffs=[]

    #Reducing the amount of qubits to 2 qubits
    names_of_dict=  [(), ((0, 'Z'), (1, 'Z')), ((0, 'Z'), (2, 'Z')),
                    ((1, 'Z'), (2, 'Z')), ((0, 'Z'), (3, 'Z')),
                    ((0, 'X'), (1, 'X') ,(2, 'Y'),(3, 'Y')),  ((0, 'Z'),), ((1, 'Z'),)]

    #Creating a list of Hamiltonian qubits
    for i in names_of_dict:
        coeffs.append(np.real(jw_H_dict[i]))
    
    #Coefficients of the Hamiltonian
    g1=coeffs[0]-2*coeffs[1]; g2=4*coeffs[5];    g3=-2*coeffs[2]+coeffs[3]+coeffs[4]
    g4=coeffs[6]-coeffs[7];   g5=g4
    
    #Final hydrogen Hamiltonian
    hydrogen_ham=[[[g1, 'z', 0], [g1, 'z', 0]],
                [[g2, 'x', 0], [g2, 'x', 1]],
                [[g3, 'z', 0], [g3, 'z', 1]],
                [[g4, 'z', 0]],[[g5, 'z', 1]]]

    #print(hydrogen_ham)

    #Include extra qubits for phase derivatives?
    if rz_add:
        hydrogen_ans= [['ry',0, 0],['ry',0, 1],['rz',0, 0],['rz',0, 1], ['cx', 0, 1],['ry',0, 0],['ry',0, 1],['rz',0, 0],['rz',0, 1], ['rz', 0, 2]]
    else:
        hydrogen_ans= [['ry',0, 0],['ry',0, 1],['rz',0, 0],['rz',0, 1], ['cx', 0, 1],['ry',0, 0],['ry',0, 1],['rz',0, 0],['rz',0, 1]]
    
    #List -> Array
    hydrogen_ham=np.array(hydrogen_ham, dtype=object)
    
    #Regular run
    if search_params is not True:
        gs_VarITE(hydrogen_ham, hydrogen_ans, ite_steps, final_time=maxTime, distribution=['U', 1])
    #Parameter search
    else:
        #Parameters to search for
        distribution_list=[['U', 1], ['U', 0.5], ['U', 0.1], ['U', 0.01],['N', 1], ['N', 0.5], ['N', 0.1], ['N', 0.01]]
        steps_list=[0.5, 0.25, 0.1, 0.05, 0.01, 0.005]

        compute_gs_energy.energies_array=np.zeros((len(distribution_list), len(steps_list)))
        compute_gs_energy.final_t=[]
        
        #Looping through the parameters
        for dist in distribution_list:
            for del_step in steps_list:
                compute_gs_energy.counter=0
                compute_gs_energy.time_t=[]
                compute_gs_energy.computed_E=[]
                gs_VarITE(hydrogen_ham, hydrogen_ans, int(maxTime/del_step), final_time=maxTime, distribution=dist)

                np.save('results/quantum_systems/H2_MT_latest_'+str(maxTime)+str(dist[0])+str(dist[1])+str(del_step), np.array([compute_gs_energy.time_t, compute_gs_energy.computed_E]))

if __name__ == "__main__":
    main()
    #plot_qe(15)

