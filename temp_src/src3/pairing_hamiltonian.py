import numpy as np
import scipy.special as special
import itertools as it


#TODO: Not quiet done
def pairing_hamiltonian(n_states,non_e,int_e):
    """
    Produces the Jordan Wigner transformed Hamiltonian og the
    Hamiltonian pairing model.

        Parameters:
            n_states (int):  Amount of states in the model
            non_e (float):  One body energy/strength
            int_e (float):  Interaction energy/strength 
    
        Returns:
            jordan wigner transformed Hamiltonian (matrix)
    
	Inputs:
		n_states (int) - The number of possible states / qubits used in the model.
		delta (float) - One body interaction term
		g (float) - Two body interaction term
	Output:
		hamiltonian_list (list) - List containing each term of the hamiltonian in terms of factors, qubit number and gates.
	"""
    H_gates=[]
    total_energy=0

    for i in range(n_states):
        level_energy=0.5*non_e*(i-i%2)*0.5
        total_energy+=level_energy

        if level_energy!=0:
            H_gates.append([level_energy, 'z', i])
        
        #Interaction term diagonal:
        if i<int(n_states/2) and int_e!=0:
            level_energy= -(1/8)*int_e
            total_energy+=level_energy
 
            H_gates.extend([[level_energy,'z', 2*i], [level_energy,'z', 2*i+1], \
            [level_energy,'z', 2*i], [level_energy,'z', 2*i+1]])

        #Non diagonal two body interaction
        for j in range(i+1,int(n_states/2)):
            l_e=int_e/16
            H_gates.extend(\
            [[-l_e, 'x',2*i],[-l_e, 'x',2*i+1],[-l_e, 'x',2*j],[-l_e, 'x',2*j+1], \
            [l_e, 'x',2*i],[l_e, 'x',2*i+1],[l_e, 'y',2*j],[l_e, 'y',2*j+1],\
            [-l_e, 'x',2*i],[-l_e, 'y',2*i+1],[-l_e, 'x',2*j],[-l_e, 'y',2*j+1],\
            [-l_e, 'x',2*i],[-l_e, 'y',2*i+1],[-l_e, 'y',2*j],[-l_e, 'x',2*j+1],\
            [-l_e, 'y',2*i],[-l_e, 'x',2*i+1],[-l_e, 'x',2*j],[-l_e, 'y',2*j+1],\
            [-l_e, 'y',2*i],[-l_e, 'x',2*i+1],[-l_e, 'y',2*j],[-l_e, 'x',2*j+1],\
            [l_e, 'y',2*i],[l_e, 'y',2*i+1],[l_e, 'x',2*j],[l_e, 'x',2*j+1],\
            [-l_e, 'y',2*i],[-l_e, 'y',2*i+1],[-l_e, 'y',2*j],[-l_e, 'y',2*j+1]])

        #Cant bother to put every list inside a list
        new_H=[]
        for lists in H_gates:
            new_H.append([lists])

    return new_H


def ci_matrix_pairing(n_pairs,n_levels,non_e,int_e):
    """
    Produces FCI matrix for Pairing Model.

        Parameters:
            n_pairs (int):  Amount of fermion pairs 
            n_levels (int): Amount of possible energy levels
            non_e (float):  One body energy/strength
            int_e (float):  Interaction energy/strength 
    
        Returns:
            fci_matrix (matrix): Full configuration interaction matrix
    """
    n_SD = int(special.binom(n_levels,n_pairs))
    fci_matrix = np.zeros((n_SD,n_SD))

    #Makes lists of possible ways the energy orbitals can be filled
    permuted_levels=np.array(list(it.combinations(range(0,n_levels),n_pairs)))
    
    #Computing the energy/FCI elements
    for i, levels_1 in enumerate(permuted_levels):
        fci_matrix[i][i]+=2*non_e*np.sum(levels_1)
        for k, levels_2 in enumerate(permuted_levels):
            if np.all(levels_1==levels_2):
                #0.5 or 0.25?
                fci_matrix[i][k]-=0.5*int_e*n_pairs
            else:
                fci_matrix[i][k]-=0.5*int_e*len(np.intersect1d(levels_1, levels_2))
    
    return fci_matrix


#To match the result of the fys4480 slides use interaction g_interact=2 and d=0-->gs=-6
FCI_mat=ci_matrix_pairing(2, 4, 1, 1)
lam, eigv=np.linalg.eig(FCI_mat)
print(FCI_mat)
print(lam)

JW_Ham=pairing_hamiltonian(2,1,1)

print(JW_Ham)