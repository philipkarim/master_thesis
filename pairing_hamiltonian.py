import numpy as np
import scipy.special as special
import itertools as it



def pairing_hamiltonian(n_states,d_energy,g):
    """
	This function gives out the pairing hamiltonian in a format recognized by the 
	methods developed in this thesis. 
	Inputs:
		n_states (int) - The number of basis states / qubits used in the model.
		delta (float) - One body interaction term
		g (float) - Two body interaction term
	Output:
		hamiltonian_list (list) - List containing each term of the hamiltonian in terms of factors, qubit number and gates.
	"""

    #TODO: Compute H_0 by myself actually just 0.5*times the level and add a z gate
    #TODO: compute the rest by myself
    #TODO: Rewrite H to fit my own circ
    #Try to get the FCI matrix from psi4 or make the FCI function myself

    H_gates=[]
    total_energy=0

    hamiltonian_list = []
    H_0 = []
    V = []
    phase_H_0 = 0
    phase_V = 0


    for i in range(n_states):
        level_energy=0.5*d_energy*(i-i%2)*0.5
        total_energy+=level_energy

        if level_energy!=0:
            H_gates.append([level_energy, [i, 'z']])
        
        #Interaction term diagonal:
        for i in range()
    
    print(H_gates)

    for p in range(0,int(n_states)):
        if (p+1 - 1 - (1 if (p+1)%2 == 0 else 0)) != 0 and d_energy != 0:
            phase_H_0 += 0.5*d_energy*0.5*(p+1 - 1 - (1 if (p+1)%2 == 0 else 0))
            H_0.append([0.5*d_energy*0.5*(p+1 - 1 - (1 if (p+1)%2 == 0 else 0)),[p,'z']])
        if g != 0 and p < int(n_states/2):
            phase_V += -(1/8)*g
            V.append([-(1/8)*g,[2*p,'z']])
            V.append([-(1/8)*g,[2*p+1,'z']])
            V.append([-(1/8)*g,[2*p,'z'],[2*p+1,'z']])
            for q in range(p+1,int(n_states/2)):
                V.append([-(1/16)*g,[2*p,'x'],[2*p+1,'x'],[2*q,'x'],[2*q+1,'x']])
                V.append([(1/16)*g,[2*p,'x'],[2*p+1,'x'],[2*q,'y'],[2*q+1,'y']])
                V.append([-(1/16)*g,[2*p,'x'],[2*p+1,'y'],[2*q,'x'],[2*q+1,'y']])
                V.append([-(1/16)*g,[2*p,'x'],[2*p+1,'y'],[2*q,'y'],[2*q+1,'x']])
                V.append([-(1/16)*g,[2*p,'y'],[2*p+1,'x'],[2*q,'x'],[2*q+1,'y']])
                V.append([-(1/16)*g,[2*p,'y'],[2*p+1,'x'],[2*q,'y'],[2*q+1,'x']])
                V.append([(1/16)*g,[2*p,'y'],[2*p+1,'y'],[2*q,'x'],[2*q+1,'x']])
                V.append([-(1/16)*g,[2*p,'y'],[2*p+1,'y'],[2*q,'y'],[2*q+1,'y']])
    hamiltonian_list = H_0
    #hamiltonian_list.extend(V)
    #hamiltonian_list.append([phase_H_0+phase_V])
    print(phase_H_0)
    
    
    return hamiltonian_list



circ_list=pairing_hamiltonian(6,3,18)

print(circ_list)

#rewrite on the form of my own hamiltonian

def pairing_hamiltonian_solution(n_states,delta,g):
    hamiltonian_list = []
    H_0 = []
    V = []
    phase_H_0 = 0
    phase_V = 0
    for p in range(0,int(n_states)):
        if (p+1 - 1 - (1 if (p+1)%2 == 0 else 0)) != 0 and delta != 0:
            phase_H_0 += 0.5*delta*0.5*(p+1 - 1 - (1 if (p+1)%2 == 0 else 0))
            H_0.append([0.5*delta*0.5*(p+1 - 1 - (1 if (p+1)%2 == 0 else 0)),[p,'z']])
        if g != 0 and p < int(n_states/2):
            phase_V += -(1/8)*g
            V.append([-(1/8)*g,[2*p,'z']])
            V.append([-(1/8)*g,[2*p+1,'z']])
            V.append([-(1/8)*g,[2*p,'z'],[2*p+1,'z']])
            for q in range(p+1,int(n_states/2)):
                V.append([-(1/16)*g,[2*p,'x'],[2*p+1,'x'],[2*q,'x'],[2*q+1,'x']])
                V.append([(1/16)*g,[2*p,'x'],[2*p+1,'x'],[2*q,'y'],[2*q+1,'y']])
                V.append([-(1/16)*g,[2*p,'x'],[2*p+1,'y'],[2*q,'x'],[2*q+1,'y']])
                V.append([-(1/16)*g,[2*p,'x'],[2*p+1,'y'],[2*q,'y'],[2*q+1,'x']])
                V.append([-(1/16)*g,[2*p,'y'],[2*p+1,'x'],[2*q,'x'],[2*q+1,'y']])
                V.append([-(1/16)*g,[2*p,'y'],[2*p+1,'x'],[2*q,'y'],[2*q+1,'x']])
                V.append([(1/16)*g,[2*p,'y'],[2*p+1,'y'],[2*q,'x'],[2*q+1,'x']])
                V.append([-(1/16)*g,[2*p,'y'],[2*p+1,'y'],[2*q,'y'],[2*q+1,'y']])
    hamiltonian_list = H_0
    hamiltonian_list.extend(V)
    hamiltonian_list.append([phase_H_0+phase_V])
    
    return hamiltonian_list

class PairingFCIMatrix:
    """
    Produces FCI matrix for Pairing Model.
    Utilize call function to return matrix
    """
    def __init__(self):
        pass
    def __call__(self,n_pairs,n_basis,delta,g):
        """
        Inputs:
            n_pairs (int) - Number of electron pairs
            n_basis (int) - Number of spacial basis states (spin-orbitals / 2)
            delta (float) - one-body strength
            g (float) - interaction strength
        Outputs:
            H_mat (array) - FCI matrix for pairing hamiltonian
            H_mat[0,0] (float) - Reference energy
        """
        n_SD = int(special.binom(n_basis,n_pairs))
        H_mat = np.zeros((n_SD,n_SD))
        S = self.stateMatrix(n_pairs,n_basis)
        for row in range(n_SD):
            bra = S[row,:]
            for col in range(n_SD):
                ket = S[col,:]
                if np.sum(np.equal(bra,ket)) == bra.shape:
                    H_mat[row,col] += 2*delta*np.sum(bra - 1) - 0.5*g*n_pairs
                if n_pairs - np.intersect1d(bra,ket).shape[0] == 1:
                    H_mat[row,col] += -0.5*g
        return(H_mat,H_mat[0,0])

    def stateMatrix(self,n_pairs,n_basis):
        L = []
        states = range(1,n_basis+1)
        for perm in it.permutations(states,n_pairs):
            L.append(perm)
        L = np.array(L)
        L.sort(axis=1)
        L = self.unique_rows(L)
        return(L)

    def unique_rows(self,a):
        a = np.ascontiguousarray(a)
        unique_a = np.unique(a.view([('', a.dtype)]*a.shape[1]))
        return unique_a.view(a.dtype).reshape((unique_a.shape[0], a.shape[1]))

pair_CI_mat=PairingFCIMatrix()
#Denne funker
ci_matrix=pair_CI_mat(1, 4, 1, 0)
#g=0 uten interaksjon

#print(ci_matrix)