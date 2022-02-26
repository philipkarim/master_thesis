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

    #TODO: Try to get the FCI matrix from psi4 or make the FCI function myself

    H_gates=[]
    total_energy=0

    for i in range(n_states):
        level_energy=0.5*d_energy*(i-i%2)*0.5
        total_energy+=level_energy

        if level_energy!=0:
            H_gates.append([level_energy, 'z', i])
        
        #Interaction term diagonal:
        if i<int(n_states/2) and g!=0:
            level_energy= -(1/8)*g
            total_energy+=level_energy
 
            H_gates.extend([[level_energy,'z', 2*i], [level_energy,'z', 2*i+1], \
            [level_energy,'z', 2*i], [level_energy,'z', 2*i+1]])

        #Non diagonal two body interaction
        for j in range(i+1,int(n_states/2)):
            l_e=g/16
            H_gates.extend([[-l_e, 'x',2*i],[-l_e, 'x',2*i+1],[-l_e, 'x',2*j],[-l_e, 'x',2*j+1], \
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




#print(circ_list)

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



#circ_list=pairing_hamiltonian(2,1,1)
#circ_list_sol=pairing_hamiltonian_solution(2,1,1)
#print(circ_list==circ_list_sol)



def ci_matrix_pairing(n_pairs,n_basis,delta,g):
    """
    Produces FCI matrix for Pairing Model.

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
    H0_mat = np.zeros((n_SD,n_SD))
    H1_mat = np.zeros_like(H0_mat)
    H_mat = np.zeros_like(H0_mat)

    S = stateMatrix(n_pairs,n_basis)
    print(S)

    #Without interaction along diagonal
    for i in range(n_SD):            
            H0_mat[i][i]+=2*delta*i

    #Interaction part
    for j in range(n_SD):
        for k in range(n_SD):
            if j==k:
                H1_mat[j][k]+=-0.5*g*n_pairs
            else:
                H1_mat[j][k]+=-0.5*g

    for row in range(n_SD):
        bra = S[row,:]
        for col in range(n_SD):
            ket = S[col,:]
            if np.sum(np.equal(bra,ket)) == bra.shape:
                H_mat[row,col] += 2*delta*np.sum(bra - 1) - 0.5*g*n_pairs
            if n_pairs - np.intersect1d(bra,ket).shape[0] == 1:
                H_mat[row,col] += -0.5*g
    return(H_mat, H0_mat, H1_mat)

def orbital_possibilities(n_pairs, n_basis):
    """
    Find possible ways of filling the orbitals
    """
    unique_perms=1
    #permutations=it.permutations(n_basis,n_pairs)
    
    #print(list(permutations))
    L = []
    states=range(1,n_basis+1)
    for i in it.permutations(states,n_pairs):
        L.append(i)
    print(L)
    return unique_perms

orbital_possibilities(3, 4)


    

def stateMatrix(n_pairs,n_basis):
    L = []
    states = range(1,n_basis+1)
    for perm in it.permutations(states,n_pairs):
        L.append(perm)
    L = np.array(L)
    L.sort(axis=1)
    L = unique_rows(L)
    return(L)

def unique_rows(a):
    a = np.ascontiguousarray(a)
    unique_a = np.unique(a.view([('', a.dtype)]*a.shape[1]))
    return unique_a.view(a.dtype).reshape((unique_a.shape[0], a.shape[1]))

#Denne funker
#print('Uten interaction')
ci_matrix_1, trash, trash=ci_matrix_pairing(3, 4, 1, 0)
#print(ci_matrix_1)

print('Kun interaction')
ci_matrix_2, trash, trash=ci_matrix_pairing(3, 4, 0, 1)
print(ci_matrix_2)
print(trash)

#print('Full interaction')
ci_matrix, own_0, own_1=ci_matrix_pairing(3, 4, 1, 1)

#g=0 uten interaksjon
print(own_0)
print(own_1)
print(own_0+own_1)
print(ci_matrix)
print(np.all(ci_matrix==(ci_matrix_1+ci_matrix_2)))
print(np.all(ci_matrix==(own_0+own_1)))



def FCI(n,l,h,v,ret_all=False):
    """
    n - Number of particles
    l - Number of spin orbitals
    h - One-body matrix elements
    v - Two-body matrix elements
    """
    n_SD = int(special.binom(l,n))
    H = np.zeros((n_SD,n_SD),dtype=complex)
    # Get occupied indices for all states
    S = configurations(n,l) 
    print(S)
    for row,bra in enumerate(S):
        for col,ket in enumerate(S):
            if np.sum(np.equal(bra,ket)) == bra.shape:
                # One-body contributions (Assumed diagonal)
                for p in bra:
                    H[row,col] += h[p,p]
                # Two-body contributions
                for p,q in it.combinations(bra,2):
                    H[row,col] += v[p,q,p,q]
            else:
                # Two-body contributions
                to_create = list(set(bra)-set(ket))
                to_annihilate = list(set(ket)-set(bra))
                eq = [i for i in bra if i not in to_create]
                # One orbital different
                if len(to_create) == 1 and len(to_annihilate) == 1:
                    for p in eq:
                        for c,a in zip(to_create,to_annihilate):
                            H[row,col] += v[p,c,p,a]
                # Two orbitals different
                elif len(to_create) == 2 and len(to_annihilate) == 2:
                    p,q = to_create
                    r,s = to_annihilate
                    H[row,col] += v[p,q,r,s]
    Es,Vs = np.linalg.eigh(H)
    if ret_all:
        return Es,Vs
    else:
        return Es[0]

def configurations(n,l):
    states = []
    for state in it.combinations(range(l),n):
        states.append([orb for orb in state])
    return np.asarray(states)





def ci_matrix_pairing_sol(n_pairs,n_basis,delta,g):
    """
    Produces FCI matrix for Pairing Model.

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
    S = stateMatrix_sol(n_pairs,n_basis)
    print(S)
    for row in range(n_SD):
        bra = S[row,:]
        for col in range(n_SD):
            ket = S[col,:]
            if np.sum(np.equal(bra,ket)) == bra.shape:
                H_mat[row,col] += 2*delta*np.sum(bra - 1) - 0.5*g*n_pairs
            if n_pairs - np.intersect1d(bra,ket).shape[0] == 1:
                H_mat[row,col] += -0.5*g
    return(H_mat,H_mat[0,0])

def stateMatrix_sol(n_pairs,n_basis):
    L = []
    states = range(1,n_basis+1)
    for perm in it.permutations(states,n_pairs):
        L.append(perm)
    L = np.array(L)
    L.sort(axis=1)
    L = unique_rows_sol(L)
    return(L)

def unique_rows_sol(a):
    a = np.ascontiguousarray(a)
    unique_a = np.unique(a.view([('', a.dtype)]*a.shape[1]))
    return unique_a.view(a.dtype).reshape((unique_a.shape[0], a.shape[1]))