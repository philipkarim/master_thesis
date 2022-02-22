from turtle import circle


def pairing_hamiltonian(n_states,delta,g):
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



circ_list=pairing_hamiltonian(4,1,1)

print(circ_list)

#rewrite on the form of my own hamiltonian