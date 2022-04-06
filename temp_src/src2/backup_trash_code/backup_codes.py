def get_A2(V_list):
    A_mat_temp=np.zeros((len(V_list), len(V_list)))

    #Loops through the indices of A
    for i in range(len(V_list)):
        #For each gate 
        #range(1) if there is no controlled qubits?
        for j in range(len(V_list)):
            #Get f_i and f_j
            #Get, the sigma terms
            
            #4? dimension of hermitian or n pauliterms? 
            a_term=run_A2(V_list, i, j)
            
            A_mat_temp[i][j]=np.real(a_term)

    return A_mat_temp


def run_A2(U_list, first, sec):
    #gates_str=[['rx',0],['ry', 0]]

    gate_label_i=U_list[first][0]
    gate_label_j=U_list[sec][0]

    #print(U_list)

    f_k_i=np.conjugate(get_f_sigma(gate_label_i))
    f_l_j=get_f_sigma(gate_label_j)
    V_circ=encoding_circ('A')

    pauli_names=['i', 'x', 'y', 'z']
    
    sum_A=0
    for i in range(len(f_k_i)):
        for j in range(len(f_l_j)):
            if f_k_i[i]==0 or f_l_j[j]==0:
                pass
            else:
                #First lets make the circuit:
                temp_circ=V_circ.copy()
                
                """
                Implements it due to figure S1, is this right? U_i or U_j gates first, dagger?
                """
                #Then we loop through the gates in U until we reach the sigma
                for ii in range(i-1):
                    gate1=U_list[ii][0]
                    #print(gate1)
                    if gate1 == 'cx' or gate1 == 'cy' or gate1 == 'cz':
                        getattr(temp_circ, gate1)(U_list[ii][1], U_list[ii][2])
                    else:
                        getattr(temp_circ, gate1)(U_list[ii][1], 1)

                    #if len(U_list[ii])==2:
                    #    getattr(temp_circ, U_list[ii][0])(params_circ[ii], U_list[ii][1])
                    #elif len(U_list[ii])==3:
                    #    getattr(temp_circ, U_list[ii][0])(params_circ[ii], U_list[ii][1], U_list[ii][2])
                    #else:
                    #    print('Something is wrong, I can sense it')
                    #    exit()
                #Add x gate                
                temp_circ.x(0)
                #Then we add the sigma
                getattr(temp_circ, 'c'+pauli_names[i])(0,1)
                #Add x gate                
                temp_circ.x(0)
                #Continue the U_i gate:
                for keep_going in range(i-1, len(U_list)):
                    gate=U_list[keep_going][0]
                    #print(gate)
                    if gate == 'cx' or gate == 'cy' or gate == 'cz':
                        getattr(temp_circ, gate)(U_list[keep_going][1], U_list[keep_going][2])
                    else:
                        getattr(temp_circ, gate)(U_list[keep_going][1], 1)

                    """
                    if len(U_list[keep_going])==2:
                        getattr(temp_circ, U_list[keep_going][0])(params_circ[keep_going], 1)
                    elif len(U_list[keep_going])==3:
                        getattr(temp_circ, U_list[keep_going][0])(params_circ[keep_going], U_list[keep_going][1], U_list[keep_going][2])
                    else:
                        print('Something is wrong, I can feel it')
                        exit()
                    """
                for jj in range(j-1):
                    gate3=U_list[jj][0]
                    if gate3 == 'cx' or gate3 == 'cy' or gate3 == 'cz':
                        getattr(temp_circ, gate3)(U_list[jj][1], U_list[jj][2])
                    else:
                        getattr(temp_circ, gate3)(U_list[jj][1], 1)

                    """
                    if len(U_list[jj])==2:
                        getattr(temp_circ, U_list[jj][0])(params_circ[jj], 1)
                    elif len(U_list[jj])==3:
                        getattr(temp_circ, U_list[jj][0])(params_circ[jj], U_list[jj][1], U_list[jj][2])
                    else:
                        print('Something is wrong, I can feel it')
                        exit()
                    """

                getattr(temp_circ, 'c'+pauli_names[i])(0,1)
                temp_circ.h(0)
                temp_circ.measure(0,0)

                #print(temp_circ)

                """
                Measures the circuit
                """
                #print(temp_circ)
                prediction=run_circuit(temp_circ)

                sum_A+=f_k_i[i]*f_l_j[j]*prediction

    return sum_A

def get_C2(V_list, H_list):
    C_vec_temp=np.zeros(len(V_list))

    #Loops through the indices of A
    for i in range(len(V_list)):
        #For each gate 
        #range(1) if there is no controlled qubits?
            #Get f_i and f_j
            #Get, the sigma terms
            
            #4? dimension of hermitian or n pauliterms? 
        c_term=run_C2(V_list,H_list, i)
        #print(c_term)
        C_vec_temp[i]=c_term    

        #C_vec_temp[i]=np.real(c_term)

    return C_vec_temp


def run_C2(U_list, H_list, fir):
    gate_label_i=U_list[fir][0]

    f_k_i=np.conjugate(get_f_sigma(gate_label_i))
    """
    lambda is actually the coefficirents of the hamiltonian,
    but I think I should wait untill I actually have the
    Hamiltonian to implement is xD
    Also h_l are tensorproducts of the thing, find out how to compute tensor products optimized way
    """

    #The length might be longer than this
    #lambda_l=np.random.uniform(0,1,size=len(f_k_i))
    lambda_l=(np.array(H_list)[:, 0]).astype('complex')

    #arr = arr.astype('float64')

    #lambda_l=(np.array(H_list)[:, 0], dtype=np.complex)

    #This is just to have something there
    #h_l=['i', 'x', 'y', 'z']
    V_circ=encoding_circ('C')
    pauli_names=['i', 'x', 'y', 'z']
    
    sum_C=0

    for i in range(len(f_k_i)):
        for l in range(len(lambda_l)):
            #Can a complex number be 0?
            if f_k_i[i]==0 or lambda_l[l]==0:
                pass
            else:
                #First lets make the circuit:
                temp_circ=V_circ.copy()

                #Then we loop thorugh the gates in U untill we reach the sigma
                for ii in range(i-1):
                    gate1=U_list[ii][0]
                    #print(gate1)
                    if gate1 == 'cx' or gate1 == 'cy' or gate1 == 'cz':
                        getattr(temp_circ, gate1)(U_list[ii][1], U_list[ii][2])
                    else:
                        getattr(temp_circ, gate1)(U_list[ii][1], 1)

                #Add x gate                
                temp_circ.x(0)
                #Then we add the sigma
                #print(pauli_names[i])
                getattr(temp_circ, 'c'+pauli_names[i])(0,1)
                #Add x gate                
                temp_circ.x(0)
                #Continue the U_i gate:
                for keep_going in range(i-1, len(U_list)):
                    gate2=U_list[keep_going][0]
                    #print(gate1)
                    if gate2 == 'cx' or gate2 == 'cy' or gate2 == 'cz':
                        getattr(temp_circ, gate2)(U_list[keep_going][1], U_list[keep_going][2])
                    else:
                        getattr(temp_circ, gate2)(U_list[keep_going][1], 1)

                #Then add the h_l gate
                #The if statement is to not have controlled identity gates, since it is the first element but might fix this later on
                if H_list[l][1]!='I':
                    getattr(temp_circ, 'c'+H_list[l][1])(0,1)
                
                temp_circ.h(0)
                temp_circ.measure(0, 0)

                #print(temp_circ)
                prediction=run_circuit(temp_circ)
                
                sum_C+=f_k_i[i]*lambda_l[l]*prediction

                return sum_C