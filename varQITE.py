import numpy as np
from numpy.core.fromnumeric import trace
from utils import *

class varQITE:
    def __init__(self, hamil, trial_circ, maxTime=0.5, steps=10):
        """
        Class handling the variational quantum imaginary time evolution
        
        Args:
            hamil(list):        Hamiltonian as a list of gates and coefficients
            trial_circ(list):   Trial circuit as a list of gates, params and 
                                qubit placement
            max_Time(float):    Maximum time value for the propagation
            steps(int):         timesteps of varQITE
        """
        self.hamil=hamil
        self.trial_circ=trial_circ
        self.maxTime=maxTime
        self.steps=steps
        self.time_step=self.maxTime/self.steps


        n_qubits=0
        for i in range(len(trial_circ)):
            if n_qubits<trial_circ[i][2]:
                n_qubits=trial_circ[i][2]

        self.trial_qubits=n_qubits

    
    def state_prep(self):
        """
        Prepares an approximation for the gibbs states using imaginary time evolution 
        """
        #Basicly some input values, and then the returned values are the gibbs states.
        #Probably should make this an own function
        #And find out how to solve the differential equations

        #Input: page 6 in article algorithm first lines

        #initialisation of w for each theta, starting with 0?
        w_dtheta=np.zeros(len(self.hamil))
        omega_w=np.zeros(len(self.hamil))

        for t in np.arange(self.time_step, self.maxTime+1):   #+1?
            #Compute A(t) and C(t)
            #print("--------------------")
            #self.run_A2(0,1)
            #self.run_C2(5)
            #print("--------------------")
            A_mat2=np.copy(self.get_A2())
            C_vec2=np.copy(self.get_C2())
            #print(A_mat2)

            A_mat2, C_vec2=remove_constant_gates(self.trial_circ, A_mat2, C_vec2)

            print(A_mat2)
            print(C_vec2)            

            A_inv_temp=np.linalg.pinv(A_mat2)

            #print(A_inv_temp)


            omega_derivative=A_inv_temp@C_vec2

            #Solve A* derivative of \omega=C
            #No idea how to do it
            for i in range(len(self.hamil)): 
                #Compute the expression of the derivative
                dA_mat=np.copy(self.get_dA(i))
                dC_vec=np.copy(self.get_dC(i))

                print(dA_mat)
                #Now we compute the derivative of omega derivated with respect to
                #hamiltonian parameter
                #dA_mat_inv=np.inv(dA_mat)
                w_dtheta_dt= A_inv_temp@(dC_vec-dA_mat@omega_derivative)#* or @?

                w_dtheta[i]+=w_dtheta_dt*self.time_step
            
            omega_w[t+1]=omega_w[t]+omega_derivative*self.time_step
                #Solve A(d d omega)=d C -(d A)*d omega(t)
                
                #Compute:
                #dw(t)=dw(t-time_step)+d d w time_step
            #compute dw
            #w(t+time_step)=w(t)dw(t)time_step

        return omega_w, w_dtheta


    def get_A2(self):
        #Lets try to remove the controlled gates
        A_mat_temp=np.zeros((len(self.trial_circ), len(self.trial_circ)))

        #Loops through the indices of A
        for i in range(len(self.trial_circ)):
            #For each gate 
            #range(1) if there is no controlled qubits?
            for j in range(len(self.trial_circ)):
                #Get f_i and f_j
                #Get, the sigma terms
                
                #4? dimension of hermitian or n pauliterms? 
                a_term=self.run_A2(i, j)
                
                A_mat_temp[i][j]=np.real(a_term)
            
        return A_mat_temp

    def run_A2(self,first, sec):
        #gates_str=[['rx',0],['ry', 0]]

        gate_label_i=self.trial_circ[first][0]
        gate_label_j=self.trial_circ[sec][0]
        #print(gate_label_i, gate_label_j)
        #print(self.trial_circ)

        f_k_i=np.conjugate(get_f_sigma(gate_label_i))
        f_l_j=get_f_sigma(gate_label_j)
        V_circ=encoding_circ('A', self.trial_qubits)

        pauli_names=['i', 'x', 'y', 'z']
        
        sum_A=0
        for i in range(len(f_k_i)):
            for j in range(len(f_l_j)):
                if f_k_i[i]==0 or f_l_j[j]==0:
                    pass
                else:
                    #print(f_k_i[i], f_k_i[j])
                    #First lets make the circuit:
                    temp_circ=V_circ.copy()
                    """
                    Implements it due to figure S1, is this right? U_i or U_j gates first, dagger?
                    """
                    #Then we loop through the gates in U until we reach the sigma
                    for ii in range(first):
                        gate1=self.trial_circ[ii][0]
                        #print(gate1)
                        if gate1 == 'cx' or gate1 == 'cy' or gate1 == 'cz':
                            getattr(temp_circ, gate1)(1+self.trial_circ[ii][1], 1+self.trial_circ[ii][2])
                        else:
                            getattr(temp_circ, gate1)(self.trial_circ[ii][1], 1+self.trial_circ[ii][2])

                    #print(temp_circ)
                        #if len(self.trial_circ[ii])==2:
                        #    getattr(temp_circ, self.trial_circ[ii][0])(params_circ[ii], self.trial_circ[ii][1])
                        #elif len(self.trial_circ[ii])==3:
                        #    getattr(temp_circ, self.trial_circ[ii][0])(params_circ[ii], self.trial_circ[ii][1], self.trial_circ[ii][2])
                        #else:
                        #    print('Something is wrong, I can sense it')
                        #    exit()
                    #Add x gate                
                    temp_circ.x(0)
                    #Then we add the sigma
                    getattr(temp_circ, 'c'+pauli_names[i])(0,1+self.trial_circ[first][2])
                    #Add x gate                
                    temp_circ.x(0)

                    #print(temp_circ)
                    #Continue the U_i gate:
                    for keep_going in range(first, len(self.trial_circ)):
                        gate=self.trial_circ[keep_going][0]
                        #print(gate)
                        #print(gate)
                        if gate == 'cx' or gate == 'cy' or gate == 'cz':
                            #print(keep_going, self.trial_circ[keep_going][1], 1+self.trial_circ[keep_going][2])
                            getattr(temp_circ, gate)(1+self.trial_circ[keep_going][1], 1+self.trial_circ[keep_going][2])
                        else:
                            getattr(temp_circ, gate)(self.trial_circ[keep_going][1], 1+self.trial_circ[keep_going][2])
                        """
                        if len(self.trial_circ[keep_going])==2:
                            getattr(temp_circ, self.trial_circ[keep_going][0])(params_circ[keep_going], 1)
                        elif len(self.trial_circ[keep_going])==3:
                            getattr(temp_circ, self.trial_circ[keep_going][0])(params_circ[keep_going], self.trial_circ[keep_going][1], self.trial_circ[keep_going][2])
                        else:
                            print('Something is wrong, I can feel it')
                            exit()
                        """
                        #print(temp_circ)   
                    for jj in range(sec):
                        gate3=self.trial_circ[jj][0]
                        #print(gate3)
                        if gate3 == 'cx' or gate3 == 'cy' or gate3 == 'cz':
                            getattr(temp_circ, gate3)(1+self.trial_circ[jj][1], 1+self.trial_circ[jj][2])
                        else:
                            getattr(temp_circ, gate3)(self.trial_circ[jj][1], 1+self.trial_circ[jj][2])

                        """
                        if len(self.trial_circ[jj])==2:
                            getattr(temp_circ, self.trial_circ[jj][0])(params_circ[jj], 1)
                        elif len(self.trial_circ[jj])==3:
                            getattr(temp_circ, self.trial_circ[jj][0])(params_circ[jj], self.trial_circ[jj][1], self.trial_circ[jj][2])
                        else:
                            print('Something is wrong, I can feel it')
                            exit()
                        """

                    getattr(temp_circ, 'c'+pauli_names[j])(0,1+self.trial_circ[sec][2])
                    temp_circ.h(0)
                    temp_circ.measure(0,0)

                    #print(temp_circ)
                    #print(temp_circ)

                    """
                    Measures the circuit
                    """
                    #print(temp_circ)
                    prediction=run_circuit(temp_circ)

                    sum_A+=f_k_i[i]*f_l_j[j]*prediction

        return sum_A

    def get_C2(self):
        """
        counter=0
        for vec in range(len(self.trial_circ)):
            if self.trial_circ[vec][0][0]!='c':
                counter+=1
        """
        C_vec_temp=np.zeros(len(self.trial_circ))
        
        #Loops through the indices of A
        for i in range(len(C_vec_temp)):
            #For each gate 
            #range(1) if there is no controlled qubits?
                #Get f_i and f_j
                #Get, the sigma terms
                
                #4? dimension of hermitian or n pauliterms? 
            c_term=np.imag(self.run_C2(i))
            #print(c_term)
            C_vec_temp[i]=c_term    
            #print(C_vec_temp[i])
            #C_vec_temp[i]=np.real(c_term)

        return C_vec_temp


    def run_C2(self, fir):
        gate_label_i=self.trial_circ[fir][0]

        f_k_i=np.conjugate(get_f_sigma(gate_label_i))
        #print(f_k_i)
        """
        lambda is actually the coefficirents of the hamiltonian,
        but I think I should wait untill I actually have the
        Hamiltonian to implement is xD
        Also h_l are tensorproducts of the thing, find out how to compute tensor products optimized way
        """

        #The length might be longer than this
        #lambda_l=np.random.uniform(0,1,size=len(f_k_i))
        lambda_l=(np.array(self.hamil)[:, 0]).astype('float')
        #print(lambda_l)
        #arr = arr.astype('float64')

        #lambda_l=(np.array(self.hamil)[:, 0], dtype=np.complex)

        #This is just to have something there
        #h_l=['i', 'x', 'y', 'z']
        V_circ=encoding_circ('C', self.trial_qubits)
        #print(V_circ)
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

                    #Then we loop through the gates in U untill we reach the sigma
                    for ii in range(fir):
                        gate1=self.trial_circ[ii][0]
                        #print(gate1)
                        if gate1 == 'cx' or gate1 == 'cy' or gate1 == 'cz':
                            getattr(temp_circ, gate1)(1+self.trial_circ[ii][1], 1+self.trial_circ[ii][2])
                            pass
                        else:
                            getattr(temp_circ, gate1)(self.trial_circ[ii][1], 1+self.trial_circ[ii][2])

                    #Add x gate                
                    temp_circ.x(0)
                    #Then we add the sigma
                    #print(pauli_names[i])
                    getattr(temp_circ, 'c'+pauli_names[i])(0,1+self.trial_circ[fir][2])

                    #Add x gate                
                    temp_circ.x(0)
                    #Continue the U_i gate:
                    for keep_going in range(fir, len(self.trial_circ)):
                        gate2=self.trial_circ[keep_going][0]
                        #print(gate1)
                        if gate2 == 'cx' or gate2 == 'cy' or gate2 == 'cz':
                            getattr(temp_circ, gate2)(1+self.trial_circ[keep_going][1], 1+self.trial_circ[keep_going][2])
                        else:
                            getattr(temp_circ, gate2)(self.trial_circ[keep_going][1], 1+self.trial_circ[keep_going][2])

                    #Then add the h_l gate
                    #The if statement is to not have controlled identity gates, since it is the first element but might fix this later on
                    if self.hamil[l][1]!='i':
                        getattr(temp_circ, 'c'+self.hamil[l][1])(0,1+self.hamil[l][2])
                    
                    temp_circ.h(0)
                    temp_circ.measure(0, 0)

                    #print(temp_circ)
                    prediction=run_circuit(temp_circ)
                    
                    sum_C+=f_k_i[i]*lambda_l[l]*prediction
                    #print(sum_C)

        return sum_C


    def get_dA(self, i_param):
        #Lets try to remove the controlled gates
        dA_mat_temp_i=np.zeros((len(self.trial_circ), len(self.trial_circ)))

        #Loops through the indices of A
        for p in range(len(self.trial_circ)):
            for q in range(len(self.trial_circ)):
                da_term=self.run_dA(p, q, i_param)
                
                dA_mat_temp_i[p][q]=np.real(da_term)
        
        return
    
    def run_dA(self, p_index, q_index, i_theta):
        #Compute one term in the dA matrix
        sum_A_pq=0

        #A bit unsure about the len of this one
        for s in np.arange(self.time_step, self.maxTime+1): #+1?
            #Okay no idea what the fuck this term even is, compute the formula
            #in the article by hand to find out.
            #the w depends on the timestep I think
            
            test=self.dA_circ([p_index, s], [q_index])

            dCircuit_term_1=self.dA_circ([p_index, s], [q_index])
            dCircuit_term_2=self.dA_circ([p_index], [q_index, s])
            temp_dw=derivative_w[s][i_theta]

            #I guess the real and trace part automatically is computed 
            # in the cirquit.. or is it?
            sum_A_pq+=temp_dw*(dCircuit_term_1+dCircuit_term_2)
        
        return sum_A_pq

    def dA_circ(self, circ_1, circ_2):
        #gates_str=[['rx',0],['ry', 0]]

        if len(circ_1)==1:
            gate_label_k_i=self.trial_circ[circ_1[0]]
            f_k_i=np.conjugate(get_f_sigma(gate_label_k_i))

        elif len(circ_1)==2:
            gate_label_k_i=self.trial_circ[circ_1[0]][0]
            gate_label_k_j=self.trial_circ[circ_1[1]][0]

            f_k_i=np.conjugate(get_f_sigma(gate_label_k_i))
            f_k_j=np.conjugate(get_f_sigma(gate_label_k_j))

            first_der=circ_1[0]
            sec_der=circ_1[1]
            
            if first_der>sec_der:
                first_der, sec_der=sec_der, first_der

        else:
            print("Only implemented for double diff circs")
            exit()

        V_circ=encoding_circ('A', self.trial_qubits)

        pauli_names=['i', 'x', 'y', 'z']
        
        sum_dA=0

        for i in range(len(f_k_i)):
            for j in range(len(f_k_j)):
                if f_k_i[i]==0 or f_k_j[j]==0:
                    pass
                else:
                    #print(f_k_i[i], f_k_i[j])
                    #First lets make the circuit:
                    temp_circ=V_circ.copy()
   
                    #Then we loop through the gates in U until we reach the sigma
                    for ii in range(first_der):
                        gate1=self.trial_circ[ii][0]
                        #print(gate1)
                        if gate1 == 'cx' or gate1 == 'cy' or gate1 == 'cz':
                            getattr(temp_circ, gate1)(1+self.trial_circ[ii][1], 1+self.trial_circ[ii][2])
                        else:
                            getattr(temp_circ, gate1)(self.trial_circ[ii][1], 1+self.trial_circ[ii][2])
               
                    temp_circ.x(0)
                    #Then we add the sigma
                    getattr(temp_circ, 'c'+pauli_names[i])(0,1+self.trial_circ[first_der][2])

                    for sec in range(first_der, sec_der):
                        gate_sec=self.trial_circ[sec][0]
                        #print(gate1)
                        if gate_sec == 'cx' or gate_sec == 'cy' or gate_sec == 'cz':
                            getattr(temp_circ, gate_sec)(1+self.trial_circ[sec][1], 1+self.trial_circ[sec][2])
                        else:
                            getattr(temp_circ, gate_sec)(self.trial_circ[sec][1], 1+self.trial_circ[sec][2])

                    #Add second sigma gate
                    getattr(temp_circ, 'c'+pauli_names[i])(0,1+self.trial_circ[sec_der][2])

                    #Add x gate                
                    temp_circ.x(0)

                    #Then continue the circuit(remember that this is only the first derivative thing)

                    for keep_going in range(sec_der, len(self.trial_circ)):
                        gate=self.trial_circ[keep_going][0]
                        #print(gate)
                        #print(gate)
                        if gate == 'cx' or gate == 'cy' or gate == 'cz':
                            #print(keep_going, self.trial_circ[keep_going][1], 1+self.trial_circ[keep_going][2])
                            getattr(temp_circ, gate)(1+self.trial_circ[keep_going][1], 1+self.trial_circ[keep_going][2])
                        else:
                            getattr(temp_circ, gate)(self.trial_circ[keep_going][1], 1+self.trial_circ[keep_going][2])


                    #getattr(temp_circ, 'c'+pauli_names[j])(0,1+self.trial_circ[sec][2])
                    #temp_circ.h(0)
                    #temp_circ.measure(0,0)

                    #print(temp_circ)
                    #print(temp_circ)

                    """
                    Measures the circuit
                    """
                    #print(temp_circ)
                    #prediction=run_circuit(temp_circ)

                    #sum_A+=f_k_i[i]*f_l_j[j]*prediction

        return 1


    def assemble_circ(list):
        #Produce the whole term in (18.5) [[p_index, s], [q_index]] means the first term in (18.5)
        #list=[[p_index, s], [q_index]]

        #Hardcode the thing i guess
        #for i in range(len(list)):

        return
        #do the same thing as in the regular expression but instead do it twice with the "term"
            
        #first produce first circuit

        #Then produce the next circuit in the next loop. then compose them?
        #Maybe produce a circuit by sending in parameters in a function
        #and see if it is the first or second derivated?

