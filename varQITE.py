import numpy as np
from numpy.core.fromnumeric import trace
from numpy.lib.histograms import _unsigned_subtract
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
                
                A_mat_temp[i][j]=a_term
            
        return A_mat_temp

    def run_A2(self,first, sec):
        gate_label_i=self.trial_circ[first][0]
        gate_label_j=self.trial_circ[sec][0]

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
                    #First lets make the circuit:
                    temp_circ=V_circ.copy()

                    #Then we loop through the gates in U until we reach the sigma
                    for ii in range(first):
                        gate1=self.trial_circ[ii][0]
                        if gate1 == 'cx' or gate1 == 'cy' or gate1 == 'cz':
                            getattr(temp_circ, gate1)(1+self.trial_circ[ii][1], 1+self.trial_circ[ii][2])
                        else:
                            getattr(temp_circ, gate1)(self.trial_circ[ii][1], 1+self.trial_circ[ii][2])
        
                    temp_circ.x(0)
                    #Then we add the sigma
                    getattr(temp_circ, 'c'+pauli_names[i])(0,1+self.trial_circ[first][2])
                    #Add x gate                
                    temp_circ.x(0)

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

                    sum_A+=np.real(f_k_i[i]*f_l_j[j])*prediction

        return sum_A

    def get_C2(self):
        C_vec_temp=np.zeros(len(self.trial_circ))
        
        #Loops through the indices of A
        for i in range(len(C_vec_temp)):
            #For each gate 
            #range(1) if there is no controlled qubits?
                #Get f_i and f_j
                #Get, the sigma terms
                
                #4? dimension of hermitian or n pauliterms? 
            c_term=self.run_C2(i)
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
                    getattr(temp_circ, 'c'+pauli_names[i])(0,1+self.trial_circ[fir][2])

                    #Add x gate                
                    temp_circ.x(0)
                    #Continue the U_i gate:
                    for keep_going in range(fir, len(self.trial_circ)):
                        gate2=self.trial_circ[keep_going][0]
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
                    
                    sum_C+=np.real(f_k_i[i]*lambda_l[l])*prediction
                    #print(sum_C)

        return sum_C

    def get_dA(self, i_param):
        #Lets try to remove the controlled gates
        dA_mat_temp_i=np.zeros((len(self.trial_circ), len(self.trial_circ)))

        #Loops through the indices of A
        for p in range(len(self.trial_circ)):
            for q in range(len(self.trial_circ)):
                da_term=self.run_dA(p, q, i_param)
                
                dA_mat_temp_i[p][q]=da_term
        
        return dA_mat_temp_i
    
    def run_dA(self, p_index, q_index, i_theta):
        #Compute one term in the dA matrix
        sum_A_pq=0

        #A bit unsure about the len of this one
        for s in range(len(self.trial_circ)): #+1?
            #Okay no idea what the fuck this term even is, compute the formula
            #in the article by hand to find out.
            #the w depends on the timestep I think
            
            dCircuit_term_1=self.dA_circ([p_index, s], [q_index])
            dCircuit_term_2=self.dA_circ([p_index], [q_index, s])
            """
            Fix this, I dont know how dw should be computed
            """
            temp_dw=derivative_w[s][i_theta]

            #I guess the real and trace part automatically is computed 
            # in the cirquit.. or is it?
            sum_A_pq+=temp_dw*(dCircuit_term_1+dCircuit_term_2)
        
        return sum_A_pq

    def dA_circ(self, circ_1, circ_2):
        """
        Might be an error when the same indexes in the double derivative is simulated.
        Missing an rotational gate between the two sigmas, maybe can remove? Idk test it
        """
        #Does this work?      
        assert len(circ_2)==1 or len(circ_2)==2

        if len(circ_1)==1:
            gate_label_k_i=self.trial_circ[circ_1[0]][0]
            gate_label_l_i=self.trial_circ[circ_2[0]][0]
            gate_label_l_j=self.trial_circ[circ_2[1]][0]

            f_i=np.conjugate(get_f_sigma(gate_label_k_i))
            f_j=get_f_sigma(gate_label_l_i)
            f_k=get_f_sigma(gate_label_l_j)
            
            first_der=circ_1[0]
            sec_der=circ_2[0]
            thr_der=circ_2[1]
            
            if sec_der>thr_der:
                sec_der, thr_der=thr_der, sec_der

        elif len(circ_1)==2:
            gate_label_k_i=self.trial_circ[circ_1[0]][0]
            gate_label_k_j=self.trial_circ[circ_1[1]][0]
            gate_label_l_i=self.trial_circ[circ_2[0]][0]

            f_i=np.conjugate(get_f_sigma(gate_label_k_i))
            f_j=np.conjugate(get_f_sigma(gate_label_k_j))
            f_k=get_f_sigma(gate_label_l_i)

            first_der=circ_1[0]
            sec_der=circ_1[1]
            thr_der=circ_2[0]
            
            if first_der>sec_der:
                first_der, sec_der=sec_der, first_der

        else:
            print("Only implemented for double diff circs")
            exit()

        V_circ=encoding_circ('A', self.trial_qubits)

        pauli_names=['i', 'x', 'y', 'z']
        
        sum_dA=0

        for i in range(len(f_i)):
            for j in range(len(f_j)):
                for k in range(len(f_k)):
                    if f_i[i]==0 or f_j[j]==0 or f_k[k]==0:
                        pass
                    else:
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

                        if len(circ_1)==1:
                            temp_circ.x(0)
                            for keep_going in range(first_der, len(self.trial_circ)):   #+1? i think not
                                gate=self.trial_circ[keep_going][0]

                                if gate == 'cx' or gate == 'cy' or gate == 'cz':
                                    #print(keep_going, self.trial_circ[keep_going][1], 1+self.trial_circ[keep_going][2])
                                    getattr(temp_circ, gate)(1+self.trial_circ[keep_going][1], 1+self.trial_circ[keep_going][2])
                                else:
                                    getattr(temp_circ, gate)(self.trial_circ[keep_going][1], 1+self.trial_circ[keep_going][2])

                        else:
                            for sec in range(first_der, sec_der):
                                gate_sec=self.trial_circ[sec][0]
                                #print(gate1)
                                if gate_sec == 'cx' or gate_sec == 'cy' or gate_sec == 'cz':
                                    getattr(temp_circ, gate_sec)(1+self.trial_circ[sec][1], 1+self.trial_circ[sec][2])
                                else:
                                    getattr(temp_circ, gate_sec)(self.trial_circ[sec][1], 1+self.trial_circ[sec][2])

                            #Add second sigma gate, double check the index j
                            getattr(temp_circ, 'c'+pauli_names[j])(0,1+self.trial_circ[sec_der][2])

                            #Add x gate                
                            temp_circ.x(0)

                            for keep_going in range(sec_der, len(self.trial_circ)):   #+1? i think not
                                gate=self.trial_circ[keep_going][0]

                                if gate == 'cx' or gate == 'cy' or gate == 'cz':
                                    #print(keep_going, self.trial_circ[keep_going][1], 1+self.trial_circ[keep_going][2])
                                    getattr(temp_circ, gate)(1+self.trial_circ[keep_going][1], 1+self.trial_circ[keep_going][2])
                                else:
                                    getattr(temp_circ, gate)(self.trial_circ[keep_going][1], 1+self.trial_circ[keep_going][2])

                        
                        ###Done with the first "term"
                        if len(circ_1)==1:
                            for sec_term in range(sec_der):
                                gate=self.trial_circ[sec_term][0]

                                if gate == 'cx' or gate == 'cy' or gate == 'cz':
                                    getattr(temp_circ, gate)(1+self.trial_circ[sec_term][1], 1+self.trial_circ[sec_term][2])
                                else:
                                    getattr(temp_circ, gate)(self.trial_circ[sec_term][1], 1+self.trial_circ[sec_term][2])

                            getattr(temp_circ, 'c'+pauli_names[j])(0,1+self.trial_circ[sec_der][2])
                            
                            for third_term in range(sec_der, thr_der):
                                gate_sec=self.trial_circ[third_term][0]
                                if gate_sec == 'cx' or gate_sec == 'cy' or gate_sec == 'cz':
                                    getattr(temp_circ, gate_sec)(1+self.trial_circ[third_term][1], 1+self.trial_circ[third_term][2])
                                else:
                                    getattr(temp_circ, gate_sec)(self.trial_circ[third_term][1], 1+self.trial_circ[third_term][2])

                            #Add second sigma gate, double check the index j
                            getattr(temp_circ, 'c'+pauli_names[k])(0,1+self.trial_circ[thr_der][2])

                        else:
                            for third_term in range(thr_der):
                                gate=self.trial_circ[third_term][0]

                                if gate == 'cx' or gate == 'cy' or gate == 'cz':
                                    getattr(temp_circ, gate)(1+self.trial_circ[third_term][1], 1+self.trial_circ[third_term][2])
                                else:
                                    getattr(temp_circ, gate)(self.trial_circ[third_term][1], 1+self.trial_circ[third_term][2])

                            getattr(temp_circ, 'c'+pauli_names[k])(0,1+self.trial_circ[thr_der][2])

                        #Just have to add the last h gate
                        temp_circ.h(0)
                        temp_circ.measure(0,0)

                        """
                        Measures the circuit
                        """
                        prediction=run_circuit(temp_circ)

                        sum_dA+=np.real(f_i[i]*f_j[j]*f_k[k])*prediction

                        print(temp_circ)

        return sum_dA

    def get_dC(self, i_param):
        #Lets try to remove the controlled gates
        dC_vec_temp_i=np.zeros((len(self.trial_circ)))
        #Loops through the indices of A
        for p in range(len(self.trial_circ)):
            dc_term=self.run_dC(p, i_param)
            dC_vec_temp_i[p]=dc_term
        
        return dC_vec_temp_i
    
    def run_dC(self, p_index, i_theta):
        dCircuit_term_0=self.dC_circ0(p_index, i_theta)
        
        sum_C_p=0
        for s in range(len(self.trial_circ)): #+1?
            for i in range(len(self.hamil)):
                dCircuit_term_1=self.dC_circ1(p_index, i, s)
                dCircuit_term_2=self.dC_circ2(p_index, s, i)
                
                ## TODO: Fix this, I dont know how dw should be computed
                temp_dw=self.hamil[i][0]*derivative_w[s][i_theta]

            #I guess the real and trace part automatically is computed 
            # in the cirquit.. or is it?
                sum_C_p+=temp_dw*(dCircuit_term_1+dCircuit_term_2)
        
        return -dCircuit_term_0-sum_C_p

    def dC_circ0(self, p, j):
        gate_label_k_i=self.trial_circ[p][0]

        f_i=np.conjugate(get_f_sigma(gate_label_k_i))
        
        first_der=p

        V_circ=encoding_circ('C', self.trial_qubits)

        pauli_names=['i', 'x', 'y', 'z']
        
        sum_dC=0

        for i in range(len(f_i)):
            #Can a complex number be 0?
            if f_i[i]==0 or self.hamil[j][0]==0:
                pass
            else:
                #First lets make the circuit:
                temp_circ=V_circ.copy()

                #Then we loop through the gates in U untill we reach the sigma
                for ii in range(first_der):
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
                getattr(temp_circ, 'c'+pauli_names[i])(0,1+self.trial_circ[first_der][2])

                #Add x gate                
                temp_circ.x(0)
                #Continue the U_i gate:
                for keep_going in range(first_der, len(self.trial_circ)):
                    gate2=self.trial_circ[keep_going][0]
                    #print(gate1)
                    if gate2 == 'cx' or gate2 == 'cy' or gate2 == 'cz':
                        getattr(temp_circ, gate2)(1+self.trial_circ[keep_going][1], 1+self.trial_circ[keep_going][2])
                    else:
                        getattr(temp_circ, gate2)(self.trial_circ[keep_going][1], 1+self.trial_circ[keep_going][2])

                #The if statement is to not have controlled identity gates, since it is the first element but might fix this later on
                if self.hamil[j][1]!='i':
                    getattr(temp_circ, 'c'+self.hamil[j][1])(0,1+self.hamil[j][2])
                
                temp_circ.h(0)
                temp_circ.measure(0, 0)

                #print(temp_circ)
                prediction=run_circuit(temp_circ)
                
                sum_dC+=np.real(f_i[i]*self.hamil[j][0])*prediction
        
        #print(temp_circ)

        return sum_dC

    def dC_circ1(self, p, i_index, s):
        gate_label_i=self.trial_circ[p][0]
        gate_label_j=self.trial_circ[s][0]

        lmd=self.hamil[i_index][0]
        first=p
        sec=s

        f_k_i=np.conjugate(get_f_sigma(gate_label_i))
        f_l_j=get_f_sigma(gate_label_j)
        V_circ=encoding_circ('C', self.trial_qubits)

        pauli_names=['i', 'x', 'y', 'z']
        
        sum_dc=0
        for i in range(len(f_k_i)):
            for j in range(len(f_l_j)):
                if f_k_i[i]==0 or f_l_j[j]==0 or lmd==0:
                    pass
                else:
                    #First lets make the circuit:
                    temp_circ=V_circ.copy()

                    #Then we loop through the gates in U until we reach the sigma
                    for ii in range(first):
                        gate1=self.trial_circ[ii][0]
                        if gate1 == 'cx' or gate1 == 'cy' or gate1 == 'cz':
                            getattr(temp_circ, gate1)(1+self.trial_circ[ii][1], 1+self.trial_circ[ii][2])
                        else:
                            getattr(temp_circ, gate1)(self.trial_circ[ii][1], 1+self.trial_circ[ii][2])
        
                    temp_circ.x(0)
                    #Then we add the sigma
                    getattr(temp_circ, 'c'+pauli_names[i])(0,1+self.trial_circ[first][2])
                    #Add x gate                
                    temp_circ.x(0)

                    #Continue the U_i gate:
                    for keep_going in range(first, len(self.trial_circ)):
                        gate=self.trial_circ[keep_going][0]
   
                        if gate == 'cx' or gate == 'cy' or gate == 'cz':
                            getattr(temp_circ, gate)(1+self.trial_circ[keep_going][1], 1+self.trial_circ[keep_going][2])
                        else:
                            getattr(temp_circ, gate)(self.trial_circ[keep_going][1], 1+self.trial_circ[keep_going][2])
                    
                    #Adds the hamiltonian gate
                    #Should this be a controlled gate?
                    if self.hamil[i_index][1]!='i':
                        getattr(temp_circ, 'c'+self.hamil[i_index][1])(0,1+self.hamil[i_index][2])
                
                    for jj in range(sec):
                        gate3=self.trial_circ[jj][0]
                        #print(gate3)
                        if gate3 == 'cx' or gate3 == 'cy' or gate3 == 'cz':
                            getattr(temp_circ, gate3)(1+self.trial_circ[jj][1], 1+self.trial_circ[jj][2])
                        else:
                            getattr(temp_circ, gate3)(self.trial_circ[jj][1], 1+self.trial_circ[jj][2])

                    getattr(temp_circ, 'c'+pauli_names[j])(0,1+self.trial_circ[sec][2])
                    temp_circ.h(0)
                    temp_circ.measure(0,0)

                    """
                    Measures the circuit
                    """
                    prediction=run_circuit(temp_circ)

                    sum_dc+=np.real(f_k_i[i]*f_l_j[j])*prediction
        #print(temp_circ)
        
        return sum_dc

    def dC_circ2(self, p, s, i_index):
        gate_label_k_i=self.trial_circ[p][0]
        gate_label_k_j=self.trial_circ[s][0]

        f_i=np.conjugate(get_f_sigma(gate_label_k_i))
        f_j=np.conjugate(get_f_sigma(gate_label_k_j))

        first_der=p
        sec_der=s
        
        if first_der>sec_der:
            first_der, sec_der=sec_der, first_der

        V_circ=encoding_circ('C', self.trial_qubits)

        pauli_names=['i', 'x', 'y', 'z']
        
        sum_dC=0
        for i in range(len(f_i)):
            for j in range(len(f_j)):
                    if f_i[i]==0 or f_j[j]==0 or self.hamil[i_index][0]==0:
                        pass
                    else:
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

                            if gate_sec == 'cx' or gate_sec == 'cy' or gate_sec == 'cz':
                                getattr(temp_circ, gate_sec)(1+self.trial_circ[sec][1], 1+self.trial_circ[sec][2])
                            else:
                                getattr(temp_circ, gate_sec)(self.trial_circ[sec][1], 1+self.trial_circ[sec][2])

                        #Add second sigma gate, double check the index j
                        getattr(temp_circ, 'c'+pauli_names[j])(0,1+self.trial_circ[sec_der][2])

                        #Add x gate                
                        temp_circ.x(0)

                        for keep_going in range(sec_der, len(self.trial_circ)):   #+1? i think not
                            gate=self.trial_circ[keep_going][0]

                            if gate == 'cx' or gate == 'cy' or gate == 'cz':
                                getattr(temp_circ, gate)(1+self.trial_circ[keep_going][1], 1+self.trial_circ[keep_going][2])
                            else:
                                getattr(temp_circ, gate)(self.trial_circ[keep_going][1], 1+self.trial_circ[keep_going][2])
                        
                        #The if statement is to not have controlled identity gates, since it is the first element but might fix this later on
                        if self.hamil[i_index][1]!='i':
                            getattr(temp_circ, 'c'+self.hamil[i_index][1])(0,1+self.hamil[i_index][2])
                
                        temp_circ.h(0)
                        temp_circ.measure(0, 0)

                        #print(temp_circ)
                        prediction=run_circuit(temp_circ)
                        
                        sum_dC+=np.real(f_i[i]*f_j[j])*prediction

        #print(temp_circ)

        return sum_dC