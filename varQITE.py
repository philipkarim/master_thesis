import numpy as np

from sklearn.linear_model import Ridge

from qiskit.circuit import ParameterVector

from utils import *
import random
#import multiprocessing as mp
#import pathos
import itertools as it
#mp=pathos.helpers.mp
import time

#from pathos.pools import ProcessPool

#from numba import jit
#from numba.experimental import jitclass

#@jitclass
class varQITE:
    def __init__(self, hamil, trial_circ, rot_indices, n_qubit_param, maxTime=0.5, steps=10):
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
        self.trial_qubits=n_qubit_param
        
        #TODO: It is called indices not indexes
        self.rot_indexes=np.array(rot_indices, dtype=int)
    
    def initialize_circuits(self):
        """
        Initialising all circuits without parameters
        """

        """
        Creating circ A
        """
        A_circ= np.empty(shape=(len(self.trial_circ), len(self.trial_circ)), dtype=object)
        
        #Assuming there is only one circ per i,j, due to r? only having 1 element in f
        for i in range(len(self.trial_circ)):
            for j in range(len(self.trial_circ)):
                #Just the circuits
                A_circ[i][j]=self.init_A(i,j)
        #print(A_circ)


        #Remember to multiply with (0+0.5j)*(0-0.5j)


        """
        Creating circ C
        """
        #Trying to add everythinh to lists
        C_vec= np.empty(len(self.trial_circ), dtype=object)
        C_lmb_index= np.empty(len(self.trial_circ), dtype=object)
        C_vec_list=[]
        C_lmb=[]
        #Loops through the indices of A
        for i in range(len(self.trial_circ)):
            temp, temp_index =self.init_C(i)
            if len(temp)>0:
                C_vec_list.append(temp)
                C_lmb.append(temp_index)
                #print(i,temp)
                
            #C_vec[i], C_lmb_index[i] =self.init_C(i)
            #print(len(C_lmb))
        #print(C_vec)

  
        """
        Creating circ dA
        """




        """
        Creating circ dC
        """

        #print(np.where(C_vec==0.075732421875))

        self.A_init=A_circ
        self.C_init=C_vec_list
        self.C_lmb_index=C_lmb

        #return A_circ, C_vec #, dA_circ, dc_circ 

    def state_prep(self, gradient_stateprep=False):
        """
        Prepares an approximation for the gibbs states using imaginary time evolution 
        """
        omega_w=np.copy((np.array(self.trial_circ)[:, 1]).astype('float'))

        #print(f'init omega{omega_w}')
        self.dwdth=np.zeros((len(self.hamil), len(self.trial_circ)))

        labels=np.concatenate((omega_w[self.rot_indexes], omega_w[self.rot_indexes]), axis=0)
        lmbs=(np.array(self.hamil)[:, 0]).astype('float')
        #print(labels)

        for t in np.linspace(self.time_step, self.maxTime, num=self.steps):
            #print(f'VarQITE steps: {np.around(t, decimals=2)}/{self.maxTime}')
            
            #start_mat=time.time()
            A_mat2=np.copy(self.get_A2())
            #end_mat=time.time()
            C_vec2=np.copy(self.get_C2())

            """
            if i want to use this:   
                #A_mat_test=(self.A_init)
                #C_vec_test=self.C_init.copy()
                #Remember to multiply with (0+0.5j)*(0-0.5j)
                circ=[]
                start_loop=time.time()
                for ii in range(len(A_mat_test)):
                    for jj in range(len(A_mat_test[0])):
                        circ_test=A_mat_test[ii][jj]
                        
                        if circ_test!=None:
                            n_rotations=len(circ_test.parameters)
                            circ_test=circ_test.bind_parameters(labels[:n_rotations])
                            circ.append(circ_test)
                            circ_pred=run_circuit(circ_test)
                            A_mat_test[ii][jj]=circ_pred*0.25
                        else:
                            A_mat_test[ii][jj]=0.
                end_loop=time.time()
                #print(f'old mat {end_mat-start_mat}')
                #print(f'loop {end_loop-start_loop}')


                circ_pred=0

                for ii in range(len(C_vec_test)):
                    circ_test=C_vec_test[ii]
                    gate=self.trial_circ[ii][0]
                    if gate== 'cx' or gate == 'cy' or gate == 'cz':
                        C_vec_test[ii]=0
                    else:
                        for jj in range(len(C_vec_test[ii])):
        
                            temp_list=[]

                            n_rotations=len(circ_test[jj].parameters)
                            circ_test[jj]=circ_test[jj].bind_parameters(labels[:n_rotations])
                            #Just appending for option to run paralell circuits
                            temp_list.append(circ_test[jj])
                            #print(f'lambda is {self.hamil[:0][self.C_lmb_index[ii]]}')
                            circ_pred=run_circuit(circ_test[jj])

                            C_vec_test[ii]=circ_pred*0.5*lmbs[self.C_lmb_index[ii][jj]]
                        #else:
                                #C_vec_test[ii]=0.
                            
                            circ.append(temp_list)

                Just do it the old way, make it work and optimize later
                #print(A_mat2)
                print(C_vec2)
                C_vec2_2=np.zeros(len(self.trial_circ))
                C_vec2_2[self.rot_indexes]=C_vec_test
                print(C_vec2)
                print(C_vec2_2)
                print(np.all(C_vec2==C_vec2_2))     

                C_vec2=C_vec2_2
            """
            
            #print(A_mat2)
            #print(C_vec2)
            #Kan bruke Ridge regression på den inverterte
            #print(C_vec2)
            #print(A_mat2)
            A_inv_temp=np.linalg.pinv(A_mat2)
            #print(A_inv_temp)
            #ridge = Ridge(alpha=1.0)
            #A_inv_ridge=ridge.fit(A_mat2, C_vec2)


            #print(np.all(A_inv_temp=A_inv_ridge))
            omega_derivative=A_inv_temp@C_vec2
            #print(A_inv_temp)
            if gradient_stateprep==False:
                #print("This loop takes some time to complete")
                for i in range(len(self.hamil)):
                    #Compute the expression of the derivative
                    dA_mat=np.copy(self.get_dA(i))
                    dC_vec=np.copy(self.get_dC(i))
                    #print(dA_mat)
                    #print(dC_vec)
                    #Now we compute the derivative of omega derivated with respect to
                    #hamiltonian parameter
                    #dA_mat_inv=np.inv(dA_mat)
                    w_dtheta_dt= A_inv_temp@(dC_vec-dA_mat@omega_derivative)#* or @?
                    #print(dC_vec)
                    #print(dA_mat)
                    #print(w_dtheta_dt)
                    self.dwdth[i]+=w_dtheta_dt*self.time_step

            #*t instead of timestep->0.88 for H2, but bad for H1    
            omega_w+=(omega_derivative*self.time_step)
            #print(omega_w)

            #print(f'omega after step {omega_w}')

            #print(omega_derivative)
            #Update parameters
            self.trial_circ=update_parameters(self.trial_circ, omega_w)
            #print(omega_w)
                #Solve A(d d omega)=d C -(d A)*d omega(t)
                
                #Compute:
                #dw(t)=dw(t-time_step)+d d w time_step
            #compute dw
            #w(t+time_step)=w(t)dw(t)time_step

        return omega_w, self.dwdth

    def update_H(self, new_H):
        self.hamil=new_H

        return
    #@jit(nopython=True)
    def get_A2(self):
        #Lets try to remove the controlled gates
        #A_mat_temp=np.zeros((len(self.trial_circ), len(self.trial_circ)))



        #test_list=[]
        #start_zip=time.time()
        #x_array=range(len(self.trial_circ))
        #matrix_indices=([(x,y) for x in x_array for y in x_array])
        #results=np.array((it.starmap(self.run_A2, np.array(matrix_indices))))
        #A_mat_temp_zip=np.reshape(results, (len(self.trial_circ),len(self.trial_circ)))
        #end_zip=time.time()

        #print(f'zip {end_zip-start_zip}')
        #test_list=([(self.trial_circ, self.trial_qubits,x,y) for x in x_array for y in x_array])
        #print(test_list)
        #start_loop=time.time()
        A_mat_temp=np.zeros((len(self.trial_circ), len(self.trial_circ)))
        for i in range(len(self.trial_circ)):
            #For each gate 
            #range(1) if there is no controlled qubits?
            for j in range(len(self.trial_circ)):
                A_term=self.run_A2(i,j)
                #TODO: Changed the real part
                #print(A_term)
                A_mat_temp[i][j]=np.real(A_term)

        #end_loop=time.time()
        #print(f'Time loop {end_loop-start_loop}')

        #if (end_zip-start_zip)<(end_loop-start_loop):
        #    print('zip is faster')
        #else:
        #    print('loop is faster')

        #comparison=A_mat_temp==A_mat_temp_zip
        #print(comparison.all())

                #Get f_i and f_j
                #Get, the sigma terms
                #4? dimension of hermitian or n pauliterms? 
                #A_mat_temp[i][j]=self.run_A2(i, j)
                #print("Running loop..")
                #A_mat_temp[i][j]=pool.apply_async(self.run_A2, args=(i,j))
        #test_list=
                #test_list.append((i,j))
        
        #print(f' test_list{test_list}')

        #print(results)

        #Parallel here
        #print("Init pool")
        #print(mp.cpu_count())

        #with mp.Pool() as pool:
        #    print("test")
        #test_results=pool.starmap(self.run_A2, [(1,1),(2,1)])
        #L = pool.starmap(func, [(1, 1), (2, 1), (3, 1)])
        #M = pool.starmap(func, zip(a_args, repeat(second_arg)))
        #N = pool.map(partial(func, b=second_arg), a_args)

        #pool = mp.Pool(mp.cpu_count())
        #test_results=np.array(pool.starmap(run_A, test_list))
        #pool.close()
        #pool.join()

        #print(f'test_res {test_results}')
        #print(f'The results are {results}')

        #print(results)
        #print(A_mat_temp)

        #print(f'Same? {A_mat_temp==results}')

        #starmap


        #results = pool.starmap(howmany_within_range, [(row, 4, 8) for row in data])
        #pool.join()
        #print("closing pool")

        return A_mat_temp

    def run_A2(self,first, sec):
        gate_label_i=self.trial_circ[first][0]
        gate_label_j=self.trial_circ[sec][0]

        f_k_i=np.conjugate(get_f_sigma(gate_label_i))
        f_l_j=get_f_sigma(gate_label_j)

        V_circ=encoding_circ('A', self.trial_qubits)

        pauli_names=np.array(['i', 'x', 'y', 'z'])
        
        sum_A=0
        for i in range(len(f_k_i)):
            for j in range(len(f_l_j)):
                if f_k_i[i]==0 or f_l_j[j]==0:
                    pass
                else:
                    #First lets make the circuit:
                    temp_circ=V_circ.copy()
                    
                    """
                    #BEST UNTILL NOW I GUESS
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
                    
                    #I gotta reverse it here, going from opposite side
                    ##New loop: all the way to sec from N
                    
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

                    #print(temp_circ)
                    
                    """
                    if first<sec:
                        for ii in range(first):
                            gate1=self.trial_circ[ii][0]
                            if gate1 == 'cx' or gate1 == 'cy' or gate1 == 'cz':
                                getattr(temp_circ, gate1)(1+self.trial_circ[ii][1], 1+self.trial_circ[ii][2])
                            else:
                                getattr(temp_circ, gate1)(self.trial_circ[ii][1], 1+self.trial_circ[ii][2])
            
                        #temp_circ.x(0)
                        #Then we add the sigma
                        getattr(temp_circ, 'c'+pauli_names[i])(0,1+self.trial_circ[first][2])
                        #Add x gate
                        #temp_circ.x(0)

                        #Continue the U_i gate:
                        for keep_going in range(first, sec):
                            gate=self.trial_circ[keep_going][0]
                            if gate == 'cx' or gate == 'cy' or gate == 'cz':
                                getattr(temp_circ, gate)(1+self.trial_circ[keep_going][1], 1+self.trial_circ[keep_going][2])
                            else:
                                getattr(temp_circ, gate)(self.trial_circ[keep_going][1], 1+self.trial_circ[keep_going][2])

                    else:
                        for ii in range(first):
                            gate1=self.trial_circ[ii][0]
                            if gate1 == 'cx' or gate1 == 'cy' or gate1 == 'cz':
                                getattr(temp_circ, gate1)(1+self.trial_circ[ii][1], 1+self.trial_circ[ii][2])
                            else:
                                getattr(temp_circ, gate1)(self.trial_circ[ii][1], 1+self.trial_circ[ii][2])
            
                        #temp_circ.x(0)
                        #Then we add the sigma
                        getattr(temp_circ, 'c'+pauli_names[i])(0,1+self.trial_circ[first][2])
                        #Add x gate                
                        #temp_circ.x(0)

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
                        
                        #I gotta reverse it here, going from opposite side
                        ##New loop: all the way to sec from N
                        #TODO: Only thing to check up is this range, shuld it be reversed?
                        for jj in range(len(self.trial_circ)-1, sec-1, -1):
                            #print(jj)                       
                            gate3=self.trial_circ[jj][0]
                            if gate3 == 'cx' or gate3 == 'cy' or gate3 == 'cz':
                                getattr(temp_circ, gate3)(1+self.trial_circ[jj][1], 1+self.trial_circ[jj][2])
                            else:
                                getattr(temp_circ, gate3)(self.trial_circ[jj][1], 1+self.trial_circ[jj][2])
                        
                        #print(temp_circ)
                        #Forgott to add a controll gate how could I?
                        #getattr(temp_circ, 'c'+self.trial_circ[sec][0])(0,1+self.trial_circ[sec][2])
                    
                    getattr(temp_circ, 'c'+pauli_names[j])(0,1+self.trial_circ[sec][2])
                    temp_circ.h(0)
                    temp_circ.measure(0,0)
                    
                    #Measures the circuit
                    
                    #print(temp_circ)
                    #start=time.time()
                    
                    prediction=run_circuit(temp_circ)
                    #end=time.time()

                    #print(end-start)
                    #print(f_k_i[i],f_l_j[j])
                    #print(np.real(f_k_i[i]*f_l_j[j])*prediction)
                    #print(f_k_i[i]*f_l_j[j])

                    sum_A+=prediction*f_k_i[i]*f_l_j[j]

        return sum_A

    def init_A(self,first, sec):
        gate_label_i=self.trial_circ[first][0]
        gate_label_j=self.trial_circ[sec][0]

        f_k_i=np.conjugate(get_f_sigma(gate_label_i))
        f_l_j=get_f_sigma(gate_label_j)

        V_circ=encoding_circ('A', self.trial_qubits)

        pauli_names=np.array(['i', 'x', 'y', 'z'])
        
        #save_circ= np.empty(shape=(len(f_k_i),len(f_l_j)), dtype=object)
        
        p_vec = ParameterVector('Init_param', 2*len(self.trial_circ))

        counter=0
        for i in range(len(f_k_i)):
            for j in range(len(f_l_j)):
                if f_k_i[i]==0 or f_l_j[j]==0:
                    pass
                else:
                    counter+=1
                    #First lets make the circuit:
                    temp_circ=V_circ.copy()

                    #Then we loop through the gates in U until we reach the sigma
                    for ii in range(first):
                        gate1=self.trial_circ[ii][0]
                        if gate1 == 'cx' or gate1 == 'cy' or gate1 == 'cz':
                            getattr(temp_circ, gate1)(1+self.trial_circ[ii][1], 1+self.trial_circ[ii][2])
                        else:
                            getattr(temp_circ, gate1)(p_vec[ii], 1+self.trial_circ[ii][2])
        
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
                            getattr(temp_circ, gate)(p_vec[keep_going], 1+self.trial_circ[keep_going][2])
 
                    for jj in range(sec):
                        gate3=self.trial_circ[jj][0]
                        #print(gate3)
                        if gate3 == 'cx' or gate3 == 'cy' or gate3 == 'cz':
                            getattr(temp_circ, gate3)(1+self.trial_circ[jj][1], 1+self.trial_circ[jj][2])
                        else:
                            getattr(temp_circ, gate3)(p_vec[jj+len(self.trial_circ)], 1+self.trial_circ[jj][2])


                    getattr(temp_circ, 'c'+pauli_names[j])(0,1+self.trial_circ[sec][2])
                    temp_circ.h(0)
                    temp_circ.measure(0,0)

                    ## Adding the circs:
                    #print(first, sec)
                    #print(len(temp_circ.parameters))
                    #print(temp_circ)
                    
                    if counter>1:
                        print("Something is wrong, this should maximum be 1")
                        exit()

                    return temp_circ

    def init_C(self, fir):
        gate_label_i=self.trial_circ[fir][0]

        f_k_i=np.conjugate(get_f_sigma(gate_label_i))
        lambda_l=(np.array(self.hamil)[:, 0]).astype('float')

        #print(f_k_i)

        p_vec = ParameterVector('Init_param', 2*len(self.trial_circ))

        V_circ=encoding_circ('C', self.trial_qubits)
        #print(V_circ)
        pauli_names=['i', 'x', 'y', 'z']

        #Assumes that f_k_i only has one nonzero element
        circs=np.empty(len(lambda_l), dtype=object)
        lamba_l=[]
        circ_list=[]

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
                        else:
                            getattr(temp_circ, gate1)(p_vec[ii], 1+self.trial_circ[ii][2])

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
                            getattr(temp_circ, gate2)(p_vec[keep_going], 1+self.trial_circ[keep_going][2])

                    #Then add the h_l gate
                    #The if statement is to not have controlled identity gates, since it is the first element but might fix this later on
                    if self.hamil[l][1]!='i':
                        getattr(temp_circ, 'c'+self.hamil[l][1])(0,1+self.hamil[l][2])
                    
                    temp_circ.h(0)
                    temp_circ.measure(0, 0)

                    #print(fir, lambda_l[l])
                    #print(temp_circ)
                    #print(temp_circ)
                    #circs[i]=temp_circ
                    circ_list.append(temp_circ)
                    #circs.append(np.array(temp_circ, dtype=object))
                    lamba_l.append(l)
        
        #circs=np.empty(len(circs), dtype=object)
            #print(circ_list)
            #circ_array=np.array(circ_list, dtype=object)
            #print(np.array(circs))
            #print(len(circ_list))
        return circ_list, lamba_l

    def get_C2(self):
        C_vec_temp=np.zeros(len(self.trial_circ))
        
        #Loops through the indices of A
        for i in range(len(self.trial_circ)):
            #For each gate 
            #range(1) if there is no controlled qubits?
                #Get f_i and f_j
                #Get, the sigma terms
                
                #4? dimension of hermitian or n pauliterms? 
            #TODO: Changed this one too, the imag part
            c_term=self.run_C2(i)
            #print(c_term)
            C_vec_temp[i]=np.imag(c_term)
            #print(c_term)
                
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

        #TODO: Put the params of H in a self.variable
        lambda_l=(np.array(self.hamil)[:, 0]).astype('float')
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
                        #TODO: Kan bruke if ii in self.rotgate indexes, (slipper 2 'or' statements)
                        if gate1 == 'cx' or gate1 == 'cy' or gate1 == 'cz':
                            getattr(temp_circ, gate1)(1+self.trial_circ[ii][1], 1+self.trial_circ[ii][2])
                            pass
                        else:
                            getattr(temp_circ, gate1)(self.trial_circ[ii][1], 1+self.trial_circ[ii][2])

                    #Add x gate                
                    #temp_circ.x(0)
                    #Then we add the sigma
                    getattr(temp_circ, 'c'+pauli_names[i])(0,1+self.trial_circ[fir][2])
                    #Add x gate                
                    #temp_circ.x(0)

                    #Continue the U_i gate:
                    for keep_going in range(fir, len(self.trial_circ)):
                        gate2=self.trial_circ[keep_going][0]
                        if gate2 == 'cx' or gate2 == 'cy' or gate2 == 'cz':
                            getattr(temp_circ, gate2)(1+self.trial_circ[keep_going][1], 1+self.trial_circ[keep_going][2])
                        else:
                            getattr(temp_circ, gate2)(self.trial_circ[keep_going][1], 1+self.trial_circ[keep_going][2])

                    #Then add the h_l gate
                    #The if statement is to not have controlled identity gates, since it is the first element but might fix this later on
                    
                    #The h gate doesn't change the answer??
                    #TODO: Do I need this if statement?
                    if self.hamil[l][1]!='i':
                        #print(self.hamil[l][1])
                        #TODO: Rememeber to make the hamiltonian be used in the gradient circs 
                        # also, important that they match
                        
                        #Think this is right, remember to extend it for an arbitrary amount of qubits
                        if self.hamil[l][2]==0:
                            temp=self.hamil[l][2]+1
                        else: 
                            temp=self.hamil[l][2]+1+1
                        
                        getattr(temp_circ, 'c'+self.hamil[l][1])(0,temp)

                        #getattr(temp_circ, 'c'+self.hamil[l][1])(0,1+self.hamil[l][2])
                        #getattr(temp_circ, 'c'+self.hamil[l][1])(0,1)
                        #getattr(temp_circ, 'c'+self.hamil[l][1])(0,1+self.trial_circ[fir][2])

                    temp_circ.h(0)
                    temp_circ.measure(0, 0)

                    #print(temp_circ)
                    prediction=run_circuit(temp_circ)
                    #Imaginary here?
                    #print(f_k_i[i], lambda_l[l])
                    sum_C+=prediction*f_k_i[i]*lambda_l[l]
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
            #the w depends on the timestep I guess
            dCircuit_term_1=self.dA_circ([p_index, s], [q_index])
            dCircuit_term_2=self.dA_circ([p_index], [q_index, s])
            """
            Fix this, dCircuit 1 and 2 are the same absolute value, why?
            To fix this, I changed the sign of sum_A_pq
            """

            temp_dw=self.dwdth[i_theta][s]
            #I guess the real and trace part automatically is computed 
            # in the cirquit.. or is it?
            sum_A_pq+=temp_dw*(dCircuit_term_1+dCircuit_term_2)
        
        return sum_A_pq
    
    #TODO: add a if j<i statement to make circs mindre
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

                        #TODO: Make an if statement removing the extra gates when i<j
                        if len(circ_1)==1:
                            for sec_term in range(len(self.trial_circ)-1, sec_der-1, -1):
                            #for sec_term in range(sec_der):
                                gate=self.trial_circ[sec_term][0]

                                if gate == 'cx' or gate == 'cy' or gate == 'cz':
                                    getattr(temp_circ, gate)(1+self.trial_circ[sec_term][1], 1+self.trial_circ[sec_term][2])
                                else:
                                    getattr(temp_circ, gate)(self.trial_circ[sec_term][1], 1+self.trial_circ[sec_term][2])

                            getattr(temp_circ, 'c'+pauli_names[j])(0,1+self.trial_circ[sec_der][2])
                            
                            for third_term in range(sec_der-1, thr_der-1, -1):
                            #for third_term in range(sec_der, thr_der):
                                gate_sec=self.trial_circ[third_term][0]
                                if gate_sec == 'cx' or gate_sec == 'cy' or gate_sec == 'cz':
                                    getattr(temp_circ, gate_sec)(1+self.trial_circ[third_term][1], 1+self.trial_circ[third_term][2])
                                else:
                                    getattr(temp_circ, gate_sec)(self.trial_circ[third_term][1], 1+self.trial_circ[third_term][2])

                            #Add second sigma gate, double check the index j
                            getattr(temp_circ, 'c'+pauli_names[k])(0,1+self.trial_circ[thr_der][2])

                        else:
                            for third_term in range(len(self.trial_circ)-1, thr_der-1, -1):
                            #for third_term in range(thr_der):
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

                        #TODO: - or +?
                        sum_dA+=np.imag(f_i[i]*f_j[j]*f_k[k])*prediction

                        #print(temp_circ)

        return sum_dA

    #TODO: #Start here and see if these are alright
    def get_dC(self, i_param):
        #Lets try to remove the controlled gates
        dC_vec_temp_i=np.zeros((len(self.trial_circ)))
        #Loops through the indices of C
        for p in range(len(self.trial_circ)):
            dc_term=self.run_dC(p, i_param)
            dC_vec_temp_i[p]=dc_term
        
        return dC_vec_temp_i
    
    def run_dC(self, p_index, i_theta):
        dCircuit_term_0=self.dC_circ0(p_index, i_theta)
        #TODO: Check if dc is right 
        sum_C_p=0
        for s in range(len(self.trial_circ)): #+1?
            for i in range(len(self.hamil)):
                dCircuit_term_1=self.dC_circ1(p_index, i, s)
                dCircuit_term_2=self.dC_circ2(p_index, s, i)
                
                ## TODO: Fix this, I dont know how dw should be computed
                temp_dw=self.hamil[i][0]*self.dwdth[i_theta][s]

            #I guess the real and trace part automatically is computed 
            # in the cirquit.. or is it?
                #+ or - is this what is wrong?
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
                
                #TODO: - or +?
                sum_dC+=np.imag(f_i[i]*self.hamil[j][0])*prediction
        
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
                

                    for jj in range(len(self.trial_circ)-1, sec-1, -1):
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

                    #TODO: + or -?
                    sum_dc+=np.imag(f_k_i[i]*f_l_j[j])*prediction
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
                        
                        #TODO: + or minus?
                        sum_dC+=np.imag(f_i[i]*f_j[j])*prediction

        return sum_dC
