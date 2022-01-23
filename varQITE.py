import numpy as np
from numpy.core.numeric import zeros_like

from sklearn.linear_model import Ridge, RidgeCV

from qiskit.circuit import ParameterVector

from utils import *
import random
#import multiprocessing as mp
#import pathos
import itertools as it
#mp=pathos.helpers.mp
import time
import copy

#from pathos.pools import ProcessPool

#from numba import jit
#from numba.experimental import jitclass

#@jitclass
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
        self.best=True
        self.hamil=hamil
        self.trial_circ=trial_circ
        self.maxTime=maxTime
        self.steps=steps
        self.time_step=self.maxTime/self.steps        
        
        self.rot_loop=np.zeros(len(trial_circ), dtype=int)

        #TODO: It is called indices not indexes
        rotational_indices1=[]
        n_qubits_params1=0
        for i in range(len(trial_circ)):
            if trial_circ[i][0]=='cx' or trial_circ[i][0]=='cy' or trial_circ[i][0]=='cz':
                if n_qubits_params1<trial_circ[i][1]:
                    n_qubits_params1=trial_circ[i][1]
                self.rot_loop[i]=1

            else:
                rotational_indices1.append(i)

            if n_qubits_params1<trial_circ[i][2]:
                n_qubits_params1=trial_circ[i][2]

        self.rot_indexes=np.array(rotational_indices1, dtype=int)
        #print(f'rot_indexes: {self.rot_indexes}')
        self.trial_qubits=n_qubits_params1

    
    def initialize_circuits(self):
        """
        Initialising all circuits without parameters
        """

        """
        Creating circ A
        """
        A_circ= np.empty(shape=(len(self.rot_indexes), len(self.rot_indexes)), dtype=object)
        
        #Assuming there is only one circ per i,j, due to r? only having 1 element in f
        for i in range(len(self.rot_indexes)):
            for j in range(len(self.rot_indexes)):
                #Just the circuits
                A_circ[i][j]=self.init_A(self.rot_indexes[i],self.rot_indexes[j])
        #print(A_circ)
        """
        A_mat_temp=np.zeros((len(self.rot_indexes), len(self.rot_indexes)))
        for i in range(len(self.rot_indexes)):
            #For each gate 
            for j in range(len(self.rot_indexes)):
                A_term=self.run_A2(self.rot_indexes[i],self.rot_indexes[j])
                #TODO: Changed the real part
                derivative_const=0.25
                A_mat_temp[i][j]=A_term*derivative_const
        #Remember to multiply with (0+0.5j)*(0-0.5j)
        """

        #hamiltonian_params=np.array(self.hamil)[:, 0].astype('float')
        """
        C_circ= np.empty(shape=(len(self.hamil),len(self.rot_indexes)), dtype=object)
        
        #Assuming there is only one circ per i,j, due to r? only having 1 element in f
        for i in range(len(self.hamil)):
            for j in range(len(self.rot_indexes)):
                #Just the circuits
                C_circ[i][j]=self.init_C(i, self.rot_indexes[i])
        """
        """
        Creating circ C
        """
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
  
        """
        Creating circ dA
        """




        """
        Creating circ dC
        """

        #print(np.where(C_vec==0.075732421875))

        self.A_init=A_circ
        #self.C_init=C_circ
        #self.C_lmb_index=C_lmb

        #return A_circ, C_vec #, dA_circ, dc_circ
        return

    def state_prep(self, gradient_stateprep=False):
        """
        Prepares an approximation for the gibbs states using imaginary time evolution 
        """
        omega_w=copy.deepcopy((np.array(self.trial_circ)[:, 1]).astype('float'))

        #print(f'init omega{omega_w}')
        self.dwdth=np.zeros((len(self.hamil), len(self.trial_circ)))
        
        #lmbs=(np.array(self.hamil)[:, 0]).astype('float')
        #print(labels)

        A_mat=zeros_like(self.A_init, dtype='float')
        #C_vec=zeros_like(self.rot_indexes, dtype='float')

        for t in np.linspace(self.time_step, self.maxTime, num=self.steps):
            print(f'VarQITE steps: {np.around(t, decimals=2)}/{self.maxTime}')
            
            #start_mat=time.time()
            #print(self.A_init[0][3])
            #print(len(self.A_init[0][3].parameters))
            #n_rotations=len(self.A_init[0][3].parameters)
            #circ_test=self.A_init.tolist()
            #print(type(circ_test))
            #print(type(circ_test.tolist()))
            #circ_test=self.A_init[0][3].bind_parameters([1,1,1])

            #circ_test=circ_test.bind_parameters(labels[:n_rotations])
            #print(circ_test)
            #print(self.A_init[0][3])
            """
            Something like this?
            """
            #circ_pred=run_circuit(circ_test)
            #A_mat_test[ii][jj]=circ_pred*0.25
            
            #Fills a list of quantum circuits and binds values to them while at it 
            qc_list_A=[]
            qc_list_C=[]
            labels=np.concatenate((omega_w[self.rot_indexes], omega_w[self.rot_indexes]), axis=0)
            
            from qiskit.quantum_info import Statevector

            #Expression A: Binds the parameters to the circuits
            counter2=0
            for i in range(len(self.rot_indexes)):
                for j in range(len(self.rot_indexes)):
                    #Just the circuits
                    qc_list_A.append(self.A_init[i][j].bind_parameters(labels[:len(self.A_init[i][j].parameters)]))
                    counter2+=1

            
            #Expression C: Binds the parameters to the circuits
            """
            counter_C=0
            for i in range(len(self.hamil)):
                for j in range(len(self.rot_indexes)):
                    #Just the circuits
                    qc_list_C.append(self.C_init[i][j].bind_parameters(labels[len(self.rot_indexes):len(self.C_init[i][j].parameters):-1]))
                    counter_C+=1
                    #n_rotations=len(self.A_init[i][j].parameters)
                    #print('-------------')
                    #print(self.A_init[i][j])
                    #print(len(self.A_init[i][j].parameters), i, j)
            
            
                    
                    They produces the same circuits
                    trash, circ=self.run_A2(self.rot_indexes[i],self.rot_indexes[j])
                    print(i,j,Statevector.from_instruction(self.A_init[i][j].bind_parameters(\
                    labels[:len(self.A_init[i][j].parameters)])).equiv(Statevector.from_instruction(circ)))
            """

                    #print(i,j, counter2)
                    #print(qc_list[0]==circ)
                    #print(labels[:len(self.A_init[i][j].parameters)])

            #print(qc_list[0])
            #print(circ)


            #print(qc_list[0])
            #exit()
            #print(f'qc_list length {len(qc_list)}')
            #print(run_circuit(qc_list[0]))

            #matrix_values=run_circuit(qc_list, multiple_circuits=True)
            #print(f' Values from the matrix{matrix_values}')
            
           
            #Expression A: Running circuits produced above,  this can most certainly be done in parallel
            counter=0
            for i in range(len(self.rot_indexes)):
                for j in range(len(self.rot_indexes)):
                    #A_mat[i][j]=matrix_values[counter]*0.25
                    A_mat[i][j]=run_circuit(qc_list_A[counter])
                    counter+=1
            A_mat*=0.25

            """
            #Expression C: Running circuits produced above
            counter2_C=0
            for i in range(len(self.hamil)):
                for j in range(len(self.rot_indexes)):
                    #A_mat[i][j]=matrix_values[counter]*0.25
                    C_vec[j]+=run_circuit(qc_list_C[counter2_C])
                    counter2_C+=1
            C_vec*=0.5
            """


            """
            A_mat2=np.copy(self.get_A2())
            print(f'Matrix 1: {A_mat}')
            print(f'MAtrix 2: {A_mat2}')

            print(f'Is the matrices same?: {np.all(A_mat==A_mat2)}')
            A2_inv=np.linalg.pinv(A_mat2, hermitian=False)
            print(f'inverted matrix 2: {A2_inv}')
            A_inv=np.linalg.pinv(A_mat, hermitian=False)
            print('inverted matrix 1: {A_inv}')
            """
            
            #testing=self.run_A2(11, 11)

            #exit()
            #print(A_mat)        
            

            #A_mat2=A_mat
            #print(A_mat2)
            #print('----')
            #print(A_mat)

            #A_mat2=A_mat
            """Basicly the same matrices"""
            #print(np.all(A_mat2==A_mat))

            #exit()
            """
            #TODO fix this, always same values, does not get updated
            if len(omega_w)>8:
                print(omega_w[2],omega_w[3], omega_w[7])
            """

            #exit()
            #end_mat=time.time()
            #A_mat2=np.copy(self.get_A_from_init())
            
            #print(A_mat2)
            #A_mat2=A_mat
            
            A_mat2=np.copy(self.get_A2())
            C_vec2=np.copy(self.get_C2())
           
            #C_vec2=C_vec
            #A_mat2, C_vec2=self.last_try()
            #print(A_mat2)
            #print(C_vec2)
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
            #C_small=np.zeros(len(self.rot_indexes))
            #C_small=C_vec2[self.rot_indexes]
            #print(C_small)

            #A_small=np.zeros((len(self.rot_indexes), len(self.rot_indexes)))
            #for i in range(len(self.rot_indexes)):
            #print(A_mat2[self.rot_indexes][self.rot_indexes])

            #A_small=A_mat2[self.rot_indexes][self.rot_indexes]

            #for i in range(len(self.rot_indexes)):
            #    for j in range(len(self.rot_indexes)):
            #        A_small[i][j]=A_mat2[self.rot_indexes[i]][self.rot_indexes[j]]

            #print(A_small)



            #print(self.rot_indexes)
            #exit()
            #print(A_mat2)
            #print(C_vec2)
            #Kan bruke Ridge regression på den inverterte
            #print(C_vec2)
            #print(A_mat2)
            
            """Full matrix"""   
            ridge_inv=True
            if ridge_inv==False:
                #TODO: Might be something wrong here?
                A_inv=np.linalg.pinv(A_mat2, hermitian=False)
                omega_derivative_temp=A_inv@C_vec2
            else:
                I=np.eye(A_mat2.shape[1])
                regr_cv = RidgeCV(alphas= np.logspace(-16, -2))
                model_cv = regr_cv.fit(A_mat2, C_vec2)
                #This is better
                #TODO: Add try/catch statement with inv/pinv?
                omega_derivative_temp=np.linalg.inv(A_mat2.T @ A_mat2 + model_cv.alpha_*I) @ A_mat2.T @ C_vec2
                #omega_derivative_temp=regr_cv.coef_
                #print(f'best alpha: {model_cv.alpha_}')

                #rr.fit(A_mat2, C_vec2) 
                #pred_train_rr= rr.predict(A_mat2)

            omega_derivative=np.zeros(len(self.trial_circ))
            omega_derivative[self.rot_indexes]=omega_derivative_temp
            #print(f'Is this large also? {omega_derivative_temp}')

            if gradient_stateprep==False:
                #print("This loop takes some time to complete")
                for i in range(len(self.hamil)):
                    #Compute the expression of the derivative
                    #TODO: Deep copy takes a lot of time, fix this
                    dA_mat=np.copy(self.get_dA(i))
                    dC_vec=np.copy(self.get_dC(i))

                    if ridge_inv==False:
                        w_dtheta_dt=A_inv@(dC_vec-dA_mat@omega_derivative_temp)#* or @?
                    else:
                        I=np.eye(A_mat2.shape[1])
                        regr_cv_der = RidgeCV(alphas= np.logspace(-16, -2))
                        y_target=dC_vec-dA_mat@omega_derivative_temp
                        model_cv_der = regr_cv_der.fit(A_mat2, y_target)
                        #This is better
                        #TODO: Add try/catch statement with inv/pinv?
                        w_dtheta_dt=np.linalg.inv(A_mat2.T @ A_mat2 + model_cv_der.alpha_*I) @ A_mat2.T @ y_target



                    #Now we compute the derivative of omega derivated with respect to
                    #hamiltonian parameter
                    #dA_mat_inv=np.inv(dA_mat)
                    #print(dC_vec)
                    #print(dA_mat)
                    #print(w_dtheta_dt)
                    self.dwdth[i][self.rot_indexes]+=w_dtheta_dt*self.time_step
                    #print(f'w_dtheta: {w_dtheta_dt}')

            #*t instead of timestep->0.88 for H2, but bad for H1    
            omega_w+=(omega_derivative*self.time_step)
            #print(omega_w)

            #print(f'omega after step {omega_w}')

            #print(omega_derivative)
            #Update parameters
            #print(self.trial_circ)
            #print(omega_w)
            self.trial_circ=update_parameters(self.trial_circ, omega_w)
            #print(self.trial_circ)
            #exit()
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
    def get_A_from_init(self):

        A_mat_temp=np.zeros((len(self.rot_indexes), len(self.rot_indexes)))
        for i in range(len(self.rot_indexes)):
            #For each gate 
            for j in range(len(self.rot_indexes)):

                A_term=self.run_A2(self.rot_indexes[i],self.rot_indexes[j])
                #TODO: Changed the real part
                derivative_const=0.25
                A_mat_temp[i][j]=A_term*derivative_const

        return A_mat_temp
        
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
        """
        A_mat_temp=np.zeros((len(self.trial_circ), len(self.trial_circ)))
        for i in range(len(self.trial_circ)):
            #For each gate 
            #range(1) if there is no controlled qubits?
            for j in range(len(self.trial_circ)):
                A_term=self.run_A2(i,j)1
                #TODO: Changed the real part
                A_mat_temp[i][j]=np.real(A_term)
        """
        
        A_mat_temp=np.zeros((len(self.rot_indexes), len(self.rot_indexes)))
        for i in range(len(self.rot_indexes)):
            #For each gate 
            for j in range(len(self.rot_indexes)):
                A_term=self.run_A2(self.rot_indexes[i],self.rot_indexes[j])
                #TODO: Changed the real part
                derivative_const=0.25
                A_mat_temp[i][j]=A_term*derivative_const
        

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
        V_circ=encoding_circ('A', self.trial_qubits)
        temp_circ=V_circ.copy()

        for i, j in enumerate(self.rot_loop[:first]):
            getattr(temp_circ, self.trial_circ[i][0])(self.trial_circ[i][1]+j, 1+self.trial_circ[i][2])
            #print(i)
        
        #TODO: Then we add the sigma and x gate(?)
        #temp_circ.x(0)
        getattr(temp_circ, 'c'+self.trial_circ[first][0][-1])(0,1+self.trial_circ[first][2])
        #temp_circ.x(0)

        if first<sec:
            #Continue the U_i gate:

            #list_to_fix_indexes=np.concatenate(self.rot_loop[:first], first)
            for ii, jj in enumerate(self.rot_loop[first:sec], start=first):
                getattr(temp_circ, self.trial_circ[ii][0])(self.trial_circ[ii][1]+jj, 1+self.trial_circ[ii][2])
                #print(first, sec, self.trial_circ[ii][0], self.trial_circ[ii][1]+jj)
                #print(self.rot_loop)

        else:
            #Continue the U_i gate:
            for ii, jj in enumerate(self.rot_loop[first:], start=first):
                getattr(temp_circ, self.trial_circ[ii][0])(self.trial_circ[ii][1]+jj, 1+self.trial_circ[ii][2])
                #print(ii, self.trial_circ[ii][0], self.trial_circ[ii][1]+jj, 1+self.trial_circ[ii][2])
            #TODO: Only thing to check up is this range, shuld it be reversed?
            #Something wrong here, I can feel it
            for kk in range(len(self.trial_circ)-1, sec-1, -1):
                #print(kk, self.trial_circ[kk][0], self.trial_circ[kk][1]+self.rot_loop[kk], 1+self.trial_circ[kk][2])
                #print(kk, self.trial_circ[kk][1], self.rot_loop[kk], 1+self.trial_circ[kk][2])
                #print(self.trial_circ[kk][1]+self.rot_loop[kk], 1+self.trial_circ[kk][2])
                getattr(temp_circ, self.trial_circ[kk][0])(self.trial_circ[kk][1]+self.rot_loop[kk], 1+self.trial_circ[kk][2])

        #TODO: add x?
        temp_circ.x(0)
        getattr(temp_circ, 'c'+self.trial_circ[sec][0][-1])(0,1+self.trial_circ[sec][2])
        temp_circ.x(0)

        temp_circ.h(0)
        #TODO Add this back
        temp_circ.measure(0,0)
        #print(temp_circ)
        #print(f'---------{first}{sec}-------')
        #print(temp_circ)
        #print('_-----------------------')
        prediction=run_circuit(temp_circ)
 
        sum_A=prediction

        #if first==3 or sec==3:
            #print(first, sec)
            #print(temp_circ)
        #print(f'prediction {prediction}')

        return sum_A

    def init_A(self,first, sec):
        #TODO: Remember to switch everything I switch here, elsewhere
        V_circ=encoding_circ('A', self.trial_qubits)
        temp_circ=V_circ.copy()
        
        p_vec = ParameterVector('Init_param', 2*len(self.rot_indexes))

        for i, j in enumerate(self.rot_loop[:first]):
            if i in self.rot_indexes:
                name=p_vec[i]
            else:
                name=self.trial_circ[i][1]+j
            getattr(temp_circ, self.trial_circ[i][0])(name, 1+self.trial_circ[i][2])
        
        getattr(temp_circ, 'c'+self.trial_circ[first][0][-1])(0,1+self.trial_circ[first][2])

        if first<sec:
            #Continue the U_i gate:
            for ii, jj in enumerate(self.rot_loop[first:sec], start=first):
                if ii in self.rot_indexes:
                    name=p_vec[ii]
                else:
                    name=self.trial_circ[ii][1]+jj
                getattr(temp_circ, self.trial_circ[ii][0])(name, 1+self.trial_circ[ii][2])

        else:
            #Continue the U_i gate:
            for ii, jj in enumerate(self.rot_loop[first:], start=first):
                if ii in self.rot_indexes:
                    name=p_vec[ii]
                else:
                    name=self.trial_circ[ii][1]+jj
                getattr(temp_circ, self.trial_circ[ii][0])(name, 1+self.trial_circ[ii][2])

            #TODO: Only thing to check up is this range, shuld it be reversed? Rewrite it as enumerate?
            for jjj in range(len(self.trial_circ)-1, sec-1, -1):
                if jjj in self.rot_indexes:
                    name=p_vec[jjj]
                else:
                    name=self.trial_circ[jjj][1]+self.rot_loop[jjj]

                getattr(temp_circ, self.trial_circ[jjj][0])(name, 1+self.trial_circ[jjj][2])

        #TODO: add x?
        #temp_circ.x(0)
        getattr(temp_circ, 'c'+self.trial_circ[sec][0][-1])(0,1+self.trial_circ[sec][2])
        
        #temp_circ.x(0)

        temp_circ.h(0)
        #TODO add this
        temp_circ.measure(0,0)
  
        return temp_circ

    def init_C(self,lamb, fir):
        #TODO: Remember to switch everything I switch here, elsewhere
        V_circ=encoding_circ('C', self.trial_qubits)
        temp_circ=V_circ.copy()
        p_vec_C = ParameterVector('Init_C_param', len(self.rot_indexes))

        #Then we loop through the gates in U until we reach sigma-gate
        for i in range(len(self.trial_circ)-1, fir-1, -1):
            #print(len(p_vec_C))
            if i in self.rot_indexes:
                name=p_vec_C[i]
            else:
                name=self.trial_circ[i][1]+self.rot_loop[i]
            #getattr(temp_circ, self.trial_circ[i][0])(self.trial_circ[i][1]+self.rot_loop[i], 1+self.trial_circ[i][2])

        #Add x gate                
        temp_circ.x(0)
        #Then we add the sigma
        getattr(temp_circ, 'c'+self.trial_circ[fir][0][-1])(0,1+self.trial_circ[fir][2])
        #Add x gate                
        temp_circ.x(0)

        for ii, jj in enumerate(self.rot_loop[fir:], start=fir):
            if ii in self.rot_indexes:
                name=p_vec_C[ii]
            else:
                name=self.trial_circ[ii][1]+jj
            getattr(temp_circ, self.trial_circ[ii][0])(name, 1+self.trial_circ[ii][2])


        #Continue the U_i gate:
        #for keep_going in range(ind, len(self.trial_circ)):
        #for ii in range(fir-1, -1, -1):
        #    getattr(temp_circ, self.trial_circ[ii][0])(self.trial_circ[ii][1]+self.rot_loop[ii], 1+self.trial_circ[ii][2])

        #TODO: x gates?
        #temp_circ.x(0)
        getattr(temp_circ, 'c'+self.hamil[lamb][1])(0,self.hamil[lamb][2]+1)
        #temp_circ.x(0)

        temp_circ.h(0)
        temp_circ.measure(0, 0)
        
        return temp_circ

    def get_C2(self):
        C_vec_temp=np.zeros(len(self.rot_indexes))
        
        #Loops through the indices of A
        for i in range(len(self.rot_indexes)):
            c_term=self.run_C2(self.rot_indexes[i])
            
            #Multiplying with 0.5 due to the derivative factor
            #TODO might be - 0.5 instead?
            C_vec_temp[i]=c_term*0.5
 
        return C_vec_temp

    def run_C2(self, ind):
        
        #TODO: Put the params of H in a self.variable
        V_circ=encoding_circ('C', self.trial_qubits)
        
        sum_C=0
        for l in range(len(self.hamil)):
            if self.best==False:
                #Does not work, but makes more sense
                temp_circ=V_circ.copy()
                #Then we loop through the gates in U until we reach sigma-gate
                for i in range(len(self.trial_circ)):
                    getattr(temp_circ, self.trial_circ[i][0])(self.trial_circ[i][1]+self.rot_loop[i], 1+self.trial_circ[i][2])
                
                for theta in range(len(self.hamil[l])):
                    getattr(temp_circ, 'c'+self.hamil[l][theta][1])(0,self.hamil[l][theta][2]+1)
                

                for ii in range(len(self.trial_circ)-1, ind-1, -1):
                    getattr(temp_circ, self.trial_circ[ii][0])(self.trial_circ[ii][1]+self.rot_loop[ii], 1+self.trial_circ[ii][2])

                #for ii in range(ind):
                #    getattr(temp_circ, self.trial_circ[ii][0])(self.trial_circ[ii][1]+self.rot_loop[ii], 1+self.trial_circ[ii][2])
                
                #Add x gate                
                temp_circ.x(0)
                #Then we add the sigma
                getattr(temp_circ, 'c'+self.trial_circ[ind][0][-1])(0,1+self.trial_circ[ind][2])
                #Add x gate                
                temp_circ.x(0)

                #Continue the U_i gate:
                #for keep_going in range(ind, len(self.trial_circ)):
                #for ii in range(ind-1, -1, -1):
                #    getattr(temp_circ, self.trial_circ[ii][0])(self.trial_circ[ii][1]+self.rot_loop[ii], 1+self.trial_circ[ii][2])

                #TODO: x gates?
                #temp_circ.x(0)
                #getattr(temp_circ, 'c'+self.hamil[l][1])(0,self.hamil[l][2]+1)
                #temp_circ.x(0)

            else:
                temp_circ=V_circ.copy()

                #for theta in range(len(self.hamil[l])):
                #    getattr(temp_circ, 'c'+self.hamil[l][theta][1])(0,self.hamil[l][theta][2]+1)

                #Then we loop through the gates in U until we reach sigma-gate
                for i in range(len(self.trial_circ)-1, ind-1, -1):
                    getattr(temp_circ, self.trial_circ[i][0])(self.trial_circ[i][1]+self.rot_loop[i], 1+self.trial_circ[i][2])

                #Add x gate                
                temp_circ.x(0)
                #Then we add the sigma
                getattr(temp_circ, 'c'+self.trial_circ[ind][0][-1])(0,1+self.trial_circ[ind][2])
                #Add x gate                
                temp_circ.x(0)

                #Continue the U_i gate:
                #for keep_going in range(ind, len(self.trial_circ)):
                for ii in range(ind-1, -1, -1):
                    getattr(temp_circ, self.trial_circ[ii][0])(self.trial_circ[ii][1]+self.rot_loop[ii], 1+self.trial_circ[ii][2])

                #TODO: x gates?
                #temp_circ.x(0)
                for theta in range(len(self.hamil[l])):
                    getattr(temp_circ, 'c'+self.hamil[l][theta][1])(0,self.hamil[l][theta][2]+1)
                #temp_circ.x(0)

            temp_circ.h(0)
            temp_circ.measure(0, 0)
            prediction=run_circuit(temp_circ)
            
            sum_C-=prediction*self.hamil[l][0][0]
                
            
        return sum_C

    def get_dA(self, i_param):
        dA_mat_temp_i=np.zeros((len(self.rot_indexes), len(self.rot_indexes)))
        for p in range(len(self.rot_indexes)):
            for q in range(len(self.rot_indexes)):
                #TODO: - or +?
                dA_mat_temp_i[p][q]=-12.5*self.run_dA(self.rot_indexes[p],self.rot_indexes[q], i_param)

        """
        #Lets try to remove the controlled gates
        dA_mat_temp_i=np.zeros((len(self.trial_circ), len(self.trial_circ)))

        #Loops through the indices of A
        for p in range(len(self.trial_circ)):
            for q in range(len(self.trial_circ)):
                da_term=self.run_dA(p, q, i_param)
                
                dA_mat_temp_i[p][q]=da_term
        """


        return dA_mat_temp_i
    
    def run_dA(self, p_index, q_index, i_theta):
        #Compute one term in the dA matrix
        sum_A_pq=0
        #A bit unsure about the len of this one
        #for s in range(len(self.trial_circ)): #+1?
        for s in self.rot_indexes:
            dCircuit_term_1=self.dA_circ([p_index, s], [q_index])
            dCircuit_term_2=self.dA_circ([p_index], [q_index, s])

            temp_dw=self.dwdth[i_theta][s]

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
            first_der=circ_1[0]
            sec_der=circ_2[0]
            thr_der=circ_2[1]
            
            #TODO: Hopefully correct
            if sec_der<thr_der:
                sec_der, thr_der=thr_der, sec_der

        elif len(circ_1)==2:
            first_der=circ_1[0]
            sec_der=circ_1[1]
            thr_der=circ_2[0]
            
            if first_der>sec_der:
                first_der, sec_der=sec_der, first_der

        else:
            print("Only implemented for double differentiated circuits")
            exit()

        #According to the article this is correct
        V_circ=encoding_circ('C', self.trial_qubits)
        temp_circ=V_circ.copy()

        for i, j in enumerate(self.rot_loop[:thr_der]):
            getattr(temp_circ, self.trial_circ[i][0])(self.trial_circ[i][1]+j, 1+self.trial_circ[i][2])
        getattr(temp_circ, 'c'+self.trial_circ[thr_der][0][-1])(0,1+self.trial_circ[thr_der][2])

        if len(circ_2)==1:
            for ii, jj in enumerate(self.rot_loop[thr_der:], start=thr_der):
                getattr(temp_circ, self.trial_circ[ii][0])(self.trial_circ[ii][1]+jj, 1+self.trial_circ[ii][2])

            for kk in range(len(self.trial_circ)-1, sec_der-1, -1):
                getattr(temp_circ, self.trial_circ[kk][0])(self.trial_circ[kk][1]+self.rot_loop[kk], 1+self.trial_circ[kk][2])
            temp_circ.x(0)
            getattr(temp_circ, 'c'+self.trial_circ[sec_der][0][-1])(0,1+self.trial_circ[sec_der][2])
            temp_circ.x(0)

            for kkk in range(sec_der-1, first_der-1, -1):
                getattr(temp_circ, self.trial_circ[kkk][0])(self.trial_circ[kkk][1]+self.rot_loop[kkk], 1+self.trial_circ[kkk][2])


        else:
            for ii, jj in enumerate(self.rot_loop[thr_der:sec_der], start=thr_der):
                getattr(temp_circ, self.trial_circ[ii][0])(self.trial_circ[ii][1]+jj, 1+self.trial_circ[ii][2])
            getattr(temp_circ, 'c'+self.trial_circ[sec_der][0][-1])(0,1+self.trial_circ[sec_der][2])
            
            for iii, jjj in enumerate(self.rot_loop[sec_der:], start=sec_der):
                getattr(temp_circ, self.trial_circ[iii][0])(self.trial_circ[iii][1]+jjj, 1+self.trial_circ[iii][2])

            for kkk in range(len(self.trial_circ)-1, first_der-1, -1):
                getattr(temp_circ, self.trial_circ[kkk][0])(self.trial_circ[kkk][1]+self.rot_loop[kkk], 1+self.trial_circ[kkk][2])
        
        temp_circ.x(0)
        getattr(temp_circ, 'c'+self.trial_circ[first_der][0][-1])(0,1+self.trial_circ[first_der][2])
        temp_circ.x(0)

        temp_circ.h(0)
        #TODO Add this back
        temp_circ.measure(0,0)

        prediction=run_circuit(temp_circ)
 
        sum_dA=prediction

        return sum_dA

    #TODO: #Start here and see if these are alright
    def get_dC(self, i_param):
        #Lets try to remove the controlled gates
        dC_vec_temp_i=np.zeros(len(self.rot_indexes))
        #Loops through the indices of C
        for p in range(len(self.rot_indexes)):
            #TODO include minus from i?
            dC_vec_temp_i[p]=self.run_dC(self.rot_indexes[p], i_param)
        
        return dC_vec_temp_i
    
    def run_dC(self, p_index, i_theta):
        dCircuit_term_0=-0.5*self.dC_circ0(p_index, i_theta)

        #TODO Continue here!
        sum_C_p=0
        for s in range(len(self.rot_indexes)):
            for i in range(len(self.hamil)):
                dCircuit_term_1=self.dC_circ1(p_index, i, self.rot_indexes[s])
                dCircuit_term_2=self.dC_circ2(p_index, self.rot_indexes[s], i)
                
                ## TODO: Fix this, I dont know how dw should be computed
                #TODO: Is the [0][0] correct?
                temp_dw=self.hamil[i][0][0]*self.dwdth[i_theta][self.rot_indexes[s]]

            #I guess the real and trace part automatically is computed 
            # in the cirquit.. or is it?
                #+ or - is this what is wrong?
                sum_C_p+=0.25*temp_dw*(dCircuit_term_1+dCircuit_term_2)
        
        return dCircuit_term_0-sum_C_p

    def dC_circ0(self, p, j):
        """
        First circuit of dC in varQBM article
        """
        V_circ=encoding_circ('C', self.trial_qubits)    
        temp_circ=V_circ.copy()
        
        if self.best==False:
            for i in range(len(self.trial_circ)):
                getattr(temp_circ, self.trial_circ[i][0])(self.trial_circ[i][1]+self.rot_loop[i], 1+self.trial_circ[i][2])
            
            for theta in range(len(self.hamil[j])):
                getattr(temp_circ, 'c'+self.hamil[j][theta][1])(0,self.hamil[j][theta][2]+1)
            
            
            for ii in range(len(self.trial_circ)-1, p-1, -1):
                getattr(temp_circ, self.trial_circ[ii][0])(self.trial_circ[ii][1]+self.rot_loop[ii], 1+self.trial_circ[ii][2])

            temp_circ.x(0)
            getattr(temp_circ, 'c'+self.trial_circ[p][0][-1])(0,1+self.trial_circ[p][2])
            temp_circ.x(0)

        else:
            #Then we loop through the gates in U until we reach sigma-gate
            for i in range(len(self.trial_circ)-1, p-1, -1):
                getattr(temp_circ, self.trial_circ[i][0])(self.trial_circ[i][1]+self.rot_loop[i], 1+self.trial_circ[i][2])

            temp_circ.x(0)
            getattr(temp_circ, 'c'+self.trial_circ[p][0][-1])(0,1+self.trial_circ[p][2])
            temp_circ.x(0)

            for ii in range(p-1, -1, -1):
                getattr(temp_circ, self.trial_circ[ii][0])(self.trial_circ[ii][1]+self.rot_loop[ii], 1+self.trial_circ[ii][2])

            #TODO: x gates?
            for theta in range(len(self.hamil[j])):
                #temp_circ.x(0)
                getattr(temp_circ, 'c'+self.hamil[j][theta][1])(0,self.hamil[j][theta][2]+1)
                #temp_circ.x(0)
            
        temp_circ.h(0)
        temp_circ.measure(0, 0)
        prediction=run_circuit(temp_circ)
            
        
        return prediction

    def dC_circ1(self, p, i_index, s):
        V_circ=encoding_circ('A', self.trial_qubits)
        temp_circ=V_circ.copy()

        for i, j in enumerate(self.rot_loop[:s]):
            getattr(temp_circ, self.trial_circ[i][0])(self.trial_circ[i][1]+j, 1+self.trial_circ[i][2])
        getattr(temp_circ, 'c'+self.trial_circ[s][0][-1])(0,1+self.trial_circ[s][2])

        for ii, jj in enumerate(self.rot_loop[s:], start=s):
            getattr(temp_circ, self.trial_circ[ii][0])(self.trial_circ[ii][1]+jj, 1+self.trial_circ[ii][2])

        for theta in range(len(self.hamil[i_index])):
            getattr(temp_circ, 'c'+self.hamil[i_index][theta][1])(0,self.hamil[i_index][theta][2]+1)
            
        for kk in range(len(self.trial_circ)-1, p-1, -1):
            getattr(temp_circ, self.trial_circ[kk][0])(self.trial_circ[kk][1]+self.rot_loop[kk], 1+self.trial_circ[kk][2])

        temp_circ.x(0)
        getattr(temp_circ, 'c'+self.trial_circ[p][0][-1])(0,1+self.trial_circ[p][2])
        temp_circ.x(0)

        temp_circ.h(0)
        temp_circ.measure(0,0)

        prediction=run_circuit(temp_circ)
        
        return prediction

    def dC_circ2(self, p, s, i_index):
        V_circ=encoding_circ('A', self.trial_qubits)
        temp_circ=V_circ.copy()

        #TODO: What if p=s?
        if p>s:
            p,s=s,p

        if self.best==False:

            for i in range(len(self.trial_circ)):
                getattr(temp_circ, self.trial_circ[i][0])(self.trial_circ[i][1]+self.rot_loop[i], 1+self.trial_circ[i][2])
                        
            for theta in range(len(self.hamil[i_index])):
               getattr(temp_circ, 'c'+self.hamil[i_index][theta][1])(0,self.hamil[i_index][theta][2]+1)
         
            for kk in range(len(self.trial_circ)-1, s-1, -1):
                getattr(temp_circ, self.trial_circ[kk][0])(self.trial_circ[kk][1]+self.rot_loop[kk], 1+self.trial_circ[kk][2])
            
            temp_circ.x(0)
            getattr(temp_circ, 'c'+self.trial_circ[s][0][-1])(0,1+self.trial_circ[s][2])
            temp_circ.x(0)

            for kkk in range(s-1, p-1, -1):
                getattr(temp_circ, self.trial_circ[kkk][0])(self.trial_circ[kkk][1]+self.rot_loop[kkk], 1+self.trial_circ[kkk][2])

            temp_circ.x(0)
            getattr(temp_circ, 'c'+self.trial_circ[p][0][-1])(0,1+self.trial_circ[p][2])
            temp_circ.x(0)
        else:
            for kk in range(len(self.trial_circ)-1, s-1, -1):
                getattr(temp_circ, self.trial_circ[kk][0])(self.trial_circ[kk][1]+self.rot_loop[kk], 1+self.trial_circ[kk][2])
            
            temp_circ.x(0)
            getattr(temp_circ, 'c'+self.trial_circ[s][0][-1])(0,1+self.trial_circ[s][2])
            temp_circ.x(0)

            for kkk in range(s-1, p-1, -1):
                getattr(temp_circ, self.trial_circ[kkk][0])(self.trial_circ[kkk][1]+self.rot_loop[kkk], 1+self.trial_circ[kkk][2])

            temp_circ.x(0)
            getattr(temp_circ, 'c'+self.trial_circ[p][0][-1])(0,1+self.trial_circ[p][2])
            temp_circ.x(0)

            for ii in range(p-1, -1, -1):
                getattr(temp_circ, self.trial_circ[ii][0])(self.trial_circ[ii][1]+self.rot_loop[ii], 1+self.trial_circ[ii][2])

            for theta in range(len(self.hamil[i_index])):
                getattr(temp_circ, 'c'+self.hamil[i_index][theta][1])(0,self.hamil[i_index][theta][2]+1)
         
        temp_circ.h(0)
        temp_circ.measure(0, 0)
        prediction=run_circuit(temp_circ)

        return prediction


    def last_try(self):
        A_mat_temp2=np.zeros((len(self.rot_indexes), len(self.rot_indexes)))
        C_vec_temp2=np.zeros(len(self.rot_indexes))
        
        for i in range(len(self.rot_indexes)):
            #For each gate 
            for j in range(len(self.rot_indexes)):
                A_term=self.run_A2(self.rot_indexes[i],self.rot_indexes[j])
                #TODO: Changed the real part
                derivative_const=0.25
                A_mat_temp2[i][j]=A_term*derivative_const

            c_term=self.run_C2(self.rot_indexes[i])
            C_vec_temp2[i]=c_term*0.5


        return A_mat_temp2, C_vec_temp2
