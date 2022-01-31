import numpy as np
from numpy.core.numeric import zeros_like

from sklearn.linear_model import Ridge, RidgeCV, Lasso

from qiskit.circuit import ParameterVector

from utils import *
import random
#import multiprocessing as mp
#import pathos
import itertools as it
#mp=pathos.helpers.mp
import time
import copy

import scipy as sp

from sklearn.metrics import mean_squared_error

#from pathos.pools import ProcessPool

#from numba import jit
#from numba.experimental import jitclass

#@jitclass
class varQITE:
    def __init__(self, hamil, trial_circ, maxTime=0.5, steps=10, plot_fidelity=False, alpha=None):
        """
        Class handling the variational quantum imaginary time evolution
        
        Args:
            hamil(list):        Hamiltonian as a list of gates and coefficients
            trial_circ(list):   Trial circuit as a list of gates, params and 
                                qubit placement
            max_Time(float):    Maximum time value for the propagation
            steps(int):         timesteps of varQITE
        """
        self.best=False
        self.sup=False
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

        self.plot_fidelity=plot_fidelity
        if self.plot_fidelity==True:
            self.plot_fidelity_list=[]

        if alpha!=None:
            self.alpha=alpha
        else:
            #0.0005 is good with Lasso
            self.alpha=0.001

    
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
        
        C_circ= np.empty(shape=(len(self.hamil),len(self.rot_indexes)), dtype=object)
        
        #Assuming there is only one circ per i,j, due to r? only having 1 element in f
        for i in range(len(self.hamil)):
            for j in range(len(self.rot_indexes)):
                #Just the circuits
                C_circ[i][j]=self.init_C(self.hamil[i], self.rot_indexes[j])
        
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
        dA_circ= np.empty(shape=(len(self.rot_indexes), len(self.rot_indexes), len(self.rot_indexes), 2), dtype=object)
        
        for i in range(len(self.rot_indexes)):
            for j in range(len(self.rot_indexes)):
                for k in range(len(self.rot_indexes)):
                    dA_circ[i][j][k][0]=self.init_dA(self.rot_indexes[i], self.rot_indexes[j],self.rot_indexes[k], 0)
                    dA_circ[i][j][k][1]=self.init_dA(self.rot_indexes[i], self.rot_indexes[j],self.rot_indexes[k], 1)
                    #pass


        """
        Creating circ dC
        """
        dC_circ= np.empty(shape=(len(self.hamil), len(self.rot_indexes), len(self.rot_indexes), 2), dtype=object)

        for ii in range(len(self.hamil)):
            for jj in range(len(self.rot_indexes)):
                for kk in range(len(self.rot_indexes)):
                    dC_circ[ii][jj][kk][0]=self.init_dC(ii, self.rot_indexes[jj],self.rot_indexes[kk], 0)
                    dC_circ[ii][jj][kk][1]=self.init_dC(ii, self.rot_indexes[jj],self.rot_indexes[kk], 1)


        #print(np.where(C_vec==0.075732421875))

        self.A_init=A_circ
        self.C_init=C_circ
        self.dA_init=dA_circ
        self.dC_init=dC_circ
        #return A_circ, C_vec #, dA_circ, dc_circ
        return

    def state_prep(self, gradient_stateprep=False):
        """
        Prepares an approximation for the gibbs states using imaginary time evolution 
        """
        omega_w=np.array(self.trial_circ)[:, 1].astype('float')

        omega_derivative=np.zeros(len(self.trial_circ))
        self.dwdth=np.zeros((len(self.hamil), len(self.trial_circ)))
        
        A_mat=zeros_like(self.A_init, dtype='float')
        C_vec=zeros_like(self.rot_indexes, dtype='float')

        for t in np.linspace(self.time_step, self.maxTime, num=self.steps):
            print(f'VarQITE steps: {np.around(t, decimals=2)}/{self.maxTime}')

            #Expression A: Binds the parameters to the circuits
            for i_a in range(len(self.rot_indexes)):
                for j_a in range(len(self.rot_indexes)):
                    #Just the circuits
                    A_mat[i_a][j_a]=run_circuit(self.A_init[i_a][j_a].bind_parameters(omega_w[self.rot_indexes][:len(self.A_init[i_a][j_a].parameters)]))
            
            A_mat*=0.25

            for i_c in range(len(self.hamil)):
                for j_c in range(len(self.rot_indexes)):
                    C_vec[j_c]+=self.hamil[i_c][0][0]*run_circuit(self.C_init[i_c][j_c].\
                    bind_parameters(omega_w[self.rot_indexes][:len(self.C_init[i_c][j_c].parameters)]))

            C_vec*=-0.5
            
            ridge_inv=True
            CV=False
            if ridge_inv==False:
                A_inv=np.linalg.pinv(A_mat, hermitian=False)
                omega_derivative_temp=A_inv@C_vec
            else:
                if CV==True:
                    I=np.eye(A_mat.shape[1])
                    regr_cv = RidgeCV(alphas= np.logspace(-4, 4))
                    regr_cv.fit(A_mat, C_vec)
                    omega_derivative_temp=np.linalg.inv(A_mat.T @ A_mat + regr_cv.alpha_*I) @ A_mat.T @ C_vec

                else:
                    model_R = Ridge(alpha=1e-8)
                    model_R.fit(A_mat, C_vec)
                    omega_derivative_temp=model_R.coef_
                    #print(mean_squared_error(C_vec,A_mat@omega_derivative_temp))

                    #print(abs(np.min(C_vec))*0.001)<   
                    """ 
                    loss=1000
                    lmb=10.0
                    
                    loss_list=[]
                    #omega_list=[]
                    while loss>0.001:
                        lmb*=0.1
                        model_R = Ridge(alpha=lmb)
                        model_R.fit(A_mat, C_vec)
                        #TODO: Deep copy coeff?
                        omega_derivative_temp=model_R.coef_
                        #omega_list.append(omega_derivative_temp)
                        loss=mean_squared_error(C_vec,omega_derivative_temp)
                        loss_list.append(loss)

                        if lmb<1e-14:
                            lmb=10**(-1*loss_list.index(min(loss_list)))
                            break
                    
                    print(f'loss: {min(loss_list)}, lmb: {lmb}')

                    lmb=1e-8

                    model_R = Ridge(alpha=lmb)
                    model_R.fit(A_mat, C_vec)
                    omega_derivative_temp=model_R.coef_
                    print(mean_squared_error(C_vec,omega_derivative_temp))

                    #omega_derivative_temp=omega_list[loss_list.index(min(loss_list))]
                    
                    #model_R = Ridge(alpha=1e-5)
                    #model_R.fit(A_mat, C_vec)
                    #omega_derivative_temp=model_R.coef_

                    """
                    
                    #model_R = Ridge(alpha=abs(np.min(C_vec/len(C_vec)))*1e-8)
                    #print(f'lambda {abs(np.min(C_vec/len(C_vec)))*1e-4}')
                    #print(f'Loss from ridge: {loss}')
                    
            omega_derivative[self.rot_indexes]=omega_derivative_temp
            
            if gradient_stateprep==False:
                dA_mat=self.getdA_bound(omega_w)
                
                for i in range(len(self.hamil)):
                    #TODO: Deep copy takes a lot of time, fix this
                    #dA_mat=np.copy(self.get_dA(i))
                    #dC_vec=np.copy(self.get_dC(i))
                    dC_vec=self.getdC_bound(i, omega_w)

                    #dA_mat=np.copy(self.getdA_init(i))

                    if ridge_inv==False:
                        w_dtheta_dt=A_inv@(dC_vec-dA_mat[i]@omega_derivative_temp)#* or @?
                        if t==self.maxTime:
                            print('Lets find out why the derivatives are so high:')

                            print('A_inv:')
                            print(A_inv)
                            print('-----------------') 
                            
                            print('dA:')
                            print(dA_mat)
                            print('-----------------')

                            print('fC_vec:')
                            print(dC_vec)
                            print('-----------------')

                            print('omega:')
                            print(omega_derivative_temp)
                            print('-----------------')

                            print('dA_mat@omega:')
                            print(dA_mat@omega_derivative_temp)
                            print('-----------------')
                            
                            print('dc- minus the thing over:')
                            print(dC_vec-dA_mat@omega_derivative_temp)
                            print('-----------------')

                            print('A_inv(\cdot)')
                            print(A_inv@(dC_vec-dA_mat@omega_derivative_temp))
                            print('-----------------')
                        
                    else:
                        rh_side=dC_vec-dA_mat[i]@omega_derivative_temp

                        model_dR = Ridge(alpha=1e-4)
                        model_dR.fit(A_mat, rh_side)
                        w_dtheta_dt=model_dR.coef_
                        
                        temp_loss_d=mean_squared_error(rh_side,A_mat@w_dtheta_dt)
                        #print(f'Loss from ridge derivert: {temp_loss_d}')
                        """
                        loss=1000
                        lmb_2=10.0
                        
                        loss_list_2=[]
                        while loss>0.001:
                            lmb_2*=0.1
                            model_dR = Ridge(alpha=lmb_2)
                            model_dR.fit(A_mat, rh_side)
                            #TODO: Deep copy coeff?
                            w_dtheta_dt=model_dR.coef_
                            loss=mean_squared_error(rh_side,w_dtheta_dt)
                            loss_list_2.append(loss)

                            if lmb_2<1e-14:
                                lmb_2=10**(-1*loss_list_2.index(min(loss_list_2)))
                                break
                        
                        print(f'Derivert: loss_: {min(loss_list_2)}, lmb: {lmb_2}')
                        """

                        #model_dR = Ridge(alpha=1e-3)
                        #model_dR = Ridge(alpha=np.log(-1*len(rh_side)))
                        #model_dR.fit(A_mat, rh_side)
                        #w_dtheta_dt=model_dR.coef_
                        
                        #temp_loss_d=mean_squared_error(rh_side,w_dtheta_dt)
                        #print(f'Loss from ridge derivert: {temp_loss_d}')

                    self.dwdth[i][self.rot_indexes]+=w_dtheta_dt*self.time_step
                    #print(f'w_dtheta: {w_dtheta_dt}')

            omega_w+=(omega_derivative*self.time_step)
            
            if self.plot_fidelity==True:
                #print(omega_w)
                self.plot_fidelity_list.append(np.copy(omega_w))
            #print(omega_w)

            #print(f'omega after step {omega_w}')

            #print(omega_derivative)
            #Update parameters
            #print(self.trial_circ)
            #print(omega_w)
            #TODO: do I change this multiple times?
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
        
        if self.sup==True:

            for i, j in enumerate(self.rot_loop[:first]):
                getattr(temp_circ, self.trial_circ[i][0])(self.trial_circ[i][1]+j, 1+self.trial_circ[i][2])
                #print(i)
            
            #TODO: Then we add the sigma and x gate(?)
            temp_circ.x(0)
            getattr(temp_circ, 'c'+self.trial_circ[first][0][-1])(0,1+self.trial_circ[first][2])
            temp_circ.x(0)
            
            for ii, jj in enumerate(self.rot_loop[first:], start=first):
                getattr(temp_circ, self.trial_circ[ii][0])(self.trial_circ[ii][1]+jj, 1+self.trial_circ[ii][2])

            for kk in range(len(self.trial_circ)-1, sec-1, -1):
                getattr(temp_circ, self.trial_circ[kk][0])(self.trial_circ[kk][1]+self.rot_loop[kk], 1+self.trial_circ[kk][2])

            getattr(temp_circ, 'c'+self.trial_circ[sec][0][-1])(0,1+self.trial_circ[sec][2])

        else:
            for i, j in enumerate(self.rot_loop[:first]):
                getattr(temp_circ, self.trial_circ[i][0])(self.trial_circ[i][1]+j, 1+self.trial_circ[i][2])
                #print(i)
            
            #TODO: Then we add the sigma and x gate(?)
            #temp_circ.x(0)
            getattr(temp_circ, 'c'+self.trial_circ[first][0][-1])(0,1+self.trial_circ[first][2])
            #temp_circ.x(0)

            if first==22.4:#<sec:
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
        temp_circ.measure(0,0)

        prediction=run_circuit(temp_circ)
        sum_A=prediction


        return sum_A

    def init_A(self,first, sec):
        #TODO: Remember to switch everything I switch here, elsewhere
        V_circ=encoding_circ('A', self.trial_qubits)
        temp_circ=V_circ.copy()
        
        p_vec = ParameterVector('Init_param', len(self.rot_indexes))

        for i, j in enumerate(self.rot_loop[:first]):
            if i in self.rot_indexes:
                name=p_vec[np.where(self.rot_indexes==i)[0][0]]
                #name=p_vec[i] 
            else:
                name=self.trial_circ[i][1]+j
            getattr(temp_circ, self.trial_circ[i][0])(name, 1+self.trial_circ[i][2])
        
        getattr(temp_circ, 'c'+self.trial_circ[first][0][-1])(0,1+self.trial_circ[first][2])

        if first==39.3:#<sec:
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
                    #name=p_vec[ii]
                    name=p_vec[np.where(self.rot_indexes==ii)[0][0]]

                else:
                    name=self.trial_circ[ii][1]+jj
                getattr(temp_circ, self.trial_circ[ii][0])(name, 1+self.trial_circ[ii][2])

            #TODO: Only thing to check up is this range, shuld it be reversed? Rewrite it as enumerate?
            for jjj in range(len(self.trial_circ)-1, sec-1, -1):
                if jjj in self.rot_indexes:
                    #name=p_vec[jjj]
                    name=p_vec[np.where(self.rot_indexes==jjj)[0][0]]
                else:
                    name=self.trial_circ[jjj][1]+self.rot_loop[jjj]

                getattr(temp_circ, self.trial_circ[jjj][0])(name, 1+self.trial_circ[jjj][2])

        
        temp_circ.x(0)
        getattr(temp_circ, 'c'+self.trial_circ[sec][0][-1])(0,1+self.trial_circ[sec][2])
        temp_circ.x(0)

        temp_circ.h(0)
        #TODO add this
        temp_circ.measure(0,0)
  
        return temp_circ

    def init_C(self,lamb, fir):
        #TODO: Remember to switch everything I switch here, elsewhere
        V_circ=encoding_circ('C', self.trial_qubits)
        temp_circ=V_circ.copy()
        p_vec_C = ParameterVector('Init_param', len(self.rot_indexes))

        for i, j in enumerate(self.rot_loop):
            if i in self.rot_indexes:
                name=p_vec_C[np.where(self.rot_indexes==i)[0][0]]
            else:
                name=self.trial_circ[i][1]+j
            getattr(temp_circ, self.trial_circ[i][0])(name, 1+self.trial_circ[i][2])
        
        for theta in range(len(lamb)):
            getattr(temp_circ, 'c'+lamb[theta][1])(0,lamb[theta][2]+1)

        for k in range(len(self.trial_circ)-1, fir-1, -1):
            #print(len(p_vec_C))
            if k in self.rot_indexes:
                #print(k)
                #print(self.rot_indexes)
                #print(len(self.trial_circ))
                #print(p_vec_C)
                name=p_vec_C[np.where(self.rot_indexes==k)[0][0]]
            else:
                name=self.trial_circ[k][1]+self.rot_loop[k]

            getattr(temp_circ, self.trial_circ[k][0])(name, 1+self.trial_circ[k][2])

        temp_circ.x(0)
        getattr(temp_circ, 'c'+self.trial_circ[fir][0][-1])(0,1+self.trial_circ[fir][2])
        temp_circ.x(0)

        temp_circ.h(0)
        temp_circ.measure(0, 0)

        #print(f'fir {fir}')
        #print(temp_circ)
        
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
        temp_circ=V_circ.copy()
        sum_C=0

        for l in range(len(self.hamil)):
            if self.sup==True:


                for i, j in enumerate(self.rot_loop[:ind]):
                    getattr(temp_circ, self.trial_circ[i][0])(self.trial_circ[i][1]+j, 1+self.trial_circ[i][2])
                    #print(i)
                
                #TODO: Then we add the sigma and x gate(?)
                temp_circ.x(0)
                getattr(temp_circ, 'c'+self.trial_circ[ind][0][-1])(0,1+self.trial_circ[ind][2])
                temp_circ.x(0)
                
                for ii, jj in enumerate(self.rot_loop[ind:], start=ind):
                    getattr(temp_circ, self.trial_circ[ii][0])(self.trial_circ[ii][1]+jj, 1+self.trial_circ[ii][2])
                
                for theta in range(len(self.hamil[l])):
                    getattr(temp_circ, 'c'+self.hamil[l][theta][1])(0,self.hamil[l][theta][2]+1)
                

            else:    
                if self.best==False:
                    #Does not work, but makes more sense
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
                dA_mat_temp_i[p][q]=self.run_dA(self.rot_indexes[p],self.rot_indexes[q], i_param)

        """
        #Lets try to remove the controlled gates
        dA_mat_temp_i=np.zeros((len(self.trial_circ), len(self.trial_circ)))

        #Loops through the indices of A
        for p in range(len(self.trial_circ)):
            for q in range(len(self.trial_circ)):
                da_term=self.run_dA(p, q, i_param)
                
                dA_mat_temp_i[p][q]=da_term
        """
        #TODO: should be -
        return dA_mat_temp_i*(-0.125)

    def getdA_bound(self, binding_values):
        dA_mat_temp=np.zeros((len(self.rot_indexes), len(self.rot_indexes), len(self.rot_indexes), 2))

        dA=np.zeros((len(self.hamil), len(self.rot_indexes), len(self.rot_indexes)))

        for p_da in range(len(self.rot_indexes)):
            for q_da in range(len(self.rot_indexes)):
                for s_da in range(len(self.rot_indexes)):
                    #print(self.dA_init[p_da][q_da][s_da][0])
                    #print(self.dA_init[p_da][q_da][s_da][0].bind_parameters(binding_values[self.rot_indexes][:len(self.dA_init[p_da][q_da][s_da][0].parameters)]))
                    dA_mat_temp[p_da][q_da][s_da][0]=run_circuit(self.dA_init[p_da][q_da][s_da][0].bind_parameters(binding_values[self.rot_indexes][:len(self.dA_init[p_da][q_da][s_da][0].parameters)]))
                    dA_mat_temp[p_da][q_da][s_da][1]=run_circuit(self.dA_init[p_da][q_da][s_da][1].bind_parameters(binding_values[self.rot_indexes][:len(self.dA_init[p_da][q_da][s_da][1].parameters)]))
                    #Quiet high, 0.5 print(dA_mat_temp[p_da][q_da][s_da][0])
       
        for i in range(len(self.hamil)):
            for p in range(len(self.rot_indexes)):
                for q in range(len(self.rot_indexes)):
                    for s in range(len(self.rot_indexes)):
                        dA[i][p][q]+=self.dwdth[i][self.rot_indexes[s]]*(dA_mat_temp[p][q][s][0]+dA_mat_temp[p][q][s][1])

        #TODO: Changed from negative to positive
        dA*=-0.125
        
        return dA

    def getdC_bound(self,i_th, binding_values):
        dC_i=np.zeros(len(self.rot_indexes))

        for j_p in range(len(self.rot_indexes)):
            #TODO- is the same?
            dC_i[j_p]+=-0.5*run_circuit(self.C_init[i_th][j_p].\
            bind_parameters(binding_values[self.rot_indexes][:len(self.C_init[i_th][j_p].parameters)]))
            
            for i_dc in range(len(self.hamil)):
                for s_dc in range(len(self.rot_indexes)):
                    term1_temp=run_circuit(self.dC_init[i_dc][j_p][s_dc][0].bind_parameters(binding_values[self.rot_indexes][:len(self.dC_init[i_dc][j_p][s_dc][0].parameters)]))
                    term2_temp=run_circuit(self.dC_init[i_dc][j_p][s_dc][1].bind_parameters(binding_values[self.rot_indexes][:len(self.dC_init[i_dc][j_p][s_dc][1].parameters)]))
                    dC_i[j_p]+=0.25*self.hamil[i_dc][0][0]*self.dwdth[i_th][s_dc]*(term1_temp+term2_temp)
        
        return dC_i

    def init_dA(self,pp, ss, qq, type_circ):
        V_circ=encoding_circ('C', self.trial_qubits)
        temp_circ=V_circ.copy()

        p_vec = ParameterVector('Init_param', len(self.rot_indexes))

        if type_circ==0:
            if pp>ss:
                pp,ss=ss,pp

            for i, j in enumerate(self.rot_loop[:qq]):
                if i in self.rot_indexes:
                    name=p_vec[np.where(self.rot_indexes==i)[0][0]]
                else:
                    name=self.trial_circ[i][1]+j
                getattr(temp_circ, self.trial_circ[i][0])(name, 1+self.trial_circ[i][2])
            
            getattr(temp_circ, 'c'+self.trial_circ[qq][0][-1])(0,1+self.trial_circ[qq][2])

            #TODO Can probably remove a chunk of this
            for ii, jj in enumerate(self.rot_loop[qq:], start=qq):
                if ii in self.rot_indexes:
                    name=p_vec[np.where(self.rot_indexes==ii)[0][0]]
                else:
                    name=self.trial_circ[ii][1]+jj
                getattr(temp_circ, self.trial_circ[ii][0])(name, 1+self.trial_circ[ii][2])

            for jjj in range(len(self.trial_circ)-1, ss-1, -1):
                if jjj in self.rot_indexes:
                    name=p_vec[np.where(self.rot_indexes==jjj)[0][0]]
                else:
                    name=self.trial_circ[jjj][1]+self.rot_loop[jjj]   
                getattr(temp_circ, self.trial_circ[jjj][0])(name, 1+self.trial_circ[jjj][2])
        
            temp_circ.x(0)
            getattr(temp_circ, 'c'+self.trial_circ[ss][0][-1])(0,1+self.trial_circ[ss][2])

            for last in range(ss, pp-1, -1):
                if last in self.rot_indexes:
                    name=p_vec[np.where(self.rot_indexes==last)[0][0]]
                else:
                    name=self.trial_circ[last][1]+self.rot_loop[last]   
                getattr(temp_circ, self.trial_circ[last][0])(name, 1+self.trial_circ[last][2])
        
            getattr(temp_circ, 'c'+self.trial_circ[pp][0][-1])(0,1+self.trial_circ[pp][2])
            temp_circ.x(0)
        
        elif type_circ==1:
            if ss>qq:
                qq,ss=ss,qq

            for i, j in enumerate(self.rot_loop[:ss]):
                if i in self.rot_indexes:
                    name=p_vec[np.where(self.rot_indexes==i)[0][0]]
                else:
                    name=self.trial_circ[i][1]+j
                getattr(temp_circ, self.trial_circ[i][0])(name, 1+self.trial_circ[i][2])
            
            getattr(temp_circ, 'c'+self.trial_circ[ss][0][-1])(0,1+self.trial_circ[ss][2])

            for ii, jj in enumerate(self.rot_loop[ss:qq], start=ss):
                if ii in self.rot_indexes:
                    name=p_vec[np.where(self.rot_indexes==ii)[0][0]]
                else:
                    name=self.trial_circ[ii][1]+jj
                getattr(temp_circ, self.trial_circ[ii][0])(name, 1+self.trial_circ[ii][2])
            
            getattr(temp_circ, 'c'+self.trial_circ[qq][0][-1])(0,1+self.trial_circ[qq][2])

            for iii, jjj in enumerate(self.rot_loop[qq:], start=qq):
                if iii in self.rot_indexes:
                    name=p_vec[np.where(self.rot_indexes==iii)[0][0]]
                else:
                    name=self.trial_circ[iii][1]+jjj
                getattr(temp_circ, self.trial_circ[iii][0])(name, 1+self.trial_circ[iii][2])


            for last in range(len(self.trial_circ)-1, pp-1, -1):
                if last in self.rot_indexes:
                    name=p_vec[np.where(self.rot_indexes==last)[0][0]]
                else:
                    name=self.trial_circ[last][1]+self.rot_loop[last]   
                getattr(temp_circ, self.trial_circ[last][0])(name, 1+self.trial_circ[last][2])
              
            temp_circ.x(0)
            getattr(temp_circ, 'c'+self.trial_circ[pp][0][-1])(0,1+self.trial_circ[pp][2])
            temp_circ.x(0)
            
        else:
            print('Cant initialize dA circ')
            exit()

        temp_circ.h(0)
        temp_circ.measure(0,0)
  
        return temp_circ
    
    def run_dA(self, p_index, q_index, i_theta):
        #Compute one term in the dA matrix
        sum_A_pq=0
        #A bit unsure about the len of this one
        #for s in range(len(self.trial_circ)): #+1?
        for s in self.rot_indexes:
            dCircuit_term_1=self.dA_circ([p_index, s], [q_index])
            dCircuit_term_2=self.dA_circ([p_index], [q_index, s])

            #TODO: self.dwdth copy?
            temp_dw=self.dwdth[i_theta][s]

            sum_A_pq+=temp_dw*(dCircuit_term_1+dCircuit_term_2)
        
        return sum_A_pq

    def init_dC(self, i_index, pp, ss,type_of_circ):
        V_circ=encoding_circ('A', self.trial_qubits)
        temp_circ=V_circ.copy()
        p_vec = ParameterVector('Init_param', len(self.rot_indexes))

        if type_of_circ==0:
            for i, j in enumerate(self.rot_loop[:ss]):
                if i in self.rot_indexes:
                    name=p_vec[np.where(self.rot_indexes==i)[0][0]]
                else:
                    name=self.trial_circ[i][1]+j
                getattr(temp_circ, self.trial_circ[i][0])(name, 1+self.trial_circ[i][2])
            
            getattr(temp_circ, 'c'+self.trial_circ[ss][0][-1])(0,1+self.trial_circ[ss][2])
        
            for ii, jj in enumerate(self.rot_loop[ss:], start=ss):
                if ii in self.rot_indexes:
                    name=p_vec[np.where(self.rot_indexes==ii)[0][0]]
                else:
                    name=self.trial_circ[ii][1]+jj
                getattr(temp_circ, self.trial_circ[ii][0])(name, 1+self.trial_circ[ii][2])

            for theta in range(len(self.hamil[i_index])):
                getattr(temp_circ, 'c'+self.hamil[i_index][theta][1])(0,self.hamil[i_index][theta][2]+1)

            for jjj in range(len(self.trial_circ)-1, pp-1, -1):
                if jjj in self.rot_indexes:
                    name=p_vec[np.where(self.rot_indexes==jjj)[0][0]]
                else:
                    name=self.trial_circ[jjj][1]+self.rot_loop[jjj]

                getattr(temp_circ, self.trial_circ[jjj][0])(name, 1+self.trial_circ[jjj][2])
        
            temp_circ.x(0)
            getattr(temp_circ, 'c'+self.trial_circ[pp][0][-1])(0,1+self.trial_circ[pp][2])
            temp_circ.x(0)
    
        elif type_of_circ==1:
            if pp>ss:
                pp,ss=ss,pp

            for i, j in enumerate(self.rot_loop):
                if i in self.rot_indexes:
                    name=p_vec[np.where(self.rot_indexes==i)[0][0]]
                else:
                    name=self.trial_circ[i][1]+j
                getattr(temp_circ, self.trial_circ[i][0])(name, 1+self.trial_circ[i][2])
            
            for theta in range(len(self.hamil[i_index])):
                getattr(temp_circ, 'c'+self.hamil[i_index][theta][1])(0,self.hamil[i_index][theta][2]+1)
    
            for jjj in range(len(self.trial_circ)-1, ss-1, -1):
                if jjj in self.rot_indexes:
                    name=p_vec[np.where(self.rot_indexes==jjj)[0][0]]
                else:
                    name=self.trial_circ[jjj][1]+self.rot_loop[jjj]   
                getattr(temp_circ, self.trial_circ[jjj][0])(name, 1+self.trial_circ[jjj][2])
        
            temp_circ.x(0)
            getattr(temp_circ, 'c'+self.trial_circ[ss][0][-1])(0,1+self.trial_circ[ss][2])

            for last in range(ss, pp-1, -1):
                if last in self.rot_indexes:
                    name=p_vec[np.where(self.rot_indexes==last)[0][0]]
                else:
                    name=self.trial_circ[last][1]+self.rot_loop[last]   
                getattr(temp_circ, self.trial_circ[last][0])(name, 1+self.trial_circ[last][2])
        
            getattr(temp_circ, 'c'+self.trial_circ[pp][0][-1])(0,1+self.trial_circ[pp][2])
            temp_circ.x(0)

        else:
            print('Cant make the dC circ')
            exit()

        temp_circ.h(0)
        temp_circ.measure(0,0)

        return temp_circ
  
    
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

        sum_C_p=0
        for s in range(len(self.rot_indexes)):
            for i in range(len(self.hamil)):
                dCircuit_term_1=self.dC_circ1(p_index, i, self.rot_indexes[s])
                dCircuit_term_2=self.dC_circ2(p_index, self.rot_indexes[s], i)
                
                ## TODO: Fix this, I dont know how dw should be computed
                #TODO: Is the [0][0] correct?
                temp_dw=np.copy(self.hamil[i][0][0]*self.dwdth[i_theta][self.rot_indexes[s]])

            #I guess the real and trace part automatically is computed 
            # in the cirquit.. or is it?
                #+ or - is this what is wrong?
                #TODO: Should be +
                sum_C_p+=0.25*temp_dw*(dCircuit_term_1+dCircuit_term_2)
        
        return dCircuit_term_0+sum_C_p

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
    
    def fidelity_omega_list(self):
        return self.plot_fidelity_list

