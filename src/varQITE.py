from xml.etree.ElementTree import register_namespace
import numpy as np
from numpy.core.numeric import zeros_like

from sklearn.linear_model import Ridge, RidgeCV, Lasso, BayesianRidge

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
    def __init__(self, hamil, trial_circ, maxTime=0.5, steps=10, lmbs=np.logspace(-10,-4,7), reg='ridge', symmetrix_matrices=False, plot_fidelity=False, gs_computations=False):
        """
        Class handling the variational quantum imaginary time evolution
        
        Args:
            hamil(list):        Hamiltonian as a list of gates and coefficients
            trial_circ(list):   Trial circuit as a list of gates, params and 
                                qubit placement
            max_Time(float):    Maximum time value for the propagation
            steps(int):         timesteps of varQITE
        """
        self.lmbs=lmbs
        self.reg_method=reg
        self.symmetrix_matrices=symmetrix_matrices
        self.best=False
        self.sup=False
        self.hamil=hamil
        self.trial_circ=trial_circ
        self.maxTime=maxTime
        self.steps=steps
        self.time_step=self.maxTime/self.steps
        self.rot_loop=np.zeros(len(trial_circ), dtype=int)
        self.gs_computations=gs_computations

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

    
    def initialize_circuits(self):
        """
        Initialising all circuits without parameters
        """

        """
        Creating circ A
        """
        A_circ= np.empty(shape=(len(self.rot_indexes), len(self.rot_indexes)), dtype=object)

        #Assuming there is only one circ per i,j, due to r? only having 1 element in f
        if self.symmetrix_matrices==True:
            for i in range(len(self.rot_indexes)):
                            for j in range(i+1):
                                #Just the circuits
                                A_circ[i][j]=self.init_A(self.rot_indexes[i],self.rot_indexes[j])
        else:
            for i in range(len(self.rot_indexes)):
                for j in range(len(self.rot_indexes)):
                    #Just the circuits
                    A_circ[i][j]=self.init_A(self.rot_indexes[i],self.rot_indexes[j])
                
  
        """
        Creating circ C
        """
        C_circ= np.empty(shape=(len(self.hamil),len(self.rot_indexes)), dtype=object)
        
        #Assuming there is only one circ per i,j, due to r? only having 1 element in f
        for i in range(len(self.hamil)):
            for j in range(len(self.rot_indexes)):
                #Just the circuits
                C_circ[i][j]=self.init_C(self.hamil[i], self.rot_indexes[j])
        
        """
        Creating circ dA
        """
        dA_circ= np.empty(shape=(len(self.rot_indexes), len(self.rot_indexes), len(self.rot_indexes), 2), dtype=object)
        
        for i in range(len(self.rot_indexes)):
            for j in range(len(self.rot_indexes)):
                for k in range(len(self.rot_indexes)):
                    dA_circ[i][j][k][0]=self.init_dA(self.rot_indexes[i], self.rot_indexes[j],self.rot_indexes[k], 0)
                    dA_circ[i][j][k][1]=self.init_dA(self.rot_indexes[i], self.rot_indexes[j],self.rot_indexes[k], 1)

        """
        Creating circ dC
        """
        dC_circ= np.empty(shape=(len(self.hamil), len(self.rot_indexes), len(self.rot_indexes), 2), dtype=object)

        for ii in range(len(self.hamil)):
            for jj in range(len(self.rot_indexes)):
                for kk in range(len(self.rot_indexes)):
                    dC_circ[ii][jj][kk][0]=self.init_dC(ii, self.rot_indexes[jj],self.rot_indexes[kk], 0)
                    dC_circ[ii][jj][kk][1]=self.init_dC(ii, self.rot_indexes[jj],self.rot_indexes[kk], 1)

        self.A_init=A_circ
        self.C_init=C_circ
        self.dA_init=dA_circ
        self.dC_init=dC_circ

        return A_circ, C_circ, dA_circ, dC_circ

    def state_prep(self, gradient_stateprep=False):
        """
        Prepares an approximation for the gibbs states using imaginary time evolution 
        """
        omega_w=np.array(self.trial_circ)[:, 1].astype('float')
        #print(f'Start: {omega_w}')
        omega_derivative=np.zeros(len(self.trial_circ))
        self.dwdth=np.zeros((len(self.hamil), len(self.trial_circ)))
        
        A_mat=zeros_like(self.A_init, dtype='float64')
        C_vec=zeros_like(self.rot_indexes, dtype='float64')

        for t in np.linspace(self.time_step, self.maxTime, num=self.steps):
            #Expression A: Binds the parameters to the circuits
            if self.symmetrix_matrices==True:
                for i_a in range(len(self.rot_indexes)):
                    for j_a in range(i_a+1):
                        #Just the circuits
                        A_mat[i_a][j_a]=run_circuit(self.A_init[i_a][j_a].bind_parameters\
                            (omega_w[self.rot_indexes][:len(self.A_init[i_a][j_a].parameters)]), statevector_test=True)
                        A_mat[j_a][i_a]=A_mat[i_a][j_a]
            else:
                for i_a in range(len(self.rot_indexes)):
                    for j_a in range(len(self.rot_indexes)):
                        #Just the circuits
                        A_mat[i_a][j_a]=run_circuit(self.A_init[i_a][j_a].bind_parameters\
                            (omega_w[self.rot_indexes][:len(self.A_init[i_a][j_a].parameters)]), statevector_test=True)

            A_mat*=0.25
            
            for i_c in range(len(self.hamil)):
                for j_c in range(len(self.rot_indexes)):
                    C_vec[j_c]+=self.hamil[i_c][0][0]*run_circuit(self.C_init[i_c][j_c].\
                    bind_parameters(omega_w[self.rot_indexes][:len(self.C_init[i_c][j_c].parameters)]), statevector_test=True)
            
            C_vec*=-0.5

            if isinstance(self.lmbs, (np.ndarray, list)):
                #Compute multiple lambdas, and choose the one wiht lowest loss

                loss=1e8
                final_lmb=1e-10

                for lmb in range(len(self.lmbs)):
                    model_R = Ridge(alpha=self.lmbs[lmb])
                    model_R.fit(A_mat, C_vec)
                    omega_derivative_temp=model_R.coef_
                    loss_temp=mean_squared_error(C_vec,A_mat@omega_derivative_temp)
                    
                    if loss_temp<loss:
                        final_lmb=self.lmbs[lmb]
                        loss=loss_temp
                

                #final_lmb=self.lmbs[0]

                #print(f'Final lmb {final_lmb}')
                #Using the final alpha
                model_R = Ridge(final_lmb)
                model_R.fit(A_mat, C_vec)
                omega_derivative_temp=model_R.coef_

                del model_R

            elif self.reg_method=='ridge':
                #Ridge regression
                model_R = Ridge(alpha=self.lmbs)
                model_R.fit(A_mat, C_vec)
                omega_derivative_temp=model_R.coef_
                #print('It worked')

            elif self.reg_method=='lasso':
                #LASSO regression
                model_L = Lasso(alpha=self.lmbs)
                model_L.fit(A_mat, C_vec)
                omega_derivative_temp=model_L.coef_

            else:
                #Using the pseudo inverse
                A_inv=np.linalg.pinv(A_mat, hermitian=False)
                omega_derivative_temp=A_inv@C_vec
                #print('It worked')
            
            omega_derivative[self.rot_indexes]=omega_derivative_temp

            if gradient_stateprep==False:
                #Preparing the gradient states
                dA_mat=self.getdA_bound(omega_w)

                for i in range(len(self.hamil)):
                    dC_vec=self.getdC_bound(i, omega_w)

                    if isinstance(self.lmbs, (np.ndarray, list)):
                        #Compute multiple lambdas, and choose the one wiht lowest loss
                        loss=1e8
                        rh_side=dC_vec-dA_mat[i]@omega_derivative_temp

                        for lmb in range(len(self.lmbs)):
                            model_R = Ridge(alpha=self.lmbs[lmb])
                            model_R.fit(A_mat, rh_side)
                            w_dtheta_dt=model_R.coef_
                            loss_temp=mean_squared_error(rh_side,A_mat@w_dtheta_dt)
                            
                            if loss_temp<loss:
                                final_lmb=self.lmbs[lmb]
                                loss=loss_temp
                        
                        #final_lmb=self.lmbs[0]

                        model_R = Ridge(final_lmb)
                        model_R.fit(A_mat, rh_side)
                        w_dtheta_dt=model_R.coef_

                        del model_R

                        #print(f'Final lmb, grad:  {final_lmb}')

                    elif self.reg_method=='ridge':
                        #Ridge regression
                        model_R = Ridge(alpha=final_lmb)
                        model_R.fit(A_mat, rh_side)
                        w_dtheta_dt=model_R.coef_

                    elif self.reg_method=='lasso':
                        #LASSO regression
                        model_L = Ridge(alpha=final_lmb)
                        model_L.fit(A_mat, rh_side)
                        w_dtheta_dt=model_L.coef_

                    else:
                        #Using the pseudo inverse
                        dA_inv=np.linalg.pinv(dA_mat[i], hermitian=False)
                        w_dtheta_dt=dA_inv@(dC_vec-dA_mat[i]@omega_derivative_temp)
                        
                    self.dwdth[i][self.rot_indexes]+=w_dtheta_dt*self.time_step

            omega_w+=omega_derivative*self.time_step
            
            if self.plot_fidelity==True:
                self.plot_fidelity_list.append(np.copy(omega_w))
                
            self.trial_circ=update_parameters(self.trial_circ, omega_w)

            if self.gs_computations==True:
                compute_gs_energy(self.trial_circ, self.hamil, t)

        return omega_w, self.dwdth

    def update_H(self, new_H):
        self.hamil=new_H


    #@jit(nopython=True)
    def get_A_from_init(self):

        A_mat_temp=np.zeros((len(self.rot_indexes), len(self.rot_indexes)))
        for i in range(len(self.rot_indexes)):
            #For each gate 
            for j in range(len(self.rot_indexes)):

                A_term=self.run_A2(self.rot_indexes[i],self.rot_indexes[j])
                derivative_const=0.25
                A_mat_temp[i][j]=A_term*derivative_const

        return A_mat_temp
        
    def get_A2(self):        
        A_mat_temp=np.zeros((len(self.rot_indexes), len(self.rot_indexes)))
        for i in range(len(self.rot_indexes)):
            #For each gate 
            for j in range(len(self.rot_indexes)):
                A_term=self.run_A2(self.rot_indexes[i],self.rot_indexes[j])
                derivative_const=0.25
                A_mat_temp[i][j]=A_term*derivative_const
        
        return A_mat_temp

    def run_A2(self,first, sec):
        V_circ=encoding_circ('A', self.trial_qubits)
        temp_circ=V_circ.copy()
        
        if self.sup==True:

            for i, j in enumerate(self.rot_loop[:first]):
                getattr(temp_circ, self.trial_circ[i][0])(self.trial_circ[i][1]+j, 1+self.trial_circ[i][2])
            
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
            
            #temp_circ.x(0)
            getattr(temp_circ, 'c'+self.trial_circ[first][0][-1])(0,1+self.trial_circ[first][2])
            #temp_circ.x(0)

            if first==22.4:#<sec:
                #Continue the U_i gate:

                for ii, jj in enumerate(self.rot_loop[first:sec], start=first):
                    getattr(temp_circ, self.trial_circ[ii][0])(self.trial_circ[ii][1]+jj, 1+self.trial_circ[ii][2])


            else:
                #Continue the U_i gate:
                for ii, jj in enumerate(self.rot_loop[first:], start=first):
                    getattr(temp_circ, self.trial_circ[ii][0])(self.trial_circ[ii][1]+jj, 1+self.trial_circ[ii][2])

                for kk in range(len(self.trial_circ)-1, sec-1, -1):
 
                    getattr(temp_circ, self.trial_circ[kk][0])(self.trial_circ[kk][1]+self.rot_loop[kk], 1+self.trial_circ[kk][2])

            temp_circ.x(0)
            getattr(temp_circ, 'c'+self.trial_circ[sec][0][-1])(0,1+self.trial_circ[sec][2])
            temp_circ.x(0)


        temp_circ.h(0)
        temp_circ.measure(0,0)

        prediction=run_circuit(temp_circ)
        sum_A=prediction


        return sum_A

    def init_A(self,first, sec):
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

        if first==39.1: #<sec
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

            for jjj in range(len(self.trial_circ)-1, sec-1, -1):
                if jjj in self.rot_indexes:
                    name=p_vec[np.where(self.rot_indexes==jjj)[0][0]]
                else:
                    name=self.trial_circ[jjj][1]+self.rot_loop[jjj]

                getattr(temp_circ, self.trial_circ[jjj][0])(name, 1+self.trial_circ[jjj][2])

        
        #temp_circ.x(0)
        getattr(temp_circ, 'c'+self.trial_circ[sec][0][-1])(0,1+self.trial_circ[sec][2])
        #temp_circ.x(0)

        temp_circ.h(0)
        #temp_circ.measure(0,0)
  
        return temp_circ

    def init_C2(self,lamb, fir):
        V_circ=encoding_circ('C', self.trial_qubits)
        temp_circ=V_circ.copy()
        p_vec_C = ParameterVector('Init_param', len(self.rot_indexes))
        
        for i, j in enumerate(self.rot_loop[:fir]):
            if i in self.rot_indexes:
                name=p_vec_C[np.where(self.rot_indexes==i)[0][0]]
            else:
                name=self.trial_circ[i][1]+j
            getattr(temp_circ, self.trial_circ[i][0])(name, 1+self.trial_circ[i][2])
        
        temp_circ.x(0)
        getattr(temp_circ, 'c'+self.trial_circ[fir][0][-1])(0,1+self.trial_circ[fir][2])
        temp_circ.x(0)


        for ii, jj in enumerate(self.rot_loop[fir:], start=fir):
                if ii in self.rot_indexes:
                    name=p_vec_C[np.where(self.rot_indexes==ii)[0][0]]
                else:
                    name=self.trial_circ[ii][1]+jj
                getattr(temp_circ, self.trial_circ[ii][0])(name, 1+self.trial_circ[ii][2])

        for theta in range(len(lamb)):
            getattr(temp_circ, 'c'+lamb[theta][1])(0,lamb[theta][2]+1)

        temp_circ.h(0)
  
        return temp_circ

    def init_C(self,lamb, fir):
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
            if k in self.rot_indexes:
                name=p_vec_C[np.where(self.rot_indexes==k)[0][0]]
            else:
                name=self.trial_circ[k][1]+self.rot_loop[k]

            getattr(temp_circ, self.trial_circ[k][0])(name, 1+self.trial_circ[k][2])

        #temp_circ.x(0)
        getattr(temp_circ, 'c'+self.trial_circ[fir][0][-1])(0,1+self.trial_circ[fir][2])
        #temp_circ.x(0)

        temp_circ.h(0)
        #temp_circ.measure(0, 0)

        #print(f'fir {fir}')
        #print(temp_circ)
        
        return temp_circ

    def get_C2(self):
        C_vec_temp=np.zeros(len(self.rot_indexes))
        
        #Loops through the indices of A
        for i in range(len(self.rot_indexes)):
            c_term=self.run_C2(self.rot_indexes[i])
            
            C_vec_temp[i]=c_term*0.5
 
        return C_vec_temp

    def run_C2(self, ind):
        
        V_circ=encoding_circ('C', self.trial_qubits)
        temp_circ=V_circ.copy()
        sum_C=0

        for l in range(len(self.hamil)):
            if self.sup==True:


                for i, j in enumerate(self.rot_loop[:ind]):
                    getattr(temp_circ, self.trial_circ[i][0])(self.trial_circ[i][1]+j, 1+self.trial_circ[i][2])
                
                temp_circ.x(0)
                getattr(temp_circ, 'c'+self.trial_circ[ind][0][-1])(0,1+self.trial_circ[ind][2])
                temp_circ.x(0)
                
                for ii, jj in enumerate(self.rot_loop[ind:], start=ind):
                    getattr(temp_circ, self.trial_circ[ii][0])(self.trial_circ[ii][1]+jj, 1+self.trial_circ[ii][2])
                
                for theta in range(len(self.hamil[l])):
                    getattr(temp_circ, 'c'+self.hamil[l][theta][1])(0,self.hamil[l][theta][2]+1)
                

            else:    
                if self.best==False:
                    #Then we loop through the gates in U until we reach sigma-gate
                    for i in range(len(self.trial_circ)):
                        getattr(temp_circ, self.trial_circ[i][0])(self.trial_circ[i][1]+self.rot_loop[i], 1+self.trial_circ[i][2])
                    
                    for theta in range(len(self.hamil[l])):
                        getattr(temp_circ, 'c'+self.hamil[l][theta][1])(0,self.hamil[l][theta][2]+1)
                    

                    for ii in range(len(self.trial_circ)-1, ind-1, -1):
                        getattr(temp_circ, self.trial_circ[ii][0])(self.trial_circ[ii][1]+self.rot_loop[ii], 1+self.trial_circ[ii][2])
  
                    #Add x gate                
                    temp_circ.x(0)
                    #Then we add the sigma
                    getattr(temp_circ, 'c'+self.trial_circ[ind][0][-1])(0,1+self.trial_circ[ind][2])
                    #Add x gate                
                    temp_circ.x(0)

                    
                else:

                   
                    for i in range(len(self.trial_circ)-1, ind-1, -1):
                        getattr(temp_circ, self.trial_circ[i][0])(self.trial_circ[i][1]+self.rot_loop[i], 1+self.trial_circ[i][2])

                    #Add x gate                
                    temp_circ.x(0)
                    #Then we add the sigma
                    getattr(temp_circ, 'c'+self.trial_circ[ind][0][-1])(0,1+self.trial_circ[ind][2])
                    #Add x gate                
                    temp_circ.x(0)

                    #Continue the U_i gate:
                    for ii in range(ind-1, -1, -1):
                        getattr(temp_circ, self.trial_circ[ii][0])(self.trial_circ[ii][1]+self.rot_loop[ii], 1+self.trial_circ[ii][2])

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
        return dA_mat_temp_i*(-0.125)

    def getdA_bound(self, binding_values):
        dA_mat_temp=np.zeros((len(self.rot_indexes), len(self.rot_indexes), len(self.rot_indexes), 2))

        dA=np.zeros((len(self.hamil), len(self.rot_indexes), len(self.rot_indexes)))

        #da_run=time.time()
        
        """
        This loop right here is what takes the most time, due to the time it takes to run all
        the circuits
        """
        for p_da in range(len(self.rot_indexes)):
            for q_da in range(len(self.rot_indexes)):
                for s_da in range(len(self.rot_indexes)):
                    dA_mat_temp[p_da][q_da][s_da][0]=run_circuit(self.dA_init[p_da][q_da][s_da][0].\
                    bind_parameters(binding_values[self.rot_indexes][:len(self.dA_init[p_da][q_da][s_da][0].parameters)]), statevector_test=True)
                    dA_mat_temp[p_da][q_da][s_da][1]=run_circuit(self.dA_init[p_da][q_da][s_da][1].\
                    bind_parameters(binding_values[self.rot_indexes][:len(self.dA_init[p_da][q_da][s_da][1].parameters)]), statevector_test=True)
        #print(f'Time to run dA circs{time.time()-da_run}')

        #TODO: slice this, but it uses very little time to compute, so might not be necesarry
        if self.symmetrix_matrices==True:
            for i in range(len(self.hamil)):
                for p in range(len(self.rot_indexes)):
                    for q in range(p+1):
                        for s in range(len(self.rot_indexes)):
                            dA[i][p][q]+=self.dwdth[i][self.rot_indexes[s]]*(dA_mat_temp[p][q][s][0]+dA_mat_temp[p][q][s][1])
            
            for i in range(len(self.hamil)):
                for p in range(len(self.rot_indexes)):
                    for q in range(p):
                        dA[i][q][p]=dA[i][p][q]
        else:
            for i in range(len(self.hamil)):
                for p in range(len(self.rot_indexes)):
                    for q in range(len(self.rot_indexes)):
                        for s in range(len(self.rot_indexes)):
                            dA[i][p][q]+=self.dwdth[i][self.rot_indexes[s]]*(dA_mat_temp[p][q][s][0]+dA_mat_temp[p][q][s][1])

        dA*=0.125

        return dA

    def getdC_bound(self,i_th, binding_values):
        dC_i=np.zeros(len(self.rot_indexes))

        for j_p in range(len(self.rot_indexes)):
            dC_i[j_p]-=0.5*run_circuit(self.C_init[i_th][j_p].bind_parameters(binding_values[self.rot_indexes]\
                                                [:len(self.C_init[i_th][j_p].parameters)]),statevector_test=True)
            
            #Something changes inside the loop?
            for i_dc in range(len(self.hamil)):
                for s_dc in range(len(self.rot_indexes)):
                    #This looks really ugly
                    term1_temp=run_circuit(self.dC_init[i_dc][j_p][s_dc][0].\
                    bind_parameters(binding_values[self.rot_indexes]\
                    [:len(self.dC_init[i_dc][j_p][s_dc][0].parameters)]),statevector_test=True)

                    term2_temp=run_circuit(self.dC_init[i_dc][j_p][s_dc][1]\
                    .bind_parameters(binding_values[self.rot_indexes]\
                    [:len(self.dC_init[i_dc][j_p][s_dc][1].parameters)]), statevector_test=True)

                    dC_i[j_p]-=0.25*self.hamil[i_dc][0][0]*self.dwdth[i_th][s_dc]*(term1_temp+term2_temp)

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
        #temp_circ.measure(0,0)
  
        return temp_circ
    
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
        #temp_circ.measure(0,0)

        return temp_circ
  
    
    def dA_circ(self, circ_1, circ_2):
        assert len(circ_2)==1 or len(circ_2)==2

        if len(circ_1)==1:
            first_der=circ_1[0]
            sec_der=circ_2[0]
            thr_der=circ_2[1]
            
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

        temp_circ.measure(0,0)

        prediction=run_circuit(temp_circ)
 
        sum_dA=prediction

        return sum_dA

    def get_dC(self, i_param):
        #Lets try to remove the controlled gates
        dC_vec_temp_i=np.zeros(len(self.rot_indexes))
        #Loops through the indices of C
        for p in range(len(self.rot_indexes)):
            dC_vec_temp_i[p]=self.run_dC(self.rot_indexes[p], i_param)
        
        return dC_vec_temp_i
    
    def run_dC(self, p_index, i_theta):
        dCircuit_term_0=-0.5*self.dC_circ0(p_index, i_theta)

        sum_C_p=0
        for s in range(len(self.rot_indexes)):
            for i in range(len(self.hamil)):
                dCircuit_term_1=self.dC_circ1(p_index, i, self.rot_indexes[s])
                dCircuit_term_2=self.dC_circ2(p_index, self.rot_indexes[s], i)
                
                temp_dw=np.copy(self.hamil[i][0][0]*self.dwdth[i_theta][self.rot_indexes[s]])

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
                derivative_const=0.25
                A_mat_temp2[i][j]=A_term*derivative_const

            c_term=self.run_C2(self.rot_indexes[i])
            C_vec_temp2[i]=c_term*0.5


        return A_mat_temp2, C_vec_temp2
    
    def fidelity_omega_list(self):
        return self.plot_fidelity_list

