#import this as d
from utils import *

import copy
import numpy as np
from qiskit.quantum_info import DensityMatrix, partial_trace

class optimize:
    """
    Class handling everything to do with optimization of parameters and loss computations
    """
    def __init__(self, Hamil, rot_in, trace_list, learning_rate=0.1, method='Adam',circuit=None):
        """
        This class is handling everything regarding optimizing the parameters 
        and loss

        Args:
            learning_rate:  Learning rate used in gradient descent
            circuit:        Quantum circuit that is used
        """
        self.rot_in=rot_in
        if type(Hamil)==int:
            self.n_hamil_params=Hamil
        else:
            self.n_hamil_params=len(Hamil)
        self.trace_list=trace_list
        self.H_qubit_states=2**len(self.trace_list)
        self.learning_rate=learning_rate
        self.circuit=circuit
        self.t=0
        self.m = np.zeros(self.n_hamil_params).astype(float)
        self.v = np.zeros(self.n_hamil_params).astype(float)

        self.method=method
        
        if self.method=='Amsgrad':
            self.vhat= np.zeros(self.n_hamil_params).astype(float)
            
    def cross_entropy_new(self, p_data,p_BM):
        """
        Loss function from article (2)
        """
        #loss=0
        #for i in range(len(p_data)):
        #    loss-=p_data[i]*np.log(p_BM[i])

        return -np.sum(p_data*np.log(p_BM))

    def fraud_CE(self, p_data, p_BM):
        """
        Loss function(cross entropy) adapted to fit the fraud dataset
        classification.

        Args:
            p_data(list):   Target Data
            p_BM(list):     The resulting Boltzmann distribution

        Return(float):      The computed loss
        """
        
        #Something wrong witht one of the things
        #sum_x=0
        #for x in range(len(p_data)):
        #    sum_x+=p_data[x]*np.sum(p_data[x]*np.log(p_BM[x]))

        #TODO: Have to rewrite and add a sum if batch size are larger
        #than one

        #print(p_data)
        #print(p_data_vec)

        #p_data_vec=np.zeros(2)
        #p_data_vec[p_data]=1

        return -np.sum(p_data*np.log(p_BM))


    # gradient descent algorithm with adam
    #def adam(self, x, g, beta1=0.9, beta2=0.999, eps=1e-8):
    def adam(self, x, g, beta1=0.7, beta2=0.99, eps=1e-8, discriminative=False):
        """
        I guess something like this should work?
        
        based on the following article:
        https://machinelearningmastery.com/adam-optimization-from-scratch/
        """
        #Just using formulas from 
        # https://ruder.io/optimizing-gradient-descent/index.html#adam
        
        #Add 1 to the counter
        self.t+=1
        self.m = beta1 * self.m+ (1.0 - beta1) * g
        self.v = beta2 * self.v + (1.0 - beta2) * g**2
        mhat = self.m / (1.0 - beta1**(self.t))

        print(f'g: {g}')
        
        if self.method=='Amsgrad':

            #a_t = self.learning_rate*np.sqrt(1-beta2**self.t)/(1-beta1**self.t)
            a_t=self.learning_rate
            #print(self.learning_rate)
            #a_t=self.learning_rate
            self.vhat=np.maximum(self.vhat, self.v)

            if discriminative==True:
                x -= np.divide(a_t*mhat, np.sqrt(self.vhat) + eps).reshape((len(x), 1))
            else:  
                x -= np.divide(a_t*mhat, np.sqrt(self.vhat) + eps)
            
            #print(f'Change in param: {-np.divide(self.learning_rate*mhat, np.sqrt(self.vhat) + eps)}')
            #print(f'Parameters in adam: m_hat:{mhat} vhat which will be divided upon: {np.sqrt(self.vhat)}') 

        else:
            vhat = self.v / (1.0 - beta2**(self.t))

            if discriminative==True:
                x -= np.divide(self.learning_rate*mhat, np.sqrt(vhat) + eps).reshape((len(x), 1))
            else:
                x -= np.divide(self.learning_rate*mhat, np.sqrt(vhat) + eps)
    
            print(f'Change in param: {-np.divide(self.learning_rate*mhat, np.sqrt(vhat) + eps)}')
            print(f'Parameters in adam: m_hat:{mhat} vhat which will be divided upon: {np.sqrt(vhat)}')        

        return x
   
    def gradient_descent_gradient_done(self, params, gradient):
        """
        Gradient descent function.
        
        Args:
                params:     Variational parameters(list or array)
                lr:         Learning rate
                gradint:     True labels of the samples(list or array)
                samples:    Samples reprenting the true labels and 
                            predictions. (2D Matrix)
        """

        print(f'gradient {gradient}')
        print(f'change in param: {self.learning_rate*gradient}')
        params-=self.learning_rate*gradient
        return params
    
    def gradient_descent(self, params, predicted, target, samples):
        """
        Gradient descent function.
        
        Args:
                params:     Variational parameters(list or array)
                predicted:  List of predicted values(list or array)
                target:     True labels of the samples(list or array)
                samples:    Samples reprenting the true labels and 
                            predictions. (2D Matrix)
        """
        update=self.learning_rate *self.gradient_of_loss(params, predicted, target, samples)
        params-= update

        return params
    
    def cross_entropy_distribution(self, p_data, p_BM):
        """
        Computes loss by cross entropy between the visible nodes and the boltzmann distribution
        
        Args:
            p_data: (list?) distribution data to be modelled
            p_BM:    (list?)BM distributions computed by probability_function in varQBM corresponding to each visible node v
                    maybe create a list depending on the different v?
        
        Returns: (float) Loss as a scalar
        """
        loss=0
        for v in range(len(p_data)):
            loss-=p_data[v]*np.log(p_BM[v])

        return loss

    def cross_entropy(self, preds, targets, classes=2, epsilon=1e-12):
        """
        Computes cross entropy between the true labels and the predictions
        by using distributions. (Not used, binary cross entropy function beneath)
        
        Args:
            preds:   predictions as an array or list
            targets: true labels as an array or list  
        
        Returns: loss as a scalar
        """
        #Creates matrixes to use one hot encoded labels
        distribution_preds=np.zeros((len(preds), classes))
        distribution_target=np.zeros((len(targets), classes))

        #Just rewriting the predictions and labels
        for i in range(len(preds)):
            distribution_preds[i][0]=1-preds[i]
            distribution_preds[i][1]=preds[i]
            
            if targets[i]==0:
                distribution_target[i][0]=1
            elif targets[i]==1:
                distribution_target[i][1]=1

        distribution_preds = np.clip(distribution_preds, epsilon, 1. - epsilon)
        n_samples = len(preds)
        loss = -np.sum(distribution_target*np.log(distribution_preds+1e-9))/n_samples
        return loss

    def binary_cross_entropy(self, preds, targets):
        """
        Computes binary cross entropy between the true labels and the predictions
        
        Args:
            preds:   predictions as an array or list
            targets: true labels as an array or list  
        
        Returns: loss as a scalar
        """
        sum=0
        n_samples=len(preds)
        for index in range(n_samples):
            sum+=targets[index]*np.log(preds[index])+(1-targets[index])*np.log(1-preds[index])

        return -sum/n_samples


    def parameter_shift(self, sample, theta_array, theta_index):
        """
        Parameter shift funtion, that finds the derivative of a
        certain variational parameter, by evaluating the cirquit
        shiftet pi/2 to the left and right.

        Args:
                sample:     Sample that will be optimized with respect 
                            of(list or array)
                theta_array:Variational parameters(list or array)
                theta_index:Index of the parameter that we want the 
                            gradient of

        Returns: optimized variational value
        """
        #Just copying the parameter arrays
        theta_left_shift=theta_array.copy()
        theta_right_shift=theta_array.copy()
        
        #Since the cirquits are normalised the shift is 0.25 which represents pi/2
        theta_right_shift[theta_index]+=0.25
        theta_left_shift[theta_index]-=0.25

        #Predicting with the shifted parameters
        pred_right_shift=self.circuit.predict(np.array([sample]),theta_right_shift)
        pred_left_shift=self.circuit.predict(np.array([sample]),theta_left_shift)
        
        #Computes the gradients
        theta_grad=(pred_right_shift[0]-pred_left_shift[0])/2

        return theta_grad

    def gradient_of_loss(self, thetas, predicted, target, samples):
        """
        Finds the gradient used in gradient descent.

        Args:
                thetas:     Variational parameters(list or array)
                predicted:  List of predicted values(list or array)
                target:     True labels of the samples(list or array)
                samples:    Samples reprenting the true labels and 
                            predictions. (2D Matrix)

        Returns:            Gradient 
        """

        gradients=np.zeros(len(thetas))
        eps=1E-8
        #Looping through variational parameters
        for thet in range(len(thetas)):
            sum=0
            #Looping through samples
            for i in range(len(predicted)):
                grad_theta=self.parameter_shift(samples[i], thetas, thet)
                deno=(predicted[i]+eps)*(1-predicted[i]-eps)
                sum+=grad_theta*(predicted[i]-target[i])/deno

            gradients[thet]=sum
        
        return gradients

    def gradient_ps(self, H, params, d_omega):
        #TODO: added copy
        d_omega=np.array(d_omega, copy=True)
        #print(self.H_qubit_states)
        w_k_sum=np.zeros((len(H), self.H_qubit_states))
        for i in range(len(H)):
            for k in self.rot_in:
                params_left_shift=copy.deepcopy(params)
                params_right_shift=copy.deepcopy(params)
                
                params_right_shift[k][1]+=0.5*np.pi
                params_left_shift[k][1]-=0.5*np.pi
                
                trace_right=create_initialstate(params_right_shift)
                trace_left=create_initialstate(params_left_shift)
                #TODO: run this or just trace it?
                DM_right=DensityMatrix.from_instruction(trace_right)
                DM_left=DensityMatrix.from_instruction(trace_left)

                PT_right=partial_trace(DM_right,self.trace_list)
                PT_left=partial_trace(DM_left,self.trace_list)
                
                w_k_sum[i]+=((np.diag(PT_right.data).real.astype(float)-\
                            np.diag(PT_left.data).real.astype(float))/2)*d_omega[i][k]
            
        return w_k_sum.real.astype(float)

    def fraud_grad_ps(self, H, params, d_omega, n_visible):
        #TODO: added copy
        d_omega=np.array(d_omega, copy=True)
        #print(self.H_qubit_states)
        w_k_sum=np.zeros((len(H), 2))
        for i in range(len(H)):
            for k in self.rot_in:
                params_left_shift=copy.deepcopy(params)
                params_right_shift=copy.deepcopy(params)
                
                params_right_shift[k][1]+=0.5*np.pi
                params_left_shift[k][1]-=0.5*np.pi
                
                trace_right=create_initialstate(params_right_shift)
                trace_left=create_initialstate(params_left_shift)
                #TODO: run this or just trace it?
                DM_right=DensityMatrix.from_instruction(trace_right)
                DM_left=DensityMatrix.from_instruction(trace_left)

                PT_right=partial_trace(DM_right,self.trace_list)
                PT_left=partial_trace(DM_left,self.trace_list)

                #print(PT_right.probabilities(n_visible))
                #print(PT_left.probabilities(n_visible))
                #print(((PT_right.probabilities(n_visible)-\
                #            PT_left.probabilities(n_visible))/2)*d_omega[i][k])

                w_k_sum[i]+=((PT_right.probabilities(n_visible)-\
                            PT_left.probabilities(n_visible))/2)*d_omega[i][k]
            
        return w_k_sum.real.astype(float)

    def gradient_loss(self, data, p_QBM, w_k_sum2):
        dL=data/p_QBM*w_k_sum2
          
        return -np.sum(dL, axis=1).real.astype(float)




