#import this as d
from utils import *

import copy
import numpy as np
from qiskit.quantum_info import DensityMatrix, partial_trace
import torch
from sklearn.metrics import mean_squared_error


class optimize:
    """
    Class handling everything to do with optimization of parameters and loss computations
    """
    def __init__(self, Hamil, rot_in, trace_list,loss_func='classification', learning_rate=0.1, method='Adam',fraud=False):
        """
        This class is handling everything regarding optimizing the parameters 
        and loss

        Args:
            learning_rate:  Learning rate used in gradient descent
            circuit:        Quantum circuit that is used
        """
        self.rot_in=rot_in
        self.method=method
        self.loss_func=loss_func

        #print(type(Hamil)=='NN.Feedforward')
        #print(len))
        #print(type(Hamil)==class)

        if torch.is_tensor(Hamil):
            #print(len(Hamil))
            Hamil=np.zeros(len(Hamil))
            #exit()
            
        if fraud==True:
            self.m = np.zeros_like((Hamil)).astype(float)
            self.v = np.zeros_like((Hamil)).astype(float)
            if self.method=='Amsgrad':
                self.vhat= np.zeros_like((Hamil)).astype(float)

        else:
            self.n_hamil_params=len(Hamil)    
            self.m = np.zeros(self.n_hamil_params).astype(float)
            self.v = np.zeros(self.n_hamil_params).astype(float)
         
            if self.method=='Amsgrad':
                self.vhat= np.zeros(self.n_hamil_params).astype(float)
        
        self.trace_list=trace_list
        self.H_qubit_states=2**len(self.trace_list)
        self.learning_rate=learning_rate
        self.t=0

        
            
    def cross_entropy_new(self, p_data,p_BM):
        """
        Loss function from article (2)
        """
        #loss=0
        #for i in range(len(p_data)):
        #    loss-=p_data[i]*np.log(p_BM[i])

        return -np.sum(p_data*np.log(p_BM))

    def cross_entropy(self, p_data, p_BM):
        """
        Loss function(cross entropy) adapted to fit the fraud dataset
        classification.

        Args:
            p_data(list):   Target Data
            p_BM(list):     The resulting Boltzmann distribution

        Return(float):      The computed loss
        """

        return -np.sum(p_data*np.log(p_BM))

    def MSE(self, y_true, y_pred):
        """
        Function which computes the mean squared errors

        Args:
            y_true(ndarray):        Ground truth
            y_pred(ndarray):        Prediction

        Returns(float or ndarray):  loss 
        """
        #return mean_squared_error(y_true, y_pred)

        return (y_true-y_pred)**2


    # gradient descent algorithm with adam
    #def adam(self, x, g, beta1=0.9, beta2=0.999, eps=1e-8):
    def adam(self, x, g, beta1=0.7, beta2=0.99, eps=1e-8, discriminative=False, sample=None):
    #def adam(self, x, g, beta1=0.9, beta2=0.999, eps=1e-8, discriminative=False, sample=None):
        """
        I guess something like this should work?
        
        based on the following article:
        https://machinelearningmastery.com/adam-optimization-from-scratch/
        """
        #Just using formulas from 
        # https://ruder.io/optimizing-gradient-descent/index.html#adam
        
        #Add 1 to the counter

        """
        if sample!=None:
            self.t+=1
            self.m = beta1 * self.m+ (1.0 - beta1) * g
            self.v = beta2 * self.v + (1.0 - beta2) * g**2
            mhat = self.m / (1.0 - beta1**(self.t))
        """
        if sample is not None:
            gradient=np.zeros((len(g),len(sample)))
            for i in range(len(g)):
                for j in range(len(sample)):
                    gradient[i][j]=g[i]*sample[j]

            g=gradient
                    
        self.t+=1
        self.m = beta1 * self.m+ (1.0 - beta1) * g
        self.v = beta2 * self.v + (1.0 - beta2) * g**2
        mhat = self.m / (1.0 - beta1**(self.t))

        #print(f'g: {g}')
        #print(f'm: {self.m}')
        #print(f'v: {self.v}')
        

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
    
            #print(f'Change in param: {-np.divide(self.learning_rate*mhat, np.sqrt(vhat) + eps)}')
            #print(f'Parameters in adam: m_hat:{mhat} vhat which will be divided upon: {np.sqrt(vhat)}')        

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

    def cross_entropy_old(self, preds, targets, classes=2, epsilon=1e-12):
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
        d_omega=np.array(d_omega, copy=True)

        w_k_sum=np.zeros((len(H), self.H_qubit_states))
        for i in range(len(H)):
            for k in self.rot_in:
                params_left_shift=copy.deepcopy(params)
                params_right_shift=copy.deepcopy(params)
                
                params_right_shift[k][1]+=0.5*np.pi
                params_left_shift[k][1]-=0.5*np.pi
                
                trace_right=create_initialstate(params_right_shift)
                trace_left=create_initialstate(params_left_shift)

                DM_right=DensityMatrix.from_instruction(trace_right)
                DM_left=DensityMatrix.from_instruction(trace_left)

                PT_right=partial_trace(DM_right,self.trace_list)
                PT_left=partial_trace(DM_left,self.trace_list)
                
                w_k_sum[i]+=((np.diag(PT_right.data).real.astype(float)-\
                            np.diag(PT_left.data).real.astype(float))/2)*d_omega[i][k]
            
        return w_k_sum.real.astype(float)

    def fraud_grad_ps(self, H, params, d_omega, n_visible):
        d_omega=np.array(d_omega, copy=True)
        w_k_sum=np.zeros((len(H), 2**len(n_visible)))
        for i in range(len(H)):
            for k in self.rot_in:
                params_left_shift=copy.deepcopy(params)
                params_right_shift=copy.deepcopy(params)
                
                params_right_shift[k][1]+=0.5*np.pi
                params_left_shift[k][1]-=0.5*np.pi
                
                trace_right=create_initialstate(params_right_shift)
                trace_left=create_initialstate(params_left_shift)
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
        if self.loss_func=='classification':
            #Derivative of cross entropy loss function
            dL=data/p_QBM*w_k_sum2  
            return -np.sum(dL, axis=1).real.astype(float)
        else:
            #Derivative of MSE loss
            dL=(data-p_QBM)*w_k_sum2

            #Insert the correct one

            return -2*np.sum(dL, axis=1).real.astype(float)
    
    def gradient_energy(self, gradient_qbm, H_energy):
        #dL=H_energy*gradient_qbm
        #OR or or oppsosite
        dL=H_energy/gradient_qbm
        #print(f'Sec grad: {-np.sum(H_energy/gradient_qbm, axis=1).real.astype(float)}')
          
        return -np.sum(dL, axis=1).real.astype(float)






