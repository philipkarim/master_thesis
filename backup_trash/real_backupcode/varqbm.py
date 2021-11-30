from utils import *

import numpy as np

class varQBM:
    def __init__(self, circuit, n_visible, n_hidden, v_nodes, h_nodes):
        """
        Class: Variational quantum Boltzmann machine

        Args: 
            circuit(quantum cirquit):    Quantum circuit
            num_visible (int):      Number of visible nodes
		    num_hidden (int):       Number of hidden nodes
            params (array or list): The variational parameters
        """
        self.ciruit=circuit
        self.n_visible=n_visible
        self.n_hidden=n_hidden

        self.v_nodes=v_nodes
        #This might be initialize here instead of sent into the function
        self.h_nodes=h_nodes
        #Not sure if I should initialize a new object each time? rather insert the parameters into a function below.
        #self.params=params

    def energy_function(self, h_nodes, weight_b, weights_vn):
        """
        Energy function used in classical Boltzmann machine 
        
        Args: 
            weight_b(array):    visible biases
            weights_vn(matrix): weights between hidden and visible nodes
        
        Returns:(float)    Value of energyfunction
        """
        E_z_term_1=0
        E_z_term_2=0

        #Computing the enegyfunctuion used in classical BM, eq("0") in article
        for v in range(self.n_visible):
            #First term
            E_z_term_1+=weight_b[v]*self.v_nodes[v]
            #Second term
            for h in range(self.n_hidden):
                E_z_term_2+=weights_vn[v][h]*self.v_nodes[v]*self.h_nodes[h]

        return -(E_z_term_1+E_z_term_2)
        
    def probability_function(self, h_nodes, weight_b, weights_vn):
        """
        The probabillity to observe a configuration v of the visible nodes
        
        Args:

        Return:    
        """
        #Should these be set to 1?
        Z_partition=1
        k_b=1
        temp_T=1

        E_v=0
        for h in range(self.n_hidden):
            #How many hidde nodes do I have? think something is wrong here
            #h_nodes[h]??
            E_v+=self.energy_function(h_nodes[h], weight_b, weights_vn)

        return np.exp(-E_v/(k_b*temp_T))/Z_partition


    def varQITE_state_preparation(self, steps_n, params):
        """
        Prepares an approximation for the gibbs states using imaginary time evolution 
        """
        #Basicly some input values, and then the returned values are the gibbs states.
        #Probably should make this an own function
        #And find out how to solve the differential equations

        #Input: page 6 in article algorithm first lines
        k_b=1
        temp_T=1
        tau=0.5*k_b*temp_T

        time_step=tau/steps_n
        for t in range(time_step, tau+1):   #+1?
            #Compute A(t) and C(t)
            A_temp=expression_A(t)
            C_temp=expression_C(t)

            #Solve A* derivative of \omega=C
            #No idea how to do it
            for i in range(len(params)):
                #Compute the derivative of dC/dtheta_i and dA/dtheta_i
                derivative_C=1
                derivative_A=1

                #Solve A(d d omega)=d C -(d A)*d omega(t)
                
                #Compute:
                #dw(t)=dw(t-time_step)+d d w time_step
            #compute dw
            #w(t+time_step)=w(t)dw(t)time_step

        return w(t), dw(t)/dtheta 

        
    def parameter_Hamiltonian(parameters):
        """
        Should I create a paraterized quantum circuit here or just matrices?
        """
        for params in range(len(parameters)):
            for j in range():

        pass


    def expression_A(tau_value, deriv_V_dagg, deriv_V, rho_in):
        """
        Computes expression A

        Args:
            tau_value:      Value of tau to be evaluated
            deriv_V_dagg:   Derivated of V dagger with respect to \omega(\tau)
            deriv_V:        Derivated of V with respect to \omega(\tau)
            rho_in:         Input quantum state projected(?) |psi_in><psi_in|

            returns real part of trace
        """

        A=np.trace(deriv_V_dagg(tau_value)*deriv_V(tau_value)*rho_in)

        return A.real

    def expression_C(theta_params, h_qubits, tau_value, deriv_V_dagg, V_cir, rho_in):
        """
        Computes expression C

        Args:
            theta_params:   Variational parameters
            h_qubits:       Hidden qubits?
            tau_value:      The value of the tau value
            deriv_V_dagg:   Derivated of V dagger with respect to \omega(\tau)
            V_cir:              The quantum circuit V
            rho_in:         Input quantum state projected(?) |psi_in><psi_in|

            returns real part of trace
        """
        C_p=0

        for i in range(len(theta_params)):
            traced_matrix=np.trace(deriv_V_dagg(tau_value)*h_qubits[i]*V_cir(tau_value)*rho_in)
            C-=theta_params[i]*traced_matrix.real
        return C 

    def dV_dw():
        #parameter_shift?
        
        return value



    

class varQITE:
    def __init__(self, params):
        """
        Class: Variational quantum imaginary time evolution

        Args: 
            params (array or list): The variational parameters
        """

        pass

    

class Hamiltonian:
    def __init__(self, parameters, hidden ):
        """
        Class: Prepare the Hamiltonian

        Args: 
            params (array or list): The variational parameters
        """

        pass
