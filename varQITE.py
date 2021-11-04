import numpy as np
from 

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

    
    def varQITE_state_preparation(self):
        """
        Prepares an approximation for the gibbs states using imaginary time evolution 
        """
        #Basicly some input values, and then the returned values are the gibbs states.
        #Probably should make this an own function
        #And find out how to solve the differential equations

        #Input: page 6 in article algorithm first lines

        time_step=self.maxTime/self.steps

        #initialisation of w for each theta, starting with 0?
        w_dtheta=np.zeros(len(self.hamil))

        for t in range(time_step, self.maxTime+1):   #+1?
            #Compute A(t) and C(t)
            A_mat2=np.copy(get_A2(param_fig2))
            C_vec2=np.copy(get_C2(param_fig2, H_simple))

            A_temp=expression_A(t)
            C_temp=expression_C(t)

            A_inv_temp=np.inv(A_temp)
            
            omega_derivative=A_inv_temp@C_temp

            #Solve A* derivative of \omega=C
            #No idea how to do it
            for i in range(len(H_theta)): 
                #Compute the expression of the derivative
                dA_mat=np.copy(get_dA(theta_list, gates_str))
                dC_vec=np.copy(get_dC(theta_list, gates_str, H_simple))

                #Now we compute the derivative of omega derivated with respect to
                #hamiltonian parameter
                #dA_mat_inv=np.inv(dA_mat)
                w_dtheta_dt= A_inv_temp@(dC_vec-dA_mat@omega_derivative)#* or @?

                w_dtheta[i]+=w_dtheta_dt*time_step

                #Solve A(d d omega)=d C -(d A)*d omega(t)
                
                #Compute:
                #dw(t)=dw(t-time_step)+d d w time_step
            #compute dw
            #w(t+time_step)=w(t)dw(t)time_step

        return w(t), dw(t) 

    def get_A2(self, V_list):
        """
        Continue from here
        """
        #Lets try to remove the controlled gates
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