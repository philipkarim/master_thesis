"""
Expressions utilized throughout the scripts
"""
# Common imports
from typing import ValuesView
import numpy as np
import matplotlib.pyplot as plt
from qiskit import circuit
import seaborn as sns
import os
import random
import qiskit as qk
import cmath

random.seed(2021)

def accuracy_score(y, y_pred):
    """
    Computes the accuracy score
    """
    y=np.ravel(y)
    y_pred=np.ravel(y_pred)
    numerator=np.sum(y == y_pred)

    return numerator/len(y)

def hard_labels(y_array, treshold):
    """
    Rewriting the soft predictions in probabillity space
    into hard predictions, of 0 or 1
    """
    for i in range(len(y_array)):
        y_array[i]=np.where(y_array[i]>treshold, 1, 0)
    
    return y_array

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def getDistribution(type, stop, n):
    if type=="U":
        return np.random.uniform(0.,stop,size=n)
    elif type=="N":
        return np.random.normal(stop/2,stop,size=n)
    else:
        print("Choose U for uniform distribution or N for normal distribution. \n Shutting down")
        quit()

def data_path(DATA_ID, dat_id):
    return os.path.join(DATA_ID, dat_id)


def plotter(*args, x_axis,x_label, y_label):
    """
    Just a function to plot functions.

    Args: *args:    arguments passed as x1 data, y1 data, 
                    label 1, x2 data, y2 data, label 2...
          x_label:  Name of x axis(string)
          y_label:  Name of y axis(string)
    """
    sns.set_style("darkgrid")
    
    if len(args)>1:
        for i in range(0, int(len(args)),2):
            plt.plot(x_axis, args[i], label=args[i+1])
            plt.legend()
        plt.xlabel(x_label,fontsize=12)
        plt.ylabel(y_label,fontsize=12)
    else:
        plt.plot(x_axis, args[0])
        plt.xlabel(x_label,fontsize=12)
        plt.ylabel(y_label,fontsize=12)
    plt.show()

    return

def run_circuit(qc, shots=1024, backend="qasm_simulator", histogram=False):
    job = qk.execute(qc,
                    backend=qk.Aer.get_backend(backend),
                    shots=shots,
                    seed_simulator=10
                    )
    results = job.result()
    results = results.get_counts(qc)

    if histogram==True:
        qk.visualization.plot_histogram(results)
        plt.show()

    prediction = 0
    for key,value in results.items():
        #print(key, value)
        if key == '1':
            prediction += value
    prediction/=shots

    return prediction


class pauli_terms:
    def __init__(self, theta=None, label=None):
        self.theta=theta
        self.label=label
    
    def gate_to_matrix(self):
        if self.label=='rx':
            return self.rx_gate()
        elif self.label=='ry':
            return self.ry_gate()
        elif self.label=='rz':
            return self.rz_gate()

        else:
            return "Something is wrong"

        """
        Continue with controlled gates, if no better method is found
        """

    def rx_gate(self):
        #cos_term=np.cos(self.theta/2)
        #sin_term=np.sin(self.theta/2)
        #return np.array([[cos_term, -1j*sin_term], [-1j*sin_term ,cos_term]])
        return np.exp(-1j*self.get_X()*self.theta/2)

    def ry_gate(self):
        #cos_term=np.cos(self.theta/2)
        #sin_term=np.sin(self.theta/2)
        #return np.array([[cos_term, -sin_term], [sin_term ,cos_term]])
        return np.exp(-1j*self.get_Y()*self.theta/2)

    def rz_gate(self):
        return np.exp(-1j*self.get_Z()*self.theta/2)
        #return np.array([[np.exp(-1j*self.theta/2), 0], [0, np.exp(1j*self.theta/2)]])

    def get_I(self):
        """
        Returns an I matrix
        """
        return np.array([[1,0],[0,1]])

    def get_X(self):
        """
        Returns an X matrix
        """
        return np.array([[0,1], [1,0]])

    def get_Y(self):
        """
        Returns an Y matrix
        """
        return np.array([[0,-1.j], [1.j,0]])

    def get_Z(self):
        """
        Returns an Z matrix
        """
        return np.array([[1,0],[0,-1]])
    
def get_M(unitary_gate):
    """
    Args:
        An unitary gate as a matrix

    Returns: (Matrix) the hermitian matrix M in e^(iM(w))
    """

    return np.log(unitary_gate)/1j

def decomposing_to_pauli(H):
    """
    DOES NOT WORK FOR CONTROLLED QUBITS
    https://quantumcomputing.stackexchange.com/questions/8725/can-arbitrary-matrices-be-decomposed-using-the-pauli-basis

    This functions takes hermitian matrices and decomposes them into coefficients of Pauli terms

    Args:
        H(np.array)     Hermitian matrix

    Return: (np.array) size 4 containing coefficients of [I,X,Y,Z] terms

    Example: Works for RX(pi) then it decomposes into x gates, might round the smallest numbers? e-33

    """
    pauli_terms=np.array([[[1,0],[0,1]], [[0,1], [1,0]], [[0,-1.j], [1.j,0]], [[1,0], [0,-1]]])

    #Check that matrix H is hermitian:
    #assert (H.conj().T == H).all(), "How should I say it.. Well, your Hermitian matrix is not really.. Hermitian"

    decomp=np.zeros(len(pauli_terms), dtype = 'complex')
    for i in range(len(pauli_terms)):
        decomp[i]=np.trace(pauli_terms[i]@H)

    #decomp = decomp.astype(np.float)

    #Just normalizes it
    #Should I do it?
    #return np.real(decomp/np.sum(decomp))

    #why divide by 2?
    return np.real(decomp)/len(H)

def get_f_sigma(label):
    """
    Args:
        label of the gate

    returns:f, sigma(2 lists)
    """
    if label=='rx':
        return np.array([0,-0.5j,0,0])
    elif label=='ry':
        return np.array([0,0,-0.5j,0])
    elif label=='rz':
        return np.array([0,0,0,-0.5j])
    else:
        print("Only rx,ry and rz gates are implemented")
        exit()
    """
    Here it is possible to implement for arbitrary hermitian matrices by
    using the following approach under
    """
    #pauli_matrix=pauli_terms(gates_list[i][1], gates_list[i][0])
    #rx_as_matrix=pauli_matrix.gate_to_matrix()
    #mat_M=get_M(rx_as_matrix)
    #decomp_pauli_terms=decomposing_to_pauli(mat_M)

    return 



def update_parameter_dict(dict, new_parameters):
    """
    Just updating the parameters.
    This function could for sure be optimized
    """
    i=0
    for key in dict:
        dict[key]=new_parameters[i]
        i+=1
    return dict

def encoding_circ():
    """
    Just defining the quantum circuit encoder in FidS1. in the supplememnt
    """
    data_register_enc = qk.QuantumRegister(2)
    classical_register_enc = qk.ClassicalRegister(1)
    
    qc_enc = qk.QuantumCircuit(data_register_enc, classical_register_enc)
    qc_enc.h(data_register_enc[0])

    #print(qc_enc)

    return qc_enc

def run_A(U_list, params_circ, i, j):
    #gates_str=[['rx',0],['ry', 0]]

    gate_label_i=U_list[i][0]
    gate_label_j=U_list[j][0]
    
    f_k_i=np.conjugate(get_f_sigma(gate_label_i))
    f_l_j=get_f_sigma(gate_label_j)
    V_circ=encoding_circ()

    pauli_names=['i', 'x', 'y', 'z']
    
    sum_A=0
    for i in range(len(f_k_i)):
        for j in range(len(f_l_j)):
            if f_k_i[i]==0 or f_l_j[j]==0:
                pass
            else:
                #First lets make the circuit:
                temp_circ=V_circ.copy()
                
                """
                Implements it due to figure S1, is this right? U_i or U_j gates first, dagger?
                """
                #Then we loop thorugh the gates in U untill we reach the sigma
                for ii in range(i-1):
                    if len(U_list[ii])==2:
                        getattr(temp_circ, U_list[ii][0])(params_circ[ii], U_list[ii][1])
                    elif len(U_list[ii])==3:
                        getattr(temp_circ, U_list[ii][0])(params_circ[ii], U_list[ii][1], U_list[ii][2])
                    else:
                        print('Something is wrong, I can sense it')
                        exit()
                #Add x gate                
                temp_circ.x(0)
                #Then we add the sigma
                getattr(temp_circ, 'c'+pauli_names[i])(0,1)
                #Add x gate                
                temp_circ.x(0)
                #Continue the U_i gate:
                for keep_going in range(i-1, len(U_list)):
                    if len(U_list[keep_going])==2:
                        getattr(temp_circ, U_list[keep_going][0])(params_circ[keep_going], 1)
                    elif len(U_list[keep_going])==3:
                        getattr(temp_circ, U_list[keep_going][0])(params_circ[keep_going], U_list[keep_going][1], U_list[keep_going][2])
                    else:
                        print('Something is wrong, I can feel it')
                        exit()
                for jj in range(j-1):
                    if len(U_list[jj])==2:
                        getattr(temp_circ, U_list[jj][0])(params_circ[jj], 1)
                    elif len(U_list[jj])==3:
                        getattr(temp_circ, U_list[jj][0])(params_circ[jj], U_list[jj][1], U_list[jj][2])
                    else:
                        print('Something is wrong, I can feel it')
                        exit()

                getattr(temp_circ, 'c'+pauli_names[i])(0,1)
                temp_circ.h(0)
                temp_circ.measure(0, 0)
                """
                Measures the circuit
                """
                job = qk.execute(temp_circ,
                        backend=qk.Aer.get_backend('qasm_simulator'),
                        shots=1024,
                        seed_simulator=10)
                results = job.result()
                results = results.get_counts(temp_circ)

                prediction = 0
                for key,value in results.items():
                    #print(key, value)
                    if key == '1':
                        prediction += value
                prediction/=1024
                
                sum_A+=f_k_i[i]*f_l_j[j]*prediction

    return sum_A


def run_C(U_list, params_circ, H_list, i):
    gate_label_i=U_list[i][0]

    f_k_i=np.conjugate(get_f_sigma(gate_label_i))
    """
    lambda is actually the coefficirents of the hamiltonian,
    but I think I should wait untill I actually have the
    Hamiltonian to implement is xD
    Also h_l are tensorproducts of the thing, find out how to compute tensor products optimized way
    """

    #The length might be longer than this
    #lambda_l=np.random.uniform(0,1,size=len(f_k_i))
    lambda_l=(np.array(H_list)[:, 0]).astype('complex')
    #arr = arr.astype('float64')

    #lambda_l=(np.array(H_list)[:, 0], dtype=np.complex)

    #This is just to have something there
    h_l=['i', 'x', 'y', 'z']

    V_circ=encoding_circ()
    
    pauli_names=['i', 'x', 'y', 'z']
    
    sum_C=0
    for i in range(len(f_k_i)):
        for l in range(len(lambda_l)):
            if f_k_i[i]==0 or lambda_l[l]==0:
                pass
            else:
                #First lets make the circuit:
                temp_circ=V_circ.copy()

                #Then we loop thorugh the gates in U untill we reach the sigma
                for ii in range(i-1):
                    if len(U_list[ii])==2:
                        getattr(temp_circ, U_list[ii][0])(params_circ[ii], U_list[ii][1])
                    elif len(U_list[ii])==3:
                        getattr(temp_circ, U_list[ii][0])(params_circ[ii], U_list[ii][1], U_list[ii][2])
                    else:
                        print('Something is wrong')

                #Add x gate                
                temp_circ.x(0)
                #Then we add the sigma
                getattr(temp_circ, 'c'+pauli_names[i])(0,1)
                #Add x gate                
                temp_circ.x(0)
                #Continue the U_i gate:
                for keep_going in range(i-1, len(U_list)):
                    if len(U_list[keep_going])==2:
                        getattr(temp_circ, U_list[keep_going][0])(params_circ[keep_going], 1)
                    elif len(U_list[keep_going])==3:
                        getattr(temp_circ, U_list[keep_going][0])(params_circ[keep_going], U_list[keep_going][1], U_list[keep_going][2])
                    else:
                        print('Something is wrong, I can feel it')
                        exit()
                #Then add the h_l gate
                #The if statement is to not have controlled identity gates, since it is the first element but might fix this later on
                if H_list[l][1]!='I':
                    getattr(temp_circ, 'c'+H_list[l][1])(0,1)
                
                temp_circ.h(0)
                temp_circ.measure(0, 0)

                """
                Measures the circuit
                """
                job = qk.execute(temp_circ,
                        backend=qk.Aer.get_backend('qasm_simulator'),
                        shots=1024,
                        seed_simulator=10)
                results = job.result()
                results = results.get_counts(temp_circ)

                prediction = 0
                for key,value in results.items():
                    #print(key, value)
                    if key == '1':
                        prediction += value
                prediction/=1024
                
                sum_C+=f_k_i[i]*lambda_l[l]*prediction

    return sum_C

def get_A(parameters_list, gates_list):
    A_mat_temp=np.zeros((len(parameters_list), len(parameters_list)))

    #Loops through the indices of A
    for i in range(len(parameters_list)):
        #For each gate 
        #range(1) if there is no controlled qubits?
        for j in range(len(parameters_list)):
            #Get f_i and f_j
            #Get, the sigma terms
            
            #4? dimension of hermitian or n pauliterms? 
            a_term=run_A(gates_list, parameters_list, i, j)
            
            A_mat_temp[i][j]=np.real(a_term)

    return A_mat_temp

def get_C(parameters_list, gates_list, hamilton_list):
    C_vec_temp=np.zeros(len(parameters_list))
    #Lets create C also
    for i in range(len(parameters_list)):
        c_term=run_C(gates_list, parameters_list,hamilton_list, i)
        C_vec_temp[i]=np.real(c_term)

    return C_vec_temp


#Just some testing of producing the initialstate
def create_initialstate(gates_params):
    #param_fig2=[['ry',0, 0],['ry',0, 1], ['cx', 0,1], ['cx', 1, 0], ['cx', 0, 1]]

    #Creating the circuit
    qr = qk.QuantumRegister(2)
    #cr = qk.ClassicalRegister(1)

    circ = qk.QuantumCircuit(qr)

    for i in range(len(gates_params)):
        getattr(circ, gates_params[i][0])(gates_params[i][1], gates_params[i][2])
    circ.measure_all()

    #print(run_circuit(circ, shots=1024*8, histogram=True))

    return circ

def evaluate_A2():
    """
    Just trying to evaluate A with fig2 example
    """
    A_mat_temp=np.zeros((len(parameters_list), len(parameters_list)))

    #Loops through the indices of A
    for i in range(len(parameters_list)):
        #For each gate 
        #range(1) if there is no controlled qubits?
        for j in range(len(parameters_list)):
            #Get f_i and f_j
            #Get, the sigma terms
            
            #4? dimension of hermitian or n pauliterms? 
            a_term=run_A(gates_list, parameters_list, i, j)
            
            A_mat_temp[i][j]=np.real(a_term)