"""
Expressions utilized throughout the scripts
"""
# Common imports
from qiskit.circuit import gate
from typing import ValuesView
import numpy as np
import matplotlib.pyplot as plt
from qiskit import circuit
import seaborn as sns
import os
import random
import qiskit as qk
import cmath
#from qiskit.compiler import assemble
#from qiskit.backends.jobstatus import JOB_FINAL_STATES

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

def run_circuit(qc_circ, shots=1024, parallel=False, backend="statevector_simulator", histogram=False): #backend="qasm_simulator"
    
    if parallel==True:
        qobj_list = [qk.compile(qc, qk.Aer.get_backend(backend)) for qc in qc_circ]
        job_list = [backend.run(qobj) for qobj in qobj_list]

        while job_list:
            for job in job_list:
                if job.status() in JOB_FINAL_STATES:
                    job_list.remove(job)
                    print(job.result().get_counts())


    else:
        job = qk.execute(qc_circ,
                    backend=qk.Aer.get_backend(backend),
                    shots=shots,
                    seed_simulator=10,
                    optimization_level=0
                    )
        results = job.result()
        results = results.get_counts(qc_circ)

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
        return np.array([0,0,0,0])

        #print("Only rx,ry and rz gates are implemented")
        #exit()
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

def encoding_circ(circuit, input_qubits):
    """
    Just defining the quantum circuit encoder in FidS1. in the supplememnt
    """
    data_register_enc = qk.QuantumRegister(2+input_qubits)
    classical_register_enc = qk.ClassicalRegister(1)
    
    qc_enc = qk.QuantumCircuit(data_register_enc, classical_register_enc)

    """
    Note to self: Remember to fix the derivative terms. 0 and pi/2 is given by
                  H and H-S respectively, both are used in the derivative terms.
                  so might need to mix depending on the term used.
                  if statement of H-S
    """
    #return qc_enc
    if circuit=='A':
        qc_enc.h(data_register_enc[0])

    elif circuit=='C':
        qc_enc.h(data_register_enc[0])
        qc_enc.s(data_register_enc[0])

    else:
        print('A or C?')
        exit()
    
    #print(qc_enc)

    return qc_enc



#Just some testing of producing the initialstate
def create_initialstate(gates_params):
    #param_fig2=[['ry',0, 0],['ry',0, 1], ['cx', 0,1], ['cx', 1, 0], ['cx', 0, 1]]
    #gates_params=np.array(gates_params)
    #Creating the circuit
    qubits=np.amax(np.array(gates_params)[:,2].astype('int'))
    #print(np.max(gates_params[0][1]))
    qr = qk.QuantumRegister(qubits+1)
    #cr = qk.ClassicalRegister(1)

    circ = qk.QuantumCircuit(qr)
    
    #print(gates_params)
    #print(circ)

    #dict with gate in them, maybe as functions?

    #gates_params=list(gates_params)
    for i in range(len(gates_params)):
        getattr(circ, str(gates_params[i][0]))(gates_params[i][1], gates_params[i][2])
    #circ.measure_all()
    #circ.measure(0, cr)
    print(run_circuit(circ, shots=1024, histogram=True))

    print(circ)
    return circ


def remove_Nans(A, C):
    """
    Just removing the vectors and nan elements corresponding 
    to gates not dependent of theta
    """
    indexes=[]
    for nan in range(0,len(C)):
        if np.isnan(C[nan])==True:
            indexes.append(nan)

    #Remove nans from C
    C=np.delete(C, indexes)

    #Remove vectors corresponding to C's nans in A
    A=np.delete(A, indexes, axis=0)
    A=np.delete(A, indexes, axis=1)

    return A, C

def remove_constant_gates(V_circ, A, C):
    """
    Just removing the vectors and nan elements corresponding 
    to gates not dependent of theta
    """
    indexes=[]
    for i in range(0,len(C)):
        if V_circ[i][0]=='rx' or V_circ[i][0]=='ry' or V_circ[i][0]=='rz':
            pass
        else:
            indexes.append(i)

    #Remove nans from C
    C=np.delete(C, indexes)

    #Remove vectors corresponding to C's nans in A
    A=np.delete(A, indexes, axis=0)
    A=np.delete(A, indexes, axis=1)

    return A, C

def update_parameters(old_params, new_params):
    for i in range(len(new_params)):
        if old_params[i][0]=='rx' or old_params[i][0]=='ry' or old_params[i][0]=='rz':
            old_params[i][1]=new_params[i]
    
    return old_params